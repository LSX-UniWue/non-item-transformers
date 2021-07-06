from typing import Tuple, Union, List

import torch

from torch import nn

from asme.models.common.components.sequence_embedding import SequenceElementsEmbeddingComponent
from asme.models.common.layers.data.sequence import SequenceRepresentation, ModifiedSequenceRepresentation, \
    EmbeddedElementsSequence
from asme.models.common.layers.layers import IdentitySequenceRepresentationModifierLayer, \
    SequenceRepresentationModifierLayer
from asme.models.common.layers.util_layers import get_activation_layer
from asme.models.sequence_recommendation_model import SequenceRecommenderModel, SequenceRepresentationLayer, \
    ProjectionLayer
from asme.utils.hyperparameter_utils import save_hyperparameters
from data.datasets import USER_ENTRY_NAME


class UserEmbeddingConcatModifier(SequenceRepresentationModifierLayer):

    """
    a sequence representation modifier that cats a user embedding to the sequence representation
    """
    def __init__(self,
                 user_vocab_size: int,
                 embedding_size: int
                 ):
        super().__init__()
        self.user_embedding = nn.Embedding(user_vocab_size, embedding_dim=embedding_size)

        # init weights
        self.user_embedding.weight.data.normal_(0, 1.0 / self.user_embedding.embedding_dim)

    def forward(self, sequence_representation: SequenceRepresentation) -> ModifiedSequenceRepresentation:
        user = sequence_representation.embedded_elements_sequence.input_sequence.get_attribute(USER_ENTRY_NAME)
        user_emb = self.user_embedding(user).squeeze(1)

        modified_representation = torch.cat([sequence_representation.encoded_sequence, user_emb], 1)
        return ModifiedSequenceRepresentation(modified_representation)


class CaserSequenceRepresentationLayer(SequenceRepresentationLayer):

    def __init__(self,
                 embedding_size: int,
                 max_sequence_length: int,
                 num_vertical_filters: int,
                 num_horizontal_filters: int,
                 conv_activation_fn: str,
                 fc_activation_fn: str,
                 dropout: float
                 ):
        super().__init__()

        self.num_vertical_filters = num_vertical_filters
        self.num_horizontal_filters = num_horizontal_filters

        # vertical conv layer
        self.conv_vertical = nn.Conv2d(1, num_vertical_filters, (max_sequence_length, 1))

        # horizontal conv layer
        lengths = [i + 1 for i in range(max_sequence_length)]

        self.conv_horizontal = nn.ModuleList(
            [CaserHorizontalConvNet(num_filters=num_horizontal_filters,
                                    kernel_size=(length, embedding_size),
                                    activation_fn=conv_activation_fn,
                                    max_length=max_sequence_length)
             for length in lengths]
        )

        # dropout
        self.dropout = nn.Dropout(dropout)

        # fully-connected layer
        self.fc1_dim_vertical = num_vertical_filters * embedding_size
        fc1_dim_horizontal = num_horizontal_filters * len(lengths)
        fc1_dim = self.fc1_dim_vertical + fc1_dim_horizontal

        self.fc1 = nn.Linear(fc1_dim, embedding_size)

        self.fc1_activation = get_activation_layer(fc_activation_fn)

    def forward(self, embedded_sequence: EmbeddedElementsSequence) -> SequenceRepresentation:
        sequence = embedded_sequence.embedded_sequence

        sequence = sequence.unsqueeze(1)

        # Convolutional Layers
        out_vertical = None
        # vertical conv layer
        if self.num_vertical_filters > 0:
            out_vertical = self.conv_vertical(sequence)
            out_vertical = out_vertical.view(-1, self.fc1_dim_vertical)  # prepare for fully connect

        # horizontal conv layer
        out_horizontal = None
        out_hs = list()
        if self.num_horizontal_filters > 0:
            for conv in self.conv_horizontal:
                conv_out = conv(sequence)
                out_hs.append(conv_out)
            out_horizontal = torch.cat(out_hs, 1)

        # Fully-connected Layers
        out = torch.cat([out_vertical, out_horizontal], 1)
        # apply dropout
        out = self.dropout(out)

        sequence_representation = self.fc1_activation(self.fc1(out))

        # fully-connected layer
        return SequenceRepresentation(sequence_representation)


class CaserProjectionLayer(ProjectionLayer):

    """
    TODO: rename layer and move
    """

    def __init__(self,
                 item_vocab_size: int,
                 embedding_size: int,
                 ):
        super().__init__()

        # W2, b2 are encoded with nn.Embedding, as we don't need to compute scores for all items
        self.W2 = nn.Embedding(item_vocab_size, embedding_size)
        self.b2 = nn.Embedding(item_vocab_size, 1)

        # init weights
        self.W2.weight.data.normal_(0, 1.0 / self.W2.embedding_dim)
        self.b2.weight.data.zero_()

    def forward(self,
                modified_sequence_representation: ModifiedSequenceRepresentation
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        #if not self.training:
        #    w2 = pos_w2.squeeze()
        #    b2 = pos_b2.squeeze()
        #    w2 = w2.permute(1, 0, 2)
        #    return torch.matmul(x, w2).sum(dim=1) + b2
        input_sequence = modified_sequence_representation.input_sequence
        positive_samples = input_sequence.get_attribute("positive_samples")
        negative_samples = input_sequence.get_attribute("negative_samples")

        sequence_representation = modified_sequence_representation.modified_encoded_sequence
        sequence_representation = sequence_representation.unsqueeze(2)
        res_pos = self._calc_scores(positive_samples, sequence_representation)
        if negative_samples is None:
            return res_pos

        # negative items
        res_negative = self._calc_scores(negative_samples, sequence_representation)
        return res_pos, res_negative

    def _calc_scores(self,
                     item: torch.Tensor,
                     sequence_representation: torch.Tensor
                     ) -> torch.Tensor:
        w2 = self.W2(item)  # (N, I, F_D)
        b2 = self.b2(item)  # (N, I, 1)

        # TODO: use torch.einsum('nif,nf -> ni', w2, x) + b2.squeeze()
        return torch.baddbmm(b2, w2, sequence_representation).squeeze()


class CaserModel(SequenceRecommenderModel):
    """
        implementation of the Caser model proposed in
        Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding, Jiaxi Tang and Ke Wang, WSDM'18
        see https://doi.org/10.1145/3159652.3159656 for more details

        adapted from the original pytorch implementation: https://github.com/graytowne/caser_pytorch
    """

    @save_hyperparameters
    def __init__(self,
                 embedding_size: int,
                 item_vocab_size: int,
                 user_vocab_size: int,
                 max_seq_length: int,
                 num_vertical_filters: int,
                 num_horizontal_filters: int,
                 conv_activation_fn: str,
                 fc_activation_fn: str,
                 dropout: float,
                 embedding_pooling_type: str = None
                 ):
        item_embedding = SequenceElementsEmbeddingComponent(vocabulary_size=item_vocab_size,
                                                            embedding_size=embedding_size,
                                                            pooling_type=embedding_pooling_type)

        user_present = user_vocab_size != 0
        seq_rep_layer = CaserSequenceRepresentationLayer(embedding_size, max_seq_length, num_vertical_filters,
                                                         num_horizontal_filters, conv_activation_fn, fc_activation_fn,
                                                         dropout)
        if user_present:
            mod_layer = UserEmbeddingConcatModifier(user_vocab_size, embedding_size)
        else:
            mod_layer = IdentitySequenceRepresentationModifierLayer()

        projection_layer = CaserProjectionLayer(item_vocab_size, 2 * embedding_size if user_present else embedding_size)
        super().__init__(item_embedding, seq_rep_layer, mod_layer, projection_layer)

    def optional_metadata_keys(self) -> List[str]:
        return [USER_ENTRY_NAME]


class CaserHorizontalConvNet(nn.Module):
    """
    the horizontal convolution module for the Caser model
    """
    def __init__(self,
                 num_filters: int,
                 kernel_size: Tuple[int, int],
                 activation_fn: str,
                 max_length: int
                 ):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=kernel_size)
        self.conv_activation = get_activation_layer(activation_fn)
        length = kernel_size[0]
        self.conv_pooling = nn.MaxPool1d(kernel_size=max_length + 1 - length)

    def forward(self,
                input_tensor: torch.Tensor):
        conv_out = self.conv(input_tensor).squeeze(3)
        conv_out = self.conv_activation(conv_out)
        return self.conv_pooling(conv_out).squeeze(2)

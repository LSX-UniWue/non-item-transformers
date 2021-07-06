import torch
import torch.nn as nn
from typing import List

from asme.models.caser.caser_model import UserEmbeddingConcatModifier, CaserProjectionLayer
from asme.models.common.components.sequence_embedding import SequenceElementsEmbeddingComponent
from asme.models.common.layers.data.sequence import EmbeddedElementsSequence, SequenceRepresentation
from asme.models.common.layers.layers import IdentitySequenceRepresentationModifierLayer
from asme.models.common.layers.util_layers import get_activation_layer
from asme.models.sequence_recommendation_model import SequenceRecommenderModel, SequenceRepresentationLayer

from data.datasets import USER_ENTRY_NAME


class CosRecSequenceRepresentationLayer(SequenceRepresentationLayer):

    def __init__(self,
                 embedding_size: int,
                 block_num: int,
                 block_dim: List[int],
                 fc_dim: int,
                 activation_function: str,
                 dropout: float):
        super().__init__()

        # TODO: why do we need the block_num parameter?
        assert len(block_dim) == block_num

        # build cnn block
        block_dim.insert(0, 2 * embedding_size)  # adds first input dimension of first cnn block
        # holds submodules in a list
        self.cnnBlock = nn.ModuleList(CNNBlock(block_dim[i], block_dim[i + 1]) for i in range(block_num))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.cnn_out_dim = block_dim[-1]  # dimension of output of last cnn block

        # dropout and fc layer
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.cnn_out_dim, fc_dim)
        self.activation_function = get_activation_layer(activation_function)

    def forward(self, embedded_sequence: EmbeddedElementsSequence) -> SequenceRepresentation:
        sequence = embedded_sequence.embedded_sequence
        sequence_shape = sequence.size()

        batch_size = sequence_shape[0]
        max_sequence_length = sequence_shape[1]
        # cast all item embeddings pairs against each other
        item_i = torch.unsqueeze(sequence, 1)  # (N, 1, S, D)
        item_i = item_i.repeat(1, max_sequence_length, 1, 1)  # (N, S, S, D)
        item_j = torch.unsqueeze(sequence, 2)  # (N, S, 1, D)
        item_j = item_j.repeat(1, 1, max_sequence_length, 1)  # (N, S, S, D)
        all_embed = torch.cat([item_i, item_j], 3)  # (N, S, S, 2*D)
        out = all_embed.permute(0, 3, 1, 2)  # (N, 2 * D, S, S)

        # 2D CNN
        for cnn_block in self.cnnBlock:
            out = cnn_block(out)

        out = self.avg_pool(out).reshape(batch_size, self.cnn_out_dim)  # (N, C_O)
        out = out.squeeze(-1).squeeze(-1)

        # apply fc and dropout
        out = self.activation_function(self.fc1(out))  # (N, F_D)
        representation = self.dropout(out)

        return SequenceRepresentation(representation)


class CosRecModel(SequenceRecommenderModel):
    """
    A 2D CNN for sequential Recommendation.
    Based on paper "CosRec: 2D Convolutional Neural Networks for Sequential Recommendation" which can be found at
    https://dl.acm.org/doi/10.1145/3357384.3358113.
    Original code used for this model is available at: https://github.com/zzxslp/CosRec.

    Args:
        user_vocab_size: number of users.
        item_vocab_size: number of items.
        max_seq_length: length of sequence, Markov order.
        embed_dim: dimensions for user and item embeddings. (latent dimension in paper)
        block_num: number of cnn blocks. (convolutional layers??)
        block_dim: the dimensions for each block. len(block_dim)==block_num -> List
        fc_dim: dimension of the first fc layer, mainly for dimension reduction after CNN.
        activation_function: type of activation functions (string) to use for the output fcn
        dropout: dropout ratio.
    """

    def __init__(self,
                 user_vocab_size: int,
                 item_vocab_size: int,
                 max_seq_length: int,
                 embed_dim: int,
                 block_num: int,
                 block_dim: List[int],
                 fc_dim: int,
                 activation_function: str,
                 dropout: float,
                 embedding_pooling_type: str = None
                 ):
        user_present = user_vocab_size != 0

        item_embedding = SequenceElementsEmbeddingComponent(vocabulary_size=item_vocab_size,
                                                            embedding_size=embed_dim,
                                                            pooling_type=embedding_pooling_type)

        seq_rep_layer = CosRecSequenceRepresentationLayer(embed_dim, block_num, block_dim, fc_dim, activation_function,
                                                          dropout)
        mod_layer = UserEmbeddingConcatModifier(user_vocab_size, embed_dim) if user_present else IdentitySequenceRepresentationModifierLayer()

        repesentation_size = fc_dim + embed_dim if user_present else fc_dim

        projection_layer = CaserProjectionLayer(item_vocab_size, repesentation_size)

        super().__init__(item_embedding, seq_rep_layer, mod_layer, projection_layer)

        # user and item embeddings
        item_embedding.elements_embedding.get_weight().data.normal_(0, 1.0 / embed_dim)

    def optional_metadata_keys(self) -> List[str]:
        return [USER_ENTRY_NAME]


class CNNBlock(nn.Module):
    """
    CNN block with two layers.
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int
                 ):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size=1)
        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm2d(output_dim)
        self.batch_norm2 = nn.BatchNorm2d(output_dim)

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        out = self.relu(self.batch_norm1(self.conv1(x)))
        return self.relu(self.batch_norm2(self.conv2(out)))

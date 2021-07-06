from typing import Optional, List, Union, Tuple

import torch
import torch.nn as nn

from asme.models.common.layers.data.sequence import InputSequence, EmbeddedElementsSequence, SequenceRepresentation, \
    ModifiedSequenceRepresentation
from asme.models.common.layers.layers import SequenceElementsRepresentationLayer, SequenceRepresentationLayer,\
    IdentitySequenceRepresentationModifierLayer, ProjectionLayer
from asme.models.common.layers.sequence_embedding import SequenceElementsEmbeddingLayer
from asme.models.sequence_recommendation_model import SequenceRecommenderModel
from asme.utils.hyperparameter_utils import save_hyperparameters
from data.datasets import USER_ENTRY_NAME


class HGNEEmbeddedElementsSequence(EmbeddedElementsSequence):

    embedded_user: Optional[torch.Tensor] = None
    """
    the representation of an the user that interacted with the sequence :math: `(N, H)`
    """


class HGNEmbeddingLayer(SequenceElementsRepresentationLayer):

    def __init__(self,
                 user_vocab_size: int,
                 dims: int,
                 item_embedding_layer: nn.Module  # TODO: change
                 ):
        super().__init__()

        self.item_embedding_layer = item_embedding_layer

        self.user_embeddings = nn.Embedding(user_vocab_size, dims)

        # init weights
        self.user_embeddings.weight.data.normal_(0, 1.0 / dims)

    def forward(self, sequence: InputSequence) -> HGNEEmbeddedElementsSequence:
        item_embedding = self.item_embedding_layer(sequence.sequence)

        sequence_result = HGNEEmbeddedElementsSequence(item_embedding)

        if sequence.has_attribute(USER_ENTRY_NAME):
            user = sequence.get_attribute(USER_ENTRY_NAME)
            user_embedding = self.user_embeddings(user)
            sequence_result.embedded_user = user_embedding

        return sequence_result


# XXX: does python support generics?!
class HGNSequenceRepresentationLayer(SequenceRepresentationLayer):

    def __init__(self,
                 hidden_size: int,
                 num_successive_items: int,
                 ):
        super().__init__()
        self.feature_gate_item = nn.Linear(hidden_size, hidden_size)
        self.feature_gate_user = nn.Linear(hidden_size, hidden_size)

        self.instance_gate_item = nn.Parameter(torch.zeros(hidden_size, 1, dtype=torch.float))
        self.instance_gate_user = nn.Parameter(torch.zeros(hidden_size, num_successive_items, dtype=torch.float))

        # weight initialization
        self.instance_gate_item = nn.init.xavier_uniform_(self.instance_gate_item)
        self.instance_gate_user = nn.init.xavier_uniform_(self.instance_gate_user)

    def forward(self, embedded_sequence: HGNEEmbeddedElementsSequence) -> SequenceRepresentation:
        item_embedding = embedded_sequence.embedded_sequence

        user_embedding = embedded_sequence.embedded_user
        user_provided = user_embedding is not None

        # feature gating to select salient latent features of items
        param = self.feature_gate_item(item_embedding)
        if user_provided:
            param += self.feature_gate_user(user_embedding).unsqueeze(1)  # personalized feature gating
        gate = torch.sigmoid(param)
        gated_item = item_embedding * gate  # (N, S, D)

        # instance gating to select the informative items
        secondparam = self.instance_gate_item.unsqueeze(0).squeeze()
        if user_provided:
            secondparam += user_embedding.mm(self.instance_gate_user)  # personalized instance gating
        instance_score = torch.sigmoid(torch.matmul(gated_item, secondparam))  # (N, S)
        union_out = gated_item * instance_score.unsqueeze(2)
        union_out = torch.sum(union_out, dim=1)
        sequence_representation = union_out / torch.sum(instance_score, dim=1).unsqueeze(1)  # (N, D)
        return SequenceRepresentation(sequence_representation)


class HGNProjectionLayer(ProjectionLayer):

    def __init__(self,
                 item_vocab_size: int,
                 hidden_size: int
                 ):
        super(HGNProjectionLayer, self).__init__()

        # TODO: the padding id can be not zero
        self.W2 = nn.Embedding(item_vocab_size, hidden_size, padding_idx=0)
        self.b2 = nn.Embedding(item_vocab_size, 1, padding_idx=0)

        self.W2.weight.data.normal_(0, 1.0 / self.W2.embedding_dim)
        self.b2.weight.data.zero_()

    def forward(self,
                modified_sequence_representation: ModifiedSequenceRepresentation
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        sequence = modified_sequence_representation.sequence_representation.embedded_elements_sequence
        item_embedding = sequence.embedded_sequence
        user_embedding = sequence.embedded_user
        sequence_representation = modified_sequence_representation.modified_encoded_sequence

        input_sequence = modified_sequence_representation.input_sequence
        positive_samples = input_sequence.get_attribute("positive_samples")
        negative_samples = input_sequence.get_attribute("negative_samples")

        positive_item_score = self._calc_scores(positive_samples, sequence_representation, item_embedding,
                                                user_embedding)

        if negative_samples is None:
            return positive_item_score  # (N, I)

        negative_item_score = self._calc_scores(negative_samples, sequence_representation, item_embedding,
                                                user_embedding)

        return positive_item_score, negative_item_score

    def _calc_scores(self,
                     items: torch.Tensor,
                     sequence_representation: torch.Tensor,
                     item_embedding: torch.Tensor,
                     user_embedding: torch.Tensor
                     ) -> torch.Tensor:
        w2 = self.W2(items)  # (N, I, D) for train
        b2 = self.b2(items)  # (N, I, 1) for train

        # matrix factorization
        if user_embedding is not None:
            res = torch.baddbmm(b2, w2, user_embedding.unsqueeze(2)).squeeze()
            # union-level
            res += torch.bmm(sequence_representation.unsqueeze(1), w2.permute(0, 2, 1)).squeeze()
        else:
            res = torch.baddbmm(b2, w2, sequence_representation.unsqueeze(2)).squeeze()  # (N, I)

        # item-item product (to model the relation between two single items)
        rel_score = item_embedding.bmm(w2.permute(0, 2, 1))  # (N, S, I)
        rel_score = torch.sum(rel_score, dim=1)  # (N, I)

        return res + rel_score


class HGNModel(SequenceRecommenderModel):

    """
    The HGN model for Sequential Recommendation

    The implementation of the paper:
    Chen Ma, Peng Kang, and Xue Liu, "Hierarchical Gating Networks for Sequential Recommendation",
    in the 25th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD 2019)
    Arxiv: https://arxiv.org/abs/1906.09217
    """

    @save_hyperparameters
    def __init__(self,
                 user_vocab_size: int,
                 item_vocab_size: int,
                 num_successive_items: int,
                 dims: int,
                 embedding_pooling_type: str = None
                 ):
        """
        inits the HGN model
        :param user_vocab_size: the number of users in the dataset
        :param item_vocab_size: the number of items in the dataset (I)
        :param num_successive_items: the number of successive items
        :param dims: the dimension of the item embeddings (D)
        :param embedding_pooling_type: average or max (seems to be optional?)
        """
        item_embedding_layer = SequenceElementsEmbeddingLayer(item_vocab_size, dims, embedding_pooling_type=embedding_pooling_type)

        seq_elements_embedding = HGNEmbeddingLayer(user_vocab_size, dims, item_embedding_layer)

        seq_rep_layer = HGNSequenceRepresentationLayer(dims, num_successive_items)

        projection_layer = HGNProjectionLayer(item_vocab_size, dims)

        super().__init__(seq_elements_embedding,
                         seq_rep_layer,
                         IdentitySequenceRepresentationModifierLayer(),
                         projection_layer)

        item_embedding_layer.embedding.weight.data.normal_(0, 1.0 / dims)

    def optional_metadata_keys(self) -> List[str]:
        return [USER_ENTRY_NAME]

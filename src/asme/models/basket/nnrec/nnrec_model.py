from typing import List

from asme.models.common.layers.data.sequence import EmbeddedElementsSequence, SequenceRepresentation
from asme.models.sequence_recommendation_model import SequenceRecommenderModel, SequenceRepresentationLayer
from data.datasets import USER_ENTRY_NAME
from torch import nn

from asme.models.common.layers.layers import LinearProjectionLayer, IdentitySequenceRepresentationModifierLayer
from asme.models.common.layers.sequence_embedding import SequenceElementsEmbeddingLayer
from asme.utils.hyperparameter_utils import save_hyperparameters


class NNRecSequenceRepresentationLayer(SequenceRepresentationLayer):

    def __init__(self,
                 embedding_size: int,
                 hidden_size: int):
        super().__init__()
        self.hidden_layer = nn.Linear(embedding_size, hidden_size)
        self.act1 = nn.Tanh()

    def forward(self,
                embedded_sequence: EmbeddedElementsSequence
                ) -> SequenceRepresentation:
        sequence = embedded_sequence.embedded_sequence
        return SequenceRepresentation(self.act1(self.hidden_layer(sequence)))


class NNRecModel(SequenceRecommenderModel):

    """
    NNRec implementation from the paper "Next Basket Recommendation with Neural Networks"

    http://ceur-ws.org/Vol-1441/recsys2015_poster15.pdf
    """

    @save_hyperparameters
    def __init__(self,
                 item_vocab_size: int,
                 user_vocab_size: int,
                 item_embedding_size: int,
                 user_embedding_size: int,
                 hidden_size: int,  # l in the paper
                 max_sequence_length: int,  # k, number of last baskets in the paper
                 embedding_pooling_type: str
                 ):
        # layer 1 in the paper
        seq_rep_layer = NNRecSequenceRepresentationLayer(max_sequence_length * item_embedding_size + user_embedding_size, hidden_size)

        projection_layer = LinearProjectionLayer(hidden_size, item_vocab_size) # layer 2 in the paper
        super().__init__(None, seq_rep_layer, IdentitySequenceRepresentationModifierLayer(), projection_layer)

        self.item_embedding = SequenceElementsEmbeddingLayer(item_vocab_size, item_embedding_size,
                                                             embedding_pooling_type=embedding_pooling_type)
        # batch_size = embedded_items.size()[0]
        # embedded_items = embedded_items.view(batch_size, -1)

        # if user is not None:
        #    embedded_user = self.user_embedding(user)
        #    overall_representation = torch.cat([embedded_user, embedded_items])
        # else:
        #    overall_representation = embedded_items

        # if we have no user the embedding can be ignored
        self.user_embedding_size = user_embedding_size
        if user_embedding_size > 0:
            self.user_embedding = nn.Embedding(user_vocab_size, user_embedding_size)

    def optional_metadata_keys(self) -> List[str]:
        return [USER_ENTRY_NAME]

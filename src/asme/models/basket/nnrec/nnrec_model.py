from typing import List

from asme.models.basket.nnrec.components import NNRecSequenceRepresentationComponent
from asme.models.common.components.representations.sequence_elements_embedding_user import \
    SequenceElementsEmbeddingWithUserEmbeddingComponent
from asme.models.common.layers.sequence_embedding import SequenceElementsEmbeddingLayer
from asme.models.sequence_recommendation_model import SequenceRecommenderModel
from data.datasets import USER_ENTRY_NAME

from asme.models.common.layers.layers import LinearProjectionLayer, IdentitySequenceRepresentationModifierLayer
from asme.utils.hyperparameter_utils import save_hyperparameters


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
        item_embedding_layer = SequenceElementsEmbeddingLayer(item_vocab_size, item_embedding_size,
                                                              embedding_pooling_type=embedding_pooling_type)

        seq_elements_embedding = SequenceElementsEmbeddingWithUserEmbeddingComponent(user_vocab_size,
                                                                                     user_embedding_size,
                                                                                     item_embedding_layer)

        # layer 1 in the paper
        seq_rep_layer = NNRecSequenceRepresentationComponent(max_sequence_length * item_embedding_size + user_embedding_size, hidden_size)

        projection_layer = LinearProjectionLayer(hidden_size, item_vocab_size)  # layer 2 in the paper
        super().__init__(seq_elements_embedding, seq_rep_layer, IdentitySequenceRepresentationModifierLayer(),
                         projection_layer)

    def optional_metadata_keys(self) -> List[str]:
        return [USER_ENTRY_NAME]

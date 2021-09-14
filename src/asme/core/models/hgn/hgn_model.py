from typing import List

from asme.core.models.common.components.representations.sequence_elements_embedding_user import \
    SequenceElementsEmbeddingWithUserEmbeddingComponent

from asme.core.models.common.layers.layers import IdentitySequenceRepresentationModifierLayer
from asme.core.models.common.layers.sequence_embedding import SequenceElementsEmbeddingLayer
from asme.core.models.hgn.components import HGNProjectionComponent, HGNSequenceRepresentationComponent
from asme.core.models.sequence_recommendation_model import SequenceRecommenderModel
from asme.core.utils.hyperparameter_utils import save_hyperparameters
from asme.data.datasets import USER_ENTRY_NAME


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
        item_embedding_layer = SequenceElementsEmbeddingLayer(item_vocab_size, dims,
                                                              embedding_pooling_type=embedding_pooling_type)

        seq_elements_embedding = SequenceElementsEmbeddingWithUserEmbeddingComponent(user_vocab_size, dims,
                                                                                     item_embedding_layer)

        seq_rep_layer = HGNSequenceRepresentationComponent(dims, num_successive_items)

        projection_layer = HGNProjectionComponent(item_vocab_size, dims)

        super().__init__(seq_elements_embedding,
                         seq_rep_layer,
                         IdentitySequenceRepresentationModifierLayer(),
                         projection_layer)

        item_embedding_layer.embedding.weight.data.normal_(0, 1.0 / dims)

    def optional_metadata_keys(self) -> List[str]:
        return [USER_ENTRY_NAME]

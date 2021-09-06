from typing import List

from asme.models.caser.components import CaserSequenceRepresentationComponent

from asme.models.common.components.projection.sparse_projection_component import SparseProjectionComponent
from asme.models.common.components.representation_modifier.user_embedding_concat_modifier import \
    UserEmbeddingConcatModifier
from asme.models.common.components.representations.sequence_embedding import SequenceElementsEmbeddingComponent
from asme.models.common.layers.layers import IdentitySequenceRepresentationModifierLayer
from asme.models.sequence_recommendation_model import SequenceRecommenderModel
from asme.utils.hyperparameter_utils import save_hyperparameters
from data.datasets import USER_ENTRY_NAME


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
        seq_rep_layer = CaserSequenceRepresentationComponent(embedding_size, max_seq_length, num_vertical_filters,
                                                             num_horizontal_filters, conv_activation_fn, fc_activation_fn,
                                                             dropout)
        if user_present:
            mod_layer = UserEmbeddingConcatModifier(user_vocab_size, embedding_size)
        else:
            mod_layer = IdentitySequenceRepresentationModifierLayer()

        projection_layer = SparseProjectionComponent(item_vocab_size, 2 * embedding_size if user_present else embedding_size)
        super().__init__(item_embedding, seq_rep_layer, mod_layer, projection_layer)

        # init layers
        if user_present:
            mod_layer.user_embedding.weight.data.normal_(0, 1.0 / embedding_size)

        item_embedding.elements_embedding.embedding.weight.data.normal_(0, 1.0 / embedding_size)
        projection_layer.W2.weight.data.normal_(0, 1.0 / embedding_size)
        projection_layer.b2.weight.data.zero_()

    def optional_metadata_keys(self) -> List[str]:
        return [USER_ENTRY_NAME]

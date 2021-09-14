from asme.core.models.common.components.representations.sequence_embedding import SequenceElementsEmbeddingComponent
from asme.core.models.common.layers.layers import IdentitySequenceRepresentationModifierLayer
from asme.core.models.narm.components import NARMSequenceRepresentationComponent, BilinearProjectionComponent
from asme.core.models.sequence_recommendation_model import SequenceRecommenderModel
from asme.core.utils.hyperparameter_utils import save_hyperparameters


class NarmModel(SequenceRecommenderModel):
    """
        Implementation of "Neural Attentive Session-based Recommendation." (https://dl.acm.org/doi/10.1145/3132847.3132926).

        See https://github.com/lijingsdu/sessionRec_NARM for the original Theano implementation.

        Shapes:
        * N - batch size
        * I - number of items
        * E - item embedding size
        * H - representation size of the encoder
        * S - sequence length
    """

    @save_hyperparameters
    def __init__(self,
                 item_vocab_size: int,
                 item_embedding_size: int,
                 global_encoder_size: int,
                 global_encoder_num_layers: int,
                 embedding_dropout: float,
                 context_dropout: float,
                 batch_first: bool = True,
                 embedding_pooling_type: str = None):

        """
        :param item_vocab_size: number of items (I)
        :param item_embedding_size: item embedding size (E)
        :param global_encoder_size: hidden size of the GRU used as the encoder (H)
        :param global_encoder_num_layers: number of layers of the encoder GRU
        :param embedding_dropout: dropout applied after embedding the items
        :param context_dropout: dropout applied on the full context representation
        :param batch_first: whether data is batch first.
        :param embedding_pooling_type: the embedding mode to use if multiple items per
        """

        sequence_embedding_layer = SequenceElementsEmbeddingComponent(
            vocabulary_size=item_vocab_size,
            embedding_size=item_embedding_size,
            pooling_type=embedding_pooling_type,
            dropout=embedding_dropout
        )

        sequence_representation_layer = NARMSequenceRepresentationComponent(
            item_embedding_size=item_embedding_size,
            global_encoder_size=global_encoder_size,
            global_encoder_num_layers=global_encoder_num_layers,
            context_dropout=context_dropout,
            batch_first=batch_first
        )

        sequence_representation_modifier_layer = IdentitySequenceRepresentationModifierLayer()
        projection_layer = BilinearProjectionComponent(embedding_layer=sequence_embedding_layer.elements_embedding,
                                                       encoded_representation_size=2 * global_encoder_size)

        super().__init__(sequence_embedding_layer,
                         sequence_representation_layer,
                         sequence_representation_modifier_layer,
                         projection_layer)

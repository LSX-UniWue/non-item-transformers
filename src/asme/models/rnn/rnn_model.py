from asme.models.common.components.sequence_embedding import SequenceElementsEmbeddingComponent
from asme.models.common.layers.layers import PROJECT_TYPE_LINEAR
from asme.models.rnn.components import RNNSequenceRepresentationComponent, RNNProjectionComponent, RNNPoolingComponent
from asme.models.sequence_recommendation_model import SequenceRecommenderModel
from asme.utils.hyperparameter_utils import save_hyperparameters


class RNNModel(SequenceRecommenderModel):

    @save_hyperparameters
    def __init__(self,
                 cell_type: str,
                 item_vocab_size: int,
                 item_embedding_dim: int,
                 hidden_size: int,
                 num_layers: int,
                 dropout: float,
                 bidirectional: bool = False,
                 nonlinearity: str = None,  # for Elman RNN
                 embedding_pooling_type: str = None,
                 project_layer_type: str = PROJECT_TYPE_LINEAR):

        sequence_embedding_component = SequenceElementsEmbeddingComponent(vocabulary_size=item_vocab_size,
                                                                          embedding_size=item_embedding_dim,
                                                                          pooling_type=embedding_pooling_type)

        sequence_representation_component = RNNSequenceRepresentationComponent(cell_type,
                                                                               item_embedding_dim,
                                                                               hidden_size,
                                                                               num_layers,
                                                                               dropout,
                                                                               bidirectional,
                                                                               nonlinearity)
        pooling_component = RNNPoolingComponent(bidirectional)

        projection_component = RNNProjectionComponent(sequence_embedding_component.elements_embedding.embedding,
                                                      item_vocab_size,
                                                      hidden_size,
                                                      bidirectional,
                                                      project_layer_type)

        super().__init__(sequence_embedding_layer=sequence_embedding_component,
                         sequence_representation_layer=sequence_representation_component,
                         sequence_representation_modifier_layer=pooling_component,
                         projection_layer=projection_component)



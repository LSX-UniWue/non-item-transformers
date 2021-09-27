import functools

from asme.core.models.bert4rec.components import BidirectionalTransformerSequenceRepresentationComponent
from asme.core.models.common.components.representation_modifier.ffn_modifier import \
    FFNSequenceRepresentationModifierComponent
from torch import nn

from asme.core.models.common.layers.layers import build_projection_layer
from asme.core.models.common.layers.transformer_layers import TransformerEmbedding
from asme.core.models.sequence_recommendation_model import SequenceRecommenderModel
from asme.core.utils.hyperparameter_utils import save_hyperparameters


class BERT4RecModel(SequenceRecommenderModel):
    """
        implementation of the paper "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations
        from Transformer"
        see https://doi.org/10.1145%2f3357384.3357895 for more details.
        Using own transformer implementation to be able to pass batch first tensors to the model
    """
    @save_hyperparameters
    def __init__(self,
                 transformer_hidden_size: int,
                 num_transformer_heads: int,
                 num_transformer_layers: int,
                 item_vocab_size: int,
                 max_seq_length: int,
                 transformer_dropout: float,
                 project_layer_type: str = 'transpose_embedding',
                 embedding_pooling_type: str = None,
                 initializer_range: float = 0.02,
                 transformer_intermediate_size: int = None,
                 transformer_attention_dropout: float = None
                 ):
        embedding_layer = TransformerEmbedding(item_vocab_size, max_seq_length, transformer_hidden_size,
                                               transformer_dropout, embedding_pooling_type)

        representation_layer = BidirectionalTransformerSequenceRepresentationComponent(transformer_hidden_size,
                                                                                       num_transformer_heads,
                                                                                       num_transformer_layers,
                                                                                       transformer_dropout,
                                                                                       transformer_attention_dropout,
                                                                                       transformer_intermediate_size)

        transform_layer = FFNSequenceRepresentationModifierComponent(transformer_hidden_size)

        projection_layer = build_projection_layer(project_layer_type, transformer_hidden_size, item_vocab_size,
                                                  embedding_layer.item_embedding.embedding)

        super().__init__(embedding_layer, representation_layer, transform_layer, projection_layer)

        # init the parameters
        self.apply(functools.partial(normal_initialize_weights, initializer_range=initializer_range))


def normal_initialize_weights(module: nn.Module, initializer_range: float = 0.2) -> None:
    is_linear_layer = isinstance(module, nn.Linear)
    is_embedding_layer = isinstance(module, nn.Embedding)
    if is_linear_layer or is_embedding_layer:
        module.weight.data.normal_(mean=0.0, std=initializer_range)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if is_linear_layer and module.bias is not None:
        module.bias.data.zero_()

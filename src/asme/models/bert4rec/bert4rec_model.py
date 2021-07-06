import functools
from typing import Optional

from torch import nn

from asme.models.common.layers.data.sequence import SequenceRepresentation, ModifiedSequenceRepresentation, \
    EmbeddedElementsSequence
from asme.models.common.layers.layers import build_projection_layer
from asme.models.common.layers.transformer_layers import TransformerLayer, TransformerEmbedding
from asme.models.sequence_recommendation_model import SequenceRecommenderModel, SequenceRepresentationLayer, \
    SequenceRepresentationModifierLayer


class FFNSequenceRepresentationModifierLayer(SequenceRepresentationModifierLayer):
    """
    layer that applies a linear layer and a activation function
    """
    def __init__(self,
                 feature_size: int
                 ):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(feature_size, feature_size),
            nn.GELU(),
            nn.LayerNorm(feature_size)
        )

    def forward(self, sequence_representation: SequenceRepresentation) -> ModifiedSequenceRepresentation:
        transformation = self.transform(sequence_representation.encoded_sequence)
        return ModifiedSequenceRepresentation(transformation)


class BidirectionalTransformerSequenceRepresentationLayer(SequenceRepresentationLayer):
    """
    A representation layer that uses a bidirectional transformer layer(s) to encode the given sequence
    """

    def __init__(self,
                 transformer_hidden_size: int,
                 num_transformer_heads: int,
                 num_transformer_layers: int,
                 transformer_dropout: float,
                 transformer_attention_dropout: Optional[float] = None,
                 transformer_intermediate_size: Optional[int] = None):
        super().__init__()

        if transformer_intermediate_size is None:
            transformer_intermediate_size = 4 * transformer_hidden_size

        self.transformer_encoder = TransformerLayer(transformer_hidden_size, num_transformer_heads,
                                                    num_transformer_layers, transformer_intermediate_size,
                                                    transformer_dropout,
                                                    attention_dropout=transformer_attention_dropout)

    def forward(self, embedded_sequence: EmbeddedElementsSequence) -> SequenceRepresentation:
        sequence = embedded_sequence.embedded_sequence
        padding_mask = embedded_sequence.input_sequence.padding_mask

        attention_mask = None

        if padding_mask is not None:
            attention_mask = padding_mask.unsqueeze(1).repeat(1, sequence.size()[1], 1).unsqueeze(1)

        encoded_sequence = self.transformer_encoder(sequence, attention_mask=attention_mask)
        return SequenceRepresentation(encoded_sequence)


class BERT4RecModel(SequenceRecommenderModel):
    """
        implementation of the paper "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations
        from Transformer"
        see https://doi.org/10.1145%2f3357384.3357895 for more details.
        Using own transformer implementation to be able to pass batch first tensors to the model
    """

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

        representation_layer = BidirectionalTransformerSequenceRepresentationLayer(transformer_hidden_size,
                                                                                   num_transformer_heads,
                                                                                   num_transformer_layers,
                                                                                   transformer_dropout,
                                                                                   transformer_attention_dropout,
                                                                                   transformer_intermediate_size)

        transform_layer = FFNSequenceRepresentationModifierLayer(transformer_hidden_size)

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

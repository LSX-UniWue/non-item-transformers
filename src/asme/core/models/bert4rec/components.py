from typing import Optional

from asme.core.models.common.layers.data.sequence import EmbeddedElementsSequence, SequenceRepresentation
from asme.core.models.common.layers.layers import SequenceRepresentationLayer
from asme.core.models.common.layers.transformer_layers import TransformerLayer


class BidirectionalTransformerSequenceRepresentationComponent(SequenceRepresentationLayer):
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
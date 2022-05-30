import torch

from asme.core.models.common.layers.data.sequence import EmbeddedElementsSequence, SequenceRepresentation
from asme.core.models.common.layers.layers import SequenceRepresentationLayer
from asme.core.models.common.layers.transformer_layers import TransformerLayer
from asme.core.utils.hyperparameter_utils import save_hyperparameters


class TransformerSequenceRepresentationComponent(SequenceRepresentationLayer):

    @save_hyperparameters
    def __init__(self, transformer_layer: TransformerLayer, bidirectional: bool):
        super().__init__()
        self.transformer_layer = transformer_layer
        self.bidirectional = bidirectional

    def forward(self, embedded_sequence: EmbeddedElementsSequence) -> SequenceRepresentation:
        sequence = embedded_sequence.embedded_sequence
        padding_mask = embedded_sequence.input_sequence.padding_mask

        input_size = sequence.size()
        batch_size = input_size[0]
        sequence_length = input_size[1]

        """ 
        We have to distinguish 4 cases here:
            - Bidirectional and no padding mask: Transformer can attend to all tokens with no restrictions
            - Bidirectional and padding mask: Transformer can attend to all tokens but those marked with the padding 
              mask
            - Unidirectional and no padding mask: Transformer can attend to all tokens up to the current sequence index
            - Unidirectional and padding mask: Transformer can attend to all tokens up to the current sequence index
              except those marked by the padding mask
        """
        if self.bidirectional:
            if padding_mask is None:
                attention_mask = None
            else:
                attention_mask = padding_mask.unsqueeze(1).repeat(1, sequence_length, 1).unsqueeze(1)
        else:
            if padding_mask is None:
                attention_mask = torch.tril(
                    torch.ones([sequence_length, sequence_length], device=sequence.device)).unsqueeze(0).repeat(
                    batch_size, 1, 1).unsqueeze(1)
            else:
                attention_mask = torch.tril(
                    torch.ones([sequence_length, sequence_length], device=sequence.device)).unsqueeze(0).repeat(
                    batch_size, 1, 1).unsqueeze(1)
                attention_mask *= padding_mask.unsqueeze(1).repeat(1, sequence_length, 1).unsqueeze(1)

        encoded_sequence = self.transformer_layer(sequence, attention_mask=attention_mask)
        return SequenceRepresentation(encoded_sequence)

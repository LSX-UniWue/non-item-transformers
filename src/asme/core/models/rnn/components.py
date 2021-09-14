from typing import Union, Tuple

import torch
from torch import nn
from torch.nn import Embedding

from asme.core.models.common.layers.data.sequence import EmbeddedElementsSequence, SequenceRepresentation, \
    ModifiedSequenceRepresentation
from asme.core.models.common.layers.layers import SequenceRepresentationLayer, ProjectionLayer, PROJECT_TYPE_LINEAR, \
    build_projection_layer, SequenceRepresentationModifierLayer
from asme.core.models.rnn.util import _build_rnn_cell
from asme.core.models.rnn.layers import RNNPooler


class RNNSequenceRepresentationComponent(SequenceRepresentationLayer):

    def __init__(self,
                 cell_type: str,
                 item_embedding_dim: int,
                 hidden_size: int,
                 num_layers: int,
                 dropout: float,
                 bidirectional: bool = False,
                 nonlinearity: str = None  # for Elman RNN
                 ):

        super().__init__()

        # FIXME: maybe this should not be done here
        rnn_dropout = dropout
        if num_layers == 1 and dropout > 0:
            print("setting the dropout of the rnn to 0 because the number of layers is 1")
            rnn_dropout = 0.0

        self.rnn = _build_rnn_cell(cell_type, item_embedding_dim, hidden_size, num_layers, bidirectional, rnn_dropout,
                                   nonlinearity)

    def forward(self, embedded_sequence: EmbeddedElementsSequence) -> SequenceRepresentation:
        input_sequence = embedded_sequence.input_sequence

        lengths = input_sequence.padding_mask.sum(dim=-1).cpu()  # required by torch >= 1.7, no cuda tensor allowed
        packed_embedded_session = nn.utils.rnn.pack_padded_sequence(
            embedded_sequence.embedded_sequence,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )
        outputs, _ = self.rnn(packed_embedded_session)

        return SequenceRepresentation(outputs, embedded_sequence)


class RNNProjectionComponent(ProjectionLayer):

    def __init__(self,
                 elements_embedding: Embedding,
                 item_vocab_size: int,
                 hidden_size: int,
                 bidirectional: bool = False,
                 project_layer_type: str = PROJECT_TYPE_LINEAR):

        super().__init__()
        hidden_size_projection = hidden_size if not bidirectional else 2 * hidden_size
        self.projection = build_projection_layer(project_layer_type,
                                                 hidden_size_projection,
                                                 item_vocab_size,
                                                 elements_embedding)

    def forward(self, modified_sequence_representation: ModifiedSequenceRepresentation) -> Union[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor]
    ]:
        return self.projection(modified_sequence_representation)


class RNNPoolingComponent(SequenceRepresentationModifierLayer):

    def __init__(self,
                 bidirectional: bool = False):

        super().__init__()
        self.pooling = RNNPooler(bidirectional=bidirectional)

    def forward(self, sequence_representation: SequenceRepresentation) -> ModifiedSequenceRepresentation:
        modified_sequence = self.pooling(sequence_representation.encoded_sequence)
        return ModifiedSequenceRepresentation(modified_sequence, sequence_representation)
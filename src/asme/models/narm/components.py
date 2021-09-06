from typing import Union, Tuple

import torch
from torch import nn as nn

from asme.models.common.layers.data.sequence import EmbeddedElementsSequence, SequenceRepresentation, \
    ModifiedSequenceRepresentation
from asme.models.common.layers.layers import SequenceRepresentationLayer, ProjectionLayer
from asme.models.common.layers.sequence_embedding import SequenceElementsEmbeddingLayer
from asme.models.narm.layers import LocalEncoderLayer, BilinearDecoderLayer


class NARMSequenceRepresentationComponent(SequenceRepresentationLayer):

    def __init__(self,
                 item_embedding_size: int,
                 global_encoder_size: int,
                 global_encoder_num_layers: int,
                 context_dropout: float,
                 batch_first: bool = True):

        super().__init__()
        self.batch_first = batch_first
        self.global_encoder = nn.GRU(item_embedding_size, global_encoder_size,
                                     num_layers=global_encoder_num_layers,
                                     batch_first=batch_first)
        self.local_encoder = LocalEncoderLayer(global_encoder_size, global_encoder_size)
        self.context_dropout = nn.Dropout(context_dropout)

    def forward(self, embedded_sequence: EmbeddedElementsSequence) -> SequenceRepresentation:
        sequence = embedded_sequence.embedded_sequence
        padding_mask = embedded_sequence.input_sequence.padding_mask

        max_seq_length = sequence.size()[1]  # should be S
        lengths = padding_mask.sum(dim=-1).cpu()
        packed_embedded_session = nn.utils.rnn.pack_padded_sequence(
            sequence,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )

        h_i, h_t = self.global_encoder(packed_embedded_session)
        c_tg = h_t = h_t[-1]  # we use the last hidden state of the last layer

        # we only use the hidden size and throw away the lengths, since we already have them
        h_i, _ = nn.utils.rnn.pad_packed_sequence(h_i, batch_first=self.batch_first, total_length=max_seq_length)
        c_tl = self.local_encoder(h_t, h_i, padding_mask)

        c_t = torch.cat([c_tg, c_tl], dim=1)

        c_tdo = self.context_dropout(c_t)

        return SequenceRepresentation(c_tdo, embedded_sequence)


class BilinearProjectionComponent(ProjectionLayer):

    def __init__(self,
                 embedding_layer: SequenceElementsEmbeddingLayer,
                 encoded_representation_size: int,
                 apply_softmax: bool = False):

        super().__init__()
        self.decoder = BilinearDecoderLayer(embedding_layer, encoded_representation_size, apply_softmax)

    def forward(self, modified_sequence_representation: ModifiedSequenceRepresentation) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        sequence = modified_sequence_representation.modified_encoded_sequence
        return self.decoder(sequence)

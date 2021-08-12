from typing import List

import torch
from asme.models.common.layers.data.sequence import EmbeddedElementsSequence, SequenceRepresentation
from asme.models.common.layers.layers import SequenceRepresentationLayer
from asme.models.common.layers.util_layers import get_activation_layer
from asme.models.cosrec.layers import CNNBlock
from torch import nn as nn


class CosRecSequenceRepresentationComponent(SequenceRepresentationLayer):

    def __init__(self,
                 embedding_size: int,
                 block_num: int,
                 block_dim: List[int],
                 fc_dim: int,
                 activation_function: str,
                 dropout: float):
        super().__init__()

        # TODO: why do we need the block_num parameter?
        assert len(block_dim) == block_num

        # build cnn block
        block_dim.insert(0, 2 * embedding_size)  # adds first input dimension of first cnn block
        # holds submodules in a list
        self.cnnBlock = nn.ModuleList(CNNBlock(block_dim[i], block_dim[i + 1]) for i in range(block_num))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.cnn_out_dim = block_dim[-1]  # dimension of output of last cnn block

        # dropout and fc layer
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.cnn_out_dim, fc_dim)
        self.activation_function = get_activation_layer(activation_function)

    def forward(self, embedded_sequence: EmbeddedElementsSequence) -> SequenceRepresentation:
        sequence = embedded_sequence.embedded_sequence
        sequence_shape = sequence.size()

        batch_size = sequence_shape[0]
        max_sequence_length = sequence_shape[1]
        # cast all item embeddings pairs against each other
        item_i = torch.unsqueeze(sequence, 1)  # (N, 1, S, D)
        item_i = item_i.repeat(1, max_sequence_length, 1, 1)  # (N, S, S, D)
        item_j = torch.unsqueeze(sequence, 2)  # (N, S, 1, D)
        item_j = item_j.repeat(1, 1, max_sequence_length, 1)  # (N, S, S, D)
        all_embed = torch.cat([item_i, item_j], 3)  # (N, S, S, 2*D)
        out = all_embed.permute(0, 3, 1, 2)  # (N, 2 * D, S, S)

        # 2D CNN
        for cnn_block in self.cnnBlock:
            out = cnn_block(out)

        out = self.avg_pool(out).reshape(batch_size, self.cnn_out_dim)  # (N, C_O)
        out = out.squeeze(-1).squeeze(-1)

        # apply fc and dropout
        out = self.activation_function(self.fc1(out))  # (N, F_D)
        representation = self.dropout(out)

        return SequenceRepresentation(representation)

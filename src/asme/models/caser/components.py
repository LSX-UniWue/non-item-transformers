import torch
from asme.models.caser.layers import CaserHorizontalConvNet
from asme.models.common.layers.data.sequence import EmbeddedElementsSequence, SequenceRepresentation
from asme.models.common.layers.layers import SequenceRepresentationLayer
from asme.models.common.layers.util_layers import get_activation_layer
from torch import nn


class CaserSequenceRepresentationComponent(SequenceRepresentationLayer):

    def __init__(self,
                 embedding_size: int,
                 max_sequence_length: int,
                 num_vertical_filters: int,
                 num_horizontal_filters: int,
                 conv_activation_fn: str,
                 fc_activation_fn: str,
                 dropout: float
                 ):
        super().__init__()

        self.num_vertical_filters = num_vertical_filters
        self.num_horizontal_filters = num_horizontal_filters

        # vertical conv layer
        self.conv_vertical = nn.Conv2d(1, num_vertical_filters, (max_sequence_length, 1))

        # horizontal conv layer
        lengths = [i + 1 for i in range(max_sequence_length)]

        self.conv_horizontal = nn.ModuleList(
            [CaserHorizontalConvNet(num_filters=num_horizontal_filters,
                                    kernel_size=(length, embedding_size),
                                    activation_fn=conv_activation_fn,
                                    max_length=max_sequence_length)
             for length in lengths]
        )

        # dropout
        self.dropout = nn.Dropout(dropout)

        # fully-connected layer
        self.fc1_dim_vertical = num_vertical_filters * embedding_size
        fc1_dim_horizontal = num_horizontal_filters * len(lengths)
        fc1_dim = self.fc1_dim_vertical + fc1_dim_horizontal

        self.fc1 = nn.Linear(fc1_dim, embedding_size)

        self.fc1_activation = get_activation_layer(fc_activation_fn)

    def forward(self, embedded_sequence: EmbeddedElementsSequence) -> SequenceRepresentation:
        sequence = embedded_sequence.embedded_sequence

        sequence = sequence.unsqueeze(1)

        # Convolutional Layers
        out_vertical = None
        # vertical conv layer
        if self.num_vertical_filters > 0:
            out_vertical = self.conv_vertical(sequence)
            out_vertical = out_vertical.view(-1, self.fc1_dim_vertical)  # prepare for fully connect

        # horizontal conv layer
        out_horizontal = None
        out_hs = list()
        if self.num_horizontal_filters > 0:
            for conv in self.conv_horizontal:
                conv_out = conv(sequence)
                out_hs.append(conv_out)
            out_horizontal = torch.cat(out_hs, 1)

        # Fully-connected Layers
        out = torch.cat([out_vertical, out_horizontal], 1)
        # apply dropout
        out = self.dropout(out)

        sequence_representation = self.fc1_activation(self.fc1(out))

        # fully-connected layer
        return SequenceRepresentation(sequence_representation)

from typing import Tuple

import torch
from asme.core.models.common.layers.util_layers import get_activation_layer
from torch import nn


class CaserHorizontalConvNet(nn.Module):
    """
    the horizontal convolution module for the Caser model
    """
    def __init__(self,
                 num_filters: int,
                 kernel_size: Tuple[int, int],
                 activation_fn: str,
                 max_length: int
                 ):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=kernel_size)
        self.conv_activation = get_activation_layer(activation_fn)
        length = kernel_size[0]
        self.conv_pooling = nn.MaxPool1d(kernel_size=max_length + 1 - length)

    def forward(self,
                input_tensor: torch.Tensor):
        conv_out = self.conv(input_tensor).squeeze(3)
        conv_out = self.conv_activation(conv_out)
        return self.conv_pooling(conv_out).squeeze(2)

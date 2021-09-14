import torch
from torch import nn as nn


class CNNBlock(nn.Module):
    """
    CNN block with two layers.
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int
                 ):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size=1)
        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm2d(output_dim)
        self.batch_norm2 = nn.BatchNorm2d(output_dim)

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        out = self.relu(self.batch_norm1(self.conv1(x)))
        return self.relu(self.batch_norm2(self.conv2(out)))

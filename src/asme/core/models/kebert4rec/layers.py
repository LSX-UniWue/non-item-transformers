import torch
from torch import nn


class LinearUpscaler(nn.Module):

    def __init__(self, vocab_size: int, embed_size: int):
        super().__init__()
        self.linear = nn.Linear(vocab_size, embed_size)
        self.vocab_size = vocab_size

    def forward(self,
                content_input: torch.Tensor
                ) -> torch.Tensor:
        """
        :param content_input: a tensor containing the ids of each
        :return:
        """
        # the input is a sequence of content ids without any order
        # so we convert them into a multi-hot encoding
        multi_hot = torch.nn.functional.one_hot(content_input, self.vocab_size).sum(2).float()
        # 0 is the padding category, so zero it out
        multi_hot[:, :, 0] = 0
        return self.linear(multi_hot)

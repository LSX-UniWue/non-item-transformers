from abc import abstractmethod
from typing import Tuple

import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence

from asme.core.utils.hyperparameter_utils import save_hyperparameters


class LSTMAdapter(nn.Module):
    """
    Changes the output of `torch.nn.LSTM` to comply with the API for `torch.nn.GRU` by omitting the internal
    context states `c_n`.
    """
    @save_hyperparameters
    def __init__(self,
                 item_embedding_size: int,
                 hidden_size: int,
                 num_layers: int,
                 bidirectional: bool,
                 dropout: float
                 ):
        super().__init__()

        self.lstm = nn.LSTM(item_embedding_size, hidden_size,
                            bidirectional=bidirectional,
                            dropout=dropout,
                            num_layers=num_layers,
                            batch_first=True)

    def forward(self,
                packed_embedded_session: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs, final_state = self.lstm(packed_embedded_session)

        # currently we ignore the context information and only return
        # the hidden representation (as the gru model)
        return outputs, final_state[0]


class RNNStatePooler(nn.Module):

    @save_hyperparameters
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self,
                outputs: torch.Tensor,
                ) -> torch.Tensor:
        pass


class RNNPooler(RNNStatePooler):
    """
    Pooling component that can be used to either return state for every step or only the state after the last element
    of the sequence.
    """

    @save_hyperparameters
    def __init__(self,
                 bidirectional: bool = False,
                 parallel: bool = False
                 ):
        """
        Initializes the pooler.

        :param bidirectional: indicates whether the input to the pooler comes from a bidirectional RNN.
        :param parallel: indicates whether states for every step in the sequence should be returned or not.
        """
        super().__init__()

        self.directions = 2 if bidirectional else 1
        self.parallel = parallel

    def forward(self,
                outputs: PackedSequence
                ) -> torch.Tensor:

        sequence, lengths = pad_packed_sequence(outputs, batch_first=True)

        batch_size, seq_len, hidden_size = sequence.size()
        hidden_size = int(hidden_size / self.directions)

        sequence = sequence.view(batch_size, seq_len, self.directions, hidden_size)  # B, S, D, H

        # concatenate states for each direction
        if self.directions == 1:
            sequence = sequence.squeeze()
        else:
            sequence = sequence.view(batch_size, seq_len, self.directions * hidden_size)

        # "pool" the model by either selecting the last state, or if requested returning states for every step.
        if self.parallel:
            return sequence
        else:
            seq_pos_index = lengths - 1
            batch_index = torch.arange(0, batch_size)
            return sequence[batch_index, seq_pos_index]

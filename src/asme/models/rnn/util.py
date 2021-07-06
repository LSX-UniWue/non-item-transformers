from torch import nn

from asme.models.rnn.layers import LSTMAdapter


def _build_rnn_cell(cell_type: str,
                    item_embedding_size: int,
                    hidden_size: int,
                    num_layers: int,
                    bidirectional: bool,
                    dropout: float,
                    nonlinearity: str  # only for Elman RNN
                    ) -> nn.Module:
    if cell_type == 'gru':
        return nn.GRU(item_embedding_size, hidden_size,
                      bidirectional=bidirectional,
                      dropout=dropout,
                      num_layers=num_layers,
                      batch_first=True)

    if cell_type == 'lstm':
        return LSTMAdapter(item_embedding_size, hidden_size,
                           bidirectional=bidirectional,
                           dropout=dropout,
                           num_layers=num_layers)

    if cell_type == 'rnn':
        return nn.RNN(item_embedding_size, hidden_size,
                      bidirectional=bidirectional,
                      dropout=dropout,
                      num_layers=num_layers,
                      nonlinearity=nonlinearity)

    raise ValueError(f'cell type "{cell_type}" not supported')
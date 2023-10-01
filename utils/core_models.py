""" Core models for our experiments."""
import torch
import torch.nn as nn

from typing import List

class BaseRnn(nn.Module):
    """LSTM-based network for experiments with Synthetic Normal data and Human Activity."""
    def __init__(
            self,
            input_size: int,
            hidden_dim: int,
            n_layers: int,
            drop_prob: float,
    ) -> None:
        """Initialize model's parameters.

        :param input_size: size of elements in input sequence
        :param output_size: length of the generated sequence
        :param hidden_dim: size of the hidden layer(-s)
        :param n_layers: number of recurrent layers
        :param drop_prob: dropout probability
        """
        super().__init__()

        self.input_size = input_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)
        self.activation = nn.Sigmoid()

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        """Forward propagation through model.

        :param input_seq: batch of generated sunthetic normal sequences
        :return: probabilities of changes for each sequence
        """
        batch_size = input_seq.size(0)
        lstm_out, hidden = self.lstm(input_seq.float())
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.linear(lstm_out)
        out = self.activation(out)
        out = out.view(batch_size, -1)
        return out
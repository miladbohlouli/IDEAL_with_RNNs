from torch.nn import LSTM, Module, Linear
import torch
import numpy as np

class LSTM_IDEAL(Module):
    def __init__(self, hidden_dim, input_size, output_size):
        super(LSTM_IDEAL, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm_layer = LSTM(input_size, hidden_dim, batch_first=True)
        self.decoder = Linear(hidden_dim, output_size)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(batch_size, self.hidden_dim)
        return hidden

    def forward(self, data):
        h0 = torch.zeros(1, data.size(0), self.hidden_dim)
        c0 = torch.zeros(1, data.size(0), self.hidden_dim)
        lstm_out, _ = self.lstm_layer(data, (h0, c0))
        return self.decoder(lstm_out[:, -1, :])





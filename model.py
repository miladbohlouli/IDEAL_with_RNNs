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
        h0 = torch.zeros(1, batch_size, self.hidden_dim)
        c0 = torch.zeros(1, batch_size, self.hidden_dim)
        return h0, c0

    def forward(self, data, prediction_time_steps):
        predictions = torch.zeros(data.size()[0], prediction_time_steps, 1)
        for i in range(prediction_time_steps):
            lstm_out, states = self.lstm_layer(data, self.init_hidden(data.size(0)))
            new_temp_data = self.decoder(lstm_out[:, -1, :])
            data[:, :-1, :] = data[:, 1:, :]
            data[:, -1, :] = new_temp_data
            predictions[:, i, :] = new_temp_data

        return predictions






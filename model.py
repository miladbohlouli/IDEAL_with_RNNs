from torch.nn import LSTM, Module, Linear
import torch
import numpy as np

class Encoder(Module):
    def __init__(self, hidden_dim, input_size, output_size):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.decoder = LSTM(input_size, hidden_dim, batch_first=True)
        self.hidden2temp = Linear(hidden_dim, output_size)

    def init_hidden(self, batch_size):
        h0 = torch.zeros(1, batch_size, self.hidden_dim)
        c0 = torch.zeros(1, batch_size, self.hidden_dim)
        return h0, c0

    def forward(self, data, prediction_time_steps):
        predictions = torch.zeros(data.size()[0], prediction_time_steps, 1)
        decoder_h = self.init_hidden(data.size()[0])
        decoder_input = data

        for i in range(prediction_time_steps):
            decoder_out, decoder_h = self.decoder(decoder_input, decoder_h)
            new_temp_data = self.hidden2temp(decoder_out[:, -1, :])
            predictions[:, i, :] = new_temp_data
            decoder_input = new_temp_data.unsqueeze(1)
        return predictions







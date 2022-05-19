from torch.nn import LSTM, Module, Linear
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torch.optim import Adam
import torch
import numpy as np
from torch_data_loader import IDEAL_RNN
from model import LSTM_IDEAL

hidden_dims = 64
batch_size = 64
predict_seq_len = 70
train_seq_len = 30
seq_len = 30
seq_stride = 1
lstm_hidden = 16
ontput_size = 1
num_epochs = 10

def train():
    train_dataset = IDEAL_RNN(
        seq_length=seq_len,
        multi_room_training=True,
        stride=100
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size
    )

    input_size = train_dataset[0].shape[-1]

    model = LSTM_IDEAL(
        hidden_dim=lstm_hidden,
        input_size=input_size,
        output_size=1
    ).float()

    loss = MSELoss()
    optimizer = Adam(model.parameters())

    for i in range(num_epochs):
        losses = []
        for sequence in train_loader:
            sequence = sequence.float()
            results = model(sequence[:, :-1, :])
            mse_error = loss(results.ravel(), sequence[:, -1, 0])

            optimizer.zero_grad()
            mse_error.backward()
            optimizer.step()

            losses.append(mse_error.detach().numpy())

        print(f"Epoch ({i}/{num_epochs}\t\tmse_error: {np.mean(losses):2.4})")


if __name__ == '__main__':
    train()
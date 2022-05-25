from torch.nn import LSTM, Module, Linear
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torch.optim import Adam
import torch
import numpy as np
from torch_data_loader import IDEAL_RNN
from model import LSTM_IDEAL
from utils import custom_collate, visulaize, MSE_loss
from matplotlib import pyplot as plt


hidden_dims = 64
batch_size = 64
predict_seq_len = 70
stride = 100
total_seq_len = 30
train_seq_len = 20
seq_stride = 1
lstm_hidden = 16
ontput_size = 1
num_epochs = 10

def train():
    # Some assertions
    assert train_seq_len < total_seq_len

    train_dataset = IDEAL_RNN(
        seq_length=total_seq_len,
        multi_room_training=True,
        stride=stride
    )

    test_dataset = IDEAL_RNN(
        seq_length=total_seq_len,
        multi_room_training=True,
        train=False,
        stride=stride
    )

    train_loader = DataLoader(
        train_dataset,
        collate_fn=custom_collate,
        batch_size=batch_size
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size
    )

    input_size = train_dataset[0][0].shape[-1]

    model = LSTM_IDEAL(
        hidden_dim=lstm_hidden,
        input_size=input_size,
        output_size=1
    ).float()

    loss = MSELoss()
    optimizer = Adam(model.parameters())

    for i in range(num_epochs):
        losses = []
        for (sequences, dates) in train_loader:
            sequences = sequences.float()
            results = model(
                sequences[:, :train_seq_len, :],
                prediction_time_steps = (total_seq_len - train_seq_len))
            mse_error = MSE_loss(results, sequences[:, train_seq_len:])

            optimizer.zero_grad()
            mse_error.backward()
            optimizer.step()

            losses.append(mse_error.detach().numpy())

        # Visualizing the results for train step
        print(f"Epoch ({i}/{num_epochs}\t\tmse_error: {np.mean(losses):2.6f})")

        selected = np.random.randint(0, dates.shape[0])
        visulaize(dates[selected],
                  target=sequences[selected].detach().numpy(),
                  predicted=results[selected].detach().numpy())
        plt.show()


if __name__ == '__main__':
    train()
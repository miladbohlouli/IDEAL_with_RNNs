import logging
import sys

import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torch.optim import Adam
import numpy as np
from torch_data_loader import IDEAL_RNN
from model import Encoder
from utils import custom_collate, visulaize, MSE_loss
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
import pickle as pk
import argparse

arg_params = argparse.ArgumentParser()

global line
line = 20*"*"


def train_evaluate(
        args,
        params,
        temp_dir,
        save_dir
    ):
    assert params["observed_sequence_len"] < params["total_seq_len"]

    train_writer = SummaryWriter(os.path.join(temp_dir, "train"))
    test_writer = SummaryWriter(os.path.join(temp_dir, "test"))
    logging.basicConfig(filename=os.path.join(save_dir, "running.log"),
                                              format='%(asctime)s %(filename)s %(levelname)s: %(message)s',
                                              level=logging.INFO)

    train_dataset = IDEAL_RNN(
        seq_length=params["total_seq_len"],
        sampling_rate=params["sampling_rate"],
        multi_room_training=True,
        stride=params["stride"],
    )

    test_dataset = IDEAL_RNN(
        seq_length=params["total_seq_len"],
        sampling_rate=params["sampling_rate"],
        multi_room_training=True,
        train=False,
        stride=params["stride"]
    )

    train_loader = DataLoader(
        train_dataset,
        collate_fn=custom_collate,
        batch_size=params["batch_size"]
    )

    test_loader = DataLoader(
        test_dataset,
        collate_fn=custom_collate,
        batch_size=params["batch_size"]
    )

    params["input_size"] = train_dataset[0][0].shape[-1]

    model = Encoder(
        hidden_dim=params["lstm_hidden"],
        input_size=params["input_size"],
        output_size=params["output_size"]
    ).float()

    logging.info(f"Experimenting with {len(train_dataset)} train | {len(test_dataset)} samples")
    logging.info(model)
    logging.info(line)

    loss = MSELoss()
    optimizer = Adam(model.parameters())
    step = 0
    start_epoch = 0
    model_saving_dict = {}

    if args.load:
        with open(os.path.join(save_dir, "run_params.pickle"), "rb") as file:
            params = pk.load(file)

        try:
            saved_models_list = [file for file in os.listdir(save_dir) if "checkpoint" in file]
            if len(saved_models_list) == 0:
                logging.info("There are no saved models")
            else:
                model_saving_dict = torch.load(os.path.join(save_dir, sorted(saved_models_list)[-1]))
                logging.info(f"Loading the model from checkpoint {sorted(saved_models_list)[:-1]}")
                start_epoch = model_saving_dict["epoch"] + 1
                optimizer.load_state_dict(model_saving_dict["optimizer_state_dict"])
                model.load_state_dict(model_saving_dict["model_state_dict"])
                step = model_saving_dict["step"]

        except:
            raise "There was a problem loading the model"

    else:
        # saving the parameters
        with open(os.path.join(save_dir, "run_params.pickle"), "wb") as file:
            pk.dump(params, file)

    for i in tqdm(range(start_epoch, start_epoch + params["num_epochs"])):
        model.train()
        train_losses = []
        test_losses = []
        for (sequences, dates) in train_loader:
            sequences = sequences.float()
            results = model(
                sequences[:, :params["observed_sequence_len"], :],
                prediction_time_steps = (params["total_seq_len"] - params["observed_sequence_len"]))
            mse_error = loss(results, sequences[:, params["observed_sequence_len"]:])

            optimizer.zero_grad()
            mse_error.backward()
            optimizer.step()

            train_losses.append(mse_error.detach().numpy())
            train_writer.add_scalar("MSE", mse_error.detach().numpy(), step)
            step += 1

        model.eval()
        for (sequences, dates) in test_loader:
            sequences = sequences.float()
            results = model(
                sequences[:, :params["observed_sequence_len"], :],
                prediction_time_steps=(params["total_seq_len"] - params["observed_sequence_len"]))
            mse_error = loss(results, sequences[:, params["observed_sequence_len"]:])

            test_losses.append(mse_error.detach().numpy())

        # Visualizing the results for train step
        logging.info(f"Epoch ({i+1:3}/{start_epoch + params['num_epochs']:3} | train_mse: {np.mean(train_losses):2.5f} | test_mse: {np.mean(test_losses):2.5f})")

        test_writer.add_scalar("MSE", np.mean(train_losses), step)

        selected = np.random.randint(0, dates.shape[0])
        fig = visulaize(dates[selected],
                  target=test_dataset.rescale(sequences[selected].detach().numpy()),
                  predicted=test_dataset.rescale(results[selected].detach().numpy()),
                  y_lim=[10, 25])

        test_writer.add_figure("test samples", fig, i)

        # saving the model
        if i % args.save_every_n_epochs == 0:
            logging.info(f"Saving the model as {args.tag + '-checkpoint-'+ str(i)}")
            model_saving_dict["epoch"] = i
            model_saving_dict["model_state_dict"] = model.state_dict()
            model_saving_dict["optimizer_state_dict"] = optimizer.state_dict()
            model_saving_dict["step"] = step
            model_saving_dict["train_loss"] = np.mean(train_losses)
            model_saving_dict["test_loss"] = np.mean(test_losses)

            torch.save(model_saving_dict, os.path.join(save_dir, args.tag + "-checkpoint-" + str(i)))


arg_params.add_argument("--load", type=bool, default=True)
arg_params.add_argument("--tag", type=str, default="static_down_sampling")
arg_params.add_argument("--save_every_n_epochs", type=int, default=10)

if __name__ == '__main__':

    args = arg_params.parse_args()
    model_tag = args.tag

    temp_files_list = os.listdir("temp")
    temp_dir = os.path.join("temp", model_tag)

    save_files_list = os.listdir("save")
    save_dir = os.path.join("save", model_tag)

    if model_tag in temp_files_list and not args.load:
        shutil.rmtree(temp_dir)
        os.mkdir(temp_dir)
    elif model_tag not in temp_files_list:
        os.mkdir(temp_dir)

    if model_tag in save_files_list and not args.load:
        shutil.rmtree(save_dir)
    elif model_tag not in save_files_list:
        os.mkdir(save_dir)

    params = dict()
    # Model parameters
    params["lstm_hidden"] = 64
    params["output_size"] = 1

    # Running Parameters
    params["batch_size"] = 64
    params["num_epochs"] = 40

    # Data preparation parameters
    params["stride"] = 100
    params["total_seq_len"] = 50
    params["observed_sequence_len"] = 30
    params["seq_stride"] = 1
    params["sampling_rate"] = 1000

    train_evaluate(
        args,
        params,
        temp_dir,
        save_dir
    )


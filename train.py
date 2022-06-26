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
from ray import tune


global line
line = 20*"*"


def train_evaluate(
        config,
        checkpoint_dir,
        data_dir,
        logging_dir,
        args,
        tuning_mode=False,
    ):
    assert config["observed_sequence_len"] < config["total_seq_len"]

    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"

    device = torch.device(dev)
    logging.info(f"Using the {dev} as the running device")

    train_writer = SummaryWriter(os.path.join(os.path.join(logging_dir, "tensorboard"), "train"))
    test_writer = SummaryWriter(os.path.join(os.path.join(logging_dir, "tensorboard"), "test"))
    logging.basicConfig(filename=os.path.join(logging_dir, "running.log"),
                        format='%(asctime)s %(filename)s %(levelname)s: %(message)s',
                        level=logging.INFO)

    logging.info("Preparing the train loaders...")
    train_dataset = IDEAL_RNN(
        data_path=data_dir,
        multi_room_training=True,
        logger=logging,
        params=config
    )

    train_loader = DataLoader(
        train_dataset,
        collate_fn=custom_collate,
        batch_size=config["batch_size"]
    )

    logging.info("Preparing the test loaders...")
    test_dataset = IDEAL_RNN(
        data_path=data_dir,
        multi_room_training=True,
        logger=logging,
        train=False,
        params=config
    )

    test_loader = DataLoader(
        test_dataset,
        collate_fn=custom_collate,
        batch_size=config["batch_size"]
    )

    config["input_size"] = train_dataset[0][0].shape[-1]

    model = Encoder(
        hidden_dim=config["lstm_hidden"],
        input_size=config["input_size"],
        output_size=config["output_size"],
        device=device 
    ).float()

    logging.info(f"Experimenting with {len(train_dataset)} train | {len(test_dataset)} samples")
    logging.info(model)
    logging.info(line)

    loss = MSELoss()
    optimizer = Adam(model.parameters())
    step = 0
    start_epoch = 0
    model_saving_dict = {}

    if checkpoint_dir:
        with open(os.path.join(logging_dir, "run_params.pickle"), "rb") as file:
            config = pk.load(file)

        model_saving_dict = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
        logging.info(f"Loading the model from checkpoint {os.path.join(checkpoint_dir, 'checkpoint')}")
        start_epoch = model_saving_dict["epoch"] + 1
        optimizer.load_state_dict(model_saving_dict["optimizer_state_dict"])
        model.load_state_dict(model_saving_dict["model_state_dict"])
        step = model_saving_dict["step"]

    model = model.to(device)
    for i in range(start_epoch, start_epoch + config["num_epochs"]):
        model.train()
        train_losses = []
        test_losses = []
        for (sequences, dates) in train_loader:
            sequences = sequences.float()
            sequences = sequences.to(device)
            results = model(
                sequences[:, :config["observed_sequence_len"], :],
                prediction_time_steps = (config["total_seq_len"] - config["observed_sequence_len"])).cpu()
            sequences = sequences.cpu()
            mse_error = loss(results, sequences[:, config["observed_sequence_len"]:])

            optimizer.zero_grad()
            mse_error.backward()
            optimizer.step()

            train_losses.append(mse_error.detach().numpy())
            train_writer.add_scalar("MSE", mse_error.detach().numpy(), step)
            step += 1

        model.eval()
        for (sequences, dates) in test_loader:
            sequences = sequences.float()
            sequences = sequences.to(device)
            results = model(
                sequences[:, :config["observed_sequence_len"], :],
                prediction_time_steps=(config["total_seq_len"] - config["observed_sequence_len"])).cpu()
            sequences = sequences.cpu()
            mse_error = loss(results, sequences[:, config["observed_sequence_len"]:])

            test_losses.append(mse_error.detach().numpy())

        
        # Visualizing the results for train step
        logging.info(f"Epoch ({i+1:3}/{start_epoch + config['num_epochs']:3} | train_mse: {np.mean(train_losses):2.5f} | test_mse: {np.mean(test_losses):2.5f})")
        test_writer.add_scalar("MSE", np.mean(train_losses), step)
        if tuning_mode:
            tune.report(loss=np.mean(test_losses))
        selected = np.random.randint(0, dates.shape[0])
        fig = visulaize(dates[selected],
                  target=test_dataset.rescale(sequences[selected].detach().numpy()),
                  predicted=test_dataset.rescale(results[selected].detach().numpy()),
                  y_lim=[10, 25])

        test_writer.add_figure("test samples", fig, i)

        # saving the model
        if i % args.save_every_n_epochs == 0:
            model_saving_dict["epoch"] = i
            model_saving_dict["model_state_dict"] = model.state_dict()
            model_saving_dict["optimizer_state_dict"] = optimizer.state_dict()
            model_saving_dict["step"] = step
            model_saving_dict["train_loss"] = np.mean(train_losses)
            model_saving_dict["test_loss"] = np.mean(test_losses)

            if tuning_mode:
                with tune.checkpoint_dir(step) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save(model_saving_dict, path)



if __name__ == '__main__':

    # For saving and loading customization edit these parameters
    arg_params = argparse.ArgumentParser()
    arg_params.add_argument("--load", type=bool, default=False)
    arg_params.add_argument("--tag", type=str, default="static_down_sampling_64")
    arg_params.add_argument("--save_every_n_epochs", type=int, default=3)

    args = arg_params.parse_args()
    model_tag = args.tag

    save_files_list = os.listdir("save")
    save_dir = os.path.join("save", model_tag)

    if model_tag in save_files_list and not args.load:
        shutil.rmtree(save_dir)
    elif model_tag not in save_files_list:
        os.mkdir(save_dir)

    # For model params edit this part
    params = dict()

    # Model parameters
    params["lstm_hidden"] = tune.choice([64, 128, 256, 512])
    params["output_size"] = 3

    # Running Parameters
    params["batch_size"] = tune.choice([32, 64])
    params["num_epochs"] = 100

    # Data preparation parameters
    params["sampling_method"] = "static"
    params["sampling_rate"] = 1000
    params["total_seq_len"] = 50
    params["observed_sequence_len"] = 30
    params["seq_stride"] = 100
    params["shuffle"] = True
    params["train_split"] = 0.8

    train_evaluate(
        args,
        params,
        checkpoint_dir=save_dir
    )


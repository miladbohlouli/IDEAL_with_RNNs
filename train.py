import logging
import sys

import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torch.optim import Adam
import numpy as np
from data_loader import IDEAL_RNN
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
import logging


global line
line = 20*"*"


def train_evaluate(
        config,
        checkpoint_dir,
        data_dir,
        logging_dir,
        args,
        tuning_mode=False

    ):
    assert config["observed_sequence_len"] < config["total_seq_len"]

    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"

    device = torch.device(dev)

    train_writer = SummaryWriter(os.path.join(os.path.join(logging_dir, "tensorboard"), "train"))
    test_writer = SummaryWriter(os.path.join(os.path.join(logging_dir, "tensorboard"), "test"))
    logger = logging.getLogger("runtime_logs")
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    hdlr1 = logging.FileHandler(os.path.join(logging_dir, "runtime.log"))
    hdlr1.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(hdlr1)
    hdlr2 = logging.StreamHandler()
    hdlr2.setFormatter(formatter)
    logger.addHandler(hdlr2)

    logger.info(f"Using the {dev} as the running device")

    if config["load_data"] and "preprocessed_train.pk" in os.listdir(data_dir):
        logger.info("Found the preprocessed train data, loading....")
        with open(os.path.join(data_dir, "preprocessed_train.pk"), "rb") as file:
            train_dataset = pk.load(file)
    else:
        logger.info("Preparing the train loaders...")
        train_dataset = IDEAL_RNN(
            data_path=data_dir,
            chosen_sensor=['temperature'],
            logger=logger,
            params=config
        )
        with open(os.path.join(data_dir, "preprocessed_train.pk"), "wb") as file:
            pk.dump(train_dataset, file)

    train_loader = DataLoader(
        train_dataset,
        collate_fn=custom_collate,
        batch_size=config["batch_size"]
    )

    if params["load_data"] and "preprocessed_test.pk" in os.listdir(data_dir):
        logger.info("Found the preprocessed test data, loading....")
        with open(os.path.join(data_dir, "preprocessed_test.pk"), "rb") as file:
            test_dataset = pk.load(file)
    else:
        logger.info("Preparing the test loaders...")
        test_dataset = IDEAL_RNN(
            data_path=data_dir,
            logger=logger,
            chosen_sensor=['temperature'],
            train=False,
            params=config
        )
        with open(os.path.join(data_dir, "preprocessed_test.pk"), "wb") as file:
            pk.dump(test_dataset, file)

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

    logger.info(f"Experimenting with {len(train_dataset)} train | {len(test_dataset)} samples")
    logger.info(model)
    logger.info(line)

    loss = MSELoss()
    optimizer = Adam(model.parameters())
    step = 0
    start_epoch = 0
    model_saving_dict = {}

    if args.load:
        with open(os.path.join(logging_dir, "run_params.pickle"), "rb") as file:
            config = pk.load(file)

        checkpoint_dir = sorted(x for x in os.listdir(logging_dir) if "checkpoint" in x)
        print("found the checkpoint", os.path.join(logging_dir, checkpoint_dir[-1]))

        model_saving_dict = torch.load(os.path.join(logging_dir, checkpoint_dir[-1]))
        logger.info(f"Loading the model from checkpoint {os.path.join(checkpoint_dir, 'checkpoint')}")
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
        logger.info(f"Epoch ({i+1:3}/{start_epoch + config['num_epochs']:3} | train_mse: {np.mean(train_losses):2.5f} | test_mse: {np.mean(test_losses):2.5f})")
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

            else:
                path = os.path.join(logging_dir, f"-checkpoint{i}.pk")
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

    data_dir = os.path.abspath("dataset")

    # For model params edit this part
    params = dict()

    # Model parameters
    params["lstm_hidden"] = 128
    params["output_size"] = 1

    # Running Parameters
    params["batch_size"] = 64
    params["num_epochs"] = 100

    # Data preparation parameters
    params["sampling_method"] = "static"
    params["sampling_rate"] = 20
    params["total_seq_len"] = 70
    params["observed_sequence_len"] = 40
    params["seq_stride"] = 100
    params["shuffle"] = True
    params["train_split"] = 0.8
    params["single_house"] = False
    params["load_data"] = False

    train_evaluate(
        args=args,
        config=params,
        checkpoint_dir=None,
        logging_dir=save_dir,
        data_dir=data_dir,
    )


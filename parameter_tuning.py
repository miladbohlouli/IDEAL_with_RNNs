import logging
from functools import partial

import torch.cuda

from train import train_evaluate
import argparse
import shutil
import os
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import pickle as pk


def parameter_search(num_samples, max_epochs):
    # For saving and loading customization edit these parameters
    data_dir = os.path.abspath("dataset")
    arg_params = argparse.ArgumentParser()
    # Set this to always to False, when parameter tuning
    arg_params.add_argument("--load", type=bool, default=False)
    arg_params.add_argument("--tag", type=str, default="static_down_sampling")
    arg_params.add_argument("--save_every_n_epochs", type=int, default=10)

    args = arg_params.parse_args()
    model_tag = args.tag

    save_files_list = os.listdir("save")
    save_dir = os.path.join("save", model_tag)

    if model_tag in save_files_list:
        shutil.rmtree(save_dir)
    else:
        os.mkdir(save_dir)

    save_dir = os.path.abspath(save_dir)

    # For model params edit this part
    params = dict()

    # Model parameters
    params["lstm_hidden"] = tune.choice([64, 128, 256, 512])
    # params["lstm_hidden"] = 64
    params["output_size"] = 1

    # Running Parameters
    params["batch_size"] = tune.choice([32, 64])
    # params["batch_size"] = 32
    params["num_epochs"] = 10

    # Data preparation parameters
    params["sampling_method"] = "static"
    params["sampling_rate"] = tune.choice(list(range(30, 100, 10)))
    # params["sampling_rate"] = 1000
    params["total_seq_len"] = 60
    params["observed_sequence_len"] = tune.choice([30, 40, 50])
    # params["observed_sequence_len"] = 30
    params["seq_stride"] = 100
    params["shuffle"] = True
    params["train_split"] = 0.8

    reporter = CLIReporter(
        "loss"
    )
    scheduler = ASHAScheduler(
        grace_period=1,
        reduction_factor=2
    )
    gpu_per_trial = 1 if torch.cuda.is_available() else 0
    results = tune.run(
        partial(train_evaluate, data_dir=data_dir, args=args, tuning_mode=True, logging_dir="."),
        config=params,
        scheduler=scheduler,
        resources_per_trial={"cpu" : 8, "gpu" : gpu_per_trial},
        metric="loss",
        progress_reporter=reporter,
        local_dir=save_dir,
        log_to_file=True,
        mode="min",
        num_samples=num_samples
    )
    logger = logging.getLogger("final_results")
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    hdlr1 = logging.FileHandler(os.path.join(save_dir, "final_results.log"))
    hdlr1.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(hdlr1)

    best_trial = results.get_best_trial("loss", "min", "last")
    logger.info("Best trial config: {}".format(best_trial.config))
    logger.info("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))

    with open(os.path.join(save_dir, "best_results.pk"), "wb") as file:
        pk.dump(results.best_trial, file)

if __name__ == '__main__':
    parameter_search(1, 2)
import logging
from torch.utils.data import Dataset
from dataset.IdealDataInterface import *
from dataset.IdealMetadataInterface import *
import os
import argparse
import shutil
line = 20*"*"
from tqdm import tqdm
from functools import partial
from data_loader import IDEAL_RNN


class IDEAL_RNN_static(IDEAL_RNN):
    def sample(self,
               data,
               sampling_rate,
               ):
        data_len = len(data)
        indexes = np.array(list(range(data_len)))
        return data[indexes % sampling_rate == 0]


if __name__ == '__main__':
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
    # Model parameters
    params["lstm_hidden"] = 128
    params["output_size"] = 1

    # Running Parameters
    params["batch_size"] = 64
    params["num_epochs"] = 100

    # Data preparation parameters
    params["sampling_method"] = "static"
    params["sampling_rate"] = 60
    params["total_seq_len"] = 70
    params["observed_sequence_len"] = 40
    params["seq_stride"] = 100
    params["shuffle"] = True
    params["train_split"] = 0.8
    params["single_house"] = True

    logger = logging.getLogger("data_loader_test")
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    hdlr1 = logging.StreamHandler()
    hdlr1.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(hdlr1)

    train_dataset = IDEAL_RNN_static(
        data_path=data_dir,
        logger=logger,
        params=params,
        chosen_sensor=["temperature"]
    )

    print(train_dataset[0][0].__len__())


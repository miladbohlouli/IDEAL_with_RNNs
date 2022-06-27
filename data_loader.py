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

# Todo
#   1. Make a saving and loading for the overall house setting for a faster loading (done)
#   2. Implement the entropy-based sampling approach


class IDEAL_RNN(Dataset):
    def __init__(self,
                 data_path="dataset",
                 home_id=None,
                 train=True,
                 seed=20,
                 params=None,
                 logger=None,
                 chosen_sensor=None):
        assert params is not None
        self.sensor_data_path = os.path.join(data_path, "sensordata")
        self.room_appliance_data_path = os.path.join(data_path, "room_and_appliance_sensors")
        self.metadata_path = os.path.join(data_path, "metadata")
        self.seq_length = params["total_seq_len"]
        self.sampling_rate = params["sampling_rate"]
        self.stride = params["seq_stride"]
        self.train_split = params["train_split"]
        self.shuffle = params["shuffle"]
        self.train = train
        self.single_house = params["single_house"]

        # initialize the metadata interface
        mdi = IdealMetadataInterface(self.metadata_path)
        ideal = IdealDataInterface(self.sensor_data_path)
        appliance = IdealDataInterface(self.room_appliance_data_path)
        home_ids = np.unique(mdi.metadata["homes"]["homeid"])
        np.random.seed(seed)
        if home_id is None:
            home_id = np.random.choice(home_ids)

        if self.single_house:
            room_ids = mdi.metadata["rooms"][mdi.metadata["rooms"]["homeid"] == home_id]["roomid"].to_numpy()
            logger.info(f" Considering the home: {home_id} with {len(room_ids)} rooms...")
        elif not self.single_house:
            room_ids = []
            for home_id in home_ids:
                room_ids.append(mdi.metadata["rooms"][mdi.metadata["rooms"]["homeid"] == home_id]["roomid"].to_numpy())
            room_ids = np.concatenate(room_ids)
            logger.info(f" Considering all of the houses with overall {len(room_ids)} number of the rooms...")

        home_sensors = mdi.metadata["sensors"][mdi.metadata["sensors"]["roomid"].isin(room_ids)]
        self.chosen_sensors = chosen_sensor
        rnn_dataset_metdata = home_sensors.loc[
            home_sensors["type"].isin(self.chosen_sensors), ["roomid", "sensorid", "type"]].sort_values("roomid")
        logger.info(f"available sensor types: {home_sensors['type'].unique()}")
        logger.info(f"chosen sensor types: {self.chosen_sensors}")
        logger.info(line)

        room_pure_dataset = {}
        self.room_mean_std = {}

        for room_id in tqdm(rnn_dataset_metdata["roomid"].unique()):
            sensor_ids = rnn_dataset_metdata[rnn_dataset_metdata["roomid"] == room_id]["sensorid"].to_numpy()
            temp_data = []
            temp_data += ideal.get(sensorid=sensor_ids)
            temp_data += appliance.get(sensorid=sensor_ids)
            if temp_data.__len__() != 0:
                room_pure_dataset[room_id] = temp_data

        if room_pure_dataset.keys().__len__() == 0:
            raise Exception(f"There are no valid files for the home_id: {home_id}")

        train_dataset = []
        train_dataset_timesteps = []
        test_dataset = []
        test_dataset_timesteps = []
        sampling_function = partial(
            self.sample,
            sampling_rate=self.sampling_rate,
        )
        for key, value in room_pure_dataset.items():

            temp_ds = [sampling_function(val['readings'].to_numpy())[:, None] for val in value]
            dates = [sampling_function(val['readings'].index.to_numpy())[:, None] for val in value]

            temp_ds = np.concatenate(temp_ds, axis=0)
            dates = np.concatenate(dates, axis=0)

            data_len = len(temp_ds)
            np.random.seed(10)
            indexes = list(range(data_len))

            train_indexes = indexes[:int(data_len * self.train_split)]
            train_dataset.append(temp_ds[train_indexes])
            train_dataset_timesteps.append(dates[train_indexes])

            test_indexes = list(range(data_len))[int(data_len * self.train_split):]
            test_dataset.append(temp_ds[test_indexes])
            test_dataset_timesteps.append(dates[test_indexes])

        train_dataset = np.concatenate(train_dataset, axis=0)
        mean, std = train_dataset.mean(), train_dataset.std()
        self.room_mean_std = (mean, std)
        train_dataset = (train_dataset - mean) / std
        train_dates = np.concatenate(train_dataset_timesteps, axis=0)

        test_dataset = np.concatenate(test_dataset, axis=0)
        test_dataset = (test_dataset - mean) / std
        test_dates = np.concatenate(test_dataset_timesteps, axis=0)

        self.train_dataset = IDEAL_RNN.__make_sequence(
            train_dataset,
            seq_len=self.seq_length,
            stride=self.stride
        )

        self.train_dates = IDEAL_RNN.__make_sequence(
            train_dates,
            seq_len=self.seq_length,
            stride=self.stride
        )

        if self.shuffle:
            indexes = list(range(len(self.train_dataset)))
            np.random.shuffle(indexes)
            self.train_dataset = self.train_dataset[indexes]
            self.train_dates = self.train_dates[indexes]

        self.test_dataset = IDEAL_RNN.__make_sequence(
            test_dataset,
            seq_len=self.seq_length,
            stride=self.stride
        )

        self.test_dates = IDEAL_RNN.__make_sequence(
            test_dates,
            seq_len=self.seq_length,
            stride=self.stride
        )

    def rescale(self, data):
        return (data * self.room_mean_std[1]) + self.room_mean_std[0]

    def __getitem__(self, idx):
        if self.train:
            return self.train_dataset[idx], self.train_dates[idx]
        elif not self.train:
            return self.test_dataset[idx], self.test_dates[idx]

    def __len__(self):
        if self.train:
            return len(self.train_dataset)
        elif not self.train:
            return len(self.test_dataset)

    def sample(self,
               data,
               sampling_rate,
               ):
        raise "Not implemented error"

    @staticmethod
    def __make_sequence(data, seq_len, stride):
        return np.array([np.array(data[i:i + seq_len]) for i in range(0, len(data) - seq_len, stride)])


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

    train_dataset = IDEAL_RNN(
        data_path=data_dir,
        logger=logger,
        params=params,
        chosen_sensor=["temperature"]
    )

    print(train_dataset[0][0].__len__())

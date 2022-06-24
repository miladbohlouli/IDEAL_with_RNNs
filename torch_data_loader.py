import logging

import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from dataset.IdealDataInterface import *
from dataset.IdealMetadataInterface import *
import os

line = 20*"*"


class IDEAL_RNN(Dataset):
    def __init__(self,
                 data_path="dataset",
                 multi_room_training=True,
                 train=True,
                 seed=20,
                 params=None):
        assert params is not None
        sensor_data_path = os.path.join(data_path, "sensordata")
        room_appliance_data_path = os.path.join(data_path, "room_and_appliance_sensors")
        metadata_path = os.path.join(data_path, "metadata")

        self.sampling_method = params["sampling_method"]
        self.seq_length = params["total_seq_len"]
        self.sampling_rate = params["sampling_rate"]
        self.stride = params["seq_stride"]
        self.train_split = params["train_split"]
        shuffle = params["shuffle"]
        self.train = train

        # initialize the metadata interface
        mdi = IdealMetadataInterface(metadata_path)
        ideal = IdealDataInterface(sensor_data_path)
        appliance = IdealDataInterface(room_appliance_data_path)
        home_ids = np.unique(mdi.metadata["homes"]["homeid"])
        np.random.seed(seed)
        home_id = np.random.choice(home_ids)
        room_ids = mdi.metadata["rooms"][mdi.metadata["rooms"]["homeid"] == home_id]["roomid"].to_numpy()
        home_sensors = mdi.metadata["sensors"][mdi.metadata["sensors"]["roomid"].isin(room_ids)]
        chosen_sensors = ["temperature"]
        print("home_id: ", home_id)
        print("room_ids: ", room_ids)
        print("available sensor types: ", home_sensors["type"].unique())
        print("chosen sensor types: ", chosen_sensors)
        print(line)

        rnn_dataset_metdata = home_sensors.loc[home_sensors["type"].isin(chosen_sensors), ["roomid", "sensorid", "type"]].sort_values("roomid")

        room_pure_dataset = {}
        self.room_mean_std = {}
        if multi_room_training:
            for room_id in rnn_dataset_metdata["roomid"].unique():
                sensor_ids = rnn_dataset_metdata[rnn_dataset_metdata["roomid"] == room_id]["sensorid"]
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
        for key, value in room_pure_dataset.items():
            temp_ds = [val['readings'].to_numpy()[:, None] for val in value]
            dates = [val['readings'].index.to_numpy()[:, None] for val in value]

            temp_ds = np.concatenate(temp_ds, axis=1)
            temp_ds = self.__sample(temp_ds)
            dates = np.concatenate(dates, axis=1)

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

        if shuffle:
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

    def __sample(self, data):
        data_len = len(data)
        indexes = np.array(list(range(data_len)))
        return data[indexes % 10 == 0]



    @staticmethod
    def __make_sequence(data, seq_len, stride):
        return np.array([np.array(data[i:i + seq_len]) for i in range(0, len(data) - seq_len, stride)])

if __name__ == '__main__':
    id = IDEAL_RNN()

    ds_loader = DataLoader(id,
               batch_size=64)

    print(iter(ds_loader).__next__().shape)

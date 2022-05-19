import os

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
                 seed=20,
                 multi_room_training=True,
                 seq_length=100,
                 shuffle=True,
                 train=True,
                 train_split=0.8,
                 stride=100):
        sensor_data_path = os.path.join(data_path, "sensordata")
        room_appliance_data_path = os.path.join(data_path, "room_and_appliance_sensors")
        metadata_path = os.path.join(data_path, "metadata")

        self.seq_length = seq_length
        self.stride = stride
        self.train_split = train_split
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
        chosen_sensors = ["humidity", "temperature", "light"]
        print("home_id: ", home_id)
        print("room_ids: ", room_ids)
        print("available sensor types: ", home_sensors["type"].unique())
        print("chosen sensor types: ", chosen_sensors)
        print(line)

        rnn_dataset_metdata = home_sensors.loc[home_sensors["type"].isin(chosen_sensors), ["roomid", "sensorid", "type"]].sort_values("roomid")

        room_pure_dataset = {}
        room_mean_std = {}
        if multi_room_training:
            for room_id in rnn_dataset_metdata["roomid"].unique():
                sensor_ids = rnn_dataset_metdata[rnn_dataset_metdata["roomid"] == room_id]["sensorid"]
                temp_data = []
                temp_data += ideal.get(sensorid=sensor_ids)
                temp_data += appliance.get(sensorid=sensor_ids)
                if temp_data.__len__() != 0:
                    room_pure_dataset[room_id] = temp_data
                    break

        if room_pure_dataset.keys().__len__() == 0:
            raise Exception(f"There are no valid files for the home_id: {home_id}")

        train_dataset = []
        test_dataset = []
        room_dataset = {}
        for key, value in room_pure_dataset.items():
            temp_ds = [val['readings'].to_numpy()[:, None] for val in value]
            room_dataset[key] = np.concatenate(temp_ds, axis=1)

            data_len = len(room_dataset[key])
            indexes = list(range(data_len))
            np.random.seed(10)
            if shuffle: np.random.shuffle(indexes)
            train_indexes = indexes[:int(data_len * self.train_split)]
            mean = np.mean(room_dataset[key][train_indexes], axis=0)
            std = np.std(room_dataset[key][train_indexes], axis=0)
            room_dataset[key] = (room_dataset[key] - mean) / std
            room_mean_std[key] = (mean, std)

            train_dataset.append(
                IDEAL_RNN.__make_sequence(
                    room_dataset[key][train_indexes],
                    self.seq_length,
                    self.stride
                ))

            test_indexes = list(range(data_len))[int(data_len * self.train_split):]
            test_dataset.append(
                IDEAL_RNN.__make_sequence(
                    room_dataset[key][test_indexes],
                    self.seq_length,
                    self.stride
                ))

        self.train_dataset = np.concatenate(train_dataset, axis=0)
        self.test_dataset = np.concatenate(test_dataset, axis=0)


    @staticmethod
    def __make_sequence(data, seq_len, stride):
        return [np.array(data[i:i+seq_len]) for i in range(0, len(data)-seq_len, stride)]

    def __getitem__(self, idx):
        if self.train:
            return self.train_dataset[idx]
        elif not self.train:
            return self.test_dataset[idx]

    def __len__(self):
        if self.train:
            return len(self.train_dataset)
        elif not self.train:
            return len(self.test_dataset)


if __name__ == '__main__':
    id = IDEAL_RNN()

    ds_loader = DataLoader(id,
               batch_size=64)

    print(iter(ds_loader).__next__().shape)

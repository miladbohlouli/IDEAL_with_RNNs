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
                 stride=100):
        sensor_data_path = os.path.join(data_path, "sensordata")
        room_appliance_data_path = os.path.join(data_path, "room_and_appliance_sensors")
        metadata_path = os.path.join(data_path, "metadata")

        self.seq_length = seq_length

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

        room_dataset = {}
        room_mean_std = {}
        if multi_room_training:
            for room_id in rnn_dataset_metdata["roomid"].unique():
                sensor_ids = rnn_dataset_metdata[rnn_dataset_metdata["roomid"] == room_id]["sensorid"]
                temp_data = []
                temp_data += ideal.get(sensorid=sensor_ids)
                temp_data += appliance.get(sensorid=sensor_ids)
                if temp_data.__len__() != 0:
                    room_dataset[room_id] = temp_data
                    break

        if room_dataset.keys().__len__() == 0:
            raise Exception(f"There are no valid files for the home_id: {home_id}")

        dataset = []
        for key, value in room_dataset.items():
            temp_ds = [val['readings'].to_numpy()[:, None] for val in value]
            mean = np.mean(temp_ds, axis=0)
            std = np.std(temp_ds, axis=0)
            temp_ds = (temp_ds - mean) / std
            dataset.append(np.concatenate(temp_ds, axis=1))
            room_mean_std[key] = (mean, std)

        self.rnn_dataset = []
        for ds in dataset:
            self.rnn_dataset += [np.array(ds[i:i+seq_length]) for i in range(0, len(ds)-seq_length, stride)]

        self.rnn_dataset = np.asarray(self.rnn_dataset, dtype=np.float)

    def __getitem__(self, idx):
        return self.rnn_dataset[idx]

    def __len__(self):
        return len(self.rnn_dataset)


if __name__ == '__main__':
    id = IDEAL_RNN()

    ds_loader = DataLoader(id,
               batch_size=64)

    print(iter(ds_loader).__next__().shape)

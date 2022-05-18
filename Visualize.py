import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
import seaborn as sns
sns.set(color_codes=True)
import os
import gzip
import shutil
from matplotlib.dates import DateFormatter
from tqdm import tqdm


def preprocess(df):
    df.iloc[:, 0] = df.iloc[:, 0].map(lambda x: pd.to_datetime(x, format="%Y-%m-%d %H:%M:%S"))
    df.iloc[:, 1] = df.iloc[:, 1].map(lambda x: float(x/10))
    return df


def save_fig(name, save_path:str = "save"):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    plt.savefig(f"{os.path.join(save_path, name)}.png")


def visualize(y_limit, figure_size, starting_date, ending_date, path: str = ".", saving_format="%a_%d_%m"):
    ds_pathes = sorted([ps for ps in os.listdir(path) if (".csv" in ps) and ("humidity" in ps or "temperature" in ps)])
    datasets = []
    ds_indexes = []
    ds_names = []

    # ds_pathes = []
    # ds_pathes.append(dataset_pathes[0])
    #     Reading and preprocessing the csv files
    for ds_path in tqdm(ds_pathes):
        df = preprocess(pd.read_csv(os.path.join(path, ds_path), header=None, nrows=100000))

        ds_indexes.append((datetime.strptime(starting_date, "%d.%m.%y %H:%M:%S") < df.iloc[:, 0]) & (
                df.iloc[:, 0] < datetime.strptime(ending_date, "%d.%m.%y %H:%M:%S")))

        if ds_indexes.__len__() == 0:
            raise Exception("There are no samples in the defined temporal period")

        ds_names.append("_".join("".join(ds_path.split(".")[:-1]).split("_")[1:]))

        datasets.append(df)

    for i, ds_name in tqdm(enumerate(ds_names)):
        fig, ax = plt.subplots(figsize=figure_size)
        max_val = np.ceil(max(datasets[i][ds_indexes[i]].iloc[:, 1]))
        min_val = np.floor(min(datasets[i][ds_indexes[i]].iloc[:, 1]))
        y_limit = [min_val, max_val]

        ax.set(
            xlabel="date(Weekday dd.mm)",
            ylabel="C" if "temperature" in ds_name else "%",
            ylim=y_limit
        )

        ax.set_title(ds_name, fontsize=20)

        ax.plot(
            datasets[i][ds_indexes[i]].iloc[:, 0],
            datasets[i][ds_indexes[i]].iloc[:, 1]
        )

        date_form = DateFormatter(axis_dateformat)
        ax.xaxis.set_major_formatter(date_form)
        ax.legend(ds_names)
        name = ds_name + "_" + \
               datetime.strptime(starting_date, "%d.%m.%y %H:%M:%S").strftime(saving_format) + "-" + \
               datetime.strptime(ending_date, "%d.%m.%y %H:%M:%S").strftime(saving_format)

        save_fig(name)
    return ds_names, datasets


if __name__ == '__main__':
    home_id = 144
    data_dir = "room_and_appliance_sensors"
    temp_dir = "temp"

    files_list = os.listdir("room_and_appliance_sensors")
    home_id_files = []
    for file_name in files_list:
        if file_name.split(".")[-1] == "gz" and home_id == int(file_name.split("_")[0].replace("home", "")):
            home_id_files.append(file_name)

    for file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, file))
    # extracting the related csv files
    for file in home_id_files:
        with gzip.open(os.path.join(data_dir, file), 'rb') as f_in:
            with open(os.path.join(temp_dir, "".join(file.split(".")[:-2]) + ".csv"), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    starting_date = "10.07.17 00:00:00"
    ending_date = "11.07.17 00:00:00"
    figure_size = [20, 10]
    axis_dateformat = "%a %d.%m %H:%M"
    saving_dateformat = "(%d_%m_%Y)"

    _, _ = visualize(
        path=temp_dir,
        y_limit=None,
        figure_size=figure_size,
        starting_date=starting_date,
        ending_date=ending_date,
        saving_format=saving_dateformat)


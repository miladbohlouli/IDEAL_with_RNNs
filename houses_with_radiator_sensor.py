import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
import seaborn as sns
sns.set(color_codes=True)
import os
from IPython.display import display


def collect_save_data(home_id=None, save_dir=None):
    meta_data_dir = "DS_10283_3647/metadata_and_surveys/metadata"
    home_meta_data = pd.read_csv(os.path.join(meta_data_dir, "home.csv"))
    if home_id is None:
        home_ids = home_meta_data["homeid"].unique()
        home_id = np.random.choice(home_ids)
    location_meta_data = pd.read_csv(os.path.join(meta_data_dir, "location.csv"))
    person_meta_data = pd.read_csv(os.path.join(meta_data_dir, "person.csv"))
    room_meta_data = pd.read_csv(os.path.join(meta_data_dir, "room.csv"))
    sensor_meta_data = pd.read_csv(os.path.join(meta_data_dir, "sensor.csv"))
    appliance_meta_data = pd.read_csv(os.path.join(meta_data_dir, "appliance.csv"))

    assert home_id in home_meta_data["homeid"].unique()
    home_df = home_meta_data[home_meta_data["homeid"] == home_id]
    persons_df = person_meta_data[person_meta_data["homeid"] == home_id]
    rooms_df = room_meta_data[room_meta_data["homeid"] == home_id]
    room_ids = rooms_df["roomid"]
    sensor_df = sensor_meta_data[sensor_meta_data["roomid"].isin(room_ids.values)]
    appliance_df = appliance_meta_data[appliance_meta_data["roomid"].isin(room_ids.values)]

    save_path = os.path.join(save_dir, f"{home_id}.xlsx")
    writer = pd.ExcelWriter(save_path, engine="xlsxwriter")
    home_df.to_excel(writer, sheet_name="home info")
    persons_df.to_excel(writer, sheet_name="occupants info")
    rooms_df.to_excel(writer, sheet_name="rooms info")
    sensor_df.to_excel(writer, sheet_name="sensors info")
    appliance_df.to_excel(writer, sheet_name="appliance info")

    writer.save()


if __name__ == '__main__':
    save_dir = "save"
    "home97_hall1010_sensor4238c4242_electric-mains_electric-combined.csv"
    collect_save_data(save_dir=save_dir)


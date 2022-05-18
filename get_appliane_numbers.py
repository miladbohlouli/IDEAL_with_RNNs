import os
import numpy as np
data_dir = "room_and_appliance_sensors"


def check_name(x):
    if x.split(".")[-1] == "gz" and x.split("_")[3] == "electric-appliance":
        return x.split("_")[0].replace("home", ""), x.split("_")[4].split(".")[0]


files_list = os.listdir(data_dir)
appliance_home_list = []


print(homes_with_electric_appliances)

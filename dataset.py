from torch.utils.data import Dataset, DataLoader
# from const import *
# from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import pandas as pd

import pickle
import random
import time
import torch

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


from utils import my_min_max, decode

class MyDataset(Dataset):
    def __init__(self, path, filter):
        print(f"loading data from {path} ...")
        df = pd.read_csv(path, skiprows=1)
        if df.values.shape[-1] != 11:
            df = pd.read_csv(path, header=None)

        n_col = df.values.shape[-1]
        assert n_col in [6, 7, 11]

        record = dict()

        if n_col == 7:  # for 3->1
            self.x_data = torch.tensor(df.values, dtype=torch.float64)[:, :3]
            self.y_data = torch.tensor(df.values, dtype=torch.float64)[:, 3:4]
            self.x_data[:, 0:1], x1_min, x1_max = my_min_max(self.x_data[:, 0:1])
            self.x_data[:, 1:2], x2_min, x2_max = my_min_max(self.x_data[:, 1:2])
            self.x_data[:, 2:3], x3_min, x3_max = my_min_max(self.x_data[:, 2:3])
            record["x1_min"] = x1_min
            record["x1_max"] = x1_max
            record["x2_min"] = x2_min
            record["x2_max"] = x2_max
            record["x3_min"] = x3_min
            record["x3_max"] = x3_max
        elif n_col == 6:  # for 2->1
            self.x_data = torch.tensor(df.values, dtype=torch.float64)[:, :2]
            self.y_data = torch.tensor(df.values, dtype=torch.float64)[:, 2:3]
            self.x_data[:, 0:1], x1_min, x1_max = my_min_max(self.x_data[:, 0:1])
            self.x_data[:, 1:2], x2_min, x2_max = my_min_max(self.x_data[:, 1:2])
            record["x1_min"] = x1_min
            record["x1_max"] = x1_max
            record["x2_min"] = x2_min
            record["x2_max"] = x2_max
        else:  # for 7->1
            self.x_data = torch.tensor(df.values, dtype=torch.float64)[:, :7]
            self.y_data = torch.tensor(df.values, dtype=torch.float64)[:, 7:8]
            self.x_data[:, 0:1], x1_min, x1_max = my_min_max(self.x_data[:, 0:1])
            self.x_data[:, 1:2], x2_min, x2_max = my_min_max(self.x_data[:, 1:2])
            self.x_data[:, 2:3], x3_min, x3_max = my_min_max(self.x_data[:, 2:3])
            self.x_data[:, 3:4], x4_min, x4_max = my_min_max(self.x_data[:, 3:4])
            self.x_data[:, 4:5], x5_min, x5_max = my_min_max(self.x_data[:, 4:5])
            self.x_data[:, 5:6], x6_min, x6_max = my_min_max(self.x_data[:, 5:6])
            self.x_data[:, 6:7], x7_min, x7_max = my_min_max(self.x_data[:, 6:7])
            record["x1_min"] = x1_min
            record["x1_max"] = x1_max
            record["x2_min"] = x2_min
            record["x2_max"] = x2_max
            record["x3_min"] = x3_min
            record["x3_max"] = x3_max
            record["x4_min"] = x4_min
            record["x4_max"] = x4_max
            record["x5_min"] = x5_min
            record["x5_max"] = x5_max
            record["x6_min"] = x6_min
            record["x6_max"] = x6_max
            record["x7_min"] = x7_min
            record["x7_max"] = x7_max


        print(f"In generating dataset, n_col = {n_col}, and keys of record: {list(record.keys())}")

        self.y_data, y_min, y_max = my_min_max(self.y_data)

        record["y_min"] = y_min
        record["y_max"] = y_max
        with open(f"processed/filter={filter}/record_min_max.pkl", "wb") as f:
            pickle.dump(record, f)
        self.x_dim = self.x_data.shape[-1]
        self.y_dim = self.y_data.shape[-1]
        print(f"Full x shape: {self.x_data.shape}")
        print(f"Full y shape: {self.y_data.shape}")

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]
        return x, y


# def one_time_generate_dataset():
#     t0 = time.time()
#     dataset = MyDataset("data/dataset_osci_0_1_2_v0604.csv")
#
#     print(dataset.x_data[0], dataset.y_data[0])
#
#     # train_idxs, val_idxs, test_idxs = train_test_split(np.arange(len(dataset)), test_size=0.2, random_state=0)
#     train_idxs, val_idxs = train_test_split(np.arange(len(dataset)), test_size=0.2, random_state=42)
#     print(len(train_idxs), len(val_idxs))
#     train_dataset = torch.utils.data.Subset(dataset, train_idxs)
#     val_dataset = torch.utils.data.Subset(dataset, val_idxs)
#     with open("processed/train_idx.pkl", "wb") as f:
#         pickle.dump(train_idxs, f)
#     with open("processed/val_idx.pkl", "wb") as f:
#         pickle.dump(val_idxs, f)
#
#     with open("processed/x_raw.pkl", "wb") as f:
#         pickle.dump(dataset.x_data, f)
#     with open("processed/y_raw.pkl", "wb") as f:
#         pickle.dump(dataset.y_data, f)
#
#     with open("processed/all.pkl", "wb") as f:
#         pickle.dump(dataset, f)
#     with open("processed/train.pkl", "wb") as f:
#         pickle.dump(train_dataset, f)
#     with open("processed/valid.pkl", "wb") as f:
#         pickle.dump(val_dataset, f)
#     print("cost {0:.6f} min".format((time.time() - t0) / 60.0))

def one_time_generate_dataset(source_path, filter):
    t0 = time.time()
    source_path = source_path.replace(".csv", f"_{filter}.csv")
    save_folder_path = f"processed/filter={filter}/"
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
    dataset = MyDataset(source_path, filter)


    print(dataset.x_data[0], dataset.y_data[0])

    # train_idxs, val_idxs, test_idxs = train_test_split(np.arange(len(dataset)), test_size=0.2, random_state=0)
    train_idxs, val_idxs = train_test_split(np.arange(len(dataset)), test_size=0.2, random_state=42)
    print(len(train_idxs), len(val_idxs))
    train_dataset = torch.utils.data.Subset(dataset, train_idxs)
    val_dataset = torch.utils.data.Subset(dataset, val_idxs)
    with open(save_folder_path + "train_idx.pkl", "wb") as f:
        pickle.dump(train_idxs, f)
    with open(save_folder_path + "val_idx.pkl", "wb") as f:
        pickle.dump(val_idxs, f)

    with open(save_folder_path + "x_raw.pkl", "wb") as f:
        pickle.dump(dataset.x_data, f)
    with open(save_folder_path + "y_raw.pkl", "wb") as f:
        pickle.dump(dataset.y_data, f)

    with open(save_folder_path + "all.pkl", "wb") as f:
        pickle.dump(dataset, f)
    with open(save_folder_path + "train.pkl", "wb") as f:
        pickle.dump(train_dataset, f)
    with open(save_folder_path + "valid.pkl", "wb") as f:
        pickle.dump(val_dataset, f)
    print(f"saved to {save_folder_path}")
    print("cost {0:.6f} min".format((time.time() - t0) / 60.0))


if __name__ == "__main__":
    # one_time_generate_dataset("data/dataset_osci_0_1_2_v0604.csv", "all")
    # one_time_generate_dataset("data/dataset_osci_0_1_2_v0604.csv", "200")
    # one_time_generate_dataset("data/dataset_osci_0_1_2_v0604.csv", "100")

    # one_time_generate_dataset("data/dataset_osci_3_4_5_v0604.csv", "all")
    # one_time_generate_dataset("data/dataset_osci_3_4_5_v0604.csv", "200")
    # one_time_generate_dataset("data/dataset_osci_3_4_5_v0604.csv", "100")

    # one_time_generate_dataset("data/dataset_osci_0_1_v0618.csv", "all")
    # one_time_generate_dataset("data/dataset_osci_0_1_v0618.csv", "200")
    # one_time_generate_dataset("data/dataset_osci_0_1_v0618.csv", "100")

    # one_time_generate_dataset("data/dataset_osci_v0628_large.csv", "all")
    # one_time_generate_dataset("data/dataset_osci_v0628_large.csv", "200")
    # one_time_generate_dataset("data/dataset_osci_v0628_large.csv", "100")

    one_time_generate_dataset("data/dataset_osci_v0628_small.csv", "all")
    one_time_generate_dataset("data/dataset_osci_v0628_small.csv", "200")
    one_time_generate_dataset("data/dataset_osci_v0628_small.csv", "100")

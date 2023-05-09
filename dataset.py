from torch.utils.data import Dataset, DataLoader
from const import *
from utils import fill_nan
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

import pickle
import random
import time
import torch

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class MyDataset(Dataset):
    def __init__(self, x_path, y_path):
        x_df = pd.read_csv(x_path)
        y_df = pd.read_csv(y_path)

        self.x_dim = 6
        self.y_dim = 6
        self.x_data = torch.tensor(x_df.values, dtype=torch.float32)[:, :self.x_dim]
        self.y_data = torch.tensor(y_df.values, dtype=torch.float32)[:, :self.y_dim]
        print(f"Full x shape: {self.x_data.shape}")
        print(f"Full y shape: {self.y_data.shape}")

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]
        return x, y


def one_time_generate_dataset():
    t0 = time.time()
    dataset = MyDataset("data/x.csv", "data/y.csv")

    print(dataset.x_data[0], dataset.y_data[0])

    # train_idxs, val_idxs, test_idxs = train_test_split(np.arange(len(dataset)), test_size=0.2, random_state=0)
    train_idxs, val_idxs = train_test_split(np.arange(len(dataset)), test_size=0.2, random_state=42)
    print(len(train_idxs), len(val_idxs))
    train_dataset = torch.utils.data.Subset(dataset, train_idxs)
    val_dataset = torch.utils.data.Subset(dataset, val_idxs)
    with open("processed/train_idx.pkl", "wb") as f:
        pickle.dump(train_idxs, f)
    with open("processed/val_idx.pkl", "wb") as f:
        pickle.dump(val_idxs, f)

    with open("processed/all.pkl", "wb") as f:
        pickle.dump(dataset, f)
    with open("processed/train.pkl", "wb") as f:
        pickle.dump(train_dataset, f)
    with open("processed/valid.pkl", "wb") as f:
        pickle.dump(val_dataset, f)
    print("cost {} min".format((time.time() - t0) / 60.0))


if __name__ == "__main__":
    one_time_generate_dataset()

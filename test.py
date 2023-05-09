import pickle
import torch
import os
from torch.utils.data import Dataset, DataLoader

from model import MyModel
from utils import get_now_string, generate_output
from dataset import MyDataset


if __name__ == "__main__":
    generate_output("saves/model_20230509_052152_722761.pt")

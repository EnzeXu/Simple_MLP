import math
import numpy as np
from datetime import datetime

import pickle
import torch
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader


def get_now_string():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def fill_nan(clinic_list):
    mean = np.nanmean(np.asarray(clinic_list))
    return [item if not math.isnan(item) else mean for item in clinic_list]


def my_min_max(data):
    assert isinstance(data, torch.Tensor) or isinstance(data, np.ndarray)
    if isinstance(data, torch.Tensor):
        assert torch.min(data) != torch.max(data)
        return (data - torch.min(data)) / (torch.max(data) - torch.min(data)), float(torch.min(data)), float(torch.max(data))
    else:
        assert np.min(data) != np.max(data)
        return (data - np.min(data)) / (np.max(data) - np.min(data)), np.min(data), np.max(data)


def decode(data_encoded, data_min, data_max):
    return data_encoded * (data_max - data_min) + data_min


if __name__ == "__main__":
    # a = np.asarray([1.0, 2.0, 3.0])
    a = torch.tensor([1.0, 2.0, 3.0])
    print(my_min_max(a))

import math
import numpy as np
from datetime import datetime

import pickle
import torch
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader

from model import MyModel



def get_now_string():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def fill_nan(clinic_list):
    mean = np.nanmean(np.asarray(clinic_list))
    return [item if not math.isnan(item) else mean for item in clinic_list]


def generate_output(pt_path, timestring=None):
    main_path = "./"
    with open(main_path + "processed/all.pkl", "rb") as f:
        dataset = pickle.load(f)
    with open(main_path + "processed/valid.pkl", "rb") as f:
        val_dataset = pickle.load(f)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    gpu_id = 2
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device('cuda', gpu_id)
    else:
        device = torch.device('cpu')
    print("using {}".format(device))

    model = MyModel(x_dim=dataset.x_dim, y_dim=dataset.y_dim).to(device)
    model.load_state_dict(torch.load(pt_path))

    save_output_folder = "./record/output/"
    if not os.path.exists(save_output_folder):
        os.makedirs(save_output_folder)
    if not timestring:
        timestring = get_now_string()
    save_output_path = f"{save_output_folder}/output_{timestring}.txt"

    x_df = pd.read_csv("data/x.csv")
    y_df = pd.read_csv("data/y.csv")
    x_data_raw = x_df.values
    y_data_raw = y_df.values

    with open("processed/val_idx.pkl", "rb") as f:
        val_idx = pickle.load(f)

    print("saved output to {}".format(save_output_path))
    with open(save_output_path, "a") as f:
        f.write("val_id,[x],[y],[y_pred]\n")
        row_id = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(dtype=torch.float32).to(device), labels.to(dtype=torch.float32).to(device)
                outputs = model(inputs)
                inputs = inputs.cpu().detach().numpy()
                labels = labels.cpu().detach().numpy()
                outputs = outputs.cpu().detach().numpy()

                for i in range(len(inputs)):
                    print("[model] input: {} / labels: {} / output: {}".format(str(list(inputs[i])), str(list(labels[i])), str(list(outputs[i]))))
                    print("[original] x: {} / y: {} ".format(str(list(x_data_raw[val_idx[row_id]])), str(list(y_data_raw[val_idx[row_id]]))))
                    f.write("{0:d},{1},{2},{3}\n".format(
                        row_id,
                        ",".join([str("{0:.12e}".format(item)) for item in inputs[i]]),
                        ",".join([str("{0:.12e}".format(item)) for item in labels[i]]),
                        ",".join([str("{0:.12e}".format(item)) for item in outputs[i]]),
                    ))
                    row_id += 1

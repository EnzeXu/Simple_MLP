import os.path

import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle
import wandb
import time
import math
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from utils import get_now_string, decode
from tqdm import tqdm
from dataset import MyDataset
import os


class MyModel(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(MyModel, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.fc = nn.Sequential(
            nn.Linear(self.x_dim, 128),
            nn.Dropout(0.2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Dropout(0.2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.Linear(256, 256),  #
            # nn.BatchNorm1d(256),  #
            # nn.ReLU(),  #
            # nn.Linear(256, 128),
            # nn.BatchNorm1d(128),
            # nn.ReLU(),
            nn.Linear(128, self.y_dim),
        )
        # print("{} layers".format(len(self.fc)))

    def forward(self, x):
        return self.fc(x)


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:  # tqdm(dataloader, total=len(dataloader)):
        inputs, labels = inputs.to(dtype=torch.float32).to(device), labels.to(dtype=torch.float32).to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(dtype=torch.float32).to(device), labels.to(dtype=torch.float32).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    return running_loss / len(dataloader)


# def test(model, dataloader, device):
#     model.eval()
#     running_loss = 0.0
#     with torch.no_grad():
#         for inputs, labels in dataloader:
#             inputs, labels = inputs.to(dtype=torch.float32).to(device), labels.to(dtype=torch.float32).to(device)
#             output = model(inputs)
#             loss = criterion(output, labels)
#             running_loss += loss.item()
#     return running_loss / len(dataloader), math.sqrt(running_loss / len(dataloader))


def run(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device, main_path):
    train_loss_record = []
    valid_loss_record = []
    min_val_loss = float('Inf')
    best_epoch = 0
    record_timestring_start = get_now_string()
    record_t0 = time.time()
    record_time_epoch_step = record_t0
    init_lr = optimizer.param_groups[0]["lr"]
    print("start training: epochs = {}".format(epochs))

    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        valid_loss = validate(model, val_loader, criterion, device)
        if valid_loss < min_val_loss:
            min_val_loss = valid_loss
            best_epoch = epoch
            torch.save(model.state_dict(), main_path + "saves/model_{}.pt".format(record_timestring_start))
        # test_loss, test_loss_rmse = test(model, test_loader, device)
        train_loss_record.append(train_loss)
        valid_loss_record.append(valid_loss)
        try:
            wandb.log({'epoch': epoch, 'val_loss': valid_loss, 'train_loss': train_loss, 'lr': optimizer.param_groups[0]["lr"]})
        except:
            pass
        scheduler.step()
        if epoch % 10 == 0:
            # print("[{}] Epoch: {}/{}  train Loss: {:.9f}  val Loss: {:.9f}  min val Loss: {:.9f}  lr: {:.9f}".format(timestring, epoch, epochs, train_loss, valid_loss, min_val_loss, optimizer.param_groups[0]["lr"]))
            record_time_epoch_step_tmp = time.time()
            info_epoch = f'Epoch:{epoch}/{epochs}  train loss:{train_loss:.4e}  val loss:{valid_loss:.4e}  '
            info_best = f'best epoch:{best_epoch}  min loss:{min_val_loss:.4e}  '
            info_extended = f'lr:{optimizer.param_groups[0]["lr"]:.9e}  time:{(record_time_epoch_step_tmp - record_time_epoch_step):.2f}s  time total:{((record_time_epoch_step_tmp - record_t0) / 60.0):.2f}min  time remain:{((record_time_epoch_step_tmp - record_t0) / 60.0 / epoch * (epochs - epoch)):.2f}min'
            record_time_epoch_step = record_time_epoch_step_tmp
            print(info_epoch + info_best + info_extended)

            # print("model saved to {}".format(main_path + "saves/model_{}.pt".format(timestring)))
        if epoch % 50 == 0:
            generate_output(main_path + "saves/model_{}.pt".format(record_timestring_start), record_timestring_start, device)

        scheduler.step()
        # if (epoch + 1) % 10 == 0:
        #     plt.figure(figsize=(16, 9))
        #     plt.plot(range(1, len(train_loss_record) + 1), train_loss_record, label="train loss")
        #     plt.plot(range(1, len(valid_loss_record) + 1), valid_loss_record, label="valid loss")
        #     plt.legend()
        #     plt.show()
        #     plt.close()

    record_timestring_end = get_now_string()
    record_time_cost_min = (time.time() - record_t0) / 60.0
    record_folder_path = "./record/"
    if not os.path.exists(record_folder_path):
        os.makedirs(record_folder_path)
    with open(record_folder_path + "record.csv", "a") as f:
        f.write("{0},{1},{2},{3:.2f},{4},{5},{6:.9f},{7:.9f},{8},{9:.9f},{10},{11}\n".format(
            # func_name,timestring_start,timestring_end,time_cost_min,epochs,layer,activation,layer_size,lr,lr_end,scheduler,dropout,best_epoch,min_loss
            "Simple_MLP",  # 0
            record_timestring_start,  # 1
            record_timestring_end,  # 2
            record_time_cost_min,  # 3
            epochs,  # 4
            len(model.fc),  # 5
            init_lr,  # 6
            optimizer.param_groups[0]["lr"],  # 7
            best_epoch,  # 8
            min_val_loss,  # 9
            model.x_dim,  # 10
            model.y_dim,  # 11
        ))

    save_comparison_folder = "./record/comparison/"
    if not os.path.exists(save_comparison_folder):
        os.makedirs(save_comparison_folder)
    save_comparison_path = f"{save_comparison_folder}/val_{record_timestring_start}.txt"

    model.load_state_dict(torch.load(main_path + "saves/model_{}.pt".format(record_timestring_start)))
    with open(save_comparison_path, "a") as f:
        row_id = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(dtype=torch.float32).to(device), labels.to(dtype=torch.float32).to(device)
                outputs = model(inputs)
                labels = labels.cpu().detach().numpy()
                outputs = outputs.cpu().detach().numpy()
                for i in range(len(inputs)):
                    row_id += 1
                    f.write("[Truth   {0:05d}] {1}\n".format(
                        row_id,
                        " ".join([str("{0:.12f}".format(item)) for item in labels[i]]),
                    ))
                    f.write("[Predict {0:05d}] {1}\n".format(
                        row_id,
                        " ".join([str("{0:.12f}".format(item)) for item in outputs[i]]),
                    ))
    print("saved comparison to {}".format(save_comparison_path))
    generate_output(main_path + "saves/model_{}.pt".format(record_timestring_start), record_timestring_start, device)

def relative_loss(prediction, target):
    criterion = nn.MSELoss(reduction="none")
    mse_non_reduce = criterion(prediction, target)
    errors = mse_non_reduce / (torch.pow(target, 2) + 1e-6)
    errors = torch.mean(errors)
    return errors


def generate_output(pt_path, timestring=None, device=None):
    main_path = "./"
    with open(main_path + "processed/all.pkl", "rb") as f:
        dataset = pickle.load(f)
    with open(main_path + "processed/valid.pkl", "rb") as f:
        val_dataset = pickle.load(f)
    with open(main_path + "processed/train.pkl", "rb") as f:
        train_dataset = pickle.load(f)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    if not device:
        gpu_id = 0
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            device = torch.device('cuda', gpu_id)
        else:
            device = torch.device('cpu')
        print("using {}".format(device))

    model = MyModel(x_dim=dataset.x_dim, y_dim=dataset.y_dim).to(device)
    # print("load pt from {}".format(pt_path))
    model.load_state_dict(torch.load(pt_path))

    save_output_folder = "./record/output/"
    if not os.path.exists(save_output_folder):
        os.makedirs(save_output_folder)
    if not timestring:
        timestring = get_now_string()
    save_output_path_val = f"{save_output_folder}/output_{timestring}_val.txt"
    save_output_path_train = f"{save_output_folder}/output_{timestring}_train.txt"

    # x_df = pd.read_csv("data/x.csv")
    # y_df = pd.read_csv("data/y.csv")
    # x_data_raw = x_df.values
    # y_data_raw = y_df.values

    with open("processed/x_raw.pkl", "rb") as f:
        x_data_raw = pickle.load(f)
    with open("processed/y_raw.pkl", "rb") as f:
        y_data_raw = pickle.load(f)


    with open("processed/val_idx.pkl", "rb") as f:
        val_idx = pickle.load(f)
    with open("processed/train_idx.pkl", "rb") as f:
        train_idx = pickle.load(f)


    with open("processed/record_min_max.pkl", "rb") as f:
        record = pickle.load(f)
    x1_min = record["x1_min"]
    x1_max = record["x1_max"]
    x2_min = record["x2_min"]
    x2_max = record["x2_max"]
    x3_min = record["x3_min"]
    x3_max = record["x3_max"]
    y_min = record["y_min"]
    y_max = record["y_max"]
    x_data_raw[:, 0:1] = decode(x_data_raw[:, 0:1], x1_min, x1_max)
    x_data_raw[:, 1:2] = decode(x_data_raw[:, 1:2], x2_min, x2_max)
    x_data_raw[:, 2:3] = decode(x_data_raw[:, 2:3], x3_min, x3_max)
    y_data_raw[:, :] = decode(y_data_raw[:, :], y_min, y_max)

    with open(save_output_path_val, "a") as f:
        f.write("id,[x],[y],[y_pred]\n")
        row_id = 0
        sorted_output_val = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(dtype=torch.float32).to(device), labels.to(dtype=torch.float32).to(device)
                outputs = model(inputs)
                inputs = inputs.cpu().detach().numpy()
                labels = labels.cpu().detach().numpy()
                outputs = outputs.cpu().detach().numpy()

                outputs = decode(outputs, y_min, y_max)

                for i in range(len(inputs)):
                    sorted_output_val.append(
                        [val_idx[row_id],
                        ",".join([str("{0:.12f}".format(item)) for item in x_data_raw[val_idx[row_id]]]),
                        ",".join([str("{0:.12f}".format(item)) for item in y_data_raw[val_idx[row_id]]]),
                        ",".join([str("{0:.12f}".format(item)) for item in outputs[i]])]
                    )
                    row_id += 1

        sorted_output_val = sorted(sorted_output_val, key=lambda x: x[0])
        for one_output in sorted_output_val:
            # print("[model] input: {} / labels: {} / output: {}".format(str(list(inputs[i])), str(list(labels[i])), str(list(outputs[i]))))
            # print("[original] x: {} / y: {} ".format(str(list(x_data_raw[val_idx[row_id]])), str(list(y_data_raw[val_idx[row_id]]))))
            f.write("{0:d},{1},{2},{3}\n".format(
                one_output[0],
                one_output[1],
                one_output[2],
                one_output[3],
            ))
    print("saved val output to {}".format(save_output_path_val))

    with open(save_output_path_train, "a") as f:
        f.write("id,[x],[y],[y_pred]\n")
        row_id = 0
        sorted_output_train = []
        with torch.no_grad():
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(dtype=torch.float32).to(device), labels.to(dtype=torch.float32).to(device)
                outputs = model(inputs)
                inputs = inputs.cpu().detach().numpy()
                labels = labels.cpu().detach().numpy()
                outputs = outputs.cpu().detach().numpy()

                outputs = decode(outputs, y_min, y_max)

                for i in range(len(inputs)):
                    sorted_output_train.append(
                        [train_idx[row_id],
                        ",".join([str("{0:.12f}".format(item)) for item in x_data_raw[train_idx[row_id]]]),
                        ",".join([str("{0:.12f}".format(item)) for item in y_data_raw[train_idx[row_id]]]),
                        ",".join([str("{0:.12f}".format(item)) for item in outputs[i]])]
                    )
                    row_id += 1

        sorted_output_train = sorted(sorted_output_train, key=lambda x: x[0])
        for one_output in sorted_output_train:
            # print("[model] input: {} / labels: {} / output: {}".format(str(list(inputs[i])), str(list(labels[i])), str(list(outputs[i]))))
            # print("[original] x: {} / y: {} ".format(str(list(x_data_raw[val_idx[row_id]])), str(list(y_data_raw[val_idx[row_id]]))))
            f.write("{0:d},{1},{2},{3}\n".format(
                one_output[0],
                one_output[1],
                one_output[2],
                one_output[3],
            ))
    print("saved train output to {}".format(save_output_path_train))


if __name__ == "__main__":
    main_path = "./"
    with open(main_path + "processed/all.pkl", "rb") as f:
        dataset = pickle.load(f)
    with open(main_path + "processed/train.pkl", "rb") as f:
        train_dataset = pickle.load(f)
    with open(main_path + "processed/valid.pkl", "rb") as f:
        val_dataset = pickle.load(f)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_id = 0
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device('cuda', gpu_id)
    else:
        device = torch.device('cpu')
    print("using {}".format(device))

    model = MyModel(x_dim=dataset.x_dim, y_dim=dataset.y_dim).to(device)
    # model.load_state_dict(torch.load(main_path + "saves/model_20230228_211049_069082.pt"))
    epochs = 1000
    criterion = nn.MSELoss()  #  relative_loss  # nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: 1 / (e / (0.01 * epochs) + 1))
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0.000001 * 0.1)

    wandb_flag = True
    if wandb_flag:
        with wandb.init(project='Simple_MLP', name='test'):
            run(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device, main_path)
    else:
         run(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device, main_path)

    # input = torch.tensor([1.01, 202.0])
    # target = torch.tensor([1.0, 200.0])
    # output = relative_loss(input, target)
    # print(output)







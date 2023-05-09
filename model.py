import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle
import math
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import get_now_string
from tqdm import tqdm
from dataset import MyDataset


class MyModel(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(MyModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(x_dim, 128),
            nn.BatchNorm1d(128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Tanh(),
            nn.Linear(128, y_dim),
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
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        valid_loss = validate(model, val_loader, criterion, device)
        if valid_loss < min_val_loss:
            min_val_loss = valid_loss
        # test_loss, test_loss_rmse = test(model, test_loader, device)
        timestring = get_now_string()
        train_loss_record.append(train_loss)
        valid_loss_record.append(valid_loss)
        print("[{}] Epoch: {}/{}  train Loss: {:.9f}  val Loss: {:.9f}  min val Loss: {:.9f}  lr: {:.9f}".format(timestring, epoch, epochs, train_loss, valid_loss, min_val_loss, optimizer.param_groups[0]["lr"]))
        scheduler.step()
        if epoch % 500 == 0:
            torch.save(model.state_dict(), main_path + "saves/model_{}.pt".format(timestring))
            # print("model saved to {}".format(main_path + "saves/model_{}.pt".format(timestring)))
        # if (epoch + 1) % 10 == 0:
        #     plt.figure(figsize=(16, 9))
        #     plt.plot(range(1, len(train_loss_record) + 1), train_loss_record, label="train loss")
        #     plt.plot(range(1, len(valid_loss_record) + 1), valid_loss_record, label="valid loss")
        #     plt.legend()
        #     plt.show()
        #     plt.close()


if __name__ == "__main__":
    main_path = "./"
    with open(main_path + "processed/all.pkl", "rb") as f:
        dataset = pickle.load(f)
    with open(main_path + "processed/train.pkl", "rb") as f:
        train_dataset = pickle.load(f)
    with open(main_path + "processed/valid.pkl", "rb") as f:
        val_dataset = pickle.load(f)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_id = 2
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device('cuda', gpu_id)
    else:
        device = torch.device('cpu')
    print("using {}".format(device))

    model = MyModel(x_dim=dataset.x_dim, y_dim=dataset.y_dim).to(device)
    # model.load_state_dict(torch.load(main_path + "saves/model_20230228_211049_069082.pt"))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: 1 / (e / 1000 + 1))
    epochs = 5000

    run(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device, main_path)





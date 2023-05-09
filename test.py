import pickle
import torch
import os
from torch.utils.data import Dataset, DataLoader

from model import MyModel
from utils import get_now_string
from dataset import MyDataset


def one_time_generate_res(pt_path, timestring=None):
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

    with open(save_output_path, "a") as f:
        f.write("val_id,x1,x2,x3,x4,x5,x6,y1,y2,y3,y4,y5,y6,y_pred_1,y_pred_2,y_pred_3,y_pred_4,y_pred_5,y_pred_6\n")
        row_id = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(dtype=torch.float32).to(device), labels.to(dtype=torch.float32).to(device)
                outputs = model(inputs)
                inputs = inputs.cpu().detach().numpy()
                labels = labels.cpu().detach().numpy()
                outputs = outputs.cpu().detach().numpy()
                for i in range(len(inputs)):
                    row_id += 1
                    f.write("{0:d},{1},{2},{3}\n".format(
                        row_id,
                        ",".join([str("{0:.9e}".format(item)) for item in inputs[i]]),
                        ",".join([str("{0:.9e}".format(item)) for item in labels[i]]),
                        ",".join([str("{0:.9e}".format(item)) for item in outputs[i]]),
                    ))


if __name__ == "__main__":
    one_time_generate_res("saves/model_20230509_052152_722761.pt")

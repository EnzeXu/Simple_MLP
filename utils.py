import math
import numpy as np
from datetime import datetime

import pickle
import torch
import pandas as pd
import os
import math
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from matplotlib.cm import ScalarMappable


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


def draw_3d_points(data_truth, data_prediction, data_error, data_error_remarkable, save_path, title=None):
    # data_truth = data_truth
    # data_prediction = data_prediction
    np.random.shuffle(data_truth)
    np.random.shuffle(data_prediction)
    np.random.shuffle(data_error)
    np.random.shuffle(data_error_remarkable)
    fig = plt.figure(figsize=(16, 16))

    truth_y_min = np.min(data_truth[:, -1])
    truth_y_max = np.max(data_truth[:, -1])
    error_y_min = np.min(data_error[:, -1])
    error_y_max = np.max(data_error[:, -1])

    x_label = "k_hyz"
    y_label = "k_pyx"
    z_label = "k_smzx"

    ax1 = fig.add_subplot(221, projection='3d')
    x = [point[0] for point in data_truth]
    y = [point[1] for point in data_truth]
    z = [point[2] for point in data_truth]
    val = [point[3] for point in data_truth]

    cmap = 'hot'

    scatter = ax1.scatter(x, y, z, c=val, cmap=cmap, alpha=0.4)
    colorbar = plt.colorbar(ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=truth_y_min, vmax=truth_y_max)), ax=ax1, shrink=0.5)

    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax1.set_zlabel(z_label)
    ax1.tick_params(axis='x', labelsize=10)
    ax1.tick_params(axis='y', labelsize=10)
    ax1.tick_params(axis='z', labelsize=10)
    ax1.set_title("Truth of CYCLE_TIME", fontsize=20)


    ax2 = fig.add_subplot(222, projection='3d')
    x = [point[0] for point in data_prediction]
    y = [point[1] for point in data_prediction]
    z = [point[2] for point in data_prediction]
    val = [point[3] for point in data_prediction]

    cmap = 'hot'

    scatter = ax2.scatter(x, y, z, c=val, cmap=cmap, alpha=0.4)
    colorbar = plt.colorbar(ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=truth_y_min, vmax=truth_y_max)), ax=ax2, shrink=0.5)

    ax2.set_xlabel(x_label)
    ax2.set_ylabel(y_label)
    ax2.set_zlabel(z_label)
    ax2.tick_params(axis='x', labelsize=10)
    ax2.tick_params(axis='y', labelsize=10)
    ax2.tick_params(axis='z', labelsize=10)
    ax2.set_title("Prediction of CYCLE_TIME", fontsize=20)

    ax3 = fig.add_subplot(223, projection='3d')
    x = [point[0] for point in data_error]
    y = [point[1] for point in data_error]
    z = [point[2] for point in data_error]
    val = [point[3] for point in data_error]

    cmap = 'cool'

    scatter = ax3.scatter(x, y, z, c=val, cmap=cmap, alpha=0.4)
    colorbar = plt.colorbar(ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=error_y_min, vmax=0.05)), ax=ax3, shrink=0.5)

    ax3.set_xlabel(x_label)
    ax3.set_ylabel(y_label)
    ax3.set_zlabel(z_label)
    ax3.tick_params(axis='x', labelsize=10)
    ax3.tick_params(axis='y', labelsize=10)
    ax3.tick_params(axis='z', labelsize=10)
    ax3.set_title("Error Distribution", fontsize=20)

    ax4 = fig.add_subplot(224, projection='3d')
    x = [point[0] for point in data_error_remarkable]
    y = [point[1] for point in data_error_remarkable]
    z = [point[2] for point in data_error_remarkable]
    val = [point[3] for point in data_error_remarkable]

    cmap = 'cool'

    scatter = ax4.scatter(x, y, z, c=val, cmap=cmap, alpha=0.4)
    colorbar = plt.colorbar(ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=error_y_min, vmax=0.05)), ax=ax4,
                            shrink=0.5)

    ax4.set_xlabel(x_label)
    ax4.set_ylabel(y_label)
    ax4.set_zlabel(z_label)
    ax4.tick_params(axis='x', labelsize=10)
    ax4.tick_params(axis='y', labelsize=10)
    ax4.tick_params(axis='z', labelsize=10)
    ax4.set_title("Remarkable Error ($e>0.05$, $n_{{R}}={0:d}$) Distribution".format(len(data_error_remarkable)), fontsize=20)  # , len(data_error_remarkable) / len(data_error) * 100.0

    #remarkable

    # plt.show()
    if title:
        fig.suptitle(title, fontsize=20)
    plt.tight_layout()
    # plt.subplots_adjust(right=0.8)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print("max", np.max(data_error[:, -1]))
    print("min", np.min(data_error[:, -1]))
    plot_value_distribution(data_error[:, -1], save_path=save_path.replace(".png", "_distribution.png"))


def draw_2d_points(data_truth, data_prediction, data_error, data_error_remarkable, save_path, title=None):
    # data_truth = data_truth
    # data_prediction = data_prediction
    np.random.shuffle(data_truth)
    np.random.shuffle(data_prediction)
    np.random.shuffle(data_error)
    np.random.shuffle(data_error_remarkable)
    fig = plt.figure(figsize=(16, 16))

    truth_y_min = np.min(data_truth[:, -1])
    truth_y_max = np.max(data_truth[:, -1])
    error_y_min = np.min(data_error[:, -1])
    error_y_max = np.max(data_error[:, -1])
    error_y_min_remarkable = np.min(data_error_remarkable[:, -1])
    error_y_max_remarkable = np.max(data_error_remarkable[:, -1])

    x_label = "k_hyz"
    y_label = "k_pyx"
    # z_label = "k_smzx"

    ax1 = fig.add_subplot(221)
    x = [point[0] for point in data_truth]
    y = [point[1] for point in data_truth]
    # z = [point[2] for point in data_truth]
    val = [point[2] for point in data_truth]

    cmap = 'hot'

    scatter = ax1.scatter(x, y, c=val, cmap=cmap, alpha=0.4)
    colorbar = plt.colorbar(ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=truth_y_min, vmax=truth_y_max)), ax=ax1, shrink=0.5)

    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    # ax1.set_zlabel(z_label)
    ax1.tick_params(axis='x', labelsize=10)
    ax1.tick_params(axis='y', labelsize=10)
    # ax1.tick_params(axis='z', labelsize=10)
    ax1.set_title("Truth of CYCLE_TIME", fontsize=20)


    ax2 = fig.add_subplot(222)
    x = [point[0] for point in data_prediction]
    y = [point[1] for point in data_prediction]
    # z = [point[2] for point in data_prediction]
    val = [point[2] for point in data_prediction]

    cmap = 'hot'

    scatter = ax2.scatter(x, y, c=val, cmap=cmap, alpha=0.4)
    colorbar = plt.colorbar(ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=truth_y_min, vmax=truth_y_max)), ax=ax2, shrink=0.5)

    ax2.set_xlabel(x_label)
    ax2.set_ylabel(y_label)
    # ax2.set_zlabel(z_label)
    ax2.tick_params(axis='x', labelsize=10)
    ax2.tick_params(axis='y', labelsize=10)
    # ax2.tick_params(axis='z', labelsize=10)
    ax2.set_title("Prediction of CYCLE_TIME", fontsize=20)

    ax3 = fig.add_subplot(223)
    x = [point[0] for point in data_error]
    y = [point[1] for point in data_error]
    # z = [point[2] for point in data_error]
    val = [point[2] for point in data_error]

    cmap = 'cool'

    scatter = ax3.scatter(x, y, c=val, cmap=cmap, alpha=0.4)
    colorbar = plt.colorbar(ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=error_y_min, vmax=error_y_max)), ax=ax3, shrink=0.5) # vmin=error_y_min, vmax=0.05

    ax3.set_xlabel(x_label)
    ax3.set_ylabel(y_label)
    # ax3.set_zlabel(z_label)
    ax3.tick_params(axis='x', labelsize=10)
    ax3.tick_params(axis='y', labelsize=10)
    # ax3.tick_params(axis='z', labelsize=10)
    ax3.set_title("Error Distribution", fontsize=20)

    ax4 = fig.add_subplot(224)
    x = [point[0] for point in data_error_remarkable]
    y = [point[1] for point in data_error_remarkable]
    # z = [point[2] for point in data_error_remarkable]
    val = [point[2] for point in data_error_remarkable]

    cmap = 'cool'

    scatter = ax4.scatter(x, y, c=val, cmap=cmap, alpha=0.4)
    colorbar = plt.colorbar(ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=error_y_min_remarkable, vmax=error_y_max_remarkable)), ax=ax4,
                            shrink=0.5)

    ax4.set_xlabel(x_label)
    ax4.set_ylabel(y_label)
    # ax4.set_zlabel(z_label)
    ax4.tick_params(axis='x', labelsize=10)
    ax4.tick_params(axis='y', labelsize=10)
    # ax4.tick_params(axis='z', labelsize=10)
    ax4.set_title("Remarkable Error ($e>0.10$, $n_{{R}}={0:d}$) Distribution".format(len(data_error_remarkable)), fontsize=20)  # , len(data_error_remarkable) / len(data_error) * 100.0

    #remarkable

    # plt.show()
    if title:
        fig.suptitle(title, fontsize=20)
    plt.tight_layout()
    # plt.subplots_adjust(right=0.8)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print("max", np.max(data_error[:, -1]))
    print("min", np.min(data_error[:, -1]))
    plot_value_distribution(data_error[:, -1], save_path=save_path.replace(".png", "_distribution.png"))

def one_time_draw_3d_points_from_txt(txt_path, save_path, title=None, log_flag=False):
    with open(txt_path, "r") as f:
        lines = f.readlines()
    lines = [line for line in lines if "," in line and "x" not in line]
    data_truth = []
    data_prediction = []
    data_error = []
    data_error_remarkable = []
    # data_2D_error = []
    # data_2D_truth = []
    error_sum = 0.0
    for one_line in lines[:]:
        parts = one_line.split(",")
        if log_flag:
            parts[1] = np.log(float(parts[1]))
            parts[2] = np.log(float(parts[2]))
            parts[3] = np.log(float(parts[3]))
        data_truth.append((float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))

        data_prediction.append((float(parts[1]), float(parts[2]), float(parts[3]), float(parts[5])))
        one_error = abs((float(parts[5]) - float(parts[4])) / float(parts[4]))
        error_sum += one_error
        data_error.append((float(parts[1]), float(parts[2]), float(parts[3]), one_error))
        if one_error > 0.05:
            data_error_remarkable.append((float(parts[1]), float(parts[2]), float(parts[3]), one_error))
    # print("data_truth:")
    # print(data_truth)
    # print("data_prediction:")
    # print(data_prediction)
    # print("data_error:")
    # print(data_error)

    print(f"## Average Error: {error_sum:.6f} / {len(lines)} = {error_sum / len(lines):.12f}")

    draw_3_2d_points(data_error, data_truth, save_path.replace(".png", "_2D.png"))

    data_truth = np.asarray(data_truth)
    data_prediction = np.asarray(data_prediction)
    data_error = np.asarray(data_error)
    draw_3d_points(data_truth, data_prediction, data_error, data_error_remarkable, save_path, title.format(len(lines)))
    print(f"saved \"{title}\" to {save_path}")

def one_time_draw_2d_points_from_txt(txt_path, save_path, title=None, log_flag=False):
    with open(txt_path, "r") as f:
        lines = f.readlines()
    lines = [line for line in lines if "," in line and "x" not in line]
    data_truth = []
    data_prediction = []
    data_error = []
    data_error_remarkable = []
    # data_2D_error = []
    # data_2D_truth = []
    error_sum = 0.0
    for one_line in lines[:]:
        parts = one_line.split(",")
        if log_flag:
            parts[1] = np.log(float(parts[1]))
            parts[2] = np.log(float(parts[2]))
            # parts[3] = np.log(float(parts[3]))
        data_truth.append((float(parts[1]), float(parts[2]), float(parts[3])))

        data_prediction.append((float(parts[1]), float(parts[2]), float(parts[4])))
        one_error = abs((float(parts[4]) - float(parts[3])) / float(parts[3]))
        error_sum += one_error
        data_error.append((float(parts[1]), float(parts[2]), one_error))
        if one_error > 0.1:
            data_error_remarkable.append((float(parts[1]), float(parts[2]), one_error))
    # print("data_truth:")
    # print(data_truth)
    # print("data_prediction:")
    # print(data_prediction)
    # print("data_error:")
    # print(data_error)

    print(f"## Average Error: {error_sum:.6f} / {len(lines)} = {error_sum / len(lines):.12f}")

    # draw_3_2d_points(data_error, data_truth, save_path.replace(".png", "_2D.png"))

    data_truth = np.asarray(data_truth)
    data_prediction = np.asarray(data_prediction)
    data_error = np.asarray(data_error)
    data_error_remarkable = np.asarray(data_error_remarkable)
    draw_2d_points(data_truth, data_prediction, data_error, data_error_remarkable, save_path, title.format(len(lines)))
    print(f"saved \"{title}\" to {save_path}")


def plot_value_distribution(data, save_path):
    fig = plt.figure(figsize=(12, 6))
    bin_edges = np.arange(0.0, 1.0, 0.05)

    # Calculate the histogram of the data using the defined bins
    hist, _ = np.histogram(data, bins=bin_edges)

    # Calculate the frequencies as the relative count in each bin
    frequencies = hist / len(data)

    ax = fig.add_subplot(111)

    # Plot the bars with the frequencies
    bars = ax.bar(bin_edges[:-1], frequencies, width=0.05, align='edge', color="orange")

    ax.set_xlabel('Relative Error')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution')

    # Set x-axis ticks to match the desired range [0.1, 0.2, ..., 0.9, 1.0]
    x_ticks = np.arange(0.0, 1.0, 0.05)
    plt.xticks(x_ticks)

    # Add count labels on top of each bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.annotate(f'{hist[i]:d}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom')

    plt.savefig(save_path, dpi=300)
    plt.close()


def one_time_filter_data(data_path, filter_list):
    with open(data_path, "r") as f:
        lines = f.readlines()
    lines = [line for line in lines if len(line) > 10 and "k_" not in line]

    n_col = len(lines[0].split(","))
    print(f"n_col={n_col}, lines[0]={lines[0]}")
    assert n_col in [6, 7, 11], "n_col should be in [6,7,11], but {} was found in row '{}'".format(n_col, lines[0])
    if n_col == 7:
        y_start_col = 3
    elif n_col == 6:
        y_start_col = 2
    else:
        y_start_col = 7
    print(f"# n_col = {n_col}, so y_start_col = {y_start_col}")

    print(f"Initial: all {len(lines)} lines")

    for one_filter in filter_list:
        save_path = data_path.replace(".csv", f"_{'all' if one_filter > 1000 else one_filter}.csv")
        # with open(save_path, "w") as f_tmp:
        #     pass
        f_write = open(save_path, "w")

        count_inf = 0
        count_normal = 0
        count_normal_remain = 0
        count_bad = 0

        print(f"# filter: <{one_filter} or inf")
        for one_line in lines:
            parts = one_line.split(",")
            c1, c2, c3 = parts[y_start_col], parts[y_start_col + 1], parts[y_start_col + 2]
            if c1 == c2 == c3 == "inf":
                count_inf += 1
                f_write.write(one_line)
            elif c1 == "inf" or c2 == "inf" or c3 == "inf":
                count_bad += 1
            else:
                c1_f, c2_f, c3_f = float(c1), float(c2), float(c3)
                if max(c1_f, c2_f, c3_f) - min(c1_f, c2_f, c3_f) > 5:
                    count_bad += 1
                else:
                    count_normal += 1
                    if c1_f < one_filter:
                        count_normal_remain += 1
                        f_write.write(one_line)
        f_write.close()
        print(f"count_inf: {count_inf}")
        print(f"count_normal: {count_normal} ({count_normal_remain} remain for matching \"<{one_filter}\"))")
        print(f"count_bad: {count_bad}")


def draw_3_2d_points(data_error, data_truth, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))

    x_label = "k_hyz"
    y_label = "k_pyx"
    z_label = "k_smzx"

    x_error = np.array([item[0] for item in data_error])
    y_error = np.array([item[1] for item in data_error])
    z_error = np.array([item[2] for item in data_error])
    value_error = np.array([item[3] for item in data_error])

    x_truth = np.array([item[0] for item in data_truth])
    y_truth = np.array([item[1] for item in data_truth])
    z_truth = np.array([item[2] for item in data_truth])
    value_truth = np.array([item[3] for item in data_truth])

    point_size = 15
    alpha_level = 0.5

    cmap = "cool"

    ax = axes[0][0]
    sc = ax.scatter(x_error, y_error, c=value_error, alpha=alpha_level, s=point_size, cmap=cmap)
    ax.set_xlabel(x_label, fontsize=20)
    ax.set_ylabel(y_label, fontsize=20)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_title('2D-XY: Error Rate of Circle Time', fontsize=20)
    colorbar = plt.colorbar(ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(value_error), vmax=0.05)),
                            ax=ax, shrink=0.5)

    ax = axes[0][1]
    sc = ax.scatter(x_error, z_error, c=value_error, alpha=alpha_level, s=point_size, cmap=cmap)
    ax.set_xlabel(x_label, fontsize=20)
    ax.set_ylabel(z_label, fontsize=20)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_title('2D-XZ: Error Rate of Circle Time', fontsize=20)
    colorbar = plt.colorbar(ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(value_error), vmax=0.05)),
                            ax=ax, shrink=0.5)

    ax = axes[0][2]
    sc = ax.scatter(y_error, z_error, c=value_error, alpha=alpha_level, s=point_size, cmap=cmap)
    ax.set_xlabel(y_label, fontsize=20)
    ax.set_ylabel(z_label, fontsize=20)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_title('2D-YZ: Error Rate of Circle Time', fontsize=20)
    colorbar = plt.colorbar(ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(value_error), vmax=0.05)),
                            ax=ax, shrink=0.5)

    cmap = "hot"

    ax = axes[1][0]
    sc = ax.scatter(x_truth, y_truth, c=value_truth, alpha=alpha_level, s=point_size, cmap=cmap)
    ax.set_xlabel(x_label, fontsize=20)
    ax.set_ylabel(y_label, fontsize=20)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_title('2D-XY: Truth of Circle Time', fontsize=20)
    colorbar = plt.colorbar(ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(value_truth), vmax=max(value_truth))),
                            ax=ax, shrink=0.5)

    ax = axes[1][1]
    sc = ax.scatter(x_truth, z_truth, c=value_truth, alpha=alpha_level, s=point_size, cmap=cmap)
    ax.set_xlabel(x_label, fontsize=20)
    ax.set_ylabel(z_label, fontsize=20)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_title('2D-XZ: Truth of Circle Time', fontsize=20)
    colorbar = plt.colorbar(ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(value_truth), vmax=max(value_truth))),
                            ax=ax, shrink=0.5)

    ax = axes[1][2]
    sc = ax.scatter(y_truth, z_truth, c=value_truth, alpha=alpha_level, s=point_size, cmap=cmap)
    ax.set_xlabel(y_label, fontsize=20)
    ax.set_ylabel(z_label, fontsize=20)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_title('2D-YZ: Truth of Circle Time', fontsize=20)
    colorbar = plt.colorbar(ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(value_truth), vmax=max(value_truth))),
                            ax=ax, shrink=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    # a = np.asarray([1.0, 2.0, 3.0])
    # a = torch.tensor([1.0, 2.0, 3.0])
    # print(my_min_max(a))

    # timestring = "20230603_063106_059898"#"20230603_044727_785177"
    # one_time_draw_3d_points_from_txt(f"record/output/output_{timestring}_best_train.txt", f"test/comparison_{timestring}_best_train_log.png", title="Results of the Train Set (n=28944) [dataset=k_hyz_k_pyx_k_smzx]", log_flag=True)
    # one_time_draw_3d_points_from_txt(f"record/output/output_{timestring}_best_val.txt", f"test/comparison_{timestring}_best_test_log.png", title="Results of the Test Set (n=7237) [dataset=k_hyz_k_pyx_k_smzx]", log_flag=True)
    # one_time_draw_3d_points_from_txt(f"record/output/output_{timestring}_last_train.txt", f"test/comparison_{timestring}_last_train_log.png", title="Results of the Train Set (n=28944) [dataset=k_hyz_k_pyx_k_smzx]", log_flag=True)
    # one_time_draw_3d_points_from_txt(f"record/output/output_{timestring}_last_val.txt", f"test/comparison_{timestring}_last_test_log.png", title="Results of the Test Set (n=7237) [dataset=k_hyz_k_pyx_k_smzx]", log_flag=True)

    # a = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
    # a = np.asarray(a)
    # np.random.shuffle(a)
    # print(a)
    # one_time_filter_data("data/dataset_osci_3_4_5_v0604.csv", [999999, 200, 100])
    # one_time_filter_data("data/dataset_osci_0_1_v0618.csv", [999999, 200, 100])
    # one_time_filter_data("data/dataset_osci_v0628_large.csv", [999999, 200, 100])

    one_time_filter_data("data/dataset_osci_v0628_small.csv", [999999, 200, 100])

    # !output_20230612_000531_827740 *
    # !output_20230612_000514_516716 *
    # !output_20230612_000525_078931 *

    # !output_20230621_175954_194414 *
    # !output_20230621_175959_346470 *
    # !output_20230621_180002_774007 *

    # for timestring in ["20230621_175954_194414", "20230621_175959_346470", "20230621_180002_774007"]:
    #     # one_time_draw_3d_points_from_txt(f"record/output/output_{timestring}_last_train.txt",
    #     #                                  f"test/comparison_{timestring}_last_train.png",
    #     #                                  title="Results of the Train Set (n={}) [dataset=k_hyz_k_pyx_k_smzx]", log_flag=False)
    #     one_time_draw_2d_points_from_txt(f"record/output/output_{timestring}_last_val.txt",
    #                                      f"test/comparison_{timestring}_last_test.png",
    #                                      title="Results of the Test Set (n={}) [dataset=k_hyz_k_pyx_k_smzx]", log_flag=True)
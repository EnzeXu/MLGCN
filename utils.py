import os
import random
import torch
import scipy.special
import math
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Optional
from torch import Tensor
from torch_scatter import scatter


def generate_gap_file(folder, save_path, length, file_format="EIGENVAL_{}"):
    # files = os.listdir(folder)
    print("{}: {} files".format(folder, length))
    # files.sort()
    gaps = []
    # print(files[:30])
    for i in range(length):
        filename = os.path.join(folder, file_format.format(i + 1))
        with open(filename, "r") as f:
            lines = f.readlines()
        if len(lines) < 80 or (float(lines[40]) - float(lines[39]) < 1.0 and float(lines[41]) - float(lines[40]) < 1.0) or float(lines[40]) - float(lines[39]) > 10 or float(lines[41]) - float(lines[40]) > 10:
            print(filename, len(lines))
            one_gap = np.mean(np.asarray(gaps))
        else:
            one_gap = float(lines[40]) - float(lines[39])
        # print(one_gap)
        gaps.append(one_gap)
    gaps = np.asarray(gaps)
    print(len(gaps))
    np.save(save_path, gaps)


def my_global_add_pool(x: Tensor, batch: Optional[Tensor],
                    size: Optional[int] = None) -> Tensor:
    if batch is None:
        return x.sum(dim=1, keepdim=True)
    size = int(batch.max().item() + 1) if size is None else size
    return scatter(x, batch, dim=0, dim_size=size, reduce='add')


def worker_init_fn(worker_id, seed=0):
    random.seed(seed + worker_id)


def func_loss3(x):
    return 1e-4 * (scipy.special.lambertw(1e2 * math.e ** (1e2 - 1e4 * x)).real + 1e4 * x - 1e2)


def loss_function(pred, true, loss_type, loss_id):
    criterion = nn.MSELoss()
    assert loss_type in ["train", "test"] and loss_id in [1, 2, 3]
    loss = criterion(pred, true)
    if loss_type == "train":
        if loss_id == 1:
            loss = criterion(pred, true)
        elif loss_id == 2:
            loss = criterion(pred, torch.round(true, decimals=2))
        elif loss_id == 3:
            loss = criterion(func_loss3(np.abs(pred - true)))
    elif loss_type == "test":
        if loss_id == 1:
            loss = criterion(pred, true)
        elif loss_id == 2:
            loss = criterion(torch.round(pred, decimals=2), torch.round(true, decimals=2))
        elif loss_id == 3:
            loss = criterion(func_loss3(np.abs(pred - true)))
    return loss


def compute_correlation(x, y):
    xBar = np.mean(x)
    yBar = np.mean(y)
    SSR = 0.0
    varX = 0.0
    varY = 0.0
    for i in range(0, len(x)):
        diffXXbar = x[i] - xBar
        difYYbar = y[i] - yBar
        SSR += (diffXXbar * difYYbar)
        varX += diffXXbar ** 2
        varY += difYYbar ** 2
    SST = math.sqrt(varX * varY)
    return SSR / SST



def draw_two_dimension(
        y_lists,
        x_list,
        color_list,
        line_style_list,
        legend_list=None,
        legend_fontsize=15,
        fig_title=None,
        fig_x_label="time",
        fig_y_label="val",
        show_flag=True,
        save_flag=False,
        save_path=None,
        save_dpi=300,
        fig_title_size=20,
        fig_grid=False,
        marker_size=0,
        line_width=2,
        x_label_size=15,
        y_label_size=15,
        number_label_size=15,
        fig_size=(8, 6)
) -> None:
    """
    Draw a 2D plot of several lines
    :param y_lists: (list[list]) y value of lines, each list in which is one line. e.g., [[2,3,4,5], [2,1,0,-1], [1,4,9,16]]
    :param x_list: (list) x value shared by all lines. e.g., [1,2,3,4]
    :param color_list: (list) color of each line. e.g., ["red", "blue", "green"]
    :param line_style_list: (list) line style of each line. e.g., ["solid", "dotted", "dashed"]
    :param legend_list: (list) legend of each line, which CAN BE LESS THAN NUMBER of LINES. e.g., ["red line", "blue line", "green line"]
    :param legend_fontsize: (float) legend fontsize. e.g., 15
    :param fig_title: (string) title of the figure. e.g., "Anonymous"
    :param fig_x_label: (string) x label of the figure. e.g., "time"
    :param fig_y_label: (string) y label of the figure. e.g., "val"
    :param show_flag: (boolean) whether you want to show the figure. e.g., True
    :param save_flag: (boolean) whether you want to save the figure. e.g., False
    :param save_path: (string) If you want to save the figure, give the save path. e.g., "./test.png"
    :param save_dpi: (integer) If you want to save the figure, give the save dpi. e.g., 300
    :param fig_title_size: (float) figure title size. e.g., 20
    :param fig_grid: (boolean) whether you want to display the grid. e.g., True
    :param marker_size: (float) marker size. e.g., 0
    :param line_width: (float) line width. e.g., 1
    :param x_label_size: (float) x label size. e.g., 15
    :param y_label_size: (float) y label size. e.g., 15
    :param number_label_size: (float) number label size. e.g., 15
    :param fig_size: (tuple) figure size. e.g., (8, 6)
    :return:
    """
    assert len(y_lists[0]) == len(x_list), "Dimension of y should be same to that of x"
    assert len(y_lists) == len(line_style_list) == len(color_list), "number of lines should be fixed"
    y_count = len(y_lists)
    plt.figure(figsize=fig_size)
    for i in range(y_count):
        plt.plot(x_list, y_lists[i], markersize=marker_size, linewidth=line_width, c=color_list[i], linestyle=line_style_list[i])
    plt.xlabel(fig_x_label, fontsize=x_label_size)
    plt.ylabel(fig_y_label, fontsize=y_label_size)
    plt.tick_params(labelsize=number_label_size)
    if legend_list:
        plt.legend(legend_list, fontsize=legend_fontsize)
    if fig_title:
        plt.title(fig_title, fontsize=fig_title_size)
    if fig_grid:
        plt.grid(True)
    if save_flag:
        plt.savefig(save_path, dpi=save_dpi)
    if show_flag:
        plt.show()
    plt.clf()
    plt.close()


def draw_two_dimension_regression(
        y_lists,
        x_lists,
        color_list,
        line_style_list,
        legend_list=None,
        legend_fontsize=15,
        fig_title=None,
        fig_x_label="time",
        fig_y_label="val",
        show_flag=True,
        save_flag=False,
        save_path=None,
        save_dpi=300,
        fig_title_size=20,
        fig_grid=False,
        marker_size=0,
        line_width=2,
        x_label_size=15,
        y_label_size=15,
        number_label_size=15,
        fig_size=(8, 6)
) -> None:
    """
    Draw a 2D plot of several lines
    :param y_lists: (list[list]) y value
    :param x_lists: (list[list]) x value
    :param color_list: (list) color of each line. e.g., ["red", "blue", "green"]
    :param line_style_list: (list) line style of each line. e.g., ["solid", "dotted", "dashed"]
    :param legend_list: (list) legend of each line, which CAN BE LESS THAN NUMBER of LINES. e.g., ["red line", "blue line", "green line"]
    :param legend_fontsize: (float) legend fontsize. e.g., 15
    :param fig_title: (string) title of the figure. e.g., "Anonymous"
    :param fig_x_label: (string) x label of the figure. e.g., "time"
    :param fig_y_label: (string) y label of the figure. e.g., "val"
    :param show_flag: (boolean) whether you want to show the figure. e.g., True
    :param save_flag: (boolean) whether you want to save the figure. e.g., False
    :param save_path: (string) If you want to save the figure, give the save path. e.g., "./test.png"
    :param save_dpi: (integer) If you want to save the figure, give the save dpi. e.g., 300
    :param fig_title_size: (float) figure title size. e.g., 20
    :param fig_grid: (boolean) whether you want to display the grid. e.g., True
    :param marker_size: (float) marker size. e.g., 0
    :param line_width: (float) line width. e.g., 1
    :param x_label_size: (float) x label size. e.g., 15
    :param y_label_size: (float) y label size. e.g., 15
    :param number_label_size: (float) number label size. e.g., 15
    :param fig_size: (tuple) figure size. e.g., (8, 6)
    :return:
    """
    y_count = len(y_lists)
    for i in range(y_count):
        assert len(y_lists[i]) == len(x_lists[i]), "Dimension of y should be same to that of x"
    assert len(y_lists) == len(x_lists) == len(line_style_list) == len(color_list), "number of lines should be fixed"

    plt.figure(figsize=fig_size)
    for i in range(y_count):
        # plt.plot(x_lists[i], y_lists[i], markersize=marker_size, linewidth=line_width, c=color_list[i], linestyle=line_style_list[i])
        fit = np.polyfit(x_lists[i], y_lists[i], 1)
        line_fn = np.poly1d(fit)
        y_line = line_fn(x_lists[i])
        plt.scatter(x_lists[i], y_lists[i])
        plt.plot(x_lists[i], y_line, markersize=marker_size, linewidth=line_width, c=color_list[i], linestyle=line_style_list[i])
    plt.xlabel(fig_x_label, fontsize=x_label_size)
    plt.ylabel(fig_y_label, fontsize=y_label_size)
    plt.tick_params(labelsize=number_label_size)
    if legend_list:
        plt.legend(legend_list, fontsize=legend_fontsize)
    if fig_title:
        plt.title(fig_title, fontsize=fig_title_size)
    if fig_grid:
        plt.grid(True)
    if save_flag:
        plt.savefig(save_path, dpi=save_dpi)
    if show_flag:
        plt.show()
    plt.clf()
    plt.close()



if __name__ == "__main__":
    # dataset = "GCN_N3P"
    # length = 1600
    # generate_gap_file("data/{}/EIGENVALS".format(dataset), "data/{}/{}_gaps.npy".format(dataset, dataset), length, "EIGENVAL_{}")
    # generate_gap_file("data/{}/EIGENVALS".format(dataset), "data/{}/{}_gaps.npy".format(dataset, dataset), length, "EIGENVAL_BETA_{}")
    # print(func_loss3(0.005))
    # draw_two_dimension_regression(
    #     x_lists=[[2.3, 2.4, 2.6, 2.5, 2.8]],
    #     y_lists=[[2.26, 2.41, 2.56, 2.52, 2.81]],
    #     color_list=["red"],
    #     legend_list=None,
    #     line_style_list=["solid"],
    #     fig_title="Regression - {} - Loss{} - Train".format("GCN_N3P", 1),
    #     fig_x_label="Truth",
    #     fig_y_label="Predict",
    #     fig_size=(8, 6),
    #     show_flag=True,
    #     save_flag=False,
    #     save_path=None
    # )
    # pass

    print(compute_correlation([2.3, 2.4, 2.6, 2.5, 2.4], [2.26, 2.41, 2.56, 2.52, 2.81]))

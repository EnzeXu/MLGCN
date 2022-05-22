import os
import random
import numpy as np


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


def worker_init_fn(worker_id, seed=0):
    random.seed(seed + worker_id)

if __name__ == "__main__":
    # dataset = "GCN_C2P"
    # length = 1858
    # generate_gap_file("data/{}/EIGENVALS".format(dataset), "data/{}/{}_gaps.npy".format(dataset, dataset), length, "EIGENVAL_ALPHA_{}")
    # generate_gap_file("data/{}/EIGENVALS".format(dataset), "data/{}/{}_gaps.npy".format(dataset, dataset), length, "EIGENVAL_BETA_{}")
    pass

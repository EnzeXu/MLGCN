import random
import torch
import os
import linecache
import pickle
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn import Embedding, Linear, ModuleList, ReLU, Sequential
from torch_geometric.utils import degree
from torch_geometric.nn import BatchNorm, PNAConv, global_add_pool
from torch_geometric.nn import ChebConv, GCNConv  # noqa
from torch_geometric.nn import PNAConv, BatchNorm, global_mean_pool
from torch.backends import cudnn
from utils import worker_init_fn

from config import config


class MolDataset(Dataset):
    def __init__(self, properties, Y):
        self.properties = properties
        self.dic = prepare_io_data(Y)

    def __len__(self):
        return len(self.properties)

    def __getitem__(self, idx):
        return self.dic[idx]

class Net(torch.nn.Module):
    def __init__(self, device, seed=None):
        super(Net, self).__init__()
        self.seed = seed if seed else 0
        self.setup_seed(self.seed)
        # self.node_emb = Embedding(21, 75)
        self.edg_emb = Embedding(10, 50)
        self.embedding1 = nn.Linear(5, 126)
        # self.embedding2= nn.Linear(450,126)
        self.device = device

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
        max_degree = 4

        deg = generate_deg()
        # print("A=", A)
        # with open(config.main_path + 'data/smiles.txt') as f:
        #     smiles = f.readlines()[:]
        #     # print(smiles[0])
        # smiles = [s.strip() for s in smiles]
        #max_degree = 4
        # print("A=", A)
        # with open(config.main_path + 'data/smiles.txt') as f:
        #     smiles = f.readlines()[:]
        #     # print(smiles[0])
        # smiles = [s.strip() for s in smiles]
        '''
        A = np.load(config.main_path + 'data1/A.npy')

        adj = sp.coo_matrix(A)
        # print("adj=", adj)
        values = adj.data
        indices = np.vstack((adj.row, adj.col))
        edge_index = torch.LongTensor(indices)
        deg = torch.zeros(max_degree + 1, dtype=torch.long)
        d = degree(edge_index[1], num_nodes=126, dtype=torch.long)
        # print(edge_index)
        deg += torch.bincount(d, minlength=deg.numel())
        '''

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(4):
            conv = PNAConv(in_channels=126, out_channels=126,
                           aggregators=aggregators, scalers=scalers, deg=deg,
                           edge_dim=50, towers=1, pre_layers=1, post_layers=1,
                           divide_input=False)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(126))

        self.mlp = Sequential(Linear(126, 100), ReLU(), Linear(100, 50), ReLU(),
                              Linear(50, 25), ReLU(), Linear(25, 1))

    def forward(self, x, edge_index, edge_attr, batch):
        # x = self.node_emb(x.squeeze())
        x = self.embedding1(x)
        edge_attr = edge_attr.reshape(1, -1)  # edge_attr = edge_attr.reshape(1,1800)
        # edge_attr = self.embedding2(edge_attr)
        edge_attr = self.edg_emb(edge_attr)
        # print(edge_attr.shape)

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index, edge_attr)))

        print("cp 1 x:", x.shape)
        x= x.reshape(-1,126,126)
        x = global_add_pool(x, batch)
        print("cp 2 x:", x.shape)
        x =x.reshape(-1,126)
        res = self.mlp(x)
        print("cp 3 res:", res.shape)
        return res

    @staticmethod
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False


def pool2d(X, pool_size, mode='max'):
    X = X.float()
    p_h, p_w = pool_size
    Y = torch.zeros(X.shape[0] - p_h + 1, X.shape[1] - p_w + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y


def atom_feature(f1, atom_i, atomic, val):
    # atom = m.GetAtomWithIdx(atom_i)
    # lines = f.read()
    # lines = lines.strip().split(' ')
    # x_y_z=f.read()
    # x_y_z = x_y_z.strip().split(' ')
    # print(f1)
    x_y_z = linecache.getline(f1, atom_i + 1)
    # print("x_y_z:", x_y_z)
    x_y_z = x_y_z.strip().split()
    # print("x_y_z:", x_y_z)
    # print(x_y_z)
    # print(type(x_y_z[2]))
    x_y_z = list(float(x_y_z[i]) for i in range(3))
    # print(x_y_z)
    # x_y_z=
    # print(x_y_z)
    x_y_z = np.array(x_y_z)
    x_y_z = x_y_z.reshape(3, 1)
    # print(x_y_z)
    ele = []
    ele.append(val[atom_i])
    ele = list(float(ele[i]) for i in range(1))
    ele = np.array(ele)
    ele = ele.reshape(1, 1)
    # print(ele)
    atomic_num = []
    atomic_num.append(atomic[atom_i])
    atomic_num = list(float(atomic[i]) for i in range(1))
    atomic_num = np.array(atomic_num)
    atomic_num = atomic_num.reshape(1, 1)
    # print(atomic_num)
    # x_y_z=x_y_z.append(atomic_num)
    x_y_z = np.row_stack((x_y_z, atomic_num, ele))
    # x_y_z=np.hstack((x_y_z,ele))
    # ele=np.hstack((ele,atomic_num))
    # print(ele)
    # print(x_y_z)
    return x_y_z


def prepare_io_data(Y):
    data_save_path = config.main_path + "data/{0}/{0}_data.pkl".format(config.dataset)
    if os.path.exists(data_save_path):
        with open(data_save_path, "rb") as f:
            dic = pickle.load(f)
        print("load data_dic from {}".format(data_save_path))
    else:
        dic = dict()
        with open(config.main_path + 'data/{}/ATOMIC_NUMBERS'.format(config.dataset)) as f:
            atomic = f.readlines()
            atomic = [a.strip() for a in atomic]
            # atomic = [a for a in atomic[0:126]]

        with open(config.main_path + 'data/{}/VALENCE_ELECTRONS'.format(config.dataset)) as f:
            val = f.readlines()
            val = [v.strip() for v in val]
            # val = [v for v in val[0:126]]

        files_bmat = [os.path.join(config.root_bmat, config.format_bmat.format(i + 1)) for i in range(config.length)]
        files_dmat = [os.path.join(config.root_dmat, config.format_dmat.format(i + 1)) for i in range(config.length)]
        files_conf = [os.path.join(config.root_conf, config.format_conf.format(i + 1)) for i in range(config.length)]

        for idx in tqdm(range(config.length)):
            # t0 = time.time()

            # s = smiles[idx]
            # m = Chem.MolFromSmiles(s)

            # t1 = time.time()
            # print("cp 1:", t1 - t0)
            # natoms = m.GetNumAtoms()
            natoms = 126
            # print(natoms)
            # print(natoms)

            # adjacency matrix
            # print(GetAdjacencyMatrix(m))

            # files = root_files
            # print(files)
            # idxx = np.random.permutation(len(files))
            # idxx = idxx.tolist()
            graph_file = files_bmat[idx]
            # print(graph_file)
            # t2 = time.time()
            # print("cp 2:", t2 - t1)
            with open(graph_file, 'r') as f:
                lines = f.read()
                lines = lines.strip().split(' ')
                # print(lines)
                A = list(float(x) for x in lines)
                # print(A)
                A = np.array(A)
                A = A.reshape(126, 126)
                # print(A)
            # t3 = time.time()
            # print("cp 3:", t3 - t2)
            A = A + np.eye(natoms)
            A_padding = np.zeros((config.max_natoms, config.max_natoms))
            A_padding[:natoms, :natoms] = A
            # print(A.shape)
            max_degree = -1
            adj = sp.coo_matrix(A_padding)
            indices = np.vstack((adj.row, adj.col))
            edge_index = torch.LongTensor(indices)
            b = np.zeros((2, 2), dtype=np.int64)
            b = torch.from_numpy(b)
            if len(edge_index[1]) < 450:
                edge_index = torch.hstack((edge_index, b))
            d = degree(edge_index[1], num_nodes=126, dtype=torch.long)
            # print(data.edge_index[1])
            max_degree = max(max_degree, int(d.max()))
            deg = torch.zeros(max_degree + 1, dtype=torch.long)
            d = degree(edge_index[1], num_nodes=126, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())
            # print(deg)
            # t4 = time.time()
            # print("cp 4:", t4 - t3)
            if len(deg) > 5:
                deg = deg[:5]
            # print(edge_index.shape)
            # root1 = main_path + 'data/GCN_N3P/DMATRIXES/'
            # files = root1_files
            graph_file = files_dmat[idx]
            # print(graph_file)
            # t5 = time.time()
            # print("cp 5:", t5 - t4)
            with open(graph_file, 'r') as f:
                lines = f.readlines()
                lines = [lines[i].strip().replace('    ', ' ') for i in range(126)]
                lines = [lines[i].strip().replace('  ', ' ') for i in range(126)]
                lines = ' '.join(lines)
                lines = lines.split(' ')
                # lines.replace(' ','a')
                # print(lines)
                DIN = list(float(x) for x in lines)
                # print(A)
                DIN = np.array(DIN)
                DIN = DIN.reshape(126, 126)
                edge_attr = []
                for i in range(450):
                    d1 = np.array(edge_index[0][i])
                    d2 = np.array(edge_index[1][i])
                    # print(d1)
                    # print(d2)
                    # print(DIN[d1][d2])
                    if 0 < DIN[d1][d2] < 9:
                        edge_attr.append(DIN[d1][d2])
                    else:
                        edge_attr.append(1)
            edge_attr = np.array(edge_attr)
            edge_attr = edge_attr.reshape(1, 450)
            edge_attr = torch.LongTensor(edge_attr)
            # t6 = time.time()
            # print("cp 6:", t6 - t5)
            # print(edge_attr)

            # print(edge_attr)

            # atom feature
            # root2 = main_path + 'data/GCN_N3P/CONFIGS/'
            # files = root2_files
            graph_file = files_conf[idx]
            # print(graph_file)
            # root2 = './GCN_N3P/ATOMIC_NUMBERS/'
            # files1 = [f for f in os.listdir(root2) if os.path.isfile(os.path.join(root2, f))]
            # with open(graph_file, 'r') as f:
            f1 = graph_file
            # f2='./GCN_N3P/ATOMIC_NUMBERS/'

            X = [atom_feature(f1, k, atomic, val) for k in range(natoms)]
            # print(X)
            '''
            for i in range(natoms, max_natoms):
                X.append(np.zeros(34))'''
            X = np.array(X)
            X = X.reshape(126, 5)
            # print(edge_attr)
            # t7 = time.time()
            # print("cp 7:", t7 - t6)
            sample = dict()
            sample['X'] = torch.from_numpy(X)
            sample['A'] = torch.from_numpy(A_padding)
            sample['EI'] = edge_index
            sample['EA'] = edge_attr
            sample['Y'] = Y[idx]
            sample['D'] = deg
            dic[idx] = sample
        with open(data_save_path, "wb") as f:
            pickle.dump(dic, f)
        print("save data to {}".format(data_save_path))
    return dic


def generate_deg():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # device = "cpu"
    batch_size = 64
    seed = 0
    num_nodes = 126
    Y = np.load(config.main_path + "data/{0}/{0}_gaps.npy".format(config.dataset))
    train_logp = Y[:config.train_length]
    test_logp = Y[config.train_length:]
    train_dataset = MolDataset(train_logp, Y)
    test_dataset = MolDataset(test_logp, Y)
    g = torch.Generator()
    g.manual_seed(seed)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2,
                                  worker_init_fn=worker_init_fn,
                                  generator=g)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=2,
                                 worker_init_fn=worker_init_fn,
                                 generator=g)
    max_degree = -1
    for i_batch, batch in enumerate(train_dataloader):
        edge_index = \
            batch['EI'].long().to(device)
        d = degree(edge_index[0][1], num_nodes=num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for i_batch, batch in enumerate(train_dataloader):
        edge_index = \
            batch['EI'].long().to(device)
        d = degree(edge_index[0][1], num_nodes=num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    return deg



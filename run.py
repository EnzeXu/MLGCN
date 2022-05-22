import torch
import time
import torch.nn as nn
import numpy as np
from torch_geometric.loader import DataLoader

from config import config
from model import Net, MolDataset
from utils import worker_init_fn


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # device = "cpu"
    batch_size = 64
    epoch_max = 100
    initial_lr = 1e-2
    seed = 0

    print("using: {}".format(device))
    print("batch_size: {}".format(batch_size))
    print("epoch_max: {}".format(epoch_max))
    print("initial_lr: {0:06f}".format(initial_lr))
    print("random_seed: {}".format(seed))

    Y = np.load(config.main_path + "data/{0}/{0}_gaps.npy".format(config.dataset))
    model = Net(device, seed).to(device)
    model.train()
    # summary(model,[(32,126,28),(32,126,126)])

    for param in model.parameters():
        if param.dim() == 1:
            continue
            nn.init.constant(param, 0)
        else:
            nn.init.xavier_normal_(param)

    # Dataset
    # train_smiles = smiles[:1000] ##
    # test_smiles = smiles[1000:1600]
    train_logp = Y[:config.train_length]
    test_logp = Y[config.train_length:]
    train_dataset = MolDataset(train_logp, Y)
    test_dataset = MolDataset(test_logp, Y)

    # Dataloader

    g = torch.Generator()
    g.manual_seed(seed)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2, worker_init_fn=worker_init_fn, generator=g)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=2, worker_init_fn=worker_init_fn, generator=g)

    criterion = nn.MSELoss()
    # loss_list = []
    # st = time.time()
    # true=[]
    # pre=[]

    # mse=[]

    # lr=1e-4

    start_time = time.time()
    start_time_0 = start_time

    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (epoch / 10.0 + 1))  # decade

    for epoch in range(1, epoch_max + 1):
        epoch_loss = []
        # lr = start_lr * (0.1/(epoch+1))
        for i_batch, batch in enumerate(train_dataloader):
            x, y, A, D, edge_index, edge_attr = \
                batch['X'].float().to(device), batch['Y'].float().to(device), batch['A'].float().to(device), batch[
                    'D'].float().to(device), batch['EI'].long().to(device), batch['EA'].long().to(device)
            # print("x:", x.shape)
            # print("y:", y.shape, y)
            # print("A:", A.shape)
            # print("D:", D.shape)
            # print("edge_index:", edge_index.shape)
            # print("edge_attr:", edge_attr.shape)
            """
            x: torch.Size([64, 126, 5])
            y: torch.Size([64])
            A: torch.Size([64, 126, 126])
            D: torch.Size([64, 5])
            edge_index: torch.Size([64, 2, 450])
            edge_attr: torch.Size([64, 1, 450])
            """
            edge_index = edge_index.reshape(2, -1)  # edge_index = edge_index.reshape(2,1800)
            pred = model(x, edge_index, edge_attr, batch=None).squeeze(-1)
            # print("pred:", pred.shape)
            pred = pred.repeat(y.shape[0])
            # a=abs(pred.data.cpu().numpy()-y.data.cpu().numpy())
            # a=np.min(a)
            # print(a)
            optimizer.zero_grad()
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

        now_time = time.time()
        print("Epoch [{0:05d}/{1:05d}] Loss:{2:.6f} Lr:{3:.6f} Time:{4:.6f}s ({5:.2f}min in total)".format(epoch, epoch_max, loss.item(), optimizer.param_groups[0]["lr"], now_time - start_time, (now_time - start_time_0) / 60.0))
        start_time = now_time
        scheduler.step()
        # mse.append(np.mean(np.array(epoch_loss)))


if __name__ == "__main__":
    run()

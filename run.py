import torch
import time
import os
import argparse
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import DataLoader

from config import config
from model import Net, MolDataset
from utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  #（保证程序cuda序号与实际cuda序号对应）
os.environ['CUDA_VISIBLE_DEVICES'] = "3"  #（代表仅使用第0，1号GPU）


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=config.epoch, help="epoch")
    parser.add_argument("--epoch_step", type=int, default=config.epoch_step, help="epoch_step")
    parser.add_argument("--batch_size", type=int, default=config.batch_size, help="batch_size")
    parser.add_argument('--lr', type=float, default=config.lr, help='learning rate, default=0.01')
    parser.add_argument('--seed', type=int, default=config.seed, help='seed')
    parser.add_argument("--main_path", type=str, default=config.main_path, help="main_path")
    parser.add_argument("--dataset", type=str, default=config.dataset, help="dataset")
    parser.add_argument("--dataset_save_as", type=str, default=config.dataset, help="dataset_save_as")
    parser.add_argument("--max_natoms", type=int, default=config.max_natoms, help="max_natoms")
    parser.add_argument("--length", type=int, default=config.length, help="important: data length")
    parser.add_argument("--root_bmat", type=str, default=config.root_bmat, help="root_bmat")
    parser.add_argument("--root_dmat", type=str, default=config.root_dmat, help="root_dmat")
    parser.add_argument("--root_conf", type=str, default=config.root_conf, help="root_conf")
    parser.add_argument("--format_bmat", type=str, default=config.format_bmat, help="format_bmat")
    parser.add_argument("--format_dmat", type=str, default=config.format_dmat, help="format_dmat")
    parser.add_argument("--format_conf", type=str, default=config.format_conf, help="format_conf")
    parser.add_argument("--loss_fn_id", type=int, default=config.loss_fn_id, help="loss_fn_id")
    args = parser.parse_args()
    args.train_length = int(args.length * 0.8)
    args.test_length = args.length - args.train_length

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # device = "cpu"

    print("using: {}".format(args.device))
    for item in args.__dict__.items():
        if item[0][0] == "_":
            continue
        print("{}: {}".format(item[0], item[1]))


    Y = np.load(args.main_path + "data/{0}/{0}_gaps.npy".format(args.dataset))
    model = Net(args).to(args.device)
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
    train_logp = Y[:args.train_length]
    test_logp = Y[args.train_length:args.train_length + args.test_length]
    train_dataset = MolDataset(train_logp, Y, args)
    test_dataset = MolDataset(test_logp, Y, args)

    # Dataloader

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=2, worker_init_fn=worker_init_fn, generator=g, shuffle=False)
    # test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=2, worker_init_fn=worker_init_fn, generator=g, shuffle=False)

    criterion = nn.MSELoss()
    # loss_list = []
    # st = time.time()
    # true=[]
    # pre=[]

    # mse=[]

    # lr=1e-4

    start_time = time.time()
    start_time_0 = start_time
    main_save_path = f"{args.main_path}train/{args.dataset_save_as}/"
    if not os.path.exists(main_save_path):
        os.makedirs(main_save_path)

    model_save_path = f"{main_save_path}/model_last.pt"
    figure_save_path_loss_whole = f"{main_save_path}/loss_whole.png"
    figure_save_path_loss_last_half = f"{main_save_path}/loss_last_half.png"
    figure_save_path_loss_last_quarter = f"{main_save_path}/loss_last_quarter.png"

    regression_result_train_true = f"{main_save_path}/train_true.npy"
    regression_result_train_pred = f"{main_save_path}/train_pred.npy"
    regression_result_test_true = f"{main_save_path}/test_true.npy"
    regression_result_test_pred = f"{main_save_path}/test_pred.npy"
    figure_regression_train_path = f"{main_save_path}/regression_train.png"
    figure_regression_test_path = f"{main_save_path}/regression_test.png"
    print("Paths:")
    print("main_save_path: {}".format(main_save_path))
    print("model_save_path: {}".format(model_save_path))
    print("figure_save_path_loss_whole: {}".format(figure_save_path_loss_whole))
    print("figure_save_path_loss_last_half: {}".format(figure_save_path_loss_last_half))
    print("figure_save_path_loss_last_quarter: {}".format(figure_save_path_loss_last_quarter))
    print("regression_result_train_true: {}".format(regression_result_train_true))
    print("regression_result_train_pred: {}".format(regression_result_train_pred))
    print("regression_result_test_true: {}".format(regression_result_test_true))
    print("regression_result_test_pred: {}".format(regression_result_test_pred))
    print("figure_regression_train_path: {}".format(figure_regression_train_path))
    print("figure_regression_test_path: {}".format(figure_regression_test_path))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ep: 1 / (ep / 100.0 + 1))  # decade

    print("Training...")
    epoch_loss_list = []
    for epoch in range(1, args.epoch + 1):

        # lr = start_lr * (0.1/(epoch+1))
        for i_batch, batch in enumerate(train_dataloader):
            x, y, edge_index, edge_attr = \
                batch['X'].float().to(args.device), batch['Y'].float().to(args.device),\
                batch['EI'].long().to(args.device), batch['EA'].long().to(args.device)
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
            #batch=torch.
            pred = model(x, edge_index, edge_attr, batch=None).squeeze(-1)
            #print(pred.shape)
            # print("pred:", pred.shape)
            # print("y_true", y.shape)
            #pred = pred.repeat(y.shape[0])
            # a=abs(pred.data.cpu().numpy()-y.data.cpu().numpy())
            # a=np.min(a)
            # print(a)
            optimizer.zero_grad()
            # loss = criterion(pred, y)
            loss = loss_function(pred, y, "train", args.loss_fn_id)
            loss.backward()
            optimizer.step()

        if epoch % args.epoch_step == 0:
            now_time = time.time()
            print("Epoch [{0:05d}/{1:05d}] Loss:{2:.6f} Lr:{3:.6f} (Time:{4:.6f}s Time total:{5:.2f}min Time remain: {6:.2f}min)".format(epoch, args.epoch, loss.item(), optimizer.param_groups[0]["lr"], now_time - start_time, (now_time - start_time_0) / 60.0, (now_time - start_time_0) / 60.0 / epoch * (args.epoch - epoch)))
            start_time = now_time
            torch.save(
                {
                    "args": args,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "loss": loss.item()
                }, model_save_path)
        scheduler.step()  # per epoch
        epoch_loss_list.append(loss.item())

    # Draw loss
    print("Drawing loss...")
    loss_length = len(epoch_loss_list)
    loss_x = range(1, args.epoch + 1)
    # draw loss_whole
    draw_two_dimension(
        y_lists=[epoch_loss_list],
        x_list=loss_x,
        color_list=["blue"],
        legend_list=["loss: last={0:.6f}, min={1:.6f}".format(epoch_loss_list[-1], min(epoch_loss_list))],
        line_style_list=["solid"],
        fig_title="Loss - epoch",
        fig_x_label="Epoch",
        fig_y_label="Loss - whole",
        fig_size=(8, 6),
        show_flag=False,
        save_flag=True,
        save_path=figure_save_path_loss_whole
    )

    # draw loss_last_half
    draw_two_dimension(
        y_lists=[epoch_loss_list[-(loss_length // 2):]],
        x_list=loss_x[-(loss_length // 2):],
        color_list=["blue"],
        legend_list=["loss: last={0:.6f}, min={1:.6f}".format(epoch_loss_list[-1], min(epoch_loss_list))],
        line_style_list=["solid"],
        fig_title="Loss - epoch",
        fig_x_label="Epoch",
        fig_y_label="Loss - last half",
        fig_size=(8, 6),
        show_flag=False,
        save_flag=True,
        save_path=figure_save_path_loss_last_half
    )

    # draw loss_last_quarter
    draw_two_dimension(
        y_lists=[epoch_loss_list[-(loss_length // 4):]],
        x_list=loss_x[-(loss_length // 4):],
        color_list=["blue"],
        legend_list=["loss: last={0:.6f}, min={1:.6f}".format(epoch_loss_list[-1], min(epoch_loss_list))],
        line_style_list=["solid"],
        fig_title="Loss - epoch",
        fig_x_label="Epoch",
        fig_y_label="Loss - last quarter",
        fig_size=(8, 6),
        show_flag=False,
        save_flag=True,
        save_path=figure_save_path_loss_last_quarter
    )

    # Test
    print("Testing...")
    model.eval()
    train_true_list = []
    train_pred_list = []
    test_true_list = []
    test_pred_list = []
    # train_dataset = MolDataset(train_logp, Y, args)
    # test_dataset = MolDataset(test_logp, Y, args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=2,
                                  worker_init_fn=worker_init_fn, generator=g, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=2, worker_init_fn=worker_init_fn,
                                 generator=g, shuffle=False)
    for batch in train_dataloader:
        x, y, edge_index, edge_attr = \
            batch['X'].float().to(args.device), batch['Y'].float().to(args.device),\
            batch['EI'].long().to(args.device), batch['EA'].long().to(args.device)
        edge_index = edge_index.reshape(2, -1)
        pred = model(x, edge_index, edge_attr, batch=None).squeeze(-1)
        train_true_list += list(y.cpu().detach().numpy())
        train_pred_list += list(pred.cpu().detach().numpy())

    for batch in test_dataloader:
        x, y, edge_index, edge_attr = \
            batch['X'].float().to(args.device), batch['Y'].float().to(args.device),\
            batch['EI'].long().to(args.device), batch['EA'].long().to(args.device)
        edge_index = edge_index.reshape(2, -1)
        pred = model(x, edge_index, edge_attr, batch=None).squeeze(-1)
        test_true_list += list(y.cpu().detach().numpy())
        test_pred_list += list(pred.cpu().detach().numpy())

    train_true_list = np.asarray(train_true_list)
    train_pred_list = np.asarray(train_pred_list)
    test_true_list = np.asarray(test_true_list)
    test_pred_list = np.asarray(test_pred_list)

    np.save(regression_result_train_true, train_true_list)
    np.save(regression_result_train_pred, train_pred_list)
    np.save(regression_result_test_true, test_true_list)
    np.save(regression_result_test_pred, test_pred_list)

    r_train = compute_correlation(train_true_list, train_pred_list)
    draw_two_dimension_regression(
        x_lists=[train_true_list],
        y_lists=[train_pred_list],
        color_list=["red"],
        legend_list=["Regression: R={0:.6f}, R^2={1:.6f}".format(r_train, r_train ** 2.0)],
        line_style_list=["solid"],
        fig_title="Regression - {} - Loss{} - Train - {} points".format(args.dataset, args.loss_fn_id, len(train_true_list)),
        fig_x_label="Truth",
        fig_y_label="Predict",
        fig_size=(8, 6),
        show_flag=False,
        save_flag=True,
        save_path=figure_regression_train_path
    )

    r_test = compute_correlation(test_true_list, test_pred_list)
    draw_two_dimension_regression(
        x_lists=[test_true_list],
        y_lists=[test_pred_list],
        color_list=["red"],
        legend_list=["Regression: R={0:.6f}, R^2={1:.6f}".format(r_test, r_test ** 2.0)],
        line_style_list=["solid"],
        fig_title="Regression - {} - Loss{} - Test - {} points".format(args.dataset, args.loss_fn_id, len(test_true_list)),
        fig_x_label="Truth",
        fig_y_label="Predict",
        fig_size=(8, 6),
        show_flag=False,
        save_flag=True,
        save_path=figure_regression_test_path
    )



if __name__ == "__main__":
    run()

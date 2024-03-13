import argparse
from model import LogReg, Model
from utils import *
import torch
import torch as th
import torch.nn as nn
import  numpy as np
import warnings
import random
import torch.nn.functional as F
warnings.filterwarnings('ignore')

#set seed
set_seed(0)

#set parameters
parser = argparse.ArgumentParser(description='GraphACL')
parser.add_argument('--dataname', type=str, default='cora', help='Name of dataset.')
parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
parser.add_argument('--epochs', type=int, default=50, help='Training epochs.')
parser.add_argument('--lr1', type=float, default=5e-4, help='Learning rate of pretraining.')
parser.add_argument('--lr2', type=float, default=1e-3, help='Learning rate of linear evaluator.')
parser.add_argument('--wd1', type=float, default=1e-6, help='Weight decay of pretraining.')
parser.add_argument('--wd2', type=float, default=1e-5, help='Weight decay of linear evaluator.')
parser.add_argument('--n_layers', type=int, default=2, help='Number of GNN layers')
parser.add_argument('--temp', type=float, default=1, help='Temperature hyperparameter.')
parser.add_argument("--hid_dim", type=int, default=2048, help='Hidden layer dim.')
parser.add_argument('--moving_average_decay', type=float, default=0.97)
parser.add_argument('--num_MLP', type=int, default=1)
args = parser.parse_args()

# cuda_setting
if args.gpu != -1 and th.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'

if __name__ == '__main__':
    print(args)
    # load hyperparameters
    dataname = args.dataname
    hid_dim = args.hid_dim
    out_dim = args.hid_dim
    n_layers = args.n_layers
    temp = args.temp
    epochs = args.epochs
    lr1 = args.lr1
    wd1 = args.wd1
    lr2 = args.lr2
    wd2 = args.wd2
    device = args.device

    graph, feat, labels, num_class, train_idx, val_idx, test_idx = load(dataname)
    in_dim = feat.shape[1]
    graph = graph.to(device)
    feat = feat.to(device)

    model = Model(in_dim, hid_dim, out_dim, n_layers, temp, args.moving_average_decay, args.num_MLP)
    model = model.to(device)
   
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr1, weight_decay=args.wd1)
    graph = graph.remove_self_loop().add_self_loop()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = model(graph, feat)
        loss.backward()
        optimizer.step()
        model.update_moving_average()
        print('Epoch={:03d}, loss={:.4f}'.format(epoch, loss.item()))
    
    # save model parameters
    save_name = args.dataname + '.pth'
    torch.save(model.state_dict(), save_name)
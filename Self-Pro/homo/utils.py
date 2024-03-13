import numpy as np
import torch as th

from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import AmazonCoBuyPhotoDataset, AmazonCoBuyComputerDataset
from dgl.data import CoauthorCSDataset, CoauthorPhysicsDataset
from torch_geometric.datasets import WebKB, Actor
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected
import networkx as nx
import dgl
import torch
import random

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

def load(name):
    if name == 'cora':
        dataset = CoraGraphDataset()
    elif name == 'citeseer':
        dataset = CiteseerGraphDataset()
    elif name == 'pubmed':
        dataset = PubmedGraphDataset()
    elif name == 'photo':
        dataset = AmazonCoBuyPhotoDataset()
    elif name == 'comp':
        dataset = AmazonCoBuyComputerDataset()
    elif name == 'cs':
        dataset = CoauthorCSDataset()
    elif name == 'physics':
        dataset = CoauthorPhysicsDataset()

    citegraph = ['cora', 'citeseer', 'pubmed']
    cograph = ['photo', 'comp', 'cs', 'physics']
    if name in citegraph:
        graph = dataset[0]
        train_mask = graph.ndata.pop('train_mask')
        val_mask = graph.ndata.pop('val_mask')
        test_mask = graph.ndata.pop('test_mask')

        train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
        val_idx = th.nonzero(val_mask, as_tuple=False).squeeze()
        test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()

        num_class = dataset.num_classes

    if name in cograph:
        graph = dataset[0]
        train_ratio = 0.1
        val_ratio = 0.1
        test_ratio = 0.8

        N = graph.number_of_nodes()
        train_num = int(N * train_ratio)
        val_num = int(N * (train_ratio + val_ratio))

        idx = np.arange(N)
        np.random.shuffle(idx)

        train_idx = idx[:train_num]
        val_idx = idx[train_num:val_num]
        test_idx = idx[val_num:]

        train_idx = th.LongTensor(train_idx)
        val_idx = th.LongTensor(val_idx)
        test_idx = th.LongTensor(test_idx)
        num_class = dataset.num_classes

    feat = graph.ndata.pop('feat')
    labels = graph.ndata.pop('label')

    return graph, feat, labels, num_class, train_idx, val_idx, test_idx


def get_centeremb(input,index,label_num):
    device = input.device
    mean = torch.ones(index.size(0), index.size(1)).to(device)
    index=torch.tensor(index,dtype=int).to(device)
    label_num = torch.max(index) + 1
    _mean = torch.zeros(label_num, 1,device=device).scatter_add_(dim=0, index=index, src=mean)
    preventnan=torch.ones(_mean.size(),device=device)*0.0000001
    _mean=_mean + preventnan
    index = index.expand(input.size())
    c = torch.zeros(label_num, input.size(1)).to(device)
    c = c.scatter_add_(dim=0, index=index, src=input)
    c = c / _mean
    return c

def accuracy(y_pred, y_true):
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1].type_as(y_true)
    correct = preds.eq(y_true).double()
    correct = correct.sum().item()
    return correct / len(y_true)

def index2mask(index,node_num):
    ret=torch.zeros(node_num)
    ones=torch.ones_like(ret)
    ret=torch.scatter_add(input=ret,dim=0,index=index,src=ones)
    return ret.type(torch.bool)

def fewshot_split(node_num,task_num,train_shot,val_shot,label_num,labels):
    train_set, val_set, test_set = [],[],[]
    for count in range(task_num):
        index = random.sample(range(0, node_num), node_num)
        trainindex=[]
        valindex=[]
        testindex=[]
        traincount = torch.zeros(label_num)
        valcount = torch.zeros(label_num)
        for i in index:
            label=labels[i]
            if traincount[label]<train_shot:
                trainindex.append(i)
                traincount[label]+=1
            elif valcount[label]<val_shot:
                valcount[label]+=1
                valindex.append(i)
            else:
                testindex.append(i)
        train_set.append(trainindex)
        val_set.append(valindex)
        test_set.append(testindex)
    return train_set, val_set, test_set
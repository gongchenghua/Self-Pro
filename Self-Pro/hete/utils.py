### Random tools useful for saveing stuff and manipulating pickle/numpy objects
import numpy as np
import networkx as nx
import dgl
import torch
import logging
import random
from functools import lru_cache
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.datasets import WebKB, Actor
from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T
from hete.dataset import WikipediaNetwork

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

class DataSplit:

    def __init__(self, dataset, train_ind, val_ind, test_ind, shuffle=True):
        self.train_indices = train_ind
        self.val_indices = val_ind
        self.test_indices = test_ind
        self.dataset = dataset

        self.train_sampler = SubsetRandomSampler(self.train_indices)
        self.val_sampler = SubsetRandomSampler(self.val_indices)
        self.test_sampler = SubsetRandomSampler(self.test_indices)

    def get_train_split_point(self):
        return len(self.train_sampler) + len(self.val_indices)

    def get_validation_split_point(self):
        return len(self.train_sampler)

    @lru_cache(maxsize=4)
    def get_split(self, batch_size=50, num_workers=4):
        logging.debug('Initializing train-validation-test dataloaders')
        self.train_loader = self.get_train_loader(batch_size=batch_size, num_workers=num_workers)
        self.val_loader = self.get_validation_loader(batch_size=batch_size, num_workers=num_workers)
        self.test_loader = self.get_test_loader(batch_size=batch_size, num_workers=num_workers)
        return self.train_loader, self.val_loader, self.test_loader

    @lru_cache(maxsize=4)
    def get_train_loader(self, batch_size=50, num_workers=4):
        logging.debug('Initializing train dataloader')
        self.train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, sampler=self.train_sampler, shuffle=False, num_workers=num_workers)
        return self.train_loader

    @lru_cache(maxsize=4)
    def get_validation_loader(self, batch_size=50, num_workers=4):
        logging.debug('Initializing validation dataloader')
        self.val_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, sampler=self.val_sampler, shuffle=False, num_workers=num_workers)
        return self.val_loader

    @lru_cache(maxsize=4)
    def get_test_loader(self, batch_size=50, num_workers=4):
        logging.debug('Initializing test dataloader')
        self.test_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, sampler=self.test_sampler, shuffle=False, num_workers=num_workers)
        return self.test_loader


def read_real_datasets(datasets):
    if datasets in ["cornell", "texas", "wisconsin"]:
        data_set = WebKB(root=f'../datasets/', name=datasets, transform=T.NormalizeFeatures())
    elif datasets in ['squirrel', 'chameleon']:
        data_set = WikipediaNetwork(root=f'../datasets/', name=datasets, geom_gcn_preprocess=True)
    elif datasets in ['crocodile']:
        data_set = WikipediaNetwork(root=f'../datasets/', name=datasets, geom_gcn_preprocess=False)
    elif datasets == 'film':
        data_set = Actor(root=f'../datasets/film/', transform=T.NormalizeFeatures())
    data = data_set[0]
    data.edge_index = to_undirected(data.edge_index)
    G = nx.from_edgelist(data.edge_index.transpose(0, 1).numpy().tolist())
    g = dgl.from_networkx(G)
    g.ndata['attr'] = data.x
    data.train_mask = data.train_mask.transpose(0, 1)
    data.val_mask = data.val_mask.transpose(0, 1)
    data.test_mask = data.test_mask.transpose(0, 1)
    split_list = []
    for i in range(0, len(data.train_mask)):
        split_list.append({'train_idx': torch.where(data.train_mask[i])[0],
                           'valid_idx': torch.where(data.val_mask[i])[0],
                           'test_idx': torch.where(data.test_mask[i])[0]})
    labels = data.y
    return g, labels, split_list

def get_centeremb(input,index,label_num):
    device=input.device
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
            label_index=labels[i].type(torch.long)
            if traincount[label_index]<train_shot:
                trainindex.append(i)
                traincount[label_index]+=1
            elif valcount[label_index]<val_shot:
                valcount[label_index]+=1
                valindex.append(i)
            else:
                testindex.append(i)
        train_set.append(trainindex)
        val_set.append(valindex)
        test_set.append(testindex)
    return train_set, val_set, test_set    



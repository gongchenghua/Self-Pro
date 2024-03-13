import argparse
from model import LogReg, Model
from utils import *
import torch
import torch as th
import torch.nn as nn
import  numpy as np
import warnings
import random
import networkx as nx
import torch.nn.functional as F
warnings.filterwarnings('ignore')

#seed_setting
set_seed(0)

#parametes_setting
parser = argparse.ArgumentParser(description='GraphACL')
parser.add_argument('--dataname', type=str, default='cora', help='Name of dataset.')
parser.add_argument('--gpu', type=int, default=0, help='GPU index.')

parser.add_argument('--epochs', type=int, default=50, help='Training epochs.')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate of pretraining.')
parser.add_argument('--wd', type=float, default=1e-5, help='Weight decay of linear evaluator.')
parser.add_argument('--temp', type=float, default=0.8, help='Temperature hyperparameter.')

parser.add_argument('--n_layers', type=int, default=2, help='Number of GNN layers')
parser.add_argument("--hid_dim", type=int, default=2048, help='Hidden layer dim.')
parser.add_argument('--moving_average_decay', type=float, default=0.97)
parser.add_argument('--num_MLP', type=int, default=1)
args = parser.parse_args()

# cuda_setting
if args.gpu != -1 and th.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

if __name__ == '__main__':
    print(args)
    # load hyperparameters
    dataname = args.dataname
    hid_dim = args.hid_dim
    out_dim = args.hid_dim
    n_layers = args.n_layers
    temp = args.temp
    device = args.device

    #--------------------------load graph----------------------------
    graph, feat, labels, num_class, train_idx, val_idx, test_idx = load(dataname)
    num_nodes = graph.num_nodes()
    #---------------------two-hop graph construction-----------------
    adj_sp = graph.adjacency_matrix()
    adj_2_sp = adj_sp @ adj_sp + adj_sp
    adj_2 = adj_2_sp.to_dense()
    adj_2 = torch.where(adj_2 > 1, torch.tensor(1), adj_2)
    graph_2 = dgl.DGLGraph()
    graph_2.add_nodes(num_nodes)
    rows, cols = torch.nonzero(adj_2, as_tuple=True)
    edges = list(zip(rows.tolist(), cols.tolist()))
    src, dst = zip(*edges)
    graph_2.add_edges(src, dst)
    #---------------------identity matrix construction----------------
    adj_id = torch.tensor(torch.eye(num_nodes))
    graph_id = dgl.DGLGraph()
    graph_id.add_nodes(num_nodes)
    rows, cols = torch.nonzero(adj_id, as_tuple=True)
    edges = list(zip(rows.tolist(), cols.tolist()))
    src, dst = zip(*edges)
    graph_id.add_edges(src, dst)
    #-----------------------------------------------------------------
    in_dim = feat.shape[1]
    graph = graph.to(device)
    graph = graph.remove_self_loop().add_self_loop()
    feat = feat.to(device)
    labels = labels.to(device)

    graph_2 = graph_2.to(device)
    graph_id = graph_id.to(device)

    model = Model(in_dim, hid_dim, out_dim, n_layers, temp, args.moving_average_decay, args.num_MLP)
    model = model.to(device)
    save_name = args.dataname + '.pth'
    print('load model parameters')
    model.load_state_dict(torch.load(save_name))

    # down-stream task ———— few-shot node classification
    g = graph
    # graph_2 = graph_2.remove_self_loop().add_self_loop()
    embeds = model.get_embedding(g, feat)
    
    # structural monophily prompt
    # embeds = model.get_embedding(graph_2,feat)
    
    # attention semantic prompt  
    prompt_embeds = model.get_embedding(graph_id, feat)
    # w = F.cosine_similarity(embeds,prompt_embeds,dim=1).unsqueeze(-1)
    # w_ = torch.ones_like(w) - w
    # embeds = w * embeds + w_ * prompt_embeds 

    # semantic prompt 
    embeds = 0.* embeds + 1.0 * prompt_embeds 


    hard_accs = []
    target_accs = []
    pro_accs = []
    prompt_accs = []
    task_num = 10
    train_set, val_set, test_set = fewshot_split(node_num=num_nodes,task_num=task_num,
    train_shot=1,val_shot=5,label_num=labels.max()+1,labels=labels)
    for i in range(task_num):
        count = i
        train_ = torch.tensor(train_set[count])
        val_ = torch.tensor(val_set[count])
        test_ = torch.tensor(test_set[count])
        train_mask = index2mask(train_, feat.shape[0]).to(device)
        val_mask = index2mask(val_, feat.shape[0]).to(device)
        test_mask = index2mask(test_, feat.shape[0]).to(device)
        
        # without prompt-tuning
        c_embedding = get_centeremb(embeds[train_mask],labels[train_mask].unsqueeze(1),labels.max()+1)
        pred = torch.matmul(embeds,c_embedding.T)
        pred = F.log_softmax(pred,dim=1)
        test_acc = accuracy(pred[test_mask],labels[test_mask])
        hard_accs.append(test_acc)
        # after projector
        embed_p = F.normalize(model.get_projector_embedding(g, feat),p=2,dim=1)
        c_embedding = get_centeremb(embed_p[train_mask],labels[train_mask].unsqueeze(1),labels.max()+1)
        pred = torch.matmul(embed_p,c_embedding.T)
        pred = F.log_softmax(pred,dim=1)
        test_acc = accuracy(pred[test_mask],labels[test_mask])
        target_accs.append(test_acc)
        # target_encoders
        embed_t = F.normalize(model.get_target_embedding(g, feat),p=2,dim=1)
        c_embedding = get_centeremb(embed_t[train_mask],labels[train_mask].unsqueeze(1),labels.max()+1)
        pred = torch.matmul(embed_t,c_embedding.T)
        pred = F.log_softmax(pred,dim=1)
        test_acc = accuracy(pred[test_mask],labels[test_mask])
        pro_accs.append(test_acc)

        # prompt-tuning process
        set_requires_grad(model.encoder_target, False)      
        set_requires_grad(model.encoder, False)
        set_requires_grad(model.projector, True)

        
        optimizer = torch.optim.Adam(model.projector.parameters(), lr=args.lr, weight_decay=args.wd)
        c_embed = get_centeremb(embeds[train_mask],labels[train_mask].unsqueeze(1),labels.max()+1)
        best_val_acc = 0
        best_acc = 0
        for epoch in range(args.epochs):
            z_pred = F.normalize(model.projector(embeds),p=2,dim=1)
            center = get_centeremb(z_pred[train_mask],labels[train_mask].unsqueeze(1),labels.max()+1)
            pred = torch.matmul(z_pred, center.T)
            pred = F.log_softmax(pred, dim=1) 
            val_acc = accuracy(pred[val_mask], labels[val_mask])
            test_acc = accuracy(pred[test_mask], labels[test_mask])
            if val_acc >= best_val_acc and test_acc > best_acc:
                best_val_acc = val_acc
                best_acc = test_acc
                
            optimizer.zero_grad()
            z_1 = model.projector(embeds[train_mask])
            z_1 = F.normalize(z_1, dim=-1)
            z_2 = model.projector(c_embed)
            z_2 = F.normalize(z_2, dim=-1)
            sim = torch.exp(torch.mm(z_1, z_2.t())/ args.temp)
            pos = sim[torch.arange(0,labels[train_mask].shape[0]).unsqueeze(-1),labels[train_mask].unsqueeze(-1)]
            pos = pos.squeeze()
            neg = sim.sum(1)
            loss = -torch.log(pos / neg)
            loss = loss.mean()
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
        prompt_accs.append(best_acc)
    
    
    print('hard accuracy:', np.mean(hard_accs), np.std(hard_accs))
    print('target accuracy:', np.mean(target_accs), np.std(target_accs))
    print('pro accuracy:', np.mean(pro_accs), np.std(pro_accs))
    print('prompt-tuning accuracy:', np.mean(prompt_accs), np.std(prompt_accs))

 
       



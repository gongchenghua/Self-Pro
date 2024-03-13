import sys
sys.path.append("..")
from hete import utils
import seaborn as sb
import torch
from hete.model import Model
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from layers import LogReg
from torch.utils.data import Dataset
import torch.nn.functional as F
import statistics
from utils import *
import argparse

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

class NodeClassificationDataset(Dataset):
    def __init__(self, node_embeddings, labels):
        self.len = node_embeddings.shape[0]
        self.x_data = node_embeddings
        self.y_data = labels.long()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
    

# Training
def train(g, feats, model, opt, lr, epoch, device, hidden_dim, sample_size=10,moving_average_decay=0.0):
    # obtain neighbor list    
    in_nodes, out_nodes = g.edges()
    neighbor_dict = {}
    for in_node, out_node in zip(in_nodes, out_nodes):
        if in_node.item() not in neighbor_dict:
            neighbor_dict[in_node.item()] = []
        neighbor_dict[in_node.item()].append(out_node.item())

    for i in tqdm(range(epoch)):
        loss, node_embeddings = model(g, feats, neighbor_dict, device=device)
        opt.zero_grad()
        loss.backward()
        print(i, loss.item())
        opt.step()
        model.update_moving_average()
    return node_embeddings.cpu().detach(), loss.item()


def evaluate(model, loader):
    with torch.no_grad():
        correct = 0
        total = 0
        for data in loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += torch.sum(predicted == labels)
    return (correct / total).item()

def write_results(acc, best_epoch):
    best_epoch = [str(tmp) for tmp in best_epoch]
    f = open("results/" + args.dataset + '_heterSSL', 'a+')
    f.write(args.dataset + ' --epochs ' + str(args.epoch_num) + ' --seed ' + str(args.seed) + ' --lr ' + str(args.lr) + ' --moving_average_decay ' + str(args.moving_average_decay) + ' --dimension ' + str(args.dimension) + ' --sample_size ' + str(args.sample_size) + ' --wd2 ' + str(args.wd2) + ' --num_MLP ' + str(args.num_MLP) + ' --tau ' + str(args.tau) + ' --best_epochs ' + " ".join(best_epoch) + f'   Final Test: {np.mean(acc):.4f} ± {np.std(acc):.4f}\n')
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--dataset', type=str, default="texas")
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate of pretraining.')
    parser.add_argument('--wd', type=float, default=1e-5, help='Weight decay of linear evaluator.')
    parser.add_argument('--temp', type=float, default=0.8, help='Temperature hyperparameter.')
    parser.add_argument('--sample_size', type=int, default=5)
    parser.add_argument('--dimension', type=int, default=2048)
    parser.add_argument('--moving_average_decay', type=float, default=0.95)
    parser.add_argument('--num_MLP', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    args = parser.parse_args()
    if args.gpu != -1 and torch.cuda.is_available():
        device = 'cuda:{}'.format(args.gpu)
    else:
        device = 'cpu'
    #----------------------seed_setting-----------------------
    set_seed(0)
    # -------------------load parameters----------------------
    dataset=args.dataset
    hidden_dim=args.dimension
    sample_size=args.sample_size
    moving_average_decay=args.moving_average_decay
    g, labels, split_lists = utils.read_real_datasets(dataset)
    g = g.to(device)
    feat = g.ndata['attr'].to(device)
    labels = labels.to(torch.int64)
    num_nodes = g.num_nodes()
    in_dim = feat.shape[1]
    #----------------initialize model--------------------------
    model = Model(in_dim, hidden_dim, 2, sample_size, tau=args.temp, 
    moving_average_decay=moving_average_decay, num_MLP=args.num_MLP).to(device)
    
    save_name = dataset + '.pth'
    print('load model parameters')
    model.load_state_dict(torch.load(save_name))

    #---------------------two-hop graph construction-----------------
    adj_sp = g.adjacency_matrix()
    adj_2_sp = adj_sp @ adj_sp + adj_sp
    adj_2 = adj_2_sp.to_dense()
    adj_2 = torch.where(adj_2 > 1, torch.tensor(1), adj_2)
    graph_2 = dgl.DGLGraph()
    graph_2.add_nodes(num_nodes)
    rows, cols = torch.nonzero(adj_2, as_tuple=True)
    edges = list(zip(rows.tolist(), cols.tolist()))
    src, dst = zip(*edges)
    graph_2.add_edges(src, dst)
    graph_2 = graph_2.to(device)
    #---------------------identity matrix construction----------------
    adj_id = torch.tensor(torch.eye(num_nodes))
    graph_id = dgl.DGLGraph()
    graph_id.add_nodes(num_nodes)
    rows, cols = torch.nonzero(adj_id, as_tuple=True)
    edges = list(zip(rows.tolist(), cols.tolist()))
    src, dst = zip(*edges)
    graph_id.add_edges(src, dst)
    graph_id = graph_id.to(device)

    # prompt-tuning
    embeds = model.get_embedding(g, feat)

    # structural prompt 
    # embeds = model.get_embedding(graph_2, feat)

    # attention semantic prompt 
    prompt_embeds = model.get_embedding(graph_id, feat)
    p = 0.9
    embeds = (1-p) * embeds + p * prompt_embeds 
    # w = F.cosine_similarity(embeds,prompt_embeds,dim=1).unsqueeze(-1)
    # w_ = torch.ones_like(w) - w
    # embeds = w * embeds + w_ * prompt_embeds 
    
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
    
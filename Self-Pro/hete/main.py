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
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epoch_num', type=int, default=1)
    parser.add_argument('--sample_size', type=int, default=5)
    parser.add_argument('--dimension', type=int, default=2048)
    parser.add_argument('--moving_average_decay', type=float, default=0.95)
    parser.add_argument('--tau', type=float, default=0.75)
    parser.add_argument('--wd2', type=float, default=1e-5)
    parser.add_argument('--num_MLP', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=1, help='GPU index.')
    args = parser.parse_args()
    if args.gpu != -1 and torch.cuda.is_available():
        device = 'cuda:{}'.format(args.gpu)
    else:
        device = 'cpu'
    #----------------------seed_setting-----------------------
    set_seed(0)
    # -------------------load parameters----------------------
    lr=args.lr
    dataset=args.dataset
    epoch_num=args.epoch_num
    hidden_dim=args.dimension
    sample_size=args.sample_size
    moving_average_decay=args.moving_average_decay
    if hidden_dim == None:
        hidden_dim = node_features.shape[1]
    else:
        hidden_dim = hidden_dim
    g, labels, split_lists = utils.read_real_datasets(dataset)
    g = g.to(device)
    node_features = g.ndata['attr'].to(device)
    node_labels = labels
    in_dim = node_features.shape[1]
    #----------------initialize model--------------------------
    model = Model(in_dim, hidden_dim, 2, sample_size, tau=args.tau, 
    moving_average_decay=moving_average_decay, num_MLP=args.num_MLP).to(device)
    opt = torch.optim.Adam([{'params': model.parameters()}], lr=lr, weight_decay=0.0003)
    #-------------------pre-training----------------------------
    node_embeddings, loss = train(g, node_features, model, opt, lr=lr, epoch=epoch_num, device=device,
                                  sample_size=sample_size, hidden_dim=hidden_dim,
                                  moving_average_decay=moving_average_decay)
    #-------------------evaluation------------------------------
    acc = []
    epochs = []
    for index in range(5):
        input_dims = node_embeddings.shape
        class_number = int(max(node_labels)) + 1
        FNN = LogReg(input_dims[1], class_number).to(device)
        FNN = FNN.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(FNN.parameters(), lr =1e-2, weight_decay=args.wd2)
        dataset = NodeClassificationDataset(node_embeddings, node_labels)
        split = utils.DataSplit(dataset, split_lists[index]['train_idx'], split_lists[index]['valid_idx'], split_lists[index]['test_idx'], shuffle=True)
        train_loader, val_loader, test_loader = split.get_split(batch_size=100000, num_workers=0)
        best = -float('inf')
        best_epoch = 0
        test_acc = 0
        for epoch in range(500):
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                y_pred = FNN(inputs)
                loss = criterion(y_pred, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for data in val_loader:
                        inputs, labels = data
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        outputs = FNN(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        loss = criterion(outputs, labels)
                        total += labels.size(0)
                        correct += torch.sum(predicted == labels)
                if correct / total > best:
                    best = correct / total
                    test_acc = evaluate(FNN, test_loader)
                    best_epoch = epoch
        print(test_acc)
        acc.append(test_acc)
        epochs.append(best_epoch)
    print("mean:")
    print(statistics.mean(acc))
    print("std:")
    print(statistics.stdev(acc))
    write_results(acc, epochs)

    # save parameters
    save_name = args.dataset + '.pth'
    torch.save(model.state_dict(), save_name)

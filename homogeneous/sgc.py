import os
import os.path as osp
import random
from time import perf_counter

import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import add_self_loops, to_scipy_sparse_matrix, degree
from tqdm import tqdm

from models import PyG_SGC, SGC

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = osp.join(osp.dirname(os.getcwd())) + '\data'
names = ['CiteSeer', 'Cora', 'PubMed']

dataset = Planetoid(root=path, name=names[0])
dataset = dataset[0]
edge_index, _ = add_self_loops(dataset.edge_index)
dataset = dataset.to(device)
num_in_feats, num_out_feats = dataset.num_node_features, torch.max(dataset.y).item() + 1
# get adj
adj = to_scipy_sparse_matrix(edge_index).todense()
adj = torch.tensor(adj).to(device)
# print(adj)
deg = degree(edge_index[0], dataset.num_nodes)
deg = torch.diag_embed(deg)
deg_inv_sqrt = torch.pow(deg, -0.5)
deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
deg_inv_sqrt = deg_inv_sqrt.to(device)
# normalization adj
s = torch.mm(torch.mm(deg_inv_sqrt, adj), deg_inv_sqrt)
# normalization feature
feature = dataset.x
k = 2
norm_x = torch.mm(torch.matrix_power(s, k), feature)
# train
y = dataset.y
train_mask = dataset.train_mask
val_mask = dataset.val_mask
test_mask = dataset.test_mask


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(42)


def train(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-6)
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.4)
    min_epochs = 10
    min_val_loss = 5
    final_best_acc = 0
    model.train()
    t = perf_counter()
    for epoch in tqdm(range(100)):
        out = model(norm_x)
        optimizer.zero_grad()
        loss = loss_function(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
        scheduler.step()
        # validation
        val_loss, test_acc = test(model)
        if val_loss < min_val_loss and epoch + 1 > min_epochs:
            min_val_loss = val_loss
            final_best_acc = test_acc
        model.train()
        print('Epoch{:3d} train_loss {:.5f} val_loss {:.3f} test_acc {:.3f}'.
              format(epoch, loss.item(), val_loss, test_acc))

    train_time = perf_counter() - t

    return final_best_acc, train_time


@torch.no_grad()
def test(model):
    model.eval()
    out = model(norm_x)
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    val_loss = loss_function(out[val_mask], y[val_mask])
    _, pred = out.max(dim=1)
    correct = int(pred[test_mask].eq(y[test_mask]).sum().item())
    test_acc = correct / int(test_mask.sum())

    return val_loss.cpu().item(), test_acc


def pyg_train(model, data):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-6)
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.4)
    min_epochs = 10
    min_val_loss = 5
    final_best_acc = 0
    model.train()
    t = perf_counter()
    for epoch in tqdm(range(100)):
        out = model(data)
        optimizer.zero_grad()
        loss = loss_function(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
        scheduler.step()
        # validation
        val_loss, test_acc = pyg_test(model, data)
        if val_loss < min_val_loss and epoch + 1 > min_epochs:
            min_val_loss = val_loss
            final_best_acc = test_acc
        model.train()
        tqdm.write('Epoch{:3d} train_loss {:.5f} val_loss {:.3f} test_acc {:.3f}'.
                   format(epoch, loss.item(), val_loss, test_acc))

    train_time = perf_counter() - t

    return final_best_acc, train_time


@torch.no_grad()
def pyg_test(model, data):
    model.eval()
    out = model(data)
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    val_loss = loss_function(out[val_mask], y[val_mask])
    _, pred = out.max(dim=1)
    correct = int(pred[test_mask].eq(y[test_mask]).sum().item())
    test_acc = correct / int(test_mask.sum())

    return val_loss.cpu().item(), test_acc


def main():
    model = SGC(num_in_feats, num_out_feats).to(device)
    best_acc, _train_time = train(model)

    # pyg
    model = PyG_SGC(num_in_feats, num_out_feats).to(device)
    pyg_best_acc, train_time = pyg_train(model, dataset)
    #
    print('---------------------------------')

    print('pytorch train_time:', _train_time)
    print('pytorch best test acc:', best_acc)

    print('pyg train_time:', train_time)
    print('pyg best test acc:', pyg_best_acc)


if __name__ == '__main__':
    main()

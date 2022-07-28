import copy
import os
import os.path as osp

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_scatter import scatter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = osp.join(osp.dirname(os.getcwd())) + '\data'

dataset = Planetoid(root=path, name='CiteSeer')
dataset = dataset[0]
dataset.edge_index, _ = add_self_loops(dataset.edge_index)
dataset = dataset.to(device)
num_in_feats, num_out_feats = dataset.num_node_features, torch.max(dataset.y).item() + 1


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.Tensor(out_channels))

    def message(self, x, edge_index):
        x = self.linear(x)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        x_j = x[col]
        x_j = norm.view(-1, 1) * x_j

        return x_j

    def aggregate(self, x_j, edge_index):
        row, col = edge_index
        # row(15758), x_j(15758, out_channels)
        out = scatter(x_j, row, dim=0, reduce='sum')
        return out

    def update(self, out):
        return out + self.bias

    def propagate(self, x, edge_index):
        out = self.message(x, edge_index)
        out = self.aggregate(out, edge_index)
        out = self.update(out)

        return out

    def forward(self, x, edge_index):
        return self.propagate(x, edge_index)


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 32)
        self.conv2 = GCNConv(32, num_classes)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = F.softmax(x, dim=1)

        return x


def get_val_loss(model):
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    model.eval()
    f = model(dataset)
    loss = loss_function(f[dataset.val_mask], dataset.y[dataset.val_mask])

    return loss.item()


def train(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    model.train()
    min_epochs = 10
    best_model = None
    min_val_loss = 5
    for epoch in range(200):
        out = model(dataset)
        optimizer.zero_grad()
        loss = loss_function(out[dataset.train_mask], dataset.y[dataset.train_mask])
        loss.backward()
        optimizer.step()
        # validation
        val_loss = get_val_loss(model)
        if epoch + 1 >= min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)
        print('Epoch: {:3d} train_Loss: {:.5f} val_loss: {:.5f}'.format(epoch, loss.item(), val_loss))
        model.train()

    return best_model


def test(model):
    model.eval()
    _, pred = model(dataset).max(dim=1)
    correct = int(pred[dataset.test_mask].eq(dataset.y[dataset.test_mask]).sum().item())
    acc = correct / int(dataset.test_mask.sum())
    print('GCN Accuracy: {:.4f}'.format(acc))


def main():
    model = GCN(num_in_feats, num_out_feats).to(device)
    best_model = train(model)
    test(best_model)


if __name__ == '__main__':
    main()
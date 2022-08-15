import copy
import os
import os.path as osp

import torch
from torch_geometric.datasets import Planetoid, NELL
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = osp.join(osp.dirname(os.getcwd())) + '\data'


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 32)
        self.conv2 = GCNConv(32, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = F.softmax(x, dim=1)

        return x


def load_data(name):
    if name == 'NELL':
        dataset = NELL(root=path + '/')
    else:
        dataset = Planetoid(root=path, name=name)

    data = dataset[0].to(device)
    if name == 'NELL':
        data.x = data.x.to_dense()
    return data, dataset.num_node_features, dataset.num_classes


def get_val_loss(model, data):
    model.eval()
    out = model(data)
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    loss = loss_function(out[data.val_mask], data.y[data.val_mask])
    model.train()
    return loss.item()


def train(model, data, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    min_val_loss = 5
    best_model = None
    min_epochs = 5
    model.train()
    for epoch in tqdm(range(200)):
        out = model(data)
        optimizer.zero_grad()
        loss = loss_function(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # validation
        val_loss = get_val_loss(model, data)
        if val_loss < min_val_loss and epoch + 1 > min_epochs:
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)
        tqdm.write('Epoch {:03d} train_loss {:.4f} val_loss {:.4f}'
                   .format(epoch, loss.item(), val_loss))

    return best_model


def test(model, data):
    model.eval()
    _, pred = model(data).max(dim=1)
    correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / int(data.test_mask.sum())
    print('GCN Accuracy: {:.4f}'.format(acc))


def main():
    names = ['CiteSeer', 'Cora', 'PubMed']
    for name in names:
        print(name + '...')
        data, num_node_features, num_classes = load_data(name)
        print(data)
        model = GCN(num_node_features, num_classes).to(device)
        model = train(model, data, device)
        test(model, data)


if __name__ == '__main__':
    main()

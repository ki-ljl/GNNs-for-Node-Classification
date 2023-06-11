import os

import torch
from torch import nn
from torch_geometric.datasets import DBLP
from torch_geometric.nn import HANConv
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = os.path.abspath(os.path.dirname(os.getcwd())) + '\data\DBLP'
dataset = DBLP(path)
graph = dataset[0]
num_classes = torch.max(graph['author'].y).item() + 1
graph['conference'].x = torch.ones((graph['conference'].num_nodes, 1))
graph = graph.to(device)
train_mask, val_mask, test_mask = graph['author'].train_mask, graph['author'].val_mask, graph['author'].test_mask
y = graph['author'].y


class HAN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(HAN, self).__init__()
        # H, D = self.heads, self.out_channels // self.heads
        self.conv1 = HANConv(in_channels, hidden_channels, graph.metadata(), heads=4)
        self.conv2 = HANConv(hidden_channels, out_channels, graph.metadata(), heads=4)

    def forward(self, data):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        x = self.conv1(x_dict, edge_index_dict)
        x = self.conv2(x, edge_index_dict)
        x = x['author']

        return x


def train():
    model = HAN(-1, 64, num_classes).to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    min_epochs = 5
    best_val_acc = 0
    final_best_acc = 0
    model.train()
    for epoch in tqdm(range(100)):
        f = model(graph)
        loss = loss_function(f[train_mask], y[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # validation
        val_acc, val_loss = test(model, val_mask)
        test_acc, test_loss = test(model, test_mask)
        if epoch + 1 > min_epochs and val_acc > best_val_acc:
            best_val_acc = val_acc
            final_best_acc = test_acc
        tqdm.write('Epoch{:3d} train_loss {:.5f} val_acc {:.3f} test_acc {:.3f}'
                   .format(epoch, loss.item(), val_acc, test_acc))

    return final_best_acc


def test(model, mask):
    model.eval()
    with torch.no_grad():
        out = model(graph)
        loss_function = torch.nn.CrossEntropyLoss().to(device)
        loss = loss_function(out[mask], y[mask])
    _, pred = out.max(dim=1)
    correct = int(pred[mask].eq(y[mask]).sum().item())
    acc = correct / int(test_mask.sum())

    return acc, loss.item()


def main():
    final_best_acc = train()
    print('HAN Accuracy:', final_best_acc)


if __name__ == '__main__':
    main()

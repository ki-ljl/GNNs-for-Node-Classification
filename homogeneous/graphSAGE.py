import copy
import os
import os.path as osp

import torch
from torch.optim.lr_scheduler import StepLR
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import SAGEConv
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = osp.join(osp.dirname(os.getcwd())) + '\data'


def load_data():
    dataset = Planetoid(root=path, name='CiteSeer')
    dataset = dataset[0]
    dataset.edge_index, _ = add_self_loops(dataset.edge_index)
    dataset = dataset.to(device)
    num_in_feats, num_out_feats = dataset.num_node_features, torch.max(dataset.y).item() + 1

    return dataset, num_in_feats, num_out_feats


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, normalize=True)
        self.conv2 = SAGEConv(h_feats, out_feats, normalize=True)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)

        return x


def get_val_loss(model, data):
    model.eval()
    out = model(data)
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    loss = loss_function(out[data.val_mask], data.y[data.val_mask])
    model.train()
    return loss.item()


def train(model, data):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    min_epochs = 10
    min_val_loss = 5
    best_model = None
    model.train()
    for epoch in tqdm(range(200)):
        out = model(data)
        optimizer.zero_grad()
        loss = loss_function(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        scheduler.step()
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
    print('GraphSAGE Accuracy: {:.4f}'.format(acc))


def main():
    dataset, num_in_feats, num_out_feats = load_data()
    model = GraphSAGE(num_in_feats, 64, num_out_feats).to(device)
    # print(model)
    model = train(model, dataset)
    test(model, dataset)


if __name__ == '__main__':
    main()
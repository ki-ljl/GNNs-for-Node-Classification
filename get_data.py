# -*- coding:utf-8 -*-
from torch_geometric.datasets import NELL, Planetoid

from util import device


def load_data(path, name):
    if name == 'NELL':
        dataset = NELL(root=path + '/NELL')
    else:
        dataset = Planetoid(root=path, name=name)

    data = dataset[0].to(device)
    if name == 'NELL':
        data.x = data.x.to_dense()

    return data, dataset.num_node_features, dataset.num_classes




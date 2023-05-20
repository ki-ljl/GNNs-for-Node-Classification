# -*- coding:utf-8 -*-
import copy

import numpy as np
import torch
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@torch.no_grad()
def test(model, data):
    model.eval()
    out = model(data)
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    loss = loss_function(out[data.val_mask], data.y[data.val_mask])
    _, pred = out.max(dim=1)
    correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / int(data.test_mask.sum())
    model.train()

    return loss.item(), acc


def train(model, data):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    min_val_loss = np.Inf
    best_model = None
    min_epochs = 5
    model.train()
    final_test_acc = 0
    for epoch in tqdm(range(200)):
        out = model(data)
        optimizer.zero_grad()
        loss = loss_function(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # validation
        val_loss, test_acc = test(model, data)
        if val_loss < min_val_loss and epoch + 1 > min_epochs:
            min_val_loss = val_loss
            final_test_acc = test_acc
            best_model = copy.deepcopy(model)
        tqdm.write('Epoch {:03d} train_loss {:.4f} val_loss {:.4f} test_acc {:.4f}'
                   .format(epoch, loss.item(), val_loss, test_acc))

    return best_model, final_test_acc

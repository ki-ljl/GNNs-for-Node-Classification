import os

from get_data import load_data
from models import GCN
from util import device, train

path = os.path.abspath(os.path.dirname(os.getcwd())) + "/data"


def main():
    # name: CiteSeer Cora NELL PubMed
    dataset, num_in_feats, num_out_feats = load_data(path, name='Cora')
    model = GCN(num_in_feats, 64, num_out_feats).to(device)
    model, test_acc = train(model, dataset)
    print('test acc:', test_acc)


if __name__ == '__main__':
    main()

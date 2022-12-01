'''
note:
    1. debatch 将batch进行拆分 √
    2. filter 对nan值进行过滤 ×
'''

import pandas as pd
import numpy as np
import torch
from graduate_design.graph_dataset import GNNDataset
from torch_geometric.data import DataLoader


def debatch(graph_dataset):
    batch = graph_dataset.batch.tolist()
    x = graph_dataset.x.tolist()
    df = pd.DataFrame({'batch': batch, 'x': x})

    batched_data = []
    for graph_id, features in df.groupby('batch'):
        data = []
        a = features['x'].tolist()
        data.append(a)
        batched_data.append(torch.tensor(np.array(data)))
    return batched_data


# TODO 调试！！！
def filter(drug_emb, protein_embs, drug_emb_pos, binding_affinity):
    drug_interacts = binding_affinity[drug_emb_pos]
    pos = np.where(np.isnan(drug_interacts) == False)
    interact_proteins = protein_embs[pos]

    # todo dim参数需要调整！！！
    return torch.cat((drug_emb, interact_proteins), dim=1)


def main():
    davis_path = './data/Davis/'
    davis_trainset = GNNDataset(davis_path, train=True)
    train_loader = DataLoader(davis_trainset, batch_size=54)

    for batch in train_loader:
        print(batch)
        transformed_data = debatch(batch)
        print(transformed_data)


if __name__ == '__main__':
    main()

'''
note:
    1. GNN自定义数据集 √
    2. 相似度和亲和力值数据 √
todo:
    1. protein数据是否处理未确定！！！
'''

import os
import pickle

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric import data as DATA
from torch_geometric.data import DataLoader
from graduate_design.data_preprogressing import SMILES_to_Graph, split_data


class GNNDataset(InMemoryDataset):
    def __init__(self, root, train=True, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.root = root
        if train:
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            self.data, self.slices = torch.load(self.processed_paths[1])

    @property
    def raw_file_names(self):
        return ['data_train.csv', 'data_test.csv']

    @property
    def processed_file_names(self):
        return ['processed_data_train.pt', 'processed_data_test.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def process_data(self, dataset):
        data_list = []
        for smi in dataset:
            x, edge_index = SMILES_to_Graph(smi)
            try:
                data = DATA.Data(
                    x=x,
                    edge_index=edge_index
                )
            except:
                print("unable to process: ", smi)

            data_list.append(data)
        return data_list

    def process(self):

        trainset, testset = split_data(self.root)

        train_list = self.process_data(trainset)
        test_list = self.process_data(testset)

        if self.pre_filter is not None:
            train_list = [train for train in train_list if self.pre_filter(train)]
            test_list = [test for test in test_list if self.pre_filter(test)]

        if self.pre_transform is not None:
            train_list = [self.pre_transform(train) for train in train_list]
            test_list = [self.pre_transform(test) for test in test_list]

        print('Graph construction done. Saving to file.')

        data, slices = self.collate(train_list)
        # save preprocessed train data:
        torch.save((data, slices), self.processed_paths[0])

        data, slices = self.collate(test_list)
        # save preprocessed test data:
        torch.save((data, slices), self.processed_paths[1])


class sim_BindingAffinity_matrix(object):
    def __init__(self, datasetname, root='./data'):
        f_read = open(os.path.join(root, datasetname, 'splited_sim_BA_matrix.pkl'), 'rb')
        dict2 = pickle.load(f_read)
        self.__dict__ = dict2
        f_read.close()


def main():
    davis_path = './data/Davis/'
    kiba_path = './data/KIBA/'
    # GNNDataset(davis_path)
    # GNNDataset(kiba_path)

    davis_sim_BA = sim_BindingAffinity_matrix('Davis')
    kiba_sim_BA = sim_BindingAffinity_matrix('KIBA')

    # davis_trainset = GNNDataset(davis_path, train=True)
    # davis_testset = GNNDataset(davis_path, train=False)
    #
    # train_loader = DataLoader(davis_trainset, batch_size=54)
    #
    # for batch in train_loader:
    #     print(batch)


if __name__ == '__main__':
    main()

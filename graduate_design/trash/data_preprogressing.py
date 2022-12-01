import csv
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from graduate_design.utils import SMILES_to_Graph


class objectview(object):
    def __init__(self, d):
        self.__dict__ = d


class DT_Info(object):
    def __init__(self, drug_num, drug, protein_num, protein, drug_sim, protein_sim):
        self.drug = drug
        self.protein = protein
        self.drug_sim = drug_sim
        self.protein_sim = protein_sim
        self.drug_num = drug_num
        self.protein_num = protein_num

        #  attributes of the nodes in the drug structure graph
        self.atom_attributes = None


class DTADataset(Dataset):
    def __init__(self, filepath):
        self.DT, self.Affinity = self.get_data(filepath)

    def load_data(self, path, sim):
        data = {}
        with open(path, encoding='utf-8') as file:
            i = 1
            for row in csv.reader(file, skipinitialspace=True):
                data[i - 1] = {'seq': row[1], 'sim': sim[i - 1]}
                i += 1
        return data

    def get_data(self, filepath):

        binding_affinity = np.loadtxt(open(filepath + "binding_affinity.csv", "rb"), delimiter=",", skiprows=0)

        drug_sim = np.loadtxt(open(filepath + "drug_sim.csv", "rb"), delimiter=",", skiprows=0)
        protein_sim = np.loadtxt(open(filepath + "target_sim.csv", "rb"), delimiter=",", skiprows=0)

        drug_dict = self.load_data(filepath + "drug.csv", drug_sim)
        protein_dict = self.load_data(filepath + "protein.csv", protein_sim)

        DT_info_list = list()
        for i in range(binding_affinity.shape[0]):  # number of drugs
            drug_num = i
            drug = drug_dict[i]['seq']
            drug_sim = drug_dict[i]['sim']
            for j in range(binding_affinity.shape[1]):  # number of proteins
                protein_num = j
                protein = protein_dict[j]['seq']
                protein_sim = protein_dict[j]['sim']
                dt_info = DT_Info(drug_num, drug, protein_num, protein, drug_sim, protein_sim)
                DT_info_list.append(dt_info)

        binding_affinity = binding_affinity.flatten()
        return DT_info_list, binding_affinity

    def __getitem__(self, item):
        return self.DT[item], self.Affinity[item]

    def __len__(self):
        return len(self.DT)


def transform_data(dataset):
    d_isexit = [0] * len(dataset[0].drug_sim)  # zero means not exit, one means exit in the dataset
    p_isexit = [0] * len(dataset[0].protein_sim)

    for data in dataset:
        p_isexit[data.protein_num] = 1 if p_isexit[data.protein_num] == 0 else p_isexit[data.protein_num]
        d_isexit[data.drug_num] = 1 if d_isexit[data.drug_num] == 0 else d_isexit[data.drug_num]

    for data in dataset:
        data.drug_sim *= d_isexit
        data.protein_sim *= p_isexit
        data.atom_attributes = objectview(SMILES_to_Graph(data.drug))


def main():  # only for test and debug
    davis_path = './data/Davis/'
    kiba_path = './data/KIBA/'

    davis_dataset = DTADataset(davis_path)
    kiba_dataset = DTADataset(kiba_path)

    train_X, test_X, train_y, test_y = train_test_split(davis_dataset.DT, davis_dataset.Affinity, test_size=0.001,
                                                        random_state=5)
    transform_data(test_X)
    print(davis_dataset[0])


if __name__ == '__main__':
    main()

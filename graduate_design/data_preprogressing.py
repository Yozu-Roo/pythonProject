'''
note:
    1. drug数据预处理部分 √
    2. 相似度和亲和力值数据预处理部分 √
'''

import os
import pickle
import pandas as pd
import networkx as nx
import torch
from rdkit import Chem
from sklearn import preprocessing
import numpy as np
from graduate_design.graph_dataset import sim_BindingAffinity_matrix

ATOM_ELEMENT = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                'Pt', 'Hg', 'Pb', 'X']  # the kinds of atoms

ATOM_ELEMENT_ONEHOT_MAP = dict()  # map the atom element to its onehot encoding
ATOM_ELEVEN_VECTOR_ONEHOT_MAP = dict()  # map the feature that dimension is eleven to its onehot encoding


def onehot_encoder(symbol):
    onehot = np.array(symbol).reshape([-1, 1])
    Encoder = preprocessing.OneHotEncoder()
    Encoder.fit(onehot)
    onehot = Encoder.transform(onehot).toarray()
    onehot = np.asarray(onehot, dtype=np.int32)
    return onehot.tolist()


def atom_onehot_encoder():
    global ATOM_ELEMENT_ONEHOT_MAP
    global ATOM_ELEVEN_VECTOR_ONEHOT_MAP
    ATOM_ELEMENT_ONEHOT_MAP = dict(zip(ATOM_ELEMENT, onehot_encoder(ATOM_ELEMENT)))
    ATOM_ELEVEN_VECTOR_ONEHOT_MAP = dict(zip(np.arange(11), onehot_encoder(np.arange(11))))


def atom_features(atom):
    atom_onehot_encoder()
    return np.array(ATOM_ELEMENT_ONEHOT_MAP[atom.GetSymbol()] +
                    ATOM_ELEVEN_VECTOR_ONEHOT_MAP[atom.GetDegree()] +
                    ATOM_ELEVEN_VECTOR_ONEHOT_MAP[atom.GetTotalNumHs()] +
                    ATOM_ELEVEN_VECTOR_ONEHOT_MAP[atom.GetImplicitValence()] +
                    [atom.GetIsAromatic()])


def SMILES_to_Graph(smi):
    mol = Chem.MolFromSmiles(smi)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

    g = nx.Graph(edges).to_directed()
    edge_index = []
    mol_adj = np.zeros((c_size, c_size))
    for e1, e2 in g.edges:
        mol_adj[e1, e2] = 1
        mol_adj[e2, e1] = 1
        # edge_index.append([e1, e2])
    mol_adj += np.matrix(np.eye(mol_adj.shape[0]))
    index_row, index_col = np.where(mol_adj >= 0.5)
    for i, j in zip(index_row, index_col):
        edge_index.append([i, j])

    features = torch.Tensor(features).type(torch.float32)
    edge_index = torch.Tensor([np.array(edge_index)[:, 0], np.array(edge_index)[:, 1]]).type(torch.int64)
    return features, edge_index


def split_data(path, split_ratio=[8, 2], kinds='drug.csv'):
    path = os.path.join(path, kinds)
    dataset = pd.read_csv(path, header=None, delimiter=",")

    ratio = len(dataset) / sum(split_ratio)
    train_size = int(ratio * split_ratio[0])

    trainset, testset = dataset.loc[0:train_size - 1, 1].tolist(), dataset.loc[train_size:, 1].tolist()

    return trainset, testset


def split_matrix(matrix, row_bound, col_bound):
    left_top = matrix[0:row_bound, 0:col_bound]
    right_top = matrix[0:row_bound, col_bound:]
    left_bottom = matrix[row_bound:, 0:col_bound]
    right_bottom = matrix[row_bound:, col_bound:]
    return left_top, right_top, left_bottom, right_bottom


def split_simlarity_BindingAffinity(filepath, split_ratio=[8, 2]):
    binding_affinity = np.loadtxt(open(os.path.join(filepath, "binding_affinity.csv"), "rb"), delimiter=",", skiprows=0)
    drug_sim = np.loadtxt(open(os.path.join(filepath, "drug_sim.csv"), "rb"), delimiter=",", skiprows=0)
    protein_sim = np.loadtxt(open(os.path.join(filepath, "target_sim.csv"), "rb"), delimiter=",", skiprows=0)

    drug_size, protein_size = binding_affinity.shape[0], binding_affinity.shape[1]

    drug_ratio = drug_size / sum(split_ratio)
    protein_ratio = protein_size / sum(split_ratio)

    drug_train_size, protein_train_size = int(drug_ratio * split_ratio[0]), int(protein_ratio * split_ratio[0])

    splited_sim_BA_matrix = {'binding_affinity': binding_affinity, 'drug_sim': drug_sim, 'protein_sim': protein_sim}
    for key, matrix in splited_sim_BA_matrix.items():
        splited_set = dict()
        row_bound = drug_train_size if key == 'binding_affinity' or key == 'drug_sim' else protein_train_size
        col_bound = protein_train_size if key == 'binding_affinity' or key == 'protein_sim' else drug_train_size

        splited_set['train_set'], \
        splited_set['valid_set1'], \
        splited_set['valid_set2'], \
        splited_set['test_set'] = split_matrix(matrix, row_bound, col_bound)

        splited_sim_BA_matrix[key] = splited_set

    # dict save as pickle file
    f_save = open(filepath + 'splited_sim_BA_matrix.pkl', 'wb')
    pickle.dump(splited_sim_BA_matrix, f_save)
    f_save.close()

    return splited_sim_BA_matrix


def main():  # only for test
    davis_sim_BA = sim_BindingAffinity_matrix('Davis')
    kiba_sim_BA = sim_BindingAffinity_matrix('KIBA')


if __name__ == '__main__':
    main()

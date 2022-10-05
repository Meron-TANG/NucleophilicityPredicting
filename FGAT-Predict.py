from dgllife.model.gnn import AttentiveFPGNN
from dgllife.model.readout import AttentiveFPReadout
import pandas as pd
import dgl
from dgllife.data import MoleculeCSVDataset
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from xgboost import XGBRegressor
from sklearn.svm import SVR
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
import warnings
warnings.filterwarnings('ignore')

from sklearn.utils import shuffle
# GAT
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from dgllife.model import model_zoo
from dgllife.utils import smiles_to_bigraph
from dgllife.utils import EarlyStopping, Meter
from dgllife.utils import AttentiveFPAtomFeaturizer
from dgllife.utils import AttentiveFPBondFeaturizer
import random
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from numpy import *
from matplotlib.offsetbox import AnchoredText

from dgllife.model.gnn import AttentiveFPGNN
from dgllife.model.readout import AttentiveFPReadout

from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
import rdkit

df = pd.read_csv('D:/Dataset/亲核test.csv',index_col = 0)
df = df[['SMILES','SOL','N']]
df['com'] = df.SMILES+'.'+df.SOL

def load_data(data, name, load, task_name, smiles_column):
    dataset = MoleculeCSVDataset(data,
                                 smiles_to_graph=smiles_to_bigraph,
                                 node_featurizer=atom_featurizer,
                                 edge_featurizer=bond_featurizer,
                                 smiles_column=smiles_column,
                                 cache_file_path=str(name) + '_dataset.bin',
                                 task_names=[task_name],
                                 load=load, init_mask=True, n_jobs=20
                                 )
    return dataset

atom_featurizer = AttentiveFPAtomFeaturizer(atom_data_field='hv')
bond_featurizer = AttentiveFPBondFeaturizer(bond_data_field='he')
n_feats = atom_featurizer.feat_size('hv')
e_feats = bond_featurizer.feat_size('he')

val_datasets = load_data(df, 'test_N', True, 'N', 'com')

def collate_emodel(data):

    assert len(data[0]) in [3, 4], \
        'Expect the tuple to be of length 3 or 4, got {:d}'.format(len(data[0]))
    if len(data[0]) == 3:
        smiles, graphs, labels = map(list, zip(*data))
        masks = None
    else:
        smiles, graphs, labels, masks = map(list, zip(*data))

    self_feats = np.empty((len(data), self_feats_dim), dtype=np.float32)

    def get_bit(smile):

        mol = Chem.MolFromSmiles(smile)

        fp1 = MACCSkeys.GenMACCSKeys(mol)
        a = np.array(fp1, int).reshape(1, -1)  # len = 167

        return a

    for i in range(len(smiles)):
        self_feats[i, :] = get_bit(smiles[i])
    # 对moe208 数据归一化
    # self_feats =  self_feats - self_feats.mean() / self_feats.std

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    if masks is None:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)

    return smiles, bg, labels, masks, torch.tensor(self_feats).to(device)

vali_loader = DataLoader(val_datasets, batch_size=1, shuffle=False,collate_fn=collate_emodel)
self_feats_dim = 167 #+ 15#208  # 添加的分子指纹长度
linear_feats_len = 167 #+ 15#208  # 经过mlp之后的分子指纹长度
device = 'cpu'
a = vali_loader


class FPSAT(nn.Module):
    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 self_feats_dim,
                 linear_feats_len,
                 n_tasks,
                 dropout,
                 num_layers,
                 num_timesteps,
                 graph_feat_size
                 ):
        super(FPSAT, self).__init__()

        self.gnn = AttentiveFPGNN(
            node_feat_size=node_feat_size,
            edge_feat_size=edge_feat_size,
            graph_feat_size=graph_feat_size,
            num_layers=num_layers,
            dropout=dropout
        )
        self.readout = AttentiveFPReadout(
            feat_size=graph_feat_size,
            dropout=dropout,
            num_timesteps=num_timesteps

        )

        self.fp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self_feats_dim, linear_feats_len),
        )

        self.predict = nn.Sequential(
            # nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(graph_feat_size + linear_feats_len, n_tasks))

    def forward(self, graph, node_feats, edge_feats, self_feats, get_node_weight=None):
        node_feats = self.gnn(graph, node_feats, edge_feats)
        graph_feats = self.readout(graph, node_feats, get_node_weight)

        new_feats = torch.cat((graph_feats, self_feats), dim=1)

        pred = self.predict(new_feats)

        return pred


def FGAT_model(self_feats_dim, linear_feats_len):
    # self_feats_dim = 167+208 # 添加的分子指纹长度
    # linear_feats_len = 167+208 # 经过mlp之后的分子指纹长度

    model = FPSAT(node_feat_size=n_feats,
                  edge_feat_size=e_feats,
                  linear_feats_len=linear_feats_len,
                  self_feats_dim=self_feats_dim,
                  n_tasks=1,
                  dropout=0.5,
                  num_layers=2,
                  num_timesteps=1,
                  graph_feat_size=300).to(device)
    return model


model = FGAT_model(167, 167)


fna = r'D:\课题相关\code\My_code_N\result folder\macc_relu_kfold_r2_wholeseed_33\macc_relu_kfold10_result_r2_wholeseed_33.pth'

model.load_state_dict(torch.load(fna,map_location=torch.device('cpu'))['model_state_dict'])

model.eval()
preds = []

with torch.no_grad():
    for batch_id, batch_data in enumerate(vali_loader):
        smiles, bg, labels, masks, self_feats = batch_data
        bg = bg.to(device)
        labels = labels.to(device)
        masks = masks.to(device)

        n_feats = bg.ndata.pop('hv').to(device)
        e_feats = bg.edata.pop('he').to(device)
        vali_prediction = model(bg, n_feats, e_feats, self_feats)
        preds.append(vali_prediction)


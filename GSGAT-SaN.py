from dgllife.model.gnn import AttentiveFPGNN
from dgllife.model.readout import AttentiveFPReadout
import pandas as pd
import dgl
from dgllife.data import MoleculeCSVDataset

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from mlxtend.regressor import StackingCVRegressor
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
from numpy import mean
from net import FGAT_SaN

# class FPSAT(nn.Module):
#     def __init__(self,
#                  node_feat_size,
#                  edge_feat_size,
#                  self_feats_dim,
#                  linear_feats_len,
#                  n_tasks,
#                  dropout,
#                  num_layers,
#                  num_timesteps,
#                  graph_feat_size
#                  ):
#         super(FPSAT, self).__init__()
#
#         self.gnn = AttentiveFPGNN(
#             node_feat_size=node_feat_size,
#             edge_feat_size=edge_feat_size,
#             graph_feat_size=graph_feat_size,
#             num_layers=num_layers,
#             dropout=dropout
#         )
#         self.readout = AttentiveFPReadout(
#             feat_size=graph_feat_size,
#             dropout=dropout,
#             num_timesteps=num_timesteps
#
#         )
#
#         self.fp = nn.Sequential(
#             nn.Dropout(dropout),
#             nn.Linear(self_feats_dim, linear_feats_len),
#         )
#
#         self.predict = nn.Sequential(
#             nn.ReLU(),
#             nn.Linear(2 * (graph_feat_size + linear_feats_len), n_tasks))
#
#     def forward(self, graph1, graph2, node_feats, edge_feats, node_feats2, edge_feats2, self_feats1, self_feats2,
#                 get_node_weight=None):
#         node_feats1 = self.gnn(graph1, node_feats, edge_feats)
#         graph_feats1 = self.readout(graph1, node_feats1, get_node_weight)
#         new_feats1 = torch.cat((graph_feats1, self_feats1), dim=1)
#
#         node_feats2 = self.gnn(graph2, node_feats2, edge_feats2)
#         graph_feats2 = self.readout(graph2, node_feats2, get_node_weight)
#         new_feats2 = torch.cat((graph_feats2, self_feats2), dim=1)
#
#         new_feats = torch.cat((new_feats1, new_feats2), dim=1)
#
#         pred = self.predict(new_feats)
#
#         return pred


from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
import rdkit

descriptor_names = [descriptor_name[0] for descriptor_name in Descriptors._descList]
descriptor_calculation = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)


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
        a = np.array(fp1, int).reshape(1, -1)

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

    return smiles, bg, labels, torch.tensor(masks).to(device), torch.tensor(self_feats).to(device)


def load_data(data, name, load, task_name, smiles_column):
    dataset = MoleculeCSVDataset(data,
                                 smiles_to_graph=smiles_to_bigraph,
                                 node_featurizer=atom_featurizer,
                                 edge_featurizer=bond_featurizer,
                                 smiles_column=smiles_column,
                                 cache_file_path=str(name) + '_dataset.bin',
                                 task_names=[task_name],
                                 load=load, init_mask=True
                                 )
    return dataset


def get_k_fold_data_limu(k, i, X, y):
    assert k >= 1
    fold_size = X.shape[0] // k
    X_train = None
    y_train = None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_part)], 0)
            y_train = pd.concat([pd.Series(y_train), pd.Series(y_part)], 0)

    return X_train, y_train, X_valid, y_valid


def run_a_train_epoch(n_epochs, epoch, model, data_loader, data_loader_sol, loss_criterion, optimizer):
    model.train()
    losses = []
    train_meter = Meter()
    for batch_data, batch_data_sol in zip(data_loader, data_loader_sol):
        batch_data
        smiles1, bg1, labels1, masks, self_feats1 = batch_data
        smiles2, bg2, labels2, masks1, self_feats2 = batch_data_sol

        bg1 = bg1.to(device)
        labels1 = labels1.to(device)
        masks1 = masks1.to(device)
        bg2 = bg2.to(device)

        n_feats = bg1.ndata.pop('hv').to(device)
        e_feats = bg1.edata.pop('he').to(device)

        n_feats2 = bg2.ndata.pop('hv').to(device)
        e_feats2 = bg2.edata.pop('he').to(device)

        prediction = model(bg1, bg2, n_feats, e_feats, n_feats2, e_feats2, self_feats1, self_feats2)

        loss = (loss_criterion(prediction, labels1) * (masks != 0).float()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_meter.update(prediction, labels1, masks)
        losses.append(loss.data.item())

    total_r2 = np.mean(train_meter.compute_metric('r2'))
    total_loss = np.mean(losses)
    if epoch % 5 == 0:
        print('epoch {:d}/{:d}, training_r2 {:.4f}, training_loss {:.4f}'.format(epoch + 1, n_epochs, total_r2,
                                                                                 total_loss))
    return total_r2, total_loss


def run_an_eval_epoch(n_epochs, model, data_loader, data_loader_sol, loss_criterion):
    model.eval()
    val_losses = []
    eval_meter = Meter()
    preds = None
    targets = None
    with torch.no_grad():
        for batch_data, batch_data_sol in zip(data_loader, data_loader_sol):

            smiles1, bg1, labels1, masks, self_feats1 = batch_data
            smiles2, bg2, labels2, masks1, self_feats2 = batch_data_sol

            bg1 = bg1.to(device)
            labels1 = labels1.to(device)
            masks1 = masks.to(device)
            bg2 = bg2.to(device)

            n_feats = bg1.ndata.pop('hv').to(device)
            e_feats = bg1.edata.pop('he').to(device)
            n_feats2 = bg2.ndata.pop('hv').to(device)
            e_feats2 = bg2.edata.pop('he').to(device)

            vali_prediction = model(bg1, bg2, n_feats, e_feats, n_feats2, e_feats2, self_feats1, self_feats2)
            val_loss = (loss_criterion(vali_prediction, labels1) * (masks != 0).float()).mean()
            val_loss = val_loss.detach().cpu().numpy()
            val_losses.append(val_loss)
            eval_meter.update(vali_prediction, labels1, masks)
            if preds is None:
                preds = vali_prediction.clone().detach()
                targets = labels1.clone().detach()
            else:
                preds = torch.cat((preds, vali_prediction), dim=0)
                targets = torch.cat((targets, labels1), dim=0)

        total_score = np.mean(eval_meter.compute_metric('rmse'))
        total_loss = np.mean(val_losses)

    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()

    np.savetxt('singal_result.csv', np.concatenate((targets, preds), axis=1), delimiter=',')
    return total_score, total_loss


# seed = 79 33
seed = 33
random.seed(seed)
np.random.seed(seed)
# os.environ['PYTHONHASHSEED'] = str(seed)

torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)

df = pd.read_csv('D:/Dataset/统一smile亲核试剂1072版本.csv', index_col=0)
df = shuffle(df, random_state=seed)
df = df[['homo_smiles', 'homo_sol_smiles', 'N']]
# df['com'] = df.homo_smiles+'.'+df.homo_sol_smiles
# df = df.dropna()
# df.isnull().sum()
# dfn = df[['com','N']]
# dfn.columns = ['SMILES','VALUES']
# n_train = dfn.iloc[:750,:]
# n_val = dfn.iloc[750:,:]

atom_featurizer = AttentiveFPAtomFeaturizer(atom_data_field='hv')
bond_featurizer = AttentiveFPBondFeaturizer(bond_data_field='he')
n_feats = atom_featurizer.feat_size('hv')
e_feats = bond_featurizer.feat_size('he')

task_name = 'N'
smiles_column_chem = 'homo_smiles'
smiles_column_sol = 'homo_sol_smiles'

OUTCOME = []
RMSE = []
fold_num = 10

for i in range(fold_num):
    outcome = []
    X_train, y_train, X_valid, y_valid = get_k_fold_data_limu(fold_num, i,
                                                              df[[smiles_column_chem, smiles_column_sol]].values,
                                                              df[task_name].values)
    train = pd.concat([X_train, y_train], 1)
    test = pd.concat([pd.DataFrame(X_valid), pd.Series(y_valid)], 1)
    train.columns = ['homo_smiles', 'homo_sol_smiles', 'N']
    test.columns = ['homo_smiles', 'homo_sol_smiles', 'N']

    train_datasets_chem = load_data(train, 'train_{}'.format(i), True, task_name, smiles_column_chem)
    val_datasets_chem = load_data(test, 'test_{}'.format(i), True, task_name, smiles_column_chem)

    train_datasets_sol = load_data(train, 'train_sol_{}'.format(i), True, task_name, 'homo_sol_smiles')
    val_datasets_sol = load_data(test, 'test_sol_{}'.format(i), True, task_name, 'homo_sol_smiles')

    loader_batch_size = 32

    train_loader_chem = DataLoader(train_datasets_chem, batch_size=loader_batch_size, shuffle=False,
                                   collate_fn=collate_emodel)
    vali_loader_chem = DataLoader(val_datasets_chem, batch_size=loader_batch_size, shuffle=False,
                                  collate_fn=collate_emodel)

    train_loader_sol = DataLoader(train_datasets_sol, batch_size=loader_batch_size, shuffle=False,
                                  collate_fn=collate_emodel)
    vali_loader_sol = DataLoader(val_datasets_sol, batch_size=loader_batch_size, shuffle=False,
                                 collate_fn=collate_emodel)

    from dgllife.model.gnn import AttentiveFPGNN
    from dgllife.model.readout import AttentiveFPReadout

    device = 'cpu'
    self_feats_dim = 167  # 添加的分子指纹长度
    linear_feats_len = 167  # 经过mlp之后的分子指纹长度

    model = FGAT_SaN(node_feat_size=n_feats,
                  edge_feat_size=e_feats,
                  linear_feats_len=linear_feats_len,
                  self_feats_dim=self_feats_dim,
                  n_tasks=1,
                  dropout=0.5,
                  num_layers=2,
                  num_timesteps=1,
                  graph_feat_size=300).to(device)

    loss_fn = nn.MSELoss(reduction='none')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.000001)

    stopper = EarlyStopping(mode='lower', patience=50)

    n_epochs = 1000
    rmse = None
    Losses = []
    train_r2 = []
    test_r2 = []
    Loss_test = []
    epoch = []
    for e in range(n_epochs):
        epoch.append(e + 1)
        score = run_a_train_epoch(n_epochs, e, model, train_loader_chem, train_loader_sol, loss_fn, optimizer)
        Losses.append(score[1])
        train_r2.append(score[0])

        val_score = run_an_eval_epoch(n_epochs, model, vali_loader_chem, vali_loader_sol, loss_fn)

        test_r2.append(val_score[0])
        Loss_test.append(val_score[-1])
        early_stop = stopper.step(val_score[0], model)
        if e % 5 == 0:
            print('epoch {:d}/{:d}, validation {} {:.4f}, validation {} {:.4f}, best validation {} {:.4f}'.format(
                e + 1, n_epochs, 'r2', val_score[0], 'loss', val_score[-1],
                'r2', stopper.best_score))
            if not rmse:
                rmse = val_score[-1]
            if val_score[-1] < rmse:
                rmes = val_score[-1]
        if early_stop:
            # outcome.append(stopper.best_score)
            # outcome.append(train)
            # outcome.append(test)
            # outcome.append(model)
            OUTCOME.append(stopper.best_score)
            RMSE.append(rmse)
            break

    print('best R2:')
    print(OUTCOME)
    print('best mse:')
    print(RMSE)

from numpy import *

print(mean(OUTCOME), sqrt(mean(RMSE)))

import matplotlib.pyplot as plt
import seaborn as sns

plt.plot(epoch,train_r2)
plt.plot(epoch,test_r2)
plt.plot(epoch,np.sqrt(Losses))
plt.plot(epoch,np.sqrt(Loss_test))
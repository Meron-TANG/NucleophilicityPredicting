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
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
# from xgboost import XGBRegressor
# from mlxtend.regressor import StackingCVRegressor
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
            nn.ReLU(),
            nn.Linear(graph_feat_size + linear_feats_len, n_tasks))

    def forward(self, graph, node_feats, edge_feats, self_feats, get_node_weight=None):
        node_feats = self.gnn(graph, node_feats, edge_feats)
        graph_feats = self.readout(graph, node_feats, get_node_weight)

        new_feats = torch.cat((graph_feats, self_feats), dim=1)

        pred = self.predict(new_feats)

        return pred


from sklearn.preprocessing import MinMaxScaler

from rdkit import Chem


def collate_emodel(data):

    descriptor_names = ['MolWt', 'RingCount']

    descriptor_calculation = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

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
        moe = np.array(descriptor_calculation.CalcDescriptors(mol)).reshape(1, -1)  # len = 49
        num_atom = np.array(mol.GetNumAtoms()).reshape(1, -1)
        moe = np.nan_to_num(moe)
        EGCN = np.concatenate([moe, num_atom], 1)

        scaler = MinMaxScaler()
        scaler.fit(EGCN)
        desc = scaler.fit_transform(EGCN)

        return desc

    for i in range(len(smiles)):
        self_feats[i, :] = get_bit(smiles[i])
    # ???moe208 ???????????????
    # self_feats =  self_feats - self_feats.mean() / self_feats.std

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    if masks is None:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)

    return smiles, bg, labels, masks, torch.tensor(self_feats)


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


# estimate_type = r2 rmse mae


def run_a_train_epoch(n_epochs, epoch, model, data_loader, loss_criterion, optimizer, estimate_type):
    model.train()
    losses = []
    train_meter = Meter()
    for batch_id, batch_data in enumerate(data_loader):
        batch_data
        smiles, bg, labels, masks, self_feats = batch_data
        bg = bg.to(device)
        labels = labels.to(device)
        masks = masks.to(device)
        self_feats = self_feats.to(device)
        n_feats = bg.ndata.pop('hv').to(device)
        e_feats = bg.edata.pop('he').to(device)
        prediction = model(bg, n_feats, e_feats, self_feats)
        loss = (loss_criterion(prediction, labels) * (masks != 0).float()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_meter.update(prediction, labels, masks)
        losses.append(loss.data.item())
    total_r2 = np.mean(train_meter.compute_metric(estimate_type))
    total_loss = np.mean(losses)
    if epoch % 5 == 0:
        print('epoch {:d}/{:d}, training_{} {:.4f}, training_loss {:.4f}'.format(epoch + 1, n_epochs, estimate_type,
                                                                                 total_r2, total_loss))
    return total_r2, total_loss


def run_an_eval_epoch(n_epochs, model, data_loader, loss_criterion, estimate_type):
    model.eval()
    val_losses = []
    eval_meter = Meter()
    preds = None
    targets = None
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks, self_feats = batch_data
            bg = bg.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            self_feats = self_feats.to(device)
            n_feats = bg.ndata.pop('hv').to(device)
            e_feats = bg.edata.pop('he').to(device)
            vali_prediction = model(bg, n_feats, e_feats, self_feats)
            val_loss = (loss_criterion(vali_prediction, labels) * (masks != 0).float()).mean()
            val_loss = val_loss.detach().cpu().numpy()
            val_losses.append(val_loss)
            eval_meter.update(vali_prediction, labels, masks)

            if preds is None:
                preds = vali_prediction.clone().detach()
                targets = labels.clone().detach()
            else:
                preds = torch.cat((preds, vali_prediction), dim=0)
                targets = torch.cat((targets, labels), dim=0)

        total_score = np.mean(eval_meter.compute_metric(estimate_type))
        total_loss = np.mean(val_losses)

    preds = torch.tensor(preds).cpu()
    targets = torch.tensor(targets).cpu()

    # np.savetxt('singal_result.csv', np.concatenate((targets, preds), axis=1), delimiter=',')

    return total_score, total_loss, preds, targets


def load_csv(random_seed):
    df = pd.read_csv('D:/Dataset/??????smile????????????1072??????.csv', index_col=0)
    df = df[['homo_smiles', 'homo_sol_smiles', 'N']]
    data = shuffle(df, random_state=random_seed)

    if 'com' in data.columns:
        data = data[['homo_smiles', 'homo_sol_smiles', 'N', 'com']]

    else:
        data['com'] = df.homo_smiles + '.' + df.homo_sol_smiles

    if not "N" in data.columns:
        print('plz,Change your traget columns to ???N???')

    # scaler = MinMaxScaler()
    # label = np.array(data['N']).reshape(-1,1)
    # scaler.fit(label)
    # data['N'] = scaler.fit_transform(label)

    return data
def load_train_valid_test(random_seed):
    df = pd.read_csv('D:/Dataset/??????smile????????????1072??????.csv', index_col=0)
    df = df[['homo_smiles', 'homo_sol_smiles', 'N']]
    data = shuffle(df, random_state=random_seed)

    # data = data.drop_duplicates(subset=['homo_smiles','homo_sol_smiles'])  ####  !!!???????????????
    # print('???????????????'+str(len(data)))

    if 'com' in data.columns:
        data = data[['homo_smiles', 'homo_sol_smiles', 'N','com']]

    else:
        data['com'] = df.homo_smiles + '.' + df.homo_sol_smiles


    if not "N" in data.columns:
        print('plz,Change your traget columns to ???N???')

    scaler = MinMaxScaler()
    label = np.array(data['N']).reshape(-1,1)
    scaler.fit(label)
    data['N'] = scaler.fit_transform(label)

    train_valid = data.iloc[:1000, :]
    test = data.iloc[1000:, :]

    print('trian_valid length:'+str(len(train_valid)))

    print('test length:'+str(len(test)))

    return train_valid,test


def random_split_data(data, split_rate, random_seed, batch_size):
    df = shuffle(data, random_state=random_seed)

    train = df.iloc[:int(float(split_rate) * len(df)), :]
    test = df.iloc[int(float(split_rate) * len(df)):, :]

    train_datasets = load_data(train, 'train_{}'.format(random_seed), True, task_name, smiles_column)
    val_datasets = load_data(test, 'test_{}'.format(random_seed), True, task_name, smiles_column)

    loader_batch_size = batch_size
    train_loader = DataLoader(train_datasets, batch_size=loader_batch_size, shuffle=True,
                              collate_fn=collate_emodel)
    vali_loader = DataLoader(val_datasets, batch_size=loader_batch_size, shuffle=True,
                             collate_fn=collate_emodel)

    return train_loader, vali_loader


def get_k_fold_data_limu(k, i, X, y):
    assert k >= 1
    fold_size = X.shape[0] // k
    X_train = None
    y_train = None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        if (j==9) and ((j+1)*fold_size!=len(X)):
            idx = slice(j * fold_size, len(X))
        X_part, y_part = X[idx], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = pd.concat([pd.Series(X_train), pd.Series(X_part)], 0)
            y_train = pd.concat([pd.Series(y_train), pd.Series(y_part)], 0)

    return X_train, y_train, X_valid, y_valid


def k_fold_data(data, k_num, i_num, batch_size):
    X_train, y_train, X_valid, y_valid = get_k_fold_data_limu(k_num, i_num, data[smiles_column].values,
                                                              data[task_name].values)
    train = pd.DataFrame({smiles_column: X_train, task_name: y_train})
    test = pd.DataFrame({smiles_column: X_valid, task_name: y_valid})

    train_datasets = load_data(train, 'train_{}_fold_{}'.format(k_num, i_num), True, task_name, smiles_column)
    val_datasets = load_data(test, 'test_{}_fold_{}'.format(k_num, i_num), True, task_name, smiles_column)
    loader_batch_size = batch_size
    train_loader = DataLoader(train_datasets, batch_size=loader_batch_size, shuffle=True,
                              collate_fn=collate_emodel)
    vali_loader = DataLoader(val_datasets, batch_size=loader_batch_size, shuffle=True,
                             collate_fn=collate_emodel)

    return train_loader, vali_loader,train,test


def FGAT_model(self_feats_dim, linear_feats_len, Dropout, num_layer, num_timestep, grah_feat_size):
    # self_feats_dim = 167+208 # ???????????????????????????
    # linear_feats_len = 167+208 # ??????mlp???????????????????????????

    model = FPSAT(node_feat_size=n_feats,
                  edge_feat_size=e_feats,
                  linear_feats_len=linear_feats_len,
                  self_feats_dim=self_feats_dim,
                  n_tasks=1,
                  dropout=Dropout,
                  num_layers=num_layer,
                  num_timesteps=num_timestep,
                  graph_feat_size=grah_feat_size).to(device)
    return model


def FP_Attention():
    # ???FP_attention model ????????????????????????????????????????????????
    return model_zoo.AttentiveFPPredictor(node_feat_size=n_feats,
                                          edge_feat_size=e_feats,
                                          num_layers=2,
                                          num_timesteps=1,
                                          graph_feat_size=300,
                                          n_tasks=1,
                                          dropout=0.5
                                          )


def plots(y_pred, y_test, model_name, model_name_pic):
    f, ax = plt.subplots(1, 1)
    # actual = np.array(scaler_y.inverse_transform(y_test))
    # predicted = np.array(scaler_y.inverse_transform(y_pred))
    actual = np.array(y_test)
    predicted = np.array(y_pred)
    sns.regplot(actual, predicted, scatter_kws={'marker': '.', 's': 5, 'alpha': 0.8},
                line_kws={'color': 'c'})
    print("Mean absolute error (MAE):      %f" % mean_absolute_error(actual, predicted))
    print("Mean squared error (MSE):       %f" % mean_squared_error(actual, predicted))
    print("Root mean squared error (RMSE): %f" % sqrt(mean_squared_error(actual, predicted)))
    print("R square (R^2):                 %f" % r2_score(actual, predicted))

    plt.xlabel('Acutal')
    plt.ylabel('Predicted')
    # plt.suptitle("Actual Vs Predicted")
    anchored_text = AnchoredText('model' + ':      ' + model_name_pic + '\n' \
                                 + 'RMSE:       ' + str(round(sqrt(mean_squared_error(actual, predicted)), 3)) + '\n' \
                                 + "R2 Score:  " + str(round(r2_score(actual, predicted), 3)), loc=4,
                                 prop=dict(size=12))

    ax.add_artist(anchored_text)
    plt.savefig(model_name)
    plt.tight_layout()
    # plt.show()


def train(model, train_loader, vali_loader, estimate_type, cross_vali, seed_recard, stopper_filename, k):
    loss_fn = nn.MSELoss(reduction='none')

    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.000001)

    if estimate_type == 'r2':
        early_stop_mode = 'higher'
    else:
        early_stop_mode = 'lower'

    stopper = EarlyStopping(mode=early_stop_mode, patience=50, filename=stopper_filename)

    n_epochs = 1000

    for e in range(n_epochs):

        score = run_a_train_epoch(n_epochs, e, model, train_loader, loss_fn, optimizer, estimate_type)
        val_score = run_an_eval_epoch(n_epochs, model, vali_loader, loss_fn, estimate_type)
        early_stop = stopper.step(val_score[0], model)
        if e % 5 == 0:
            print('epoch {:d}/{:d}, validation {} {:.4f}, validation {} {:.4f}, best validation {} {:.4f}'.format(
                e + 1, n_epochs, estimate_type, val_score[0], 'loss', val_score[1],
                estimate_type, stopper.best_score))

        if early_stop:
            # outcome.append(stopper.best_score)
            # outcome.append(train)
            # outcome.append(test)
            # outcome.append(model)\
            np.savetxt(str(stopper_filename)[:-4] + 'predandtrue' + '.csv',
                       np.concatenate((val_score[2], val_score[3]), axis=1), delimiter=',')
            OUTCOME.append(stopper.best_score)
            estimate.append(estimate_type)
            Lr.append(lr)
            SEED.append(seed)
            splid_seed.append(seed_recard)
            # pic_save_path = 'FGAT_kfold_macc17rdk9__pic'

            # if pic_save_path not in os.listdir('result folder'):
            #     os.makedirs('result folder/' + pic_save_path)

            # plots(val_score[2],val_score[3],'result folder/' + pic_save_path+'/'+'_fold'+str(k),'FGAT')

            break


#     if  cross_vali:
#         outname = 'kfold'
#         return OUTCOME,estimate,Lr,SEED,outname

#     else:
#         out = pd.DataFrame({'result':OUTCOME,'estimate':estimate,'learning_rate':Lr,'whole_seed':SEED,'splid_seed':splid_seed})
#         outname = 'random_split'
#         return OUTCOME,estimate,Lr,SEED,splid_seed,outname

#     return out.to_csv('result_{}_{}.csv'.format(estimate,outname))

# seed = 79 33 1
seed = 33
all_seed = seed

random.seed(seed)
np.random.seed(seed)
# os.environ['PYTHONHASHSEED'] = str(seed)

torch.manual_seed(seed)  # ???CPU??????????????????
torch.cuda.manual_seed(seed)  # ?????????GPU??????????????????
torch.cuda.manual_seed_all(seed)

self_feats_dim = 3  # 167#208  # ???????????????????????????
linear_feats_len = 3  # 167#208  # ??????mlp???????????????????????????

atom_featurizer = AttentiveFPAtomFeaturizer(atom_data_field='hv')
bond_featurizer = AttentiveFPBondFeaturizer(bond_data_field='he')
n_feats = atom_featurizer.feat_size('hv')
e_feats = bond_featurizer.feat_size('he')

device = 'cpu'
task_name = 'N'
smiles_column = 'com'

data = load_csv(random_seed=seed)
# data,test_data = load_train_valid_test(random_seed=seed)

# model = FGAT_model()
cross_vali = True
estimate_type = 'rmse'

if cross_vali:
    k_fold = 10

    drop = [0.5]
    Layer = [2]
    num_timestep = [1]
    graph_feat_size = [100, 200, 300]

    for dropout in drop:
        for layer in Layer:
            for num_time in num_timestep:
                OUTCOME = []
                estimate = []
                Lr = []
                SEED = []
                splid_seed = []
                for _ in range(k_fold):
                    print('---------------------------------------------------------------')
                    print('-------------------------{}----fold----------------------------'.format(_))
                    print('dropout----------------' + str(dropout))
                    print('num_layer----------------' + str(layer))
                    print('num_time----------------' + str(num_time))
                    print('variance:'+str(self_feats_dim))
                    print('dataset:'+str(len(data)))
                    model = FGAT_model(self_feats_dim, linear_feats_len, dropout, layer, num_time, 300)

                    x, y,orgin_trian,orgin_test = k_fold_data(data, k_fold, _, 32)
                    print('train_set:' + str(len(orgin_trian)))
                    print('test_set:' + str(len(orgin_test)))

                    stopper_filename = 'macc17rdk9_kfold{}_result_{}_wholeseed_{}.pth'.format(_ + 1, estimate_type,
                                                                                              seed)

                    train(model, x, y, estimate_type, cross_vali, seed, stopper_filename, _)

                    out = pd.DataFrame({'result': OUTCOME, 'estimate': estimate, 'learning_rate': Lr, 'seed': SEED})

                print('dropout----------------' + str(dropout))
                print('num_layer----------------' + str(layer))
                print('num_time----------------' + str(num_time))

                print(np.mean(OUTCOME))

                print(out)

            # out.to_csv('result/macc17rdk9_kfold_result_{}_wholeseed_{}.csv'.format(estimate[0],seed))

else:
    split_size = 5
    train_rate = 0.9
    OUTCOME = []
    estimate = []
    Lr = []
    SEED = []
    splid_seed = []
    for _ in range(split_size):
        # np.random.seed(123)

        model = FGAT_model(self_feats_dim, linear_feats_len)

        seed_spl = int(np.random.randint(1, 999, 1))  # ???????????????????????? ????????????????????????randam??????state

        x, y = random_split_data(data, train_rate, seed_spl, 32)

        stopper_filename = 'spl_result_{}_size_{}_seedsple_{}.pth'.format(estimate_type, split_size, seed_spl)

        train(model, x, y, estimate_type, cross_vali, seed_spl, stopper_filename)

    # out = pd.DataFrame(
    #     {'result': OUTCOME, 'estimate': estimate, 'learning_rate': Lr, 'whole_seed': SEED, 'splid_seed': splid_seed})
    # out.to_csv('rsp_result_{}_size_{}.csv'.format(estimate[0],split_size))

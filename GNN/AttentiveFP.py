import pandas as pd

import dgl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dgllife.model import model_zoo
from dgllife.utils import smiles_to_bigraph
from dgllife.utils import EarlyStopping, Meter
from dgllife.utils import AttentiveFPAtomFeaturizer
from dgllife.utils import AttentiveFPBondFeaturizer
from sklearn.utils import shuffle
# if torch.cuda.is_available():
#     print('use GPU')
#     device = 'cuda'
# else:
#     print('use CPU')
#     device = 'cpu'

device = 'cpu'
# 设置全局随机种子
import os
import random
import numpy as np

# seed = 79 33
seed = 33

random.seed(seed)
np.random.seed(seed)
# os.environ['PYTHONHASHSEED'] = str(seed)

torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)

def set_random_seed(seed=33):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def collate_molgraphs(data):
    assert len(data[0]) in [3, 4], \
        'Expect the tuple to be of length 3 or 4, got {:d}'.format(len(data[0]))
    if len(data[0]) == 3:
        smiles, graphs, labels = map(list, zip(*data))
        masks = None
    else:
        smiles, graphs, labels, masks = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    if masks is None:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)
    return smiles, bg, labels, masks


atom_featurizer = AttentiveFPAtomFeaturizer(atom_data_field='hv')
bond_featurizer = AttentiveFPBondFeaturizer(bond_data_field='he')
n_feats = atom_featurizer.feat_size('hv')
e_feats = bond_featurizer.feat_size('he')

from dgllife.data import MoleculeCSVDataset

df = pd.read_csv('D:/Dataset/统一smile亲核试剂1072版本.csv',index_col=0)
df = shuffle(df, random_state=seed)
df = df[['homo_smiles','homo_sol_smiles','N']]
df['com'] = df.homo_smiles+'.'+df.homo_sol_smiles
df = df.dropna()
df.isnull().sum()
dfa = df[['com','N']]  # !!  没有加溶剂描述符
dfa.columns = ['SMILES','VALUES']

# dftest = dfa.iloc[1000:, :]
# dfn = dfa.iloc[:1000, :]


def load_data(data,name,load):
    dataset = MoleculeCSVDataset(data,
                                 smiles_to_graph=smiles_to_bigraph,
                                 node_featurizer=atom_featurizer,
                                 edge_featurizer= bond_featurizer,
                                 smiles_column='SMILES',
                                 cache_file_path=str(name)+'_dataset.bin',
                                 task_names=['VALUES'],
                                 load=load,init_mask=True,n_jobs=8
                            )
    return dataset


def load_train_valid_test(random_seed):
    df = pd.read_csv('D:/Dataset/统一smile亲核试剂1072版本.csv', index_col=0)
    df = df[['homo_smiles', 'homo_sol_smiles', 'N']]
    data = shuffle(df, random_state=random_seed)

    # data = data.drop_duplicates(subset=['homo_smiles','homo_sol_smiles'])  ####  !!!删除重复值
    # print('数据集大小'+str(len(data)))

    if 'com' in data.columns:
        data = data[['homo_smiles', 'homo_sol_smiles', 'N','com']]

    else:
        data['com'] = df.homo_smiles + '.' + df.homo_sol_smiles


    if not "N" in data.columns:
        print('plz,Change your traget columns to “N”')
    #
    # scaler = MinMaxScaler()
    # label = np.array(data['N']).reshape(-1,1)
    # scaler.fit(label)
    # data['N'] = scaler.fit_transform(label)

    train_valid = data.iloc[:1000, :]
    test = data.iloc[1000:, :]

    print('trian_valid length:'+str(len(train_valid)))

    print('test length:'+str(len(test)))

    return train_valid,test

def run_a_train_epoch(n_epochs, epoch, model, data_loader, loss_criterion, optimizer):
    model.train()
    losses = []
    train_meter = Meter()
    for batch_id, batch_data in enumerate(data_loader):
        batch_data
        smiles, bg, labels, masks = batch_data
        bg=bg.to(device)
        labels = labels.to(device)
        masks = masks.to(device)
        n_feats = bg.ndata.pop('hv').to(device)
        e_feats = bg.edata.pop('he').to(device)
        prediction = model(bg, n_feats, e_feats)
        loss = (loss_criterion(prediction, labels) * (masks != 0).float()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_meter.update(prediction, labels, masks)
        losses.append(loss.data.item())
    total_r2 = np.mean(train_meter.compute_metric('r2'))
    total_loss = np.mean(losses)
    if epoch % 10 == 0:
        print('epoch {:d}/{:d}, training_r2 {:.4f}, training_loss {:.4f}'.format(epoch + 1, n_epochs, total_r2,total_loss))
    return total_r2, total_loss

def run_an_eval_epoch(n_epochs, model, data_loader,loss_criterion):
    model.eval()
    val_losses=[]
    eval_meter = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            bg = bg.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            n_feats = bg.ndata.pop('hv').to(device)
            e_feats = bg.edata.pop('he').to(device)
            vali_prediction = model(bg, n_feats, e_feats)
            val_loss = (loss_criterion(vali_prediction, labels) * (masks != 0).float()).mean()
            val_loss=val_loss.detach().cpu().numpy()
            val_losses.append(val_loss)
            eval_meter.update(vali_prediction, labels, masks)
        total_score = np.mean(eval_meter.compute_metric('rmse'))
        total_loss = np.mean(val_losses)
    return total_score, total_loss


# def get_k_fold_data_limu(k, i, X, y):
#     assert k >= 1
#     fold_size = X.shape[0] // k
#     X_train = None
#     y_train = None
#     for j in range(k):
#         idx = slice(j * fold_size, (j + 1) * fold_size)
#         X_part, y_part = X[idx], y[idx]
#         if j == i:
#             X_valid, y_valid = X_part, y_part
#         elif X_train is None:
#             X_train, y_train = X_part, y_part
#         else:
#             X_train = pd.concat([pd.Series(X_train), pd.Series(X_part)], 0)
#             y_train = pd.concat([pd.Series(y_train), pd.Series(y_part)], 0)
#
#     return X_train, y_train, X_valid, y_valid

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

OUTCOME = []
fold_num = 10
test_r2 = []
# test_datasets = load_data(dftest,'testss',True)
# test_daterloader = DataLoader(test_datasets, batch_size=1, shuffle=True, collate_fn=collate_molgraphs)

def load_model(stopper_filename,val_datasets,n_feats,e_feats):

    Model = model_zoo.AttentiveFPPredictor(node_feat_size=n_feats,
                                 edge_feat_size=e_feats,
                                 num_layers=2,
                                 num_timesteps=2,
                                 graph_feat_size=200,
                                 n_tasks=1,
                                 dropout=0.2).to(device)

    fna = 'D:/课题相关/code/My_code_N/'+stopper_filename

    Model.load_state_dict(torch.load(fna, map_location=torch.device('cuda'))['model_state_dict'])

    # vali_loader = DataLoader(val_datasets, batch_size=1, shuffle=True,
    #                          collate_fn=collate_emodel)

    Label = None
    Pred = None

    from sklearn.metrics import r2_score

    for batch_id, batch_data in enumerate(val_datasets):
        # try:

        smiles, bg, labels, masks = batch_data

        bg = bg.to(device)

        labels = labels.to(device)
        masks = masks.to(device)

        try:
            n_feat = bg.ndata.pop('hv').to(device)
            e_feat = bg.edata.pop('he').to(device)

            preds = Model(bg, n_feat, e_feat)

            if Pred is None:
                Pred = preds
                Label = labels
            else:
                Pred = torch.cat((Pred, preds), dim=0)
                Label = torch.cat((Label, labels), dim=0)
        except:
            print(smiles)

    Preds = Pred.cpu().detach().numpy()
    targets = Label.cpu().detach().numpy()

    return r2_score(Preds,targets)

for i in range(fold_num):
    outcome = []
    print(len(dfa))
    X_train, y_train, X_valid, y_valid = get_k_fold_data_limu(fold_num, i, dfa.SMILES.values, dfa.VALUES.values)
    train = pd.DataFrame({'SMILES': X_train, 'VALUES': y_train})
    test = pd.DataFrame({'SMILES': X_valid, 'VALUES': y_valid})
    train_datasets = load_data(train,'train_{}'.format(i),True)
    val_datasets = load_data(test,'test_{}'.format(i),True)

    loader_batch_size = 32

    # model = model_zoo.AttentiveFPPredictor(node_feat_size=n_feats,
    #                                        edge_feat_size=e_feats,
    #                                        num_layers=2,
    #                                        num_timesteps=1,
    #                                        graph_feat_size=300,
    #                                        n_tasks=1,
    #                                        dropout=0.5
    #                                        )

    model = model_zoo.AttentiveFPPredictor(node_feat_size=n_feats,
                                 edge_feat_size=e_feats,
                                 num_layers=2,
                                 num_timesteps=2,
                                 graph_feat_size=200,
                                 n_tasks=1,
                                 dropout=0.2)
    model = model.to(device)



    train_loader = DataLoader(train_datasets, batch_size=loader_batch_size,shuffle=True,
                              collate_fn=collate_molgraphs)
    vali_loader = DataLoader(val_datasets,batch_size=loader_batch_size,shuffle=True,
                              collate_fn=collate_molgraphs)

    loss_fn = nn.MSELoss(reduction='none')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.000001)

    stopper_filename = 'attentiveFP_kfold{}.pth'.format(i + 1)

    stopper = EarlyStopping(mode='lower', patience=50, filename=stopper_filename)


    n_epochs = 1000

    for e in range(n_epochs):
        score = run_a_train_epoch(n_epochs, e, model, train_loader, loss_fn, optimizer)
        val_score = run_an_eval_epoch(n_epochs, model, vali_loader, loss_fn)
        early_stop = stopper.step(val_score[0], model)
        if e % 5 == 0:
            print('epoch {:d}/{:d}, validation {} {:.4f}, validation {} {:.4f}, best validation {} {:.4f}'.format(
                e + 1, n_epochs, 'r2', val_score[0], 'loss', val_score[-1],
                'r2', stopper.best_score))
        if early_stop:
            outcome.append(stopper.best_score)
            OUTCOME.append(outcome)

            # r2 = load_model(stopper_filename,test_daterloader,n_feats,e_feats)
            # print(r2)
            # test_r2.append(r2)
            break

    print(OUTCOME)
    print(np.mean(OUTCOME))
    # print('---test-r2---')
    # print(test_r2)
    # print(np.mean(test_r2))





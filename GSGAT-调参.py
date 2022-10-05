
import pandas as pd
import dgl
from dgllife.data import MoleculeCSVDataset
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from sklearn.utils import shuffle
# GAT
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from dgllife.utils import smiles_to_bigraph
from dgllife.utils import EarlyStopping, Meter
from dgllife.utils import AttentiveFPAtomFeaturizer
from dgllife.utils import AttentiveFPBondFeaturizer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from numpy import *
from matplotlib.offsetbox import AnchoredText


from net import FPSAT


from rdkit.Chem import MACCSkeys
from rdkit import Chem



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


        return np.concatenate([a], 1)

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

    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()

    return total_score, total_loss,preds,targets


def load_csv(random_seed):
    df = pd.read_csv('D:/Dataset/统一smile亲核试剂1072版本.csv', index_col=0)
    df = df[['homo_smiles', 'homo_sol_smiles', 'N']]
    data = shuffle(df, random_state=random_seed)

    if 'com' in data.columns:
        data = data[['homo_smiles', 'homo_sol_smiles', 'N','com']]

    else:
        data['com'] = df.homo_smiles + '.' + df.homo_sol_smiles


    if not "N" in data.columns:
        print('plz,Change your traget columns to “N”')


    return data


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

    return train_loader, vali_loader


def FGAT_model(self_feats_dim, linear_feats_len,dropout,num_layers,graph_size):
    # self_feats_dim = 167+208 # 添加的分子指纹长度
    # linear_feats_len = 167+208 # 经过mlp之后的分子指纹长度

    model = FPSAT(node_feat_size=n_feats,
                  edge_feat_size=e_feats,
                  linear_feats_len=linear_feats_len,
                  self_feats_dim=self_feats_dim,
                  n_tasks=1,
                  dropout=dropout,
                  num_layers=num_layers,
                  num_timesteps=1,
                  graph_feat_size=graph_size).to(device)
    return model

def plots(y_pred, y_test, model_name,model_name_pic):

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
    print("R square (R^2):                 %f" % r2_score(actual,predicted))

    plt.xlabel('Acutal')
    plt.ylabel('Predicted')
    # plt.suptitle("Actual Vs Predicted")
    anchored_text = AnchoredText('model' + ':      ' + model_name_pic + '\n' \
                                 + 'RMSE:       ' + str(round(sqrt(mean_squared_error(actual, predicted)), 3)) + '\n' \
                                 + "R2 Score:  " + str(round(r2_score(actual, predicted), 3)), loc=4,
                                 prop=dict(size=14))

    ax.add_artist(anchored_text)
    plt.savefig(model_name)
    plt.tight_layout()
    # plt.show()


def train(model, train_loader, vali_loader, estimate_type, cross_vali, seed_recard, stopper_filename,k,lr,patience):
    loss_fn = nn.MSELoss(reduction='none')

    # lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.000001)

    if estimate_type == 'r2':
        early_stop_mode = 'higher'
    else:
        early_stop_mode = 'lower'

    stopper = EarlyStopping(mode=early_stop_mode, patience=patience,filename=stopper_filename)

    n_epochs = 1000

    for e in range(n_epochs):
        epoch.append(e+1)

        score = run_a_train_epoch(n_epochs, e, model, train_loader, loss_fn, optimizer, estimate_type)
        Losses_train.append(score[1])
        train_r2.append(score[0])

        val_score = run_an_eval_epoch(n_epochs, model, vali_loader, loss_fn, estimate_type)
        Loss_test.append(val_score[1])
        test_r2.append(val_score[0])

        early_stop = stopper.step(val_score[0], model)
        if e % 5 == 0:
            print('epoch {:d}/{:d}, validation {} {:.4f}, validation {} {:.4f}, best validation {} {:.4f}'.format(
                e + 1, n_epochs, estimate_type, val_score[0], 'loss', val_score[1],
                estimate_type, stopper.best_score))

        if early_stop:

            np.savetxt(str(stopper_filename)[:-4]+'predandtrue'+'.csv', np.concatenate((val_score[2],val_score[3]), axis=1), delimiter=',')
            OUTCOME.append(stopper.best_score)
            estimate.append(estimate_type)
            Lr.append(lr)
            SEED.append(seed)
            splid_seed.append(seed_recard)
            pic_save_path = 'FGAT_kfold_6.17_relu_pic'

            if pic_save_path not in os.listdir('result folder'):
                os.makedirs('result folder/' + pic_save_path)

            plots(val_score[2],val_score[3],'result folder/' + pic_save_path+'/'+'fold_'+str(k),'FGAT')

            break


def load_model(stopper_filename,val_datasets,n_feats,e_feats):

    Model = FPSAT(node_feat_size=n_feats,
                  edge_feat_size=e_feats,
                  linear_feats_len=linear_feats_len,
                  self_feats_dim=self_feats_dim,
                  n_tasks=1,
                  dropout=0.,
                  num_layers=2,
                  num_timesteps=2,
                  graph_feat_size=300).to(device)

    fna = 'D:/课题相关/code/My_code_N/'+stopper_filename

    Model.load_state_dict(torch.load(fna, map_location=torch.device('cuda'))['model_state_dict'])

    Label = None
    Pred = None


    for batch_id, batch_data in enumerate(val_datasets):
        # try:

        smiles, bg, labels, masks, self_feats = batch_data

        bg = bg.to(device)

        labels = labels.to(device)
        masks = masks.to(device)

        g = dgl.batch([bg])
        try:
            n_feat = bg.ndata.pop('hv').to(device)
            e_feat = bg.edata.pop('he').to(device)

            preds = Model(bg, n_feat, e_feat, self_feats.to(device))

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

seed = 33
all_seed = seed

random.seed(seed)
np.random.seed(seed)
# os.environ['PYTHONHASHSEED'] = str(seed)

torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)

self_feats_dim = 167 #+ 15#208  # 添加的分子指纹长度
linear_feats_len = 167 #+ 15#208  # 经过mlp之后的分子指纹长度


atom_featurizer = AttentiveFPAtomFeaturizer(atom_data_field='hv')
bond_featurizer = AttentiveFPBondFeaturizer(bond_data_field='he')
n_feats = atom_featurizer.feat_size('hv')
e_feats = bond_featurizer.feat_size('he')

device = 'cuda'
task_name = 'N'
smiles_column = 'com'  #!!!  homo_smiles or com

data = load_csv(random_seed=seed)  # !!!!


cross_vali = True
estimate_type = 'r2'

dropout = [0.5]
layers = [ 3]
lr = [0.001,0.005, 0.01]
graph_size = [200, 300, 500]
batch = [24,32,64]

Learning_Rate = []
Layers = []
GS = []
Batch = []
Dp= []
Result = []

if cross_vali:
    k_fold = 10
    OUTCOME = []
    estimate = []
    Lr = []
    SEED = []
    splid_seed = []
    Test_Score = []
    ak = 0
    AK = 1*3*3*3*3
    for dpo in dropout:
        for layer in layers:
            for learning_rate in lr:
                for bs in batch:
                    for gs in graph_size:
                        ak += 1

                        Learning_Rate.append(learning_rate)
                        Layers.append(layer)
                        GS.append(gs)
                        Batch.append(bs)
                        Dp.append(dpo)

                        for _ in range(k_fold):
                            Loss_test = []
                            Losses_train = []
                            train_r2 = []
                            test_r2 = []
                            epoch = []


                            model = FGAT_model(self_feats_dim, linear_feats_len,dpo,layer,gs)

                            x, y = k_fold_data(data, k_fold, _, bs)

                            stopper_filename = 'FAGT_kfold{}_result_{}_wholeseed_{}.pth'.format(_+1,estimate_type, seed)

                            train(model, x, y, estimate_type, cross_vali, seed, stopper_filename,_,learning_rate,15)
                            print('dropout:'+str(dpo)+'; num_layer:'+str(layer)+'; lr:'+str(learning_rate)+';graph_size: '+str(gs)+';Batch_size'+str(bs))
                            print('processing:'+str(ak)+'/'+str(AK))
                            print('fold: '+str(_))
                            print('---------------------------------------------')

                        Result.append(np.mean(OUTCOME))

                        DF = pd.DataFrame({'lr':Learning_Rate,'dropout':Dp,'Graph_size':GS,'Num_layer':Layers,'Batch_size':Batch,'R2':Result})
                        DF.to_csv('D:/Dataset/parameters4.csv')

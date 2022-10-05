import pandas as pd
import matplotlib.pyplot as plt
from sklearn import manifold
import rdkit.Chem as Chem
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from sklearn.model_selection import train_test_split
import os
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
import rdkit

def split_map(x):
    split = 1 if x == 'train' else 0
    return split


def split_tsne(data, smiles_columns, hue_columns,save_path,save=False, three_dim=False):
    tsn_df = data[smiles_columns]
    mol = [Chem.MolFromSmiles(x) for x in tsn_df]
    fp_mac = [MACCSkeys.GenMACCSKeys(y) for y in mol]
    bit_metrics = np.array(fp_mac, int)
    if three_dim:
        n_components = 3
        filename = '3d'
    else:
        n_components = 2
        filename = '2d'

    tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=5)
    bit_tsne = tsne.fit_transform(bit_metrics)

    #  PiYG  seismic coolwarm Pastel1 tab10  YlGn BuGn PuBu inferno viridis
    color_name = 'coolwarm'
    color_level = 50
    cm = plt.cm.get_cmap(color_name)

    if three_dim:
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(111, projection='3d')
        for x, y, z, m in zip(bit_tsne[:, 0], bit_tsne[:, 1], bit_tsne[:, 2], data[hue_columns]):
            if m == 1:
                a = ax.scatter(x, y, z, marker='.', color='r', alpha=0.3)
            else:
                b = ax.scatter(x, y, z, c='b', marker='x')
        plt.legend([a, b], ['training set', 'testing set'])
        ax.set_xlabel('t_SNE Reduced Dimension 1', fontproperties='Times New Roman')
        ax.set_ylabel('t_SNE Reduced Dimension 2', fontproperties='Times New Roman')
        ax.set_zlabel('t_SNE Reduced Dimension 3', fontproperties='Times New Roman')
    else:
        for x, y, z in zip(bit_tsne[:, 0], bit_tsne[:, 1], data[hue_columns]):
            if z == 1:
                a = plt.scatter(x, y, c='b', marker='.', alpha=0.2)
            else:
                b = plt.scatter(x, y, c='g', marker='x', alpha=0.8)

        plt.legend([a, b], ['training set', 'testing set'])

        plt.xlabel('t_SNE Reduced Dimension 1 ', size=16, fontproperties='Times New Roman')
        plt.ylabel('t_SNE Reduced Dimension 2 ', size=16, fontproperties='Times New Roman')

    if save:
        plt.savefig(save_path + '/' + 't_sne_split_{}.png'.format(filename))

    return plt.show()



def split_N_distribution(train_set,test_set,save_path,save=False):
    plt.figure(figsize=(16,8))
    plt.subplot(121)
    x = ['training set','testing set']
    y = [train_set.N.mean(),test_set.N.mean()]

    sns.barplot(x= x,y=y)
    for i in range(len(x)):
        plt.text(i - 0.1, y[i] + 1, round(y[i], 2), color='black', size=16)
    plt.text(-0.8, 14.5, 'A', color='black', size=30)

    plt.ylim(0, 15)
    plt.ylabel('Average of N values', size=15)

    plt.subplot(122)
    plt.hist(train_set.N, alpha=0.8)
    plt.hist(test_set.N)
    plt.ylabel('Distribution of N values', size=15)
    plt.text(-18, 190, 'B', color='black', size=30)

    if save:
        plt.savefig(save_path + '/' + 'split_N_distribution.png')
    return plt.show()


def k_fold(k, i, X, y):
    assert k >= 1
    fold_size = X.shape[0]//k
    X_train = None
    y_train = None
    for j in range(k):
        idx = slice(j*fold_size,(j+1)*fold_size)
        if (j==9) and ((j+1)*fold_size!=len(X)):
            idx = slice(j * fold_size, len(X))
        X_part,y_part = X[idx],y[idx]
        if j == i:
            X_valid, y_valid = X_part,y_part

        elif X_train is None:
            X_train, y_train = X_part,y_part
        else:
            X_train = np.concatenate([X_train,X_part],0)
            y_train =np.concatenate([y_train,y_part],0)

    return X_train,y_train,X_valid,y_valid


def Min_Max_Scaler(X,y):
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaler_y.fit(y)
    scaler_X.fit(X)
    X_ = scaler_X.transform(X)
    y_ = scaler_y.transform(y)
    return X_,y_,scaler_X,scaler_y

def MFF(data,smile_col):
    def Get_bitvetors(fingerprints):
        bit = pd.DataFrame(np.array(fingerprints, int))
        return bit

    def Get_fp(mol):
        fp_mac = [MACCSkeys.GenMACCSKeys(x) for x in mol]
        fp_ava = [pyAvalonTools.GetAvalonFP(x, nBits=3096) for x in mol]
        fp_mor = [AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=1024) for x in mol]
        fp_atm = [rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(x, nBits=3096) for x in mol]
        fp_top = [rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(x, nBits=3096) for x in mol]
        fp_lyr = [Chem.LayeredFingerprint(x, maxPath=2, fpSize=3096) for x in mol]
        fp_rdk = [Chem.RDKFingerprint(mol=x, maxPath=2, fpSize=3096) for x in mol]
        fp_rdl = [Chem.RDKFingerprint(mol=x, maxPath=2, branchedPaths=False, fpSize=3096) for x in mol]
        fp_rdk_4 = [Chem.RDKFingerprint(mol=x, maxPath=4, fpSize=3096) for x in mol]
        fp_rdk_6 = [Chem.RDKFingerprint(mol=x, maxPath=6, fpSize=3096) for x in mol]
        fp_rdk_8 = [Chem.RDKFingerprint(mol=x, maxPath=8, fpSize=3096) for x in mol]
        fp_rdl_4 = [Chem.RDKFingerprint(mol=x, maxPath=4, branchedPaths=False, fpSize=3096) for x in mol]
        fp_rdl_6 = [Chem.RDKFingerprint(mol=x, maxPath=6, branchedPaths=False, fpSize=3096) for x in mol]
        fp_rdl_8 = [Chem.RDKFingerprint(mol=x, maxPath=8, branchedPaths=False, fpSize=3096) for x in mol]
        fp_mor_4 = [AllChem.GetMorganFingerprintAsBitVect(x, 4, nBits=1024) for x in mol]
        fp_mor_6 = [AllChem.GetMorganFingerprintAsBitVect(x, 6, nBits=1024) for x in mol]

        return fp_mac, fp_ava, fp_mor, fp_atm, fp_top, fp_lyr, fp_rdk, fp_rdl, fp_rdk_4, fp_rdk_6, fp_rdk_8, fp_rdl_4, fp_rdl_6, fp_rdl_8, fp_mor_4, fp_mor_6

    # 获取分子指纹的比特矩阵
    def Get_singal_mol_bitvector(mol):
        singal_fp_all = []
        fp_name = ['fp_mac', 'fp_ava', 'fp_mor', 'fp_atm', 'fp_top', 'fp_lyr', 'fp_rdk', 'fp_rdl', 'fp_rdk_4',
                   'fp_rdk_6', 'fp_rdk_8', 'fp_rdl_4', 'fp_rdl_6', 'fp_rdl_8', 'fp_mor_4', 'fp_mor_6']

        for i in range(len(Get_fp(mol))):
            fp = Get_fp(mol)[i]
            fp_bit = Get_bitvetors(fp)
            fp_bit.columns = [fp_name[i] + '_' + str(j) for j in range(len(fp_bit.columns))]
            singal_fp_all.append(fp_bit)

        return singal_fp_all

    N_Mol = [Chem.MolFromSmiles(x) for x in data[smile_col]]

    chemical_fp_all = Get_singal_mol_bitvector(N_Mol)
    co_chem = pd.concat(chemical_fp_all, axis=1)

    return co_chem

def Final_dataset(fp_data,raw_data,N_col):
    N_value = raw_data[N_col]
    N_value.index = fp_data.index
    new_csv = pd.concat([fp_data,N_value],1)
    X = new_csv.values[:,:-1]
    y = new_csv.values[:,-1].reshape(-1,1)
    return X,y


def random_split(X,y,seed):
    X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=seed)
    return X_train,X_test,y_train,y_test


def one_fp(data,smile_col,fp_cho):

    def Get_bitvetors(fingerprints):
        bit = pd.DataFrame(np.array(fingerprints, int))
        return bit

    mol = [Chem.MolFromSmiles(x) for x in data[smile_col]]

    fp_name = { 'fp_mac':[MACCSkeys.GenMACCSKeys(x) for x in mol],
                'fp_ava' :[pyAvalonTools.GetAvalonFP(x, nBits=3096) for x in mol],
                'fp_mor': [AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=1024) for x in mol],
                'fp_atm' :[rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(x, nBits=3096) for x in mol],
                'fp_top' : [rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(x, nBits=3096) for x in mol],
                'fp_lyr' :[Chem.LayeredFingerprint(x, maxPath=2, fpSize=3096) for x in mol],
                'fp_rdk': [Chem.RDKFingerprint(mol=x, maxPath=2, fpSize=3096) for x in mol],
                'fp_rdl' : [Chem.RDKFingerprint(mol=x, maxPath=2, branchedPaths=False, fpSize=3096) for x in mol],
                'fp_rdk_4' : [Chem.RDKFingerprint(mol=x, maxPath=4, fpSize=3096) for x in mol],
                'fp_rdk_6': [Chem.RDKFingerprint(mol=x, maxPath=6, fpSize=3096) for x in mol],
                'fp_rdk_8' :[Chem.RDKFingerprint(mol=x, maxPath=8, fpSize=3096) for x in mol],
                'fp_rdl_4' : [Chem.RDKFingerprint(mol=x, maxPath=4, branchedPaths=False, fpSize=3096) for x in mol],
                'fp_rdl_6' : [Chem.RDKFingerprint(mol=x, maxPath=6, branchedPaths=False, fpSize=3096) for x in mol],
                'fp_rdl_8' : [Chem.RDKFingerprint(mol=x, maxPath=8, branchedPaths=False, fpSize=3096) for x in mol],
                'fp_mor_4':[AllChem.GetMorganFingerprintAsBitVect(x, 4, nBits=1024) for x in mol],
                'fp_mor_6':[AllChem.GetMorganFingerprintAsBitVect(x, 6, nBits=1024) for x in mol]}

    A_fp = fp_name[fp_cho]
    A_bit = Get_bitvetors(A_fp)
    A_bit.columns = [fp_cho+'_'+str(j) for j in range(len(A_bit.columns))]

    return A_bit


def rdkit_desc(data,smile_col):
    descriptor_names = [descriptor_name[0] for descriptor_name in Descriptors._descList]
    descriptor_calculation = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    mol = [Chem.MolFromSmiles(x) for x in data[smile_col]]
    # moe = [descriptor_calculation.CalcDescriptors(i) for i in mol]
    moe = np.nan_to_num(np.array([descriptor_calculation.CalcDescriptors(i) for i in mol]))
    rdkit_2d = pd.DataFrame(moe, columns=descriptor_names, index=data.index)

    return rdkit_2d


def special_data(data,train_or_test,choose_fp,mff=False):

    smiles_col = 'com'

    if mff:

        if not 'mff_features.csv' in os.listdir():

            fp_bit = MFF(data, smiles_col)  # 多分子指纹数据
            fp_bit.to_csv('mff_features_{}.csv'.format(train_or_test))
        else:
            fp_bit = pd.read_csv('mff_features_{}.csv'.format(train_or_test))

    else:
        '''
        fp_name = 
        ['fp_mac', 'fp_ava', 'fp_mor', 'fp_atm', 'fp_top', 'fp_lyr', 'fp_rdk', 'fp_rdl', 'fp_rdk_4',
       'fp_rdk_6', 'fp_rdk_8', 'fp_rdl_4', 'fp_rdl_6', 'fp_rdl_8', 'fp_mor_4', 'fp_mor_6']
       '''
        choose_fp = choose_fp

        if not 'one_fp_features.csv' in os.listdir():
            fp_bit = one_fp(data, smiles_col, choose_fp)
            fp_bit.to_csv('one_fp_features_{}.csv'.format(train_or_test))
        else:
            fp_bit = pd.read_csv('one_fp_features_{}.csv'.format(train_or_test),index_col=0)

    X, y = Final_dataset(fp_bit, data, 'N')  # 特征向量 X 目标 y
    X, y, scaler_X, scaler_y = Min_Max_Scaler(X, y)  # 大小归一化 numpy 型数据

    return X,y, scaler_X, scaler_y


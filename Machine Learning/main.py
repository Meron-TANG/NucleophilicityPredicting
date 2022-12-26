from typing import List

import numpy as np
import pandas as pd
import time
import data_visualization
from load_dataset import *
import warnings
from ML_Model import *
import os
from sklearn.utils import shuffle
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')
random_seed = 1

file_name = 'D:/Dataset/统一smile亲核试剂1072版本.csv'
save_path = 'D:/Dataset'
data = data_visualization.read_data(file_name,random_seed=random_seed,Shuffle=True)


# 数据可视化
#
# t_sne = data_visualization.t_SNE(data,"SMILES",'N',save_path,three_dim=True)
# Neoclephile_class = data_visualization.nucleo_class(data,'Type','N','Solvent',save_path)
# N_hist = data_visualization.hist_N(data,'N',save_path)
# nucleo_num = data_visualization.nucleo_num(data,'Type',save_path)
# nucleo_average_N = data_visualization.nucleo_meanN(data,'Type','N',save_path)



# 分割数据集
# df = pd.read_csv('D:/Dataset/8-2分割过的1100亲核数据.csv',encoding='gbk')
# df = df.dropna()
# df['split'] = df['train_test_split'].map(split_map)
#
train = pd.read_csv('D:/Dataset/N_train_1100-878_0.926_threshold=1_xgboost_1.csv',index_col='Sort')
test = pd.read_csv('D:/Dataset/N_test_1100-223_0.926_threshold=1_xgboost_1.csv',index_col='Sort')
# train['train_test_split'] = 'train'
# test['train_test_split'] = 'test'
# train['split'] = train['train_test_split'].map(split_map)
# test['split'] = test['train_test_split'].map(split_map)
#
# dim_2_split_tsne = split_tsne(df,'SMILES','split',save_path=save_path,three_dim=True,save=True)
# dim_3_split_tsne = split_tsne(df,'SMILES','split',save_path=save_path,three_dim=True,save=True)
# split_N_distribution = split_N_distribution(train,test,save_path=save_path,save=True)

# X_special_train,y_special_train = special_data(train,'train','fp_mac')
# X_special_test,y_special_test = special_data(train,'test','fp_mac')
#



# 加载数据
if 'com' in data.columns:
    df = data[['SMILES', 'SOL', 'N', 'com']]

else:
    df = data[['SMILES', 'SOL', 'N']]
    df['com'] = df.SMILES + '.' + df.SOL


SRMG = True

if SRMG:
    smiles_col = 'com'   #  nol sol !!
else:
    smiles_col = 'SMILES'   #  nol sol !!


mff= True
rdkit_2d = False

if mff:
    if not 'mff_features.csv'  in os.listdir():
        print('loading---mff--features')
        fp_bit = MFF(df,smiles_col) # 多分子指纹数据
        fp_bit.to_csv('mff_features.csv') # 加载文件的时候注意seed  seed不同文件排序不同
    else:
        fp_bit = pd.read_csv('mff_features.csv',index_col=0)

elif rdkit_2d:
    choose_fp = 'rdkit2d'
    if not 'rdk2d_features.csv'  in os.listdir():

        fp_bit = rdkit_desc(df,smiles_col) # 多分子指纹数据
        fp_bit.to_csv('rdk2d_features.csv')
    else:
        fp_bit = pd.read_csv('rdk2d_features.csv',index_col=0)

else:
    '''
    fp_name = 
    ['fp_mac', 'fp_ava', 'fp_mor', 'fp_atm', 'fp_top', 'fp_lyr', 'fp_rdk', 'fp_rdl', 'fp_rdk_4',
   'fp_rdk_6', 'fp_rdk_8', 'fp_rdl_4', 'fp_rdl_6', 'fp_rdl_8', 'fp_mor_4', 'fp_mor_6']
   '''

    choose_fp = 'fp_mor'

    sff_csv = 'one_fp_features_sol_'+choose_fp+'.csv'

    if not sff_csv in os.listdir():
        fp_bit = one_fp(df,smiles_col,choose_fp)
        fp_bit.to_csv('one_fp_features_sol_{}.csv'.format(choose_fp))
    else:
        fp_bit = pd.read_csv('one_fp_features_sol_{}.csv'.format(choose_fp))




X,y = Final_dataset(fp_bit,df,'N') # 特征向量 X 目标 y

X,y,scaler_X,scaler_y = Min_Max_Scaler(X,y)  # 大小归一化 numpy 型数据


# 数据分割并且建模



# ml_model_list = {'rf':rf(),'xgb':xgboost(),'lgbm':lgbm(),'gbr':gbr(),'svr':svr()}
# ml_model_name = 'rf'
# model = ml_model_list[ml_model_name]
# print('start training----{}-----'.format(ml_model_name))


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from numpy import *
from matplotlib.offsetbox import AnchoredText


def plots(X_train, y_test, model, model_name,model_name_pic):
    f, ax = plt.subplots(1, 1)
    actual = np.array(scaler_y.inverse_transform(y_test))
    predicted = np.array(scaler_y.inverse_transform(model.predict(X_train).reshape(-1,1)))

    sns.regplot(actual, predicted, scatter_kws={'marker': '.', 's': 5, 'alpha': 0.8},
                line_kws={'color': 'c'})
    print("Mean absolute error (MAE):      %f" % mean_absolute_error(actual, predicted))
    print("Mean squared error (MSE):       %f" % mean_squared_error(actual, predicted))
    print("Root mean squared error (RMSE): %f" % sqrt(mean_squared_error(actual, predicted)))
    print("R square (R^2):                 %f" % model.score(X_train,y_test))

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

for i in range(10):

    model_list = ['MLR','Adboost','Xtratree','Bagging','PLS','RF','KNN','SVR','LGBM','Xgboost']
    ml_model_name = model_list[i]
    model = defining_model(ml_model_name)

    print('start training----{}-----'.format(ml_model_name))

    # 输入想要训练的模型


    is_kfold = True

    SCORE = []
    MAE_ = []
    MSE_ = []
    RMSE_ = []


    if is_kfold:
        k = 10
        for i in range(k):

            X_train, y_train, X_valid, y_valid = k_fold(k, i, X, y)

            model.fit(X_train, y_train)

            r2 = model.score(X_valid, y_valid)

            r2_train = model.score(X_train, y_train)

            # SCORE.append(r2)
            #
            actual = np.array(scaler_y.inverse_transform(y_valid))
            predicted = np.array(scaler_y.inverse_transform(model.predict(X_valid).reshape(-1, 1)))

            MAE_.append(mean_absolute_error(actual, predicted))
            MSE_.append(mean_squared_error(actual, predicted))
            RMSE_.append(sqrt(mean_squared_error(actual, predicted)))
            SCORE.append(r2_score(actual, predicted))

            if not mff:
                pic_save_path = '{}_1072_kfold_fp_{}'.format(ml_model_name,choose_fp)

                if pic_save_path not in os.listdir('result folder'):
                    os.makedirs('result folder/'+pic_save_path)

                plots(X_valid, y_valid, model, 'result folder/{}_kfold_fp_{}'.format(ml_model_name,choose_fp)+'/'+ml_model_name + str(i), ml_model_name)

            else:
                pic_save_path = '{}_mff_1072_sol'.format(ml_model_name)

                if pic_save_path not in os.listdir('result folder'):
                    os.makedirs('result folder/' + pic_save_path)

                plots(X_valid, y_valid, model,
                      'result folder/'+pic_save_path + '/' + ml_model_name + str(i),
                      ml_model_name)


            print('{}-fold:'.format(i + 1) + 'r2_valid:' + str(round(r2, 3)) + ';' + 'r2_train:' + str(round(r2_train, 3)))

        if mff:
            print('use fingerprint: mff; use model:{}'.format(ml_model_name))

            print('10-fold mean R2 = {}'.format(round(np.mean(SCORE),4)))
            print('10-fold mean MAE = {}'.format(round(np.mean(MAE_),4)))
            print('10-fold mean MSE = {}'.format(round(np.mean(MSE_),4)))
            print('10-fold mean RMSE = {}'.format(round(np.mean(RMSE_),4)))

            save_ = pd.DataFrame({'r2':SCORE,'MAE':MAE_,'RMSE':RMSE_,'MSE':MSE_})
            for i in save_.columns:
                save_.loc['mean', i] = round(save_[i].mean(),2)
            save_.to_csv('result/ml/MFF_{}_1072.csv'.format(ml_model_name))

        else:
            print(' use fingerprint: {} \n use model {}  '.format(choose_fp,ml_model_name) )

            print('10-fold mean R2 = {}'.format(round(np.mean(SCORE),4)))
            print('10-fold mean MAE = {}'.format(round(np.mean(MAE_),4)))
            print('10-fold mean MSE = {}'.format(round(np.mean(MSE_),4)))
            print('10-fold mean RMSE = {}'.format(round(np.mean(RMSE_),4)))

            save_ = pd.DataFrame({'r2':SCORE,'MAE':MAE_,'RMSE':RMSE_,'MSE':MSE_})
            for i in save_.columns:
                save_.loc['mean', i] = round(save_[i].mean(),2)
            save_.to_csv('result/ml/{}_{}_1072.csv'.format(choose_fp,ml_model_name))

    else:
        special = False

        if special:
            X_train, y_train,scaler_X, scaler_y = special_data(train, 'train', choose_fp)
            X_test, y_test ,scaler_X, scaler_y = special_data(test, 'test', choose_fp)
            model.fit(X_train, y_train)
            r2 = model.score(X_test, y_test)
            r2_train = model.score(X_train, y_train)
            print('---spcial---')
            print(' use fingerprint: {} \n use model: {} \n '.format(choose_fp, ml_model_name))
            print('r2_valid:' + str(round(r2, 3)) + ';' + 'r2_train:' + str(round(r2_train, 3)))

        else:
            R2 = []
            for _ in range(10):
                seed = int(np.random.randint(0,1000,1))
                X_train,X_test,y_train,y_test = random_split(X,y,seed)
                model.fit(X_train, y_train)
                r2 = model.score(X_test, y_test)
                r2_train = model.score(X_train, y_train)
                R2.append(r2)
                print(r2)
                plots(X_test,y_test,model,ml_model_name+str(_),ml_model_name)
            print('十次随机采样平均 R2= '+str(np.mean(r2)))

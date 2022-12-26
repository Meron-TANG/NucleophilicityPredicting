import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn import svm
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.neural_network import MLPRegressor


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,ExtraTreesRegressor,BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

def defining_model(x):
    if x == 'MLR':
        model = LinearRegression()
    elif x=='Adboost':
        model = AdaBoostRegressor()
    elif x=='Xtratree':
        model = ExtraTreesRegressor(n_jobs=20)
    elif x=='Bagging':
        model = BaggingRegressor()
    elif x=='PLS':
        model = PLSRegression()
    elif x=='RF':
        model = RandomForestRegressor(n_jobs=20)
    elif x=='KNN':
        model = KNeighborsRegressor()
    elif x=='SVR':
        model = SVR()
    elif x=='LGBM':
        model = LGBMRegressor(n_jobs=20)
    elif x=='Xgboost':
        model = XGBRegressor(n_jobs=20)
    else:
        print("wrong selection")
    return model


# Gradient Boosting Regressor
def gbr():
    clf = GradientBoostingRegressor(learning_rate=0.05,
                                max_depth=31,
                                max_features=500,
                                min_samples_leaf=20,
                                n_estimators=1000)
    return clf

def xgboost():
    return XGBRegressor(n_estimators=150, learning_rate=0.1, max_depth=10, n_jobs=20)

def rf():
    return RandomForestRegressor(n_estimators=150,max_depth=10,n_jobs=20)


def svr():
    # kernel = ['linear','rbf','poly','sigmoid']
    return svm.SVR(kernel='poly',C=0.5,gamma=0.5)


def lgbm():
    # lightgbm = LGBMRegressor(objective='regression',
    #                          num_leaves=20,
    #                          learning_rate=0.1,
    #                          n_estimators=1000,
    #                          max_bin=200,
    #                          bagging_fraction=0.8,
    #                          bagging_freq=4,
    #                          bagging_seed=8,
    #                          feature_fraction=0.2,
    #                          feature_fraction_seed=8,
    #                          min_sum_hessian_in_leaf=11,
    #                          verbose=-1,
    #                          random_state=42)
    lightgbm = LGBMRegressor(objective='regression',
                             num_leaves=10,
                             learning_rate=0.1,
                             n_estimators=1000,
                             max_bin=200,
                             verbose=-1,
                             random_state=42)
    return  lightgbm

def mlp():
    mlp = MLPRegressor(hidden_layer_sizes=3000, max_iter=100)
    return mlp


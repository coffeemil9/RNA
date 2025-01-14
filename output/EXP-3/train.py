import sys
import shutil
import pandas as pd
import numpy as np
import re
import gc
import os
import math
import json
import importlib
import random
import warnings
import pickle
import joblib
from tqdm import tqdm
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
from xgboost import XGBRegressor, XGBClassifier
from sklearn.svm import SVR

import optuna
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import lightgbm as lgb
import catboost as cat, gc
import xgboost as xgb
print(f'lgb version:{lgb.__version__}')
print(f'catboost version:{cat.__version__}')
print(f'xgboost version:{xgb.__version__}')
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')
sys.path.append('/notebooks/CIBMTR/utils')
sys.path.append('/notebooks/CIBMTR/output')

import utils
import metric
from utils import clr
from metric import score

def get_model(model_name, 
              exp,
              cat_cols):
    params_path = f"/notebooks/CIBMTR/output/{exp}/params.json"
    with open(params_path, "r") as f:
        params = json.load(f)
        
    if model_name=='lgb':
        model = LGBMRegressor(**params, 
                              categorical_feature=cat_cols)
    elif model_name=='cat':
        model = CatBoostRegressor(**params, 
                                  cat_features=cat_cols)
    elif model_name == 'xgb':
        model = XGBRegressor(**params, 
                             enable_categorical=True)
    return model



def train(data,
          features,
          target_type,
          n_split, 
          cat_cols,
          model_name='cat',
          exp:str='experiment_num'):
    
    SAVE_PATH =f"/notebooks/CIBMTR/output/{exp}"
    oof_preds = np.zeros(len(data))
    for fold in range(n_split):
        print('')
        print(clr.GREEN+'#'*25+clr.END)
        print(clr.GREEN+f'### Fold {fold+1}'+clr.END)
        print(clr.GREEN+'#'*25+clr.END)
        print('')
        
        """ Dataset """
        X_train = data[data["fold"] != fold][features]
        X_val = data[data["fold"] == fold][features]
        if target_type=='target1':
            y_train = data[data["fold"] != fold]['target1']
            y_val = data[data["fold"] == fold]['target1']

        # Cox loss Model
        elif target_type=='target4':
            y_train = data[data["fold"] != fold]['target4']
            y_val = data[data["fold"] == fold]['target4']

        val_idx = data[data["fold"] == fold].index
        model = get_model(model_name=model_name,
                         exp=exp,
                         cat_cols=cat_cols)
        '''
        lightgbm
        '''
        if model_name == 'lgb':
            early_stopping_callback = lgb.early_stopping(400, first_metric_only=True, verbose=False)
            verbose_callback = lgb.log_evaluation(300)

            model.fit(X_train, y_train,
                      eval_metric='rmse',
                      eval_set=[(X_val, y_val)], 
                      callbacks=[early_stopping_callback, verbose_callback])

            oof_preds[val_idx] = model.predict(X_val)
            model.booster_.save_model(f'{SAVE_PATH}/LGB_{exp}_f{fold}.txt')

        '''
        catboost
        '''
        if model_name == 'cat':
            model.fit(X_train,
                      y_train,
                      eval_set=[(X_val, y_val)],
                      verbose=400, # log300毎に表示
                      early_stopping_rounds=400)

            oof_preds[val_idx] = model.predict(X_val)
            joblib.dump(model, f'{SAVE_PATH}/CAT_{exp}_f{fold}.pkl')

        '''
        xgboost
        '''
        if model_name == 'xgb':
            eval_set = [(X_train, y_train), (X_val, y_val)]
            model.fit(X_train, y_train, 
                      eval_set=eval_set, 
                      early_stopping_rounds=400,
                      verbose=400)
            oof_preds[val_idx] = model.predict(X_val)
            joblib.dump(model, f'{SAVE_PATH}/XGB_{exp}_f{fold}.pkl')
        
        ##
        # Calculate Score
        ##
        y_true_fold = data.iloc[val_idx][['ID', 'efs', 'efs_time', 'race_group']].copy()
        y_pred_fold = data.iloc[val_idx][['ID']].copy()
        y_pred_fold['prediction'] = oof_preds[val_idx]
        loss_val = score(y_true_fold, y_pred_fold, 'ID')
        print(clr.RED + f'Overall fold-{fold} C-index : {loss_val:.3}' + clr.END)
        
        del X_train, X_val, y_train, y_val, model
        gc.collect()
    joblib.dump(oof_preds, f'{SAVE_PATH}/oof_preds_{exp}.pkl')
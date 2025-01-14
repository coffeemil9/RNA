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
import warnings
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
sys.path.append('/notebooks/CIBMTR/train')

import config
import utils
import metric
from utils import clr
from metric import score


def save_best_params(trial, exp, fixed_params):
    
    """Best parametersをJSON形式で保存し、フォルダが存在しない場合は作成"""
    path = f'/notebooks/CIBMTR/output/{exp}'
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory {path} created.")
    
    # ファイルパスを作成
    full_params = {**fixed_params, **trial.params}
    params_path = os.path.join(path, "params.json")
    
    # JSONファイルにベストパラメータを保存
    with open(params_path, "w") as f:
        json.dump(full_params, f, indent=4)
    print(f"Best parameters saved to {params_path}")


def get_params(trial,
               target_type,
               model_name,
               seed=2025):
    # model_name: cat, lgb, xgb
    if model_name=='cat' and target_type=='target4':
        fixed_params = {'loss_function': 'Cox',
                        'grow_policy': 'Depthwise',
                        'random_seed' : seed,
                        'verbose': 500}
        trial_params = {
            'n_estimators': trial.suggest_int('n_estimators', 800, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.09, log=True), 
            'max_depth': trial.suggest_int('max_depth', 5, 8),
            'task_type': 'CPU',
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.05, 0.9, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 6, 15),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.05, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.01, 0.9, log=True),
        }
        full_params = {**fixed_params, **trial_params}
        
    return full_params, fixed_params

def objective(trial, 
              data,
              features,
              target_type,
              n_split, 
              cat_cols,
              cv_fold_func=np.average,
              model_name='cat',
              seed=2025):
    
    params, _ = get_params(trial,
                          target_type, 
                          model_name,
                          seed)

    
    """ Fold Iter """
    best_loss_score = []
    oof_preds = np.zeros(len(data))
    print(f'target_type:{target_type}')
    for fold in range(n_split):
        
        """ Dataset """
        X_train = data[data["fold"] != fold][features]
        X_val = data[data["fold"] == fold][features]
        
        if target_type=='target1':
            Y_train = data[data["fold"] != fold]['target1']
            Y_val = data[data["fold"] == fold]['target1']
            
        # Cox loss Model
        elif target_type=='target4':
            Y_train = data[data["fold"] != fold]['target4']
            Y_val = data[data["fold"] == fold]['target4']
        
        val_idx = data[data["fold"] == fold].index
        
        """ Train Model """
        if model_name=='lgb':
            model = LGBMRegressor(**param, 
                                  categorical_feature=cat_cols)
            model.fit(X_train,
                      Y_train,
                      eval_metric='rmse',
                      eval_set=[(X_val, Y_val)],
                      callbacks=[lgb.early_stopping(stopping_rounds=400, verbose=True),
                                 lgb.log_evaluation(period=400)], # Logの出力 100毎に出力
                     )
            
        elif model_name=='cat':
            model = CatBoostRegressor(**params, cat_features=cat_cols)
            model.fit(X_train,
                      Y_train,
                      eval_set=[(X_val, Y_val)],
                      early_stopping_rounds=400)
            
        elif model_name == 'xgb':
            model = XGBRegressor(**param, 
                                 enable_categorical=True,
                                 tree_method='gpu_hist')
            model.fit(X_train,
                      Y_train,
                      eval_set=[(X_train, Y_train), (X_val, Y_val)],
                      early_stopping_rounds=400,
                      verbose=400)

        """ Eval """
        preds = model.predict(X_val)
        oof_preds[val_idx] = preds
        y_true_fold = data.iloc[val_idx][['ID', 'efs', 'efs_time', 'race_group']].copy()
        y_pred_fold = data.iloc[val_idx][['ID']].copy()
        y_pred_fold['prediction'] = preds
        loss_val = score(y_true_fold, y_pred_fold, 'ID')
        best_loss_score.append(loss_val)
        
        print(clr.RED + f'Overall fold-{fold} C-index : {loss_val:.3}' + clr.END)
        
        del X_train, X_val, Y_train, Y_val, model, preds
        gc.collect()

    # save_trial_params(trial.number, param, score)
    """ Finaly Logging """
    print('-'*70)
    print(f"-[INFO] Trial OOF(Log Loss): {np.mean(best_loss_score)}")
    print(f"-[INFO] Trial All Fold(Log Loss): {best_loss_score}")
    print('-'*70)
    return cv_fold_func(best_loss_score)

def optimize_params(data, 
                    features, 
                    target_type, 
                    n_split, 
                    cat_cols, 
                    cv_fold_func=np.average, 
                    model_name='cat',
                    n_trials=1,
                    exp:str='experiment_number',
                    seed=2025):
    study = optuna.create_study(direction='maximize', 
                                study_name='Optimize boosting hyperparameters')
    study.optimize(lambda trial: objective(trial, data, features, target_type, n_split, cat_cols, cv_fold_func, model_name, seed), 
                   n_trials=n_trials)
    
    trial = study.best_trial
    print('-' * 70)
    print(f"- Best Trial OOF: {trial.value}")
    print('Best trial parameters:')
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    _, fixed_params = get_params(trial, target_type, model_name, seed)
    save_best_params(trial, exp, fixed_params)
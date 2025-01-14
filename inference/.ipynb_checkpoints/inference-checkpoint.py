import numpy as np
import pandas as pd
import os
import sys
import lightgbm as lgb
import catboost as cat, gc
import xgboost as xgb
import importlib
import importlib.util
import joblib
import warnings
from scipy.stats import rankdata 
print(lgb.__version__)
print(cat.__version__)
print(xgb.__version__)


warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

sys.path.append('/kaggle/input/utililies')
import utils as my_utils
from utils import clr

def export_config(exp_list:list=[]):
    config_modules = {}
    if len(exp_list) == 0:
        print('[ALERT] The exp_list is empty. Please provide valid experiment names.')
        return config_modules
    
    for exp in exp_list:
        try:
            folder_path = f'/kaggle/input/tree-{exp.lower()}'
            sys.path.insert(0, folder_path)
            module_name = f'config_{exp.lower()}'
            
            spec = importlib.util.spec_from_file_location(module_name, f'{folder_path}/config.py')
            if spec is None:
                raise FileNotFoundError(f"[ALERT] config.py not found in {folder_path}")
            
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            config_modules[module_name] = config_module
            sys.path.pop(0)
        except Exception as e:
            print(f"[ALERT] Failed to load config for {exp}: {e}")
            sys.path.pop(0)    
    return config_modules

def get_model(exp:str, 
              model_name:str, 
              fold:int):
    
    # MODEL_PATH = /kaggle/input/tree-exp-1/CAT_EXP-1_f0.pkl
    MODEL_NAME = model_name
    EXP_NUM = exp
    MODEL_PATH = f'/kaggle/input/tree-{EXP_NUM.lower()}'
    if model_name == 'lgb':
        with open(f'{MODEL_PATH}/LGB_{EXP_NUM}_f{fold}.txt', 'r', encoding='utf-8') as f:
            model_str = f.read()
        model = lgb.Booster(model_str=model_str)
        
    elif model_name == 'cat':
        model = joblib.load(f'{MODEL_PATH}/CAT_{EXP_NUM}_f{fold}.pkl')
        
    elif model_name == 'xgb':
        model = joblib.load(f'{MODEL_PATH}/XGB_{EXP_NUM}_f{fold}.pkl')
        
    return model

def inference(test_df,
              exp_list:list=[]):
    # exp_list:['EXP-1']
    module_list = export_config(exp_list)
    preds = np.zeros((len(exp_list), len(test_df)))
    for i, exp in enumerate(exp_list):

        module = module_list[f'config_{exp.lower()}']
        n_splits = module.general.n_splits
        model_name = module.training.model_name
        features = joblib.load(f'/kaggle/input/tree-{exp.lower()}/features.pkl')
        categorical_cols = joblib.load(f'/kaggle/input/tree-{exp.lower()}/categorical_cols.pkl')

        data = test_df[features]
        data[categorical_cols] = data[categorical_cols].astype(str).astype('category')

        pred_list = []
        print(clr.GREEN + f'{exp} : {model_name}' + clr.END)
        for fold in range(n_splits):
            print(clr.RED + '*'*10 + clr.END)
            print(clr.RED + f'* fold {fold}' + clr.END)
            print(clr.RED + '*'*10 + clr.END)
            model = get_model(exp=exp,
                              model_name=model_name,
                              fold=fold)
            pred = model.predict(data)
            pred_list.append(pred)
            print(pred)

        preds[i, :] = np.mean(np.array(pred_list), axis=0)
        print()
    print(preds)
    return preds
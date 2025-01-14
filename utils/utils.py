import re
import joblib
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, LabelEncoder, StandardScaler, QuantileTransformer, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer, HashingVectorizer
from itertools import combinations
from lifelines import KaplanMeierFitter
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

class CIBMTR_FE:
    def __init__(self, mode='train'):
        self.mode = mode
        self.TARGET = ['efs', 'efs_time']
        self.AGGREGATIONS = ['count', 'mean', 'min', 'max', 'first', 'last', self.q1, 'median', self.q3, 'sum']
    
    def q1(self, x):
        return x.quantile(0.25)
    def q3(self, x):
        return x.quantile(0.75)


    ###########################
    # Feature Engineering Fanctions
    ###########################
    
    def fe1(self, df):
        print(f'fe1 Done! DATA Shape:{df.shape}')
        return df
    
    def fe2(self, df):        
        print(f'fe2 Done! DATA Shape:{df.shape}')
        return df
    
    def fe3(self, df):
        print(f'fe3 Done! DATA Shape:{df.shape}')
        return df

    def fe4(self, df, train=None):
        print(f'fe4 Done! DATA Shape:{df.shape}')
        return df
    
    def fe5(self, df):
        print(f'fe5 Done! DATA Shape:{df.shape}')
        return df
    
    ######################
    # DROP COLS FUNCTIONS
    ######################
    def drop_cols(self, df, game_drop=False):

        return df
    
    ##################
    # Trainsform Object to Categorical
    ##################
    def trans_obj_cate(self, df):
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.to_list()
        for col in categorical_cols:
            df[col] = df[col].astype('category')
        self.categorical_cols = categorical_cols
        return df, categorical_cols
    
    
    ###################
    # Functions For NN Model
    ###################
    def one_hot_encoder(self, train_df, test_df, test_flag=False):
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        encoder.fit(train_df[self.encode_features])
        
        if test_flag:
            # Test Transform
            encoded_data_test = encoder.transform(test_df[self.encode_features])  # testデータに適用
            encoded_df_test = pd.DataFrame(encoded_data_test, columns=encoder.get_feature_names_out(self.encode_features))
            test_data = pd.concat([test_df, encoded_df_test], axis=1)
            
            return test_data
        else:
            encoded_data_train = encoder.transform(train_df[self.encode_features])  # trainデータに適用
            encoded_df_train = pd.DataFrame(encoded_data_train, columns=encoder.get_feature_names_out(self.encode_features))
            train_df = pd.concat([train_df, encoded_df_train], axis=1)
            
            return train_df
        
    def robust_scaler(self, df):
        exclude_cols = self.targets + ['Id'] + ['fold']
        numerical_cols = df.select_dtypes(include=['number']).columns
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        # Inf Columns
        inf_columns = df[numerical_cols].columns[np.isinf(df[numerical_cols]).any()].tolist()
        if inf_columns:
            raise ValueError(f"Dataframe contains 'inf' values in the following columns: {inf_columns}. Please handle them before scaling.")
            
        if self.mode=='train':
            scaler = RobustScaler()
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
            joblib.dump(scaler, 'UM_utilities/robust_scaler.pkl')
        else:
            path = f'/kaggle/input/um-utils/rubust_scaler.pkl'
            scaler = joblib.load(path)
            df[numerical_cols] = scaler.transform(df[numerical_cols])
        return df
    
    def standard_scaler(self, df, exp_num=None):
        exclude_cols = self.targets + ['Id'] + ['fold']
        numerical_cols = df.select_dtypes(include=['number']).columns
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        # Inf Columns
        inf_columns = df[numerical_cols].columns[np.isinf(df[numerical_cols]).any()].tolist()
        if inf_columns:
            raise ValueError(f"Dataframe contains 'inf' values in the following columns: {inf_columns}. Please handle them before scaling.")
            
        if self.mode=='train':
            scaler = StandardScaler()
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
            joblib.dump(scaler, 'UM_utilities/standard_scaler.pkl')
        else:
            
            path = f'/kaggle/input/{exp_num}/standard_scaler.pkl'
            scaler = joblib.load(path)
            df[numerical_cols] = scaler.transform(df[numerical_cols])
        return df
    
    def min_max_scaler(self, df, exp_num=None):
        exclude_cols = self.targets + ['Id'] + ['fold']
        numerical_cols = df.select_dtypes(include=['number']).columns
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        # Inf Columns
        inf_columns = df[numerical_cols].columns[np.isinf(df[numerical_cols]).any()].tolist()
        if inf_columns:
            raise ValueError(f"Dataframe contains 'inf' values in the following columns: {inf_columns}. Please handle them before scaling.")
            
        if self.mode=='train':
            scaler = MinMaxScaler()
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
            joblib.dump(scaler, 'UM_utilities/min_max_scaler.pkl')
        else:
            path = f'/kaggle/input/{exp_num}/min_max_scaler.pkl'
            scaler = joblib.load(path)
            df[numerical_cols] = scaler.transform(df[numerical_cols])
        return df
    
    def quant_trans(self, df):
        exclude_cols = self.targets + ['Id'] + ['fold']
        numerical_cols = df.select_dtypes(include=['number']).columns
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        # Inf Columns
        inf_columns = df[numerical_cols].columns[np.isinf(df[numerical_cols]).any()].tolist()
        if inf_columns:
            raise ValueError(f"Dataframe contains 'inf' values in the following columns: {inf_columns}. Please handle them before scaling.")
            
        if self.mode=='train':
            scaler = QuantileTransformer()
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
            joblib.dump(scaler, 'UM_utilities/standard_scaler.pkl')
        else:
            path = f'/kaggle/input/{exp_num}/quant_scaler.pkl'
            scaler = joblib.load(path)
            df[numerical_cols] = scaler.transform(df[numerical_cols])
        return df
    
    def label_encoder(self, df, exp_num=None):
        exclude_cols = self.targets + ['Id'] + ['fold']
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        categorical_cols = [col for col in categorical_cols if col not in exclude_cols]

        # NaN値の処理（Label EncoderはNaNを扱えないため、一時的に文字列に変換）
        for col in categorical_cols:
            df[col] = df[col].cat.add_categories('missing').fillna('missing')

        if self.mode == 'train':
            encoders = {}
            for col in categorical_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                encoders[col] = le
            joblib.dump(encoders, 'UM_utilities/label_encoder.pkl')
        else:
            path = f'/kaggle/input/{exp_num}/label_encoder.pkl'
            encoders = joblib.load(path)
            for col in categorical_cols:
                if col in encoders:
                    df[col] = encoders[col].transform(df[col])
                else:
                    raise ValueError(f"Column {col} was not found in the encoder. Please check the data.")
        return df
    
    def clip_outliers(self, df, exp_num=None):
        exclude_cols = self.targets + ['Id', 'fold']
        numerical_cols = df.select_dtypes(include=['number']).columns
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        if self.mode == 'train':
            lowerbound, upperbound = np.percentile(df[numerical_cols], [1, 99], axis=0)
            bounds = {'lowerbound': lowerbound, 'upperbound': upperbound}
            joblib.dump(bounds, 'UM_utilities/bounds.pkl')
            
        elif self.mode == 'test':
            path = f'/kaggle/input/{exp_num}/bounds.pkl'
            if os.path.exists(path):
                bounds = joblib.load(path)
                lowerbound = bounds['lowerbound']
                upperbound = bounds['upperbound']
            else:
                raise FileNotFoundError(f"Saved bounds file not found at {self.save_path}")
        else:
            raise ValueError("Mode must be either 'train' or 'test'")
        
        df[numerical_cols] = np.clip(df[numerical_cols], lowerbound, upperbound)
        
        return df
        
    def log_trans(self, df):
        exclude_cols = self.targets + ['Id'] + ['fold']
        numerical_cols = df.select_dtypes(include=['number']).columns
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        df[numerical_cols] = np.log(df[numerical_cols]+1)
        return df
    
    ###
    # Create Targets
    ###
    def create_target1(self, data, cat_cols):
        cph_data = pd.get_dummies(data, columns=cat_cols, drop_first=True)
        cph = CoxPHFitter(penalizer=self.penalizer)
        cph.fit(cph_data, duration_col='efs_time', event_col='efs')
        data['target1'] = cph.predict_partial_hazard(cph_data)
        return data

    def create_target2(self, data):
        kmf = KaplanMeierFitter()
        kmf.fit(durations=data['efs_time'], event_observed=data['efs'])
        data['target2'] = kmf.survival_function_at_times(data['efs_time']).values
        return data

    def create_target3(self, data):
        naf = NelsonAalenFitter()
        naf.fit(durations=data['efs_time'], event_observed=data['efs'])
        data['target3'] = naf.cumulative_hazard_at_times(data['efs_time']).values
        data['target3'] = data['target3'] * -1
        return data

    def create_target4(self, data):
        data['target4'] = data.efs_time.copy()
        data.loc[data.efs == 0, 'target4'] *= -1
        return data

    
    

class clr:
    BLACK     = '\033[30m'
    RED       = '\033[31m'
    GREEN     = '\033[32m'
    YELLOW    = '\033[33m'
    BLUE      = '\033[34m'
    PURPLE    = '\033[35m'
    CYAN      = '\033[36m'
    WHITE     = '\033[37m'
    END       = '\033[0m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'
    INVISIBLE = '\033[08m'
    REVERCE   = '\033[07m'
    S = '\033[1m' + '\033[95m'
    E = '\033[0m'

my_colors = ["#761D80", "#9926A6", "#9C69C9",
             "#6C91BF", "#58BCC6", "#4AD1B2",
             "#4BF1B2"]

CMAP1 = ListedColormap(my_colors)
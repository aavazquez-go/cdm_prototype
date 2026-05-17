import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

import torch
import scipy.integrate

if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson


SEED = 1234
torch.manual_seed(123)
np.random.seed(SEED)


def load_datasets(train_path: str, test_path: str):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    return df_train, df_test


def load_single_dataset(path: str, test_frac: float = 0.2, val_frac: float = 0.2):
    df = pd.read_csv(path)
    df_test = df.sample(frac=test_frac)
    df_train = df.drop(df_test.index)
    df_val = df_train.sample(frac=val_frac)
    df_train = df_train.drop(df_val.index)
    print(f"Train: {df_train.shape}, Val: {df_val.shape}, Test: {df_test.shape}")
    return df_train, df_val, df_test


def make_preprocessor(df_train: pd.DataFrame, exclude_cols: list[str], verbose_names: bool = False):
    cols = [c for c in df_train.columns.tolist() if c not in exclude_cols]
    preprocessor = ColumnTransformer(
        transformers=[('num', StandardScaler(), cols)],
        remainder='passthrough',
        verbose_feature_names_out=verbose_names
    )
    return preprocessor


def preprocess_data(preprocessor, df_train, df_val, df_test):
    x_train = preprocessor.fit_transform(df_train).astype('float32')
    x_val = preprocessor.transform(df_val).astype('float32')
    x_test = preprocessor.transform(df_test).astype('float32')
    return x_train, x_val, x_test


def get_target(df, time_col='time', event_col='event'):
    return (df[time_col].values, df[event_col].values)


def get_target_cc(df):
    return (df['Stop'].values, df['Event'].values)

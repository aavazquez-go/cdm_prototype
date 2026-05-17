import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

import torch
import torchtuples as tt

from pycox.models import DeepHitSingle
from pycox.evaluation import EvalSurv

import sys
sys.path.append('..')
from scripts.utils import *


DATA_PATH = './datasets/arhv_prepro_5.0.csv'
MODEL_DIR = './notebooks/'
BATCH_SIZE = 256
EPOCHS = 100
LR = 0.01
NUM_DURATIONS = 10
ALPHA = 0.2
SIGMA = 0.1
TEST_FRAC = 0.2
VAL_FRAC = 0.2


def main():
    df_train, df_val, df_test = load_single_dataset(DATA_PATH, TEST_FRAC, VAL_FRAC)

    exclude_cols = ['event', 'time']
    preprocessor = make_preprocessor(df_train, exclude_cols)
    x_train, x_val, x_test = preprocess_data(preprocessor, df_train, df_val, df_test)

    labtrans = DeepHitSingle.label_transform(NUM_DURATIONS)
    get_target = lambda df: (df['time'].values, df['event'].values)
    y_train = labtrans.fit_transform(*get_target(df_train))
    y_val = labtrans.transform(*get_target(df_val))
    durations_test, events_test = get_target(df_test)
    val = (x_val, y_val)

    in_features = x_train.shape[1]
    num_nodes = [32, 32]
    out_features = labtrans.out_features
    batch_norm = True
    dropout = 0.1

    net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)
    model = DeepHitSingle(net, tt.optim.Adam, alpha=ALPHA, sigma=SIGMA, duration_index=labtrans.cuts)

    lr_finder = model.lr_finder(x_train, y_train, BATCH_SIZE, tolerance=3)
    model.optimizer.set_lr(LR)

    callbacks = [tt.callbacks.EarlyStopping()]
    log = model.fit(x_train, y_train, BATCH_SIZE, EPOCHS, callbacks, val_data=val)

    surv = model.interpolate(10).predict_surv_df(x_test)

    ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
    cindex = ev.concordance_td()
    print(f"Concordance index: {cindex}")

    time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)
    ibs = ev.integrated_brier_score(time_grid)
    inbll = ev.integrated_nbll(time_grid)
    print(f"Integrated Brier Score: {ibs}")
    print(f"Integrated NBLL: {inbll}")

    model.save_model_weights(MODEL_DIR + 'deephit_model_weights.pt')
    model.save_net(MODEL_DIR + 'deephit_net.pt')
    import joblib
    joblib.dump(preprocessor, MODEL_DIR + 'deephit_preprocessor.pkl')
    joblib.dump(labtrans, MODEL_DIR + 'deephit_labtrans.pkl')

    print("DeepHit training complete.")


if __name__ == '__main__':
    main()

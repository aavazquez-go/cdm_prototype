import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

import torch
import torchtuples as tt

from pycox.models import CoxTime
from pycox.models.cox_time import MLPVanillaCoxTime
from pycox.evaluation import EvalSurv
import joblib

import sys
sys.path.append('..')
from scripts.utils import *


DATA_PATH = './datasets/arhv_prepro_5.0.csv'
MODEL_DIR = './notebooks/cox_time_models/'
BATCH_SIZE = 256
EPOCHS = 512
LR = 0.01
TEST_FRAC = 0.2
VAL_FRAC = 0.2


def main():
    df_train, df_val, df_test = load_single_dataset(DATA_PATH, TEST_FRAC, VAL_FRAC)

    exclude_cols = ['event', 'time']
    preprocessor = make_preprocessor(df_train, exclude_cols)
    x_train, x_val, x_test = preprocess_data(preprocessor, df_train, df_val, df_test)

    labtrans = CoxTime.label_transform()
    get_target = lambda df: (df['time'].values.astype("float32"), df['event'].values.astype("float32"))
    y_train = get_target(df_train)
    y_val = get_target(df_val)
    durations_test, events_test = get_target(df_test)
    val = tt.tuplefy(x_val, y_val)

    in_features = x_train.shape[1]
    num_nodes = [32, 32]
    batch_norm = True
    dropout = 0.1

    net = MLPVanillaCoxTime(in_features, num_nodes, batch_norm, dropout)
    model = CoxTime(net, tt.optim.Adam, labtrans=labtrans)

    lrfinder = model.lr_finder(x_train, y_train, BATCH_SIZE, tolerance=2)
    model.optimizer.set_lr(LR)

    callbacks = [tt.callbacks.EarlyStopping()]
    log = model.fit(x_train, y_train, BATCH_SIZE, EPOCHS, callbacks, True, val_data=val.repeat(10).cat())

    labtrans.fit(*y_train)
    _ = model.compute_baseline_hazards()
    surv = model.predict_surv_df(x_test)
    surv.index = np.sort(np.unique(df_train['time'].values))

    ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
    cindex = ev.concordance_td()
    print(f"Concordance index: {cindex}")

    time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)
    ibs = ev.integrated_brier_score(time_grid)
    inbll = ev.integrated_nbll(time_grid)
    print(f"Integrated Brier Score: {ibs}")
    print(f"Integrated NBLL: {inbll}")

    joblib.dump(ev, MODEL_DIR + 'cox_time_evaluation.joblib')
    joblib.dump(preprocessor, MODEL_DIR + 'cox_time_preprocessor.joblib')
    joblib.dump(labtrans, MODEL_DIR + 'cox_time_labtrans.joblib')
    model.save_model_weights(MODEL_DIR + 'cox_time_model_weights.pt')
    model.save_net(MODEL_DIR + 'cox_time_net.pt')
    joblib.dump(model.baseline_hazards_, MODEL_DIR + 'cox_time_baseline_hazards.joblib')

    print("Cox-Time training complete. Models saved to", MODEL_DIR)


if __name__ == '__main__':
    main()

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

import torch
import torchtuples as tt

from pycox.models import CoxCC
from pycox.evaluation import EvalSurv
import joblib

import sys
sys.path.append('..')
from scripts.utils import *


TRAIN_PATH = './datasets/train_set.csv'
TEST_PATH = './datasets/test_set.csv'
MODEL_DIR = './notebooks/cox_cc_models/'
BATCH_SIZE = 256
EPOCHS = 512
LR = 0.01


def main():
    df_train, df_test = load_datasets(TRAIN_PATH, TEST_PATH)

    df_val = df_train.sample(frac=0.2)
    df_train = df_train.drop(df_val.index)

    exclude_cols = ['Event', 'Start', 'Stop']
    preprocessor = make_preprocessor(df_train, exclude_cols)
    x_train, x_val, x_test = preprocess_data(preprocessor, df_train, df_val, df_test)

    get_target = lambda df: (df['Stop'].values, df['Event'].values)
    y_train = get_target(df_train)
    y_val = get_target(df_val)
    durations_test, events_test = get_target(df_test)
    val = tt.tuplefy(x_val, y_val)

    in_features = x_train.shape[1]
    num_nodes = [32, 32]
    out_features = 1
    batch_norm = True
    dropout = 0.1
    output_bias = False

    net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout, output_bias=output_bias)
    model = CoxCC(net, tt.optim.Adam)

    lrfinder = model.lr_finder(x_train, y_train, BATCH_SIZE, tolerance=2)
    model.optimizer.set_lr(LR)

    callbacks = [tt.callbacks.EarlyStopping()]
    log = model.fit(x_train, y_train, BATCH_SIZE, EPOCHS, callbacks, True, val_data=val.repeat(10).cat())

    _ = model.compute_baseline_hazards()
    surv = model.predict_surv_df(x_test)

    ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
    cindex = ev.concordance_td()
    print(f"Concordance index: {cindex}")

    time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)
    ibs = ev.integrated_brier_score(time_grid)
    inbll = ev.integrated_nbll(time_grid)
    print(f"Integrated Brier Score: {ibs}")
    print(f"Integrated NBLL: {inbll}")

    joblib.dump(ev, MODEL_DIR + 'coxcc_evaluation.joblib')
    joblib.dump(preprocessor, MODEL_DIR + 'cox_cc_preprocessor.pkl')
    torch.save(model.net.state_dict(), MODEL_DIR + 'cox_cc_modelo_weights.pt')
    joblib.dump(model.baseline_hazards_, MODEL_DIR + 'cox_cc_baseline_hazards.joblib')

    print("Cox-CC training complete. Models saved to", MODEL_DIR)


if __name__ == '__main__':
    main()

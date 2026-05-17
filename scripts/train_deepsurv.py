import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

import torch
import torchtuples as tt

from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
import joblib

import sys
sys.path.append('..')
from scripts.utils import *


DATA_PATH = './datasets/arhv_prepro_5.0.csv'
MODEL_DIR = './notebooks/deepsurv_models/'
BATCH_SIZE = 256
EPOCHS = 512
TEST_FRAC = 0.2
VAL_FRAC = 0.2


def main():
    df_train, df_val, df_test = load_single_dataset(DATA_PATH, TEST_FRAC, VAL_FRAC)

    exclude_cols = ['event', 'time']
    preprocessor = make_preprocessor(df_train, exclude_cols)
    x_train, x_val, x_test = preprocess_data(preprocessor, df_train, df_val, df_test)

    get_target = lambda df: (df['time'].values, df['event'].values)
    y_train = get_target(df_train)
    y_val = get_target(df_val)
    durations_test, events_test = get_target(df_test)
    val = (x_val, y_val)

    in_features = x_train.shape[1]
    num_nodes = [32, 32]
    out_features = 1
    batch_norm = True
    dropout = 0.1
    output_bias = False

    net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout, output_bias=output_bias)
    model = CoxPH(net, tt.optim.Adam)

    lrfinder = model.lr_finder(x_train, y_train, BATCH_SIZE, tolerance=10)
    best_lr = lrfinder.get_best_lr()
    model.optimizer.set_lr(best_lr)

    callbacks = [tt.callbacks.EarlyStopping()]
    log = model.fit(x_train, y_train, BATCH_SIZE, EPOCHS, callbacks, True, val_data=val, val_batch_size=BATCH_SIZE)

    model.partial_log_likelihood(*val).mean()
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

    joblib.dump(ev, MODEL_DIR + 'deepsurv_evaluation.joblib')
    torch.save(model.net.state_dict(), MODEL_DIR + 'deepsurv_model_weights.pt')
    joblib.dump(preprocessor, MODEL_DIR + 'deepsurv_preprocessor.pkl')
    joblib.dump(model.baseline_hazards_, MODEL_DIR + 'deepsurv_baseline_hazards.joblib')

    print("DeepSurv training complete. Models saved to", MODEL_DIR)


if __name__ == '__main__':
    main()

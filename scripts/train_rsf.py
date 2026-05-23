import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
import joblib

import sys
sys.path.append('..')
from scripts.utils import *


DATA_PATH = './datasets/arhv_prepro_5.0.csv'
MODEL_DIR = './notebooks/rsf_models/'
TEST_FRAC = 0.2
VAL_FRAC = 0.2
N_ESTIMATORS = 500
MAX_DEPTH = 10
MIN_SAMPLES_SPLIT = 10


def main():
    df_train, df_val, df_test = load_single_dataset(DATA_PATH, TEST_FRAC, VAL_FRAC)

    exclude_cols = ['event', 'time']
    preprocessor = make_preprocessor(df_train, exclude_cols)
    x_train, x_val, x_test = preprocess_data(preprocessor, df_train, df_val, df_test)

    y_train = Surv.from_arrays(df_train['event'].values.astype(bool), df_train['time'].values)
    y_val = Surv.from_arrays(df_val['event'].values.astype(bool), df_val['time'].values)
    y_test = Surv.from_arrays(df_test['event'].values.astype(bool), df_test['time'].values)

    rsf = RandomSurvivalForest(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        random_state=42,
        n_jobs=-1
    )
    rsf.fit(x_train, y_train)

    c_index = concordance_index_censored(
        y_test['event'], y_test['time'], rsf.predict(x_test)
    )[0]
    print(f"Concordance index: {c_index}")

    surv_array = rsf.predict_survival_function(x_test, return_array=True)
    surv_df = pd.DataFrame(surv_array.T, index=rsf.unique_times_)
    print(f"Survival curves: {surv_df.shape[0]} times, {surv_df.shape[1]} samples")

    joblib.dump(preprocessor, MODEL_DIR + 'rsf_preprocessor.pkl')
    joblib.dump(rsf, MODEL_DIR + 'rsf_model.joblib')
    joblib.dump(c_index, MODEL_DIR + 'rsf_evaluation.joblib')

    print("RSF training complete. Models saved to", MODEL_DIR)


if __name__ == '__main__':
    main()

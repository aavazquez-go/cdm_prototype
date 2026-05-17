import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter
import joblib

import sys
sys.path.append('..')
from scripts.utils import *


DATA_PATH = './datasets/arhv_prepro_5.0.csv'
MODEL_DIR = './notebooks/coxph_models/'
TEST_FRAC = 0.2
VAL_FRAC = 0.2


def ipcw_brier_score(model, X, event_times, event_observed, times,
                     train_event_times=None, train_event_observed=None):
    if train_event_times is None:
        train_event_times = event_times
        train_event_observed = event_observed

    censoring_event = 1 - train_event_observed
    km_censoring = KaplanMeierFitter()
    km_censoring.fit(train_event_times, event_observed=censoring_event)

    G_times = np.concatenate([times, event_times])
    G_times = np.unique(G_times)
    G_vals = km_censoring.survival_function_at_times(G_times).values
    G_dict = dict(zip(G_times, G_vals))

    def G(t):
        idx = np.searchsorted(G_times, t, side='right') - 1
        idx = np.clip(idx, 0, len(G_times) - 1)
        return G_dict[G_times[idx]]

    G_vec = np.vectorize(G)
    surv_funcs = model.predict_survival_function(X)
    S_pred = surv_funcs.loc[times].T.values
    n_samples = len(event_times)
    brier_scores = []

    for j, t in enumerate(times):
        event_mask = (event_times <= t) & (event_observed == 1)
        denom1 = G_vec(event_times[event_mask])
        denom1 = np.clip(denom1, 1e-8, None)
        term1 = np.sum(((0 - S_pred[event_mask, j]) ** 2) / denom1)

        censor_mask = (event_times > t)
        denom2 = G_vec(t)
        denom2 = np.clip(denom2, 1e-8, None)
        term2 = np.sum(((1 - S_pred[censor_mask, j]) ** 2) / denom2)

        brier = (term1 + term2) / n_samples
        brier_scores.append(brier)

    return np.array(brier_scores)


def ipcw_negative_log_likelihood(model, X, event_times, event_observed,
                                  train_event_times=None, train_event_observed=None):
    if train_event_times is None:
        train_event_times = event_times
        train_event_observed = event_observed

    censoring_event = 1 - train_event_observed
    km_censoring = KaplanMeierFitter()
    km_censoring.fit(train_event_times, event_observed=censoring_event)

    G_Ti = km_censoring.survival_function_at_times(event_times).values
    G_Ti = np.clip(G_Ti, 1e-8, None)

    linear_predictor = model.predict_log_partial_hazard(X).values
    partial_hazard = np.exp(linear_predictor)

    surv_funcs = model.predict_survival_function(X)
    S_Ti = np.array([np.interp(t, surv_funcs.index, surv_funcs.iloc[:, i]) for i, t in enumerate(event_times)])

    baseline_hazard = model.baseline_hazard_
    h0_Ti = np.array([
        baseline_hazard.loc[:t].iloc[-1, 0] if t >= baseline_hazard.index.min() else baseline_hazard.iloc[0, 0]
        for t in event_times
    ])

    h_Ti = h0_Ti * partial_hazard
    f_Ti = h_Ti * S_Ti

    log_likelihood = np.zeros_like(event_times, dtype=float)

    event_mask = (event_observed == 1)
    if np.any(event_mask):
        f_event = f_Ti[event_mask]
        f_event = np.clip(f_event, 1e-15, None)
        log_likelihood[event_mask] = np.log(f_event)

    censor_mask = (event_observed == 0)
    if np.any(censor_mask):
        S_censor = S_Ti[censor_mask]
        S_censor = np.clip(S_censor, 1e-15, None)
        log_likelihood[censor_mask] = np.log(S_censor)

    weighted_loglik = (log_likelihood / G_Ti)
    ipcw_nll = -np.mean(weighted_loglik)
    return ipcw_nll


def main():
    df_train, df_val, df_test = load_single_dataset(DATA_PATH, TEST_FRAC, VAL_FRAC)

    exclude_cols = ['event', 'time']
    preprocessor = make_preprocessor(df_train, exclude_cols, verbose_names=True)

    x_train = preprocessor.fit_transform(df_train).astype('float32')
    x_val = preprocessor.transform(df_val).astype('float32')
    x_test = preprocessor.transform(df_test).astype('float32')

    cols_after_transform = preprocessor.get_feature_names_out()
    base_cols = [c for c in df_train.columns if c not in exclude_cols]
    new_train_df = pd.DataFrame(x_train, columns=cols_after_transform)
    new_train_df['time'] = df_train['time'].values
    new_train_df['event'] = df_train['event'].values

    cph = CoxPHFitter()
    cph.fit(new_train_df, duration_col='time', event_col='event')
    cph.print_summary()

    X_test_df = pd.DataFrame(x_test, columns=cols_after_transform)
    X_test = X_test_df
    T_test = df_test['time'].values
    E_test = df_test['event'].values

    times = np.percentile(T_test[E_test == 1], np.linspace(10, 90, 9))
    brier_scores = ipcw_brier_score(
        model=cph, X=X_test, event_times=T_test, event_observed=E_test,
        times=times, train_event_times=df_train['time'].values,
        train_event_observed=df_train['event'].values
    )
    print("Brier scores at times:", times)
    print("Mean Brier score:", np.mean(brier_scores))

    nll = ipcw_negative_log_likelihood(
        model=cph, X=X_test, event_times=T_test, event_observed=E_test,
        train_event_times=df_train['time'].values,
        train_event_observed=df_train['event'].values
    )
    print(f"IPCW Negative Log-Likelihood: {nll:.4f}")

    joblib.dump(preprocessor, MODEL_DIR + 'preprocessor.joblib')
    joblib.dump(cph, MODEL_DIR + 'coxph_model.joblib')

    print("CoxPH training complete. Models saved to", MODEL_DIR)


if __name__ == '__main__':
    main()

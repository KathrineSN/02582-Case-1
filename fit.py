from __future__ import annotations
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from pelutils import log, TT

def predict(model, x):
    y = model.predict(normalize(x))
    y[y<0] = 0
    y[y>1.3] = 1.3
    return y

def normalize(x):
    x = x.copy()
    for col in x.columns:
        if x.dtypes[col] != np.uint8 and x.dtypes[col] != bool:
            x[col] = (x[col].values - x[col].values.mean()) / (x[col].values.std()+1e6)
    return x

def error(y: np.ndarray, y_hat: np.ndarray) -> float:
    non_zero = y > 0
    y, y_hat = y[non_zero], y_hat[non_zero]
    dev = 1 - y_hat / y
    return np.abs(dev).mean()

def fit(x: pd.DataFrame, y: np.ndarray, models: list, num_splits: int) -> list[tuple]:
    # Returns a list of tuples (model, weight, mean total accuracy)
    # Page 175 in ML book
    N = len(x)
    S = len(models)
    log.section("Fitting %i models using %i splits" % (S, num_splits))

    E_val = np.empty((S, num_splits))
    for i, (train_idx, val_idx) in enumerate(KFold(n_splits=num_splits, shuffle=True).split(x)):
        log("Split %i / %i" % (i+1, num_splits))
        x_train, y_train = x.iloc[train_idx], y[train_idx]
        x_val, y_val = x.iloc[val_idx], y[val_idx]
        for s in range(S):
            log.debug("Fitting model %i / %i: %s" % (s+1, S, models[s]))
            with TT.profile("Fit %s" % models[s]):
                models[s].fit(x_train, y_train)
            y_hat = predict(models[s], x_val)
            E_val[s, i] = error(y_val, y_hat)

    best_models = E_val.argmin(axis=0)
    model_idcs, counts = np.unique(best_models, return_counts=True)
    return [(models[s], c/num_splits, E_val[s].mean()) for s, c in zip(model_idcs, counts)]

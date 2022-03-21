from __future__ import annotations
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from pelutils import log, TT, Levels
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')


def fit_model(model, x, y):
    x = x.copy()

    means = dict()
    stds = dict()
    for col in x.columns:
        if x.dtypes[col] != np.uint8 and x.dtypes[col] != bool:
            means[col] = x[col].mean()
            stds[col] = x[col].std() + 1e-6
            x[col] = (x[col]-means[col]) / stds[col]

    model.fit(x, y)

    best_bias = 0
    best_bias_error = float("inf")

    for b in np.linspace(0, 0.1, 11):
        y_hat = predict(model, x, (b, dict(), dict()))
        e = error(y, y_hat)
        if e < best_bias_error:
            best_bias = b
            best_bias_error = e

    return best_bias, means, stds

def predict(model, x, metadata):
    x = x.copy()
    for col in x.columns:
        if col in metadata[1]:
            x[col] = (x[col]-metadata[1][col]) / metadata[2][col]
    y = model.predict(x) - metadata[0]
    y[y<0] = 0
    y[y>1.3] = 1.3
    return y

def normalize(x):
    x = x.copy()
    for col in x.columns:
        if x.dtypes[col] != np.uint8 and x.dtypes[col] != bool:
            x[col] = (x[col].values - x[col].values.mean()) / (x[col].values.std()+1e-6)
    return x

def error(y: np.ndarray, y_hat: np.ndarray) -> float:
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
        metadatas = list()
        for s in range(S):
            log.debug("Fitting model %i / %i: %s" % (s+1, S, models[s]))
            with TT.profile("%s" % models[s]):
                b, means, stds = fit_model(models[s], x_train, y_train)
            metadatas.append((b, means, stds))
            y_hat = predict(models[s], x_val, metadatas[-1])
            E_val[s, i] = error(y_val, y_hat)

    best_models = E_val.argmin(axis=0)
    weights = np.zeros(S)
    model_idcs, counts = np.unique(best_models, return_counts=True)
    return [(models[s], c/num_splits, E_val[s].mean()) for s, c in zip(model_idcs, counts)]

def feature_importance(X,y):
    num_train = int(0.2*len(X))
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)
    model = RandomForestRegressor(n_estimators=20, n_jobs=-1)
    model.fit(X.iloc[:num_train], y[:num_train]) 
    feature_importances = model.feature_importances_
    forest_importances = pd.Series(feature_importances, index=X.columns)
    topten = forest_importances.sort_values(ascending = False)[0:10]
    lowten = forest_importances.sort_values(ascending = True)[0:10]
    feature_vals = np.flip(np.sort(feature_importances))
    feature_vals_low = np.sort(feature_importances)
    tol = 0.5
    tol_low = 0.01
    gini_im = 0
    gini_im_low = 0
    for i in range(len(feature_vals)):
        gini_im += feature_vals[i]
        if gini_im >= tol:
            num_features = i+1
            break
    for j in range(len(feature_vals_low)):
        gini_im_low += feature_vals_low[i]
        if gini_im_low >= tol_low:
            num_features_low = j+1
            break
    topten.plot.bar()
    plt.title('10 most important features')
    plt.ylabel('Gini importance')
    plt.tight_layout()
    plt.savefig('Feature_importance.png', dpi = 200)
    plt.close()
    log("Number of features to obtain a gini importance of 0.5:", num_features)
    log("Ten most important features", topten)
    log("Ten least important features", lowten)

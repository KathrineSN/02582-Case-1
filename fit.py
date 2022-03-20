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
            with TT.profile("%s" % models[s]):
                models[s].fit(x_train, y_train)
            y_hat = predict(models[s], x_val)
            E_val[s, i] = error(y_val, y_hat)

    best_models = E_val.argmin(axis=0)
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
    feature_vals = np.flip(np.sort(feature_importances))
    tol = 0.5
    gini_im = 0
    for i in range(len(feature_vals)):
        gini_im += feature_vals[i]
        if gini_im >= tol:
            num_features = i+1
            break
    topten.plot.bar()
    plt.title('10 most important features')
    plt.ylabel('Gini importance')
    plt.tight_layout()
    plt.savefig('Feature_importance.png', dpi = 200)
    plt.close()
    log("Number of features to obtain a gini importance of 0.5:", num_features)
    log("Ten most important features", topten) 

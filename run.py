from data import load
from fit import fit, error, predict, feature_importance
from nn import NN
from fit import fit, error, predict, fit_model
from pelutils import log, Levels, TT, Table
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set_style('darkgrid')
import warnings
from pelutils.ds.plot import update_rc_params, rc_params
update_rc_params(rc_params)

def predicter(models, x):
    y = np.zeros(len(x))
    for model, weight, m in models:
        y += weight * predict(model, x, m)
    return y

if __name__ == "__main__":
    log.configure("model.log", print_level=Levels.DEBUG)
    with log.log_errors:
        warnings.simplefilter("ignore")
        X, y, keep, all_cats,_ = load("train.xlsx")
        feature_importance(X,y)
        log("%i data points with %i features" % (len(X), len(X.columns)))

        all_models = [
            RandomForestRegressor(n_estimators=16, n_jobs=-1),
            RandomForestRegressor(n_estimators=16, n_jobs=-1, min_samples_leaf=3),
            RandomForestRegressor(n_estimators=16, n_jobs=-1, max_features=5),
            Lasso(alpha=0.0001),
            LinearRegression(),
            MLPRegressor((24, 12)),
            KNeighborsRegressor(n_neighbors=10, n_jobs=-1),
        ]

        K2 = 4
        Epe = fit(X, y, all_models, K1=10, K2=K2)

        log.debug("CV time distribution", TT)
        log("Epe: %.4f" % Epe, "Acc: %.2f %%" % (100*(1-Epe)))

        S = len(all_models)
        E = np.zeros((S, K2))
        for i, (train_idx, val_idx) in enumerate(KFold(n_splits=K2).split(X)):
            x_train, y_train = X.iloc[train_idx], y[train_idx]
            x_val, y_val = X.iloc[val_idx], y[val_idx]
            for s in range(S):
                meta = fit_model(all_models[s], x_train, y_train)
                y_hat = predict(all_models[s], x_val, meta)
                E[s, i] = error(y_val, y_hat)
        s_star = np.argmin(E.mean(axis=1))
        final_model = all_models[s_star]
        log("Final model: %s" % final_model)

        log.section("Predicting on test set")
        log("Retraining model")
        meta = fit_model(final_model, X, y)
        y_hat = predict(final_model, X, meta)
        plt.figure(figsize=(20, 10))
        plt.subplot(121)
        plt.gca().axis("equal")
        plt.scatter(y, y_hat, s=1)
        plt.grid()
        plt.xlabel("LoadFactor")
        plt.ylabel("Predicted LoadFactor")
        plt.subplot(122)
        e = 1 - np.abs(y_hat / y)
        plt.scatter(y, e, s=1)
        plt.grid()
        plt.xlabel("LoadFactor")
        plt.ylabel("Deviance")
        plt.tight_layout()
        plt.savefig("preds.png")
        plt.close()
        col_order = X.columns
        X, y, *_ = load("test.xlsx", keep, all_cats)
        X = X[col_order].copy()
        log("Predicting test data")
        y_hat = predict(final_model, X, meta)
        log("Saving to output")
        with open("output.txt", "w") as f:
            for pred in y_hat:
                f.write("%f\n" % pred)

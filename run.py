from data import load
from nn import NN
from fit import fit, error, predict
from pelutils import log, Levels, TT, Table
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import warnings


def predicter(models, x):
    y = np.zeros(len(x))
    for model, weight, _ in models:
        y += weight * predict(model, x)
    return y

if __name__ == "__main__":
    log.configure("model.log", print_level=Levels.DEBUG)
    with log.log_errors, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X, y, keep, all_cats, df_init = load("train.xlsx")
        log("%i data points with %i features" % (len(X), len(X.columns)))
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

        models = fit(
            X_train,
            y_train,
            [
                # RandomForestRegressor(n_estimators=20),
                # RandomForestRegressor(n_estimators=20, n_jobs=-1),
                # RandomForestRegressor(n_estimators=50, n_jobs=-1),
                # RandomForestRegressor(n_estimators=20, n_jobs=-1, min_samples_leaf=4),
                # RandomForestRegressor(n_estimators=50, n_jobs=-1, min_samples_leaf=4),
                # RandomForestRegressor(n_estimators=20, n_jobs=-1, min_samples_leaf=8),
                # RandomForestRegressor(n_estimators=50, n_jobs=-1, min_samples_leaf=8),
                # BaggingRegressor(RandomForestRegressor(n_estimators=20, n_jobs=-1, min_samples_leaf=4)),
                # BaggingRegressor(RandomForestRegressor(n_estimators=50, n_jobs=-1, min_samples_leaf=4)),
                # Lasso(alpha=0.0001),
                # # Lasso(alpha=0.001),
                # LinearRegression(),
                # # MLPRegressor((20, 10)),
                # MLPRegressor((50, 20)),
                NN(),
            ],
            num_splits=5,
        )

        log.debug("CV time distribution", TT)

        table = Table()
        table.add_header(["Model", "weight"])
        for model, weight, _ in models:
            table.add_row([model, "%.4f" % weight], [1, 0])
            model.fit(X_train, y_train)
        #     feature_importances = model.feature_importances_().sort_values(ascending = False)
        #     tol = 0.5
        #     gini_im = 0
        #     for i in range(len(feature_importances)):
        #         gini_im += feature_importances[i]
        #         if gini_im >= tol:
        #             log("Number of features to obtain a gini importance of 0.5:", gini_im)
        #             log("Ten most important features", feature_importances[0,10])

        log("Models", table)
        y_hat = predicter(models, X_val)
        test_error = error(y_val, y_hat)
        log("Final error: %.4f" % test_error)
        plt.figure(figsize=(20, 10))
        plt.subplot(121)
        plt.scatter(y_val, y_hat, s=2)
        plt.gca().set_aspect('equal')
        plt.grid()
        plt.subplot(122)
        plt.scatter(y_val, np.abs(1-y_hat/y_val), s=2)
        plt.grid()
        plt.tight_layout()
        plt.savefig("preds.png")
        plt.close()

        log.section("Predicting on test set")
        log("Retraining model")
        for model, weight, _ in models:
            log.debug(model)
            model.fit(X, y)
        col_order = X.columns
        X, y, *_ = load("test.xlsx", keep, all_cats)
        X = X[col_order].copy()
        log("Predicting test data")
        y_hat = predicter(models, X)
        log("Saving to output")
        with open("output.txt", "w") as f:
            for pred in y_hat:
                f.write("%f\n" % pred)

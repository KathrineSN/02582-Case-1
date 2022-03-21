from data import load
from fit import fit, error, predict, feature_importance
from nn import NN
from fit import fit, error, predict, fit_model
from pelutils import log, Levels, TT, Table
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import warnings


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
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

        models = fit(
            X_train,
            y_train,
            [
                RandomForestRegressor(n_estimators=30, n_jobs=-1),
                RandomForestRegressor(n_estimators=100, n_jobs=-1),
                RandomForestRegressor(n_estimators=30, n_jobs=-1, min_samples_leaf=3),
                RandomForestRegressor(n_estimators=100, n_jobs=-1, min_samples_leaf=3),
                BaggingRegressor(LinearRegression(), n_estimators=30, max_features=50, n_jobs=-1),
                BaggingRegressor(LinearRegression(), n_estimators=30, max_features=50, n_jobs=-1),
                Lasso(alpha=0.0001),
                Lasso(alpha=0.001),
                LinearRegression(),
                MLPRegressor((20, 10)),
                MLPRegressor((50, 20)),
                MLPRegressor((100, 50, 20)),
                ElasticNet(),
            ],
            num_splits=20,
        )

        log.debug("CV time distribution", TT)

        table = Table()
        table.add_header(["Model", "Weight", "Error"])
        final_models = list()
        for model, weight, e in models:
            table.add_row([model, "%.4f" % weight, "%.4f" % e], [1, 0, 0])
            metadata = fit_model(model, X_train, y_train)
            if weight > 0:
                final_models.append([model, weight, metadata])

        log("Models", table)
        y_hat = predicter(final_models, X_val)
        test_error = error(y_val, y_hat)
        log("Final error:    %.4f" % test_error)
        log("Final accuracy: %.2f %%" % (100 * (1-test_error)))
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
        for i, (model, weight, m) in enumerate(final_models):
            log.debug("Refitting %s" % model)
            final_models[i][2] = fit_model(model, X, y)
        col_order = X.columns
        X, y, *_ = load("test.xlsx", keep, all_cats)
        X = X[col_order].copy()
        log("Predicting test data")
        y_hat = predicter(final_models, X)
        log("Saving to output")
        with open("output.txt", "w") as f:
            for pred in y_hat:
                f.write("%f\n" % pred)

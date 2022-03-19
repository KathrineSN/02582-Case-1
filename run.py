from data import load
from fit import fit, error, predict
from pelutils import log, Levels, TT, Table
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neural_network import MLPRegressor
import numpy as np


def predicter(models, x):
    y = np.zeros(len(x))
    for model, weight, _ in models:
        y += weight * predict(model, x)
    return y

if __name__ == "__main__":
    log.configure("model.log", print_level=Levels.DEBUG)
    with log.log_errors:
        df, df_coded, X, y = load("train.xlsx")
        num_train = int(0.2*len(X))
        models = fit(
            X.iloc[:num_train],
            y[:num_train],
            [
                RandomForestRegressor(n_estimators=20, n_jobs=-1),
                RandomForestRegressor(n_estimators=50, n_jobs=-1),
                RandomForestRegressor(n_estimators=20, n_jobs=-1, min_samples_leaf=4),
                RandomForestRegressor(n_estimators=50, n_jobs=-1, min_samples_leaf=4),
                RandomForestRegressor(n_estimators=20, n_jobs=-1, min_samples_leaf=8),
                RandomForestRegressor(n_estimators=50, n_jobs=-1, min_samples_leaf=8),
                BaggingRegressor(RandomForestRegressor(n_estimators=20, n_jobs=-1, min_samples_leaf=4)),
                BaggingRegressor(RandomForestRegressor(n_estimators=50, n_jobs=-1, min_samples_leaf=4)),
                Lasso(alpha=0.0001),
                # Lasso(alpha=0.001),
                LinearRegression(n_jobs=-1),
                # MLPRegressor((20, 10)),
                MLPRegressor((50, 20)),
            ],
            num_splits=15,
        )

        log.debug("CV time distribution", TT)

        table = Table()
        table.add_header(["Model", "weight"])
        for model, weight, _ in models:
            table.add_row([model, "%.4f" % weight], [1, 0])
            model.fit(X.iloc[:num_train], y[:num_train])
        log("Models", table)
        y_hat = predicter(models, X[num_train:])
        test_error = error(y[num_train:], y_hat)
        log("Final error: %.4f" % test_error)

        # log.section("Predicting on test set")
        # log("Retraining model")
        # for model, weight, _ in models:
        #     log.debug(model)
        #     model.fit(X, y)
        # TODO Fix missing features
        # df, df_coded, X, y = load("test.xlsx")
        # log("Predicting test data")
        # y_hat = predicter(models, X)
        # log("Saving to output")
        # with open("output.txt", "w") as f:
        #     for pred in y_hat:
        #         f.write("%f\n" % pred)

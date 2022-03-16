from data import load
from fit import fit
from pelutils import log, Levels
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


if __name__ == "__main__":
    log.configure("model.log")
    with log.log_errors:
        x, y, cols = load("train.xlsx")
        E = fit(x, y, cols,
            [RandomForestRegressor(n_estimators=10, n_jobs=-1), LinearRegression()], 5)
        log("Final generalization error: %.4f" % E)

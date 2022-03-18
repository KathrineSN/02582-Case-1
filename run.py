from data import load
from fit import fit
from pelutils import log, Levels
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso


if __name__ == "__main__":
    log.configure("model.log")
    with log.log_errors:
        x, y, cols = load("train.xlsx")
        E = fit(x, y, cols,
            [
                RandomForestRegressor(n_estimators=10, n_jobs=-1),
                # Lasso(alpha=0.0001),
                # LinearRegression(n_jobs=-1),
            ], 2)
        log("Final accuracy: %.2f %%" % (100*(1-E)))

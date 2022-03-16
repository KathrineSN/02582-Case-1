from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from pelutils import log


def total_accuracy(y: np.ndarray, y_hat: np.ndarray) -> float:
    non_zero = y > 0
    y, y_hat = y[non_zero], y_hat[non_zero]
    return (1 - np.abs(y-y_hat)).mean()
    dev = (y* - y_hat) / y
    acc = 1 - np.abs(dev)
    return acc.mean()

def fit(x: np.ndarray, y: np.ndarray, cols: dict[str, int], models, num_splits: int):
    # Page 175 in ML book
    log.section("Fitting model using %i splits" % num_splits)
    N = len(x)
    S = len(models)
    K1 = K2 = num_splits

    E_test = np.zeros(K1)
    D_test_sizes = np.zeros(K1)
    for i, (par_idx, test_idx) in enumerate(KFold(n_splits=K1).split(x)):
        log("Outer loop %i / %i" % (i+1, K1))
        D_test_sizes[i] = len(test_idx)
        x_par, x_test = x[par_idx], x[test_idx]
        y_par, y_test = y[par_idx], y[test_idx]

        E_val = np.zeros((K2, S))
        D_val_sizes = np.zeros(K2)
        for j, (train_idx, val_idx) in enumerate(KFold(n_splits=K2).split(par_idx)):
            D_val_sizes[j] = len(val_idx)
            x_train, y_train = x[train_idx], y[train_idx]
            x_val, y_val = x[val_idx], y[val_idx]

            for s in range(S):
                models[s].fit(x_train, y_train)
                y_hat = models[s].predict(x_val)
                E_val[j, s] = total_accuracy(y_val, y_hat)

        E_gen_hat = np.zeros(S)
        for s in range(S):
            E_gen_hat[s] = np.sum(D_val_sizes / len(x_par) * E_val[:, s])

        s_star = np.argmin(E_gen_hat)
        log("Best model: %i" % s_star)
        models[s_star].fit(x_par, y_par)
        y_hat = models[s_star].predict(x_test)
        E_test[i] = total_accuracy(y_test, y_hat)

    E_gen_hat = np.sum(D_test_sizes/N*E_test)
    return E_gen_hat

        # M = models[s]
        # model.fit(x_train, y_train)
        # y_pred = model.predict(x_test)

        # # Deviation per flight
        # n_pas_pred = np.ceil(y_pred * x_test[cols['SeatCapacity']]) # number of predicted passengers
        # n_pas = np.ceil(y_test * x_test[cols['SeatCapacity']]) # actual number of passengers

        # diff = (n_pas - n_pas_pred)
        # dev_per_fli = np.divide(diff, n_pas, out=np.zeros_like(diff), where=n_pas!=0) # Deviating per flight
        # acc_per_fli = np.ones(len(dev_per_fli))- abs(dev_per_fli) # Accuracy per flight
        # acc = np.mean(acc_per_fli) # Overall accuracy

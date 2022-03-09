from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from pelutils import log


def fit(df: pd.DataFrame, num_splits: int):
    log.section("Fitting model using %i splits" % num_splits)
    # breakpoint()
    for i, (train_idx, test_idx) in enumerate(KFold(n_splits=num_splits).split(df)):
        log.debug("Fitting for split %i / %i" % (i+1, num_splits))
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        x_train, y_train = train_df.drop("LoadFactor", axis=1), train_df.LoadFactor.values
        x_test, y_test = test_df.drop("LoadFactor", axis=1), test_df.LoadFactor.values
        model = RandomForestRegressor(n_estimators=10)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        
        # Deviation per flight
        n_pas_pred = np.ceil(y_pred * x_test['SeatCapacity']) # number of predicted passengers
        n_pas = np.ceil(y_test * x_test['SeatCapacity']) # actual number of passengers
        
        diff = (n_pas - n_pas_pred)
        dev_per_fli = np.divide(diff, n_pas, out=np.zeros_like(diff), where=n_pas!=0) # Deviating per flight
        acc_per_fli = np.ones(len(dev_per_fli))- abs(dev_per_fli) # Accuracy per flight
        acc = np.mean(acc_per_fli) # overall accuracy
        


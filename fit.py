from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import pandas as pd
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
        model.predict(x_test)

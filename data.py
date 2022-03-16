import numpy as np
import pandas as pd
from pelutils import log


def _process(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """ Performs some in-place preprocessing, e.g.
    converting ScheduleTime to float epoch time and
    converting categorical variables to integers. """
    df["ScheduleTime"] = df["ScheduleTime"].values.astype(float) / 1e9
    log("Data has %i points before removing missing" % len(df))
    df = df.dropna()
    log("Data has %i points after removing missing" % len(df))
    str_cols = "Airline", "Destination", "AircraftType", "FlightType", "Sector"
    datas = list()
    cols = dict()
    num_cols = 0
    for col in str_cols:
        df[col] = df[col].astype(str)
        keys = { key: i for i, key in enumerate(pd.unique(df[col])) }
        data = np.zeros((len(df), len(keys)))
        data[np.arange(len(df)), [keys[v] for v in df[col].values]] = 1
        cols[col] = num_cols
        num_cols += len(keys)
        datas.append(data)
        log.debug("Found %i unique values for %s" % (len(keys), col))
    for col in df.columns:
        if col in str_cols:
            continue
        if col == "LoadFactor":
            y = df[col].values
            continue
        datas.append(np.expand_dims(df[col].values, axis=1))
        cols[col] = num_cols
        num_cols += 1
    return np.hstack(datas), y, cols

def load(path: str) -> tuple[np.ndarray]:
    log.section("Loading data from %s" % path)
    df = pd.read_excel(path)
    log("Preprocessing data")
    x, y, cols = _process(df)
    log("Loaded %i data points with %i features" % (x.shape[0], x.shape[1]))
    return x, y, cols

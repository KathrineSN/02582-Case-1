import pandas as pd
from pelutils import log


def _process(df: pd.DataFrame):
    """ Performs some in-place preprocessing, e.g.
    converting ScheduleTime to float epoch time and
    converting categorical variables to integers. """
    df["ScheduleTime"] = df["ScheduleTime"].values.astype(float) / 1e9
    str_cols = "Airline", "Destination", "AircraftType", "FlightType", "Sector"
    for col in str_cols:
        df[col] = df[col].astype(str)
        keys = { key: i for i, key in enumerate(pd.unique(df[col])) }
        log.debug("Found %i unique values for %s" % (len(keys), col))
        df[col] = [keys[key] for key in df[col]]

def load(path: str) -> pd.DataFrame:
    log.section("Loading data from %s" % path)
    df = pd.read_excel(path)
    log("Preprocessing data")
    _process(df)
    log("Loaded %i data points with features" % len(df), df.dtypes)
    return df

from __future__ import annotations
import numpy as np
import pandas as pd
from pelutils import log
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from enum import IntEnum


class TimeHandling(IntEnum):
    NONE = 0
    ONEHOT = 1
    CIRCULAR = 2


time_handling = TimeHandling.CIRCULAR

def _group_categoricals(df: pd.DataFrame, keep_cats: dict[str, pd.Series]) -> dict[str, pd.Series]:
    log("Removing rare categories")
    cats = [("Airline", 0.95), ("Destination", 0.95), ("AircraftType", 0.95), ("Sector", 0.97)]
    df = df.copy()
    if keep_cats is None:
        keep_cats = dict()
        for col, threshold in cats:
            counts = df[col].value_counts()
            cumfreq = counts.cumsum() / counts.sum()
            keep = cumfreq < threshold
            keep_cats[col] = keep
            log.debug("Kept %i / %i of %s categories" % (keep.sum(), len(keep), col))
            df[col] = [x if keep[x] else "Other" for x in df[col]]
    else:
        for col, _ in cats:
            df[col] = [x if x in keep_cats and keep_cats[x] else "Other" for x in df[col]]
    return df.copy(), keep_cats

def _process(df: pd.DataFrame, keep: dict[str, pd.Series], all_cats):
    """ Performs some in-place preprocessing, e.g.
    converting ScheduleTime to float epoch time and
    converting categorical variables to integers. """
    log("Data has %i points before removing missing" % len(df))
    df = df.dropna()
    # Splitting scheduled time into multiple columns
    log("Data has %i points after removing missing" % len(df))
    df['Year'] = df.ScheduleTime.apply(lambda text: int(str(text).split('-')[0]))
    df['Month'] = df.ScheduleTime.apply(lambda text: str(text).split('-')[1])
    df['Date'] = df.ScheduleTime.apply(lambda x: str(x)[8:10])
    df['Hour'] = df.ScheduleTime.apply(lambda x: str(x)[11:13])
    df['Weekday'] = df['ScheduleTime'].dt.day_name()

    df_init = df.copy()
    df = df.copy()
    df.loc[(df["FlightType"]=="O").values, "FlightType"] = "C"
    df.loc[(df["FlightType"]=="G").values, "FlightType"] = "J"

    df = df.drop(["FlightNumber"], axis=1).copy()

    # One-hot encode
    time_cols = ["Month", "Date", "Hour"]
    periods = {"Month": 12, "Date": 31, "Hour": 24}
    oh_cols = ['Airline', 'Destination', 'AircraftType', 'FlightType', 'Sector', 'Weekday']
    if time_handling == TimeHandling.ONEHOT:
        oh_cols += time_cols
    elif time_handling == TimeHandling.NONE:
        for tc in time_cols:
            df[tc] = df[tc].apply(int)
    elif time_handling == TimeHandling.CIRCULAR:
        for tc in time_cols:
            df[tc] = df[tc].apply(int)
            if tc != "Hour":
                df[tc] = df[tc] - 1
            df[tc+"half"] = df[tc] % periods[tc] > periods[tc] / 2
            df[tc] = np.sin(2*np.pi/periods[tc]*(df[tc]-periods[tc]/4)) / 2 + 1 / 2

    df, keep = _group_categoricals(df, keep)

    df = pd.get_dummies(df, columns=oh_cols)
    df = df.drop('ScheduleTime', axis=1).copy()

    for cat in all_cats:
        if cat not in df.columns:
            df[cat] = [0] * len(df)
    df = df.copy()
    all_cats = [x for x in df.columns if any(x.startswith(y) for y in oh_cols)]

    X = df.loc[:, df.columns != 'LoadFactor']
    y = df[['LoadFactor']].values.ravel()

    return df, X, y, keep, all_cats, df_init

def load(path: str, keep: dict[str, pd.Series]=None, all_cats=list()):
    log.section("Loading data from %s" % path)
    df = pd.read_excel(path)
    if path == "test.xlsx":
        df["LoadFactor"] = 0
    log("Preprocessing data")
    df, X, y, keep, all_cats, df_init = _process(df, keep, all_cats)
    if path == "train.xlsx":
        df = df.loc[df.LoadFactor > 0].copy()
        log("Data has %i data points after limiting to positive LoadFactor" % len(df))
    return df, X, y, keep, all_cats, df_init

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

def _process(df: pd.DataFrame):
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

    df["FlightType"].loc[df["FlightType"]=="O"] = "C"
    df["FlightType"].loc[df["FlightType"]=="G"] = "J"

    df = df.drop(["FlightNumber"], axis=1)

    # TODO Group one hot stuff with few instances into "other" group

    # One-hot encode
    time_cols = ["Month", "Date", "Hour"]
    periods = {"Month": 12, "Date": 31, "Hour": 24}
    oh_cols = ['Airline', 'Destination', 'AircraftType', 'FlightType', 'Sector']
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

    df_coded = pd.get_dummies(df, columns=oh_cols)
    df_coded = df_coded.drop('ScheduleTime', axis=1)
    X = df_coded.loc[:, df_coded.columns != 'LoadFactor']
    y = df_coded[['LoadFactor']].values.ravel()

    return df, df_coded, X, y

def load(path: str):
    log.section("Loading data from %s" % path)
    df = pd.read_excel(path)
    if path == "test.xlsx":
        df["LoadFactor"] = 0
    log("Preprocessing data")
    df, df_coded, X, y = _process(df)
    return df, df_coded, X, y

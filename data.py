from __future__ import annotations
import numpy as np
import pandas as pd
from pelutils import log
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def _process(df: pd.DataFrame):
    """ Performs some in-place preprocessing, e.g.
    converting ScheduleTime to float epoch time and
    converting categorical variables to integers. """
    log("Data has %i points before removing missing" % len(df))
    df = df.dropna()
    # Splitting scheduled time into multiple columns
    log("Data has %i points after removing missing" % len(df))
    df['Year'] = df.ScheduleTime.apply(lambda text: str(text).split('-')[0])
    df['Month'] = df.ScheduleTime.apply(lambda text: str(text).split('-')[1])
    df['Date'] = df.ScheduleTime.apply(lambda x: int(str(x)[8:10]))
    df['Hour'] = df.ScheduleTime.apply(lambda x: int(str(x)[11:13]))
    
    df_coded = df.copy()
    
    # Normalizing seat capacity and load factor
    df_coded['LoadFactor'] = (df_coded['LoadFactor']-df_coded['LoadFactor'].mean())/df_coded['LoadFactor'].std()
    df_coded['SeatCapacity'] = (df_coded['SeatCapacity']-df_coded['SeatCapacity'].mean())/df_coded['SeatCapacity'].std()
    
    # One-hot encode
    df_coded = pd.get_dummies(df, columns = ['Airline', 'Destination', 'AircraftType', 'FlightType', 'Sector', 'Year', 'Month', 'Date', 'Time'])
    df_coded = df_coded.drop('ScheduleTime', axis = 1)
    X = df_coded.loc[:, df_coded.columns != 'LoadFactor']
    y = df_coded[['LoadFactor']]
    
    return df, df_coded, X, y

def load(path: str):
    log.section("Loading data from %s" % path)
    df = pd.read_excel(path)
    log("Preprocessing data")
    df, df_coded, X, y = _process(df)
    return df, df_coded, X, y

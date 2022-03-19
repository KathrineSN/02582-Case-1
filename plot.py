# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 10:46:30 2022

@author: kathr
"""
from data import load
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import numpy as np
from typing import Iterable, Callable

def linear_binning(x: Iterable, bins: int) -> np.ndarray:
    """ Standard linear binning """
    return np.linspace(min(x), max(x), bins)

def get_bins(
    data:         Iterable,
    binning_fn:   Callable[[Iterable, int], Iterable] = linear_binning,
    bins:         int  = 25,
    density:      bool = True,
    ignore_zeros: bool = False,
):
    """ Create bins for plotting a line histogram. Simplest usage is plt.plot(*get_bins(data)) """
    bins = binning_fn(data, bins+1)
    y, edges = np.histogram(data, bins=bins, density=density)
    x = (edges[1:] + edges[:-1]) / 2
    if ignore_zeros:
        x, y = x[y>0], y[y>0]
    return x, y

#%% Data load
df,df_coded,_,_ = load("train.xlsx")

#%% Information about dataframe
print('Number of unique instances in each category')
print(df.nunique())

#%% Plotting correlations
sns.pairplot(df)
plt.savefig('pairplot.png', dpi = 200)
plt.close()

#%% Number of flight per month in each sector
# Number of flights in each sector by month
df_grouped_month = df.groupby(['Sector','Month']).size().unstack()
print(df_grouped_month)

fig = plt.figure(figsize=(15, 18))
for plot_index in range(1, 13):
    fig = plt.subplot(6, 2, plot_index)
    plt.subplots_adjust(hspace = 0.55)
    
    ax = sns.barplot(x = df_grouped_month.iloc[plot_index - 1].index, y = df_grouped_month.iloc[plot_index - 1].values, color = sns.color_palette()[0])
    plt.margins(y = 0.25, x = 0.05) # set the inner margins between plot values and plot
    plt.title(df_grouped_month.iloc[plot_index - 1].name, y = 0.80, x = 0.05, loc = 'left') # set indentation 'left' and x, y added/subtracted, compared to the default values
    plt.xlim([-1,12])
plt.tight_layout()
plt.savefig('Flights_per_month_per_sector.png', dpi = 200)
plt.close()


# Number of flights in each sector by date
df_grouped_date = df.groupby(['Sector','Date']).size().unstack()

fig = plt.figure(figsize=(15, 18))
for plot_index in range(1, 13):
    fig = plt.subplot(6, 2, plot_index)
    plt.subplots_adjust(hspace = 0.55)

    ax = sns.barplot(x = df_grouped_date.iloc[plot_index - 1].index, y = df_grouped_date.iloc[plot_index - 1].values, color = sns.color_palette()[0])
    plt.margins(y = 0.25, x = 0.05) # set the inner margins between plot values and plot
    plt.xlim([-1,31])
    plt.title(df_grouped_date.iloc[plot_index - 1].name, y = 0.80, x = 0.05, loc = 'left') # set indentation 'left' and x, y added/subtracted, compared to the default values
plt.tight_layout()
plt.savefig('Flights_per_day_per_sector.png', dpi = 200)
plt.close()

#%% Histograms of continuous variables
fig, ax = plt.subplots(figsize=(17,12), 
                       nrows=2, 
                       ncols=3)

fig.suptitle('Distribution of Numerical Features', 
             fontsize=21) #use a for loop to create each subplot:
features = ['LoadFactor', 
       'SeatCapacity',
       'Month', 
       'Date', 
       'Year', 
       'Hour']
for feat in features:
    row = features.index(feat)//3
    col = features.index(feat)%3
    
    ax[row, col].hist(df[feat], bins=20)
    ax[row, col].set_title(feat.title(), 
                           fontsize=20)
    ax[row, col].set_xlabel(feat.title(),
                            fontsize=18)
    ax[row, col].set_ylabel('Count',
                            fontsize=18)
    
plt.subplots_adjust(top=0.9, wspace=0.3, hspace=0.3)
plt.savefig('hist_numerical_features.png', dpi = 200)
plt.close()


#%% Additional histogram of flightnumber
plt.figure(figsize = (5,5))
plt.hist(df['FlightNumber'], bins = 20)
plt.title('FlightNumber', fontsize=18)
plt.xlabel('FlightNumber',fontsize=14)
plt.ylabel('Count',fontsize=14)
plt.tight_layout()
plt.savefig('hist_FlightNumber.png', dpi = 200)
plt.close()

#%% Bar plots of continuous variables
features = ['FlightType',  
       'AircraftType', 
       'Sector']
fig = plt.figure(figsize = (20,5))
fig.suptitle('Distribution of Categorical features', fontsize = 21)
for i in range(len(features)):
    plt.subplot(1,3,i+1)
    df[features[i]].value_counts().plot.bar(title = features[i])
plt.tight_layout()
plt.savefig('bar_categorical_features.png', dpi = 200)
plt.close()
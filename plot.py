# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 10:46:30 2022

@author: kathr
"""
from data import load
import matplotlib.pyplot as plt
import seaborn as sns

df,df_coded,_,_ = load("train.xlsx")

# Data visualizations
print('Number of unique instances in each category')
print(df.nunique())

# Plotting correlations
sns.pairplot(df_coded)

# Number of flights in each sector by month
df_grouped_month = df_coded.groupby(['Sector','Month']).size().unstack()
print(df_grouped_month)

# Plotting the locations
fig = plt.figure(figsize=(15, 18))
for plot_index in range(1, 13):
    # Creating a subplot placeholder corresponding to a category
    fig = plt.subplot(6, 2, plot_index)
    # Make a space between the different rows of plots
    plt.subplots_adjust(hspace = 0.55)

    # P.S. I can of course remove the labels from all but the last 2 plots, and make the x-axis year labels show from 2 to 2 years, however I kept this style, since I believe it is more beautiful this way
    
    ax = sns.barplot(x = df_grouped_month.iloc[plot_index - 1].index, y = df_grouped_month.iloc[plot_index - 1].values, color = sns.color_palette()[0])
    plt.margins(y = 0.25, x = 0.05) # set the inner margins between plot values and plot
    plt.title(df_grouped_month.iloc[plot_index - 1].name, y = 0.80, x = 0.05, loc = 'left') # set indentation 'left' and x, y added/subtracted, compared to the default values

# Number of flights in each sector by date
df_grouped_date = df_coded.groupby(['Sector','Date']).size().unstack()

# Plotting the locations
fig = plt.figure(figsize=(15, 18))
for plot_index in range(1, 13):
    # Creating a subplot placeholder corresponding to a category
    fig = plt.subplot(6, 2, plot_index)
    # Make a space between the different rows of plots
    plt.subplots_adjust(hspace = 0.55)

    # P.S. I can of course remove the labels from all but the last 2 plots, and make the x-axis year labels show from 2 to 2 years, however I kept this style, since I believe it is more beautiful this way
    ax = sns.barplot(x = df_grouped_date.iloc[plot_index - 1].index, y = df_grouped_date.iloc[plot_index - 1].values, color = sns.color_palette()[0])
    plt.margins(y = 0.25, x = 0.05) # set the inner margins between plot values and plot
    plt.title(df_grouped_date.iloc[plot_index - 1].name, y = 0.80, x = 0.05, loc = 'left') # set indentation 'left' and x, y added/subtracted, compared to the default values

import math
import pandas as pd
import numpy as np
import datetime as dt
import statistics as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, TargetEncoder
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from scipy.stats import iqr, spearmanr, linregress, ecdf
from typing import List, Union, Literal

def plot_correlation_matrix(df: pd.DataFrame, drop: List[str] = [],
                            title: str = None,
                            annot: bool = True):
    """
    Plots a correlation matrix of a pandas dataframe

    Parameters:
        df (DataFrame): Pandas dataframe cotaining values of variables to plot
        drop (List): List of columns to drop due to non numerical variables
    """
    df = df.drop(columns=drop)
    correlation_matrix = df.corr(method="spearman")
    plt.figure()    
    sns.heatmap(correlation_matrix, annot=annot, cmap='viridis', fmt=".2f")
    if title is not None:
        plt.title(title)
    plt.show()

def plot_hist_continous(df: pd.DataFrame, variable: str) -> None:
    """
    Plots the histogram of a continous variable of a pandas
    dataframe

    Parameters:
        df (DataFrame): Pandas dataframe cotaining values of variable to plot
        variabl (str): Name of the variable to plot
    """

    interquartile_range = iqr(df[variable])
    n = len(df.index)
    min = df[variable].min()
    max = df[variable].max()
    bins = int((max-min)/(2*(interquartile_range/(n**(1/3)))))

    plt.figure(figsize=(8, 6))
    plt.hist(df[variable], bins=bins, density=True)
    plt.axvline(x=st.mean(df[variable]), color='r', linestyle='--')
    plt.axvline(x=max, color='g', linestyle='--')
    plt.axvline(x=min, color='g', linestyle='--')
    plt.title(f'{variable}')
    plt.xlabel(f'{variable}')
    plt.ylabel('Density')
    plt.show()

def plot_relationship_continous(df: pd.DataFrame, variable_x: str,
                                variable_y: str, x_unit: str = "",
                                y_unit: str = "",
                                y_name: str = None,
                                x_name: str = None,
                                sig_level: float = 0.95,
                                title: str = None,
                                export: bool = False):
    """
    Plots the relationship between continous variables of a pandas
    dataframe

    Parameters:
        df (DataFrame): Pandas dataframe cotaining values of variables to plot
        variable_x (str): Name of the variable x to plot
        variable_y (str): Name of the variable y to plot
        x_unit (str): Unit of variable x
        y_unit (str): Unit of variable y
    """
    x_label = variable_x
    if x_name is not None:
        x_label = x_name

    y_label = variable_y
    if y_name is not None:
        y_label = y_name

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    if title is not None:
        ax1.set_title(title, fontsize=16)
    else:
        ax1.set_title(f'{x_label} ~ {y_label}')
    ax1.scatter(df[variable_x], df[variable_y], marker='x', color='blue')
    ax1.set_ylabel(y_label + " " + y_unit, fontsize=14)
    ax1.set_xlabel(x_label + " " + x_unit, fontsize=14)

    corr = spearmanr(df[variable_x], df[variable_y])
    stat = round(corr.statistic, 4)
    p_value = round(corr.pvalue, 4)
    thres = 1-sig_level

    s = " "
    if p_value <= thres:
        s = "*"

    if abs(stat) < 0.2:
        a = "very weak"
    elif abs(stat) >= 0.2 and abs(stat) < 0.4:
        a = "weak"
    elif abs(stat) >= 0.4 and abs(stat) < 0.6:
        a = "moderate"
    elif abs(stat) >= 0.6 and abs(stat) < 0.8:
        a = "strong"
    elif abs(stat) >= 0.8:
        a = "very strong"

    slope, intercept, _, _, _ = linregress(df[variable_x], df[variable_y])
    x_reg = np.linspace(df[variable_y].min(), df[variable_x].max(), 100)
    y_reg = slope * x_reg + intercept
    residuals = df[variable_y] - (slope * df[variable_x] + intercept)
    std_err = np.std(residuals)
    confInterval = 1.96 * std_err
    yUpper = y_reg + confInterval
    yLower = y_reg - confInterval

    ax2.plot(x_reg, y_reg, color='red')
    ax2.fill_between(x_reg, yLower, yUpper, alpha=0.2, color='blue')
    ax2.set_xlabel(x_label + " " + x_unit, fontsize=14)
    ax2.set_ylabel(y_label + " " + y_unit, fontsize=14)
    ax2.set_title(f'ρ: {stat} ({a} / ({sig_level*100}% {s})', fontsize=16)
    fig.tight_layout()
    if export:
        plt.savefig(f"{paths.FIGURE_PATH}cont_relatioship_{variable_x}_{variable_y}.png", format="png", bbox_inches="tight", dpi=300)
    plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import iqr, spearmanr, linregress, ecdf
from typing import List, Literal

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
    plt.figure(figsize=(16, 14)) 
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
    #plt.axvline(x=st.mean(df[variable]), color='r', linestyle='--')
    #plt.axvline(x=max, color='g', linestyle='--')
    #plt.axvline(x=min, color='g', linestyle='--')
    plt.title(f'Histogram of {variable}')
    plt.xlabel(f'{variable}')
    plt.ylabel('Density')
    plt.show()

def plot_relationship_continous(df: pd.DataFrame, variable_x: str,
                                variable_y: str, x_unit: str = "",
                                y_unit: str = "",
                                y_name: str = None,
                                x_name: str = None,
                                sig_level: float = 0.95,
                                title: str = None):
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
    ax1.scatter(df[variable_x], df[variable_y], marker='+', color='blue', s=2.5)
    ax1.set_ylabel(y_label + " " + y_unit, fontsize=14)
    ax1.set_xlabel(x_label + " " + x_unit, fontsize=14)

    corr = spearmanr(df[variable_x], df[variable_y])
    stat = round(corr.statistic, 4)
    p_value = round(corr.pvalue, 4)
    thres = 1-sig_level

    s = "~"
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
    ax2.set_title(f'ρ: {stat} ({a})', fontsize=16)
    fig.tight_layout()
    plt.show()


def correction_factor(lower_estimator, upper_estimator, X_calib, y_calib, miscoverage: float):
    """
    Computes the correction factor for confromalised qunatile regression based on
    
    Romano, Y., Patterson, E., & Cand`es, E. J. (2019). Conformalized quantile regression. 
    Proceedings of the 33rd International Conference on Neural Information Processing Systems. 
    https://doi.org/10.48550/arXiv.1905.03222 

    Parameters:
        lower_estimator: Lower quantile estimator
        upper_estimator: Upper quantile estimator
        X_calib: Predictors for calibration as pandas dataframe
        y_calib: Target variable for calibration as pandas dataframe
        miscoverage (float): Miscoverage level for which to compute the
                             corection factor       

    Returns:
        correction_factor (float): Correction factor to pad the prediction intervals
    """
    y_calib = y_calib.to_numpy().ravel()
    n = len(y_calib)

    lower_errors = lower_estimator.predict(X_calib)-y_calib
    upper_errors = y_calib-upper_estimator.predict(X_calib)

    conformity_scores = np.maximum(lower_errors, upper_errors)
    empirical_quantile = (1-miscoverage)*(1+1/n)

    correction_factor = np.quantile(conformity_scores, empirical_quantile)

    return correction_factor


def signed_correction_factor(lower_estimator, upper_estimator, X_calib, y_calib, miscoverage: float):
    """
    Computes the correction factors for signed confromalised qunatile regression based on

    Romano, Y., Patterson, E., & Cand`es, E. J. (2019). Conformalized quantile regression.
    Proceedings of the 33rd International Conference on Neural Information Processing Systems.
    https://doi.org/10.48550/arXiv.1905.03222
    and 
    Linusson, H., Johansson, U., & Löfström, T. (2014). Signed-Error Conformal Regression.
    Proceedings of the 18th Pacific-Asia Conference.
    https://doi.org/10.1007/978-3-319-06608-0


    Parameters:
        lower_estimator: Lower quantile estimator
        upper_estimator: Upper quantile estimator
        X_calib: Predictors for calibration as pandas dataframe
        y_calib: Target variable for calibration as pandas dataframe
        miscoverage (float): Miscoverage level for which to compute the
                             corection factor

    Returns:
        lower_correction_factor (float): Lower correction factor to pad the prediction intervals
        upper_correction_factor (float): Upper correction factor to pad the prediction intervals
    """
    y_calib = y_calib.to_numpy().ravel()
    n = len(y_calib)

    lower_errors = lower_estimator.predict(X_calib)-y_calib
    upper_errors = y_calib-upper_estimator.predict(X_calib)

    lower_empirical_quantile = (1-round(miscoverage/2, 3))*(1+1/n)
    upper_empirical_quantile = (1-round(miscoverage/2, 3))*(1+1/n)

    lower_correction_factor = np.quantile(lower_errors,
                                          lower_empirical_quantile)
    upper_correction_factor = np.quantile(upper_errors,
                                          upper_empirical_quantile)

    return lower_correction_factor, upper_correction_factor


def coverage(y_pred, y_lower, y_upper):
    """
    Computes the coverage of the prediction intervals defined by upper and
    lower bounds for given point estimates

    Parameters:
        y_pred: Point estimates
        y_lower: Lower bound of the prediction intervals for y_pred
        y_upper: Upper bound of the prediction intervals for y_pred

    Returns:
        coverage (float): Coverage that is achived by the prediction intervals
                          defined by y_lower, y_upper on y_pred
    """
    return np.mean((y_pred >= y_lower) & (y_pred <= y_upper))

def size(y_lower, y_upper, mode: Literal['mean', 'median'] = 'mean'):
    """
    Computes the size of the prediction intervals defined by upper and
    lower bounds

    Parameters:
        y_lower: Lower bound of the prediction intervals
        y_upper: Upper bound of the prediction intervals
        mode (str)

    Returns:
        size (float): Coverage that is achived by the prediction intervals
                          defined by y_lower, y_upper on y_pred
    """
    match mode:
        case "mean":
            return np.mean(abs(y_upper - y_lower))
        case "median":
            return np.median(abs(y_upper - y_lower))
        case _:
            return np.mean(abs(y_upper - y_lower))
        
def plot_scatter(df, var_x, var_y, color_var, title, x_label, y_label, cmap: str = "viridis"):
    """
    Plots a colored scatterplot 

    Parameters:
        df (DataFrame): Pandas dataframe cotaining values to plot
        var_x (str): Variable x
        var_y (str): Variable y
        color_var (str): Variable to use for coloring
        title (str): Title of the plot
        x_label (str): Lable of variable x 
        y_label (str): Lable of variable y
        cmap (str): Colormap to use for plotting
    """
    plt.figure(figsize=(12, 8))
    plt.scatter(df[var_x], df[var_y], c=df[color_var], s=8, cmap=cmap)
    plt.colorbar()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
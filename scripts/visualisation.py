import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import pause
from typing import List
from sklearn.tree import DecisionTreeRegressor


def visualise_dataset(df):
    """
    Utility function for visualising the preprocessed dataset
    Args:
        df: Pandas dataframe containing the data
    """
    red = "#b20710"

    fig, ax = plt.subplots(1, 1, figsize=(40, 10))

    lanes = ["northBound", "southBound"]
    colours = ["blue", "red"]

    ax.set_xlabel("", fontsize=20)
    ax.set_title("Lane Volume", fontsize=20)
    ax.set_ylabel("Volume", fontsize=20)

    include = df[df.index.year == 2017]
    for lane, colour in zip(lanes, colours):
        include[lane].plot(color=colour, label=lane)

    plt.legend(fontsize=20)
    plt.show()


def visualise_features(
    feature_names: List[str], df: pd.DataFrame, large_title: str, target_var: str
):
    """
    Utility function for visualising features effect on data.
    Args:
        feature_names : A list of feature names
        df : Pandas dataframe of the dataset
        large_title : Large title used for the sub plots
        target_var : String of the column name we are trying to predict
    """

    num_plots: int = len(feature_names)
    fig, ax = plt.subplots(num_plots, 1, figsize=(20, num_plots * 5))

    fig.suptitle(large_title, fontsize=30)

    for feature, ax in zip(feature_names, ax.flatten()):
        grouped = df.groupby(feature)[target_var].mean()
        grouped.plot(ax=ax, color="red", marker="o", label="Mean Volume")
        ax.set_ylabel("Mean Volume")
        ax.set_title(f"{feature} mean volume")


def visualise_decision_tree(decision_tree_model, X, y):
    # Fit regression model

    # Predict
    y_pred = decision_tree_model.predict(X)

    # Plot the results
    plt.figure()

    x_axis = np.arange(0, y.size, 1)
    plt.scatter(x_axis, y, s=20, edgecolor="black", c="orange", label="data")
    plt.plot(x_axis, y_pred, color="blue", label="decision tree regressor", linewidth=2)
    plt.xlabel("hour index")
    plt.ylabel("target")
    plt.title("Decision Tree Regression")
    plt.legend()
    plt.show()


def visualise_forecast_vs_true(
    X_test: np.array, y_test: np.array, y_forecast: np.array, model_name: str, baseline_y
):
    """
    Function to visualise forecasted traffic volumes against the true values
    Args:
        X_test : Numpy array of test feature vectors
        y_test : Numpy array of testing output features
        y_forecast : Numpy array of forecasted (predicted) output features
        baseline_y : Numpy array of dummy (mean) predictions made as our baseline
    """
    # Plot the results
    plt.figure()

    plt.scatter(
        X_test, y_test, s=20, edgecolor="black", c="orange", label="Testing data"
    )
    plt.plot(X_test, y_forecast, color="blue", label="Forecasted data", linewidth=2)
    plt.plot(X_test, baseline_y, color="red", label="(Mean) Baseline", linewidth=2)
    plt.xlabel("Hours")
    plt.ylabel("Traffic Volume")
    plt.title(f"{model_name} predicted traffic volume")
    plt.legend()
    plt.show()


def visualise_multiple_forecast_vs_true(
    X_test: np.array, y_test: np.array, y_forecast_list, model_names: list
):
    num_points=24*8
    X_test=X_test[24:num_points]
    
    
    # Plot the results
    plt.figure()
    plt.scatter(
        X_test, y_test[24:num_points]*6606, s=20, edgecolor="black", c="orange", label="Testing data"
    )

    for i in range(0, len(model_names)):
        transparency = None
        if model_names[i]=="Neural Net":
            transparency=0.5

        plt.plot(X_test, y_forecast_list[i][24:num_points]*6606.0, label=model_names[i], linewidth=2,alpha=transparency)

    plt.xlabel("Hours")
    plt.ylabel("Traffic Volume")
    plt.title(f"Predicted traffic volume (Monday - Sunday)")
    plt.legend(bbox_to_anchor=(1, 1.05))
    plt.xlim([X_test[0],X_test[-1]])
    plt.show()

def forecast_plot(df, title, x_label, log_scale: bool):
    df.plot.bar(log=log_scale)
    plt.title(title)
    plt.ylabel("MSE")
    plt.xlabel(x_label)

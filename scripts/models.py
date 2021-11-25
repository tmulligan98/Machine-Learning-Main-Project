from typing import List
from utilities import cross_validation_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from typing import Any


def evaluate_lasso_hyperparams(df: pd.DataFrame, test_params: List[Any]):
    """ """
    plt.rc("font", size=18)
    plt.rcParams["figure.constrained_layout.use"] = True

    cv = TimeSeriesSplit(n_splits=5)

    mean_mse = []
    std_mse = []

    for i, q in enumerate(test_params):

        model = KNeighborsRegressor(weights="distance")
        temp_mse = []

        X = df.drop(target_var, axis=1).to_numpy()
        y = df["northBound"]

        metrics_named_tuple = cross_validation_model()

    plt.show()

    plt.errorbar(test_params, mean_mse, yerr=std_mse)
    plt.title(f"{feature_type} Cross Validation")
    plt.xlabel("Lag")
    plt.ylabel("Mean MSE")
    plt.xlim((1, test_params[-1]))


def evaluate_ridge_hyperparams():
    """ """


def evaluate_decision_tree_hyperparams():
    """ """


def evaluate_MLP_hyperparams():
    """ """

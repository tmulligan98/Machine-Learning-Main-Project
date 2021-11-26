from typing import List
from utilities import cross_validation_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
import matplotlib.pyplot as plt
from typing import Any


def evaluate_lasso_hyperparams(X, y, test_params: List[Any]):
    """ """
    plt.rc("font", size=18)
    plt.rcParams["figure.constrained_layout.use"] = True

    cv = TimeSeriesSplit(n_splits=5)

    mean_mse = []
    std_mse = []

    for c in test_params:

        model = linear_model.Lasso(alpha=1 / c)

        metrics_named_tuple = cross_validation_model(X, y, model, cv)
        mean_mse.append(np.array(metrics_named_tuple).mean())
        std_mse.append(np.array(metrics_named_tuple).std())

    plt.show()

    plt.errorbar(test_params, mean_mse, yerr=std_mse)
    plt.title(f"Lasso Cross Validation")
    plt.xlabel("Ci")
    plt.ylabel("MSE")
    plt.xlim((0, test_params[-1]))


def evaluate_ridge_hyperparams(X, y, test_params: List[Any]):
    """ """
    plt.rc("font", size=18)
    plt.rcParams["figure.constrained_layout.use"] = True

    cv = TimeSeriesSplit(n_splits=5)

    mean_mse = []
    std_mse = []

    for c in test_params:

        model = linear_model.Ridge(alpha=1 / (2 * c))

        metrics_named_tuple = cross_validation_model(X, y, model, cv)
        mean_mse.append(np.array(metrics_named_tuple).mean())
        std_mse.append(np.array(metrics_named_tuple).std())

    plt.show()

    plt.errorbar(test_params, mean_mse, yerr=std_mse)
    plt.title(f"Ridge Cross Validation")
    plt.xlabel("Ci")
    plt.ylabel("MSE")
    plt.xlim((0, test_params[-1]))


def evaluate_decision_tree_hyperparams(X,y,test_params: List[Any]):
    """ """
    plt.rc("font", size=18)
    plt.rcParams["figure.constrained_layout.use"] = True

    cv = TimeSeriesSplit(n_splits=5)

    mean_mse = []
    std_mse = []

    for depth in test_params:

        model = DecisionTreeRegressor(max_depth=depth)

        metrics_named_tuple = cross_validation_model(X, y, model, cv)
        mean_mse.append(np.array(metrics_named_tuple).mean())
        std_mse.append(np.array(metrics_named_tuple).std())

    plt.show()

    plt.errorbar(test_params, mean_mse, yerr=std_mse)
    plt.title(f"Decision Tree Cross Validation")
    plt.xlabel("Max Depth")
    plt.ylabel("MSE")
    plt.xlim((0, test_params[-1]))

    
    
  

def evaluate_MLP_hyperparams():
    """ """

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


def evaluate_decision_tree_hyperparams(X,y,x_axis):
    """ """


    # Create a random dataset
    #rng = np.random.RandomState(1)
    #X = np.sort(5 * rng.rand(80, 1), axis=0)
    #y = np.sin(X).ravel()
    #y[::5] += 3 * (0.5 - rng.rand(16))

    # Fit regression model
    regr_1 = DecisionTreeRegressor(max_depth=2)
    regr_2 = DecisionTreeRegressor(max_depth=5)
    regr_1.fit(X, y)
    regr_2.fit(X, y)

    # Predict
    
    y_1 = regr_1.predict(X)
    y_2 = regr_2.predict(X)

    # Plot the results
    plt.figure()
    y=y[:200]
    x_axis=np.arange(0,y.size,1)
    plt.scatter(x_axis,y, s=20, edgecolor="black", c="darkorange", label="data")
    plt.plot(x_axis,y_1[:200], color="cornflowerblue", label="max_depth=2", linewidth=2)
    plt.plot(x_axis,y_2[:200], color="yellowgreen", label="max_depth=5", linewidth=2)
    plt.xlabel("data")
    plt.ylabel("target")
    plt.title("Decision Tree Regression")
    plt.legend()
    plt.show()




def evaluate_MLP_hyperparams():
    """ """

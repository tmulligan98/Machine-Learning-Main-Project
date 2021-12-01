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
from typing import Any, Dict
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate, TimeSeriesSplit


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


def evaluate_decision_tree_hyperparams(X, y, test_params: List[Any]):
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


def evaluate_MLP_hidden_nodes(X: np.array, y: np.array, test_params: List[Any]):
    """ """
    plt.rc("font", size=18)
    plt.rcParams["figure.constrained_layout.use"] = True

    cv = TimeSeriesSplit(n_splits=5)

    mean_mse = []
    std_mse = []

    for n in test_params:

        model = MLPRegressor(hidden_layer_sizes=n, max_iter=500)

        metrics_named_tuple = cross_validation_model(X, y, model, cv)
        mean_mse.append(np.array(metrics_named_tuple).mean())
        std_mse.append(np.array(metrics_named_tuple).std())

    plt.show()

    plt.errorbar(test_params, mean_mse, yerr=std_mse)
    plt.title(f"MLP Hidden Layer Size Cross Validation")
    plt.xlabel("Hidden Nodes")
    plt.ylabel("MSE")
    plt.xlim((0, test_params[-1]))


def evaluate_MLP_penalty_weight(X: np.array, y: np.array, test_params: List[Any]):
    """ """
    plt.rc("font", size=18)
    plt.rcParams["figure.constrained_layout.use"] = True

    cv = TimeSeriesSplit(n_splits=5)

    mean_mse = []
    std_mse = []

    for ci in test_params:

        model = MLPRegressor(hidden_layer_sizes=5, alpha=1.0 / ci, max_iter=500)

        metrics_named_tuple = cross_validation_model(X, y, model, cv)
        mean_mse.append(np.array(metrics_named_tuple).mean())
        std_mse.append(np.array(metrics_named_tuple).std())

    plt.show()

    plt.errorbar(test_params, mean_mse, yerr=std_mse)
    plt.title(f"MLP L2 Penalty weight Cross Validation")
    plt.xlabel("L2 Penalty Weight")
    plt.ylabel("MSE")
    plt.xlim((0, test_params[-1]))


def evaluate_random_forest_hyperparams(
    X: np.array, y: np.array, param_grid: Dict[str, Any]
):
    """
    Going to use a gridsearch here to get the best hyper params. (There's a lot of hyper params to tune!)
    """
    plt.rc("font", size=18)
    plt.rcParams["figure.constrained_layout.use"] = True

    cv = TimeSeriesSplit(n_splits=5)
    model = RandomForestRegressor()
    rf_grid_search = RandomizedSearchCV(
        estimator=model, param_distributions=param_grid, cv=cv
    )

    rf_grid_search.fit(X, y)

    # Print the best params
    print(rf_grid_search.best_params_)


def evaluate_ada_boost_hyperparams(X, y, params_grid):
    ada_boost_model = AdaBoostRegressor()

    cv = TimeSeriesSplit(n_splits=5)
    # grid_search = GridSearchCV(estimator=ada_boost_model, param_grid=params_grid, n_jobs=-1, cv=cv)
    grid_search = RandomizedSearchCV(
        estimator=ada_boost_model, cv=cv, param_distributions=params_grid
    )

    grid_result = grid_search.fit(X, y)
    print("best params: " + str(grid_result.best_params_))


# ------------------K Nearest Neighbours------------------
def gaussian_weighting(distances: np.array, gamma=20) -> float:
    """
    Take in a numpy array of distances between the input x point, and each training point.
    Using this, compute the weights for each distance, then compute weighted mean.

    Args:
        distances : A numpy array of distances between the input x point

    Returns:
        A weighted mean of the computed gaussian weights

    """
    weights = np.exp(-gamma * (distances ** 2))
    return weights / np.sum(weights)


def evaluate_knn_k(X: np.array, y: np.array, test_params: List[Any]):
    plt.rc("font", size=18)
    plt.rcParams["figure.constrained_layout.use"] = True

    cv = TimeSeriesSplit(n_splits=5)

    mean_mse = []
    std_mse = []

    for k in test_params:

        model = KNeighborsRegressor(n_neighbors=k, weights="distance")

        metrics_named_tuple = cross_validation_model(X, y, model, cv)
        mean_mse.append(np.array(metrics_named_tuple).mean())
        std_mse.append(np.array(metrics_named_tuple).std())

    plt.show()

    plt.errorbar(test_params, mean_mse, yerr=std_mse)
    plt.title(f"K-Nearest Neighbours Cross Validation for k")
    plt.xlabel("Number of Neighbours (k)")
    plt.ylabel("MSE")
    plt.xlim((0, test_params[-1]))


def evaluate_knn_gamma(X: np.array, y: np.array):
    def gaussian_weighting_a(distances: np.array, gamma=200) -> float:
        weights = np.exp(-gamma * (distances ** 2))
        return weights / np.sum(weights)

    def gaussian_weighting_b(distances: np.array, gamma=500) -> float:
        weights = np.exp(-gamma * (distances ** 2))
        return weights / np.sum(weights)

    def gaussian_weighting_c(distances: np.array, gamma=1000) -> float:
        weights = np.exp(-gamma * (distances ** 2))
        return weights / np.sum(weights)

    def gaussian_weighting_d(distances: np.array, gamma=5000) -> float:
        weights = np.exp(-gamma * (distances ** 2))
        return weights / np.sum(weights)

    def gaussian_weighting_e(distances: np.array, gamma=10000) -> float:
        weights = np.exp(-gamma * (distances ** 2))
        return weights / np.sum(weights)

    plt.rc("font", size=18)
    plt.rcParams["figure.constrained_layout.use"] = True

    cv = TimeSeriesSplit(n_splits=5)

    gamma_range = [200, 500, 1000, 5000, 10000]
    gaussian_range_functions = {
        200: gaussian_weighting_a,
        500: gaussian_weighting_b,
        1000: gaussian_weighting_c,
        5000: gaussian_weighting_d,
        10000: gaussian_weighting_e,
    }

    mean_mse = []
    std_mse = []

    for g in gamma_range:

        model = KNeighborsRegressor(n_neighbors=5, weights=gaussian_range_functions[g])

        metrics_named_tuple = cross_validation_model(X, y, model, cv)
        mean_mse.append(np.array(metrics_named_tuple).mean())
        std_mse.append(np.array(metrics_named_tuple).std())

    plt.show()

    plt.errorbar(gamma_range, mean_mse, yerr=std_mse)
    plt.title(f"K-Nearest Neighbours Cross Validation for k")
    plt.xlabel("Number of Neighbours (k)")
    plt.ylabel("MSE")
    plt.xlim((0, gamma_range[-1]))

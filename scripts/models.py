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
from forecasting import n_one_step_ahead_prediction
import copy
from utilities import train_test_split
from visualisation import visualise_multiple_forecast_vs_true


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

def get_model_MSE(model,X,y):
    model.fit(X,y)

    cv =TimeSeriesSplit(5)
    scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring="neg_mean_squared_error",
            return_estimator=True,
        )

    return np.sqrt(-np.mean(scores["test_score"]))

month_map = {
        0:"Dec",
        1:"Nov",
        2:"Oct",
        3:"Sep",
        4:"Aug",
        5:"Jul",
        6:"Jun",
        7:"May",
        8:"Apr",
        9:"Mar",
        10:"Feb",
        11:"Jan"
    }

def calc_test_and_validation_MSE(test_scores,val_scores,val_week_scores,model_name,model,X,y,X_val,y_val,n_forecast,lagged_points,month_index):
    

    model.fit(X,y)
    test_scores[month_map.get(month_index)]=np.trunc(get_model_MSE(model,X,y))
    y_forecast_temp = n_one_step_ahead_prediction(model, X_val, n_forecast, lagged_points, relevant_lag_indexes=[0,1,2,3,4,5,11])
    val_scores[month_map.get(month_index)]=np.trunc(mean_squared_error(y_val, y_forecast_temp))
    val_week_scores[month_map.get(month_index)]=np.trunc(mean_squared_error(y_val[24*7:], y_forecast_temp[24*7:]))


def twleve_month_forecast(
    df_north,df_south,n_forecast,lag_range,model_name,model
    ):
    test_scores={}
    val_month_scores={}
    val_week_scores={}

    for month_index in range(11,-1,-1):

        date_time_axis,lagged_points, y_north_temp, X_north_temp, X_north_val_temp, y_north_val_temp = one_month_data_prep(df_north, df_south, n_forecast, lag_range, month_index)
        calc_test_and_validation_MSE(test_scores,val_month_scores,val_week_scores,model_name,model,X_north_temp,y_north_temp,X_north_val_temp,y_north_val_temp,n_forecast,lagged_points,month_index)

    return (test_scores,val_month_scores,val_week_scores)


def visualise_month_forecast(model_list,month,df_north, df_south, n_forecast, lag_range):
    month_index = list(month_map.keys())[list(month_map.values()).index(month)]
    date_time_axis,lagged_points, y_north_temp, X_north_temp, X_north_val_temp, y_north_val_temp=one_month_data_prep(df_north, df_south, n_forecast, lag_range, month_index)
    
    #put this in a loop for each model
    y_forecast_north_temp = []
    for model in model_list.values():
        y_forecast_north_temp.append(n_one_step_ahead_prediction(model, X_north_val_temp, n_forecast, lagged_points, relevant_lag_indexes=[0,1,2,3,4,5,11]))
    
    visualise_multiple_forecast_vs_true(date_time_axis,y_north_val_temp,y_forecast_north_temp,list(model_list.keys()))


def one_month_data_prep(df_north, df_south, n_forecast, lag_range, month_index):
    df_north_temp=df_north.copy()
    df_south_temp=df_south.copy()

        #start by removing 0 days for december, 
    remove_rows=24*30*month_index
    if(month_index!=0):
        df_north_temp=df_north_temp[:-remove_rows]
        df_south_temp=df_south_temp[:-remove_rows]
        
    date_time_axis = df_north_temp.iloc[-n_forecast:].index.to_numpy()

        # 12-previous points
    lagged_points = df_north_temp.to_numpy()[-12-lag_range-1:-12]  # This gives us last twelve rows of training data
    lagged_points = lagged_points[
            :, 3
        ]  # Get the volume for each row, these are our lagged points


        # Target Variable
    y_north_temp = df_north_temp["northBound"].to_numpy()
    y_south_temp = df_south_temp["southBound"].to_numpy()

        # Feature Vectors
    X_north_temp = df_north_temp.drop(columns=["northBound"]).to_numpy()
    X_south_temp = df_south_temp.drop(columns=["southBound"]).to_numpy()


        # Hold out a validation set
    (X_north_temp, X_north_val_temp, y_north_temp, y_north_val_temp) = train_test_split(X_north_temp, y_north_temp, test_size=n_forecast)
    (X_south_temp, X_south_val_temp, y_south_temp, y_south_val_temp) = train_test_split(X_south_temp, y_south_temp, test_size=n_forecast)
    
    return date_time_axis,lagged_points,y_north_temp,X_north_temp,X_north_val_temp,y_north_val_temp

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

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

def get_and_print_test_and_validation_MSE(test_scores,val_scores,val_week_scores,model_name,model,X,y,X_val,y_val,n_forecast,lagged_points,month_index):
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

    model.fit(X,y)
    

    #print(model_name+","+
     #   str(month_map.get(month_index))+
      #   ","+
       # str(np.trunc(get_model_MSE(model,X,y)))+
        #","+
        #str(np.trunc(mean_squared_error(y_val, y_forecast_north_temp)))
    #)
    
    test_scores[model_name][month_map.get(month_index)]=np.trunc(get_model_MSE(model,X,y))
    y_forecast_temp = n_one_step_ahead_prediction(model, X_val, n_forecast, lagged_points, relevant_lag_indexes=[0,1,2,3,4,5,11])
    val_scores[model_name][month_map.get(month_index)]=np.trunc(mean_squared_error(y_val, y_forecast_temp))
    val_week_scores[model_name][month_map.get(month_index)]=np.trunc(mean_squared_error(y_val[24*7:], y_forecast_temp[24*7:]))



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


def evaluate_ada_boost_hyperparams(X,y,params_grid):
    ada_boost_model=AdaBoostRegressor()
    

    cv = TimeSeriesSplit(n_splits=5)
    #grid_search = GridSearchCV(estimator=ada_boost_model, param_grid=params_grid, n_jobs=-1, cv=cv)
    grid_search = RandomizedSearchCV(estimator=ada_boost_model,cv=cv,param_distributions=params_grid)

    grid_result = grid_search.fit(X, y)
    print("best params: " +str(grid_result.best_params_))

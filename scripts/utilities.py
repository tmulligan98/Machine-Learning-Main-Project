import collections
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import glob
import os
import time
from datetime import date, datetime
from sklearn import neighbors
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from visualisation import visualise_features
from pathlib import Path
from typing import List, Any, Dict
import matplotlib.pyplot as plt

YEARS = ["2018","2017","2016"]


def find_csvs() -> List[str]:
    """
    Function to help us find csv files in our sub-directories
    """
    path = os.path.dirname(os.path.realpath(__file__))
    path = str(Path(path).parent)

    all_files: List[str] = []
    files: List[str]
    for year in YEARS:
        files = glob.glob(path + f"/Data/Data{year}/preprocessed/*.csv")
        all_files.extend(files)
    return all_files


def load_dataframe():
    """
    Function to fetch data from our preprocessed CSVs.
    """

    # Lambda function to parse datetime information
    dateparse = lambda dates: [datetime.strptime(d, "%Y/%m/%d %H:%M") for d in dates]

    # Load data from csvs
    all_files = find_csvs()
    df = pd.DataFrame()
    cur_df: pd.DataFrame
    for file in all_files:
        cur_df = pd.read_csv(file)
        df = df.append(cur_df)
    # Tidy a little
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(by=["datetime"])
    df = df.set_index("datetime")

    # replace '-' to -1 to change dataframe type to int
    df = df.replace("-", -1)
    df["northBound"] = df["northBound"].astype(str).astype(np.float32).astype(int)
    df["southBound"] = df["southBound"].astype(str).astype(np.float32).astype(int)
    df = df.replace(-1, np.NaN)  # change -1 to NaN for interpolating NaN values

    # impute missing data with linear interpolation
    df["northBound"].interpolate(method="linear", axis=0, inplace=True)
    df["southBound"].interpolate(method="linear", axis=0, inplace=True)

    return df


def cross_validation_feature_params(
    test_params: List[Any], df: pd.DataFrame, feature_type: str, target_var: str
):
    """
    Function to carry out cross validation, given a timeseries split

    Args:
        df : Pandas dataframe
        test_params: A list of parameters to carry out cross validation on
        feature_type : A string denoting the type of feature we are using for this particular evaluation
        target_var : The target variable we want to predict
    """

    plt.rc("font", size=18)
    plt.rcParams["figure.constrained_layout.use"] = True

    cv = TimeSeriesSplit(n_splits=5)

    mean_mse = []
    std_mse = []

    for i, q in enumerate(test_params):

        model = KNeighborsRegressor(weights="distance")
        temp_mse = []
        if feature_type == "lag":
            df[feature_type] = df[target_var].shift(q)
        elif feature_type == "rolling_window_mean":
            df[feature_type] = df[target_var].rolling(window=q).mean()
        elif feature_type == "rolling_window_max":
            df[feature_type] = df[target_var].rolling(window=q).max()
        elif feature_type == "rolling_window_min":
            df[feature_type] = df[target_var].rolling(window=q).min()

        df[feature_type] = df[feature_type].fillna(0)

        y = df[target_var]
        X = df.drop(target_var, axis=1).to_numpy()

        for train, test in cv.split(X):
            model.fit(X[train], y[train])
            ypred = model.predict(X[test])
            temp_mse.append(mean_squared_error(y[test], ypred))

        mean_mse.append(np.array(temp_mse).mean())
        std_mse.append(np.array(temp_mse).std())

    plt.show()

    plt.errorbar(test_params, mean_mse, yerr=std_mse)
    plt.title(f"{feature_type} Cross Validation")
    plt.xlabel("Lag")
    plt.ylabel("Mean MSE")
    plt.xlim((1, test_params[-1]))


def cross_validation_model(
    X: np.array, y: np.array, model: Any, cv: TimeSeriesSplit
) -> List[Any]:
    """
    Function to carry out a chained cross validation using TimeSeriesSplit

    Args:
        X : A numpy array of the feature vector
        y: A numpy array of the target variable
        model : Any given trained model to use
        cv : TimeSeriesSplit

    Returns:
        A named tuple containing the mean and std Mean Squared Error
    """

    Metrics = collections.namedtuple("Metrics", ["mean", "std"])

    mean_mse = []
    std_mse = []
    temp_mse = []

    for train, test in cv.split(X):
        model.fit(X[train], y[train])
        ypred = model.predict(X[test])
        temp_mse.append(mean_squared_error(y[test], ypred))

    return temp_mse


def performance(df: pd.DataFrame, target_var: str):
    """
    Function to give us a baseline performance measure, using cross validation and K-Means

    Args:
        df (Pandas Dataframe) : A Dataframe containing the preprocessed traffic data.
        target_var : Column name of the variable we are attempting to predict

    Returns:
        Pandas dataframe of data to be used in models
    """
    cv = TimeSeriesSplit(n_splits=5)
    neighbors_model = KNeighborsRegressor(weights="distance")

    X = df.drop(target_var, axis=1)
    y = df[target_var]

    scores = cross_validate(
        neighbors_model,
        X,
        y,
        cv=cv,
        scoring="neg_mean_squared_error",
        return_estimator=True,
    )

    # Base RMSLE
    base_mse = np.sqrt(-np.mean(scores["test_score"]))
    print(f"Base MSE for {target_var} traffic is: {format(base_mse)}")


# if __name__ == "__main__":
#     # Get our dataframe
#     df = load_dataframe()

#     # We're going to hold out 12 hours of data points to predict on!~
#     def train_test_split(X, y, test_size):
#         return (X[:-test_size, :], X[-test_size:, :], y[:-test_size], y[-test_size:])

#     df_north = df.drop(columns=["southBound"])

#     df_north["volume_lag_1"] = df_north["northBound"].shift(1, fill_value=0)
#     df_north["volume_lag_1_diff"] = df_north["volume_lag_1"] - df_north[
#         "northBound"
#     ].shift(2, fill_value=0)

#     # Target Variable
#     y_north = df_north["northBound"].to_numpy()
#     # Feature Vectors
#     X_north = df_north.drop(columns=["northBound"]).to_numpy()

#     # Hold out a validation set
#     (X_north, X_north_val, y_north, y_north_val) = train_test_split(
#         X_north, y_north, test_size=12
#     )

#     neighbors_model = KNeighborsRegressor(weights="distance").fit(X_north, y_north)
#     predictions = n_one_step_ahead_prediction(neighbors_model, X_north_val, n=12)

#     # Base features performance, where we use K Nearest Neighbors.
#     # Here is our baseline, now we add features.
#     # base_performance(df)
#     # visualise_features(["dayOfWeek", "month", "time"], df, "Base Features")

#     # Now have to add more (basic) features like:
#     # Business quarter of the year, week of year, day of year etc
#     # Also, use holidays as features!

#     # Then we can add more features like:
#     # Lagging, rolling window, expanding window
#     # Then maybe also some domain specific features

#     # Once that's done, we can start training (and validating) some models!

from sklearn.tree import DecisionTreeRegressor

def get_decision_tree_models(max_depth_array):
    models=[]
    for depth in max_depth_array:
        models.append(DecisionTreeRegressor(max_depth=depth))
    return models
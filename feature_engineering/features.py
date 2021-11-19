import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import glob
import os
import time
from datetime import date, datetime
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.neighbors import KNeighborsRegressor
from visualisation import visualise_features


def load_dataframe():
    """
    Function to fetch data from our preprocessed CSVs.
    """

    # Lambda function to parse datetime information
    dateparse = lambda dates: [datetime.strptime(d, "%Y/%m/%d %H:%M") for d in dates]

    # Load data from csvs
    path = "/mnt/c/Users/Tom/Documents/ML/ML_main_project/Data2017/preprocessed"
    # all_files = glob.glob(path + "/*.csv")
    df = pd.DataFrame()
    # cur_df : pd.DataFrame
    # for file in all_files:
    #     cur_df = pd.read_csv(file)
    #     df = df.append(cur_df, ignore_index=True)
    print(path)
    df = pd.read_csv(
        f"{path}/Jan.csv",
        parse_dates=["datetime"],
        date_parser=dateparse,
        index_col="datetime",
    )
    return df


def performance(df: pd.DataFrame):
    """
    Function to extract time series data in a way that we can process
    in our models.

    Args:
        df (Pandas Dataframe) : A Dataframe containing the preprocessed traffic data.

    Returns:
        Pandas dataframe of data to be used in models
    """
    cv = TimeSeriesSplit(n_splits=5)
    neighbors_model = KNeighborsRegressor(weights="distance")

    X = df.drop("volume", axis=1)
    y = df[["volume"]]

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
    print("Base MSE is: {:.5f}".format(base_mse))


# def short_term_features():
#     """
#     Function to fetch time series features for short term
#     trends. (Every hour)
#     """


# def daily_trend():

# def weekly_trend():


# if __name__ == "__main__":
#     # Get our dataframe
#     df = load_dataframe()

#     # Base features performance, where we use K Nearest Neighbors.
#     # Here is our baseline, now we add features.
#     base_performance(df)
#     visualise_features(["dayOfWeek", "month", "time"], df, "Base Features")

#     # Now have to add more (basic) features like:
#     # Business quarter of the year, week of year, day of year etc
#     # Also, use holidays as features!

#     # Then we can add more features like:
#     # Lagging, rolling window, expanding window
#     # Then maybe also some domain specific features

#     # Once that's done, we can start training (and validating) some models!

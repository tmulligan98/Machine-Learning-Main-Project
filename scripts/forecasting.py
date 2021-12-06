from typing import Any, List, Dict
from sklearn.neighbors import KNeighborsRegressor
from utilities import load_dataframe
import numpy as np


def construct_lag_features(
    new_prediction: float, existing_lags: np.array, relevant_lag_indexes: List[int]
) -> np.array:
    """
    Function to construct lag features, based on a list of indexes we want to take.
    For example, if the list of indexes is [1,2,3,6], we will only take those lag features from
    the numpy array (existing_lags) containing all lags up to 12.

    Args:
        new_prediction : The new prediction to prepend to the existing lags
        existing_lags : A numpy array containing lag features up to 12 hours previously
        relevant_lag_indexes : A list of relevant lag indexes to use.

    Returns:
        The full list of previous volumes up to twelve time steps previously
        AND
        The relevant output lag features
    """
    existing_lags = np.append(existing_lags, new_prediction)
    existing_lags = np.delete(existing_lags, 0)
    output_features = []
    for ind in relevant_lag_indexes:
        output_features.append(existing_lags[-ind - 1])

    # Add difference feature
    output_features.append(new_prediction - existing_lags[-2])
    return existing_lags, np.array(output_features)


# For validation/validation
def one_step_ahead_prediction(trained_model: Any, testX_single: np.array) -> int:
    """
    Given a previous datapoint, predict one step ahead.

    Args:

    Returns:
    """
    ypred = trained_model.predict(np.array([testX_single]))

    # This will give us the prediction for one step ahead.
    # Use this predicition as the lag of the next point.
    return ypred[0]


def n_one_step_ahead_prediction(
    trained_model: Any,
    testX: np.array,
    n: int,
    previous_data: np.array,
    relevant_lag_indexes: List[int],
):
    """ """

    # Predict first step. (This first one has it's lag features already there from the last
    # element of the test set!)
    y_predictions = []
    # Generate our initial lagged features
    
    ypred = one_step_ahead_prediction(trained_model, testX[0])

    y_predictions.append(ypred)

    for i in range(1, n):
        # Using previous prediction, add to the feature
        temp = np.delete(testX[i], list(range(8, len(testX[0]))))
        previous_data, input_feature_vector = construct_lag_features(
            ypred, previous_data, relevant_lag_indexes
        )
        temp = np.concatenate([temp, input_feature_vector])

        # Prediction
        ypred = one_step_ahead_prediction(trained_model, temp)
        y_predictions.append(ypred)

    return y_predictions


# if __name__ == "__main__":
#     # Get our dataframe
#     df = load_dataframe()

#     # We're going to hold out 12 hours of data points to predict on!~
#     def train_test_split(X, y, test_size):
#         return (X[:-test_size, :], X[-test_size:, :], y[:-test_size], y[-test_size:])

#     df_north = df.drop(columns=["southBound"])

#     df_north["volume_lag_1"] = df_north["northBound"].shift(1, fill_value=0)
#     df_north["volume_lag_2"] = df_north["northBound"].shift(2, fill_value=0)
#     df_north["volume_lag_3"] = df_north["northBound"].shift(3, fill_value=0)
#     df_north["volume_lag_4"] = df_north["northBound"].shift(4, fill_value=0)
#     df_north["volume_lag_5"] = df_north["northBound"].shift(5, fill_value=0)
#     df_north["volume_lag_6"] = df_north["northBound"].shift(6, fill_value=0)
#     df_north["volume_lag_12"] = df_north["northBound"].shift(12, fill_value=0)
#     df_north["volume_lag_1_diff"] = df_north["volume_lag_1"] - df_north[
#         "northBound"
#     ].shift(2, fill_value=0)

#     lagged_points = df_north.to_numpy()[
#         -13 - 12 : -12
#     ]  # This gives us last twelve rows of training data
#     lagged_points = lagged_points[
#         :, 3
#     ]  # Get the volume for each row, these are our lagged points

#     # Target Variable
#     y_north = df_north["northBound"].to_numpy()
#     # Feature Vectors
#     X_north = df_north.drop(columns=["northBound"]).to_numpy()

#     # Hold out a validation set
#     (X_north, X_north_val, y_north, y_north_val) = train_test_split(
#         X_north, y_north, test_size=12
#     )

#     neighbors_model = KNeighborsRegressor(weights="distance").fit(X_north, y_north)
#     predictions = n_one_step_ahead_prediction(
#         neighbors_model, X_north_val, 12, lagged_points, [0, 1, 2, 3, 4, 5, 11]
#     )
#     print()

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

import pandas as pd
import features.features_utils as features_utils


def add_hourly_lagged_deltas(p_df, p_colname, n_lags=72, n_minutes=60):
    """
    Add hourly lagged delta columns for the last 3 days (72 hours) to a DataFrame with 1-minute data.

    """
    tempdf = p_df[[p_colname]].copy()
    col_names = []
    feature_list = []

    # Calculate hourly lagged deltas for each hour up to n_hours
    for lag in range(1, n_lags + 1):
        # Calculate the lag in minutes
        lag_minutes = n_minutes * lag

        # Create the lagged series
        tempdf[p_colname + "lagged"] = tempdf[p_colname].shift(lag_minutes)

        # Calculate the delta
        feature_id = f"F_delta_rel_{p_colname}_{lag_minutes}" + "_f16"
        feature_obj = features_utils.Feature(
            feature_id=feature_id,
            activated=True,
            feature_type=features_utils.FeatureType.LAG_DELTAS,
        )

        tempdf[feature_id] = (tempdf[p_colname] - tempdf[p_colname + "lagged"]) / tempdf[
            p_colname + "lagged"
        ]
        col_names.append(feature_id)
        feature_list.append(feature_obj)

    p_df[col_names] = tempdf[col_names]

    return p_df, col_names, feature_list

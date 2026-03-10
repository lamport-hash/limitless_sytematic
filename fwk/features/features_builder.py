import pandas as pd
import features.features_utils as features_utils
from features.feature_ta_utils import (
    numba_parabolic_sar,
    numba_atr,
    numba_rsi,
    numba_ema,
    rolling_std,
    add_accumulation_distribution_index,
)


def add_emas(p_df, p_colname, p_periods_emas):
    """
    Add emas and related cols

    """
    mid_col = p_colname
    tempdf = p_df[[p_colname]].copy()
    col_names = []
    feature_list = []
    epsilon = 0.0001

    # EMAs
    periods = p_periods_emas
    ema_names = []
    for i in range(len(periods)):
        ema_id = "F_ema_" + str(periods[i]) + "_" + mid_col + "_f32"
        ema_names.append(ema_id)
        col_names.append(ema_id)
        tempdf[ema_names[i]] = numba_ema(tempdf[mid_col].to_numpy(), periods[i])
        feature_obj = features_utils.Feature(
            feature_id=ema_id,
            activated=False,
            feature_type=features_utils.FeatureType.EMA,
        )
        feature_list.append(feature_obj)

    # delta_rel to other ema
    for i in range(len(periods) - 1):
        name_id = (
            "F_delta_rel_ema_"
            + str(periods[i])
            + "_"
            + str(periods[i + 1])
            + "_"
            + mid_col
            + "_f16"
        )
        tempdf[name_id] = (tempdf[ema_names[i + 1]] - tempdf[ema_names[i]]) / (
            tempdf[ema_names[i]] + epsilon
        )
        col_names.append(name_id)
        feature_obj = features_utils.Feature(
            feature_id=name_id,
            activated=True,
            feature_type=features_utils.FeatureType.SPREAD_REL_EMA,
        )
        feature_list.append(feature_obj)

    # diff_rel to price
    for i in range(len(periods) - 1):
        name_id = "F_diff_rel_ema_2_" + mid_col + "_" + str(periods[i]) + "_f16"
        tempdf[name_id] = (tempdf[ema_names[i + 1]] - tempdf[mid_col]) / (tempdf[mid_col] + epsilon)
        col_names.append(name_id)
        feature_obj = features_utils.Feature(
            feature_id=name_id,
            activated=True,
            feature_type=features_utils.FeatureType.DIFF_REL_EMA_MID,
        )
        feature_list.append(feature_obj)

    p_df[col_names] = tempdf[col_names]

    return p_df, col_names, feature_list

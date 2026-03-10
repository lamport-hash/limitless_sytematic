import pandas as pd
import numpy as np

from core.enums import (
    g_close_col,
    g_open_col,
    g_high_col,
    g_low_col,
    g_volume_col,
    g_mid_col,
    g_mid2_col,
    g_close_time_col,
    g_open_time_col,
    g_index_col,
    g_qa_vol_col,
    g_nt_col,
    g_la_vol_col,
    g_lqa_vol_col,
)
import features.features_utils as features_utils


def feature_volume_ma_ratio(df, p_window=20):
    """Ratio of current volume to its moving average. Signals unusual activity."""
    vol_ma = df[g_volume_col].rolling(window=p_window).mean()
    return df[g_volume_col] / vol_ma


def feature_volume_dollar(df):
    """Calculate dollar volume (if 'g_qa_vol_col' is not already dollar volume)."""
    # Often 'g_qa_vol_col' IS dollar volume, so you might not need this.
    # This function calculates it if you only have base volume.
    return df[g_close_col] * df[g_volume_col]


def feature_volume_dollar_ma_ratio(df, p_window=20):
    """Ratio of current dollar volume to its moving average."""
    dollar_vol = df[g_close_col] * df[g_volume_col]
    dollar_vol_ma = dollar_vol.rolling(window=p_window).mean()
    return dollar_vol / dollar_vol_ma


def feature_volume_obv(df):
    """On-Balance Volume, a cumulative volume-based momentum indicator."""
    price_change = df[g_close_col].diff()
    obv = (np.sign(price_change) * df[g_volume_col]).fillna(0).cumsum()
    return obv


def feature_trade_buy_ratio_base(df):
    """
    The ratio of buyer-initiated volume to total volume.
    The primary measure of buying pressure.
    """
    return df[g_la_vol_col] / df[g_volume_col]


def feature_trade_buy_ratio_quote(df):
    """
    The ratio of buyer-initiated dollar volume to total dollar volume.
    Often a cleaner signal than base volume ratio.
    """
    # g_lqa_vol_col is often the direct dollar amount bought by "takers"
    return df[g_lqa_vol_col] / df[g_qa_vol_col]


def feature_trade_buy_pressure_ma_ratio(df, p_window=20, ratio_type="base"):
    """
    Measures if current buying pressure is high relative to recent history.
    """
    if ratio_type == "base":
        current_ratio = df[g_la_vol_col] / df[g_volume_col]
    else:  # 'quote'
        current_ratio = df[g_lqa_vol_col] / df[g_qa_vol_col]

    ratio_ma = current_ratio.rolling(window=p_window).mean()
    return current_ratio / ratio_ma


def feature_trade_buy_pressure_ma_ratio_base(df, p_window=20):
    return feature_trade_buy_pressure_ma_ratio(df, p_window, "base")


def feature_trade_buy_pressure_ma_ratio_quote(df, p_window=20):
    return feature_trade_buy_pressure_ma_ratio(df, p_window, "quote")


def feature_trade_avg_trade_size_dollar(df):
    """Average size of a trade in quote (dollar) terms."""
    return df[g_qa_vol_col] / df[g_nt_col]


def feature_trade_avg_trade_size_dollar_ma_ratio(df, p_window=20):
    """Unusual average trade size can signal institutional vs. retail activity."""
    avg_size = df[g_qa_vol_col] / df[g_nt_col]
    avg_size_ma = avg_size.rolling(window=p_window).mean()
    return avg_size / avg_size_ma


def feature_trade_trade_count_ma_ratio(df, p_window=20):
    """Ratio of current number of trades to its moving average."""
    trade_count_ma = df[g_nt_col].rolling(window=p_window).mean()
    return df[g_nt_col] / trade_count_ma


def feature_trade_net_taker_flow(df):
    """
    Approximates the net dollar volume flow from takers.
    Taker Buy Quote Volume - (Total Quote Volume - Taker Buy Quote Volume)
    Simplifies to: 2 * Taker Buy Quote Volume - Total Quote Volume
    """
    return (2 * df[g_lqa_vol_col]) - df[g_qa_vol_col]


def add_all_orderflow_features(df, selected_columns, selected_features, p_windows=[5, 240]):
    """
    Enriches the input dataframe with engineered features using the specified functions.
    Each feature column is prefixed with 'F_' and postfixed with '_f64'.
    Window parameters are included in the column name where applicable.

    Parameters:
    df (pd.DataFrame): Input dataframe containing the raw data
    g_*_col (str): Column names in the dataframe for each data field

    Returns:
    pd.DataFrame: Enriched dataframe with original data and new feature columns
    """

    # Make a copy to avoid modifying the original dataframe
    enriched_df = df.copy()

    ids_and_functions = [
        # Volume-based features
        ("F_volume_dollar_f64", feature_volume_dollar, features_utils.FeatureType.V_USD),
        ("F_volume_obv_f64", feature_volume_obv, features_utils.FeatureType.V_OBV),
        # Trade-based features
        (
            "F_trade_buy_ratio_base_f64",
            feature_trade_buy_ratio_base,
            features_utils.FeatureType.T_BUY_RATIO_B,
        ),
        (
            "F_trade_buy_ratio_quote_f64",
            feature_trade_buy_ratio_quote,
            features_utils.FeatureType.T_BUY_RATIO_Q,
        ),
        (
            "F_trade_avg_trade_size_dollar_f64",
            feature_trade_avg_trade_size_dollar,
            features_utils.FeatureType.T_AVG_TS_USD,
        ),
        (
            "F_trade_net_taker_flow_f64",
            feature_trade_net_taker_flow,
            features_utils.FeatureType.T_NET_TAKER_FLOW,
        ),
    ]

    for id, func, type in ids_and_functions:
        enriched_df[id] = func(enriched_df)
        selected_columns.append(id)
        feature_obj = features_utils.Feature(
            feature_id=id,
            activated=True,
            feature_type=type,
        )
        selected_features.append(feature_obj)

    ids_and_functions_win = []

    for window in p_windows:
        # windows based features
        colname = "F_volume_ma_ratio_" + str(window) + "_f64"
        ids_and_functions_win.append(
            (colname, feature_volume_ma_ratio, features_utils.FeatureType.V_MA_RATIO, window)
        )

        colname = "F_volume_dollar_ma_ratio_" + str(window) + "_f64"
        ids_and_functions_win.append(
            (
                colname,
                feature_volume_dollar_ma_ratio,
                features_utils.FeatureType.V_USD_MA_RATIO,
                window,
            )
        )

        colname = "F_trade_avg_trade_size_dollar_ma_ratio_" + str(window) + "_f64"
        ids_and_functions_win.append(
            (
                colname,
                feature_trade_avg_trade_size_dollar_ma_ratio,
                features_utils.FeatureType.T_AVG_TS_USD_MA_RATIO,
                window,
            )
        )

        # Buy pressure features
        colname = "F_trade_buy_pressure_ma_ratio_base_" + str(window) + "_f64"
        ids_and_functions_win.append(
            (
                colname,
                feature_trade_buy_pressure_ma_ratio_base,
                features_utils.FeatureType.T_BUY_MA_RATIO_B,
                window,
            )
        )

        colname = "F_trade_buy_pressure_ma_ratio_quote_" + str(window) + "_f64"
        ids_and_functions_win.append(
            (
                colname,
                feature_trade_buy_pressure_ma_ratio_quote,
                features_utils.FeatureType.T_BUY_MA_RATIO_Q,
                window,
            )
        )

        # Trade count features
        colname = "F_trade_trade_count_ma_ratio_" + str(window) + "_f64"
        ids_and_functions_win.append(
            (
                colname,
                feature_trade_trade_count_ma_ratio,
                features_utils.FeatureType.T_TC_MA_RATIO,
                window,
            )
        )

    for id, func, type, window in ids_and_functions_win:
        enriched_df[id] = func(enriched_df, window)
        selected_columns.append(id)
        feature_obj = features_utils.Feature(
            feature_id=id,
            activated=True,
            feature_type=type,
        )
        selected_features.append(feature_obj)

    return enriched_df, selected_columns, selected_features

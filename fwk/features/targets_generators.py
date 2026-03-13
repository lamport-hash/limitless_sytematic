import logging
import re
from pathlib import Path
from typing import Dict, Any, Union

import pandas as pd
import numpy as np
from ruamel.yaml import YAML

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
    g_precision,
)
from merger.merger_utils import g_yaml_t_classification


logger = logging.getLogger(__name__)


# all the below functions assume that the df contains the following columns
# g_open_col
# g_high_col
# g_low_col
# g_close_col
# g_index_col


def gen_perfect_signal_class(
    df: pd.DataFrame,
    close_col: str,
    high_col: str,
    low_col: str,
    upstrong_val: float,
    downstrong_val: float,
    flat_val: float,
    N_periods: int = 60,
) -> tuple[pd.Series, str]:
    """
    Calculate perfect signal class based on candle data and thresholds.

    Args:
        df: DataFrame containing columns for 'high', 'low', 'close'
        upstrong_val (float): Threshold for strong upward class (> this = 2)
        downstrong_val (float): Threshold for strong downward class (< this = -2)
        flat_val (float): Threshold between neutral and weak movement (e.g., 0.005 or 0.5%)
        N_periods (int): Number of periods ahead to look for max high and min low

    Returns:
        pd.Series: Signal classes: 2 (strong up), 1 (weak up), 0 (neutral),
                   -1 (weak down), -2 (strong down)
    """
    try:
        # Initialize result series
        signal_class = pd.Series(0, index=df.index, dtype="int8")

        # Extract price series
        highs = df[high_col].values
        lows = df[low_col].values
        closes = df[close_col].values

        # For each row, compute max high and min low over next N periods
        for i in range(len(df) - N_periods):
            current_close = closes[i]

            # Look ahead N periods: indices [i+1] to [i+N]
            future_highs = highs[i + 1 : i + 1 + N_periods]
            future_lows = lows[i + 1 : i + 1 + N_periods]

            # Compute max and min returns relative to current close
            max_return = (future_highs.max() - current_close) / current_close
            min_return = (future_lows.min() - current_close) / current_close

            # Classify based on conditions
            if max_return >= upstrong_val:
                signal_class[i] = 2
            elif max_return >= flat_val:
                signal_class[i] = 1
            elif min_return <= downstrong_val:
                signal_class[i] = -2
            elif min_return <= -flat_val:  # Note: flat_val is positive, so we use negative
                signal_class[i] = -1
            # else remains 0

        # Last N_periods rows have no future data → leave as 0 (or NaN if preferred)
        # We keep as 0 for now
        indicator_name = f"PerfectClass_N{N_periods}"

        return signal_class, indicator_name

    except Exception as e:
        raise ValueError(f"Error calculating perfect signal class: {e}")


def gen_perfect_stoploss_signal_class(
    df: pd.DataFrame,
    close_col: str,
    high_col: str,
    low_col: str,
    up_objective: float,
    up_stoploss: float,
    down_objective: float,
    down_stoploss: float,
    N_periods: int = 60,
) -> tuple[pd.Series, str]:
    """
    Generate trading signals based on future price movements over N candles.

    This function evaluates the maximum upside and minimum downside returns
    over the next N candles to generate directional signals (1, -1, 0).

    Parameters:
        df (pd.DataFrame): DataFrame with OHLC columns and a datetime-like index.
        upobjective (float): Target upside return threshold (e.g., 0.02 for 2%).
        upstoploss (float): Maximum allowable downside during an uptrend (e.g., -0.01 for -1%).
        downobjective (float): Target downside return threshold (e.g., -0.02 for -2%).
        downstoploss (float): Maximum allowable upside during a downtrend (e.g., 0.01 for +1%).
        N (int): Number of future candles to look ahead. Default is 60.

    Returns:
        tuple[pd.Series, str]:
            - pd.Series: Signal column with values {1, -1, 0}, same length as df.
                         Last N rows are NaN (no future data).
            - str: Indicator name as a descriptive string, e.g., "PerfectStopLoss_N60"
    """

    # Initialize signal column
    signals = pd.Series(0, index=df.index, dtype="int8")

    # Extract close prices for return calculation
    closes = df[close_col].values
    highs = df[high_col].values
    lows = df[low_col].values

    # Precompute max and min returns over next N candles for each row
    for i in range(len(df) - N_periods):  # Only process rows where we have N future candles
        current_close = closes[i]

        # Look at next N candles' high and low prices
        future_highs = highs[i + 1 : i + 1 + N_periods]
        future_lows = lows[i + 1 : i + 1 + N_periods]

        # Calculate max return (highest high relative to current close)
        max_return = np.max((future_highs - current_close) / current_close)

        # Calculate min return (lowest low relative to current close)
        min_return = np.min((future_lows - current_close) / current_close)

        # Signal logic:
        if min_return >= up_stoploss and max_return >= up_objective:
            signals[i] = 1
        elif max_return <= down_stoploss and min_return <= down_objective:
            signals[i] = -1
        # else remains 0

    indicator_name = f"PerfectStopLoss_N{N_periods}"

    return signals, indicator_name


def add_min_max_log_returns(p_df, N, col_name, valid_col_name, is_log=True):
    """
    Adds minimum and maximum returns over the next N periods to the DataFrame.

    Args:
        p_df (pd.DataFrame): Input DataFrame with a 'col_name' column and 'valid_col_name' column
        N (int): Number of periods to look ahead.

    Returns:
        pd.DataFrame: DataFrame with added 'min_return' and 'max_return' columns.
    """
    # Create columns for shifted values of 'mid'
    shifted = np.column_stack([p_df[col_name].shift(-i) for i in range(1, N + 1)])
    shifted_low = np.column_stack([p_df[g_low_col].shift(-i) for i in range(1, N + 1)])
    shifted_high = np.column_stack([p_df[g_high_col].shift(-i) for i in range(1, N + 1)])

    if is_log:
        # Calculate returns (row-wise operations)
        returns = shifted / p_df[col_name].values[:, None]
        returns_low = shifted_low / p_df[col_name].values[:, None]
        returns_high = shifted_high / p_df[col_name].values[:, None]
    else:
        # Calculate returns (row-wise operations)
        returns = (shifted - p_df[col_name].values[:, None]) / p_df[col_name].values[:, None]
        returns_low = (shifted_low - p_df[col_name].values[:, None]) / p_df[col_name].values[
            :, None
        ]
        returns_high = (shifted_high - p_df[col_name].values[:, None]) / p_df[col_name].values[
            :, None
        ]

    # Calculate min and max along the rows
    min_return = np.min(returns, axis=1)
    max_return = np.max(returns, axis=1)
    mid_return = (min_return + max_return) / 2  # todo check this idea, or take the last mid value??

    min_return_low = np.min(returns_low, axis=1)
    max_return_high = np.max(returns_high, axis=1)

    if is_log:
        min_return = np.log(min_return)
        max_return = np.log(max_return)
        mid_return = np.log(mid_return)

        max_spread = np.log(max_return_high) - np.log(min_return_low)

        # Add new columns
        name_min = col_name + "_min_LR_" + str(N) + "_f16"
        name_max = col_name + "_max_LR_" + str(N) + "_f16"
        name_mid = col_name + "_mid_LR_" + str(N) + "_f16"
        name_spread = col_name + "_spread_LR_" + str(N) + "_f16"

    else:
        # Add new columns
        max_spread = max_return_high - min_return_low
        name_min = col_name + "_min_R_" + str(N) + "_f16"
        name_max = col_name + "_max_R_" + str(N) + "_f16"
        name_mid = col_name + "_mid_R_" + str(N) + "_f16"
        name_spread = col_name + "_spread_R_" + str(N) + "_f16"

    p_df[name_min] = min_return
    p_df[name_max] = max_return
    p_df[name_mid] = mid_return
    p_df[name_spread] = max_spread

    # update validity
    p_df = p_df.reset_index(drop=True)
    p_df.loc[p_df.shape[0] - N :, valid_col_name] = False

    return name_min, name_max, name_mid, name_spread


# Function mapping — add yours here
TARGETS_FUNCTIONS = {
    "gen_perfect_signal_class": gen_perfect_signal_class,
    "gen_perfect_stoploss_signal_class": gen_perfect_stoploss_signal_class,
}


def compute_single_asset_target_from_dict(df, id_single, params_dict, target_df):
    function_name = params_dict.get("function")
    asset = params_dict.get("asset")
    # Extract params (dict) — will be unpacked into function
    params = params_dict.get("params", {})
    # modify to point to the correct col of the asset
    for key, value in params.items():
        if key.endswith("_col"):
            value = asset + "_" + value
            params[key] = value

    # Validate function exists
    if function_name not in TARGETS_FUNCTIONS:
        print(f"Warning: Unknown function '{function_name}'")

    # Get the function
    func = TARGETS_FUNCTIONS[function_name]

    try:
        signal, signal_name = func(df, **params)
        target_df["T_" + id_single + "_" + signal_name] = signal
        print(f"✅ Generated target '{signal_name}' using {function_name}")
    except Exception as e:
        print(f" Generating target '{signal_name}' using {function_name}")
        print(f"❌ Error generating target: {e}")
    return target_df


def compute_spread_asset_target_from_dict(df, id_spread, params_dict, target_df):
    function_name = params_dict.get("function")
    asset1 = params_dict.get("asset1")
    factor1 = params_dict.get("factor1")
    asset2 = params_dict.get("asset2")
    factor2 = params_dict.get("factor2")

    if not all([function_name, asset1, asset2]):
        print("❌ Missing required fields for spread target")
        return

    params = params_dict.get("params", {})
    # compute spread cols for all col params
    for key, value in params.items():
        if key.endswith("_col"):
            col_name = value
            # Compute spread: asset1 - asset2
            spread_col_name = id_spread + "_" + col_name
            df[spread_col_name] = (
                factor1 * df[asset1 + "_" + col_name] - factor2 * df[asset2 + "_" + col_name]
            )

    # Now reuse single_asset function with the spread column
    params_dict["asset"] = id_spread  # created spread col

    target_df = compute_single_asset_target_from_dict(
        df, id_spread, params_dict, target_df
    )  # Reuse logic
    return target_df


def setup_framework_indicators(config_yaml, df, target_df):
    """Main setup function to load new_targets from config and call their functions."""

    results = {}

    yaml = YAML()
    if isinstance(config_yaml, str):
        config = yaml.load(config_yaml)
    else:
        config = config_yaml
    classification_targets = config.get(g_yaml_t_classification, {})

    if classification_targets == {}:
        return target_df

    for target_name, settings in classification_targets.items():
        target_type = settings.get("type")

        if target_type == "single_asset":
            target_df = compute_single_asset_target_from_dict(df, target_name, settings, target_df)
        elif target_type == "spread_asset":
            target_df = compute_spread_asset_target_from_dict(df, target_name, settings, target_df)
        else:
            print(f"⚠️ Unknown target type: {target_type}")

    return target_df


def add_targets_from_md(
    filepath: Union[str, Path],
    df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add targets defined in a markdown file containing a YAML code block.

    The markdown file should contain a YAML code block like:

    ```yaml
    t_classification:
      btc_stop_loss_signal_class_240:
        type: single_asset
        asset: BTC
        function: gen_perfect_stoploss_signal_class
        params:
          close_col: S_close_f32
          high_col: S_high_f32
          low_col: S_low_f32
          up_objective: 0.012
          up_stoploss: -0.004
          down_objective: -0.012
          down_stoploss: 0.004
          N_periods: 240
    ```

    Args:
        filepath: Path to .md file with YAML target configuration.
        df: DataFrame with OHLCV columns.
        target_df: DataFrame to add target columns to.

    Returns:
        pd.DataFrame: The target_df with added target columns.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If no YAML block found or invalid configuration.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Target config file not found: {filepath}")

    content = filepath.read_text(encoding="utf-8")

    yaml_block_match = re.search(
        r"```yaml\s*\n(.*?)\n```", content, re.DOTALL | re.IGNORECASE
    )
    if not yaml_block_match:
        raise ValueError(f"No YAML code block found in {filepath}")

    yaml_content = yaml_block_match.group(1)

    yaml = YAML()
    config = yaml.load(yaml_content)

    if not config:
        raise ValueError(f"Invalid YAML content in {filepath}")

    return setup_framework_indicators(config, df, target_df)

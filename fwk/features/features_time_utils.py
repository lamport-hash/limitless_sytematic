import pandas as pd
import numpy as np


def split_dataframe_by_time(
    df,
    col_o="S_open_f32",
    col_h="S_high_f32",
    col_l="S_low_f32",
    col_c="S_close_f32",
    col_v="S_volume_f64",
    col_d="S_close_time_i",
):
    """
    Splits a DataFrame into three DataFrames based on time of day (UTC),
    using the index as minutes since 2000-01-01 00:00.

    Time windows:
      - df_morning:   00:00 to 07:50 (minutes 0 to 470)
      - df_open:      07:51 to 20:00 (minutes 471 to 1199)
      - df_night:     20:01 to 23:59 (minutes 1200 to 1439)

    Args:
        df (pd.DataFrame): Input DataFrame with index as minutes since 2000-01-01 00:00.
        col_o, col_h, col_l, col_c, col_v, col_d (str): Column names (unused in logic but kept for signature).

    Returns:
        tuple: (df_morning, df_open, df_night) — each is a filtered DataFrame.
    """
    # Extract minute-of-day from index (modulo 1440 to handle multi-day data)
    minute_of_day = df.index % 1440

    # Define time windows (in minutes since midnight)
    morning_mask = (minute_of_day >= 0) & (minute_of_day <= 470)  # 00:00 to 07:50
    open_mask = (minute_of_day >= 471) & (minute_of_day <= 1199)  # 07:51 to 20:00
    night_mask = (minute_of_day >= 1200) & (minute_of_day <= 1439)  # 20:01 to 23:59

    df_morning = df[morning_mask].copy()
    df_open = df[open_mask].copy()
    df_night = df[night_mask].copy()

    return df_morning, df_open, df_night


def add_cyclical_time_features(p_df, drop_original=True):
    """
    Convert existing time component columns to cyclical features.

    Expects columns: t_minute_of_day_ui16, t_day_of_week_ui16,
                     t_month_of_year_ui16, t_day_of_month_ui16

    Args:
        p_df: DataFrame with pre-computed time component columns
        drop_original: If True, drop the original time columns after encoding

    Returns:
        (df_with_cyclical, cyclical_feature_names)
    """
    # Map existing columns to their periods
    time_mappings = [
        ("t_minute_of_day_ui16", 1440),  # minutes in a day
        ("t_day_of_week_ui16", 7),  # days in a week
        ("t_month_of_year_ui16", 12),  # months in a year
        ("t_day_of_month_ui16", 31),  # max days in a month
    ]

    df = p_df.copy()
    cyclical_features = []

    # Convert each time component to sin/cos pair
    for col_name, period in time_mappings:
        if col_name not in df.columns:
            raise ValueError(f"Expected column '{col_name}' not found in DataFrame")

        # Calculate angle
        angle = 2 * np.pi * df[col_name] / period

        # Create feature names matching the original column pattern
        base_name = col_name.replace("_ui16", "")  # Remove type suffix
        sin_name = f"{base_name}_sin_f32"
        cos_name = f"{base_name}_cos_f32"

        # Add cyclical features
        df[sin_name] = np.sin(angle).astype("float32")
        df[cos_name] = np.cos(angle).astype("float32")

        cyclical_features.extend([sin_name, cos_name])

    # Drop original columns if requested
    if drop_original:
        original_cols = [col for col, _ in time_mappings]
        df = df.drop(columns=original_cols)

    return df, cyclical_features

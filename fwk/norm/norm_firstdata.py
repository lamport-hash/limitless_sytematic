import pandas as pd
from pathlib import Path

from core.enums import (
    g_close_col,
    g_open_col,
    g_high_col,
    g_low_col,
    g_volume_col,
    g_open_time_col,
    g_close_time_col,
    g_index_col,
)

from norm.norm_utils import add_minutes_since_2000


def norm_firstdata(input_file: str, output_pickle: str) -> pd.DataFrame:
    """
    Load txt (CSV-like) file and create DataFrame with normalized column mapping to standard candle columns:
        - g_open_col: "S_open_f32"
        - g_high_col: "S_high_f32"
        - g_low_col: "S_low_f32"
        - g_close_col: "S_close_f32"
        - g_volume_col: "S_volume_f64"
        - g_close_time_col: "S_close_time_i"
        - g_open_time_col: "S_open_time_i"
        - g_index_col: "i_minute_i" (computed from datetime)

    Args:
        input_file: Path to the input txt file.
        output_pickle: Path to save the normalized DataFrame as pickle.

    Returns:
        DataFrame with normalized columns.
    """
    df = pd.read_csv(
        input_file,
        header=None,
        names=["_timestamp_str", "Open", "High", "Low", "Close", "Volume"],
        dtype={
            "_timestamp_str": str,
            "Open": float,
            "High": float,
            "Low": float,
            "Close": float,
            "Volume": float,
        },
    )

    df["Datetime"] = pd.to_datetime(df["_timestamp_str"])

    df = df.rename(columns={"_timestamp_str": "Timestamp"})

    df = df.sort_values("Datetime").reset_index(drop=True)

    df[g_open_col] = df["Open"].astype("float32")
    df[g_high_col] = df["High"].astype("float32")
    df[g_low_col] = df["Low"].astype("float32")
    df[g_close_col] = df["Close"].astype("float32")
    df[g_volume_col] = df["Volume"].astype("float64")

    df[g_close_time_col] = df["Timestamp"]
    df[g_open_time_col] = df["Datetime"].astype("str")

    df = add_minutes_since_2000(df, "Datetime", g_index_col)

    df = df[
        [
            g_index_col,
            g_open_col,
            g_high_col,
            g_low_col,
            g_close_col,
            g_volume_col,
            g_open_time_col,
            g_close_time_col,
        ]
    ]

    output_path = Path(output_pickle)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_pickle(output_pickle, compression="gzip")

    return df

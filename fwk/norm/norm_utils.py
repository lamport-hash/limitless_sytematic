import pandas as pd
import gzip
import glob
import logging
from pathlib import Path
from typing import cast


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_normalized_df(filepath: str | Path) -> pd.DataFrame:
    """
    Load a normalized DataFrame (gzip-compressed pickle with .parquet extension).
    
    Supports glob patterns - if the path contains wildcards, resolves to first match.
    """
    filepath = str(filepath)
    
    if "*" in filepath:
        matches = glob.glob(filepath)
        if not matches:
            raise FileNotFoundError(f"No files match pattern: {filepath}")
        filepath = matches[0]
        logger.debug(f"Resolved glob pattern to: {filepath}")
    
    with gzip.open(filepath, 'rb') as f:
        df = pd.read_pickle(f)  # pyright: ignore[reportArgumentType]
        return cast(pd.DataFrame, df)


def add_minutes_since_2000(df: pd.DataFrame, datetime_col: str, new_col_name: str) -> pd.DataFrame:
    reference_time = pd.Timestamp("2000-01-01 00:00:00")
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df[new_col_name] = ((df[datetime_col] - reference_time).dt.total_seconds() // 60).astype("int64")
    return df


def forward_fill_df(
    p_df: pd.DataFrame,
    p_columns: list | None = None,
    p_inplace: bool = False,
) -> pd.DataFrame:
    """
    Forward-fill missing values in a DataFrame.

    Useful for filling gaps in market data when the market is closed.
    Propagates last valid observation forward to fill NaN values.

    Args:
        p_df: DataFrame to forward-fill
        p_columns: List of columns to fill (default: all columns)
        p_inplace: Whether to modify DataFrame in place (default: False)

    Returns:
        DataFrame with forward-filled values

    Example:
        >>> df = forward_fill_df(df, p_columns=['QQQ_S_close_f32', 'SPY_S_close_f32'])
    """
    if p_inplace:
        if p_columns:
            p_df[p_columns] = p_df[p_columns].ffill()
        else:
            p_df.ffill(inplace=True)
        return p_df
    else:
        df = p_df.copy()
        if p_columns:
            df[p_columns] = df[p_columns].ffill()
        else:
            df = df.ffill()
        return df

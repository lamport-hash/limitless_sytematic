import pandas as pd
import gzip
import logging
from pathlib import Path


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_normalized_df(filepath: str) -> pd.DataFrame:
    """
    Load a normalized DataFrame (gzip-compressed pickle with .parquet extension).
    """
    with gzip.open(filepath, 'rb') as f:
        return pd.read_pickle(f)


def add_minutes_since_2000(df: pd.DataFrame, datetime_col: str, new_col_name: str) -> pd.DataFrame:
    reference_time = pd.Timestamp("2000-01-01 00:00:00")
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df[new_col_name] = ((df[datetime_col] - reference_time).dt.total_seconds() // 60).astype("int64")
    return df

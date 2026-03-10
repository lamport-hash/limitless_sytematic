import pandas as pd
import logging
from typing import Tuple

from core.enums import (
    g_close_col,
    g_open_col,
    g_high_col,
    g_low_col,
    g_volume_col,
    g_close_time_col,
    g_open_time_col,
    g_index_col,
    g_qa_vol_col,
    g_nt_col,
    g_la_vol_col,
    g_lqa_vol_col,
)

from norm.norm_utils import add_minutes_since_2000


logger_s = logging.getLogger(__name__)
logger_s.setLevel(logging.INFO)


def normalise_generic_df(
    inputfile: str,
    output_pickle: str,
    p_verbose: bool = False,
) -> Tuple[pd.DataFrame, bool]:
    """
    Normalize DataFrame column names to standard format from input file and save to pickle.

    Handles various input formats (Binance old/new, generic, lowercase, etc.)
    and maps them to standard column names defined in core.enums.

    Args:
        inputfile (str): Path to input file (csv, parquet, etc.).
        output_pickle (str): Path to output pickle file.
        p_verbose (bool): Enable verbose logging.

    Returns:
        tuple: (normalized_df, binance_cols_flag)
    """
    logger_s.info("******")
    logger_s.info("STARTING normalise_generic_df")

    p_df = pd.read_parquet(inputfile) if inputfile.endswith('.parquet') else pd.read_csv(inputfile)

    logger_s.warn(p_df.columns)

    col_name_close = "close"
    col_name_Close = "Close"
    col_name_close_norm = g_close_col
    col_name_binance = "Quote asset volume"
    col_name_binance_norm = g_la_vol_col
    binance_cols = False

    if col_name_binance in p_df.columns:
        logger_s.info("df with old colnames renaming")
        p_df.rename(
            columns={
                "Open": g_open_col,
                "High": g_high_col,
                "Low": g_low_col,
                "Close": g_close_col,
                "Close time": g_close_time_col,
                "Open time": g_open_time_col,
                "Volume": g_volume_col,
                "Quote asset volume": g_qa_vol_col,
                "Number of trades": g_nt_col,
                "Taker buy base asset volume": g_la_vol_col,
                "Taker buy quote asset volume": g_lqa_vol_col,
            },
            inplace=True,
        )
        binance_cols = True
    elif col_name_binance_norm in p_df.columns:
        binance_cols = True
        logger_s.info("df binance with new normalised colnames")
    elif col_name_Close in p_df.columns:
        binance_cols = False
        p_df.rename(
            columns={
                "Open": g_open_col,
                "High": g_high_col,
                "Low": g_low_col,
                "Close": g_close_col,
                "Timestamp": g_close_time_col,
                "Datetime": g_open_time_col,
                "Volume": g_volume_col,
            },
            inplace=True,
        )
    elif col_name_close in p_df.columns:
        binance_cols = False
        p_df.rename(
            columns={
                "open": g_open_col,
                "high": g_high_col,
                "low": g_low_col,
                "close": g_close_col,
                "timestamp": g_close_time_col,
                "datetime": g_open_time_col,
                "volume": g_volume_col,
            },
            inplace=True,
        )
    elif g_close_col in p_df.columns:
        binance_cols = False
        logger_s.info("df with new normalised colnames")
    else:
        logger_s.warn("df with unknown format ")

    if g_index_col not in p_df.columns and g_close_time_col in p_df.columns:
        p_df = add_minutes_since_2000(p_df, g_close_time_col, g_index_col)

    p_df.to_pickle(output_pickle)
    logger_s.info(f"Saved normalized dataframe to {output_pickle}")

    return p_df, binance_cols

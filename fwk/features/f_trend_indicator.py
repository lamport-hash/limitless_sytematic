import pandas as pd
from typing import Tuple
import logging

# Configure logging (do this once in your main module; here for completeness)
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_trend_smootheness(p_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Computes the trend smoothness indicator for a given DataFrame of 1-minute candle data.

    Args:
        p_df (pd.DataFrame): A DataFrame containing Open/High/Low/Close columns.

    Returns:
        Tuple[pd.Series, pd.Series]: A tuple containing two Series:
            - dir_trend_smooth: The directional trend smoothness values.
            - trend_smooth: The absolute trend smoothness values.
            In case of errors, returns Series filled with 0s.
    """
    # Default fallback in case of any error
    if len(p_df) == 0:
        return pd.Series([0.0], dtype=float), pd.Series([0.0], dtype=float)

    try:
        # Validate input type (still check since it's cheap and helpful)
        if not isinstance(p_df, pd.DataFrame):
            logger.warning("Input is not a pandas DataFrame. Returning zeros.")
            return pd.Series([0.0] * len(p_df), dtype=float), pd.Series([0.0] * len(p_df), dtype=float)

        required_columns = ['Open', 'High', 'Low', 'Close']
        missing_columns = [col for col in required_columns if col not in p_df.columns]
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}. Returning zeros.")
            return pd.Series([0.0] * len(p_df), dtype=float), pd.Series([0.0] * len(p_df), dtype=float)

        # Compute trend smoothness
        close_diff = p_df['Close'] - p_df['Close'].shift(1)
        range_val = p_df['High'] - p_df['Low'] + 0.00000001
        dir_trend_smooth = close_diff / range_val
        trend_smooth = dir_trend_smooth.abs()

        # Replace NaN with 0 (handles first row and any division issues)
        dir_trend_smooth = dir_trend_smooth.fillna(0)
        trend_smooth = trend_smooth.fillna(0)

        # Ensure output has same length as input (in case of misalignment)
        if len(dir_trend_smooth) != len(p_df):
            logger.warning("Computed trend smoothness length mismatch. Padding with zeros.")
            dir_trend_smooth = dir_trend_smooth.reindex(p_df.index, fill_value=0.0)
            trend_smooth = trend_smooth.reindex(p_df.index, fill_value=0.0)

        return dir_trend_smooth, trend_smooth

    except Exception as e:
        # Log the error but do NOT raise — ensure function always returns values
        logger.error(f"Unexpected error in compute_trend_smootheness: {e}")
        # Return zeros of correct length
        return pd.Series([0.0] * len(p_df), dtype=float), pd.Series([0.0] * len(p_df), dtype=float)
import gc

import pandas as pd
import numpy as np
from numba import jit

from core.enums import g_close_col, g_open_col, g_high_col, g_low_col, g_volume_col


@jit(nopython=True)
def numba_parabolic_sar(high, low, close, initial_af=0.02, max_af=0.2):
    """
    Calculate the Parabolic SAR (Stop and Reverse) using Numba for performance optimization.

    Parameters:
        high (numpy.ndarray): Array of high prices.
        low (numpy.ndarray): Array of low prices.
        close (numpy.ndarray): Array of closing prices.
        initial_af (float, optional): Initial acceleration factor. Defaults to 0.02.
        max_af (float, optional): Maximum acceleration factor. Defaults to 0.2.

    Returns:
        numpy.ndarray: Array containing the SAR values for each period.
    """
    n = len(close)
    sar = np.zeros(n)
    trend = 1  # 1 for uptrend, -1 for downtrend
    af = initial_af
    ep = high[0] if trend == 1 else low[0]
    sar[0] = low[0] if trend == 1 else high[0]

    for i in range(1, n):
        sar[i] = sar[i - 1] + af * (ep - sar[i - 1])

        if trend == 1:
            sar[i] = min(sar[i], low[i - 1], low[i - 2])
        else:
            sar[i] = max(sar[i], high[i - 1], high[i - 2])

        if trend == 1 and high[i] > ep:
            ep = high[i]
            af = min(af + initial_af, max_af)
        elif trend == -1 and low[i] < ep:
            ep = low[i]
            af = min(af + initial_af, max_af)

        if trend == 1 and close[i] < sar[i]:
            trend = -1
            sar[i] = ep
            af = initial_af
            ep = low[i]
        elif trend == -1 and close[i] > sar[i]:
            trend = 1
            sar[i] = ep
            af = initial_af
            ep = high[i]

    return sar


@jit(nopython=True)
def numba_atr(high, low, close, period=14):
    """
    Compute the Average True Range (ATR) using Numba for performance optimization.

    Args:
        high (numpy.ndarray): Array of high prices.
        low (numpy.ndarray): Array of low prices.
        close (numpy.ndarray): Array of closing prices.
        period (int, optional): The number of periods to use for the ATR calculation. Defaults to 14.

    Returns:
        numpy.ndarray: An array of ATR values with the same length as the input arrays.
    """
    n = len(close)
    if n == 0:
        return np.zeros(n)
    atr = np.zeros(n)
    tr = np.zeros(n)

    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

    atr[0] = np.mean(tr[1 : period + 1])

    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr


@jit(nopython=True)
def numba_rsi(prices, period=14):
    """
    Compute the Relative Strength Index (RSI) using Numba.
    Args:
        prices (numpy.ndarray): Array of prices.
        period (int): RSI lookback period.
    Returns:
        numpy.ndarray: RSI values.
    """
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.zeros_like(prices)
    avg_loss = np.zeros_like(prices)
    rsi = np.zeros_like(prices)

    # Initialize first average gain and loss
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])

    for i in range(period + 1, len(prices)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period
        rs = avg_gain[i] / avg_loss[i] if avg_loss[i] != 0 else 0
        rsi[i] = 100 - (100 / (1 + rs))

    return rsi


@jit(nopython=True)
def numba_ema(prices, period):
    """
    Compute the Exponential Moving Average (EMA) using Numba.
    Args:
        prices (numpy.ndarray): Array of prices.
        period (int): EMA period.
    Returns:
        numpy.ndarray: EMA values.
    """
    ema = np.zeros_like(prices)
    alpha = 2 / (period + 1)

    # Initialize first EMA value
    ema[period - 1] = np.mean(prices[:period])

    for i in range(period, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]

    return ema


@jit(nopython=True)
def smma_numba(p_values, p_period):
    """
    Calculates the Smoothed Moving Average (SMMA) of a source array.

    Parameters:
        p_values (array): The source data (numpy array).
        p_period (int): The period of the SMMA.

    Returns:
        array: The SMMA values.
    """
    n = len(p_values)
    smma_values = np.zeros(n)

    if p_period <= 0 or n < p_period:
        smma_values[:] = np.nan
        return smma_values

    sma = sum(p_values[:p_period]) / p_period
    for i in range(p_period):
        smma_values[i] = sma

    for i in range(p_period, n):
        smma_values[i] = (smma_values[i - 1] * (p_period - 1) + p_values[i]) / p_period

    return smma_values


@jit(nopython=True)
def rolling_std(values, window):
    n = len(values)
    stds = np.full(n, np.nan)

    for i in range(n):
        if i < window - 1:
            continue
        mean = np.mean(values[i - window + 1 : i + 1])
        variance = np.sum((values[i - window + 1 : i + 1] - mean) ** 2) / window
        stds[i] = np.sqrt(variance)

    return stds


@jit(nopython=True)
def numba_roc(prices, period):
    """
    Compute Rate of Change (ROC) using Numba.
    ROC = (price - price_N_periods_ago) / price_N_periods_ago * 100

    Args:
        prices (numpy.ndarray): Array of prices.
        period (int): Lookback period.

    Returns:
        numpy.ndarray: ROC values as percentage.
    """
    n = len(prices)
    roc = np.zeros(n)

    for i in range(period, n):
        if prices[i - period] != 0:
            roc[i] = (prices[i] - prices[i - period]) / prices[i - period] * 100

    return roc



@jit(nopython=True)
def numba_roc_correct_min(prices, min_index, period):
    """
    Compute Rate of Change (ROC) using Numba.
    ROC = (price - price_N_periods_ago) / price_N_periods_ago * 100

    Args:
        prices (numpy.ndarray): Array of prices.
        min_index (numpy.ndarray): Array of the minutes whem the prices were observed, min_index of the candles, is the number of minutes since 01/01/2000
        period (int): Lookback period in minutes

    Returns:
        numpy.ndarray: ROC values as percentage.
    """
    n = len(prices)
    roc = np.zeros(n)
    
    j = 0  # pointer for the past index
    
    for i in range(1, n):
        target_min = min_index[i] - period
        #print(target_min)
        
        # Move j forward until we find the first index that exceeds target_min
        while j < i and min_index[j] <= target_min:
            j += 1
        
        # Now j is the first index with min_index[j] > target_min
        # So j-1 is the last index with min_index <= target_min
        past_idx = j - 1
        
        if past_idx >= 0 and prices[past_idx] != 0:
            roc[i] = (prices[i] - prices[past_idx]) / prices[past_idx] * 100
    
    return roc




@jit(nopython=True)
def numba_sma(prices, period):
    """Simple Moving Average using Numba.
    
    Args:
        prices (numpy.ndarray): Array of prices.
        period (int): SMA period.
    
    Returns:
        numpy.ndarray: SMA values.
    """
    n = len(prices)
    sma = np.full(n, np.nan)
    for i in range(period - 1, n):
        sma[i] = np.mean(prices[i - period + 1:i + 1])
    return sma


@jit(nopython=True)
def numba_rolling_max(values, period):
    """Rolling maximum using Numba.
    
    Args:
        values (numpy.ndarray): Array of values.
        period (int): Rolling window period.
    
    Returns:
        numpy.ndarray: Rolling maximum values.
    """
    n = len(values)
    result = np.full(n, np.nan)
    for i in range(period - 1, n):
        result[i] = np.max(values[i - period + 1:i + 1])
    return result


@jit(nopython=True)
def numba_rolling_min(values, period):
    """Rolling minimum using Numba.
    
    Args:
        values (numpy.ndarray): Array of values.
        period (int): Rolling window period.
    
    Returns:
        numpy.ndarray: Rolling minimum values.
    """
    n = len(values)
    result = np.full(n, np.nan)
    for i in range(period - 1, n):
        result[i] = np.min(values[i - period + 1:i + 1])
    return result


@jit(nopython=True)
def numba_macd(prices, fast_period=12, slow_period=26, signal_period=9):
    """MACD indicator using Numba.
    
    Args:
        prices (numpy.ndarray): Array of prices.
        fast_period (int): Fast EMA period (default 12).
        slow_period (int): Slow EMA period (default 26).
        signal_period (int): Signal line EMA period (default 9).
    
    Returns:
        tuple: (macd_line, signal_line, histogram)
    """
    ema_fast = numba_ema(prices, fast_period)
    ema_slow = numba_ema(prices, slow_period)
    macd_line = ema_fast - ema_slow
    signal_line = numba_ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


@jit(nopython=True)
def numba_crossover_detect(fast_line, slow_line):
    """Detect crossovers between two lines.
    
    Args:
        fast_line (numpy.ndarray): Fast line values.
        slow_line (numpy.ndarray): Slow line values.
    
    Returns:
        numpy.ndarray: 1 for bullish crossover, -1 for bearish, 0 otherwise.
    """
    n = len(fast_line)
    signals = np.zeros(n)
    for i in range(1, n):
        if (not (np.isnan(fast_line[i]) or np.isnan(slow_line[i]) or 
                    np.isnan(fast_line[i-1]) or np.isnan(slow_line[i-1]))):
            # Bullish crossover
            if fast_line[i-1] <= slow_line[i-1] and fast_line[i] > slow_line[i]:
                signals[i] = 1
            # Bearish crossover
            elif fast_line[i-1] >= slow_line[i-1] and fast_line[i] < slow_line[i]:
                signals[i] = -1
    return signals


def calculate_atr(df, period=14):
    """Wrapper for numba_atr that works with DataFrame.
    
    Args:
        df: DataFrame with S_high_f32, S_low_f32, S_close_f32 columns.
        period: ATR period (default 14).
    
    Returns:
        numpy.ndarray: ATR values.
    """
    return numba_atr(
        df[g_high_col].to_numpy(),
        df[g_low_col].to_numpy(),
        df[g_close_col].to_numpy(),
        period
    )


def add_accumulation_distribution_index(df):
    """
    Calculate Accumulation/Distribution Index for OHLCV data.

    Parameters:
    df (pandas.DataFrame): DataFrame containing g_close_col, g_high_col, g_low_col, and 'Volume' columns

    Returns:
    pandas.DataFrame: Original DataFrame with new 'acc_dist_index' column
    """

    work_df = df.copy()
    numeric_columns = [g_close_col, g_high_col, g_low_col, g_volume_col]
    for col in numeric_columns:
        work_df[col] = pd.to_numeric(work_df[col], errors="coerce")
    # Calculate Current Money Flow Volume (CMFV)
    # Formula: ((Close - Low) - (High - Close)) / (High - Low) * Volume
    # Simplified to: (2*Close - High - Low) / (High - Low) * Volume
    df["cmfv"] = (
        (2 * work_df[g_close_col] - work_df[g_high_col] - work_df[g_low_col])
        / (work_df[g_high_col] - work_df[g_low_col])
        * work_df[g_volume_col]
    )

    # Handle potential division by zero (when High equals Low)
    df.loc[df[g_high_col] == df[g_low_col], "cmfv"] = 0

    # Calculate cumulative sum to get the A/D line
    colname = "F_acc_dist_index_f64"
    df[colname] = df["cmfv"].cumsum()

    # check my work vs pd_ta
    # test = ta.ad(work_df['High'], work_df['Low'], work_df['Close'], work_df['Volume'])

    # Drop the intermediate calculation column
    df.drop("cmfv", axis=1, inplace=True)

    del work_df
    gc.collect()

    return df, colname

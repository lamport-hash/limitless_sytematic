import pandas as pd

# ===================================================================
# COLUMN NAME VARIABLES — CHANGE THESE ONCE TO MATCH YOUR DATAFRAME
# ===================================================================
# Example for your custom column names (works with any naming):
# open_col   = "Open"
# high_col   = "High"
# low_col    = "Low"
# close_col  = "Close"
# volume_col = "Volume"

# But you can pass them directly to any function below — no globals required.

# ===================================================================
# 1. DUAL EMA CROSSOVER (Trend-Following)
# ===================================================================
def dual_ema_crossover_strategy(
    df: pd.DataFrame,
    close_col: str = "c_col",
    fast_period: int = 9,
    slow_period: int = 21
) -> pd.DataFrame:
    df = df.copy()
    
    df['ema_fast'] = df[close_col].ewm(span=fast_period, adjust=False).mean()
    df['ema_slow'] = df[close_col].ewm(span=slow_period, adjust=False).mean()
    
    prev_fast = df['ema_fast'].shift(1)
    prev_slow = df['ema_slow'].shift(1)
    
    df['signal'] = 0
    df.loc[(df['ema_fast'] > df['ema_slow']) & (prev_fast <= prev_slow), 'signal'] = 1   # long entry
    df.loc[(df['ema_fast'] < df['ema_slow']) & (prev_fast >= prev_slow), 'signal'] = -1  # short entry
    
    return df


# ===================================================================
# 2. MACD CROSSOVER (Momentum)
# ===================================================================
def macd_crossover_strategy(
    df: pd.DataFrame,
    close_col: str = "c_col",
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9
) -> pd.DataFrame:
    df = df.copy()
    
    ema_fast = df[close_col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[close_col].ewm(span=slow, adjust=False).mean()
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=signal_period, adjust=False).mean()
    
    prev_macd = df['macd'].shift(1)
    prev_sig  = df['macd_signal'].shift(1)
    
    df['signal'] = 0
    df.loc[(df['macd'] > df['macd_signal']) & (prev_macd <= prev_sig), 'signal'] = 1
    df.loc[(df['macd'] < df['macd_signal']) & (prev_macd >= prev_sig), 'signal'] = -1
    
    return df


# ===================================================================
# 3. RSI MEAN-REVERSION (with 200 EMA trend filter)
# ===================================================================
def rsi_mean_reversion_strategy(
    df: pd.DataFrame,
    close_col: str = "c_col",
    rsi_period: int = 14,
    oversold: int = 30,
    overbought: int = 70,
    ema200_period: int = 200
) -> pd.DataFrame:
    df = df.copy()
    
    delta = df[close_col].diff()
    gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=rsi_period).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Trend filter
    df['ema200'] = df[close_col].ewm(span=ema200_period, adjust=False).mean()
    
    df['signal'] = 0
    df.loc[(df['rsi'] < oversold) & (df[close_col] > df['ema200']), 'signal'] = 1   # long
    df.loc[(df['rsi'] > overbought) & (df[close_col] < df['ema200']), 'signal'] = -1  # short
    
    return df


# ===================================================================
# 4. BOLLINGER BANDS SQUEEZE + BREAKOUT (Volatility)
# ===================================================================
def bollinger_squeeze_breakout_strategy(
    df: pd.DataFrame,
    close_col: str = "c_col",
    bb_period: int = 20,
    bb_std: float = 2.0,
    squeeze_lookback: int = 100
) -> pd.DataFrame:
    df = df.copy()
    
    df['bb_mid'] = df[close_col].rolling(bb_period).mean()
    df['bb_std_val'] = df[close_col].rolling(bb_period).std()
    df['bb_upper'] = df['bb_mid'] + bb_std * df['bb_std_val']
    df['bb_lower'] = df['bb_mid'] - bb_std * df['bb_std_val']
    
    bb_width = df['bb_upper'] - df['bb_lower']
    bb_width_low = bb_width.rolling(squeeze_lookback).min()
    
    prev_width = bb_width.shift(1)
    prev_width_low = bb_width_low.shift(1)
    
    df['signal'] = 0
    # Long breakout after squeeze
    df.loc[(df[close_col] > df['bb_upper']) & (prev_width < prev_width_low), 'signal'] = 1
    # Short breakout after squeeze
    df.loc[(df[close_col] < df['bb_lower']) & (prev_width < prev_width_low), 'signal'] = -1
    
    return df


# ===================================================================
# 5. DONCHIAN CHANNEL BREAKOUT (Pure Price Action)
# ===================================================================
def donchian_channel_breakout_strategy(
    df: pd.DataFrame,
    high_col: str = "h_col",
    low_col: str = "l_col",
    close_col: str = "c_col",
    channel_period: int = 20
) -> pd.DataFrame:
    df = df.copy()
    
    df['upper_band'] = df[high_col].rolling(channel_period).max()
    df['lower_band'] = df[low_col].rolling(channel_period).min()
    
    prev_upper = df['upper_band'].shift(1)
    prev_lower = df['lower_band'].shift(1)
    
    df['signal'] = 0
    df.loc[df[close_col] > prev_upper, 'signal'] = 1   # long breakout
    df.loc[df[close_col] < prev_lower, 'signal'] = -1  # short breakout
    
    return df

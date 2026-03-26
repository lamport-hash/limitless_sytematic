"""
MACD Crossover Strategy.

Momentum strategy that trades MACD crossovers.
- Long: MACD crosses above signal line
- Short: MACD crosses below signal line
"""

from typing import Tuple

import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy

from features.base_dataframe import BaseDataFrame
from features.feature_ta_utils import numba_macd, numba_crossover_detect, calculate_atr
from core.enums import (
    g_open_col,
    g_high_col,
    g_low_col,
    g_close_col,
    g_volume_col,
    g_index_col,
)


def build_features(
    p_df: pd.DataFrame,
    p_direction: str = "both",
    p_fast: int = 12,
    p_slow: int = 26,
    p_signal: int = 9,
) -> pd.DataFrame:
    """Build MACD Crossover features and signals."""
    df = p_df.copy()
    
    # Use numba-optimized MACD
    prices = df[g_close_col].to_numpy()
    macd_line, signal_line, histogram = numba_macd(prices, p_fast, p_slow, p_signal)
    
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = histogram
    
    # Use numba-optimized ATR
    df["ATR"] = calculate_atr(df, period=14)
    
    # Use numba-optimized crossover detection
    crossovers = numba_crossover_detect(df["macd"].to_numpy(), df["macd_signal"].to_numpy())
    
    long_entry = crossovers == 1
    short_entry = crossovers == -1
    
    df["signal"] = 0
    if p_direction in ("long", "both"):
        df.loc[long_entry, "signal"] = 1
    if p_direction in ("short", "both"):
        df.loc[short_entry, "signal"] = -1
    
    return df


def convert_to_ohlcv(p_df: pd.DataFrame) -> pd.DataFrame:
    """Convert framework columns to OHLCV for backtesting."""
    ohlcv_df = pd.DataFrame(index=p_df.index)
    
    ohlcv_df["Open"] = p_df[g_open_col]
    ohlcv_df["High"] = p_df[g_high_col]
    ohlcv_df["Low"] = p_df[g_low_col]
    ohlcv_df["Close"] = p_df[g_close_col]
    ohlcv_df["Volume"] = p_df[g_volume_col]
    
    ohlcv_df["signal"] = p_df["signal"]
    ohlcv_df["ATR"] = p_df["ATR"]
    ohlcv_df["macd"] = p_df["macd"]
    ohlcv_df["macd_signal"] = p_df["macd_signal"]
    ohlcv_df["macd_hist"] = p_df["macd_hist"]
    
    if g_index_col in p_df.columns:
        base = pd.Timestamp("2000-01-01")
        ohlcv_df.index = base + pd.to_timedelta(p_df[g_index_col], unit="m")
    
    ohlcv_df = ohlcv_df.dropna(subset=["Open", "High", "Low", "Close", "ATR"])
    
    return ohlcv_df


class MACDCrossoverStrategy(Strategy):
    """MACD Crossover strategy with ATR-based SL/TP."""
    
    atr_mult: float = 2.0
    rr: float = 2.0
    direction: str = "both"
    
    def init(self):
        pass
    
    def next(self):
        if self.position:
            return
        
        close = float(self.data.Close[-1])
        sig = int(self.data.signal[-1])
        atr = float(self.data.ATR[-1])
        
        if not np.isfinite(atr) or atr <= 0:
            return
        
        sl_dist = self.atr_mult * atr
        
        if sig == 1 and self.direction in ("long", "both"):
            sl = close - sl_dist
            tp = close + self.rr * sl_dist
            self.buy(sl=sl, tp=tp)
        
        elif sig == -1 and self.direction in ("short", "both"):
            sl = close + sl_dist
            tp = close - self.rr * sl_dist
            self.sell(sl=sl, tp=tp)


def run_backtest(
    p_df: pd.DataFrame,
    p_cash: float = 100_000,
    p_commission: float = 0.0002,
    **kwargs
) -> Tuple[dict, pd.DataFrame]:
    """Run backtest with MACD Crossover strategy."""
    ohlcv_df = convert_to_ohlcv(p_df)
    
    bt = Backtest(
        ohlcv_df,
        MACDCrossoverStrategy,
        cash=p_cash,
        commission=p_commission,
        trade_on_close=True,
        hedging=False,
        exclusive_orders=False,
    )
    
    stats = bt.run(**kwargs)
    trades = stats._trades if hasattr(stats, '_trades') else pd.DataFrame()
    
    return stats, trades


def main(
    p_df: pd.DataFrame,
    p_direction: str = "both",
    p_fast: int = 12,
    p_slow: int = 26,
    p_signal: int = 9,
    p_atr_mult: float = 2.0,
    p_rr: float = 2.0,
    p_cash: float = 100_000,
    p_commission: float = 0.0002,
    p_verbose: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """Main entry point."""
    if p_verbose:
        print("=" * 80)
        print(f"MACD Crossover Strategy - {p_fast}/{p_slow}/{p_signal} - Direction: {p_direction}")
        print("=" * 80)
    
    df = p_df
    df = build_features(
        df,
        p_direction=p_direction,
        p_fast=p_fast,
        p_slow=p_slow,
        p_signal=p_signal,
    )
    
    n_signals = (df["signal"] != 0).sum()
    if p_verbose:
        print(f"Active signals: {n_signals}")
    
    stats, trades = run_backtest(
        df,
        p_cash=p_cash,
        p_commission=p_commission,
        atr_mult=p_atr_mult,
        rr=p_rr,
        direction=p_direction,
    )
    
    if p_verbose:
        print(f"\nReturn [%]: {stats['Return [%]']:.2f}")
        print(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
        print(f"# Trades: {stats['# Trades']}")
    
    return df, stats


if __name__ == "__main__":
    main()

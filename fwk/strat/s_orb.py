"""
Opening Range Breakout (ORB) Strategy.

Strategy that trades breakouts from the opening range.
- Long: Price breaks above OR high with volume confirmation
- Short: Price breaks below OR low with volume confirmation
"""

import datetime
from typing import Tuple

import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy

from features.feature_ta_utils import calculate_atr

from features.base_dataframe import BaseDataFrame
from features.features_utils import FeatureType
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
    p_or_minutes: int = 15,
    p_vol_filter: bool = True,
    p_vol_threshold: float = 1.2,
) -> pd.DataFrame:
    """Build ORB features and signals."""
    df = p_df.copy()
    
    bdf = BaseDataFrame(p_df=df)
    bdf.add_feature(FeatureType.OPENING_RANGE, periods=[5, 15, 30])
    bdf.add_feature(FeatureType.VOLUME_RATIO, periods=[20])
    bdf.add_feature(FeatureType.GAP_PCT)
    df = bdf.get_dataframe()
    
    df["ATR"] = calculate_atr(df, period=14)
    
    or_high = df[f"or{p_or_minutes}_high"]
    or_low = df[f"or{p_or_minutes}_low"]
    
    prev_close = df[g_close_col].shift(1)
    
    long_breakout = (prev_close <= or_high) & (df[g_close_col] > or_high)
    short_breakout = (prev_close >= or_low) & (df[g_close_col] < or_low)
    
    if p_vol_filter:
        vol_filter = df["vol_ratio"].fillna(0) > p_vol_threshold
        long_breakout &= vol_filter
        short_breakout &= vol_filter
    
    df["signal"] = 0
    if p_direction in ("long", "both"):
        df.loc[long_breakout, "signal"] = 1
    if p_direction in ("short", "both"):
        df.loc[short_breakout, "signal"] = -1
    
    df["or_high"] = or_high
    df["or_low"] = or_low
    
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
    ohlcv_df["or_high"] = p_df["or_high"]
    ohlcv_df["or_low"] = p_df["or_low"]
    
    if g_index_col in p_df.columns:
        base = pd.Timestamp("2000-01-01")
        ohlcv_df.index = base + pd.to_timedelta(p_df[g_index_col], unit="m")
    
    ohlcv_df = ohlcv_df.dropna(subset=["Open", "High", "Low", "Close", "ATR"])
    
    return ohlcv_df


class ORBStrategy(Strategy):
    """Opening Range Breakout strategy with ATR-based SL/TP."""
    
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
        or_low = float(self.data.or_low[-1])
        or_high = float(self.data.or_high[-1])
        
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
    """Run backtest with ORB strategy."""
    ohlcv_df = convert_to_ohlcv(p_df)
    
    bt = Backtest(
        ohlcv_df,
        ORBStrategy,
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
    p_or_minutes: int = 15,
    p_vol_filter: bool = True,
    p_atr_mult: float = 2.0,
    p_rr: float = 2.0,
    p_cash: float = 100_000,
    p_commission: float = 0.0002,
    p_verbose: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """Main entry point."""
    if p_verbose:
        print("=" * 80)
        print(f"ORB Strategy - {p_or_minutes}min - Direction: {p_direction}")
        print("=" * 80)
    
    df = p_df
    
    if p_verbose:
        print(f"\n2. Building features...")
    df = build_features(
        df,
        p_direction=p_direction,
        p_or_minutes=p_or_minutes,
        p_vol_filter=p_vol_filter,
    )
    
    n_signals = (df["signal"] != 0).sum()
    if p_verbose:
        print(f"   Active signals: {n_signals}")
    
    if p_verbose:
        print(f"\n3. Running backtest...")
    
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
        print(f"Win Rate [%]: {stats['Win Rate [%]']:.2f}")
    
    return df, stats


if __name__ == "__main__":
    main()

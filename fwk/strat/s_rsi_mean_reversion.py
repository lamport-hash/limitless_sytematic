"""
RSI Mean Reversion Strategy.

Mean-reversion strategy that trades RSI extremes with EMA200 trend filter.
- Long: RSI oversold + price above EMA200
- Short: RSI overbought + price below EMA200
"""

from typing import Tuple

import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy

from features.base_dataframe import BaseDataFrame
from features.features_utils import FeatureType
from features.feature_ta_utils import calculate_atr
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
    p_rsi_period: int = 14,
    p_oversold: int = 30,
    p_overbought: int = 70,
    p_ema_period: int = 200,
) -> pd.DataFrame:
    """Build RSI Mean Reversion features and signals."""
    df = p_df.copy()
    
    bdf = BaseDataFrame(p_df=df)
    bdf.add_feature(FeatureType.RSI, periods=[p_rsi_period])
    bdf.add_feature(FeatureType.EMA, periods=[p_ema_period])
    df = bdf.get_dataframe()
    
    df["ATR"] = calculate_atr(df, period=14)
    
    # Use BaseDataFrame's EMA instead of duplicating calculation
    ema_col = f"F_ema_{p_ema_period}_{g_close_col}_f32"
    
    rsi_col = f"rsi{p_rsi_period}"
    
    long_entry = (df[rsi_col] < p_oversold) & (df[g_close_col] > df[ema_col])
    short_entry = (df[rsi_col] > p_overbought) & (df[g_close_col] < df[ema_col])
    
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
    
    if g_index_col in p_df.columns:
        base = pd.Timestamp("2000-01-01")
        ohlcv_df.index = base + pd.to_timedelta(p_df[g_index_col], unit="m")
    
    ohlcv_df = ohlcv_df.dropna(subset=["Open", "High", "Low", "Close", "ATR"])
    
    return ohlcv_df


class RSIMeanReversionStrategy(Strategy):
    """RSI Mean Reversion strategy with ATR-based SL/TP."""
    
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
    """Run backtest with RSI Mean Reversion strategy."""
    ohlcv_df = convert_to_ohlcv(p_df)
    
    bt = Backtest(
        ohlcv_df,
        RSIMeanReversionStrategy,
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
    p_rsi_period: int = 14,
    p_oversold: int = 30,
    p_overbought: int = 70,
    p_ema_period: int = 200,
    p_atr_mult: float = 2.0,
    p_rr: float = 2.0,
    p_cash: float = 100_000,
    p_commission: float = 0.0002,
    p_verbose: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """Main entry point."""
    if p_verbose:
        print("=" * 80)
        print(f"RSI Mean Reversion Strategy - RSI{p_rsi_period} - EMA{p_ema_period} - Direction: {p_direction}")
        print("=" * 80)
    
    df = p_df
    df = build_features(
        df,
        p_direction=p_direction,
        p_rsi_period=p_rsi_period,
        p_oversold=p_oversold,
        p_overbought=p_overbought,
        p_ema_period=p_ema_period,
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

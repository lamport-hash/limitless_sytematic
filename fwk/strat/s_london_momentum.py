"""
London Momentum Strategy.

Strategy that trades pre-market/London session momentum into US open.
- Long: London session closes higher with volume confirmation
- Short: London session closes lower with volume confirmation
"""

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
    p_min_move_pct: float = 0.15,
    p_vol_mult: float = 1.30,
) -> pd.DataFrame:
    """Build London Momentum features and signals."""
    df = p_df.copy()
    
    bdf = BaseDataFrame(p_df=df)
    bdf.add_feature(FeatureType.VOLUME_RATIO, periods=[20])
    df = bdf.get_dataframe()
    
    df["ATR"] = calculate_atr(df, period=14)
    
    base = pd.Timestamp("2000-01-01")
    dt_index = base + pd.to_timedelta(df["i_minute_i"], unit="m")
    
    times = pd.Series(dt_index.dt.time, index=df.index)
    dates = pd.Series(dt_index.dt.date, index=df.index)
    
    import datetime
    is_london = (times >= datetime.time(3, 0)) & (times <= datetime.time(4, 30))
    
    lon = df[is_london].copy()
    lon["date"] = dates[is_london]
    
    if len(lon) > 0:
        lon_stats = lon.groupby("date").agg(
            lon_open=(g_open_col, "first"),
            lon_close=(g_close_col, "last"),
            lon_totvol=(g_volume_col, "sum"),
            lon_avgvol=("vol_sma20", "mean"),
        )
        lon_stats["lon_move_pct"] = (
            (lon_stats["lon_close"] - lon_stats["lon_open"]) /
            lon_stats["lon_open"].replace(0, np.nan) * 100
        )
        lon_stats["lon_vol_ok"] = (
            lon_stats["lon_totvol"] > p_vol_mult * lon_stats["lon_avgvol"]
        )
        
        df["_lon_move"] = dates.map(lon_stats["lon_move_pct"].to_dict())
        df["_lon_vol_ok"] = dates.map(lon_stats["lon_vol_ok"].to_dict())
    else:
        df["_lon_move"] = np.nan
        df["_lon_vol_ok"] = False
    
    at_open = times == datetime.time(9, 30)
    vol_ok = df["_lon_vol_ok"].fillna(False)
    
    long_entry = at_open & (df["_lon_move"] >= p_min_move_pct) & vol_ok
    short_entry = at_open & (df["_lon_move"] <= -p_min_move_pct) & vol_ok
    
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


class LondonMomentumStrategy(Strategy):
    """London Momentum strategy with ATR-based SL/TP."""
    
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
    """Run backtest with London Momentum strategy."""
    ohlcv_df = convert_to_ohlcv(p_df)
    
    bt = Backtest(
        ohlcv_df,
        LondonMomentumStrategy,
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
    p_min_move_pct: float = 0.15,
    p_vol_mult: float = 1.30,
    p_atr_mult: float = 2.0,
    p_rr: float = 2.0,
    p_cash: float = 100_000,
    p_commission: float = 0.0002,
    p_verbose: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """Main entry point."""
    if p_verbose:
        print("=" * 80)
        print(f"London Momentum Strategy - Direction: {p_direction}")
        print("=" * 80)
    
    df = p_df
    df = build_features(
        df,
        p_direction=p_direction,
        p_min_move_pct=p_min_move_pct,
        p_vol_mult=p_vol_mult,
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

"""
Gap Fill Strategy.

Strategy that trades overnight gap reversion to previous close.
- Long: Gap down expecting fill
- Short: Gap up expecting fill
"""

from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import numpy as np
import pandas_ta as ta
from backtesting import Backtest, Strategy

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

DATA_PATH = Path("data/bundle/test_etf_features_bundle.parquet")
ETF_SYMBOL = "QQQ"


def load_data(p_data_path: Optional[Path] = None, p_symbol: str = "QQQ") -> pd.DataFrame:
    """Load ETF data from bundle file."""
    data_path = p_data_path or DATA_PATH
    
    if not data_path.exists():
        raise FileNotFoundError(f"Bundle not found: {data_path}")
    
    df_bundle = pd.read_parquet(data_path)
    
    cols = [c for c in df_bundle.columns if c.startswith(f"{p_symbol}_")]
    if not cols:
        raise ValueError(f"No {p_symbol} columns found in bundle")
    
    col_mapping = {col: col.replace(f"{p_symbol}_", "") for col in cols}
    df = df_bundle[cols].rename(columns=col_mapping).copy()
    
    return df


def build_features(
    p_df: pd.DataFrame,
    p_direction: str = "both",
    p_min_gap_pct: float = 0.10,
    p_max_gap_pct: float = 1.50,
) -> pd.DataFrame:
    """Build Gap Fill features and signals."""
    df = p_df.copy()
    
    bdf = BaseDataFrame(p_df=df)
    bdf.add_feature(FeatureType.GAP_PCT)
    df = bdf.get_dataframe()
    
    df["ATR"] = ta.atr(
        df[g_high_col],
        df[g_low_col],
        df[g_close_col],
        length=14
    )
    
    base = pd.Timestamp("2000-01-01")
    dt_index = base + pd.to_timedelta(df["i_minute_i"], unit="m")
    times = pd.Series(dt_index.dt.time, index=df.index)
    
    import datetime
    at_open = times == datetime.time(9, 30)
    gap_abs = df["gap_pct"].abs()
    gap_ok = (gap_abs >= p_min_gap_pct) & (gap_abs <= p_max_gap_pct)
    
    long_entry = at_open & (df["gap_pct"] <= -p_min_gap_pct) & gap_ok
    short_entry = at_open & (df["gap_pct"] >= p_min_gap_pct) & gap_ok
    
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
    ohlcv_df["prev_close"] = p_df["prev_close"]
    
    if g_index_col in p_df.columns:
        base = pd.Timestamp("2000-01-01")
        ohlcv_df.index = base + pd.to_timedelta(p_df[g_index_col], unit="m")
    
    ohlcv_df = ohlcv_df.dropna(subset=["Open", "High", "Low", "Close", "ATR"])
    
    return ohlcv_df


class GapFillStrategy(Strategy):
    """Gap Fill strategy with ATR-based SL/TP."""
    
    atr_mult: float = 2.0
    rr: float = 2.0
    direction: str = "both"
    
    def init(self):
        pass
    
    def next(self):
        if self.position:
            return
        
        close = float(self.data.Close[-1])
        sig = int(self.data.signal.iloc[-1])
        atr = float(self.data.ATR.iloc[-1])
        
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
    """Run backtest with Gap Fill strategy."""
    ohlcv_df = convert_to_ohlcv(p_df)
    
    bt = Backtest(
        ohlcv_df,
        GapFillStrategy,
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
    p_direction: str = "both",
    p_min_gap_pct: float = 0.10,
    p_max_gap_pct: float = 1.50,
    p_atr_mult: float = 2.0,
    p_rr: float = 2.0,
    p_cash: float = 100_000,
    p_commission: float = 0.0002,
    p_verbose: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """Main entry point."""
    if p_verbose:
        print("=" * 80)
        print(f"Gap Fill Strategy - Direction: {p_direction}")
        print("=" * 80)
    
    df = load_data()
    df = build_features(
        df,
        p_direction=p_direction,
        p_min_gap_pct=p_min_gap_pct,
        p_max_gap_pct=p_max_gap_pct,
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

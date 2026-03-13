"""
Daily Signal Strategy.

Strategy that generates signals based on daily candle direction from hourly data:
- Long Signal (2): Bullish candle (Close > Open)
- Short Signal (1): Bearish candle (Close < Open)

The strategy uses stop-entry orders based on the first N candles of each day
and exits after a configurable delay.
"""

from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import numpy as np
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


def load_data(p_data_path: Optional[Path] = None, p_symbol: str = ETF_SYMBOL) -> pd.DataFrame:
    """
    Load ETF data from bundle file.
    
    Args:
        p_data_path: Path to bundle parquet file (default: DATA_PATH)
        p_symbol: ETF symbol to load (default: QQQ)
        
    Returns:
        DataFrame with framework columns (S_open_f32, S_high_f32, etc.)
    """
    data_path = p_data_path or DATA_PATH
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Bundle not found: {data_path}\n"
            f"Please ensure the bundle file exists."
        )
    
    df_bundle = pd.read_parquet(data_path)
    
    symbol_cols = [c for c in df_bundle.columns if c.startswith(f"{p_symbol}_")]
    if not symbol_cols:
        raise ValueError(f"No {p_symbol} columns found in bundle")
    
    col_mapping = {col: col.replace(f"{p_symbol}_", "") for col in symbol_cols}
    df = df_bundle[symbol_cols].rename(columns=col_mapping).copy()
    
    return df


def build_features(
    p_df: pd.DataFrame,
    p_test_candles: int = 8,
    p_exit_delay: int = 4,
) -> pd.DataFrame:
    """
    Build daily signal features from hourly data.
    
    Args:
        p_df: DataFrame with framework OHLCV columns
        p_test_candles: Number of candles to use for stop price calculation
        p_exit_delay: Number of days to delay before exit signal
        
    Returns:
        DataFrame with added signal, stop_price, and exit columns
    """
    df = p_df.copy()
    
    bdf = BaseDataFrame(p_df=df)
    bdf.add_feature(
        FeatureType.DAILY_SIGNAL,
        p_test_candles=p_test_candles,
        p_exit_delay=p_exit_delay,
    )
    df = bdf.get_dataframe()
    
    return df


def convert_to_ohlcv(p_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert framework columns to standard OHLCV format for backtesting.py.
    
    Args:
        p_df: DataFrame with framework columns (S_open_f32, etc.)
        
    Returns:
        DataFrame with OHLCV columns and datetime index
    """
    ohlcv_df = pd.DataFrame(index=p_df.index)
    
    ohlcv_df["Open"] = p_df[g_open_col]
    ohlcv_df["High"] = p_df[g_high_col]
    ohlcv_df["Low"] = p_df[g_low_col]
    ohlcv_df["Close"] = p_df[g_close_col]
    ohlcv_df["Volume"] = p_df[g_volume_col]
    
    ohlcv_df["signal"] = p_df["F_daily_signal_f16"]
    ohlcv_df["StopPrice"] = p_df["F_daily_stop_price_f64"]
    ohlcv_df["Exit"] = p_df["F_daily_exit_f16"]
    
    if g_index_col in p_df.columns:
        base = pd.Timestamp("2000-01-01")
        ohlcv_df.index = base + pd.to_timedelta(p_df[g_index_col], unit="m")
    
    ohlcv_df = ohlcv_df.dropna(subset=["Open", "High", "Low", "Close"])
    
    return ohlcv_df


class DailySignalStrategy(Strategy):
    """
    Daily Signal strategy with stop-entry orders.
    
    Signal values:
        - 2: Long signal (bullish candle) - buy stop above high
        - 1: Short signal (bearish candle) - sell stop below low
    
    Parameters:
        size: Trade size (default: 0.1)
    """
    
    size: float = 0.1
    
    def init(self):
        pass
    
    def next(self):
        if self.data.Exit[-1] == 1:
            self.cancel_orders()
            self.close_trades()
            return
        
        if self.position or len(self.orders) > 0:
            return
        
        signal = int(self.data.df["signal"].iloc[-1])
        stop_price = float(self.data.df["StopPrice"].iloc[-1])
        
        if not np.isfinite(stop_price) or stop_price <= 0:
            return
        
        if signal == 2:
            self.buy(size=self.size, stop=stop_price)
        elif signal == 1:
            self.sell(size=self.size, stop=stop_price)
    
    def close_trades(self):
        for trade in self.trades:
            trade.close()
    
    def cancel_orders(self):
        while self.orders:
            self.orders[0].cancel()


def run_backtest(
    p_df: pd.DataFrame,
    p_cash: float = 100_000,
    p_commission: float = 0.0002,
    **kwargs
) -> Tuple[dict, pd.DataFrame]:
    """
    Run backtest with Daily Signal strategy.
    
    Args:
        p_df: DataFrame with framework columns + signal + StopPrice + Exit
        p_cash: Initial cash amount
        p_commission: Commission per trade
        **kwargs: Strategy parameters (size)
        
    Returns:
        Tuple of (backtest stats dict, trades DataFrame)
    """
    ohlcv_df = convert_to_ohlcv(p_df)
    
    bt = Backtest(
        ohlcv_df,
        DailySignalStrategy,
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
    p_test_candles: int = 8,
    p_exit_delay: int = 4,
    p_size: float = 0.1,
    p_cash: float = 100_000,
    p_commission: float = 0.0002,
    p_verbose: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """
    Main entry point: Load data, build features, run backtest.
    
    Args:
        p_test_candles: Number of candles for stop price calculation
        p_exit_delay: Days to delay before exit
        p_size: Trade size
        p_cash: Initial cash
        p_commission: Commission per trade
        p_verbose: Print results to console
        
    Returns:
        Tuple of (df with signals, backtest stats dict)
    """
    if p_verbose:
        print("=" * 80)
        print("Daily Signal Strategy - QQQ ETF (Hourly Data)")
        print("=" * 80)
    
    if p_verbose:
        print(f"\n1. Loading QQQ data...")
    df = load_data()
    if p_verbose:
        print(f"   Loaded {len(df)} bars")
    
    if p_verbose:
        print(f"\n2. Building features...")
        print(f"   Test candles: {p_test_candles}")
        print(f"   Exit delay: {p_exit_delay}")
    df = build_features(df, p_test_candles=p_test_candles, p_exit_delay=p_exit_delay)
    
    n_long = (df["F_daily_signal_f16"] == 2).sum()
    n_short = (df["F_daily_signal_f16"] == 1).sum()
    n_exits = df["F_daily_exit_f16"].sum()
    if p_verbose:
        print(f"   Long signals: {n_long}")
        print(f"   Short signals: {n_short}")
        print(f"   Exit signals: {n_exits}")
    
    if p_verbose:
        print(f"\n3. Running backtest...")
    
    stats, trades = run_backtest(
        df,
        p_cash=p_cash,
        p_commission=p_commission,
        size=p_size,
    )
    
    if p_verbose:
        strategy_return = stats['Return [%]']
        buy_hold_return = stats['Buy & Hold Return [%]']
        outperformance = strategy_return - buy_hold_return
        
        print("\n" + "=" * 80)
        print("BACKTEST RESULTS")
        print("=" * 80)
        
        print("\n--- PERFORMANCE SUMMARY ---")
        print(f"Start Date:            {stats['Start']}")
        print(f"End Date:              {stats['End']}")
        print(f"Duration:              {stats['Duration']}")
        
        print("\n--- RETURNS ---")
        print(f"Strategy Return [%]:       {strategy_return:.2f}")
        print(f"Buy & Hold Return [%]:     {buy_hold_return:.2f}")
        print(f"Outperformance [%]:        {outperformance:+.2f}")
        print(f"Max. Drawdown [%]:         {stats['Max. Drawdown [%]']:.2f}")
        print(f"Exposure Time [%]:         {stats['Exposure Time [%]']:.2f}")
        
        print("\n--- TRADE STATISTICS ---")
        print(f"# Trades:                  {stats['# Trades']}")
        print(f"Win Rate [%]:              {stats['Win Rate [%]']:.2f}")
        print(f"Best Trade [%]:            {stats['Best Trade [%]']:.2f}")
        print(f"Worst Trade [%]:           {stats['Worst Trade [%]']:.2f}")
        print(f"Avg. Trade [%]:            {stats['Avg. Trade [%]']:.2f}")
        
        print("\n--- STRATEGY PARAMETERS ---")
        print(f"Test Candles:              {p_test_candles}")
        print(f"Exit Delay:                {p_exit_delay}")
        print(f"Trade Size:                {p_size}")
        
        print("\n" + "=" * 80)
        
        if outperformance > 0:
            print(f">>> Strategy OUTPERFORMED Buy & Hold by {outperformance:.2f}%")
        else:
            print(f">>> Strategy UNDERPERFORMED Buy & Hold by {abs(outperformance):.2f}%")
        print("=" * 80)
    
    return df, stats


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Run Daily Signal Strategy on EURUSD forex data.

Usage:
    python scripts/strat/run_daily_signal_eurusd.py
    python scripts/strat/run_daily_signal_eurusd.py --test_candles 8 --exit_delay 4
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from features.f_daily_signal import feature_daily_signal, feature_daily_signal_with_exit
from norm.norm_utils import load_normalized_df
from core.enums import (
    g_open_col,
    g_high_col,
    g_low_col,
    g_close_col,
    g_volume_col,
    g_index_col,
    g_open_time_col,
    g_close_time_col,
)

DATA_PATH = Path("data/normalised/candle_1min/firstrate_undefined/spot/EURUSD/EURUSD_20100103_20260226_candle_1min.df.parquet")


def load_data(p_data_path: Optional[Path] = None) -> pd.DataFrame:
    """Load EURUSD forex data."""
    data_path = p_data_path or DATA_PATH
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")
    
    df = load_normalized_df(str(data_path))
    return df


def resample_to_hourly(p_df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-minute data to hourly."""
    df = p_df.copy()
    
    base = pd.Timestamp("2000-01-01")
    df["datetime"] = base + pd.to_timedelta(df[g_index_col].astype(int), unit="m")
    df.set_index("datetime", inplace=True)
    
    ohlcv_dict = {
        g_open_col: "first",
        g_high_col: "max",
        g_low_col: "min",
        g_close_col: "last",
        g_volume_col: "sum",
    }
    
    df_hourly = df.resample("1h").apply(ohlcv_dict).dropna(how="any")
    df_hourly.reset_index(inplace=True)
    
    df_hourly[g_index_col] = ((df_hourly["datetime"] - base).dt.total_seconds() / 60).astype("int64")
    df_hourly[g_open_time_col] = df_hourly[g_index_col] - 60
    df_hourly[g_close_time_col] = df_hourly[g_index_col]
    
    df_hourly = df_hourly.drop(columns=["datetime"])
    
    return df_hourly


def build_features(
    p_df: pd.DataFrame,
    p_test_candles: int = 8,
    p_exit_delay: int = 4,
) -> pd.DataFrame:
    """Build daily signal features from hourly data."""
    df = p_df.copy()
    
    signal = feature_daily_signal(df)
    stop_price, exit_signal = feature_daily_signal_with_exit(
        df,
        p_test_candles=p_test_candles,
        p_exit_delay=p_exit_delay,
    )
    
    df["F_daily_signal_f16"] = signal
    df["F_daily_stop_price_f64"] = stop_price
    df["F_daily_exit_f16"] = exit_signal
    
    return df


def convert_to_ohlcv(p_df: pd.DataFrame) -> pd.DataFrame:
    """Convert framework columns to standard OHLCV format for backtesting.py."""
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
    """Daily Signal strategy with stop-entry orders."""
    
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
    """Run backtest with Daily Signal strategy."""
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
    p_last_n_rows: int = -1,
) -> Tuple[pd.DataFrame, dict]:
    """Main entry point."""
    if p_verbose:
        print("=" * 80)
        print("Daily Signal Strategy - EURUSD Forex (Hourly Data)")
        print("=" * 80)
    
    if p_verbose:
        print(f"\n1. Loading EURUSD data...")
    df = load_data()
    if p_verbose:
        print(f"   Loaded {len(df)} 1-minute bars")
    
    if p_last_n_rows > 0:
        df = df.tail(p_last_n_rows).copy()
        if p_verbose:
            print(f"   Using last {len(df)} bars")
    
    if p_verbose:
        print(f"\n2. Resampling to hourly...")
    df_hourly = resample_to_hourly(df)
    if p_verbose:
        print(f"   Resampled to {len(df_hourly)} hourly bars")
    
    if p_verbose:
        print(f"\n3. Building features...")
        print(f"   Test candles: {p_test_candles}")
        print(f"   Exit delay: {p_exit_delay}")
    df_features = build_features(df_hourly, p_test_candles=p_test_candles, p_exit_delay=p_exit_delay)
    
    n_long = (df_features["F_daily_signal_f16"] == 2).sum()
    n_short = (df_features["F_daily_signal_f16"] == 1).sum()
    n_exits = int(df_features["F_daily_exit_f16"].sum())
    if p_verbose:
        print(f"   Long signals: {n_long}")
        print(f"   Short signals: {n_short}")
        print(f"   Exit signals: {n_exits}")
    
    if p_verbose:
        print(f"\n4. Running backtest...")
    
    stats, trades = run_backtest(
        df_features,
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
        
        print("\n--- RISK METRICS ---")
        print(f"Sharpe Ratio:              {stats['Sharpe Ratio']:.2f}")
        print(f"Sortino Ratio:             {stats['Sortino Ratio']:.2f}")
        print(f"Calmar Ratio:              {stats['Calmar Ratio']:.2f}")
        
        print("\n--- TRADE STATISTICS ---")
        print(f"# Trades:                  {stats['# Trades']}")
        win_rate = stats['Win Rate [%]']
        if np.isnan(win_rate):
            win_rate = 0.0
        print(f"Win Rate [%]:              {win_rate:.2f}")
        
        n_trades = stats['# Trades']
        n_wins = int(n_trades * win_rate / 100) if not np.isnan(win_rate) else 0
        n_losses = n_trades - n_wins
        print(f"# Winning Trades:          {n_wins}")
        print(f"# Losing Trades:           {n_losses}")
        
        if len(trades) > 0 and 'ReturnPct' in trades.columns:
            winning_trades = trades[trades['ReturnPct'] > 0]['ReturnPct']
            losing_trades = trades[trades['ReturnPct'] <= 0]['ReturnPct']
            
            avg_win = winning_trades.mean() * 100 if len(winning_trades) > 0 else 0
            avg_loss = losing_trades.mean() * 100 if len(losing_trades) > 0 else 0
            avg_win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            
            print(f"Avg. Winning Trade [%]:    {avg_win:.2f}")
            print(f"Avg. Losing Trade [%]:     {avg_loss:.2f}")
            print(f"Avg Win / Avg Loss Ratio:  {avg_win_loss_ratio:.2f}")
        
        print(f"Best Trade [%]:            {stats['Best Trade [%]']:.2f}")
        print(f"Worst Trade [%]:           {stats['Worst Trade [%]']:.2f}")
        print(f"Avg. Trade [%]:            {stats['Avg. Trade [%]']:.2f}")
        print(f"Max. Trade Duration:       {stats['Max. Trade Duration']}")
        print(f"Avg. Trade Duration:       {stats['Avg. Trade Duration']}")
        
        print("\n--- PROFIT ANALYSIS ---")
        print(f"Profit Factor:             {stats['Profit Factor']:.2f}")
        print(f"Expectancy [%]:            {stats['Expectancy [%]']:.2f}")
        print(f"SQN (System Quality):      {stats['SQN']:.2f}")
        
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
    
    return df_features, stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Daily Signal Strategy on EURUSD Forex"
    )
    parser.add_argument(
        "--test_candles",
        type=int,
        default=8,
        help="Number of candles for stop price calculation (default: 8)",
    )
    parser.add_argument(
        "--exit_delay",
        type=int,
        default=4,
        help="Days to delay before exit (default: 4)",
    )
    parser.add_argument(
        "--size",
        type=float,
        default=0.1,
        help="Trade size (default: 0.1)",
    )
    parser.add_argument(
        "--cash",
        type=float,
        default=100_000,
        help="Initial cash (default: 100000)",
    )
    parser.add_argument(
        "--commission",
        type=float,
        default=0.0002,
        help="Commission per trade (default: 0.0002)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output",
    )
    return parser.parse_args()


def main_cli():
    args = parse_args()
    
    df, stats = main(
        p_test_candles=args.test_candles,
        p_exit_delay=args.exit_delay,
        p_size=args.size,
        p_cash=args.cash,
        p_commission=args.commission,
        p_verbose=not args.quiet,
    )
    
    if args.quiet:
        print(f"Return: {stats['Return [%]']:.2f}%")
        print(f"Sharpe: {stats['Sharpe Ratio']:.2f}")
        print(f"Win Rate: {stats['Win Rate [%]']:.2f}%")
        print(f"Profit Factor: {stats['Profit Factor']:.2f}")
    
    return df, stats


if __name__ == "__main__":
    main_cli()

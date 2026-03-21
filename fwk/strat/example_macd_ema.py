#!/usr/bin/env python3
"""
Example script demonstrating MACD EMA Trend Strategy with real data.

This script uses the framework's data organization and feature generation
to run the MACD EMA strategy on actual ETF data.
"""

from pathlib import Path
import pandas as pd
import numpy as np

from core.data_org import BUNDLE_DIR
from features.base_dataframe import BaseDataFrame
from features.features_utils import FeatureType
from strat.s_macd_ema import (
    build_features,
    MACDEMA_SwingOrATR,
    MACDEMA_ATRTrail,
    run_backtest,
    optimize_backtest,
)


def load_or_create_etf_data(etf_symbol: str = "QQQ") -> pd.DataFrame:
    """
    Load ETF data from bundle or create from normalized files.
    
    Args:
        etf_symbol: ETF symbol to load (e.g., "QQQ", "SPY")
        
    Returns:
        DataFrame with OHLCV columns
    """
    bundle_file = BUNDLE_DIR / "candle_1hour" / f"{etf_symbol}_features.parquet"
    
    if bundle_file.exists():
        print(f"Loading {etf_symbol} from bundle: {bundle_file}")
        df = pd.read_parquet(bundle_file)
    else:
        raise FileNotFoundError(
            f"Bundle not found: {bundle_file}\n"
            f"Please run: python scripts/features/compute_etf_bundle.py"
        )
    
    df = convert_framework_columns_to_ohlcv(df, etf_symbol)
    
    return df


def convert_framework_columns_to_ohlcv(df: pd.DataFrame, prefix: str = None) -> pd.DataFrame:
    """
    Convert framework column format to standard OHLCV format.
    
    Framework uses: {PREFIX}_S_close_f32, {PREFIX}_S_open_f32, etc. (in bundles)
    Or without prefix: S_close_f32, S_open_f32, etc. (in individual files)
    Strategy expects: Close, Open, High, Low, Volume
    """
    ohlcv_df = pd.DataFrame(index=df.index)
    
    # Check if prefix exists in columns
    if prefix and f"{prefix}_S_close_f32" in df.columns:
        # Bundle format with prefix
        ohlcv_df["Close"] = df[f"{prefix}_S_close_f32"]
        ohlcv_df["Open"] = df[f"{prefix}_S_open_f32"]
        ohlcv_df["High"] = df[f"{prefix}_S_high_f32"]
        ohlcv_df["Low"] = df[f"{prefix}_S_low_f32"]
        ohlcv_df["Volume"] = df[f"{prefix}_S_volume_f64"]
    elif "S_close_f32" in df.columns:
        # Individual file format without prefix
        ohlcv_df["Close"] = df["S_close_f32"]
        ohlcv_df["Open"] = df["S_open_f32"]
        ohlcv_df["High"] = df["S_high_f32"]
        ohlcv_df["Low"] = df["S_low_f32"]
        ohlcv_df["Volume"] = df["S_volume_f64"]
    else:
        # Fallback: find columns by pattern
        for col_type, target in [
            ("close", "Close"),
            ("open", "Open"),
            ("high", "High"),
            ("low", "Low"),
            ("volume", "Volume"),
        ]:
            matching_cols = [c for c in df.columns if f"_S_{col_type}_" in c.lower()]
            if matching_cols:
                ohlcv_df[target] = df[matching_cols[0]]
    
    # Convert time column to datetime index if present
    if "time" in df.columns:
        ohlcv_df.index = pd.to_datetime(df["time"], unit='s')
    elif "i_minute_i" in df.columns:
        base = pd.Timestamp("2000-01-01")
        ohlcv_df.index = base + pd.to_timedelta(df["i_minute_i"], unit='m')
    elif "i_minute" in df.columns:
        base = pd.Timestamp("2000-01-01")
        ohlcv_df.index = base + pd.to_timedelta(df["i_minute"], unit='m')
    
    ohlcv_df = ohlcv_df.dropna()
    
    return ohlcv_df


def main():
    print("=" * 80)
    print("MACD EMA Trend Strategy - Real Data Example")
    print("=" * 80)
    
    etf_symbol = "QQQ"
    print(f"\n1. Loading {etf_symbol} data from bundle...")
    
    try:
        df_raw = load_or_create_etf_data(etf_symbol)
        print(f"   Loaded {len(df_raw)} bars")
        print(f"   Date range: {df_raw.index[0]} to {df_raw.index[-1]}")
        print(f"   Columns: {list(df_raw.columns)}")
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("\nPlease ensure you have run the bundle creation script:")
        print("  python scripts/features/compute_etf_bundle.py")
        return
    
    print(f"\n2. Building MACD EMA features...")
    df_features = build_features(df_raw)
    print(f"   Features built: {df_features.shape}")
    
    n_signals = (df_features['pre_signal'] != 0).sum()
    n_long = (df_features['pre_signal'] == 1).sum()
    n_short = (df_features['pre_signal'] == -1).sum()
    print(f"   Total signals: {n_signals} (Long: {n_long}, Short: {n_short})")
    
    print(f"\n3. Running backtest with MACDEMA_SwingOrATR...")
    stats = run_backtest(
        df_raw,
        strategy_class=MACDEMA_SwingOrATR,
        rr=2.0,
        sw_window=5,
        atr_mult=2.0,
    )
    
    print("\n" + "=" * 80)
    print(f"BACKTEST RESULTS - {etf_symbol}")
    print("=" * 80)
    print(f"Start Date:            {stats['Start']}")
    print(f"End Date:              {stats['End']}")
    print(f"Duration:              {stats['Duration']}")
    print(f"Exposure Time [%]:     {stats['Exposure Time [%]']:.2f}")
    print(f"Return [%]:            {stats['Return [%]']:.2f}")
    print(f"Buy & Hold Return [%]: {stats['Buy & Hold Return [%]']:.2f}")
    print(f"Sharpe Ratio:          {stats['Sharpe Ratio']:.2f}")
    print(f"Sortino Ratio:         {stats['Sortino Ratio']:.2f}")
    print(f"Max. Drawdown [%]:     {stats['Max. Drawdown [%]']:.2f}")
    print(f"# Trades:              {stats['# Trades']}")
    print(f"Win Rate [%]:          {stats['Win Rate [%]']:.2f}")
    print(f"Best Trade [%]:        {stats['Best Trade [%]']:.2f}")
    print(f"Worst Trade [%]:       {stats['Worst Trade [%]']:.2f}")
    print(f"Avg. Trade [%]:        {stats['Avg. Trade [%]']:.2f}")
    print(f"Profit Factor:         {stats['Profit Factor']:.2f}")
    print(f"Expectancy [%]:        {stats['Expectancy [%]']:.2f}")
    print(f"SQN:                   {stats['SQN']:.2f}")
    print("=" * 80)
    
    print(f"\n4. Running optimization on {etf_symbol}...")
    
    # Use last 10000 bars for optimization to avoid memory issues
    df_opt = df_raw.tail(10000) if len(df_raw) > 10000 else df_raw
    print(f"   Using last {len(df_opt)} bars for optimization")
    
    stats_opt, heatmap = optimize_backtest(
        df_opt,
        strategy_class=MACDEMA_SwingOrATR,
        rr_grid=[1.5, 2.0, 2.5],
        sw_window_grid=[3, 5, 7],
    )
    
    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS")
    print("=" * 80)
    print(f"Best parameters: {stats_opt._strategy}")
    print(f"Best Return [%]: {stats_opt['Return [%]']:.2f}")
    print(f"Best Sharpe:     {stats_opt['Sharpe Ratio']:.2f}")
    print(f"Best Win Rate:   {stats_opt['Win Rate [%]']:.2f}%")
    
    print("\nHeatmap (Return [%]):")
    if isinstance(heatmap, pd.Series):
        for idx, val in heatmap.items():
            print(f"  rr={idx[0]}, sw_window={idx[1]}: {val:.2f}%")
    else:
        print(heatmap.to_string())
    print("=" * 80)
    
    print(f"\n5. Testing ATR Trailing strategy on {etf_symbol}...")
    stats_trail = run_backtest(
        df_raw,
        strategy_class=MACDEMA_ATRTrail,
        atr_mult=2.5,
    )
    
    print(f"\nATR Trail Results:")
    print(f"  Return [%]:       {stats_trail['Return [%]']:.2f}")
    print(f"  Sharpe Ratio:     {stats_trail['Sharpe Ratio']:.2f}")
    print(f"  Max. Drawdown:    {stats_trail['Max. Drawdown [%]']:.2f}%")
    print(f"  # Trades:         {stats_trail['# Trades']}")
    print(f"  Win Rate:         {stats_trail['Win Rate [%]']:.2f}%")
    
    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)
    
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    stats_df = pd.DataFrame([{
        'Strategy': 'MACDEMA_SwingOrATR',
        'Return [%]': stats['Return [%]'],
        'Sharpe': stats['Sharpe Ratio'],
        'Max DD [%]': stats['Max. Drawdown [%]'],
        'Trades': stats['# Trades'],
        'Win Rate [%]': stats['Win Rate [%]'],
    }])
    
    output_file = output_dir / f"{etf_symbol}_macd_ema_results.csv"
    stats_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

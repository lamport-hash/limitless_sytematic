"""
Unit tests for MACD EMA Trend Strategy.

Tests feature building, signal generation, and backtesting with real bundle data.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from core.data_org import BUNDLE_DIR
from strat.strat_macd_ema import (
    build_features,
    ema_trend_signal,
    MACDEMA_SwingWindow,
    MACDEMA_SwingOrATR,
    MACDEMA_ATRTrail,
    run_backtest,
    optimize_backtest,
)


INPUT_FILE = BUNDLE_DIR / "test_etf_features_bundle.parquet"
OUTPUT_DIR = Path(__file__).parent / "output"


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
        ohlcv_df["Close"] = df[f"{prefix}_S_close_f32"]
        ohlcv_df["Open"] = df[f"{prefix}_S_open_f32"]
        ohlcv_df["High"] = df[f"{prefix}_S_high_f32"]
        ohlcv_df["Low"] = df[f"{prefix}_S_low_f32"]
        ohlcv_df["Volume"] = df[f"{prefix}_S_volume_f64"]
    elif "S_close_f32" in df.columns:
        ohlcv_df["Close"] = df["S_close_f32"]
        ohlcv_df["Open"] = df["S_open_f32"]
        ohlcv_df["High"] = df["S_high_f32"]
        ohlcv_df["Low"] = df["S_low_f32"]
        ohlcv_df["Volume"] = df["S_volume_f64"]
    else:
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
    
    return ohlcv_df.dropna()


@pytest.mark.skipif(not INPUT_FILE.exists(), reason=f"Test bundle not found: {INPUT_FILE}")
def test_build_features_with_real_data():
    """Test that build_features creates required columns with real data."""
    df = pd.read_parquet(INPUT_FILE)
    print(f"\nLoaded test bundle: {df.shape}")
    
    ohlcv_df = convert_framework_columns_to_ohlcv(df, "QQQ")
    print(f"Converted to OHLCV: {ohlcv_df.shape}")
    
    result = build_features(ohlcv_df)
    
    assert "EMA200" in result.columns
    assert "MACD_12_26_9" in result.columns
    assert "MACDs_12_26_9" in result.columns
    assert "MACDh_12_26_9" in result.columns
    assert "ema_signal" in result.columns
    assert "MACD_signal" in result.columns
    assert "pre_signal" in result.columns
    assert "ATR" in result.columns
    
    assert not result["EMA200"].isna().any()
    assert len(result) < len(ohlcv_df)


@pytest.mark.skipif(not INPUT_FILE.exists(), reason=f"Test bundle not found: {INPUT_FILE}")
def test_build_features_signals_range():
    """Test that signals are in valid range with real data."""
    df = pd.read_parquet(INPUT_FILE)
    ohlcv_df = convert_framework_columns_to_ohlcv(df, "QQQ")
    
    result = build_features(ohlcv_df)
    
    assert result["ema_signal"].isin([0, 1, -1]).all()
    assert result["MACD_signal"].isin([0, 1, -1]).all()
    assert result["pre_signal"].isin([0, 1, -1]).all()
    
    n_signals = (result['pre_signal'] != 0).sum()
    print(f"\nTotal signals generated: {n_signals}")
    assert n_signals > 0, "Should generate at least some signals"


@pytest.mark.skipif(not INPUT_FILE.exists(), reason=f"Test bundle not found: {INPUT_FILE}")
def test_ema_trend_signal_with_real_data():
    """Test EMA trend signal logic with real data."""
    df = pd.read_parquet(INPUT_FILE)
    ohlcv_df = convert_framework_columns_to_ohlcv(df, "QQQ")
    
    df_with_features = build_features(ohlcv_df)
    
    for i in range(10, min(50, len(df_with_features))):
        signal = ema_trend_signal(df_with_features, i)
        assert signal in [0, 1, -1]


@pytest.mark.skipif(not INPUT_FILE.exists(), reason=f"Test bundle not found: {INPUT_FILE}")
def test_run_backtest_with_real_data():
    """Test backtest with real bundle data."""
    df = pd.read_parquet(INPUT_FILE)
    ohlcv_df = convert_framework_columns_to_ohlcv(df, "QQQ")
    
    stats = run_backtest(ohlcv_df, strategy_class=MACDEMA_SwingOrATR)
    
    assert stats is not None
    assert "# Trades" in stats
    assert stats["# Trades"] >= 0
    
    print(f"\nBacktest results:")
    print(f"  Return: {stats['Return [%]']:.2f}%")
    print(f"  Sharpe: {stats['Sharpe Ratio']:.2f}")
    print(f"  Trades: {stats['# Trades']}")


@pytest.mark.skipif(not INPUT_FILE.exists(), reason=f"Test bundle not found: {INPUT_FILE}")
def test_run_backtest_different_strategies():
    """Test backtest with different strategy variants using real data."""
    df = pd.read_parquet(INPUT_FILE)
    ohlcv_df = convert_framework_columns_to_ohlcv(df, "QQQ")
    
    strategies = [MACDEMA_SwingWindow, MACDEMA_SwingOrATR, MACDEMA_ATRTrail]
    
    print(f"\n{'='*60}")
    print("Testing different strategies")
    print(f"{'='*60}")
    
    for strategy in strategies:
        stats = run_backtest(ohlcv_df, strategy_class=strategy)
        assert stats is not None
        assert stats["# Trades"] >= 0
        
        print(f"\n{strategy.__name__}:")
        print(f"  Return: {stats['Return [%]']:.2f}%")
        print(f"  Sharpe: {stats['Sharpe Ratio']:.2f}")
        print(f"  Trades: {stats['# Trades']}")


@pytest.mark.skipif(not INPUT_FILE.exists(), reason=f"Test bundle not found: {INPUT_FILE}")
def test_run_backtest_with_parameters():
    """Test backtest with custom parameters on real data."""
    df = pd.read_parquet(INPUT_FILE)
    ohlcv_df = convert_framework_columns_to_ohlcv(df, "QQQ")
    
    stats = run_backtest(
        ohlcv_df,
        strategy_class=MACDEMA_SwingOrATR,
        rr=2.0,
        sw_window=5,
        atr_mult=2.0,
    )
    
    assert stats is not None
    print(f"\nCustom parameters (rr=2.0, sw_window=5, atr_mult=2.0):")
    print(f"  Return: {stats['Return [%]']:.2f}%")


@pytest.mark.skipif(not INPUT_FILE.exists(), reason=f"Test bundle not found: {INPUT_FILE}")
def test_optimize_backtest_with_real_data():
    """Test backtest optimization on real data."""
    df = pd.read_parquet(INPUT_FILE)
    ohlcv_df = convert_framework_columns_to_ohlcv(df, "QQQ")
    
    rr_grid = [1.5, 2.0]
    sw_window_grid = [4, 6]
    
    stats, heatmap = optimize_backtest(
        ohlcv_df,
        strategy_class=MACDEMA_SwingOrATR,
        rr_grid=rr_grid,
        sw_window_grid=sw_window_grid,
    )
    
    assert stats is not None
    assert heatmap is not None
    assert len(heatmap) == len(rr_grid) * len(sw_window_grid)
    
    print(f"\n{'='*60}")
    print("Optimization Results")
    print(f"{'='*60}")
    print(f"Best parameters: {stats._strategy}")
    print(f"Best Return: {stats['Return [%]']:.2f}%")
    print(f"\nHeatmap:")
    print(heatmap)


@pytest.mark.skipif(not INPUT_FILE.exists(), reason=f"Test bundle not found: {INPUT_FILE}")
def test_backtest_with_multiple_etfs():
    """Test strategy on multiple ETFs from the bundle."""
    df = pd.read_parquet(INPUT_FILE)
    
    etfs_in_bundle = sorted(set(c.split("_")[0] for c in df.columns if "_S_close_" in c))
    print(f"\nETFs in bundle: {etfs_in_bundle}")
    
    print(f"\n{'='*60}")
    print("Testing strategy on multiple ETFs")
    print(f"{'='*60}")
    
    for etf in etfs_in_bundle[:3]:
        try:
            ohlcv_df = convert_framework_columns_to_ohlcv(df, etf)
            
            if len(ohlcv_df) < 200:
                print(f"\n{etf}: Skipped (insufficient data)")
                continue
            
            stats = run_backtest(
                ohlcv_df,
                strategy_class=MACDEMA_SwingOrATR,
                rr=2.0,
                sw_window=5,
            )
            
            print(f"\n{etf}:")
            print(f"  Bars: {len(ohlcv_df)}")
            print(f"  Return: {stats['Return [%]']:.2f}%")
            print(f"  Sharpe: {stats['Sharpe Ratio']:.2f}")
            print(f"  Trades: {stats['# Trades']}")
            print(f"  Win Rate: {stats['Win Rate [%]']:.2f}%")
            
        except Exception as e:
            print(f"\n{etf}: Error - {e}")


@pytest.mark.skipif(not INPUT_FILE.exists(), reason=f"Test bundle not found: {INPUT_FILE}")
def test_signal_generation_consistency():
    """Test that signals are generated consistently with real data."""
    df = pd.read_parquet(INPUT_FILE)
    ohlcv_df = convert_framework_columns_to_ohlcv(df, "QQQ")
    
    result1 = build_features(ohlcv_df)
    result2 = build_features(ohlcv_df)
    
    pd.testing.assert_frame_equal(result1, result2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

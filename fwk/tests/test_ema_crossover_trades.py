import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from strat.s_ema_crossover import build_features, convert_to_ohlcv, EMACrossoverStrategy


def test_ema_crossover_generates_trades():
    """Test that EMA Crossover strategy generates trades with correct parameters."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='1min')
    n = len(dates)

    close = 100 + np.cumsum(np.random.randn(n) * 0.1)
    high = close + np.abs(np.random.randn(n) * 0.05)
    low = close - np.abs(np.random.randn(n) * 0.05)
    open_ = close + np.random.randn(n) * 0.02
    volume = np.random.randint(1000, 10000, n)

    df = pd.DataFrame({
        'S_open_f32': open_,
        'S_high_f32': high,
        'S_low_f32': low,
        'S_close_f32': close,
        'S_volume_f64': volume,
        'i_minute_i': [(d - pd.Timestamp('2000-01-01')).total_seconds() / 60 for d in dates]
    })

    params = {'p_direction': 'both', 'p_fast_period': 12, 'p_slow_period': 50}

    df_feat = build_features(df.copy(), **params)

    n_signals = (df_feat['signal'] != 0).sum()
    print(f"Number of signals generated: {n_signals}")
    print(f"Signal distribution:\n{df_feat['signal'].value_counts()}")

    assert n_signals > 0, "Expected at least some signals to be generated"

    ohlcv = convert_to_ohlcv(df_feat)
    print(f"OHLCV shape: {ohlcv.shape}")
    print(f"OHLCV columns: {ohlcv.columns.tolist()}")

    assert len(ohlcv) > 0, "OHLCV data should not be empty"

    from backtesting import Backtest
    bt = Backtest(
        ohlcv,
        EMACrossoverStrategy,
        cash=100_000,
        commission=0.0002,
        trade_on_close=True
    )
    result = bt.run(atr_mult=2.0, rr=2.0, direction='both')

    print(f"Backtest result:")
    print(f"  # Trades: {result['# Trades']}")
    print(f"  Return [%]: {result['Return [%]']}")
    print(f"  Sharpe Ratio: {result['Sharpe Ratio']}")

    assert result['# Trades'] > 0, "Expected at least one trade to be generated"


def test_notebook_run_backtest_params():
    """Test the exact params used in the notebook to identify the bug."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='1min')
    n = len(dates)

    close = 100 + np.cumsum(np.random.randn(n) * 0.1)
    high = close + np.abs(np.random.randn(n) * 0.05)
    low = close - np.abs(np.random.randn(n) * 0.05)
    open_ = close + np.random.randn(n) * 0.02
    volume = np.random.randint(1000, 10000, n)

    df = pd.DataFrame({
        'S_open_f32': open_,
        'S_high_f32': high,
        'S_low_f32': low,
        'S_close_f32': close,
        'S_volume_f64': volume,
        'i_minute_i': [(d - pd.Timestamp('2000-01-01')).total_seconds() / 60 for d in dates]
    })

    default_params = {'p_fast_period': 12, 'p_slow_period': 50, 'p_direction': 'both'}
    test_params = {**default_params, 'atr_mult': 2.0, 'rr': 2.0}

    print("\n--- Testing original (buggy) filtering logic ---")
    build_params_buggy = {
        k: v for k, v in test_params.items()
        if not k.startswith('atr') and not k.startswith('rr') and k != 'direction'
    }
    print(f"Buggy build_params: {build_params_buggy}")

    print("\n--- Testing fixed filtering logic ---")
    build_params_fixed = {
        k: v for k, v in test_params.items()
        if not k.startswith('atr') and not k.startswith('rr') and not k.startswith('direction')
    }
    print(f"Fixed build_params: {build_params_fixed}")

    try:
        df_feat_buggy = build_features(df.copy(), **build_params_buggy)
        print("Buggy version succeeded - UNEXPECTED!")
    except TypeError as e:
        print(f"Buggy version failed as expected: {e}")

    df_feat_fixed = build_features(df.copy(), **build_params_fixed)
    print("Fixed version succeeded")
    n_signals = (df_feat_fixed['signal'] != 0).sum()
    print(f"Number of signals: {n_signals}")
    assert n_signals > 0


if __name__ == '__main__':
    test_ema_crossover_generates_trades()
    print("\n" + "="*80)
    test_notebook_run_backtest_params()
    print("\nAll tests passed!")

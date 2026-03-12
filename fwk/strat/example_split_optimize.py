#!/usr/bin/env python3
"""
Example demonstrating split and optimize with MACD EMA strategy.
"""

from pathlib import Path
import pandas as pd

from core.data_org import BUNDLE_DIR
from strat.strat_macd_ema import (
    build_features,
    MACDEMA_SwingOrATR,
    split_and_optimize,
    run_backtest,
    optimize_backtest,
)
from strat.example_macd_ema import convert_framework_columns_to_ohlcv


def main():
    print("=" * 80)
    print("MACD EMA Trend Strategy - Split & Optimize")
    print("=" * 80)
    
    etf_symbol = "QQQ"
    print(f"\n1. Loading {etf_symbol} data from bundle...")
    
    bundle_file = BUNDLE_DIR / "candle_1hour" / f"{etf_symbol}_features.parquet"
    
    if not bundle_file.exists():
        raise FileNotFoundError(
            f"Bundle not found: {bundle_file}\n"
            f"Please run: python scripts/features/compute_etf_bundle.py"
        )
    
    df = pd.read_parquet(bundle_file)
    print(f"   Loaded bundle: {df.shape}")
    
    ohlcv_df = convert_framework_columns_to_ohlcv(df)
    print(f"   Converted to OHLCV: {ohlcv_df.shape}")
    print(f"   Columns: {list(ohlcv_df.columns)}")
    
    df_features = build_features(ohlcv_df)
    n_signals = (df_features['pre_signal'] != 0).sum()
    n_long = (df_features['pre_signal'] == 1).sum()
    n_short = (df_features['pre_signal'] == -1).sum()
    print(f"   Total signals: {n_signals} (Long: {n_long}, Short: {n_short})")
    
    print(f"\n2. Running grid search optimization...")
    print(f"   Training period: 2000-01-01 to 2018-01-01")
    print(f"   Test period:     2018-01-01 to 2026-02-26")
    
    print(f"\n3. Optimizing on training data...")
    results = split_and_optimize(
        ohlcv_df,
        strategy_class=MACDEMA_SwingOrATR,
        split_ratio=0.5,        # Use 50% for training
        ema_len=200,       # Configurable EMA period
        rr_grid=[1.5, 2.0],
        sw_window_grid=[3, 5, 7],
    )
    print(f"\n4. Training Results:")
    print("=" * 60)
    print(f"Train Return [%]:    {results['train_stats']['Return [%]']:.2f}")
    print(f"Train Sharpe: :        {results['train_stats']['Sharpe Ratio']:.5f}")
    print(f"Train Win Rate [%]: {results['train_stats']['Win Rate [%]']:.5f}")
    print("=" * 60)
    
    print(f"\n5. Best Parameters:")
    for key, val in results['best_params'].items():
        print(f"  {key}: {val}")
    
    print(f"\n6. Test Results with Best Parameters")
    print("=" * 60)
    test_stats = results['test_stats']
    print(f"Test Return [%]:     {test_stats['Return [%]']:.5f}")
    print(f"Test Sharpe: :     {test_stats['Sharpe Ratio']:.5f}")
    print(f"Test Win Rate [%]:   {test_stats['Win Rate [%]']:.5f}")
    print(f"Test Max DD [%]:  {test_stats['Max. Drawdown [%]']:.5f}")
    print("=" * 60)
            
            print(f"\n7. Comparison")
            print("=" * 60)
            print(f"{'Train Return': {results['train_stats']['Return [%]']:.5f} vs {results['test_stats']['Return [%]']:.5f}")
            print(f"{'Train Sharpe': {results['train_stats']['Sharpe Ratio']:.5f} vs {results['test_stats']['Sharpe Ratio']:.5f}")
            print("=" * 60)
            
            print("\nOptimization complete!")
            print(f"Results saved to: {Path(__file__).parent / 'output' / 'split_optimize_results.csv')
            
            output_file = Path(__file__).parent / "output" / "split_optimize_results.csv"
            results_df.to_csv(output_file, index=False)
            print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

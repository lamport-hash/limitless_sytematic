"""
Integration test for TrendOptimizer using real FX data.
Loads 10 major FX pairs from normalized data and runs optimization.
"""

import pytest
import pandas as pd
import numpy as np
from typing import List
from pathlib import Path

from features.targets_trend_optimise import TrendOptimizer
from features.targets_trend import add_sma_trends, add_zigzag_trends

from norm.norm_utils import load_normalized_df
from core.data_org import NORMALISED_DIR, MktDataFred, ExchangeNAME, ProductType

MAJOR_FX_PAIRS = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "USDCHF",
    "AUDUSD",
    "NZDUSD",
    "USDCAD",
    "EURGBP",
    "EURJPY",
    "GBPJPY",
]

DATA_FREQ = "candle_1min"
SOURCE = "firstrate_undefined"
PRODUCT_TYPE = "spot"


def get_fx_data_path(symbol: str) -> Path:
    """Get path to normalized FX data file."""
    return (
        NORMALISED_DIR
        / DATA_FREQ
        / SOURCE
        / PRODUCT_TYPE
        / symbol
    )


def load_fx_pair(symbol: str, max_rows: int = 5000) -> pd.DataFrame:
    """Load a single FX pair and limit to max_rows."""
    pair_dir = get_fx_data_path(symbol)
    
    if not pair_dir.exists():
        pytest.skip(f"Data directory not found: {pair_dir}")
    
    parquet_files = list(pair_dir.glob("*.df.parquet"))
    if not parquet_files:
        pytest.skip(f"No parquet files found in {pair_dir}")
    
    df = load_normalized_df(str(parquet_files[0]))
    
    if max_rows and len(df) > max_rows:
        df = df.head(max_rows)
    
    return df


def align_dataframes(dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """Align multiple DataFrames on their common index."""
    if not dfs:
        return dfs
    
    common_index = dfs[0].index
    for df in dfs[1:]:
        common_index = common_index.intersection(df.index)
    
    return [df.loc[common_index].copy() for df in dfs]


@pytest.fixture(scope="module")
def fx_dataframes() -> List[pd.DataFrame]:
    """Load and align 10 major FX pairs."""
    dfs = []
    loaded_symbols = []
    
    for symbol in MAJOR_FX_PAIRS:
        try:
            df = load_fx_pair(symbol, max_rows=3000)
            dfs.append(df)
            loaded_symbols.append(symbol)
            print(f"Loaded {symbol}: {len(df)} rows")
        except Exception as e:
            print(f"Skipping {symbol}: {e}")
    
    if len(dfs) < 5:
        pytest.skip(f"Not enough FX pairs loaded (got {len(dfs)}, need at least 5)")
    
    aligned_dfs = align_dataframes(dfs)
    print(f"\nAligned {len(aligned_dfs)} pairs on {len(aligned_dfs[0])} common rows")
    
    return aligned_dfs


class TestTrendOptimizerRealData:
    """Tests using real FX data."""

    def test_sma_optimization_realdata(self, fx_dataframes):
        """Test SMA trend optimization on real FX data."""
        optimizer = TrendOptimizer(
            dfs=fx_dataframes,
            trend_function=add_sma_trends,
            timeframe='1min',
            max_long=5,
            max_short=2,
            cost=0.0001,
        )
        
        param_grid = {
            'long_window': [1440, 2880],
            'threshold': [0.002, 0.003],
            'min_length': [60, 120],
            'min_intensity': [0.10, 0.15],
            'prefix': ['sma_1min'],
        }
        
        results = optimizer.random_search(param_grid, n_iter=10, n_jobs=1, verbose=True)
        
        assert 'best_params' in results
        assert 'best_return' in results
        assert results['best_return'] > -1.0
        assert len(results['top_results']) > 0
        
        print(f"\nSMA Best Return: {results['best_return']:.4f}")
        print(f"SMA Best Params: {results['best_params']}")

    def test_zigzag_optimization_realdata(self, fx_dataframes):
        """Test ZigZag trend optimization on real FX data."""
        optimizer = TrendOptimizer(
            dfs=fx_dataframes,
            trend_function=add_zigzag_trends,
            timeframe='1min',
            max_long=5,
            max_short=2,
            cost=0.0002,
        )
        
        param_grid = {
            'threshold': [0.005, 0.01],
            'min_length': [15, 30],
            'prefix': ['zigzag_1min'],
        }
        
        results = optimizer.random_search(param_grid, n_iter=8, n_jobs=1, verbose=True)
        
        assert 'best_params' in results
        assert 'best_return' in results
        assert len(results['top_results']) > 0
        
        print(f"\nZigZag Best Return: {results['best_return']:.4f}")
        print(f"ZigZag Best Params: {results['best_params']}")

    def test_backtest_realdata(self, fx_dataframes):
        """Test backtest functionality on real FX data."""
        optimizer = TrendOptimizer(
            dfs=fx_dataframes,
            trend_function=add_sma_trends,
            timeframe='1min',
            max_long=5,
            max_short=2,
            cost=0.0002,
        )
        
        param_grid = {
            'long_window': [150],
            'threshold': [0.003],
            'min_length': [40],
            'min_intensity': [0.12],
            'prefix': ['sma_1min'],
        }
        
        optimizer.random_search(param_grid, n_iter=1, n_jobs=1, verbose=False)
        backtest = optimizer.backtest(verbose=True)
        
        assert 'total_return' in backtest
        assert 'sharpe_ratio' in backtest
        assert 'max_drawdown' in backtest
        assert 'win_rate' in backtest
        assert len(backtest['daily_returns']) > 0
        
        print(f"\nBacktest Total Return: {backtest['total_return']:.4f}")
        print(f"Sharpe Ratio: {backtest['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {backtest['max_drawdown']:.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

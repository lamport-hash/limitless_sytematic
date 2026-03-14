"""
Unit tests for TrendOptimizer using synthetic data.
Tests the core optimization functionality without requiring real data files.
"""

import pytest
import pandas as pd
import numpy as np
from typing import List

from features.targets_trend_optimise import (
    TrendOptimizer,
    get_param_grid_for_timeframe,
    scale_parameters_for_timeframe,
    TIMEFRAME_MINUTES,
    BASE_PARAM_GRIDS,
)
from features.targets_trend import add_sma_trends, add_zigzag_trends

from core.enums import g_open_col, g_high_col, g_low_col, g_close_col


@pytest.fixture
def synthetic_dfs() -> List[pd.DataFrame]:
    """Generate synthetic price data for 5 assets over 200 periods."""
    np.random.seed(42)
    n_periods = 200
    n_assets = 5
    dates = pd.date_range('2020-01-01', periods=n_periods, freq='D')
    
    common_factor = np.cumsum(np.random.randn(n_periods) * 0.01)
    
    dfs = []
    for asset_id in range(n_assets):
        asset_specific = np.random.randn(n_periods) * 0.02
        prices = 100 * np.exp(common_factor + asset_specific)
        prices = np.maximum(prices, 10)
        
        df = pd.DataFrame({
            g_open_col: prices,
            g_high_col: prices * (1 + np.abs(np.random.randn(n_periods) * 0.005)),
            g_low_col: prices * (1 - np.abs(np.random.randn(n_periods) * 0.005)),
            g_close_col: prices,
            'volume': np.random.randint(1000000, 10000000, n_periods).astype(np.int64),
        }, index=dates)
        dfs.append(df)
    
    return dfs


class TestTrendOptimizer:
    """Tests for TrendOptimizer class."""

    def test_trend_optimizer_init(self, synthetic_dfs):
        """Test TrendOptimizer initialization."""
        optimizer = TrendOptimizer(
            dfs=synthetic_dfs,
            trend_function=add_sma_trends,
            timeframe='1d',
        )
        
        assert optimizer.n_assets == 5
        assert optimizer.n_periods == 200
        assert optimizer.returns_matrix.shape == (200, 5)

    def test_random_search_basic(self, synthetic_dfs):
        """Test random search optimization."""
        optimizer = TrendOptimizer(
            dfs=synthetic_dfs,
            trend_function=add_sma_trends,
            timeframe='1d',
            max_long=3,
            max_short=1,
        )
        
        param_grid = {
            'long_window': [50, 100],
            'threshold': [0.03, 0.05],
            'min_length': [20, 30],
            'min_intensity': [0.15],
            'prefix': ['sma_1d'],
        }
        
        results = optimizer.random_search(param_grid, n_iter=5, n_jobs=1, verbose=False)
        
        assert 'best_params' in results
        assert 'best_return' in results
        assert 'statistics' in results
        assert results['n_tested'] == 5
        assert optimizer.best_params is not None

    def test_backtest(self, synthetic_dfs):
        """Test backtest functionality."""
        optimizer = TrendOptimizer(
            dfs=synthetic_dfs,
            trend_function=add_sma_trends,
            timeframe='1d',
        )
        
        param_grid = {
            'long_window': [50],
            'threshold': [0.03],
            'min_length': [20],
            'min_intensity': [0.15],
            'prefix': ['sma_1d'],
        }
        
        optimizer.random_search(param_grid, n_iter=1, n_jobs=1, verbose=False)
        backtest_results = optimizer.backtest(verbose=False)
        
        assert 'total_return' in backtest_results
        assert 'sharpe_ratio' in backtest_results
        assert 'max_drawdown' in backtest_results
        assert 'win_rate' in backtest_results
        assert 'daily_returns' in backtest_results

    def test_zigzag_optimization(self, synthetic_dfs):
        """Test zigzag trend optimization."""
        optimizer = TrendOptimizer(
            dfs=synthetic_dfs,
            trend_function=add_zigzag_trends,
            timeframe='1d',
            max_long=2,
            max_short=1,
        )
        
        param_grid = {
            'threshold': [0.03, 0.05],
            'min_length': [10, 15],
            'prefix': ['zigzag_1d'],
        }
        
        results = optimizer.random_search(param_grid, n_iter=4, n_jobs=1, verbose=False)
        
        assert results['n_tested'] == 4
        assert 'top_results' in results
        assert len(results['top_results']) > 0


class TestParameterScaling:
    """Tests for parameter scaling functions."""

    def test_scale_parameters_1d_to_1h(self):
        """Test scaling parameters from daily to hourly."""
        base_params = {'long_window': [100, 200], 'threshold': [0.05]}
        scaled = scale_parameters_for_timeframe(base_params, '1d', '1h')
        
        assert scaled['long_window'][0] == 100 * 24
        assert scaled['long_window'][1] == 200 * 24
        assert scaled['threshold'] == [0.05]

    def test_scale_parameters_1d_to_1w(self):
        """Test scaling parameters from daily to weekly."""
        base_params = {'min_length': [20, 30], 'threshold': [0.05]}
        scaled = scale_parameters_for_timeframe(base_params, '1d', '1w')
        
        assert scaled['min_length'][0] == 5
        assert len(scaled['min_length']) == 1
        assert scaled['threshold'] == [0.05]

    def test_get_param_grid_for_timeframe(self):
        """Test getting parameter grid for specific timeframe."""
        grid_1d = get_param_grid_for_timeframe('sma', '1d')
        grid_1h = get_param_grid_for_timeframe('sma', '1h')
        
        assert 'long_window' in grid_1d
        assert 'prefix' in grid_1d
        assert grid_1d['prefix'] == ['sma_1d']
        assert grid_1h['prefix'] == ['sma_1h']
        
        assert grid_1h['long_window'][0] > grid_1d['long_window'][0]


class TestTimeframeDefinitions:
    """Tests for timeframe constants."""

    def test_timeframe_minutes_defined(self):
        """Test that all expected timeframes are defined."""
        expected = ['1min', '5min', '15min', '30min', '1h', '4h', '1d', '1w', '1M']
        for tf in expected:
            assert tf in TIMEFRAME_MINUTES

    def test_timeframe_scaling_factors(self):
        """Test that timeframe scaling factors are correct."""
        assert TIMEFRAME_MINUTES['1h'] == 60
        assert TIMEFRAME_MINUTES['1d'] == 1440
        assert TIMEFRAME_MINUTES['1w'] == 10080

    def test_base_param_grids_defined(self):
        """Test that all base parameter grids are defined."""
        expected_methods = ['sma', 'zigzag', 'regression', 'directional']
        for method in expected_methods:
            assert method in BASE_PARAM_GRIDS
            assert isinstance(BASE_PARAM_GRIDS[method], dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

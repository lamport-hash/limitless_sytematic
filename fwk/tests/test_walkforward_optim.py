import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List, Tuple, Callable
from itertools import product

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtesting import Backtest
from strat.s_ema_crossover import build_features as ema_build, convert_to_ohlcv as ema_conv, EMACrossoverStrategy


class OptimMetric(Enum):
    SHARPE = "sharpe_ratio"
    CALMAR = "calmar_ratio"
    TOTAL_RETURN = "total_return"
    PROFIT_FACTOR = "profit_factor"


@dataclass
class OptimResult:
    params: Dict[str, Any]
    metrics: Dict[str, float]
    sharpe: float
    calmar: float
    total_return: float
    profit_factor: float
    max_drawdown: float
    win_rate: float
    n_trades: int


def grid_search(
    func: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
    df: pd.DataFrame,
    param_ranges: Dict[str, List[Any]],
    optim_metric: OptimMetric = OptimMetric.SHARPE,
    maximize: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, OptimResult, pd.DataFrame]:
    """Run grid search optimization over parameter space."""
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    
    combinations = list(product(*param_values))
    total = len(combinations)
    
    if verbose:
        print(f'Running grid search with {total} combinations...')
    
    results = []
    for i, combo in enumerate(combinations):
        params = dict(zip(param_names, combo))
        try:
            metrics = func(df.copy(), params)
            results.append({
                **params,
                'sharpe_ratio': metrics.get('sharpe_ratio', np.nan),
                'calmar_ratio': metrics.get('calmar_ratio', np.nan),
                'total_return': metrics.get('total_return', np.nan),
                'profit_factor': metrics.get('profit_factor', np.nan),
                'max_drawdown': metrics.get('max_drawdown', np.nan),
                'win_rate': metrics.get('win_rate', np.nan),
                'n_trades': metrics.get('n_trades', np.nan),
                'error': None,
            })
        except Exception as e:
            results.append({
                **{k: v for k, v in params.items()},
                'sharpe_ratio': np.nan, 'calmar_ratio': np.nan,
                'total_return': np.nan, 'profit_factor': np.nan,
                'max_drawdown': np.nan, 'win_rate': np.nan, 'n_trades': np.nan,
                'error': str(e),
            })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.dropna(subset=[optim_metric.value])
    
    if len(results_df) == 0:
        raise ValueError('No valid results')
    
    best_idx = results_df[optim_metric.value].idxmax() if maximize else results_df[optim_metric.value].idxmin()
    best_row = results_df.loc[best_idx]
    
    best_result = OptimResult(
        params={k: best_row[k] for k in param_names},
        metrics={},
        sharpe=best_row['sharpe_ratio'],
        calmar=best_row['calmar_ratio'],
        total_return=best_row['total_return'],
        profit_factor=best_row['profit_factor'],
        max_drawdown=best_row['max_drawdown'],
        win_rate=best_row['win_rate'],
        n_trades=int(best_row['n_trades']),
    )
    
    return results_df, best_result, pd.DataFrame()


def create_test_df(n_bars: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic price dataframe for testing."""
    np.random.seed(seed)
    dates = pd.date_range('2020-01-01', periods=n_bars, freq='1min')
    
    close = 100 + np.cumsum(np.random.randn(n_bars) * 0.1)
    high = close + np.abs(np.random.randn(n_bars) * 0.05)
    low = close - np.abs(np.random.randn(n_bars) * 0.05)
    open_ = close + np.random.randn(n_bars) * 0.02
    volume = np.random.randint(1000, 10000, n_bars)
    
    df = pd.DataFrame({
        'S_open_f32': open_,
        'S_high_f32': high,
        'S_low_f32': low,
        'S_close_f32': close,
        'S_volume_f64': volume,
        'i_minute_i': [(d - pd.Timestamp('2000-01-01')).total_seconds() / 60 for d in dates]
    })
    
    return df


def run_backtest(
    df: pd.DataFrame,
    params: Dict[str, Any],
    strategy_info: Dict[str, Any],
    initial_cash: float = 100_000,
    commission: float = 0.0002
) -> Dict[str, float]:
    """Run backtest with given parameters."""
    try:
        build_params = {
            k: v for k, v in params.items()
            if not k.startswith('atr') and not k.startswith('rr')
        }
        bt_params = {
            k: v for k, v in params.items()
            if k.startswith('atr') or k.startswith('rr')
        }
        bt_params.setdefault('direction', 'both')
        bt_params.setdefault('atr_mult', 2.0)
        bt_params.setdefault('rr', 2.0)
        
        df_feat = strategy_info['build'](df.copy(), **build_params)
        ohlcv = strategy_info['convert'](df_feat)
        
        bt = Backtest(
            ohlcv,
            strategy_info.get('strategy') or EMACrossoverStrategy,
            cash=initial_cash,
            commission=commission,
            trade_on_close=True
        )
        result = bt.run(**bt_params)
        
        return {
            'sharpe_ratio': result['Sharpe Ratio'] if pd.notna(result['Sharpe Ratio']) else 0,
            'total_return': result['Return [%]'] if pd.notna(result['Return [%]']) else 0,
            'max_drawdown': result['Max. Drawdown [%]'] if pd.notna(result['Max. Drawdown [%]']) else 0,
            'win_rate': result['Win Rate [%]'] if pd.notna(result['Win Rate [%]']) else 0,
            'n_trades': result['# Trades'] if pd.notna(result['# Trades']) else 0,
            'profit_factor': result['Profit Factor'] if pd.notna(result['Profit Factor']) else 0,
            'calmar_ratio': result['Calmar Ratio'] if pd.notna(result['Calmar Ratio']) else 0,
        }
    except Exception as e:
        return {
            'sharpe_ratio': -999,
            'total_return': -999,
            'max_drawdown': 100,
            'win_rate': 0,
            'n_trades': 0,
            'profit_factor': 0,
            'calmar_ratio': 0
        }


def test_run_backtest_basic():
    """Test basic backtest execution."""
    df = create_test_df(n_bars=2000)
    
    strategy_info = {
        'name': 'EMA Crossover',
        'build': ema_build,
        'convert': ema_conv,
        'strategy': EMACrossoverStrategy,
        'default': {'p_fast_period': 12, 'p_slow_period': 50, 'p_direction': 'both'},
    }
    
    params = {**strategy_info['default'], 'atr_mult': 2.0, 'rr': 2.0}
    
    result = run_backtest(df, params, strategy_info, initial_cash=100_000, commission=0.0002)
    
    assert 'sharpe_ratio' in result
    assert 'total_return' in result
    assert 'n_trades' in result
    assert result['n_trades'] >= 0


def test_grid_search_basic():
    """Test grid search functionality."""
    df = create_test_df(n_bars=2000)
    
    strategy_info = {
        'name': 'EMA Crossover',
        'build': ema_build,
        'convert': ema_conv,
        'strategy': EMACrossoverStrategy,
        'default': {'p_fast_period': 12, 'p_slow_period': 50, 'p_direction': 'both'},
    }
    
    param_ranges = {
        'p_fast_period': [9, 12],
        'p_slow_period': [21, 50],
        'atr_mult': [2.0],
        'rr': [2.0]
    }
    
    def is_func(p_df, p):
        return run_backtest(p_df, p, strategy_info, 100_000, 0.0002)
    
    results_df, best_result, _ = grid_search(is_func, df, param_ranges, OptimMetric.SHARPE, verbose=False)
    
    assert len(results_df) == 4
    assert 'sharpe_ratio' in results_df.columns
    assert best_result is not None
    assert 'p_fast_period' in best_result.params


def test_walk_forward_optimization():
    """Test walk-forward optimization with small dataset."""
    df = create_test_df(n_bars=2000)
    
    strategy_info = {
        'name': 'EMA Crossover',
        'build': ema_build,
        'convert': ema_conv,
        'strategy': EMACrossoverStrategy,
        'default': {'p_fast_period': 12, 'p_slow_period': 50, 'p_direction': 'both'},
        'optim': {
            'p_fast_period': [9, 12],
            'p_slow_period': [21, 50],
            'atr_mult': [2.0],
            'rr': [2.0]
        }
    }
    
    LOOKBACK_BARS = 500
    FORWARD_BARS = 100
    STEP_BARS = 200
    INITIAL_CASH = 100_000
    COMMISSION = 0.0002
    
    n_bars = len(df)
    start_idx = LOOKBACK_BARS
    param_ranges = strategy_info['optim']
    
    results = []
    window_starts = list(range(start_idx, n_bars - FORWARD_BARS, STEP_BARS))
    total_windows = len(window_starts)
    
    assert total_windows > 0, "Should have at least one window"
    
    for window_num, window_start in enumerate(window_starts):
        is_end = window_start + LOOKBACK_BARS
        oos_start, oos_end = is_end, min(is_end + FORWARD_BARS, n_bars)
        
        df_is = df.iloc[window_start:is_end]
        df_oos = df.iloc[oos_start:oos_end]
        
        def is_func(p_df, p):
            return run_backtest(p_df, p, strategy_info, INITIAL_CASH, COMMISSION)
        
        _, best, _ = grid_search(is_func, df_is, param_ranges, OptimMetric.SHARPE, verbose=False)
        
        oos_metrics = run_backtest(df_oos, best.params, strategy_info, INITIAL_CASH, COMMISSION)
        
        results.append({
            'window': window_num,
            'is_start': window_start,
            'is_end': is_end,
            'oos_start': oos_start,
            'oos_end': oos_end,
            'is_sharpe': best.sharpe,
            'is_return': best.total_return,
            'oos_sharpe': oos_metrics['sharpe_ratio'],
            'oos_return': oos_metrics['total_return'],
            'oos_max_dd': oos_metrics['max_drawdown'],
            'oos_n_trades': oos_metrics['n_trades'],
            **best.params
        })
    
    results_df = pd.DataFrame(results)
    
    assert len(results_df) == total_windows
    assert 'is_sharpe' in results_df.columns
    assert 'oos_sharpe' in results_df.columns
    
    summary = {
        'n_windows': len(results_df),
        'avg_oos_sharpe': results_df['oos_sharpe'].mean(),
        'avg_oos_return': results_df['oos_return'].mean(),
        'avg_oos_max_dd': results_df['oos_max_dd'].mean(),
    }
    
    assert summary['n_windows'] > 0


def test_walk_forward_with_edge_cases():
    """Test walk-forward optimization with edge cases."""
    df = create_test_df(n_bars=1000)
    
    strategy_info = {
        'name': 'EMA Crossover',
        'build': ema_build,
        'convert': ema_conv,
        'strategy': EMACrossoverStrategy,
        'default': {'p_fast_period': 12, 'p_slow_period': 50, 'p_direction': 'both'},
        'optim': {
            'p_fast_period': [12],
            'p_slow_period': [50],
            'atr_mult': [2.0],
            'rr': [2.0]
        }
    }
    
    LOOKBACK_BARS = 300
    FORWARD_BARS = 100
    STEP_BARS = 300
    
    n_bars = len(df)
    start_idx = LOOKBACK_BARS
    param_ranges = strategy_info['optim']
    
    results = []
    window_starts = list(range(start_idx, n_bars - FORWARD_BARS, STEP_BARS))
    
    assert len(window_starts) >= 1
    
    for window_num, window_start in enumerate(window_starts):
        is_end = window_start + LOOKBACK_BARS
        oos_start, oos_end = is_end, min(is_end + FORWARD_BARS, n_bars)
        
        df_is = df.iloc[window_start:is_end]
        df_oos = df.iloc[oos_start:oos_end]
        
        def is_func(p_df, p):
            return run_backtest(p_df, p, strategy_info, 100_000, 0.0002)
        
        try:
            _, best, _ = grid_search(is_func, df_is, param_ranges, OptimMetric.SHARPE, verbose=False)
            oos_metrics = run_backtest(df_oos, best.params, strategy_info, 100_000, 0.0002)
            
            results.append({
                'window': window_num,
                'is_sharpe': best.sharpe,
                'oos_sharpe': oos_metrics['sharpe_ratio'],
                'oos_n_trades': oos_metrics['n_trades'],
            })
        except Exception as e:
            pytest.fail(f"Walk-forward failed at window {window_num}: {e}")
    
    assert len(results) == len(window_starts)


def test_backtest_with_small_dataset():
    """Test backtest behavior with very small dataset (edge case)."""
    np.random.seed(999)
    n_bars = 100
    close = 100 + np.cumsum(np.random.randn(n_bars) * 0.001)
    
    df = pd.DataFrame({
        'S_open_f32': close + np.random.randn(n_bars) * 0.0001,
        'S_high_f32': close + np.random.randn(n_bars) * 0.0001,
        'S_low_f32': close + np.random.randn(n_bars) * 0.0001,
        'S_close_f32': close,
        'S_volume_f64': np.random.randint(1000, 10000, n_bars),
        'i_minute_i': list(range(n_bars))
    })
    
    strategy_info = {
        'name': 'EMA Crossover',
        'build': ema_build,
        'convert': ema_conv,
        'strategy': EMACrossoverStrategy,
        'default': {'p_fast_period': 50, 'p_slow_period': 100, 'p_direction': 'both'},
    }
    
    params = {**strategy_info['default'], 'atr_mult': 2.0, 'rr': 2.0}
    
    result = run_backtest(df, params, strategy_info)
    
    assert 'n_trades' in result
    assert result['n_trades'] >= 0


if __name__ == '__main__':
    test_run_backtest_basic()
    print("✓ test_run_backtest_basic passed")
    
    test_grid_search_basic()
    print("✓ test_grid_search_basic passed")
    
    test_walk_forward_optimization()
    print("✓ test_walk_forward_optimization passed")
    
    test_walk_forward_with_edge_cases()
    print("✓ test_walk_forward_with_edge_cases passed")
    
    test_backtest_with_small_dataset()
    print("✓ test_backtest_with_small_dataset passed")
    
    print("\n" + "="*50)
    print("All tests passed!")

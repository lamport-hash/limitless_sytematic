"""
Memory-optimized walk-forward optimization for Jupyter notebooks.

Fixes memory issues by:
1. Minimizing dataframe copies
2. Explicit garbage collection
3. Deleting intermediate objects
4. Using smaller data slices when possible
"""
import gc
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Callable
from itertools import product

from backtesting import Backtest
from strat.s_ema_crossover import build_features as ema_build, convert_to_ohlcv as ema_conv, EMACrossoverStrategy


class OptimMetric:
    SHARPE = "sharpe_ratio"
    CALMAR = "calmar_ratio"
    TOTAL_RETURN = "total_return"
    PROFIT_FACTOR = "profit_factor"


def run_backtest_optimized(
    df: pd.DataFrame,
    params: Dict[str, Any],
    strategy_info: Dict[str, Any],
    initial_cash: float = 100_000,
    commission: float = 0.0002
) -> Dict[str, float]:
    """
    Memory-optimized backtest wrapper.
    Avoids unnecessary dataframe copies and cleans up intermediate objects.
    """
    try:
        # Filter params without copying the entire dict unnecessarily
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
        
        # Only copy once for feature building
        df_feat = strategy_info['build'](df, **build_params)
        ohlcv = strategy_info['convert'](df_feat)
        
        # Run backtest
        bt = Backtest(
            ohlcv,
            strategy_info.get('strategy') or EMACrossoverStrategy,
            cash=initial_cash,
            commission=commission,
            trade_on_close=True
        )
        result = bt.run(**bt_params)
        
        # Clean up large objects immediately
        del df_feat, ohlcv, bt
        
        metrics = {
            'sharpe_ratio': result['Sharpe Ratio'] if pd.notna(result['Sharpe Ratio']) else 0,
            'total_return': result['Return [%]'] if pd.notna(result['Return [%]']) else 0,
            'max_drawdown': result['Max. Drawdown [%]'] if pd.notna(result['Max. Drawdown [%]']) else 0,
            'win_rate': result['Win Rate [%]'] if pd.notna(result['Win Rate [%]']) else 0,
            'n_trades': result['# Trades'] if pd.notna(result['# Trades']) else 0,
            'profit_factor': result['Profit Factor'] if pd.notna(result['Profit Factor']) else 0,
            'calmar_ratio': result['Calmar Ratio'] if pd.notna(result['Calmar Ratio']) else 0,
        }
        
        return metrics
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


def grid_search_optimized(
    func: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
    df: pd.DataFrame,
    param_ranges: Dict[str, List[Any]],
    optim_metric: str = "sharpe_ratio",
    maximize: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Memory-optimized grid search.
    Minimizes memory usage during parameter sweeps.
    """
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
            # Don't copy df here - let the function handle it
            metrics = func(df, params)
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
        
        # Progress reporting
        if verbose and (i + 1) % max(1, total // 10) == 0:
            print(f'  Progress: {i + 1}/{total} ({(i+1)/total*100:.1f}%)')
    
    results_df = pd.DataFrame(results)
    results_df = results_df.dropna(subset=[optim_metric])
    
    if len(results_df) == 0:
        raise ValueError('No valid results')
    
    best_idx = results_df[optim_metric].idxmax() if maximize else results_df[optim_metric].idxmin()
    best_row = results_df.loc[best_idx]
    
    best_result = {
        'params': {k: best_row[k] for k in param_names},
        'sharpe': best_row['sharpe_ratio'],
        'calmar': best_row['calmar_ratio'],
        'total_return': best_row['total_return'],
        'profit_factor': best_row['profit_factor'],
        'max_drawdown': best_row['max_drawdown'],
        'win_rate': best_row['win_rate'],
        'n_trades': int(best_row['n_trades']),
    }
    
    return results_df, best_result


def walk_forward_optimization(
    df: pd.DataFrame,
    strategy_info: Dict[str, Any],
    lookback_bars: int = 4400,
    forward_bars: int = 100,
    step_bars: int = 500,
    initial_cash: float = 100_000,
    commission: float = 0.0002,
    optim_metric: str = "sharpe_ratio",
    verbose: bool = True
) -> pd.DataFrame:
    """
    Memory-optimized walk-forward optimization.
    """
    n_bars = len(df)
    start_idx = lookback_bars
    param_ranges = strategy_info['optim']
    
    # Pre-calculate windows to save memory
    window_configs = []
    for window_num, window_start in enumerate(range(start_idx, n_bars - forward_bars, step_bars)):
        is_end = window_start + lookback_bars
        oos_start, oos_end = is_end, min(is_end + forward_bars, n_bars)
        window_configs.append({
            'window': window_num,
            'window_start': window_start,
            'is_end': is_end,
            'oos_start': oos_start,
            'oos_end': oos_end
        })
    
    total_windows = len(window_configs)
    
    if verbose:
        print(f'Walk-forward: {n_bars} bars, IS={lookback_bars}, OOS={forward_bars}, step={step_bars}')
        print(f'Total windows: {total_windows}')
    
    results = []
    
    for config in window_configs:
        window_num = config['window']
        
        # Use .iloc slicing (views are more memory-efficient than copies)
        df_is = df.iloc[config['window_start']:config['is_end']].copy()
        df_oos = df.iloc[config['oos_start']:config['oos_end']].copy()
        
        def is_func(p_df, p):
            return run_backtest_optimized(p_df, p, strategy_info, initial_cash, commission)
        
        _, best = grid_search_optimized(
            is_func, df_is, param_ranges, optim_metric, maximize=True, verbose=False
        )
        
        oos_metrics = run_backtest_optimized(
            df_oos, best['params'], strategy_info, initial_cash, commission
        )
        
        results.append({
            'window': window_num,
            'is_start': config['window_start'],
            'is_end': config['is_end'],
            'oos_start': config['oos_start'],
            'oos_end': config['oos_end'],
            'is_sharpe': best['sharpe'],
            'is_return': best['total_return'],
            'oos_sharpe': oos_metrics['sharpe_ratio'],
            'oos_return': oos_metrics['total_return'],
            'oos_max_dd': oos_metrics['max_drawdown'],
            'oos_n_trades': oos_metrics['n_trades'],
            **best['params']
        })
        
        if verbose and (window_num + 1) % max(1, total_windows // 10) == 0:
            print(f'  Progress: {window_num + 1}/{total_windows} windows')
        
        # Force garbage collection after each window
        del df_is, df_oos, best, oos_metrics
        gc.collect()
    
    results_df = pd.DataFrame(results)
    
    return results_df


# Example usage for Jupyter notebook:
if __name__ == '__main__':
    # Small test to verify it works
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from tests.test_walkforward_optim import create_test_df
    
    print("Testing memory-optimized walk-forward...")
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
    
    results = walk_forward_optimization(
        df,
        strategy_info,
        lookback_bars=500,
        forward_bars=100,
        step_bars=200,
        verbose=True
    )
    
    print(f"\nCompleted {len(results)} windows")
    print(f"Avg OOS Sharpe: {results['oos_sharpe'].mean():.3f}")

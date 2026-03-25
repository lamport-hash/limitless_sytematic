"""
Generic parameter optimization for trading strategies.

Provides grid search optimization over parameter spaces with support for:
- Custom objective functions (sharpe, calmar, total_return, profit_factor)
- Allocation-based strategies (via create_alloc_backtest_func wrapper)
- Pivot table generation for 2D visualization
"""

from enum import Enum
from dataclasses import dataclass
from typing import Callable, Dict, List, Any, Tuple
from itertools import product

import numpy as np
import pandas as pd

from backtest.backtest_basket_alloc_based import run_full_backtest
from backtest.backtest_utils import compute_strategy_metrics


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


def grid_search(
    func: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
    df: pd.DataFrame,
    param_ranges: Dict[str, List[Any]],
    optim_metric: OptimMetric = OptimMetric.SHARPE,
    maximize: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, OptimResult, pd.DataFrame]:
    """
    Run grid search optimization over parameter space.
    
    Args:
        func: Function that takes (df, params) and returns metrics dict.
              Metrics dict must contain keys matching OptimMetric values.
        df: Input DataFrame to pass to func.
        param_ranges: Dict mapping param names to lists of values.
                      e.g., {'lookback': [5, 10, 20], 'threshold': [0.1, 0.2]}
        optim_metric: Which metric to optimize (default: SHARPE).
        maximize: If True, maximize the metric; if False, minimize.
        verbose: Print progress information.
    
    Returns:
        Tuple of:
        - results_df: DataFrame with all combinations, params, and metrics
        - best_result: OptimResult with best parameters
        - pivot_df: Pivot table for 2D visualization (empty if not 2 params)
    """
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    
    combinations = list(product(*param_values))
    total = len(combinations)
    
    if verbose:
        print(f"Running grid search with {total} parameter combinations...")
    
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
            if verbose:
                print(f"Error for params {params}: {e}")
            results.append({
                **params,
                'sharpe_ratio': np.nan,
                'calmar_ratio': np.nan,
                'total_return': np.nan,
                'profit_factor': np.nan,
                'max_drawdown': np.nan,
                'win_rate': np.nan,
                'n_trades': np.nan,
                'error': str(e),
            })
        
        if verbose and (i + 1) % max(1, total // 10) == 0:
            print(f"  Progress: {i + 1}/{total}")
    
    results_df = pd.DataFrame(results)
    
    metric_col = optim_metric.value
    valid_mask = ~results_df[metric_col].isna()
    
    if not valid_mask.any():
        raise ValueError(f"No valid results found for metric {optim_metric.name}")
    
    valid_results = results_df[valid_mask]
    
    if maximize:
        best_idx = valid_results[metric_col].idxmax()
    else:
        best_idx = valid_results[metric_col].idxmin()
    
    best_row = results_df.loc[best_idx]
    
    best_result = OptimResult(
        params={k: best_row[k] for k in param_names},
        metrics={
            'sharpe_ratio': best_row['sharpe_ratio'],
            'calmar_ratio': best_row['calmar_ratio'],
            'total_return': best_row['total_return'],
            'profit_factor': best_row['profit_factor'],
            'max_drawdown': best_row['max_drawdown'],
            'win_rate': best_row['win_rate'],
            'n_trades': best_row['n_trades'],
        },
        sharpe=float(best_row['sharpe_ratio']),
        calmar=float(best_row['calmar_ratio']),
        total_return=float(best_row['total_return']),
        profit_factor=float(best_row['profit_factor']),
    )
    
    pivot_df = pd.DataFrame()
    if len(param_names) == 2:
        x_param, y_param = param_names
        pivot_df = results_df.pivot(index=y_param, columns=x_param, values=metric_col)
    
    if verbose:
        print(f"Best result: {best_result.params}")
        print(f"Best {optim_metric.name}: {best_row[metric_col]:.4f}")
    
    return results_df, best_result, pivot_df


def create_alloc_backtest_func(
    alloc_func: Callable[[pd.DataFrame, Dict[str, Any]], pd.DataFrame],
    asset_list: List[str],
    transaction_cost_pct: float = 0.0,
) -> Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]]:
    """
    Create a wrapper function that combines allocation computation with backtesting.
    
    The allocation function should take (df, params) and return a DataFrame
    with allocation columns named 'A_{asset}_alloc' for each asset.
    
    This wrapper will:
    1. Call alloc_func to compute allocations
    2. Run backtest via run_full_backtest
    3. Compute metrics via compute_strategy_metrics
    
    Args:
        alloc_func: Function that takes (df, params) and returns df with 
                    A_*_alloc columns.
        asset_list: List of asset tickers (e.g., ['SPY', 'TLT', 'GLD']).
        transaction_cost_pct: Transaction cost as percentage (e.g., 0.1 for 0.1%).
    
    Returns:
        Function that takes (df, params) and returns metrics dict.
    """
    def wrapped_func(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, float]:
        df_with_alloc = alloc_func(df.copy(), params)
        
        for asset in asset_list:
            alloc_col = f"A_{asset}_alloc"
            if alloc_col not in df_with_alloc.columns:
                raise ValueError(f"Missing allocation column: {alloc_col}")
        
        result_df, orders_df = run_full_backtest(
            df_with_alloc, 
            asset_list, 
            transaction_cost_pct
        )
        
        metrics = compute_strategy_metrics(result_df, orders_df, asset_list)
        
        return metrics
    
    return wrapped_func

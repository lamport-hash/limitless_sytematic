"""
Parameter Optimizer for Dual Momentum Strategy.

Runs backtests across a range of lookback periods to find optimal parameters.
"""

import uuid
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ui_strat.backtest_runner import load_parquet_file, generate_roc_features


def run_optimization(
    filepath: str,
    selected_assets: List[str],
    default_asset: str,
    top_n: int = 1,
    abs_momentum_threshold: float = 0.0,
    transaction_cost_pct: float = 0.1,
    min_holding_periods: int = 0,
    switch_threshold_pct: float = 0.0,
    lookback_min: int = 100,
    lookback_max: int = 10000,
    num_steps: int = 50,
    run_id: Optional[str] = None,
) -> Dict:
    """
    Run optimization over lookback periods.
    
    Tests 50 evenly-spaced lookback values from lookback_min to lookback_max.
    Returns CAGR and Max DD for each lookback period.
    """
    if run_id is None:
        run_id = str(uuid.uuid4())[:8]
    
    df, all_assets = load_parquet_file(filepath)
    
    assets_to_use = [a for a in selected_assets if a in all_assets]
    if not assets_to_use:
        raise ValueError("No valid assets found in the selected file")
    
    if default_asset not in assets_to_use:
        raise ValueError(f"Default asset '{default_asset}' not in selected assets")
    
    lookback_values = np.linspace(lookback_min, lookback_max, num_steps, dtype=int)
    lookback_values = np.unique(lookback_values)
    
    results = []
    
    for lookback in lookback_values:
        try:
            result = _run_single_optim_step(
                df=df,
                assets_to_use=assets_to_use,
                default_asset=default_asset,
                lookback=int(lookback),
                top_n=top_n,
                abs_momentum_threshold=abs_momentum_threshold,
                transaction_cost_pct=transaction_cost_pct,
                min_holding_periods=min_holding_periods,
                switch_threshold_pct=switch_threshold_pct,
            )
            results.append(result)
        except Exception as e:
            results.append({
                'lookback': int(lookback),
                'cagr': None,
                'max_dd': None,
                'total_return': None,
                'error': str(e)
            })
    
    valid_results = [r for r in results if r.get('cagr') is not None]
    
    best_cagr = None
    best_sharpe = None
    
    if valid_results:
        best_cagr = max(valid_results, key=lambda x: x['cagr'] or -999)
        best_sharpe = max(valid_results, key=lambda x: (x['cagr'] or 0) / max(abs(x['max_dd'] or 1), 0.01))
    
    return {
        'run_id': run_id,
        'lookback_range': [int(lookback_min), int(lookback_max)],
        'num_steps': len(lookback_values),
        'results': results,
        'best_by_cagr': best_cagr,
        'best_by_sharpe': best_sharpe,
    }


def _run_single_optim_step(
    df,
    assets_to_use: List[str],
    default_asset: str,
    lookback: int,
    top_n: int,
    abs_momentum_threshold: float,
    transaction_cost_pct: float,
    min_holding_periods: int,
    switch_threshold_pct: float,
) -> Dict:
    """Run a single optimization step for one lookback value."""
    from strat.strat_backtest import compute_dual_momentum
    from backtest.backtest_basket_alloc_based import run_full_backtest
    from backtest.backtest_alloc_based import calculate_performance_metrics
    from features.feature_ta_utils import numba_roc_correct_min
    
    df_copy = df.copy()
    
    mid_col = "F_mid_f32"
    index_col = "i_minute_i"
    
    for asset in assets_to_use:
        mid_price_col = f"{asset}_{mid_col}"
        if mid_price_col not in df_copy.columns:
            continue
        
        col_name = f"{asset}_F_roctrue_{lookback}_{mid_col}_f16"
        if col_name not in df_copy.columns:
            df_copy[col_name] = numba_roc_correct_min(
                df_copy[mid_price_col].to_numpy(),
                df_copy[index_col].to_numpy(),
                lookback * 60
            )
    
    default_asset_idx = assets_to_use.index(default_asset)
    feature_id = f"F_roctrue_{lookback}_F_mid_f32_f16"
    
    df_allocations = compute_dual_momentum(
        p_df=df_copy,
        p_feature_id=feature_id,
        p_default_asset_idx=default_asset_idx,
        p_default_asset=default_asset,
        p_top_n=top_n,
        p_abs_momentum_threshold=abs_momentum_threshold,
        p_asset_list=assets_to_use,
        p_min_holding_periods=min_holding_periods,
        p_switch_threshold_pct=switch_threshold_pct,
    )
    
    p_df_result, orders_df = run_full_backtest(df_allocations, assets_to_use, transaction_cost_pct)
    
    min_idx = max(lookback, 100)
    p_df_result = p_df_result.iloc[min_idx:].copy()
    
    metrics = calculate_performance_metrics(p_df_result)
    
    return {
        'lookback': lookback,
        'cagr': round(metrics['cagr'], 2),
        'max_dd': round(metrics['max_drawdown'], 2),
        'total_return': round(metrics['total_return'], 2),
        'sharpe': round(metrics.get('sharpe_ratio', 0), 2),
        'win_rate': round(metrics.get('win_rate', 0), 1),
    }

"""
Dual Momentum Strategy Backtest Runner.

Extracted from notebooks/bkt_dual_mom_clean.ipynb
"""

import os
import uuid
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strat.strat_backtest import compute_dual_momentum
from strat.strat_cto_line import compute_cto_line_allocations, compute_cto_line_raw_allocations
from strat.strat_filters import AllocationFilter, AllocationFilterParams
from backtest.backtest_basket_alloc_based import run_full_backtest
from features.feature_ta_utils import numba_roc_correct_min, numba_rsi

from ui_strat.metrics import (
    calculate_performance_metrics,
    calculate_asset_metrics,
    minutes_to_datetime,
    MONTH_NAMES,
)
from ui_strat.charts import generate_charts
from ui_strat.trades import (
    compute_trades_from_orders,
    compute_current_positions,
    compute_trades_and_positions,
)

warnings.filterwarnings('ignore')


def extract_assets_from_df(df: pd.DataFrame) -> List[str]:
    """Extract unique asset symbols from DataFrame columns."""
    assets = set()
    for col in df.columns:
        if '_' in col and not col.startswith('F_') and not col.startswith('A_') and not col.startswith('i_'):
            parts = col.split('_')
            if len(parts) > 1:
                asset = parts[0]
                if asset and not asset.startswith('S_'):
                    assets.add(asset)
    return sorted(assets)


def load_parquet_file(filepath: str) -> Tuple[pd.DataFrame, List[str]]:
    """Load parquet file and extract assets."""
    df = pd.read_parquet(filepath)
    assets = extract_assets_from_df(df)
    return df, assets


def generate_roc_features(df: pd.DataFrame, assets: List[str], lookback: int) -> pd.DataFrame:
    """Generate ROC features for the specified lookback period."""
    df = df.copy()
    
    mid_col = "F_mid_f32"
    index_col = "i_minute_i"
    
    for asset in assets:
        mid_price_col = f"{asset}_{mid_col}"
        if mid_price_col not in df.columns:
            continue
            
        col_name = f"{asset}_F_roctrue_{lookback}_{mid_col}_f16"
        if col_name not in df.columns:
            df[col_name] = numba_roc_correct_min(
                df[mid_price_col].to_numpy(),
                df[index_col].to_numpy(),
                lookback * 60
            )
    
    return df


def generate_rsi_features(df: pd.DataFrame, assets: List[str], rsi_period: int) -> pd.DataFrame:
    """Generate RSI features for the specified period using close prices."""
    df = df.copy()
    
    close_col = "S_close_f32"
    
    for asset in assets:
        price_col = f"{asset}_{close_col}"
        if price_col not in df.columns:
            continue
            
        rsi_col_name = f"{asset}_F_rsi_{rsi_period}_{close_col}_f16"
        if rsi_col_name not in df.columns:
            prices = df[price_col].to_numpy()
            rsi_values = numba_rsi(prices, rsi_period)
            df[rsi_col_name] = rsi_values.astype(np.float32)
    
    return df


def run_backtest(
    filepath: str,
    selected_assets: List[str],
    lookback: int,
    default_asset: str,
    top_n: int,
    abs_momentum_threshold: float,
    transaction_cost_pct: float = 0.1,
    min_holding_periods: int = 0,
    switch_threshold_pct: float = 0.0,
    rsi_period: int = 14,
    use_rsi_entry_filter: bool = False,
    rsi_entry_max: float = 30.0,
    use_rsi_entry_queue: bool = False,
    use_rsi_diff_filter: bool = False,
    rsi_diff_threshold: float = 10.0,
    run_id: Optional[str] = None,
) -> Dict:
    """Run the complete dual momentum backtest."""
    if run_id is None:
        run_id = str(uuid.uuid4())[:8]
    
    df, all_assets = load_parquet_file(filepath)
    
    assets_to_use = [a for a in selected_assets if a in all_assets]
    if not assets_to_use:
        raise ValueError("No valid assets found in the selected file")
    
    if default_asset not in assets_to_use:
        raise ValueError(f"Default asset '{default_asset}' not in selected assets")
    
    default_asset_idx = assets_to_use.index(default_asset)
    
    df_with_roc = generate_roc_features(df, assets_to_use, lookback)
    
    if use_rsi_entry_filter or use_rsi_diff_filter:
        df_with_roc = generate_rsi_features(df_with_roc, assets_to_use, rsi_period)
    
    feature_id = f"F_roctrue_{lookback}_F_mid_f32_f16"
    rsi_feature_id = f"F_rsi_{rsi_period}_S_close_f32_f16"
    
    df_allocations = compute_dual_momentum(
        p_df=df_with_roc,
        p_feature_id=feature_id,
        p_default_asset_idx=default_asset_idx,
        p_default_asset=default_asset,
        p_top_n=top_n,
        p_abs_momentum_threshold=abs_momentum_threshold,
        p_asset_list=assets_to_use,
        p_min_holding_periods=min_holding_periods,
        p_switch_threshold_pct=switch_threshold_pct,
        p_use_rsi_entry_filter=use_rsi_entry_filter,
        p_rsi_entry_max=rsi_entry_max,
        p_use_rsi_entry_queue=use_rsi_entry_queue,
        p_use_rsi_diff_filter=use_rsi_diff_filter,
        p_rsi_diff_threshold=rsi_diff_threshold,
        p_rsi_feature_id=rsi_feature_id
    )
    
    p_df_result, orders_df = run_full_backtest(df_allocations, assets_to_use, transaction_cost_pct)
    
    min_idx = max(lookback, 100)
    p_df_result = p_df_result.iloc[min_idx:].copy()
    
    orders_df = orders_df[orders_df['timestamp'] >= min_idx].copy()
    orders_df['timestamp'] = orders_df['timestamp'] - min_idx
    orders_df = orders_df.reset_index(drop=True)
    
    entries_count = int((orders_df['direction'] > 0).sum()) if len(orders_df) > 0 else 0
    exits_count = int((orders_df['direction'] < 0).sum()) if len(orders_df) > 0 else 0
    
    entries_safe = int(((orders_df['direction'] > 0) & (orders_df['etf'] == default_asset)).sum()) if len(orders_df) > 0 else 0
    entries_risky = int(((orders_df['direction'] > 0) & (orders_df['etf'] != default_asset)).sum()) if len(orders_df) > 0 else 0
    exits_safe = int(((orders_df['direction'] < 0) & (orders_df['etf'] == default_asset)).sum()) if len(orders_df) > 0 else 0
    exits_risky = int(((orders_df['direction'] < 0) & (orders_df['etf'] != default_asset)).sum()) if len(orders_df) > 0 else 0
    
    metrics = calculate_performance_metrics(p_df_result)
    
    asset_metrics = calculate_asset_metrics(p_df_result, assets_to_use, orders_df)
    
    chart_files = generate_charts(metrics, run_id, assets_to_use)
    
    monthly_returns_dict = {}
    for year, row in metrics['monthly_returns'].iterrows():
        monthly_returns_dict[int(year)] = {k: (v if not pd.isna(v) else None) for k, v in row.to_dict().items()}
    
    return {
        'run_id': run_id,
        'metrics': {
            'start_date': metrics['start_date'].strftime('%Y-%m-%d %H:%M'),
            'end_date': metrics['end_date'].strftime('%Y-%m-%d %H:%M'),
            'years': metrics['years'],
            'start_value': metrics['start_value'],
            'end_value': metrics['end_value'],
            'total_return': metrics['total_return'],
            'cagr': metrics['cagr'],
            'max_drawdown': metrics['max_drawdown'],
            'max_drawdown_date': metrics['max_drawdown_date'].strftime('%Y-%m-%d'),
            'max_drawdown_peak_date': metrics['max_drawdown_peak_date'].strftime('%Y-%m-%d'),
            'annual_volatility': metrics['annual_volatility'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'calmar_ratio': metrics['calmar_ratio'],
            'win_rate': metrics['win_rate'],
            'monthly_returns': monthly_returns_dict,
        },
        'asset_metrics': asset_metrics,
        'charts': chart_files,
        'orders_count': len(orders_df),
        'entries_count': entries_count,
        'exits_count': exits_count,
        'entries_safe': entries_safe,
        'entries_risky': entries_risky,
        'exits_safe': exits_safe,
        'exits_risky': exits_risky,
        'orders_df': orders_df,
        'p_df': p_df_result,
    }


def run_backtest_cto_line(
    filepath: str,
    selected_assets: List[str],
    cto_params: Tuple[int, int, int, int] = (15, 19, 25, 29),
    direction: str = "both",
    default_asset: str,
    transaction_cost_pct: float = 0.1,
    min_holding_periods: int = 0,
    switch_threshold_pct: float = 0.0,
    use_rsi_entry_filter: bool = False,
    rsi_entry_max: float = 30.0,
    use_rsi_entry_queue: bool = False,
    use_rsi_diff_filter: bool = False,
    rsi_diff_threshold: float = 10.0,
    rsi_period: int = 14,
    run_id: Optional[str] = None,
) -> Dict:
    """Run the CTO Line basket allocation backtest with filters."""
    if run_id is None:
        run_id = str(uuid.uuid4())[:8]
    
    df, all_assets = load_parquet_file(filepath)
    
    assets_to_use = [a for a in selected_assets if a in all_assets]
    if not assets_to_use:
        raise ValueError("No valid assets found in the selected file")
    
    if default_asset is None:
        raise ValueError("default_asset is required for CTO Line strategy")
    
    if default_asset not in assets_to_use:
        raise ValueError(f"default_asset '{default_asset}' not in asset list")
    
    default_asset_idx = assets_to_use.index(default_asset)
    
    rsi_matrix = None
    if use_rsi_entry_filter or use_rsi_diff_filter:
        df = generate_rsi_features(df, assets_to_use, rsi_period)
        rsi_feature_id = f"F_rsi_{rsi_period}_S_close_f32_f16"
        rsi_cols = {}
        for asset in assets_to_use:
            col_name = f"{asset}_{rsi_feature_id}"
            if col_name in df.columns:
                rsi_cols[asset] = df[col_name].to_numpy()
        else:
            prices = df[f"{asset}_S_close_f32"].to_numpy()
            rsi_cols[asset] = numba_rsi(prices, rsi_period)
        
        n_assets = len(assets_to_use)
        rsi_matrix = np.zeros((len(df), n_assets))
        for j, range(n_assets):
            asset = assets_to_use[j]
            rsi_matrix[:, j] = rsi_cols[asset]
    
    df_allocations = compute_cto_line_allocations(
        p_df=df,
        p_asset_list=assets_to_use,
        p_cto_params=cto_params,
        p_direction=direction,
        p_min_holding_periods=min_holding_periods,
        p_switch_threshold_pct=switch_threshold_pct,
        p_default_asset=default_asset,
        p_default_asset_idx=default_asset_idx,
        p_use_rsi_entry_filter=use_rsi_entry_filter,
        p_rsi_entry_max=rsi_entry_max,
        p_use_rsi_entry_queue=use_rsi_entry_queue,
        p_use_rsi_diff_filter=use_rsi_diff_filter,
        p_rsi_diff_threshold=rsi_diff_threshold,
        p_rsi_values=rsi_matrix if use_rsi_entry_filter or use_rsi_diff_filter else None,
    )
    
    p_df_result, orders_df = run_full_backtest(df_allocations, assets_to_use, transaction_cost_pct)
    
    warmup_bars = max(cto_params)
    min_idx = max(warmup_bars, 100)
    p_df_result = p_df_result.iloc[min_idx:].copy()
    
    orders_df = orders_df[orders_df['timestamp'] >= min_idx].copy()
    orders_df['timestamp'] = orders_df['timestamp'] - min_idx
    orders_df = orders_df.reset_index(drop=True)
    
    entries_count = int((orders_df['direction'] > 0).sum()) if len(orders_df) > 0 else 0
    exits_count = int((orders_df['direction'] < 0).sum()) if len(orders_df) > 0 else 0
    
    safe_asset = default_asset if default_asset else assets_to_use[0]
    entries_safe = int(((orders_df['direction'] > 0) & (orders_df['etf'] == safe_asset)).sum()) if len(orders_df) > 0 else 0
    entries_risky = int(((orders_df['direction'] > 0) & (orders_df['etf'] != safe_asset)).sum()) if len(orders_df) > 0 else 0
    exits_safe = int(((orders_df['direction'] < 0) & (orders_df['etf'] == safe_asset)).sum()) if len(orders_df) > 0 else 0
    exits_risky = int(((orders_df['direction'] < 0) & (orders_df['etf'] != safe_asset)).sum()) if len(orders_df) > 0 else 0
    
    metrics = calculate_performance_metrics(p_df_result)
    asset_metrics = calculate_asset_metrics(p_df_result, assets_to_use, orders_df)
    chart_files = generate_charts(metrics, run_id, assets_to_use)
    
    monthly_returns_dict = {}
    for year, row in metrics['monthly_returns'].iterrows():
        monthly_returns_dict[int(year)] = {k: (v if not pd.isna(v) else None) for k, v in row.to_dict().items()}
    
    return {
        'run_id': run_id,
        'metrics': {
            'start_date': metrics['start_date'].strftime('%Y-%m-%d %H:%M'),
            'end_date': metrics['end_date'].strftime('%Y-%m-%d %H:%M'),
            'years': metrics['years'],
            'start_value': metrics['start_value'],
            'end_value': metrics['end_value'],
            'total_return': metrics['total_return'],
            'cagr': metrics['cagr'],
            'max_drawdown': metrics['max_drawdown'],
            'max_drawdown_date': metrics['max_drawdown_date'].strftime('%Y-%m-%d'),
            'max_drawdown_peak_date': metrics['max_drawdown_peak_date'].strftime('%Y-%m-%d'),
            'annual_volatility': metrics['annual_volatility'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'calmar_ratio': metrics['calmar_ratio'],
            'win_rate': metrics['win_rate'],
            'monthly_returns': monthly_returns_dict,
        },
        'asset_metrics': asset_metrics,
        'charts': chart_files,
        'orders_count': len(orders_df),
        'entries_count': entries_count,
        'exits_count': exits_count,
        'entries_safe': entries_safe,
        'entries_risky': entries_risky,
        'exits_safe': exits_safe,
        'exits_risky': exits_risky,
        'orders_df': orders_df,
        'p_df': p_df_result,
    }

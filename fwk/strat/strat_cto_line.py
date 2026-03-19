"""
CTO Line Basket Allocation Strategy.

Implements allocation strategy based on CTO (Colored Trend Oscillator) Line signals
for a basket of assets. Each asset gets individual long/short signals.

Signal Logic:
- Long signal (1) -> asset eligible for allocation
- Short signal (1) -> asset eligible for allocation
- Both mode: long and short baskets are separate, each gets 50% of portfolio
- Multiple assets can have allocations simultaneously

Filter Constraints:
Use strat_filters.AllocationFilter to apply:
- direction filtering (long/short/both)
- default_asset (when no signals)
- min_holding_periods (hysteresis)
- RSI filters (optional)
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
from numba import njit

from strat.strat_base import BaseStrategy


@njit
def smma_numba(p_src: np.ndarray, p_length: int) -> np.ndarray:
    """Calculate Smoothed Moving Average."""
    smma = np.zeros(len(p_src))
    for i in range(len(p_src)):
        if i == 0:
            smma[i] = p_src[0]
        else:
            smma[i] = (smma[i - 1] * (p_length - 1) + p_src[i]) / p_length
    return smma


def compute_cto_signals(
    p_high: np.ndarray,
    p_low: np.ndarray,
    p_close: np.ndarray,
    p_params: Tuple[int, int, int, int] = (15, 19, 25, 29),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute CTO Line signals for a single asset.
    
    Args:
        p_high: High prices array
        p_low: Low prices array
        p_close: Close prices array (not used but kept for API consistency)
        p_params: SMMA periods (v1, m1, m2, v2)
        
    Returns:
        Tuple of (long_signal, short_signal) arrays, values 0 or 1
    """
    hl2 = (p_high + p_low) / 2.0
    
    v1 = smma_numba(hl2, p_params[0])
    m1 = smma_numba(hl2, p_params[1])
    m2 = smma_numba(hl2, p_params[2])
    v2 = smma_numba(hl2, p_params[3])
    
    p2 = ((v1 < m1) != (v1 < v2)) | ((m2 < v2) != (v1 < v2))
    p3 = ~p2 & (v1 < v2)
    p1 = ~p2 & ~p3
    
    long_signal = p1.astype(np.int8)
    short_signal = p3.astype(np.int8)
    
    return long_signal, short_signal


def compute_cto_line_raw_allocations(
    p_df: pd.DataFrame,
    p_asset_list: List[str],
    p_cto_params: Tuple[int, int, int, int] = (15, 19, 25, 29),
    p_direction: str = "both",
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute RAW CTO Line allocations (no constraints applied).
    
    Each asset gets raw allocation of 1.0 when signal is active, 0.0 otherwise.
    Multiple assets can have allocations simultaneously.
    
    This function computes signals only - use strat_filters.AllocationFilter
    to apply constraints like default_asset, min_holding_periods, RSI filters, etc.
    
    Args:
        p_df: DataFrame with asset data (must have OHLC columns)
        p_asset_list: List of asset symbols
        p_cto_params: SMMA periods (v1, m1, m2, v2)
        p_direction: "long", "short", or "both"
        
    Returns:
        Tuple of (df with raw allocation columns, long_signals matrix, short_signals matrix, signal_types matrix)
        - DataFrame has A_{asset}_raw_alloc and A_{asset}_signal_type columns
        - long_signals: (n_periods, n_assets) matrix
        - short_signals: (n_periods, n_assets) matrix
        - signal_types: (n_periods, n_assets) matrix (1=long, -1=short, 0=none)
    """
    df = p_df.copy()
    n_periods = len(df)
    n_assets = len(p_asset_list)
    
    high_col = "S_high_f32"
    low_col = "S_low_f32"
    close_col = "S_close_f32"
    
    raw_allocs = np.zeros((n_periods, n_assets))
    long_signals = np.zeros((n_periods, n_assets), dtype=np.int8)
    short_signals = np.zeros((n_periods, n_assets), dtype=np.int8)
    signal_types = np.zeros((n_periods, n_assets), dtype=np.int8)
    
    for i, asset in enumerate(p_asset_list):
        asset_high = f"{asset}_{high_col}"
        asset_low = f"{asset}_{low_col}"
        asset_close = f"{asset}_{close_col}"
        
        if asset_high not in df.columns or asset_low not in df.columns or asset_close not in df.columns:
            continue
        
        long_sig, short_sig = compute_cto_signals(
            df[asset_high].to_numpy(),
            df[asset_low].to_numpy(),
            df[asset_close].to_numpy(),
            p_cto_params
        )
        
        long_signals[:, i] = long_sig
        short_signals[:, i] = short_sig
        
        if p_direction == "long":
            raw_allocs[:, i] = long_sig.astype(np.float64)
            signal_types[:, i] = long_sig.astype(np.int8)
        elif p_direction == "short":
            raw_allocs[:, i] = -short_sig.astype(np.float64)
            signal_types[:, i] = -short_sig.astype(np.int8)
        else:  # both - combine long and short
            raw_allocs[:, i] = (long_sig.astype(np.float64) + short_sig.astype(np.float64))
            signal_types[:, i] = long_sig.astype(np.int8) - short_sig.astype(np.int8)
    
    for i, asset in enumerate(p_asset_list):
        df[f"A_{asset}_raw_alloc"] = raw_allocs[:, i]
        df[f"A_{asset}_signal_type"] = signal_types[:, i]
    
    df["A_n_assets_with_signal"] = (raw_allocs > 0).sum(axis=1)
    
    return df, long_signals, short_signals, signal_types


def compute_cto_line_allocations(
    p_df: pd.DataFrame,
    p_asset_list: List[str],
    p_cto_params: Tuple[int, int, int, int] = (15, 19, 25, 29),
    p_direction: str = "both",
    p_min_holding_periods: int = 0,
    p_switch_threshold_pct: float = 0.0,
    p_default_asset: Optional[str] = None,
    p_use_rsi_entry_filter: bool = False,
    p_rsi_entry_max: float = 30.0,
    p_use_rsi_entry_queue: bool = False,
    p_use_rsi_diff_filter: bool = False,
    p_rsi_diff_threshold: float = 10.0,
    p_rsi_values: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Compute CTO Line allocations with all constraints applied.
    
    This is a convenience function that combines raw allocation computation
    with filter application. For more control, use compute_cto_line_raw_allocations
    and strat_filters.AllocationFilter directly.
    
    Args:
        p_df: DataFrame with asset data (must have OHLC columns)
        p_asset_list: List of asset symbols
        p_cto_params: SMMA periods (v1, m1, m2, v2)
        p_direction: "long", "short", or "both"
        p_min_holding_periods: Min bars to hold before allowing switch (0 = disabled)
        p_switch_threshold_pct: Not used in basket mode
        p_default_asset: Asset to use when no signals (REQUIRED - raises error if None)
        p_use_rsi_entry_filter: Enable RSI entry filter
        p_rsi_entry_max: Max RSI to allow entry
        p_use_rsi_entry_queue: Enable pending queue for RSI entry
        p_use_rsi_diff_filter: Enable RSI difference filter
        p_rsi_diff_threshold: Min RSI diff to switch
        p_rsi_values: Optional RSI matrix (n_periods, n_assets)
        
    Returns:
        DataFrame with A_{asset}_alloc columns added (filtered and normalized)
    """
    from strat.strat_filters import (
        AllocationFilter, 
        AllocationFilterParams,
        DIRECTION_LONG,
        DIRECTION_SHORT,
        DIRECTION_BOTH,
    )
    
    if p_default_asset is None:
        raise ValueError("default_asset is required for CTO Line strategy")
    
    if p_default_asset not in p_asset_list:
        raise ValueError(f"default_asset '{p_default_asset}' not in asset list")
    
    default_asset_idx = p_asset_list.index(p_default_asset)
    
    df, long_signals, short_signals, signal_types = compute_cto_line_raw_allocations(
        p_df, p_asset_list, p_cto_params, p_direction
    )
    
    filter_params = AllocationFilterParams(
        p_direction=p_direction,
        p_default_asset=p_default_asset,
        p_default_asset_idx=default_asset_idx,
        p_min_holding_periods=p_min_holding_periods,
        p_switch_threshold_pct=p_switch_threshold_pct,
        p_use_rsi_entry_filter=p_use_rsi_entry_filter,
        p_rsi_entry_max=p_rsi_entry_max,
        p_use_rsi_entry_queue=p_use_rsi_entry_queue,
        p_use_rsi_diff_filter=p_use_rsi_diff_filter,
        p_rsi_diff_threshold=p_rsi_diff_threshold,
    )
    
    filter = AllocationFilter(filter_params)
    
    raw_alloc_cols = [f"A_{asset}_raw_alloc" for asset in p_asset_list]
    raw_allocs = df[raw_alloc_cols].to_numpy()
    
    filtered_allocs = filter.apply(
        raw_allocs, long_signals, short_signals, p_rsi_values
    )
    
    for i, asset in enumerate(p_asset_list):
        df[f"A_{asset}_alloc"] = filtered_allocs[:, i]
        df[f"A_{asset}_signal_type"] = signal_types[:, i]
    
    df["A_n_assets_with_signal"] = (filtered_allocs > 0).sum(axis=1)
    
    return df


class CtoLineStrategy(BaseStrategy):
    """CTO Line basket allocation strategy implementation."""
    
    @property
    def strategy_name(self) -> str:
        return "cto_line"
    
    @property
    def strategy_display_name(self) -> str:
        return "CTO Line Strategy"
    
    def compute_allocations(
        self,
        p_df: pd.DataFrame,
        p_asset_list: List[str],
        p_cto_params: Tuple[int, int, int, int] = (15, 19, 25, 29),
        p_direction: str = "both",
        p_min_holding_periods: int = 0,
        p_switch_threshold_pct: float = 0.0,
        p_default_asset: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        return compute_cto_line_allocations(
            p_df=p_df,
            p_asset_list=p_asset_list,
            p_cto_params=p_cto_params,
            p_direction=p_direction,
            p_min_holding_periods=p_min_holding_periods,
            p_switch_threshold_pct=p_switch_threshold_pct,
            p_default_asset=p_default_asset,
        )
    
    def get_required_features(
        self,
        p_cto_params: Tuple[int, int, int, int] = (15, 19, 25, 29),
        **kwargs
    ) -> List[str]:
        return [
            "S_high_f32",
            "S_low_f32",
            "S_close_f32",
        ]
    
    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "cto_params": {
                "type": "tuple",
                "default": "(15, 19, 25, 29)",
                "description": "SMMA periods (v1, m1, m2, v2)"
            },
            "direction": {
                "type": "str",
                "default": "both",
                "description": "Trading direction: 'long', 'short', or 'both'"
            },
            "min_holding_periods": {
                "type": "int",
                "default": 0,
                "description": "Minimum bars to hold before allowing switch (0 = disabled)"
            },
            "switch_threshold_pct": {
                "type": "float",
                "default": 0.0,
                "description": "Not used in basket mode"
            },
            "default_asset": {
                "type": "str",
                "default": None,
                "description": "Asset to hold when no signals (required)"
            },
        }

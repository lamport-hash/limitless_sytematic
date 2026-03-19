"""
CTO Line Basket Allocation Strategy.

Implements allocation strategy based on CTO (Colored Trend Oscillator) Line signals
for a basket of assets. Each asset gets individual long/short signals.

Signal Logic:
- Long signal (1) -> asset eligible for allocation
- Short/neutral (0) -> no allocation
- Multiple assets can have allocations simultaneously
- Allocations normalized to sum to 1.0

Hysteresis:
- p_min_holding_periods: Min bars before switching
- p_switch_threshold_pct: Relative % new ROC must exceed current to switch
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
from numba import njit

from strat.strat_base import BaseStrategy, normalize_allocations, validate_allocations


@njit
def smma_numba(p_src: np.ndarray, p_length: int) -> np.ndarray:
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
        p_close: Close prices array
        p_params: SMMA periods (v1, m1, m2, v2)
        
    Returns:
        Tuple of (long_signal, short_signal) arrays
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


def compute_cto_line_allocations(
    p_df: pd.DataFrame,
    p_asset_list: List[str],
    p_cto_params: Tuple[int, int, int, int] = (15, 19, 25, 29),
    p_direction: str = "both",
    p_min_holding_periods: int = 0,
    p_switch_threshold_pct: float = 0.0,
    p_default_asset: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute CTO Line allocations for multi-asset basket.
    
    Basket Mode Logic:
    - Each asset gets individual long/short signal
    - Long signal (1) -> asset eligible for allocation
    - Short/neutral (0) -> no allocation
    - Multiple assets can have allocations simultaneously
    - Allocations normalized so they sum to 1.0 (or 0 if no signals)
    
    Args:
        p_df: DataFrame with asset data (must have OHLC columns)
        p_asset_list: List of asset symbols
        p_cto_params: SMMA periods (v1, m1, m2, v2)
        p_direction: "long", "short", or "both"
        p_min_holding_periods: Min bars to hold before allowing switch (0 = disabled)
        p_switch_threshold_pct: Not used in basket mode with individual signals
        p_default_asset: Asset to use when no signals (optional)
        
    Returns:
        DataFrame with A_{asset}_alloc columns added
    """
    df = p_df.copy()
    n_periods = len(df)
    n_assets = len(p_asset_list)
    
    high_col = "S_high_f32"
    low_col = "S_low_f32"
    close_col = "S_close_f32"
    
    signals = np.zeros((n_periods, n_assets))
    
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
        
        if p_direction == "long":
            signals[:, i] = long_sig
        elif p_direction == "short":
            signals[:, i] = short_sig
        else:
            if p_direction == "both":
                signals[:, i] = long_sig
    
    allocations = np.zeros((n_periods, n_assets))
    current_allocs = np.zeros(n_assets)
    periods_since_change = 0
    
    for t in range(n_periods):
        current_signals = signals[t]
        
        valid_signal_mask = current_signals > 0
        
        if not valid_signal_mask.any():
            if p_default_asset is not None:
                default_idx = p_asset_list.index(p_default_asset) if p_default_asset in p_asset_list else None
                if default_idx is not None:
                    new_allocs = np.zeros(n_assets)
                    new_allocs[default_idx] = 1.0
                else:
                    new_allocs = current_allocs.copy()
            else:
                new_allocs = np.zeros(n_assets)
        else:
            new_allocs = np.where(valid_signal_mask, 1.0, 0.0)
            n_signals = valid_signal_mask.sum()
            if n_signals > 0:
                new_allocs = new_allocs / n_signals
        
        should_change = True
        if p_min_holding_periods > 0 and periods_since_change < p_min_holding_periods:
            if not np.array_equal(new_allocs > 0, current_allocs > 0):
                should_change = False
            else:
                should_change = True
        
        if should_change:
            if not np.array_equal(new_allocs, current_allocs):
                periods_since_change = 0
            else:
                periods_since_change += 1
            allocations[t] = new_allocs
            current_allocs = new_allocs.copy()
        else:
            allocations[t] = current_allocs.copy()
            periods_since_change += 1
    
    for i, asset in enumerate(p_asset_list):
        df[f"A_{asset}_alloc"] = allocations[:, i]
    
    df["A_n_assets_with_signal"] = (allocations > 0).sum(axis=1)
    
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
                "description": "Asset to hold when no signals (optional)"
            },
        }

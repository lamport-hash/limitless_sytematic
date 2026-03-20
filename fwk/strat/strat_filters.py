"""
Strat Filters - Reusable allocation constraint utilities.

This module provides composable filter functions that can be applied
to raw strategy allocations. Each filter is a pure function that can be
tested independently and combined in different ways.

Filters:
- apply_direction_filter: Filter to long/short/both modes
- apply_default_asset_filter: Fill gaps with default asset
- apply_min_holding_filter: Hysteresis to prevent rapid switching
- apply_rsi_entry_filter: Block entries when RSI too high
- apply_rsi_diff_filter: Block switches when RSI diff too small
- apply_switch_threshold_filter: Block switches when ROC improvement too small

Usage:
    raw_allocs = strategy.compute_raw_allocs(no constraints)
    filter = AllocationFilter(p_params=...)
    filtered_allocs = filter.apply(raw_allocs, context)
"""

import numpy as np
from numba import njit
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass


DIRECTION_LONG = 1
DIRECTION_SHORT = -1
DIRECTION_BOTH = 0


@njit
def apply_direction_filter_numba(
    raw_allocs: np.ndarray,
    long_signals: np.ndarray,
    short_signals: np.ndarray,
    direction: int,
) -> np.ndarray:
    """
    Filter raw allocations based on trading direction.
    
    Args:
        raw_allocs: Raw allocation matrix (n_periods, n_assets)
        long_signals: Long signal matrix (n_periods, n_assets), values 0 or 1
        short_signals: Short signal matrix (n_periods, n_assets), values 0 or 1
        direction: DIRECTION_LONG (1), DIRECTION_SHORT (-1), or DIRECTION_BOTH (0)
    
    Returns:
        Filtered allocation matrix
    """
    n_periods, n_assets = raw_allocs.shape
    filtered = np.zeros((n_periods, n_assets))
    
    for i in range(n_periods):
        for j in range(n_assets):
            if direction == DIRECTION_LONG:
                filtered[i, j] = raw_allocs[i, j] * long_signals[i, j]
            elif direction == DIRECTION_SHORT:
                filtered[i, j] = raw_allocs[i, j] * short_signals[i, j]
            else:  # DIRECTION_BOTH
                filtered[i, j] = raw_allocs[i, j] * (long_signals[i, j] + short_signals[i, j])
    
    return filtered


@njit
def apply_default_asset_filter_numba(
    allocs: np.ndarray,
    default_asset_idx: int,
) -> np.ndarray:
    """
    Fill allocation gaps with default asset when no signals present.
    
    Args:
        allocs: Current allocation matrix (n_periods, n_assets), modified in-place
        default_asset_idx: Index of default asset in asset list
    
    Returns:
        Modified allocation matrix with default asset fills
    """
    n_periods, n_assets = allocs.shape
    result = allocs.copy()
    
    for i in range(n_periods):
        row_sum = 0.0
        for j in range(n_assets):
            row_sum += abs(result[i, j])
        
        if row_sum < 1e-9:
            for j in range(n_assets):
                result[i, j] = 0.0
            if default_asset_idx >= 0 and default_asset_idx < n_assets:
                result[i, default_asset_idx] = 1.0
    
    return result


@njit
def apply_min_holding_filter_numba(
    target_allocs: np.ndarray,
    initial_allocs: np.ndarray,
    min_holding_periods: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply minimum holding period filter (hysteresis).
    
    Prevents allocation changes until min_holding_periods have elapsed.
    
    Args:
        target_allocs: Target allocation matrix from strategy
        initial_allocs: Starting allocation (usually zeros)
        min_holding_periods: Minimum bars to hold before allowing switch
    
    Returns:
        Tuple of (filtered_allocs, periods_held_counter)
    """
    n_periods, n_assets = target_allocs.shape
    filtered = np.zeros((n_periods, n_assets))
    current_allocs = initial_allocs.copy()
    periods_held = np.zeros(n_assets, dtype=np.int64)
    
    for i in range(n_periods):
        target_row = target_allocs[i]
        
        for j in range(n_assets):
            if min_holding_periods > 0 and periods_held[j] < min_holding_periods:
                if abs(target_row[j] - current_allocs[j]) > 1e-9:
                    filtered[i, j] = current_allocs[j]
                    periods_held[j] += 1
                else:
                    filtered[i, j] = target_row[j]
                    if abs(target_row[j] - current_allocs[j]) > 1e-9:
                        periods_held[j] = 0
                    else:
                        periods_held[j] += 1
            else:
                filtered[i, j] = target_row[j]
                if abs(target_row[j] - current_allocs[j]) > 1e-9:
                    periods_held[j] = 0
                else:
                    periods_held[j] += 1
        
        current_allocs = filtered[i].copy()
    
    return filtered, periods_held


@njit
def normalize_allocations_numba(allocs: np.ndarray) -> np.ndarray:
    """
    Normalize allocations so each row sums to 1.0.
    
    Args:
        allocs: Allocation matrix (n_periods, n_assets)
    
    Returns:
        Normalized allocation matrix
    """
    n_periods, n_assets = allocs.shape
    normalized = np.zeros((n_periods, n_assets))
    
    for i in range(n_periods):
        row_sum = 0.0
        for j in range(n_assets):
            row_sum += abs(allocs[i, j])
        
        if row_sum > 1e-9:
            for j in range(n_assets):
                normalized[i, j] = allocs[i, j] / row_sum
    
    return normalized


@njit
def apply_rsi_entry_filter_numba(
    allocs: np.ndarray,
    rsi_values: np.ndarray,
    rsi_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply RSI entry filter - block entries when RSI too high.
    
    Args:
        allocs: Current allocation matrix
        rsi_values: RSI matrix (n_periods, n_assets)
        rsi_max: Maximum RSI to allow entry
    
    Returns:
        Tuple of (filtered_allocs, blocked_mask)
    """
    n_periods, n_assets = allocs.shape
    filtered = allocs.copy()
    blocked = np.zeros((n_periods, n_assets), dtype=np.bool_)
    
    for i in range(n_periods):
        for j in range(n_assets):
            if allocs[i, j] > 1e-9:
                rsi = rsi_values[i, j]
                if not np.isnan(rsi) and rsi > rsi_max:
                    filtered[i, j] = 0.0
                    blocked[i, j] = True
    
    return filtered, blocked


@njit
def apply_switch_threshold_filter_numba(
    should_switch: np.ndarray,
    current_metric: np.ndarray,
    new_metric: np.ndarray,
    threshold_pct: float,
) -> np.ndarray:
    """
    Apply switch threshold filter - only switch if new metric exceeds threshold.
    
    Args:
        should_switch: Initial switch decisions (n_periods,)
        current_metric: Current asset's metric values (n_periods,)
        new_metric: New asset's metric values (n_periods,)
        threshold_pct: Minimum % improvement required (e.g., 0.1 = 10%)
    
    Returns:
        Modified switch decisions
    """
    n_periods = len(should_switch)
    result = should_switch.copy()
    
    if threshold_pct <= 0:
        return result
    
    for i in range(n_periods):
        if should_switch[i]:
            if not np.isnan(current_metric[i]) and not np.isnan(new_metric[i]):
                threshold_value = current_metric[i] * (1 + threshold_pct)
                if new_metric[i] < threshold_value:
                    result[i] = False
    
    return result


@njit
def apply_rsi_diff_filter_numba(
    should_switch: np.ndarray,
    current_rsi: np.ndarray,
    new_rsi: np.ndarray,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply RSI difference filter - only switch if RSI diff >= threshold.
    
    Args:
        should_switch: Initial switch decisions (n_periods,)
        current_rsi: Current asset's RSI values (n_periods,)
        new_rsi: New asset's RSI values (n_periods,)
        threshold: Minimum RSI diff required (current - new)
    
    Returns:
        Tuple of (modified_switch_decisions, blocked_mask)
    """
    n_periods = len(should_switch)
    result = should_switch.copy()
    blocked = np.zeros(n_periods, dtype=np.bool_)
    
    if threshold <= 0:
        return result, blocked
    
    for i in range(n_periods):
        if should_switch[i]:
            if not np.isnan(current_rsi[i]) and not np.isnan(new_rsi[i]):
                rsi_diff = current_rsi[i] - new_rsi[i]
                if rsi_diff < threshold:
                    result[i] = False
                    blocked[i] = True
    
    return result, blocked


@njit
def apply_half_assets_cap_numba(
    allocs: np.ndarray,
    signal_types: np.ndarray,
    metric_values: np.ndarray,
) -> np.ndarray:
    """
    Cap allocations to at most half the assets, ranked by metric strength.
    
    For each period:
    - Count assets with active signals
    - If more than half, keep only top N=floor(n_assets/2) strongest signals
    - Set allocation to 1/(n_assets/2) for each selected asset
    
    Args:
        allocs: Allocation matrix (n_periods, n_assets)
        signal_types: Signal type matrix (n_periods, n_assets): 1=long, -1=short, 0=none
        metric_values: Metric for ranking (n_periods, n_assets). Higher = stronger signal.
    
    Returns:
        Filtered allocation matrix with half-assets cap applied
    """
    n_periods, n_assets = allocs.shape
    max_assets = n_assets // 2
    if max_assets < 1:
        max_assets = 1
    
    result = np.zeros((n_periods, n_assets))
    
    for i in range(n_periods):
        signal_count = 0
        for j in range(n_assets):
            if allocs[i, j] > 1e-9:
                signal_count += 1
        
        if signal_count == 0:
            continue
        
        if signal_count <= max_assets:
            for j in range(n_assets):
                result[i, j] = allocs[i, j]
        else:
            indices = np.zeros(n_assets, dtype=np.int64)
            metrics = np.zeros(n_assets)
            for j in range(n_assets):
                indices[j] = j
                if allocs[i, j] > 1e-9 and not np.isnan(metric_values[i, j]):
                    metrics[j] = metric_values[i, j]
                else:
                    metrics[j] = -1e9
            
            for j1 in range(n_assets):
                for j2 in range(j1 + 1, n_assets):
                    if metrics[j2] > metrics[j1]:
                        tmp_m = metrics[j1]
                        metrics[j1] = metrics[j2]
                        metrics[j2] = tmp_m
                        tmp_i = indices[j1]
                        indices[j1] = indices[j2]
                        indices[j2] = tmp_i
            
            for k in range(max_assets):
                idx = indices[k]
                result[i, idx] = allocs[i, idx]
        
        active_count = 0
        for j in range(n_assets):
            if result[i, j] > 1e-9:
                active_count += 1
        
        if active_count > 0:
            alloc_per_asset = 1.0 / float(max_assets)
            for j in range(n_assets):
                if result[i, j] > 1e-9:
                    result[i, j] = alloc_per_asset
    
    return result


@dataclass
class AllocationFilterParams:
    """Parameters for allocation filter pipeline."""
    
    p_direction: str = "both"
    p_default_asset: Optional[str] = None
    p_default_asset_idx: Optional[int] = None
    p_min_holding_periods: int = 0
    p_switch_threshold_pct: float = 0.0
    p_use_rsi_entry_filter: bool = False
    p_rsi_entry_max: float = 30.0
    p_use_rsi_entry_queue: bool = False
    p_use_rsi_diff_filter: bool = False
    p_rsi_diff_threshold: float = 10.0
    p_top_n: int = 1
    p_use_ranking_logic: bool = False
    p_cap_to_half_assets: bool = False


class AllocationFilter:
    """
    Composable filter pipeline for allocation strategies.
    
    Applies multiple filters in sequence:
    1. Direction filter (long/short/both)
    2. RSI entry filter (optional)
    3. Default asset filter (when no signals)
    4. Normalization
    5. Min holding filter (hysteresis)
    6. Switch threshold filter (optional)
    7. RSI diff filter (optional)
    """
    
    def __init__(self, p_params: AllocationFilterParams):
        self.params = p_params
    
    def apply(
        self,
        p_raw_allocs: np.ndarray,
        p_long_signals: np.ndarray,
        p_short_signals: np.ndarray,
        p_rsi_values: Optional[np.ndarray] = None,
        p_metric_values: Optional[np.ndarray] = None,
        p_signal_types: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Apply all filters to raw allocations.
        
        Args:
            p_raw_allocs: Raw allocation matrix (n_periods, n_assets)
            p_long_signals: Long signal matrix (n_periods, n_assets)
            p_short_signals: Short signal matrix (n_periods, n_assets)
            p_rsi_values: Optional RSI matrix (n_periods, n_assets)
            p_metric_values: Optional metric matrix for ranking (n_periods, n_assets)
            p_signal_types: Optional signal type matrix (n_periods, n_assets): 1=long, -1=short, 0=none
        
        Returns:
            Filtered allocation matrix
        """
        n_periods, n_assets = p_raw_allocs.shape
        allocs = p_raw_allocs.copy()
        
        direction_code = DIRECTION_BOTH
        if self.params.p_direction == "long":
            direction_code = DIRECTION_LONG
        elif self.params.p_direction == "short":
            direction_code = DIRECTION_SHORT
        
        allocs = apply_direction_filter_numba(
            allocs, p_long_signals, p_short_signals, direction_code
        )
        
        if self.params.p_cap_to_half_assets and p_metric_values is not None and p_signal_types is not None:
            allocs = apply_half_assets_cap_numba(allocs, p_signal_types, p_metric_values)
        
        if self.params.p_use_rsi_entry_filter and p_rsi_values is not None:
            allocs, _ = apply_rsi_entry_filter_numba(
                allocs, p_rsi_values, self.params.p_rsi_entry_max
            )
        
        if self.params.p_default_asset_idx is not None:
            allocs = apply_default_asset_filter_numba(
                allocs, self.params.p_default_asset_idx
            )
        
        allocs = normalize_allocations_numba(allocs)
        
        if self.params.p_min_holding_periods > 0:
            initial_allocs = np.zeros(n_assets)
            allocs, _ = apply_min_holding_filter_numba(
                allocs, initial_allocs, self.params.p_min_holding_periods
            )
        
        return allocs
    
    def apply_with_metrics(
        self,
        p_raw_allocs: np.ndarray,
        p_long_signals: np.ndarray,
        p_short_signals: np.ndarray,
        p_rsi_values: Optional[np.ndarray],
        p_metric_values: Optional[np.ndarray],
        p_signal_types: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Apply filters with additional output for tracking blocked/delayed switches.
        
        Returns:
            Tuple of (filtered_allocs, metrics_dict)
            metrics_dict contains: 'rsi_entry_blocked', 'rsi_diff_blocked', 'periods_held'
        """
        n_periods, n_assets = p_raw_allocs.shape
        allocs = p_raw_allocs.copy()
        metrics = {
            'rsi_entry_blocked': np.zeros((n_periods, n_assets), dtype=bool),
            'rsi_diff_blocked': np.zeros(n_periods, dtype=bool),
            'periods_held': np.zeros(n_assets, dtype=np.int64),
        }
        
        direction_code = DIRECTION_BOTH
        if self.params.p_direction == "long":
            direction_code = DIRECTION_LONG
        elif self.params.p_direction == "short":
            direction_code = DIRECTION_SHORT
        
        allocs = apply_direction_filter_numba(
            allocs, p_long_signals, p_short_signals, direction_code
        )
        
        if self.params.p_cap_to_half_assets and p_metric_values is not None and p_signal_types is not None:
            allocs = apply_half_assets_cap_numba(allocs, p_signal_types, p_metric_values)
        
        if self.params.p_use_rsi_entry_filter and p_rsi_values is not None:
            allocs, rsi_blocked = apply_rsi_entry_filter_numba(
                allocs, p_rsi_values, self.params.p_rsi_entry_max
            )
            metrics['rsi_entry_blocked'] = rsi_blocked
        
        if self.params.p_default_asset_idx is not None:
            allocs = apply_default_asset_filter_numba(
                allocs, self.params.p_default_asset_idx
            )
        
        allocs = normalize_allocations_numba(allocs)
        
        if self.params.p_min_holding_periods > 0:
            initial_allocs = np.zeros(n_assets)
            allocs, periods_held = apply_min_holding_filter_numba(
                allocs, initial_allocs, self.params.p_min_holding_periods
            )
            metrics['periods_held'] = periods_held
        
        return allocs, metrics

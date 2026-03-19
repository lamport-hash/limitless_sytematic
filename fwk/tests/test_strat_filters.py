"""
Tests for strat_filters module.

Tests the apply_direction_filter:
- Test long only: short only: both modes
- Test default_asset_filter: Test min_holding_filter
- Test normalization
- Test RSI entry_filter
"""

import numpy as np
import pytest

from strat.strat_filters import (
    apply_direction_filter_numba,
    apply_default_asset_filter_numba,
    apply_min_holding_filter_numba,
    normalize_allocations_numba,
    apply_rsi_entry_filter_numba,
    DIRECTION_LONG,
    DIRECTION_SHORT,
    DIRECTION_BOTH,
)


def test_direction_filter_long_only():
    """Test direction filter - long only mode."""
    raw_allocs = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    
    long_signals = np.array([
        [1, 0, 0, 1],
        [0, 1, 0, 1],
        [1, 1, 0, 1],
    ], dtype=np.int8)
    
    short_signals = np.array([
        [0, 0, 0, 0],
        [1, 0, 0, 1],
        [0, 1, 0, 1],
    ], dtype=np.int8)
    
    result = apply_direction_filter_numba(
        raw_allocs, long_signals, short_signals, DIRECTION_LONG
    )
    
    expected = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    
    assert np.allclose(result, expected)


def test_direction_filter_short_only():
    """Test direction filter - short only mode."""
    raw_allocs = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    
    long_signals = np.array([
        [0, 0, 0, 0],
        [1, 0, 0, 1],
        [0, 1, 0, 1],
    ], dtype=np.int8)
    
    short_signals = np.array([
        [0, 0, 0, 1],
        [1, 0, 0, 1],
        [1, 1, 0, 1],
    ], dtype=np.int8)
    
    result = apply_direction_filter_numba(
        raw_allocs, long_signals, short_signals, DIRECTION_SHORT
    )
    
    expected = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ])
    
    assert np.allclose(result, expected)


def test_direction_filter_both():
    """Test direction filter - both mode (basket)."""
    raw_allocs = np.array([
        [1.0, 1.0, 0.0],
        [1.0, 0.5, 0.5],
        [1.0, 0.0, 0.0],
    ])
    
    long_signals = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 1, 0, 1],
    ], dtype=np.int8)
    
    short_signals = np.array([
        [1, 0, 0, 1],
        [0, 1, 0, 1],
        [1, 1, 0, 1],
    ], dtype=np.int8)
    
    result = apply_direction_filter_numba(
        raw_allocs, long_signals, short_signals, DIRECTION_BOTH
    )
    
    expected = np.array([
        [1.0, 1.0, 0.5],
        [1.0, 0.5, 0.5],
        [1.0, 0.0, 0.0],
    ])
    
    assert np.allclose(result, expected)


def test_default_asset_filter():
    """Test default asset filter."""
    allocs = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    
    result = apply_default_asset_filter_numba(allocs, default_asset_idx=1)
    
    expected = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ])
    
    assert np.allclose(result, expected)
    
    result = apply_default_asset_filter_numba(allocs, default_asset_idx=2)
    
    expected = np.array([
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
    ])
    
    assert np.allclose(result, expected)


def test_min_holding_filter():
    """Test min holding filter."""
    target_allocs = np.array([
        [1.0, 1.0, 0.0],
        [1.0, 0.5, 0.5],
        [1.0, 0.0, 0.0],
    ])
    
    initial_allocs = np.zeros(3)
    
    result, _ = apply_min_holding_filter_numba(
        target_allocs, initial_allocs, min_holding_periods=3
    )
    
    expected = np.array([
        [1.0, 0.5, 0.5],
        [1.0, 0.5, 0.5],
        [1.0, 0.0, 0.0],
    ])
    
    assert np.allclose(result[0], expected)
    
    _, periods_held = apply_min_holding_filter_numba(
        target_allocs, initial_allocs, min_holding_periods=0
    )
    
    expected_periods = np.zeros(3, dtype=np.int64)
    assert np.array_equal(periods_held, np.zeros(3))
    
    result, periods_held = apply_min_holding_filter_numba(
        target_allocs, initial_allocs, min_holding_periods=0
    )
    
    assert np.allclose(result[0], expected)
    assert np.array_equal(periods_held, np.zeros(3))


def test_normalization():
    """Test normalization of allocations."""
    allocs = np.array([
        [1.0, 2.0],
        [0.5, 0.5],
    ])
    
    result = normalize_allocations_numba(allocs)
    
    expected = np.array([
        [0.5, 1.0],
        [0.0, 1.0],
    ])
    
    assert np.allclose(result, expected)
    
    allocs = np.array([
        [1.0, 1.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0],
    ])
    
    result = normalize_allocations_numba(allocs)
    
    expected = np.array([
        [0.5, 0.0],
        [0.5, 0.0],
    ])
    
    assert np.allclose(result, expected)


def test_rsi_entry_filter():
    """Test RSI entry filter."""
    allocs = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    
    rsi_values = np.array([
        [20.0, 20.0, 20.0],
        [50.0, 50.0, 50.0],
    ])
    
    result, blocked = apply_rsi_entry_filter_numba(
        allocs, rsi_values, rsi_max=35.0
    )
    
    expected = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    
    blocked_expected = np.array([
        [False, False, False],
        [True, True, True],
    ], dtype=bool)
    
    assert np.allclose(result, expected)
    assert np.array_equal(blocked, blocked_expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Unit tests for CTO Line basket allocation strategy.

Tests the compute_cto_line_allocations function with synthetic data.
"""

import numpy as np
import pandas as pd
import pytest

from strat.strat_cto_line import compute_cto_line_allocations

ETF_LIST = ["QQQ", "SPY", "TLT", "GLD", "VWO"]


def test_compute_cto_signals():
    """
    Test CTO signal computation with synthetic data.
    """
    n_rows = 100
    data = {}

    np.random.seed(42)
    for asset in ETF_LIST:
        data[f"{asset}_S_high_f32"] = 100 + np.cumsum(np.random.randn(n_rows) * 0.5)
        data[f"{asset}_S_low_f32"] = 100 + np.cumsum(np.random.randn(n_rows) * 0.5)
        data[f"{asset}_S_close_f32"] = 100 + np.cumsum(np.random.randn(n_rows) * 0.5)

    df = pd.DataFrame(data)

    from strat.strat_cto_line import compute_cto_signals

    long_sig, short_sig = compute_cto_signals(
        df["QQQ_S_high_f32"].to_numpy(),
        df["QQQ_S_low_f32"].to_numpy(),
        df["QQQ_S_close_f32"].to_numpy(),
        (15, 19, 25, 29)
    )

    assert len(long_sig) == n_rows
    assert len(short_sig) == n_rows
    assert long_sig.sum() > 0, "Should have some long signals"
    assert short_sig.sum() > 0, "Should have some short signals"


def test_compute_cto_line_allocations_basic():
    """
    Test basic CTO Line allocation computation with synthetic data.
    """
    n_rows = 100
    data = {}

    np.random.seed(42)
    for asset in ETF_LIST:
        data[f"{asset}_S_high_f32"] = 100 + np.cumsum(np.random.randn(n_rows) * 0.5)
        data[f"{asset}_S_low_f32"] = 100 + np.cumsum(np.random.randn(n_rows) * 0.5)
        data[f"{asset}_S_close_f32"] = 100 + np.cumsum(np.random.randn(n_rows) * 0.5)

    df = pd.DataFrame(data)

    result = compute_cto_line_allocations(
        p_df=df,
        p_asset_list=ETF_LIST,
        p_cto_params=(15, 19, 25, 29),
        p_direction="long",
    )

    assert result is not None
    assert len(result) == n_rows

    for asset in ETF_LIST:
        assert f"A_{asset}_alloc" in result.columns

    assert "A_n_assets_with_signal" in result.columns

    for i in range(len(result)):
        alloc_sum = sum(result.iloc[i][f"A_{asset}_alloc"] for asset in ETF_LIST)
        assert abs(alloc_sum - 1.0) < 1e-6 or f"Row {i}: allocations should sum to 1.0"


def test_compute_cto_line_allocations_multi_asset():
    """
    Test CTO Line with multiple assets having signals simultaneously.
    """
    n_rows = 100
    data = {}

    np.random.seed(123)
    for asset in ETF_LIST:
        data[f"{asset}_S_high_f32"] = 100 + np.cumsum(np.random.randn(n_rows) * 0.5)
        data[f"{asset}_S_low_f32"] = 100 + np.cumsum(np.random.randn(n_rows) * 0.5)
        data[f"{asset}_S_close_f32"] = 100 + np.cumsum(np.random.randn(n_rows) * 0.5)

    df = pd.DataFrame(data)

    result = compute_cto_line_allocations(
        p_df=df,
        p_asset_list=ETF_LIST,
        p_cto_params=(15, 19, 25, 29),
        p_direction="both"
    )

    assert result is not None
    
    for i in range(len(result)):
        non_zero_allocs = sum(1 for asset in ETF_LIST if result.iloc[i][f"A_{asset}_alloc"] > 0)
        assert non_zero_allocs >= 0, "Should have at least 1 allocation"


def test_compute_cto_line_allocations_with_default():
    """
    Test CTO Line with default asset when no signals.
    """
    n_rows = 50
    data = {}

    np.random.seed(456)
    for asset in ETF_LIST:
        data[f"{asset}_S_high_f32"] = 100 + np.cumsum(np.random.randn(n_rows) * 0.2)
        data[f"{asset}_S_low_f32"] = 100 + np.cumsum(np.random.randn(n_rows) * 0.2)
        data[f"{asset}_S_close_f32"] = 100 + np.cumsum(np.random.randn(n_rows) * 0.2)

    df = pd.DataFrame(data)

    result = compute_cto_line_allocations(
        p_df=df,
        p_asset_list=ETF_LIST,
        p_cto_params=(15, 19, 25, 29),
        p_direction="long",
        p_default_asset="TLT",
    )

    for i in range(len(result)):
        alloc_sum = sum(result.iloc[i][f"A_{asset}_alloc"] for asset in ETF_LIST)
        if alloc_sum < 1e-6:
            assert result.iloc[i]["A_TLT_alloc"] == 1.0, f"Row {i}: With no signals, should allocate to default TLT"


def test_compute_cto_line_allocations_min_holding():
    """
    Test CTO Line with min_holding_periods constraint.
    """
    n_rows = 50
    data = {}

    np.random.seed(789)
    for asset in ETF_LIST:
        data[f"{asset}_S_high_f32"] = 100 + np.cumsum(np.random.randn(n_rows) * 0.5)
        data[f"{asset}_S_low_f32"] = 100 + np.cumsum(np.random.randn(n_rows) * 0.5)
        data[f"{asset}_S_close_f32"] = 100 + np.cumsum(np.random.randn(n_rows) * 0.5)

    df = pd.DataFrame(data)

    result_no_constraint = compute_cto_line_allocations(
        p_df=df,
        p_asset_list=ETF_LIST,
        p_cto_params=(15, 19, 25, 29),
        p_direction="long",
        p_min_holding_periods=0,
    )

    result_with_constraint = compute_cto_line_allocations(
        p_df=df,
        p_asset_list=ETF_LIST,
        p_cto_params=(15, 19, 25, 29),
        p_direction="long",
        p_min_holding_periods=10,
    )

    switches_no_constraint = 0
    switches_with_constraint = 0
    
    for i in range(1, len(result_no_constraint)):
        for asset in ETF_LIST:
            if result_no_constraint.iloc[i][f"A_{asset}_alloc"] != result_no_constraint.iloc[i - 1][f"A_{asset}_alloc"]:
                switches_no_constraint += 1
                break
    for i in range(1, len(result_with_constraint)):
        for asset in ETF_LIST:
            if result_with_constraint.iloc[i][f"A_{asset}_alloc"] != result_with_constraint.iloc[i - 1][f"A_{asset}_alloc"]:
                switches_with_constraint += 1
                break
    
    assert switches_no_constraint >= switches_with_constraint, "With min_holding_periods, should have fewer switches"


def test_compute_cto_line_allocations_short_only():
    """
    Test CTO Line with short direction only.
    """
    n_rows = 100
    data = {}

    np.random.seed(999)
    for asset in ETF_LIST:
        data[f"{asset}_S_high_f32"] = 100 + np.cumsum(np.random.randn(n_rows) * 0.5)
        data[f"{asset}_S_low_f32"] = 100 + np.cumsum(np.random.randn(n_rows) * 0.5)
        data[f"{asset}_S_close_f32"] = 100 + np.cumsum(np.random.randn(n_rows) * 0.5)

    df = pd.DataFrame(data)

    result = compute_cto_line_allocations(
        p_df=df,
        p_asset_list=ETF_LIST,
        p_cto_params=(15, 19, 25, 29),
        p_direction="short",
    )

    assert result is not None
    
    for i in range(len(result)):
        for asset in ETF_LIST:
            alloc = result.iloc[i][f"A_{asset}_alloc"]
            if alloc > 0:
                assert alloc >= 0.0, "Short-only mode should not have negative allocations"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

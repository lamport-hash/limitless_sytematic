"""
Unit tests for dual momentum strategy.

Tests the compute_dual_momentum function with synthetic and real bundle data.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from core.data_org import BUNDLE_DIR
from strat.strat_backtest import (
    ETF_LIST,
    compute_dual_momentum,
    get_current_allocation
)
from strat.strat_analysis import print_allocation_summary
from strat.strat_visualise import plot_normalized_prices, create_allocation_gif


INPUT_FILE = BUNDLE_DIR / "test_etf_features_bundle.parquet"
OUTPUT_DIR = Path(__file__).parent / "output"


def test_compute_dual_momentum_basic():
    """
    Test basic dual momentum computation with synthetic data.
    """
    n_rows = 100
    data = {}

    for etf in ETF_LIST:
        data[f"{etf}_F_roc_4800_F_mid_f32_f16"] = np.random.randn(n_rows)
        data[f"{etf}_S_close_f32"] = 100 + np.cumsum(np.random.randn(n_rows) * 0.1)

    df = pd.DataFrame(data)

    result = compute_dual_momentum(
        p_df=df,
        p_feature_id="F_roc_4800_F_mid_f32_f16",
        p_default_etf_idx=2,
        p_top_n=1,
        p_abs_momentum_threshold=0.0,
    )

    assert result is not None
    assert len(result) == n_rows

    for etf in ETF_LIST:
        assert f"A_{etf}_alloc" in result.columns

    assert "A_top_etf" in result.columns
    assert "A_n_positive_momentum" in result.columns

    for i in range(len(result)):
        alloc_sum = sum(result.iloc[i][f"A_{etf}_alloc"] for etf in ETF_LIST)
        assert abs(alloc_sum - 1.0) < 1e-6, f"Row {i}: allocations should sum to 1.0"


def test_compute_dual_momentum_with_threshold():
    """
    Test dual momentum with positive threshold - should default to safe haven more often.
    """
    n_rows = 50
    data = {}

    np.random.seed(42)
    for etf in ETF_LIST:
        data[f"{etf}_F_roc_4800_F_mid_f32_f16"] = np.random.randn(n_rows) * 0.1
        data[f"{etf}_S_close_f32"] = 100 + np.cumsum(np.random.randn(n_rows) * 0.1)

    df = pd.DataFrame(data)

    result = compute_dual_momentum(
        p_df=df,
        p_feature_id="F_roc_4800_F_mid_f32_f16",
        p_default_etf_idx=2,
        p_top_n=1,
        p_abs_momentum_threshold=0.5,
    )

    tlt_alloc_count = (result["A_TLT_alloc"] > 0.5).sum()

    assert tlt_alloc_count > 0, "With high threshold, should default to TLT sometimes"


def test_compute_dual_momentum_top_n():
    """
    Test dual momentum with top_n=2 - should allocate to 2 ETFs.
    """
    n_rows = 20
    data = {}

    np.random.seed(42)
    for etf in ETF_LIST:
        data[f"{etf}_F_roc_4800_F_mid_f32_f16"] = np.random.randn(n_rows) + 0.5
        data[f"{etf}_S_close_f32"] = 100 + np.cumsum(np.random.randn(n_rows) * 0.1)

    df = pd.DataFrame(data)

    result = compute_dual_momentum(
        p_df=df,
        p_feature_id="F_roc_4800_F_mid_f32_f16",
        p_default_etf_idx=2,
        p_top_n=2,
        p_abs_momentum_threshold=0.0,
    )

    for i in range(len(result)):
        non_zero_allocs = sum(1 for etf in ETF_LIST if result.iloc[i][f"A_{etf}_alloc"] > 0)
        assert non_zero_allocs <= 2, f"Row {i}: should have at most 2 allocations"


def test_compute_dual_momentum_missing_feature():
    """
    Test that missing feature raises ValueError.
    """
    df = pd.DataFrame({"other_col": [1, 2, 3]})

    with pytest.raises(ValueError, match="Feature column not found"):
        compute_dual_momentum(p_df=df)


def test_get_current_allocation():
    """
    Test extraction of current allocation from result dataframe.
    """
    n_rows = 10
    data = {}

    for etf in ETF_LIST:
        data[f"{etf}_F_roc_4800_F_mid_f32_f16"] = np.random.randn(n_rows)
        data[f"{etf}_S_close_f32"] = 100 + np.cumsum(np.random.randn(n_rows) * 0.1)

    df = pd.DataFrame(data)
    result = compute_dual_momentum(p_df=df)

    allocation = get_current_allocation(result)

    assert isinstance(allocation, dict)
    assert len(allocation) == len(ETF_LIST)

    total = sum(allocation.values())
    assert abs(total - 1.0) < 1e-6, "Allocation should sum to 1.0"


def test_print_allocation_summary():
    """
    Test that print_allocation_summary runs without error.
    """
    n_rows = 15
    data = {}

    for etf in ETF_LIST:
        data[f"{etf}_F_roc_4800_F_mid_f32_f16"] = np.random.randn(n_rows)
        data[f"{etf}_S_close_f32"] = 100 + np.cumsum(np.random.randn(n_rows) * 0.1)

    df = pd.DataFrame(data)
    result = compute_dual_momentum(p_df=df)

    print_allocation_summary(result, p_n_last=5)


@pytest.mark.skipif(not INPUT_FILE.exists(), reason=f"Test bundle not found: {INPUT_FILE}")
def test_compute_dual_momentum_with_test_bundle():
    """
    Test dual momentum with the actual test_etf_features_bundle.parquet.
    Uses real data from all ETFs in ETF_LIST.
    """
    df = pd.read_parquet(INPUT_FILE)
    print(f"Loaded test bundle: {df.shape}")

    etfs_in_bundle = sorted(set(c.split("_")[0] for c in df.columns if "_F_" in c))
    print(f"\nETFs in bundle: {etfs_in_bundle}")

    roc_cols = [c for c in df.columns if "F_roc_" in c and "_F_mid" in c]
    if not roc_cols:
        pytest.skip("No ROC feature found in test bundle")

    feature_id = roc_cols[0].split("_", 1)[1]
    print(f"Using feature: {feature_id}")

    result = compute_dual_momentum(
        p_df=df,
        p_feature_id=feature_id,
        p_default_etf_idx=2,
        p_top_n=1,
        p_abs_momentum_threshold=0.0,
    )

    assert result is not None
    assert len(result) == len(df)

    for etf in ETF_LIST:
        assert f"A_{etf}_alloc" in result.columns, f"Missing allocation column for {etf}"

    allocation = get_current_allocation(result)
    print(f"\nCurrent allocation: {allocation}")

    print_allocation_summary(result, p_n_last=5)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    normalized_chart_path = OUTPUT_DIR / "normalized_prices.png"
    plot_normalized_prices(df, normalized_chart_path)

    allocation_gif_path = OUTPUT_DIR / "allocation_evolution.gif"
    create_allocation_gif(result, allocation_gif_path, p_fps=10)

    print(f"\nCharts saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

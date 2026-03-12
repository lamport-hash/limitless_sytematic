"""
Unit tests for dual momentum backtest module.

Tests order generation, performance computation, and period returns.
"""

import os

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from backtest.dual_momentum_backtest import (
    OHLC_COLS,
    Order,
    generate_orders_from_allocations,
    compute_strategy_performance,
    save_backtest_diagnostics,
)
from core.data_org import BUNDLE_DIR
from strat.strat_backtest import ETF_LIST, compute_dual_momentum, create_test_bundle_with_allocations
from strat.strat_analysis import compute_period_returns, print_metrics
from strat.strat_visualise import (
    plot_pnl_histogram,
    plot_portfolio_value,
    plot_allocation_history,
    plot_cumulative_pnl,
    plot_monthly_returns_chart,
)


INPUT_FILE = BUNDLE_DIR / "test_etf_features_bundle.parquet"
OUTPUT_DIR = Path(__file__).parent / "output"


def test_generate_orders_basic():
    """
    Test basic order generation from allocations.
    """
    df = create_test_bundle_with_allocations(p_n_rows=50)

    orders_df, orders = generate_orders_from_allocations(
        p_df=df,
        p_rebalance_threshold=0.05,
    )

    assert len(orders_df) > 0, "Should generate at least one order"
    assert len(orders) == len(orders_df)

    assert "timestamp" in orders_df.columns
    assert "etf" in orders_df.columns
    assert "direction" in orders_df.columns
    assert "size" in orders_df.columns
    assert "price" in orders_df.columns
    assert "allocation" in orders_df.columns

    for order in orders:
        assert isinstance(order, Order)
        assert order.etf in ETF_LIST
        assert order.direction in [1.0, -1.0]


def test_generate_orders_rebalance_threshold():
    """
    Test that higher rebalance threshold results in fewer orders.
    """
    df = create_test_bundle_with_allocations(p_n_rows=100)

    orders_df_low, _ = generate_orders_from_allocations(
        p_df=df,
        p_rebalance_threshold=0.01,
    )

    orders_df_high, _ = generate_orders_from_allocations(
        p_df=df,
        p_rebalance_threshold=0.5,
    )

    assert len(orders_df_low) >= len(orders_df_high), (
        "Higher threshold should result in fewer orders"
    )


def test_compute_strategy_performance_basic():
    """
    Test basic strategy performance computation.
    """
    df = create_test_bundle_with_allocations(p_n_rows=50)

    orders_df, _ = generate_orders_from_allocations(p_df=df)

    metrics, result_df = compute_strategy_performance(
        p_df=df,
        p_orders_df=orders_df,
    )

    assert isinstance(metrics, dict)
    assert "total_return" in metrics
    assert "sharpe_ratio" in metrics
    assert "max_drawdown" in metrics
    assert "win_rate" in metrics
    assert "n_trades" in metrics

    assert isinstance(result_df, pd.DataFrame)
    assert "pnl_per_bar" in result_df.columns
    assert "cum_pnl" in result_df.columns
    assert "portfolio_value" in result_df.columns


def test_compute_strategy_performance_metrics_values():
    """
    Test that performance metrics have valid values.
    """
    df = create_test_bundle_with_allocations(p_n_rows=100)

    orders_df, _ = generate_orders_from_allocations(p_df=df)

    metrics, result_df = compute_strategy_performance(
        p_df=df,
        p_orders_df=orders_df,
    )

    assert metrics["n_bars"] == len(df)
    assert metrics["n_trades"] == len(orders_df)
    assert 0.0 <= metrics["win_rate"] <= 1.0
    assert metrics["final_portfolio_value"] > 0
    assert metrics["max_drawdown"] <= 0


def test_compute_period_returns_basic():
    """
    Test basic period returns computation.
    """
    df = create_test_bundle_with_allocations(p_n_rows=200)

    orders_df, _ = generate_orders_from_allocations(p_df=df)

    _, result_df = compute_strategy_performance(
        p_df=df,
        p_orders_df=orders_df,
    )

    result_df.index = pd.date_range(start="2023-01-01", periods=len(result_df), freq="h")

    monthly_returns, yearly_returns = compute_period_returns(result_df)

    assert isinstance(monthly_returns, pd.DataFrame)
    assert isinstance(yearly_returns, pd.DataFrame)

    assert "return" in monthly_returns.columns
    assert "n_bars" in monthly_returns.columns
    assert "win_rate" in monthly_returns.columns

    assert "return" in yearly_returns.columns
    assert "n_bars" in yearly_returns.columns
    assert "win_rate" in yearly_returns.columns


def test_print_metrics():
    """
    Test that print_metrics runs without error.
    """
    df = create_test_bundle_with_allocations(p_n_rows=50)

    orders_df, _ = generate_orders_from_allocations(p_df=df)

    metrics, _ = compute_strategy_performance(
        p_df=df,
        p_orders_df=orders_df,
    )

    print_metrics(metrics)


def test_missing_allocation_column():
    """
    Test that missing allocation columns raise ValueError.
    """
    df = pd.DataFrame({"other_col": [1, 2, 3]})

    with pytest.raises(ValueError, match="Required column not found"):
        generate_orders_from_allocations(p_df=df)


def test_missing_close_column():
    """
    Test that missing close price columns raise ValueError.
    """
    data = {}
    for etf in ETF_LIST:
        data[f"A_{etf}_alloc"] = [0.2] * 10

    df = pd.DataFrame(data)

    with pytest.raises(ValueError, match="Required column not found"):
        generate_orders_from_allocations(p_df=df)


@pytest.mark.skipif(not INPUT_FILE.exists(), reason=f"Test bundle not found: {INPUT_FILE}")
def test_backtest_with_test_bundle():
    """
    Test backtest with actual test_etf_features_bundle.parquet.
    Uses real data from all ETFs in ETF_LIST.
    """
    df = pd.read_parquet(INPUT_FILE)
    print(f"\n{'=' * 60}")
    print(f"Loaded test bundle: {df.shape}")
    print(f"{'=' * 60}")

    etfs_in_bundle = sorted(set(c.split("_")[0] for c in df.columns if "_F_" in c))
    print(f"\nETFs in bundle: {etfs_in_bundle}")

    roc_cols = [c for c in df.columns if "F_roc_" in c and "_F_mid" in c]
    if not roc_cols:
        pytest.skip("No ROC feature found in test bundle")

    feature_id = roc_cols[0].split("_", 1)[1]
    print(f"Using feature: {feature_id}")

    df = compute_dual_momentum(
        p_df=df,
        p_feature_id=feature_id,
        p_default_etf_idx=2,
        p_top_n=1,
    )

    orders_df, orders = generate_orders_from_allocations(
        p_df=df,
        p_rebalance_threshold=0.05,
    )

    print(f"\nGenerated {len(orders_df)} orders")

    metrics, result_df = compute_strategy_performance(
        p_df=df,
        p_orders_df=orders_df,
    )

    print_metrics(metrics)

    try:
        monthly_returns, yearly_returns = compute_period_returns(result_df)
    except ValueError:
        result_df.index = pd.date_range(start="2020-01-01", periods=len(result_df), freq="h")
        monthly_returns, yearly_returns = compute_period_returns(result_df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_backtest_diagnostics(
        p_result_df=result_df,
        p_orders_df=orders_df,
        p_monthly_returns=monthly_returns,
        p_yearly_returns=yearly_returns,
        p_metrics=metrics,
        p_output_dir=OUTPUT_DIR,
    )

    if not os.environ.get('SKIP_PLOTS'):
        plot_pnl_histogram(result_df, OUTPUT_DIR / "pnl_histogram.png")
        plot_portfolio_value(result_df, OUTPUT_DIR / "portfolio_value.png")
        plot_allocation_history(result_df, OUTPUT_DIR / "allocation_history.png")
        plot_cumulative_pnl(result_df, OUTPUT_DIR / "cumulative_pnl.png")
        plot_monthly_returns_chart(monthly_returns, OUTPUT_DIR / "monthly_returns.png")

    print(f"\n{'=' * 60}")
    print(f"All diagnostics saved to: {OUTPUT_DIR}")
    print(f"{'=' * 60}")

    assert metrics["n_bars"] == len(df)
    assert metrics["n_trades"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

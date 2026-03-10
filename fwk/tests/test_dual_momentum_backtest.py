"""
Unit tests for dual momentum backtest module.

Tests order generation, performance computation, and period returns.
"""

import numpy as np
import pandas as pd
import pytest

from backtest.dual_momentum_backtest import (
    OHLC_COLS,
    Order,
    generate_orders_from_allocations,
    compute_strategy_performance,
    compute_period_returns,
    print_metrics,
)
from core.data_org import BUNDLE_DIR
from strat.dual_momentum import ETF_LIST, compute_dual_momentum


INPUT_FILE = BUNDLE_DIR / "test_etf_features_bundle.parquet"


def create_test_bundle_with_allocations(p_n_rows: int = 100) -> pd.DataFrame:
    """
    Create a test bundle with all required columns for backtesting.
    """
    np.random.seed(42)
    data = {}
    
    for etf in ETF_LIST:
        data[f"{etf}_F_roc_4800_F_mid_f32_f16"] = np.random.randn(p_n_rows) * 0.1
        close_prices = 100 + np.cumsum(np.random.randn(p_n_rows) * 0.5)
        data[f"{etf}_S_close_f32"] = close_prices
        data[f"{etf}_S_open_f32"] = close_prices * (1 + np.random.randn(p_n_rows) * 0.01)
        data[f"{etf}_S_high_f32"] = close_prices * (1 + abs(np.random.randn(p_n_rows)) * 0.01)
        data[f"{etf}_S_low_f32"] = close_prices * (1 - abs(np.random.randn(p_n_rows)) * 0.01)
        data[f"{etf}_S_volume_f64"] = np.random.randint(1000, 10000, p_n_rows).astype(float)
    
    df = pd.DataFrame(data)
    
    df = compute_dual_momentum(
        p_df=df,
        p_feature_id="F_roc_4800_F_mid_f32_f16",
        p_default_etf_idx=2,
        p_top_n=1,
    )
    
    return df


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
    
    assert len(orders_df_low) >= len(orders_df_high), \
        "Higher threshold should result in fewer orders"


def test_compute_strategy_performance_basic():
    """
    Test basic strategy performance computation.
    """
    df = create_test_bundle_with_allocations(p_n_rows=50)
    
    orders_df, _ = generate_orders_from_allocations(p_df=df)
    
    metrics, result_df = compute_strategy_performance(
        p_df=df,
        p_orders_df=orders_df,
        p_plot_histogram=False,
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
        p_plot_histogram=False,
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
        p_plot_histogram=False,
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
        p_plot_histogram=False,
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


@pytest.mark.skipif(
    not INPUT_FILE.exists(),
    reason=f"Test bundle not found: {INPUT_FILE}"
)
def test_backtest_with_test_bundle():
    """
    Test backtest with actual test_etf_features_bundle.parquet.
    Note: Bundle only has QQQ, so we create synthetic other ETFs.
    """
    df = pd.read_parquet(INPUT_FILE)
    print(f"Loaded test bundle: {df.shape}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    qqq_cols = [c for c in numeric_cols if c.startswith("QQQ_")]
    
    for etf in ["SPY", "TLT", "GLD", "VWO"]:
        for col in qqq_cols:
            new_col = col.replace("QQQ_", f"{etf}_")
            multiplier = 0.8 + 0.4 * np.random.random()
            df[new_col] = df[col].astype(float) * multiplier
    
    roc_cols = [c for c in df.columns if "F_roc_60_F_mid" in c]
    if not roc_cols:
        pytest.skip("No ROC feature found in test bundle")
    
    feature_id = "F_roc_60_F_mid_f32_f16"
    
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
    
    print(f"Generated {len(orders_df)} orders")
    
    metrics, result_df = compute_strategy_performance(
        p_df=df,
        p_orders_df=orders_df,
        p_plot_histogram=False,
    )
    
    print(f"Total return: {metrics['total_return_pct']:.2f}%")
    print(f"Sharpe ratio: {metrics['sharpe_ratio']:.4f}")
    print(f"Max drawdown: {metrics['max_dd_pct']:.2f}%")
    
    assert metrics["n_bars"] == len(df)
    assert metrics["n_trades"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

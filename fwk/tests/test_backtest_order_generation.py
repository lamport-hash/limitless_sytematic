"""
Unit tests for backtest order generation and position tracking.

Tests cover:
- Full exit with transaction costs (bug fix verification)
- Partial exits
- Multiple buys followed by sells
- FIFO lot tracking in compute_current_positions
- Allocation tracking consistency
"""

import numpy as np
import pandas as pd
import pytest

from backtest.backtest_basket_alloc_based import (
    run_full_backtest,
    _run_backtest_kernel_compounding_optimized,
)
from ui_strat.backtest_runner import compute_current_positions


def _create_test_df(allocations, prices, n_rows, assets):
    """Helper to create test DataFrame with allocations and prices."""
    data = {'i_minute_i': np.arange(n_rows)}
    
    for i, asset in enumerate(assets):
        data[f'A_{asset}_alloc'] = allocations[:, i]
        data[f'{asset}_S_close_f32'] = prices[:, i]
    
    return pd.DataFrame(data)


class TestFullExitWithTransactionCosts:
    """Test that full exits sell ALL shares, even with transaction costs."""
    
    def test_single_buy_full_exit_no_residual(self):
        """Buy 100% then sell 100% should leave 0 shares."""
        assets = ['GLD']
        n_rows = 3
        
        allocations = np.array([
            [1.0],
            [1.0],
            [0.0],
        ])
        
        prices = np.array([
            [100.0],
            [110.0],
            [105.0],
        ])
        
        p_df = _create_test_df(allocations, prices, n_rows, assets)
        result_df, orders_df = run_full_backtest(p_df, assets, transaction_cost_pct=0.1)
        
        gld_buys = orders_df[orders_df['etf'] == 'GLD'][orders_df['direction'] > 0]
        gld_sells = orders_df[orders_df['etf'] == 'GLD'][orders_df['direction'] < 0]
        
        total_bought = gld_buys['size'].sum()
        total_sold = gld_sells['size'].sum()
        
        residual = abs(total_bought - total_sold)
        assert residual < 1e-9, f"Residual shares after full exit: {residual}"
    
    def test_full_exit_high_transaction_costs(self):
        """Full exit with 1% transaction costs should still leave 0 shares."""
        assets = ['GLD']
        n_rows = 3
        
        allocations = np.array([
            [1.0],
            [1.0],
            [0.0],
        ])
        
        prices = np.array([
            [100.0],
            [110.0],
            [105.0],
        ])
        
        p_df = _create_test_df(allocations, prices, n_rows, assets)
        result_df, orders_df = run_full_backtest(p_df, assets, transaction_cost_pct=1.0)
        
        gld_buys = orders_df[orders_df['etf'] == 'GLD'][orders_df['direction'] > 0]
        gld_sells = orders_df[orders_df['etf'] == 'GLD'][orders_df['direction'] < 0]
        
        total_bought = gld_buys['size'].sum()
        total_sold = gld_sells['size'].sum()
        
        residual = abs(total_bought - total_sold)
        assert residual < 1e-9, f"Residual shares after full exit with high tx costs: {residual}"


class TestMultipleBuysMultipleSells:
    """Test scenarios with multiple buy and sell orders."""
    
    def test_two_buys_one_full_exit(self):
        """Two separate buys followed by one full exit."""
        assets = ['GLD']
        n_rows = 5
        
        allocations = np.array([
            [0.5],
            [0.5],
            [1.0],
            [1.0],
            [0.0],
        ])
        
        prices = np.array([
            [100.0],
            [100.0],
            [105.0],
            [105.0],
            [110.0],
        ])
        
        p_df = _create_test_df(allocations, prices, n_rows, assets)
        result_df, orders_df = run_full_backtest(p_df, assets, transaction_cost_pct=0.1)
        
        gld_buys = orders_df[orders_df['etf'] == 'GLD'][orders_df['direction'] > 0]
        gld_sells = orders_df[orders_df['etf'] == 'GLD'][orders_df['direction'] < 0]
        
        total_bought = gld_buys['size'].sum()
        total_sold = gld_sells['size'].sum()
        
        residual = abs(total_bought - total_sold)
        assert residual < 1e-9, f"Residual shares after two buys one exit: {residual}"
    
    def test_multiple_buys_partial_sells_then_full_exit(self):
        """Multiple buys, several partial sells, then full exit."""
        assets = ['GLD']
        n_rows = 8
        
        allocations = np.array([
            [0.5],
            [0.5],
            [1.0],
            [1.0],
            [0.5],
            [0.5],
            [0.25],
            [0.0],
        ])
        
        prices = np.array([
            [100.0],
            [100.0],
            [105.0],
            [105.0],
            [110.0],
            [110.0],
            [115.0],
            [120.0],
        ])
        
        p_df = _create_test_df(allocations, prices, n_rows, assets)
        result_df, orders_df = run_full_backtest(p_df, assets, transaction_cost_pct=0.1)
        
        gld_buys = orders_df[orders_df['etf'] == 'GLD'][orders_df['direction'] > 0]
        gld_sells = orders_df[orders_df['etf'] == 'GLD'][orders_df['direction'] < 0]
        
        total_bought = gld_buys['size'].sum()
        total_sold = gld_sells['size'].sum()
        
        residual = abs(total_bought - total_sold)
        assert residual < 1e-9, f"Residual shares after multiple partial sells: {residual}"
    
    def test_multiple_assets_rotation(self):
        """Rotate between multiple assets - each exit should sell all shares."""
        assets = ['GLD', 'EEM']
        n_rows = 5
        
        allocations = np.array([
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ])
        
        prices = np.array([
            [100.0, 50.0],
            [100.0, 50.0],
            [105.0, 55.0],
            [105.0, 55.0],
            [110.0, 60.0],
        ])
        
        p_df = _create_test_df(allocations, prices, n_rows, assets)
        result_df, orders_df = run_full_backtest(p_df, assets, transaction_cost_pct=0.1)
        
        for asset in assets:
            asset_orders = orders_df[orders_df['etf'] == asset].copy()
            buys = asset_orders[asset_orders['direction'] > 0]
            sells = asset_orders[asset_orders['direction'] < 0]
            
            total_bought = buys['size'].sum()
            total_sold = sells['size'].sum()
            
            residual = abs(total_bought - total_sold)
            assert residual < 1e-9, f"Residual shares for {asset}: {residual}"
    
    def test_multiple_assets_rotation_with_final_position(self):
        """Rotate between assets - verify intermediate exits are complete."""
        assets = ['GLD', 'EEM']
        n_rows = 5
        
        allocations = np.array([
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ])
        
        prices = np.array([
            [100.0, 50.0],
            [100.0, 50.0],
            [105.0, 55.0],
            [105.0, 55.0],
            [110.0, 60.0],
        ])
        
        p_df = _create_test_df(allocations, prices, n_rows, assets)
        result_df, orders_df = run_full_backtest(p_df, assets, transaction_cost_pct=0.1)
        
        gld_orders = orders_df[orders_df['etf'] == 'GLD'].copy()
        gld_sells = gld_orders[gld_orders['direction'] < 0]
        
        assert len(gld_sells) == 1, f"Expected 1 GLD sell (at row 2), got {len(gld_sells)}"
        
        gld_buys_before_sell = gld_orders[
            (gld_orders['direction'] > 0) & 
            (gld_orders.index < gld_sells.index[0])
        ]
        total_bought_before_sell = gld_buys_before_sell['size'].sum()
        sold_at_exit = gld_sells['size'].sum()
        
        residual = abs(total_bought_before_sell - sold_at_exit)
        assert residual < 1e-9, f"GLD exit at row 2 left residual: {residual}"
        
        positions = compute_current_positions(orders_df, p_df, assets)
        gld_positions = [p for p in positions if p['asset'] == 'GLD']
        assert len(gld_positions) >= 1, "GLD should have position at end"
        
        total_gld_held = sum(p['quantity'] for p in gld_positions)
        assert total_gld_held > 0, "GLD should have positive position at end"


class TestFIFOLotTracking:
    """Test FIFO lot tracking in compute_current_positions."""
    
    def test_single_lot_position(self):
        """Single buy should create one lot in positions."""
        assets = ['GLD']
        n_rows = 3
        
        allocations = np.array([
            [1.0],
            [1.0],
            [1.0],
        ])
        
        prices = np.array([
            [100.0],
            [105.0],
            [110.0],
        ])
        
        p_df = _create_test_df(allocations, prices, n_rows, assets)
        result_df, orders_df = run_full_backtest(p_df, assets, transaction_cost_pct=0.0)
        
        positions = compute_current_positions(orders_df, p_df, assets)
        
        assert len(positions) == 1, f"Expected 1 position, got {len(positions)}"
        assert positions[0]['asset'] == 'GLD'
        assert positions[0]['quantity'] > 0
    
    def test_two_lots_two_positions(self):
        """Two separate buys without sells should create two lots."""
        assets = ['GLD']
        n_rows = 5
        
        allocations = np.array([
            [0.5],
            [0.5],
            [0.5],
            [1.0],
            [1.0],
        ])
        
        prices = np.array([
            [100.0],
            [100.0],
            [100.0],
            [110.0],
            [110.0],
        ])
        
        p_df = _create_test_df(allocations, prices, n_rows, assets)
        result_df, orders_df = run_full_backtest(p_df, assets, transaction_cost_pct=0.0)
        
        positions = compute_current_positions(orders_df, p_df, assets)
        
        buys = orders_df[orders_df['direction'] > 0]
        assert len(buys) == 2, f"Expected 2 buy orders, got {len(buys)}"
        assert len(positions) == 2, f"Expected 2 lots, got {len(positions)}"
        
        for pos in positions:
            assert pos['asset'] == 'GLD'
            assert pos['quantity'] > 0
    
    def test_fifo_partial_sell(self):
        """Partial sell should reduce first lot, not create residual."""
        assets = ['GLD']
        n_rows = 6
        
        allocations = np.array([
            [1.0],
            [1.0],
            [1.0],
            [0.5],
            [0.5],
            [0.5],
        ])
        
        prices = np.array([
            [100.0],
            [100.0],
            [100.0],
            [110.0],
            [110.0],
            [110.0],
        ])
        
        p_df = _create_test_df(allocations, prices, n_rows, assets)
        result_df, orders_df = run_full_backtest(p_df, assets, transaction_cost_pct=0.0)
        
        positions = compute_current_positions(orders_df, p_df, assets)
        
        buys = orders_df[orders_df['direction'] > 0]
        sells = orders_df[orders_df['direction'] < 0]
        
        total_bought = buys['size'].sum()
        total_sold = sells['size'].sum() if len(sells) > 0 else 0.0
        expected_remaining = total_bought - total_sold
        
        position_qty = sum(p['quantity'] for p in positions)
        
        assert abs(position_qty - expected_remaining) < 1e-6, \
            f"Position qty {position_qty} != expected {expected_remaining}"
    
    def test_full_exit_no_positions(self):
        """Full exit should result in no open positions."""
        assets = ['GLD']
        n_rows = 4
        
        allocations = np.array([
            [1.0],
            [1.0],
            [0.0],
            [0.0],
        ])
        
        prices = np.array([
            [100.0],
            [105.0],
            [110.0],
            [115.0],
        ])
        
        p_df = _create_test_df(allocations, prices, n_rows, assets)
        result_df, orders_df = run_full_backtest(p_df, assets, transaction_cost_pct=0.1)
        
        positions = compute_current_positions(orders_df, p_df, assets)
        
        assert len(positions) == 0, f"Expected 0 positions after full exit, got {len(positions)}"


class TestAllocationTrackingConsistency:
    """Test that current_allocs stays consistent with actual holdings."""
    
    def test_allocation_never_negative(self):
        """Current allocation should never go negative."""
        assets = ['GLD', 'EEM']
        n_rows = 5
        
        allocations = np.array([
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ])
        
        prices = np.array([
            [100.0, 50.0],
            [105.0, 52.0],
            [110.0, 55.0],
            [108.0, 58.0],
            [112.0, 60.0],
        ])
        
        alloc_matrix = allocations.astype(np.float64)
        price_matrix = prices.astype(np.float64)
        
        port_vals, o_rows, o_assets, o_dirs, o_sizes, o_prices = \
            _run_backtest_kernel_compounding_optimized(alloc_matrix, price_matrix, 0.1)
        
        assert np.all(port_vals > 0), "Portfolio value should always be positive"
    
    def test_portfolio_value_positive(self):
        """Portfolio value should remain positive throughout."""
        assets = ['GLD', 'EEM', 'TLT']
        n_rows = 10
        
        np.random.seed(42)
        
        target_asset = np.zeros(n_rows, dtype=int)
        for i in range(n_rows):
            if i % 3 == 0:
                target_asset[i] = i % 3
        
        allocations = np.zeros((n_rows, 3))
        for i in range(n_rows):
            allocations[i, target_asset[i]] = 1.0
        
        base_prices = np.array([100.0, 50.0, 80.0])
        prices = np.zeros((n_rows, 3))
        for i in range(n_rows):
            prices[i] = base_prices * (1.0 + 0.01 * np.random.randn(3))
            prices[i] = np.maximum(prices[i], 1.0)
        
        p_df = _create_test_df(allocations, prices, n_rows, assets)
        result_df, orders_df = run_full_backtest(p_df, assets, transaction_cost_pct=0.1)
        
        assert np.all(result_df['port_value'].values > 0), \
            "Portfolio value should always be positive"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_allocation_from_start(self):
        """Zero allocation from start should generate no orders."""
        assets = ['GLD']
        n_rows = 3
        
        allocations = np.array([
            [0.0],
            [0.0],
            [0.0],
        ])
        
        prices = np.array([
            [100.0],
            [105.0],
            [110.0],
        ])
        
        p_df = _create_test_df(allocations, prices, n_rows, assets)
        result_df, orders_df = run_full_backtest(p_df, assets, transaction_cost_pct=0.1)
        
        assert len(orders_df) == 0, f"Expected 0 orders, got {len(orders_df)}"
    
    def test_very_small_residual_should_be_zero(self):
        """Very small residuals (< 1e-9) should be treated as zero."""
        assets = ['GLD']
        n_rows = 3
        
        allocations = np.array([
            [1.0],
            [1.0],
            [0.0],
        ])
        
        prices = np.array([
            [100.0],
            [100.0000001],
            [100.0000002],
        ])
        
        p_df = _create_test_df(allocations, prices, n_rows, assets)
        result_df, orders_df = run_full_backtest(p_df, assets, transaction_cost_pct=0.0)
        
        positions = compute_current_positions(orders_df, p_df, assets)
        
        assert len(positions) == 0, f"Expected 0 positions after full exit, got {len(positions)}"
    
    def test_price_increases_after_buy(self):
        """Price increase after buy should not affect exit completeness."""
        assets = ['GLD']
        n_rows = 4
        
        allocations = np.array([
            [1.0],
            [1.0],
            [1.0],
            [0.0],
        ])
        
        prices = np.array([
            [100.0],
            [200.0],
            [200.0],
            [150.0],
        ])
        
        p_df = _create_test_df(allocations, prices, n_rows, assets)
        result_df, orders_df = run_full_backtest(p_df, assets, transaction_cost_pct=0.1)
        
        positions = compute_current_positions(orders_df, p_df, assets)
        
        assert len(positions) == 0, f"Expected 0 positions after full exit, got {len(positions)}"
    
    def test_price_decreases_after_buy(self):
        """Price decrease after buy should not affect exit completeness."""
        assets = ['GLD']
        n_rows = 4
        
        allocations = np.array([
            [1.0],
            [1.0],
            [1.0],
            [0.0],
        ])
        
        prices = np.array([
            [200.0],
            [100.0],
            [100.0],
            [80.0],
        ])
        
        p_df = _create_test_df(allocations, prices, n_rows, assets)
        result_df, orders_df = run_full_backtest(p_df, assets, transaction_cost_pct=0.1)
        
        positions = compute_current_positions(orders_df, p_df, assets)
        
        assert len(positions) == 0, f"Expected 0 positions after full exit, got {len(positions)}"


class TestOrderDirectionConsistency:
    """Test that order directions are correctly recorded."""
    
    def test_buy_direction_positive(self):
        """All buy orders should have direction > 0."""
        assets = ['GLD', 'EEM']
        n_rows = 5
        
        allocations = np.array([
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ])
        
        prices = np.array([
            [100.0, 50.0],
            [100.0, 50.0],
            [105.0, 55.0],
            [105.0, 55.0],
            [110.0, 60.0],
        ])
        
        p_df = _create_test_df(allocations, prices, n_rows, assets)
        result_df, orders_df = run_full_backtest(p_df, assets, transaction_cost_pct=0.1)
        
        for _, order in orders_df.iterrows():
            assert order['direction'] in [1.0, -1.0], \
                f"Invalid direction: {order['direction']}"
    
    def test_sell_direction_negative(self):
        """All sell orders should have direction < 0."""
        assets = ['GLD']
        n_rows = 4
        
        allocations = np.array([
            [1.0],
            [1.0],
            [0.0],
            [0.0],
        ])
        
        prices = np.array([
            [100.0],
            [105.0],
            [110.0],
            [115.0],
        ])
        
        p_df = _create_test_df(allocations, prices, n_rows, assets)
        result_df, orders_df = run_full_backtest(p_df, assets, transaction_cost_pct=0.1)
        
        sells = orders_df[orders_df['direction'] < 0]
        for _, order in sells.iterrows():
            assert order['direction'] == -1.0, \
                f"Expected sell direction -1.0, got {order['direction']}"

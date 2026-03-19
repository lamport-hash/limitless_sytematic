import numpy as np
from numba import njit
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
from numba import njit
from typing import Tuple


warnings.filterwarnings('ignore')

# Your OHLC column mapping

OHLC_COLS = {
    "open": "S_open_f32",
    "high": "S_high_f32",
    "low": "S_low_f32",
    "close": "S_close_f32",
    "volume": "S_volume_f64",
}

@dataclass
class Order:
    timestamp: int
    etf: str
    direction: float
    size: float
    price: float
    allocation: float

def compute_order_pnl(orders_df: pd.DataFrame, current_prices: dict) -> pd.DataFrame:
    """
    Computes the unrealized/realized PnL % for each order based on current prices.
    """
    df = orders_df.copy()
    # PnL % = (Current Price - Entry Price) / Entry Price * Direction
    df['pnl_pct'] = df.apply(
        lambda x: ((current_prices[x['etf']] - x['price']) / x['price']) * x['direction'], 
        axis=1
    )
    return df
    
@njit
def _generate_orders_numba(alloc_matrix, price_matrix, epsilon=1e-9):
    """
    Core JIT-compiled logic for row-by-row rebalancing.
    Returns: arrays of (row_idx, asset_idx, direction, size, price, target_alloc)
    """
    n_rows, n_assets = alloc_matrix.shape
    current_allocs = np.zeros(n_assets)
    
    # Pre-allocate large arrays to store results (we'll trim them later)
    # Max possible orders is n_rows * n_assets
    max_orders = n_rows * n_assets
    out_row_idx = np.zeros(max_orders, dtype=np.int64)
    out_asset_idx = np.zeros(max_orders, dtype=np.int64)
    out_direction = np.zeros(max_orders, dtype=np.float64)
    out_size = np.zeros(max_orders, dtype=np.float64)
    out_price = np.zeros(max_orders, dtype=np.float64)
    out_alloc = np.zeros(max_orders, dtype=np.float64)
    
    order_count = 0
    portfolio_value = 1.0

    for i in range(n_rows):
        for j in range(n_assets):
            target_val = alloc_matrix[i, j]
            current_val = current_allocs[j]
            diff = target_val - current_val

            if abs(diff) > epsilon:
                price = price_matrix[i, j]
                if price > 0:
                    out_row_idx[order_count] = i
                    out_asset_idx[order_count] = j
                    out_direction[order_count] = 1.0 if diff > 0 else -1.0
                    out_size[order_count] = abs(diff) * portfolio_value / price
                    out_price[order_count] = price
                    out_alloc[order_count] = target_val
                    
                    order_count += 1
                    current_allocs[j] = target_val

    return (out_row_idx[:order_count], out_asset_idx[:order_count], 
            out_direction[:order_count], out_size[:order_count], 
            out_price[:order_count], out_alloc[:order_count])

def generate_orders_fast(p_df, p_asset_list):
    # Prepare NumPy arrays
    alloc_cols = [f"A_{etf}_alloc" for etf in p_asset_list]
    price_cols = [f"{etf}_close" for etf in p_asset_list] # Simplified naming
    
    alloc_matrix = p_df[alloc_cols].values
    price_matrix = p_df[price_cols].values
    timestamps = p_df.index.values

    # Run Numba
    res = _generate_orders_numba(alloc_matrix, price_matrix)
    
    # Reconstruct DataFrame
    orders_df = pd.DataFrame({
        "timestamp": timestamps[res[0]],
        "etf": [p_asset_list[j] for j in res[1]],
        "direction": res[2],
        "size": res[3],
        "price": res[4],
        "allocation": res[5]
    })
    
    return orders_df
    
@njit
def _compute_pnl_numba(price_matrix, order_row_idx, order_asset_idx, order_size, order_dir):
    n_rows, n_assets = price_matrix.shape
    port_values = np.zeros(n_rows)
    shares = np.zeros(n_assets)
    cash = 1.0
    
    order_ptr = 0
    n_orders = len(order_row_idx)

    for i in range(n_rows):
        # Process all orders that happen at this timestamp
        while order_ptr < n_orders and order_row_idx[order_ptr] == i:
            a_idx = order_asset_idx[order_ptr]
            o_size = order_size[order_ptr]
            o_dir = order_dir[order_ptr]
            o_price = price_matrix[i, a_idx]
            
            trade_val = o_size * o_price * o_dir
            cash -= trade_val
            shares[a_idx] += (o_size * o_dir)
            order_ptr += 1
        
        # Mark to market
        mkt_val = 0.0
        for j in range(n_assets):
            mkt_val += shares[j] * price_matrix[i, j]
        
        port_values[i] = cash + mkt_val
        
    return port_values




@njit
def _run_backtest_kernel_compounding_optimized(alloc_matrix, price_matrix, transaction_cost_pct=0.0, epsilon=1e-9):
    n_rows, n_assets = alloc_matrix.shape
    current_allocs = np.zeros(n_assets)
    shares_held = np.zeros(n_assets)
    portfolio_values = np.ones(n_rows)
    cash = 100.0 
    
    max_orders = n_rows * n_assets
    out_row_idx = np.zeros(max_orders, dtype=np.int64)
    out_asset_idx = np.zeros(max_orders, dtype=np.int64)
    out_dir = np.zeros(max_orders)
    out_size = np.zeros(max_orders)
    out_price = np.zeros(max_orders)
    
    order_count = 0

    for i in range(n_rows):
        # 1. Calculate Mark-to-Market at current prices
        mkt_val = 0.0
        for j in range(n_assets):
            mkt_val += shares_held[j] * price_matrix[i, j]
        total_val = cash + mkt_val
        
        # PASS 1: PROCESS SELLS FIRST (Increases Cash)
        for j in range(n_assets):
            target = alloc_matrix[i, j]
            diff = target - current_allocs[j]
            
            if diff < -epsilon:  # Negative diff means we are selling
                px = price_matrix[i, j]
                if px > 0 and shares_held[j] > epsilon:
                    if abs(target) < epsilon:
                        sell_size = shares_held[j]
                    elif abs(target) < shares_held[j] * 0.04:
                        sell_size = shares_held[j]    
                    else:
                        sell_size = (abs(diff) * total_val) / px
                    
                    trade_value = sell_size * px
                    cost = trade_value * (transaction_cost_pct / 100.0)
                    cash += (trade_value - cost)
                    shares_held[j] -= sell_size
                    
                    # Record Order
                    out_row_idx[order_count] = i
                    out_asset_idx[order_count] = j
                    out_dir[order_count] = -1.0
                    out_size[order_count] = sell_size
                    out_price[order_count] = px
                    order_count += 1
                    
                    final_mkt_val_after_sell = 0.0
                    for k in range(n_assets):
                        final_mkt_val_after_sell += shares_held[k] * price_matrix[i, k]
                    total_val_after = cash + final_mkt_val_after_sell
                    if total_val_after > epsilon:
                        current_allocs[j] = (shares_held[j] * px) / total_val_after
                    else:
                        current_allocs[j] = 0.0

        # PASS 2: PROCESS BUYS SECOND (Uses refreshed Cash)
        for j in range(n_assets):
            target = alloc_matrix[i, j]
            diff = target - current_allocs[j]
            
            if diff > epsilon:  # Positive diff means we are buying
                px = price_matrix[i, j]
                if px > 0 and shares_held[j] < -epsilon:
                    if abs(target) < epsilon:
                        buy_size = abs(shares_held[j])
                    elif abs(target) < abs(shares_held[j]) * 0.04:
                        buy_size = abs(shares_held[j])
                    else:
                        buy_size = (diff * total_val) / px
                elif px > 0:
                    buy_size = (diff * total_val) / px
                else:
                    buy_size = 0.0
                    
                if buy_size > epsilon:
                    trade_value = buy_size * px
                    cost = trade_value * (transaction_cost_pct / 100.0)
                    
                    cash -= (trade_value + cost)
                    shares_held[j] += buy_size
                    
                    # Record Order
                    out_row_idx[order_count] = i
                    out_asset_idx[order_count] = j
                    out_dir[order_count] = 1.0
                    out_size[order_count] = buy_size
                    out_price[order_count] = px
                    order_count += 1
                    
                    current_allocs[j] = target

        # PASS 3: RECALCULATE current_allocs for ALL assets based on actual positions
        # This fixes allocation drift bug - ensures tracked allocations match reality
        final_mkt_val = 0.0
        for j in range(n_assets):
            final_mkt_val += shares_held[j] * price_matrix[i, j]
        total_val_after_trades = cash + final_mkt_val
        
        if total_val_after_trades > epsilon:
            for j in range(n_assets):
                px = price_matrix[i, j]
                if px > epsilon:
                    current_allocs[j] = (shares_held[j] * px) / total_val_after_trades
                else:
                    current_allocs[j] = 0.0
        else:
            for j in range(n_assets):
                current_allocs[j] = 0.0

        # Final MTM for the candle
        portfolio_values[i] = total_val_after_trades

    return (portfolio_values, 
            out_row_idx[:order_count], out_asset_idx[:order_count], 
            out_dir[:order_count], out_size[:order_count], out_price[:order_count])
@njit
def _run_backtest_kernel_compounding_optimized(
    alloc_matrix,
    price_matrix,
    transaction_cost_pct=0.0,
    epsilon=1e-9
):
    n_rows, n_assets = alloc_matrix.shape

    current_allocs = np.zeros(n_assets)
    shares_held = np.zeros(n_assets)
    prev_targets = np.zeros(n_assets)

    portfolio_values = np.ones(n_rows)
    cash = 100.0

    max_orders = n_rows * n_assets
    out_row_idx = np.zeros(max_orders, dtype=np.int64)
    out_asset_idx = np.zeros(max_orders, dtype=np.int64)
    out_dir = np.zeros(max_orders)
    out_size = np.zeros(max_orders)
    out_price = np.zeros(max_orders)

    order_count = 0

    for i in range(n_rows):

        # ===== MARK-TO-MARKET =====
        mkt_val = 0.0
        for j in range(n_assets):
            mkt_val += shares_held[j] * price_matrix[i, j]
        total_val = cash + mkt_val

        # ============================================================
        # PASS 1: CLOSE / REDUCE POSITIONS
        # ============================================================
        for j in range(n_assets):

            # 🚫 SKIP if allocation unchanged
            if i > 0 and abs(alloc_matrix[i, j] - prev_targets[j]) < epsilon:
                continue

            target = alloc_matrix[i, j]
            px = price_matrix[i, j]

            if px <= 0:
                continue

            current_value = shares_held[j] * px
            target_value = target * total_val
            diff_value = target_value - current_value

            if diff_value < -epsilon:

                reduce_value = -diff_value
                reduce_size = reduce_value / px

                if shares_held[j] > 0:
                    # SELL LONG
                    reduce_size = min(reduce_size, shares_held[j])
                    if reduce_size <= epsilon:
                        continue

                    shares_held[j] -= reduce_size
                    trade_value = reduce_size * px
                    cost = trade_value * (transaction_cost_pct / 100.0)
                    cash += (trade_value - cost)

                    direction = -1.0

                elif shares_held[j] < 0:
                    # COVER SHORT
                    reduce_size = min(reduce_size, -shares_held[j])
                    if reduce_size <= epsilon:
                        continue

                    shares_held[j] += reduce_size
                    trade_value = reduce_size * px
                    cost = trade_value * (transaction_cost_pct / 100.0)
                    cash -= (trade_value + cost)

                    direction = 1.0
                else:
                    continue

                out_row_idx[order_count] = i
                out_asset_idx[order_count] = j
                out_dir[order_count] = direction
                out_size[order_count] = reduce_size
                out_price[order_count] = px
                order_count += 1

        # ============================================================
        # RECOMPUTE AFTER SELLS
        # ============================================================
        mkt_val_after = 0.0
        for j in range(n_assets):
            mkt_val_after += shares_held[j] * price_matrix[i, j]
        total_val_after = cash + mkt_val_after

        # ============================================================
        # PASS 2: OPEN / INCREASE POSITIONS
        # ============================================================
        for j in range(n_assets):

            # 🚫 SKIP if allocation unchanged
            if i > 0 and abs(alloc_matrix[i, j] - prev_targets[j]) < epsilon:
                continue

            target = alloc_matrix[i, j]
            px = price_matrix[i, j]

            if px <= 0:
                continue

            current_value = shares_held[j] * px
            target_value = target * total_val_after
            diff_value = target_value - current_value

            if diff_value > epsilon:

                # =========================
                # LONG SIDE
                # =========================
                if target >= 0:
                    invest_value = min(diff_value, cash)
                    buy_size = invest_value / px

                    if buy_size <= epsilon:
                        continue

                    trade_value = buy_size * px
                    cost = trade_value * (transaction_cost_pct / 100.0)

                    cash -= (trade_value + cost)
                    shares_held[j] += buy_size

                    direction = 1.0

                # =========================
                # SHORT SIDE
                # =========================
                else:
                    short_value = diff_value
                    short_size = short_value / px

                    if short_size <= epsilon:
                        continue

                    trade_value = short_size * px
                    cost = trade_value * (transaction_cost_pct / 100.0)

                    cash += (trade_value - cost)
                    shares_held[j] -= short_size

                    direction = -1.0

                out_row_idx[order_count] = i
                out_asset_idx[order_count] = j
                out_dir[order_count] = direction
                out_size[order_count] = abs(buy_size if target >= 0 else short_size)
                out_price[order_count] = px
                order_count += 1

        # Prevent float drift
        if cash < 0 and cash > -1e-6:
            cash = 0.0

        # ============================================================
        # FINAL MARK-TO-MARKET
        # ============================================================
        final_mkt_val = 0.0
        for j in range(n_assets):
            final_mkt_val += shares_held[j] * price_matrix[i, j]

        total_val_final = cash + final_mkt_val
        portfolio_values[i] = total_val_final

        if total_val_final > epsilon:
            for j in range(n_assets):
                px = price_matrix[i, j]
                if px > epsilon:
                    current_allocs[j] = (shares_held[j] * px) / total_val_final
                else:
                    current_allocs[j] = 0.0
        else:
            for j in range(n_assets):
                current_allocs[j] = 0.0

        # ✅ UPDATE PREVIOUS TARGETS
        for j in range(n_assets):
            prev_targets[j] = alloc_matrix[i, j]

    return (
        portfolio_values,
        out_row_idx[:order_count],
        out_asset_idx[:order_count],
        out_dir[:order_count],
        out_size[:order_count],
        out_price[:order_count],
    )

def run_full_backtest(p_df: pd.DataFrame, p_asset_list: list, transaction_cost_pct: float = 0.0):
    """
    Python wrapper to prepare data and format results.
    """
    alloc_cols = [f"A_{etf}_alloc" for etf in p_asset_list]
    price_cols = [f"{etf}_S_close_f32" for etf in p_asset_list]
    
    # Extract matrices
    alloc_matrix = p_df[alloc_cols].to_numpy()
    price_matrix = p_df[price_cols].to_numpy()
    timestamps = p_df.index.to_numpy()

    # Run the Numba engine
    port_vals, o_rows, o_assets, o_dirs, o_sizes, o_prices = _run_backtest_kernel_compounding_optimized(
        alloc_matrix, price_matrix, transaction_cost_pct
    )

    # 1. Create Portfolio Results DataFrame
    p_df['port_value'] = port_vals
    p_df['cum_pnl_pct'] = (p_df['port_value'] / port_vals[0]) - 1

    # 2. Create Orders DataFrame
    orders_df = pd.DataFrame({
        "timestamp": timestamps[o_rows],
        "etf": [p_asset_list[j] for j in o_assets],
        "direction": o_dirs,
        "size": o_sizes,
        "price": o_prices
    })

    return p_df, orders_df


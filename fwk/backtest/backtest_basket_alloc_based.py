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
                    if target < epsilon:
                        sell_size = shares_held[j]
                    else:
                        sell_size = (abs(diff) * total_val) / px
                        sell_size = min(sell_size, shares_held[j])
                    
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
                if px > 0:
                    buy_size = (diff * total_val) / px
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

        # Final MTM for the candle
        final_mkt_val = 0.0
        for j in range(n_assets):
            final_mkt_val += shares_held[j] * price_matrix[i, j]
        portfolio_values[i] = cash + final_mkt_val

    return (portfolio_values, 
            out_row_idx[:order_count], out_asset_idx[:order_count], 
            out_dir[:order_count], out_size[:order_count], out_price[:order_count])

@njit
def _run_backtest_kernel_with_delay(alloc_matrix, price_matrix, delay_n=1, epsilon=1e-9):
    n_rows, n_assets = alloc_matrix.shape
    current_allocs = np.zeros(n_assets)
    shares_held = np.zeros(n_assets)
    portfolio_values = np.ones(n_rows)
    
    immediate_cash = 1.0  # Cash available to spend right now
    # Queue to hold cash from sales: [amount, release_index]
    pending_cash_val = np.zeros(n_rows) 
    pending_cash_time = np.zeros(n_rows, dtype=np.int64)
    pending_ptr_in = 0
    pending_ptr_out = 0
    
    max_orders = n_rows * n_assets
    out_row_idx = np.zeros(max_orders, dtype=np.int64)
    out_asset_idx = np.zeros(max_orders, dtype=np.int64)
    out_dir = np.zeros(max_orders)
    out_size = np.zeros(max_orders)
    out_price = np.zeros(max_orders)
    order_count = 0

    for i in range(n_rows):
        # 1. Release pending cash that has finished its 'n' candle delay
        while pending_ptr_out < pending_ptr_in:
            if pending_cash_time[pending_ptr_out] <= i:
                immediate_cash += pending_cash_val[pending_ptr_out]
                pending_ptr_out += 1
            else:
                break

        # 2. Calculate Total Value (Immediate Cash + Pending Cash + Holdings)
        # We use Total Value to calculate 'Target Dollar Amount' for compounding
        sum_pending = 0.0
        for p in range(pending_ptr_out, pending_ptr_in):
            sum_pending += pending_cash_val[p]
            
        mkt_val = 0.0
        for j in range(n_assets):
            mkt_val += shares_held[j] * price_matrix[i, j]
        
        total_val = immediate_cash + sum_pending + mkt_val

        # PASS 1: SELLS (Cash goes into the pending queue)
        for j in range(n_assets):
            target = alloc_matrix[i, j]
            diff = target - current_allocs[j]
            if diff < -epsilon:
                px = price_matrix[i, j]
                if px > 0:
                    sell_size = min((abs(diff) * total_val) / px, shares_held[j])
                    sale_proceeds = sell_size * px
                    
                    # Instead of immediate_cash, add to pending queue
                    pending_cash_val[pending_ptr_in] = sale_proceeds
                    pending_cash_time[pending_ptr_in] = i + delay_n
                    pending_ptr_in += 1
                    
                    shares_held[j] -= sell_size
                    current_allocs[j] = target
                    
                    # Record Order
                    out_row_idx[order_count], out_asset_idx[order_count] = i, j
                    out_dir[order_count], out_size[order_count], out_price[order_count] = -1.0, sell_size, px
                    order_count += 1

        # PASS 2: BUYS (Only uses immediate_cash)
        for j in range(n_assets):
            target = alloc_matrix[i, j]
            diff = target - current_allocs[j]
            if diff > epsilon:
                px = price_matrix[i, j]
                if px > 0:
                    desired_buy_val = diff * total_val
                    # Constrain by actually available cash
                    actual_buy_val = min(desired_buy_val, immediate_cash)
                    
                    if actual_buy_val > 0:
                        buy_size = actual_buy_val / px
                        immediate_cash -= actual_buy_val
                        shares_held[j] += buy_size
                        current_allocs[j] = target # Note: if cash was insufficient, target isn't fully met
                        
                        out_row_idx[order_count], out_asset_idx[order_count] = i, j
                        out_dir[order_count], out_size[order_count], out_price[order_count] = 1.0, buy_size, px
                        order_count += 1

        portfolio_values[i] = immediate_cash + sum_pending + mkt_val

    return portfolio_values, out_row_idx[:order_count], out_asset_idx[:order_count], out_dir[:order_count], out_size[:order_count], out_price[:order_count]
    

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


import numpy as np
import pandas as pd
from numba import jit
from typing import Tuple, Dict

@jit(nopython=True)
def compute_trade_pnl_numba(entry_prices, exit_prices, directions, entry_idxs, exit_idxs, n_bars):
    """
    Numba-accelerated core computation.
    
    Args:
        entry_prices: array of entry prices
        exit_prices: array of exit prices
        directions: array of trade directions (1 for long, -1 for short)
        entry_idxs: array of entry bar indices
        exit_idxs: array of exit bar indices
        n_bars: total number of bars
        
    Returns:
        Tuple of (trade_pnls array, bar_pnl array)
    """
    n_trades = len(entry_prices)
    trade_pnls = np.zeros(n_trades)
    bar_pnl = np.zeros(n_bars)
    
    for i in range(n_trades):
        # Calculate trade P&L
        if directions[i] == 1:  # Long
            trade_pnls[i] = (exit_prices[i] - entry_prices[i]) / entry_prices[i]
        else:  # Short
            trade_pnls[i] = (entry_prices[i] - exit_prices[i]) / entry_prices[i]
        
        # Distribute P&L across bars (if multiple bars)
        entry_idx = entry_idxs[i]
        exit_idx = exit_idxs[i]
        
        if exit_idx > entry_idx:
            pnl_per_bar = trade_pnls[i] / (exit_idx - entry_idx + 1)
            for j in range(entry_idx, exit_idx + 1):
                bar_pnl[j] += pnl_per_bar
        else:
            bar_pnl[entry_idx] += trade_pnls[i]
    
    return trade_pnls, bar_pnl


@jit(nopython=True)
def compute_portfolio_value_numba(bar_pnl, initial_value=1.0):
    """
    Compute portfolio value by compounding returns.
    This matches your original implementation.
    """
    n_bars = len(bar_pnl)
    portfolio_value = np.zeros(n_bars)
    cum_pnl = np.zeros(n_bars)
    
    current_value = initial_value
    cum_return = 0.0
    
    for i in range(n_bars):
        # Same logic as your original: portfolio_value *= (1 + trade_pnl)
        # but distributed across bars
        current_value *= (1 + bar_pnl[i])
        cum_return = cum_return + bar_pnl[i]  # This matches your cum_pnl calculation
        
        portfolio_value[i] = current_value
        cum_pnl[i] = cum_return
    
    return portfolio_value, cum_pnl

@jit(nopython=True)
def compute_metrics_numba(returns, bar_pnl, portfolio_values):
    """
    Compute performance metrics.
    """
    n_trades = len(returns)
    n_bars = len(portfolio_values)
    
    # Filter zero returns
    non_zero_mask = returns != 0
    returns_nonzero = returns[non_zero_mask]
    n_nonzero = len(returns_nonzero)
    
    # Win/Loss metrics
    win_rate = 0.0
    avg_win = 0.0
    avg_loss = 0.0
    profit_factor = 0.0
    
    if n_nonzero > 0:
        wins = returns_nonzero[returns_nonzero > 0]
        losses = returns_nonzero[returns_nonzero < 0]
        
        n_wins = len(wins)
        n_losses = len(losses)
        
        win_rate = n_wins / n_nonzero if n_nonzero > 0 else 0
        avg_win = np.mean(wins) if n_wins > 0 else 0
        avg_loss = np.mean(losses) if n_losses > 0 else 0
        
        gross_profit = np.sum(wins)
        gross_loss = abs(np.sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    
    # Max Drawdown (using portfolio values)
    rolling_max = np.zeros(n_bars)
    current_max = portfolio_values[0]
    max_dd = 0.0
    
    for i in range(n_bars):
        if portfolio_values[i] > current_max:
            current_max = portfolio_values[i]
        rolling_max[i] = current_max
        dd = (portfolio_values[i] - current_max) / current_max
        if dd < max_dd:
            max_dd = dd
    
    # Total return
    total_return = portfolio_values[-1] - 1.0
    
    return (n_trades, win_rate, avg_win, avg_loss, profit_factor, max_dd, total_return)

def compute_strategy_performance_simple(
    p_df: pd.DataFrame,
    p_orders_df: pd.DataFrame,
    etf_list: list,
    close_col_map: dict = None
) -> Tuple[Dict, pd.DataFrame]:
    """
    Simplified strategy performance computation with numba acceleration.
    
    Args:
        p_df: DataFrame with OHLC data
        p_orders_df: Orders DataFrame from generate_orders_from_allocations
        etf_list: List of ETF tickers
        close_col_map: Optional mapping of ETF to close column names
        
    Returns:
        Tuple of (metrics_dict, result_df with bar-level P&L)
    """
    OHLC_COLS = {
    "open": "S_open_f32",
    "high": "S_high_f32",
    "low": "S_low_f32",
    "close": "S_close_f32",
    "volume": "S_volume_f64",
    }
    # Create close column mapping if not provided
    if close_col_map is None:
        close_col_map = {etf: f"{etf}_{OHLC_COLS['close']}" for etf in etf_list}
    
    # Prepare data structures for numba
    p_df = p_df.copy()
    timestamp_to_idx = {ts: i for i, ts in enumerate(p_df.index)}
    n_bars = len(p_df)
    
    # Track positions
    positions = {etf: {"entry_price": None, "entry_idx": None} for etf in etf_list}
    
    # Collect trade data
    entry_prices = []
    exit_prices = []
    directions = []
    entry_idxs = []
    exit_idxs = []
    
    # Process orders
    p_orders_sorted = p_orders_df.sort_values("timestamp")
    
    for _, order in p_orders_sorted.iterrows():
        ts = order["timestamp"]
        etf = order["etf"]
        direction = order["direction"]
        
        if ts not in p_df.index:
            continue
            
        close_col = close_col_map[etf]
        current_price = p_df.loc[ts, close_col]
        pos = positions[etf]
        
        if direction == 1.0:  # Entry
            pos["entry_price"] = current_price
            pos["entry_idx"] = ts
            
        elif direction == -1.0:  # Exit
            if pos["entry_price"] is not None and pos["entry_idx"] is not None:
                entry_prices.append(pos["entry_price"])
                exit_prices.append(current_price)
                directions.append(1.0)  # Assuming long only from your original code
                entry_idxs.append(timestamp_to_idx[pos["entry_idx"]])
                exit_idxs.append(timestamp_to_idx[ts])
                
                pos["entry_price"] = None
                pos["entry_idx"] = None
    
    # Convert to numpy arrays
    entry_prices = np.array(entry_prices)
    exit_prices = np.array(exit_prices)
    directions = np.array(directions)
    entry_idxs = np.array(entry_idxs)
    exit_idxs = np.array(exit_idxs)
    
    # Compute P&L with numba
    if len(entry_prices) > 0:
        trade_pnls, bar_pnl = compute_trade_pnl_numba(
            entry_prices, exit_prices, directions, entry_idxs, exit_idxs, n_bars
        )
    else:
        trade_pnls = np.array([])
        bar_pnl = np.zeros(n_bars)
    
    # Calculate portfolio value
    p_df["pnl_per_bar"] = bar_pnl
    p_df["cum_pnl"] = np.cumsum(bar_pnl)
    p_df["portfolio_value"] = 1.0 + p_df["cum_pnl"]
    
    portfolio_values = p_df["portfolio_value"].values
    
    # Compute metrics with numba
    n_trades, win_rate, avg_win, avg_loss, profit_factor, max_dd, total_return = compute_metrics_numba(
        trade_pnls, bar_pnl, portfolio_values
    )
    
    # Create metrics dictionary
    metrics = {
        "n_trades": n_trades,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "max_drawdown": max_dd,
        "total_return": total_return,
        "final_value": portfolio_values[-1],
    }
    
    return metrics, p_df[["pnl_per_bar", "cum_pnl", "portfolio_value"]]
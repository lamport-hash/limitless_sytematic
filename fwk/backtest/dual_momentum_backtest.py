"""
Dual Momentum Strategy Backtest.

Runs backtest on dual momentum allocations, computing performance metrics
and order generation.

Takes the output of compute_dual_momentum() as input (DataFrame with A_*_alloc columns).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd



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


@dataclass
class PerformanceMetrics:
    total_return: float
    total_return_pct: float
    pnl_per_bar: float
    pnl_per_month: float
    cum_pnl_final: float
    final_portfolio_value: float
    max_drawdown: float
    max_dd: float
    max_dd_pct: float
    sharpe_ratio: float
    calmar_ratio: float
    profit_factor: float
    win_rate: float
    n_bars: int
    n_positive_bars: int
    n_negative_bars: int
    avg_win: float
    avg_loss: float
    avg_win_loss_ratio: float
    n_trades: int



def generate_orders_from_allocations(
    p_df: pd.DataFrame,
    p_asset_list=['QQQ', 'SPY', 'TLT', 'GLD', 'VWO']
) -> Tuple[pd.DataFrame, List[Order]]:
    # Don't copy the entire DataFrame - work with references
    # df = p_df.copy()  # REMOVE THIS
    
    alloc_cols = [f"A_{etf}_alloc" for etf in p_asset_list]
    close_cols = [f"{etf}_{OHLC_COLS['close']}" for etf in p_asset_list]
    
    # Extract numpy arrays for vectorized operations
    alloc_data = p_df[alloc_cols].values
    close_data = p_df[close_cols].values
    timestamps = p_df.index.values
    
    orders = []
    current_allocs = np.zeros(len(p_asset_list))
    portfolio_value = 1.0
    epsilon = 1e-9
    
    n_assets = len(p_asset_list)
    
    # Pre-calculate which assets changed at each step
    # This avoids the inner loop for most cases
    for i in range(len(p_df)):
        target_vals = alloc_data[i]
        
        # Find assets that changed significantly
        diff_mask = np.abs(target_vals - current_allocs) > epsilon
        changed_indices = np.where(diff_mask)[0]
        
        if len(changed_indices) == 0:
            continue
        
        # Process only changed assets
        for j in changed_indices:
            target_val = target_vals[j]
            current_val = current_allocs[j]
            diff = target_val - current_val
            
            price = close_data[i, j]
            
            if price > 0:
                orders.append(Order(
                    timestamp=int(timestamps[i]),
                    etf=p_asset_list[j],
                    direction=1.0 if diff > 0 else -1.0,
                    size=abs(diff) * portfolio_value / price,
                    price=price,
                    allocation=target_val,
                ))
            
            current_allocs[j] = target_val
    
    # Create DataFrame more efficiently
    if orders:
        # Use list comprehension for better performance
        orders_data = [
            (o.timestamp, o.etf, o.direction, o.size, o.price, o.allocation)
            for o in orders
        ]
        orders_df = pd.DataFrame(
            orders_data,
            columns=['timestamp', 'etf', 'direction', 'size', 'price', 'allocation']
        )
    else:
        orders_df = pd.DataFrame(columns=['timestamp', 'etf', 'direction', 'size', 'price', 'allocation'])
    
    return orders_df, orders
    
def generate_orders_from_allocations_vectorized(
    p_df: pd.DataFrame,
    p_asset_list=['QQQ', 'SPY', 'TLT', 'GLD', 'VWO']
) -> pd.DataFrame:
    """
    Fully vectorized version - much faster but doesn't create Order objects
    """
    alloc_cols = [f"A_{etf}_alloc" for etf in p_asset_list]
    close_cols = [f"{etf}_{OHLC_COLS['close']}" for etf in p_asset_list]
    
    # Get the allocation changes
    alloc_data = p_df[alloc_cols].values
    alloc_changes = np.diff(alloc_data, axis=0, prepend=alloc_data[0:1])
    
    # Find where significant changes occur
    epsilon = 1e-9
    significant_changes = np.abs(alloc_changes) > epsilon
    
    # Get indices of changes
    rows, cols = np.where(significant_changes)
    
    if len(rows) == 0:
        return pd.DataFrame()
    
    # Extract data for all changes at once
    timestamps = p_df.index.values[rows]
    etfs = [p_asset_list[col] for col in cols]
    directions = np.where(alloc_changes[rows, cols] > 0, 1.0, -1.0)
    sizes = np.abs(alloc_changes[rows, cols])
    prices = close_data[rows, cols]
    allocations = alloc_data[rows, cols]
    
    # Filter out zero prices
    valid_prices = prices > 0
    
    # Create DataFrame directly
    orders_df = pd.DataFrame({
        'timestamp': timestamps[valid_prices],
        'etf': [etfs[i] for i in range(len(etfs)) if valid_prices[i]],
        'direction': directions[valid_prices],
        'size': sizes[valid_prices],
        'price': prices[valid_prices],
        'allocation': allocations[valid_prices]
    })
    
    return orders_df


def compute_strategy_performance(
    p_df: pd.DataFrame,
    p_orders_df: pd.DataFrame,
    p_asset_list = ['QQQ', 'SPY', 'TLT', 'GLD', 'VWO']
) -> Tuple[Dict, pd.DataFrame]:
    """
    Compute strategy performance from trade-by-trade P&L.

    Args:
        p_df: DataFrame with A_*_alloc columns and OHLC data
        p_orders_df: Orders DataFrame from generate_orders_from_allocations

    Returns:
        Tuple of (metrics_dict, result_df)
    """
    close_col_map = {etf: f"{etf}_{OHLC_COLS['close']}" for etf in p_asset_list}

    positions = {etf: {"entry_price": None, "entry_idx": None} for etf in p_asset_list}
    pnl_list = []
    portfolio_value = 1.0
    bar_pnl = np.zeros(len(p_df))
    p_df = p_df.copy()

    timestamp_to_idx = {ts: i for i, ts in enumerate(p_df.index)}
    p_orders_sorted = p_orders_df.sort_values("timestamp")

    for _, order in p_orders_sorted.iterrows():
        ts = order["timestamp"]
        etf = order["etf"]
        direction = order["direction"]
        close_col = close_col_map[etf]

        if ts not in p_df.index:
            continue
        current_price = p_df.loc[ts, close_col]

        pos = positions[etf]

        if direction == 1.0:
            pos["entry_price"] = current_price
            pos["entry_idx"] = ts

        elif direction == -1.0:
            if pos["entry_price"] is not None and pos["entry_idx"] is not None:
                trade_pnl = (current_price - pos["entry_price"]) / pos["entry_price"]
                portfolio_value *= (1 + trade_pnl)
                entry_idx = timestamp_to_idx[pos["entry_idx"]]
                exit_idx = timestamp_to_idx[ts]
                bar_pnl[entry_idx:exit_idx + 1] += trade_pnl / (exit_idx - entry_idx + 1)
                pnl_list.append(
                    {
                        "entry_time": pos["entry_idx"],
                        "exit_time": ts,
                        "etf": etf,
                        "entry_price": pos["entry_price"],
                        "exit_price": current_price,
                        "pnl": trade_pnl,
                    }
                )
                pos["entry_price"] = None
                pos["entry_idx"] = None

    p_df["pnl_per_bar"] = bar_pnl
    p_df["cum_pnl"] = np.cumsum(bar_pnl)
    p_df["portfolio_value"] = 1.0 + p_df["cum_pnl"]

    if pnl_list:
        pnl_df = pd.DataFrame(pnl_list)
        returns = pnl_df["pnl"].values
    else:
        returns = np.array([0.0])

    total_pnl = portfolio_value - 1.0

    returns_nonzero = returns[returns != 0]

    portfolio_values = p_df["portfolio_value"].values
    rolling_max = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - rolling_max) / rolling_max
    max_dd = drawdown.min()

    positive_returns = returns_nonzero[returns_nonzero > 0]
    negative_returns = returns_nonzero[returns_nonzero < 0]

    n_bars = len(p_df)
    n_months = n_bars / (30 * 24)

    avg_pnl_per_bar = np.mean(bar_pnl[bar_pnl != 0]) if np.any(bar_pnl != 0) else 0

    sharpe = 0.0
    if np.std(returns_nonzero) > 0 and len(returns_nonzero) > 1:
        annualized_return = np.mean(returns_nonzero) * 252 * 24
        annualized_vol = np.std(returns_nonzero) * np.sqrt(252 * 24)
        sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0

    calmar = 0.0
    if max_dd < 0 and n_bars > 0 and portfolio_values[0] > 0 and portfolio_values[-1] > 0:
        annualized_return = (portfolio_values[-1] / portfolio_values[0]) ** (
            252 * 24 / n_bars
        ) - 1
        calmar = annualized_return / abs(max_dd)

    profit_factor = 0.0
    gross_profit = np.sum(positive_returns) if len(positive_returns) > 0 else 0
    gross_loss = abs(np.sum(negative_returns)) if len(negative_returns) > 0 else 0
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss

    n_trades = len(p_orders_df)
    win_rate = (
        len(positive_returns) / len(returns_nonzero) if len(returns_nonzero) > 0 else 0
    )
    avg_win = np.mean(positive_returns) if len(positive_returns) > 0 else 0
    avg_loss = np.mean(negative_returns) if len(negative_returns) > 0 else 0

    metrics = {
        "total_return": total_pnl,
        "total_return_pct": total_pnl * 100,
        "pnl_per_bar": avg_pnl_per_bar,
        "pnl_per_month": total_pnl / n_months if n_months > 0 else 0,
        "cum_pnl_final": p_df["cum_pnl"].iloc[-1],
        "final_portfolio_value": portfolio_values[-1],
        "max_drawdown": max_dd,
        "max_dd_pct": max_dd * 100,
        "sharpe_ratio": sharpe,
        "calmar_ratio": calmar,
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "n_bars": n_bars,
        "n_positive_bars": len(positive_returns),
        "n_negative_bars": len(negative_returns),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_win_loss_ratio": abs(avg_win / avg_loss) if avg_loss != 0 else 0,
        "n_trades": n_trades,
    }

    return metrics, p_df


def save_backtest_diagnostics(
    p_result_df: pd.DataFrame,
    p_orders_df: pd.DataFrame,
    p_monthly_returns: pd.DataFrame,
    p_yearly_returns: pd.DataFrame,
    p_metrics: dict,
    p_output_dir: Path,
) -> None:
    """
    Save comprehensive backtest diagnostics.
    """
    p_output_dir.mkdir(parents=True, exist_ok=True)

    orders_csv = p_output_dir / "orders.csv"
    p_orders_df.to_csv(orders_csv, index=False)
    print(f"Saved orders: {orders_csv}")

    result_parquet = p_output_dir / "backtest_result.parquet"
    p_result_df.to_parquet(result_parquet)
    print(f"Saved backtest result: {result_parquet}")

    monthly_csv = p_output_dir / "monthly_returns.csv"
    p_monthly_returns.to_csv(monthly_csv, index=False)
    print(f"Saved monthly returns: {monthly_csv}")

    yearly_csv = p_output_dir / "yearly_returns.csv"
    p_yearly_returns.to_csv(yearly_csv, index=False)
    print(f"Saved yearly returns: {yearly_csv}")

    metrics_txt = p_output_dir / "metrics.txt"
    with open(metrics_txt, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("STRATEGY PERFORMANCE METRICS\n")
        f.write("=" * 60 + "\n\n")
        for key, value in p_metrics.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.6f}\n")
            else:
                f.write(f"{key}: {value}\n")
    print(f"Saved metrics: {metrics_txt}")

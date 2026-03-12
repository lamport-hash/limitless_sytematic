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

from strat.strat_backtest import ETF_LIST


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
    p_rebalance_threshold: float = 0.05,
) -> Tuple[pd.DataFrame, List[Order]]:
    """
    Generate orders from allocation DataFrame.

    Args:
        p_df: DataFrame with A_*_alloc columns (output from compute_dual_momentum)
        p_rebalance_threshold: Minimum allocation change to trigger rebalance

    Returns:
        Tuple of (orders_df, orders_list)
    """
    df = p_df.copy()

    alloc_cols = [f"A_{etf}_alloc" for etf in ETF_LIST]
    close_cols = [f"{etf}_{OHLC_COLS['close']}" for etf in ETF_LIST]

    for col in alloc_cols + close_cols:
        if col not in df.columns:
            raise ValueError(f"Required column not found: {col}")

    orders = []
    current_alloc = np.zeros(len(ETF_LIST))
    portfolio_value = 1.0

    for i, (idx, row) in enumerate(df.iterrows()):
        target_alloc = np.array([row[col] for col in alloc_cols])
        close_prices = np.array([row[col] for col in close_cols])

        if i == 0:
            for j, etf in enumerate(ETF_LIST):
                if target_alloc[j] > 0:
                    order = Order(
                        timestamp=int(idx),
                        etf=etf,
                        direction=1.0,
                        size=target_alloc[j] * portfolio_value / close_prices[j]
                        if close_prices[j] > 0
                        else 0,
                        price=close_prices[j],
                        allocation=target_alloc[j],
                    )
                    orders.append(order)
            current_alloc = target_alloc.copy()
            continue

        allocation_change = np.abs(target_alloc - current_alloc)
        needs_rebalance = np.any(allocation_change > p_rebalance_threshold)

        if needs_rebalance:
            for j, etf in enumerate(ETF_LIST):
                if allocation_change[j] > p_rebalance_threshold:
                    if target_alloc[j] > current_alloc[j]:
                        order = Order(
                            timestamp=int(idx),
                            etf=etf,
                            direction=1.0,
                            size=(target_alloc[j] - current_alloc[j])
                            * portfolio_value
                            / close_prices[j]
                            if close_prices[j] > 0
                            else 0,
                            price=close_prices[j],
                            allocation=target_alloc[j],
                        )
                        orders.append(order)
                    elif target_alloc[j] < current_alloc[j]:
                        order = Order(
                            timestamp=int(idx),
                            etf=etf,
                            direction=-1.0,
                            size=(current_alloc[j] - target_alloc[j])
                            * portfolio_value
                            / close_prices[j]
                            if close_prices[j] > 0
                            else 0,
                            price=close_prices[j],
                            allocation=target_alloc[j],
                        )
                        orders.append(order)

            current_alloc = target_alloc.copy()

    orders_df = pd.DataFrame(
        [
            {
                "timestamp": o.timestamp,
                "etf": o.etf,
                "direction": o.direction,
                "size": o.size,
                "price": o.price,
                "allocation": o.allocation,
            }
            for o in orders
        ]
    )

    return orders_df, orders


def compute_strategy_performance(
    p_df: pd.DataFrame,
    p_orders_df: pd.DataFrame,
) -> Tuple[Dict, pd.DataFrame]:
    """
    Compute strategy performance from trade-by-trade P&L.

    Args:
        p_df: DataFrame with A_*_alloc columns and OHLC data
        p_orders_df: Orders DataFrame from generate_orders_from_allocations

    Returns:
        Tuple of (metrics_dict, result_df)
    """
    close_col_map = {etf: f"{etf}_{OHLC_COLS['close']}" for etf in ETF_LIST}

    positions = {etf: {"entry_price": None, "entry_idx": None} for etf in ETF_LIST}
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

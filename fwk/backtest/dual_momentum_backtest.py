"""
Dual Momentum Strategy Backtest.

Runs backtest on dual momentum allocations, computing performance metrics,
order generation, and period returns.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from strat.dual_momentum import ETF_LIST


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
    p_output_dir: Optional[Path] = None,
    p_plot_histogram: bool = False,
    p_prefix: Optional[str] = None,
) -> Tuple[Dict, pd.DataFrame]:
    df = p_df.copy()

    alloc_cols = [f"A_{etf}_alloc" for etf in ETF_LIST]
    close_cols = [f"{etf}_{OHLC_COLS['close']}" for etf in ETF_LIST]

    returns_matrix = np.zeros(len(df))
    portfolio_alloc = np.zeros((len(df), len(ETF_LIST)))

    for i in range(len(df)):
        allocations = np.array([df.iloc[i][col] for col in alloc_cols])
        portfolio_alloc[i] = allocations

    for j in range(len(close_cols)):
        etf_returns = df[close_cols[j]].pct_change().fillna(0).values
        returns_matrix += portfolio_alloc[:, j] * etf_returns

    df["pnl_per_bar"] = returns_matrix
    df["cum_pnl"] = (1 + returns_matrix).cumprod() - 1
    portfolio_value = (1 + returns_matrix).cumprod()
    df["portfolio_value"] = portfolio_value

    returns = returns_matrix[returns_matrix != 0]
    pnl_per_bar = returns_matrix

    rolling_max = np.maximum.accumulate(portfolio_value)
    drawdown = (portfolio_value - rolling_max) / rolling_max
    max_dd = drawdown.min()

    positive_returns = returns[returns > 0]
    negative_returns = returns[returns < 0]

    n_bars = len(df)
    n_months = n_bars / (30 * 24)

    total_pnl = portfolio_value[-1] - 1.0
    avg_pnl_per_bar = (
        np.mean(pnl_per_bar[pnl_per_bar != 0]) if np.any(pnl_per_bar != 0) else 0
    )

    sharpe = 0.0
    if np.std(returns) > 0 and len(returns) > 1:
        annualized_return = np.mean(returns) * 252 * 24
        annualized_vol = np.std(returns) * np.sqrt(252 * 24)
        sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0

    calmar = 0.0
    if max_dd < 0 and n_bars > 0:
        annualized_return = (portfolio_value[-1] / portfolio_value[0]) ** (
            252 * 24 / n_bars
        ) - 1
        calmar = annualized_return / abs(max_dd)

    profit_factor = 0.0
    gross_profit = np.sum(positive_returns)
    gross_loss = abs(np.sum(negative_returns))
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss

    n_trades = len(p_orders_df)
    win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
    avg_win = np.mean(positive_returns) if len(positive_returns) > 0 else 0
    avg_loss = np.mean(negative_returns) if len(negative_returns) > 0 else 0

    metrics = {
        "total_return": total_pnl,
        "total_return_pct": total_pnl * 100,
        "pnl_per_bar": avg_pnl_per_bar,
        "pnl_per_month": total_pnl / n_months if n_months > 0 else 0,
        "cum_pnl_final": df["cum_pnl"].iloc[-1],
        "final_portfolio_value": portfolio_value[-1],
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

    return metrics, df


def compute_period_returns(
    p_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute monthly and yearly returns from the backtest results.

    Args:
        p_df: DataFrame with datetime index and 'pnl_per_bar' column

    Returns:
        Tuple of (monthly_returns_df, yearly_returns_df)
    """
    df = p_df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        index_values = df.index.values.astype(np.int64)

        try:
            index_values = index_values / 1e7
            reference_time = pd.Timestamp("2000-01-01 00:00:00")
            df["datetime"] = reference_time + pd.to_timedelta(index_values, unit="s")
            df = df.set_index("datetime")
        except Exception as e:
            raise ValueError(
                f"Cannot convert index to datetime. Index type: {type(df.index)}, sample: {df.index[:3].tolist()}, error: {e}"
            )

    df["month"] = df.index.to_period("M")
    df["year"] = df.index.to_period("Y")

    monthly_returns = df.groupby("month").agg(
        return_=("pnl_per_bar", lambda x: (1 + x).prod() - 1),
        n_bars=("pnl_per_bar", "count"),
        positive_bars=("pnl_per_bar", lambda x: (x > 0).sum()),
        negative_bars=("pnl_per_bar", lambda x: (x < 0).sum()),
    )

    monthly_returns["win_rate"] = (
        monthly_returns["positive_bars"] / monthly_returns["n_bars"]
    )
    monthly_returns = monthly_returns.rename(
        columns={"return_": "return"}
    ).reset_index()

    yearly_returns = df.groupby("year").agg(
        return_=("pnl_per_bar", lambda x: (1 + x).prod() - 1),
        n_bars=("pnl_per_bar", "count"),
        positive_bars=("pnl_per_bar", lambda x: (x > 0).sum()),
        negative_bars=("pnl_per_bar", lambda x: (x < 0).sum()),
    )

    yearly_returns["win_rate"] = (
        yearly_returns["positive_bars"] / yearly_returns["n_bars"]
    )
    yearly_returns = yearly_returns.rename(columns={"return_": "return"}).reset_index()

    return monthly_returns, yearly_returns


def print_metrics(metrics: Dict):
    """Print performance metrics in formatted table."""
    print("\n" + "=" * 60)
    print("STRATEGY PERFORMANCE METRICS")
    print("=" * 60)

    print(f"\n{'Returns':^60}")
    print("-" * 60)
    print(
        f"  Total PnL:              {metrics['total_return']:.4f} ({metrics['total_return_pct']:.2f}%)"
    )
    print(f"  Final Portfolio Value:  {metrics['final_portfolio_value']:.4f}")
    print(f"  PnL per Bar:            {metrics['pnl_per_bar']:.6f}")
    print(f"  PnL per Month:          {metrics['pnl_per_month']:.4f}")

    print(f"\n{'Risk Metrics':^60}")
    print("-" * 60)
    print(
        f"  Max Drawdown:           {metrics['max_drawdown']:.4f} ({metrics['max_dd_pct']:.2f}%)"
    )
    print(f"  Sharpe Ratio:           {metrics['sharpe_ratio']:.4f}")
    print(f"  Calmar Ratio:           {metrics['calmar_ratio']:.4f}")

    print(f"\n{'Trade Statistics':^60}")
    print("-" * 60)
    print(f"  Number of Trades:       {metrics['n_trades']}")
    print(f"  Win Rate:               {metrics['win_rate']:.2%}")
    print(f"  Avg Win:                {metrics['avg_win']:.6f}")
    print(f"  Avg Loss:               {metrics['avg_loss']:.6f}")
    print(f"  Avg Win/Loss Ratio:     {metrics['avg_win_loss_ratio']:.4f}")
    print(f"  Profit Factor:          {metrics['profit_factor']:.4f}")

    print(f"\n{'Bar Statistics':^60}")
    print("-" * 60)
    print(f"  Total Bars:             {metrics['n_bars']}")
    print(f"  Positive Bars:          {metrics['n_positive_bars']}")
    print(f"  Negative Bars:          {metrics['n_negative_bars']}")

    print("=" * 60)

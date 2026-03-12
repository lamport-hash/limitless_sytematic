"""
PnL and allocation analysis functions.
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd

ETF_LIST = ["QQQ", "SPY", "TLT", "GLD", "VWO"]


def print_allocation_summary(p_df: pd.DataFrame, p_n_last: int = 10):
    """Print summary of recent allocations."""
    print(f"\n{'=' * 60}")
    print("Recent Allocations (last {} periods)".format(p_n_last))
    print(f"{'=' * 60}")

    alloc_cols = [f"A_{etf}_alloc" for etf in ETF_LIST]
    recent = p_df[alloc_cols + ["A_top_etf", "A_n_positive_momentum"]].tail(p_n_last)

    for idx, row in recent.iterrows():
        alloc_str = ", ".join(
            [
                f"{ETF_LIST[i]}:{row[col]:.2f}"
                for i, col in enumerate(alloc_cols)
                if row[col] > 0
            ]
        )
        print(
            f"  {idx}: {alloc_str} | top: {row['A_top_etf']} | +mom: {row['A_n_positive_momentum']}"
        )


def compute_period_returns(
    p_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute monthly and yearly returns from the backtest results.

    Args:
        p_df: DataFrame with datetime index or bar index and 'pnl_per_bar' column

    Returns:
        Tuple of (monthly_returns_df, yearly_returns_df)
    """
    df = p_df.copy()

    # Check for i_minute_i column (minutes since 01/01/2000)
    if "i_minute_i" in df.columns:
        reference_time = pd.Timestamp("2000-01-01 00:00:00")
        df["datetime"] = reference_time + pd.to_timedelta(df["i_minute_i"], unit="min")
        df = df.set_index("datetime")
    elif not isinstance(df.index, pd.DatetimeIndex):
        index_values = df.index.values.astype(np.int64)

        if index_values.max() > 1e15:
            try:
                index_values = index_values / 1e7
                reference_time = pd.Timestamp("2000-01-01 00:00:00")
                df["datetime"] = reference_time + pd.to_timedelta(index_values, unit="s")
                df = df.set_index("datetime")
            except Exception:
                pass

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

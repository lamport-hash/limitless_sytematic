"""
Visualization functions for strategy results.
"""

from pathlib import Path

import pandas as pd

ETF_LIST = ["QQQ", "SPY", "TLT", "GLD", "VWO"]

ETF_COLORS = {
    "QQQ": "#1f77b4",
    "SPY": "#ff7f0e",
    "TLT": "#2ca02c",
    "GLD": "#d62728",
    "VWO": "#9467bd",
}


def plot_normalized_prices(p_df: pd.DataFrame, p_output_path: Path, p_etf_list: list = None) -> None:
    """
    Plot close prices normalized at 100.
    """
    import matplotlib.pyplot as plt

    etf_list = p_etf_list or ETF_LIST

    fig, ax = plt.subplots(figsize=(14, 8))

    for etf in etf_list:
        close_col = f"{etf}_S_close_f32"
        if close_col in p_df.columns:
            prices = p_df[close_col].astype(float)
            normalized = (prices / prices.iloc[0]) * 100
            ax.plot(
                p_df.index, normalized, label=etf, color=ETF_COLORS.get(etf), linewidth=1.5
            )

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Normalized Price (Start = 100)", fontsize=12)
    ax.set_title("Normalized ETF Prices", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(p_output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p_output_path}")


def create_allocation_gif(
    p_result: pd.DataFrame, p_output_path: Path, p_etf_list: list = None, p_fps: int = 10
) -> None:
    """
    Create animated GIF showing allocation evolution over time.
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    etf_list = p_etf_list or ETF_LIST

    fig, ax = plt.subplots(figsize=(12, 8))

    alloc_cols = [f"A_{etf}_alloc" for etf in etf_list]
    colors = [ETF_COLORS.get(etf, "#333333") for etf in etf_list]

    n_frames = len(p_result)
    frame_step = max(1, n_frames // 200)

    def animate(frame_idx):
        ax.clear()

        data_idx = frame_idx * frame_step
        if data_idx >= n_frames:
            data_idx = n_frames - 1

        allocations = [p_result.iloc[data_idx][col] for col in alloc_cols]

        bars = ax.bar(etf_list, allocations, color=colors, edgecolor="black", linewidth=1)

        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Allocation", fontsize=12)
        ax.set_xlabel("ETF", fontsize=12)
        ax.set_title(
            f"Portfolio Allocation - {p_result.index[data_idx]}", fontsize=14, fontweight="bold"
        )
        ax.grid(True, alpha=0.3, axis="y")

        for bar, alloc in zip(bars, allocations):
            if alloc > 0.01:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{alloc * 100:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

        return bars

    n_animation_frames = n_frames // frame_step + 1
    anim = animation.FuncAnimation(
        fig, animate, frames=n_animation_frames, interval=1000 // p_fps, blit=False
    )

    plt.tight_layout()
    anim.save(p_output_path, writer="pillow", fps=p_fps)
    plt.close()
    print(f"Saved: {p_output_path}")


def plot_pnl_histogram(p_result_df: pd.DataFrame, p_output_path: Path) -> None:
    """
    Plot histogram of PnL per bar.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))

    pnl = p_result_df["pnl_per_bar"].dropna()
    pnl_nonzero = pnl[pnl != 0]

    ax.hist(pnl_nonzero, bins=100, edgecolor="black", alpha=0.7)
    ax.axvline(x=0, color="red", linestyle="--", linewidth=2)
    ax.axvline(
        x=float(pnl_nonzero.mean()),
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {pnl_nonzero.mean():.6f}",
    )

    ax.set_xlabel("PnL per Bar", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Distribution of PnL per Bar", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(p_output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved PnL histogram: {p_output_path}")


def plot_portfolio_value(p_result_df: pd.DataFrame, p_output_path: Path) -> None:
    """
    Plot portfolio value over time with drawdown.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(p_result_df.index, p_result_df["portfolio_value"], linewidth=1)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)

    ax.set_xlabel("Bar", fontsize=12)
    ax.set_ylabel("Portfolio Value", fontsize=12)
    ax.set_title("Portfolio Value Over Time", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    rolling_max = p_result_df["portfolio_value"].cummax()
    drawdown = (p_result_df["portfolio_value"] - rolling_max) / rolling_max

    ax2 = ax.twinx()
    ax2.fill_between(p_result_df.index, drawdown, 0, alpha=0.3, color="red", label="Drawdown")
    ax2.set_ylabel("Drawdown", fontsize=12, color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    plt.tight_layout()
    plt.savefig(p_output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved portfolio value chart: {p_output_path}")


def plot_allocation_history(p_result_df: pd.DataFrame, p_output_path: Path) -> None:
    """
    Plot allocation history stacked area chart.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 8))

    alloc_data = {}
    for etf in ETF_LIST:
        col = f"A_{etf}_alloc"
        if col in p_result_df.columns:
            alloc_data[etf] = p_result_df[col]

    alloc_df = pd.DataFrame(alloc_data, index=p_result_df.index)

    colors = [ETF_COLORS.get(etf, "#333333") for etf in ETF_LIST]
    alloc_df.plot.area(ax=ax, stacked=True, alpha=0.7, color=colors)

    ax.set_xlabel("Bar", fontsize=12)
    ax.set_ylabel("Allocation", fontsize=12)
    ax.set_title("Allocation History", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(p_output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved allocation history: {p_output_path}")


def plot_cumulative_pnl(p_result_df: pd.DataFrame, p_output_path: Path) -> None:
    """
    Plot cumulative PnL over time.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.fill_between(
        p_result_df.index,
        0,
        p_result_df["cum_pnl"],
        where=p_result_df["cum_pnl"] >= 0,
        alpha=0.3,
        color="green",
        label="Profit",
    )
    ax.fill_between(
        p_result_df.index,
        0,
        p_result_df["cum_pnl"],
        where=p_result_df["cum_pnl"] < 0,
        alpha=0.3,
        color="red",
        label="Loss",
    )
    ax.plot(p_result_df.index, p_result_df["cum_pnl"], linewidth=1)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    ax.set_xlabel("Bar", fontsize=12)
    ax.set_ylabel("Cumulative PnL", fontsize=12)
    ax.set_title("Cumulative PnL Over Time", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(p_output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved cumulative PnL chart: {p_output_path}")


def plot_monthly_returns_chart(p_monthly_returns: pd.DataFrame, p_output_path: Path) -> None:
    """
    Plot monthly returns as a bar chart.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 8))

    colors = ["green" if x >= 0 else "red" for x in p_monthly_returns["return"]]
    ax.bar(
        range(len(p_monthly_returns)),
        p_monthly_returns["return"] * 100,
        color=colors,
        alpha=0.7,
    )

    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Return (%)", fontsize=12)
    ax.set_title("Monthly Returns", fontsize=14, fontweight="bold")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    if len(p_monthly_returns) <= 24:
        ax.set_xticks(range(len(p_monthly_returns)))
        ax.set_xticklabels([str(m) for m in p_monthly_returns["month"]], rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(p_output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved monthly returns chart: {p_output_path}")



def plot_performance_ranking_timeline(
    p_rolling_ranking: pd.DataFrame,
    p_horizon: int,
    p_output_path: Path = None,
    p_etf_list: list = None,
    p_etf_colors: dict = None,
    p_show: bool = False,
) -> None:
    """
    Plot performance ranking over time as a colored timeline.

    Shows which ETF is ranked 1st (y=5), 2nd (y=4), etc. at each point in time.
    Each ETF is colored differently, making it easy to see ranking changes.

    Args:
        p_rolling_ranking: DataFrame from compute_rolling_performance_ranking()
        p_horizon: Time horizon in hours (e.g., 1, 24, 48)
        p_output_path: Path to save the chart (optional if p_show=True)
        p_etf_list: List of ETF symbols (default: QQQ, SPY, TLT, GLD, VWO)
        p_etf_colors: Dict mapping ETF to color (default: ETF_COLORS)
        p_show: If True, display in notebook instead of saving (default: False)

    Example:
        >>> plot_performance_ranking_timeline(rolling_df, 48, "ranking_48h.png")
        >>> plot_performance_ranking_timeline(rolling_df, 48, p_show=True)  # Display in notebook
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    etf_list = p_etf_list or ETF_LIST
    etf_colors = p_etf_colors or ETF_COLORS

    rank_cols = {rank: f"bestperf_{rank}_{p_horizon}" for rank in range(1, len(etf_list) + 1)}

    for col in rank_cols.values():
        if col not in p_rolling_ranking.columns:
            raise ValueError(f"Column not found: {col}. Make sure horizon {p_horizon}h was computed.")

    fig, ax = plt.subplots(figsize=(16, 6))

    rank_positions = {etf: [] for etf in etf_list}

    for idx, row in p_rolling_ranking.iterrows():
        for rank, col in rank_cols.items():
            etf = row[col]
            if etf in etf_list:
                rank_positions[etf].append((idx, rank))
            else:
                for etf in etf_list:
                    rank_positions[etf].append((idx, np.nan))

    for etf in etf_list:
        positions = rank_positions[etf]
        if positions:
            x_vals = [p[0] for p in positions]
            y_vals = [p[1] for p in positions]
            ax.scatter(
                x_vals,
                y_vals,
                c=etf_colors.get(etf, "#888888"),
                label=etf,
                s=8,
                alpha=0.8,
            )

    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Rank (1 = Best)", fontsize=12)
    ax.set_title(f"ETF Performance Ranking Timeline ({p_horizon}h Horizon)", fontsize=14, fontweight="bold")

    ax.set_yticks(range(1, len(etf_list) + 1))
    ax.set_yticklabels([f"#{i}" for i in range(1, len(etf_list) + 1)])
    ax.invert_yaxis()

    ax.grid(True, alpha=0.3, axis="y")

    legend_patches = [mpatches.Patch(color=etf_colors.get(etf, "#888888"), label=etf) for etf in etf_list]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=10, ncol=len(etf_list))

    plt.tight_layout()
    
    if p_show:
        plt.show()
    elif p_output_path:
        plt.savefig(p_output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {p_output_path}")
        plt.close()


def plot_trades_on_candles(
    p_df: pd.DataFrame,
    p_orders_df: pd.DataFrame,
    p_output_path: Path = None,
    p_etf: str = "QQQ",
    p_n_bars: int = 500,
) -> None:
    """
    Plot trades on hourly candles for a specific ETF.
    
    Shows all hourly candles with entry/exit points marked.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    close_col = f"{p_etf}_S_close_f32"
    high_col = f"{p_etf}_S_high_f32"
    low_col = f"{p_etf}_S_low_f32"
    open_col = f"{p_etf}_S_open_f32"
    
    df_plot = p_df.iloc[-p_n_bars:].copy()
    
    ax1 = axes[0]
    for i in range(len(df_plot)):
        idx = df_plot.index[i]
        o = df_plot.loc[idx, open_col]
        c = df_plot.loc[idx, close_col]
        h = df_plot.loc[idx, high_col]
        l = df_plot.loc[idx, low_col]
        color = "green" if c >= o else "red"
        ax1.plot([i, i], [l, h], color=color, linewidth=0.5)
        ax1.plot([i-0.3, i+0.3], [o, o], color=color, linewidth=0.5)
        ax1.plot([i-0.3, i+0.3], [c, c], color=color, linewidth=0.5)
    
    buy_orders = p_orders_df[(p_orders_df["etf"] == p_etf) & (p_orders_df["direction"] == 1.0)]
    sell_orders = p_orders_df[(p_orders_df["etf"] == p_etf) & (p_orders_df["direction"] == -1.0)]
    
    for _, order in buy_orders.iterrows():
        if order["timestamp"] in df_plot.index:
            idx_pos = df_plot.index.get_loc(order["timestamp"])
            ax1.scatter(idx_pos, order["price"], marker="^", color="green", s=100, zorder=5)
    
    for _, order in sell_orders.iterrows():
        if order["timestamp"] in df_plot.index:
            idx_pos = df_plot.index.get_loc(order["timestamp"])
            ax1.scatter(idx_pos, order["price"], marker="v", color="red", s=100, zorder=5)
    
    ax1.set_ylabel(f"{p_etf} Price", fontsize=12)
    ax1.set_title(f"{p_etf} Price with Trades (Last {p_n_bars} Bars)", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    if "portfolio_value" in p_df.columns:
        ax2.plot(range(len(df_plot)), df_plot["portfolio_value"].iloc[-p_n_bars:].values, linewidth=1.5)
        ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
        ax2.set_ylabel("Portfolio Value", fontsize=12)
        ax2.set_title("Portfolio Value", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3)
    
    ax3 = axes[2]
    if "pnl_per_bar" in p_df.columns:
        pnl = p_df["pnl_per_bar"].iloc[-p_n_bars:].values
        colors = ["green" if x >= 0 else "red" for x in pnl]
        ax3.bar(range(len(pnl)), pnl, color=colors, alpha=0.5, width=1.0)
        ax3.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
        ax3.set_ylabel("PnL per Bar", fontsize=12)
        ax3.set_xlabel("Bar", fontsize=12)
        ax3.set_title("PnL per Bar", fontsize=14, fontweight="bold")
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if p_output_path:
        plt.savefig(p_output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {p_output_path}")
        plt.close()
    else:
        plt.show()


def plot_all_etfs_trades(
    p_df: pd.DataFrame,
    p_orders_df: pd.DataFrame,
    p_output_dir: Path = None,
    p_n_bars: int = 500,
) -> None:
    """
    Plot trades for all ETFs.
    """
    for etf in ETF_LIST:
        output_path = p_output_dir / f"trades_{etf}.png" if p_output_dir else None
        plot_trades_on_candles(
            p_df=p_df,
            p_orders_df=p_orders_df,
            p_output_path=output_path,
            p_etf=etf,
            p_n_bars=p_n_bars,
        )


def plot_detailed_diagnostics(
    p_df: pd.DataFrame,
    p_orders_df: pd.DataFrame,
    p_output_dir: Path,
    p_n_bars: int = 500,
) -> None:
    """
    Generate comprehensive diagnostics showing all hourly candles with trades.
    """
    p_output_dir.mkdir(parents=True, exist_ok=True)
    
    if "portfolio_value" in p_df.columns:
        plot_portfolio_value(p_df, p_output_dir / "portfolio_value.png")
    
    if "pnl_per_bar" in p_df.columns:
        plot_pnl_histogram(p_df, p_output_dir / "pnl_histogram.png")
    
    if any(f"A_{etf}_alloc" in p_df.columns for etf in ETF_LIST):
        plot_allocation_history(p_df, p_output_dir / "allocation_history.png")
    
    plot_all_etfs_trades(p_df, p_orders_df, p_output_dir, p_n_bars)
    
    print(f"All diagnostics saved to: {p_output_dir}")

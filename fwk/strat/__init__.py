"""Strategy modules for allocation and trading logic."""

from strat.strat_backtest import (
    compute_dual_momentum,
    get_current_allocation,
    create_test_bundle_with_allocations,
)
from strat.strat_analysis import (
    print_allocation_summary,
    print_metrics,
    compute_period_returns,
)
from strat.strat_visualise import (
    ETF_COLORS,
    plot_normalized_prices,
    create_allocation_gif,
    plot_pnl_histogram,
    plot_portfolio_value,
    plot_allocation_history,
    plot_cumulative_pnl,
    plot_monthly_returns_chart,
    plot_performance_ranking_timeline,
)
from strat.s_macd_ema import (
    build_features,
    ema_trend_signal,
    MACDEMA_SwingWindow_BE,
    MACDEMA_SwingWindow,
    MACDEMA_SwingOrATR,
    MACDEMA_ATRorBand,
    MACDEMA_ATRTrail,
    run_backtest,
    optimize_backtest,
    split_and_optimize,
)

__all__ = [
    "ETF_COLORS",
    "compute_dual_momentum",
    "get_current_allocation",
    "create_test_bundle_with_allocations",
    "print_allocation_summary",
    "print_metrics",
    "compute_period_returns",
    "plot_normalized_prices",
    "create_allocation_gif",
    "plot_pnl_histogram",
    "plot_portfolio_value",
    "plot_allocation_history",
    "plot_cumulative_pnl",
    "plot_monthly_returns_chart",
    "plot_performance_ranking_timeline",
    "build_features",
    "ema_trend_signal",
    "MACDEMA_SwingWindow_BE",
    "MACDEMA_SwingWindow",
    "MACDEMA_SwingOrATR",
    "MACDEMA_ATRorBand",
    "MACDEMA_ATRTrail",
    "run_backtest",
    "optimize_backtest",
    "split_and_optimize",
]

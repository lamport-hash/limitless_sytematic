"""Strategy modules for allocation and trading logic."""

from strat.dual_momentum import (
    ETF_LIST,
    compute_dual_momentum,
    get_current_allocation,
    print_allocation_summary,
)

__all__ = [
    "ETF_LIST",
    "compute_dual_momentum",
    "get_current_allocation",
    "print_allocation_summary",
]

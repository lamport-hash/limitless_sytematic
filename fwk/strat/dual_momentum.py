"""
Dual Momentum Strategy for ETF Basket.

Implements the classic dual momentum approach:
1. Absolute momentum: ROC > threshold (positive momentum)
2. Relative momentum: Rank ETFs by ROC, pick top N with positive momentum
3. If no ETF has positive momentum -> allocate to default ETF (safe haven)
"""

from typing import List

import numpy as np
import pandas as pd


ETF_LIST = ["QQQ", "SPY", "TLT", "GLD", "VWO"]


def compute_dual_momentum(
    p_df: pd.DataFrame,
    p_feature_id: str = "F_roc_4800_F_mid_f32_f16",
    p_default_etf_idx: int = 2,
    p_top_n: int = 1,
    p_abs_momentum_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Compute dual momentum allocations for ETF basket.

    Dual Momentum Strategy:
    1. Absolute momentum: ROC > threshold (positive momentum)
    2. Relative momentum: Rank ETFs by ROC, pick top N with positive momentum
    3. If no ETF has positive momentum -> allocate to default ETF (safe haven)

    Args:
        p_df: DataFrame with ETF bundle (prefixed columns)
        p_feature_id: Feature to use for momentum (e.g., "F_roc_4800_F_mid_f32_f16")
        p_default_etf_idx: Index of ETF to use when nothing has momentum (default: 2 = TLT)
        p_top_n: Number of top momentum ETFs to allocate to (default: 1)
        p_abs_momentum_threshold: Minimum ROC for absolute momentum (default: 0.0)

    Returns:
        DataFrame with allocation columns for each ETF
    """
    df = p_df.copy()

    feature_cols = {}
    for i, etf in enumerate(ETF_LIST):
        col_name = f"{etf}_{p_feature_id}"
        if col_name not in df.columns:
            raise ValueError(f"Feature column not found: {col_name}")
        feature_cols[etf] = col_name

    roc_matrix = pd.DataFrame({etf: df[col] for etf, col in feature_cols.items()})
    roc_matrix.index = df.index

    n_periods = len(df)
    n_etfs = len(ETF_LIST)

    allocations = np.zeros((n_periods, n_etfs))

    for t in range(n_periods):
        current_roc = roc_matrix.iloc[t].values

        valid_mask = ~np.isnan(current_roc)
        if not valid_mask.any():
            allocations[t, p_default_etf_idx] = 1.0
            continue

        abs_momentum_mask = current_roc > p_abs_momentum_threshold

        if not abs_momentum_mask.any():
            allocations[t, p_default_etf_idx] = 1.0
        else:
            ranked_indices = np.argsort(-current_roc)

            top_indices = []
            for idx in ranked_indices:
                if abs_momentum_mask[idx] and len(top_indices) < p_top_n:
                    top_indices.append(idx)

            if top_indices:
                weight = 1.0 / len(top_indices)
                for idx in top_indices:
                    allocations[t, idx] = weight
            else:
                allocations[t, p_default_etf_idx] = 1.0

    for i, etf in enumerate(ETF_LIST):
        df[f"A_{etf}_alloc"] = allocations[:, i]

    df["A_top_etf"] = df[[f"A_{etf}_alloc" for etf in ETF_LIST]].idxmax(axis=1)
    df["A_top_etf"] = df["A_top_etf"].str.replace("A_", "").str.replace("_alloc", "")

    df["A_n_positive_momentum"] = (roc_matrix > p_abs_momentum_threshold).sum(axis=1)

    roc_rank = roc_matrix.rank(axis=1, ascending=False)
    for etf in ETF_LIST:
        df[f"A_{etf}_roc_rank"] = roc_rank[etf]

    return df


def get_current_allocation(p_df: pd.DataFrame) -> dict:
    """Get the most recent allocation."""
    last_row = p_df.iloc[-1]

    allocations = {}
    for etf in ETF_LIST:
        col = f"A_{etf}_alloc"
        if col in last_row:
            allocations[etf] = last_row[col]

    return allocations


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

"""
Dual Momentum Strategy Backtester.

Implements the classic dual momentum approach:
1. Absolute momentum: ROC > threshold (positive momentum)
2. Relative momentum: Rank assets by ROC, pick top N with positive momentum
3. If no asset has positive momentum -> allocate to default asset (safe haven)
"""

import numpy as np
import pandas as pd

def compute_dual_momentum(
    p_df: pd.DataFrame,
    p_feature_id: str = "F_roc_4800_F_mid_f32_f16",
    p_default_asset_idx: int = 2,
    p_top_n: int = 1,
    p_abs_momentum_threshold: float = 0.0,
    p_asset_list = ["QQQ", "SPY", "TLT", "GLD", "VWO"]
) -> pd.DataFrame:
    """
    Compute dual momentum allocations for asset basket.

    Dual Momentum Strategy:
    1. Absolute momentum: ROC > threshold (positive momentum)
    2. Relative momentum: Rank assets by ROC, pick top N with positive momentum
    3. If no asset has positive momentum -> allocate to default asset (safe haven)

    Args:
        p_df: DataFrame with asset bundle (prefixed columns)
        p_feature_id: Feature to use for momentum (e.g., "F_roc_4800_F_mid_f32_f16")
        p_default_asset_idx: Index of asset to use when nothing has momentum (default: 2 = TLT)
        p_top_n: Number of top momentum assets to allocate to (default: 1)
        p_abs_momentum_threshold: Minimum ROC for absolute momentum (default: 0.0)

    Returns:
        DataFrame with allocation columns for each asset
    """
    df = p_df.copy()

    feature_cols = {}
    for i, asset in enumerate(p_asset_list):
        col_name = f"{asset}_{p_feature_id}"
        if col_name not in df.columns:
            raise ValueError(f"Feature column not found: {col_name}")
        feature_cols[asset] = col_name

    roc_matrix = pd.DataFrame({asset: df[col] for asset, col in feature_cols.items()})
    roc_matrix.index = df.index

    n_periods = len(df)
    n_assets = len(p_asset_list)

    allocations = np.zeros((n_periods, n_assets))

    for t in range(n_periods):
        current_roc = roc_matrix.iloc[t].values

        valid_mask = ~np.isnan(current_roc)
        if not valid_mask.any():
            allocations[t, p_default_asset_idx] = 1.0
            continue

        abs_momentum_mask = current_roc > p_abs_momentum_threshold

        if not abs_momentum_mask.any():
            allocations[t, p_default_asset_idx] = 1.0
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
                allocations[t, p_default_asset_idx] = 1.0

    for i, asset in enumerate(p_asset_list):
        df[f"A_{asset}_alloc"] = allocations[:, i]

    df["A_top_asset"] = df[[f"A_{asset}_alloc" for asset in p_asset_list]].idxmax(axis=1)
    df["A_top_asset"] = df["A_top_asset"].str.replace("A_", "").str.replace("_alloc", "")

    df["A_n_positive_momentum"] = (roc_matrix > p_abs_momentum_threshold).sum(axis=1)

    roc_rank = roc_matrix.rank(axis=1, ascending=False)
    for asset in p_asset_list:
        df[f"A_{asset}_roc_rank"] = roc_rank[asset]

    return df


def get_current_allocation(p_df: pd.DataFrame) -> dict:
    """Get the most recent allocation."""
    last_row = p_df.iloc[-1]

    allocations = {}
    for asset in p_asset_list:
        col = f"A_{asset}_alloc"
        if col in last_row:
            allocations[asset] = last_row[col]

    return allocations


def create_test_bundle_with_allocations(p_n_rows: int = 100) -> pd.DataFrame:
    """
    Create a test bundle with all required columns for backtesting.
    Generates synthetic data and computes allocations.
    """
    np.random.seed(42)
    data = {}

    for asset in p_asset_list:
        data[f"{asset}_F_roc_4800_F_mid_f32_f16"] = np.random.randn(p_n_rows) * 0.1
        close_prices = 100 + np.cumsum(np.random.randn(p_n_rows) * 0.5)
        data[f"{asset}_S_close_f32"] = close_prices
        data[f"{asset}_S_open_f32"] = close_prices * (1 + np.random.randn(p_n_rows) * 0.01)
        data[f"{asset}_S_high_f32"] = close_prices * (1 + abs(np.random.randn(p_n_rows)) * 0.01)
        data[f"{asset}_S_low_f32"] = close_prices * (1 - abs(np.random.randn(p_n_rows)) * 0.01)
        data[f"{asset}_S_volume_f64"] = np.random.randint(1000, 10000, p_n_rows).astype(float)

    df = pd.DataFrame(data)

    df = compute_dual_momentum(
        p_df=df,
        p_feature_id="F_roc_4800_F_mid_f32_f16",
        p_default_asset_idx=2,
        p_top_n=1,
    )

    return df


def create_test_bundle_with_multiple_assets(p_base_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a test bundle with multiple assets by copying QQQ data to other assets.
    This allows testing the full dual momentum strategy.
    """
    df = p_base_df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    qqq_cols = [c for c in numeric_cols if c.startswith("QQQ_")]

    for asset in ["SPY", "TLT", "GLD", "VWO"]:
        for col in qqq_cols:
            new_col = col.replace("QQQ_", f"{asset}_")
            multiplier = 0.8 + 0.4 * np.random.random()
            df[new_col] = df[col].astype(float) * multiplier

    return df

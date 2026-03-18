"""
Dual Momentum Strategy Backtester.

Implements the classic dual momentum approach:
1. Absolute momentum: ROC > threshold (positive momentum)
2. Relative momentum: Rank assets by ROC, pick top N with positive momentum
3. If no asset has positive momentum -> allocate to default asset (safe haven)
"""

import numpy as np
import pandas as pd

ETF_LIST = ["QQQ", "SPY", "TLT", "GLD", "VWO"]


def compute_dual_momentum(
    p_df: pd.DataFrame,
    p_feature_id: str = "F_roc_4800_F_mid_f32_f16",
    p_default_asset_idx: int = 2,
    p_default_asset: str = "TLT",
    p_top_n: int = 1,
    p_abs_momentum_threshold: float = 0.0,
    p_asset_list = ["QQQ", "SPY", "TLT", "GLD", "VWO"],
    p_min_holding_periods: int = 0,
    p_switch_threshold_pct: float = 0.0,
    p_use_rsi_entry_filter: bool = False,
    p_rsi_entry_max: float = 30.0,
    p_use_rsi_entry_queue: bool = False,
    p_use_rsi_diff_filter: bool = False,
    p_rsi_diff_threshold: float = 10.0,
    p_rsi_feature_id: str = "F_rsi_14_S_close_f32_f16"
) -> pd.DataFrame:
    """
    Compute dual momentum allocations for asset basket.

    Dual Momentum Strategy:
    1. Absolute momentum: ROC > threshold (positive momentum)
    2. Relative momentum: Rank assets by ROC, pick top N with positive momentum
    3. If no asset has positive momentum -> allocate to default asset (safe haven)

    Hysteresis (anti-flip-flop):
    - p_min_holding_periods: Minimum bars to hold before allowing a switch
    - p_switch_threshold_pct: New asset's ROC must exceed current by this % (relative)

    RSI Filters (optional):
    - RSI Entry Filter: Only enter when asset RSI < p_rsi_entry_max
      * When queue disabled: Immediate per-bar filter
      * When queue enabled: Signals queued until RSI drops, then entered
    - RSI Difference Filter: Switch only when (current RSI - target RSI) >= threshold

    Args:
        p_df: DataFrame with asset bundle (prefixed columns)
        p_feature_id: Feature to use for momentum (e.g., "F_roc_4800_F_mid_f32_f16")
        p_default_asset_idx: Index of asset to use when nothing has momentum
        p_default_asset: Symbol of default asset (safe haven) for verification
        p_top_n: Number of top momentum assets to allocate to (default: 1)
        p_abs_momentum_threshold: Minimum ROC for absolute momentum (default: 0.0)
        p_asset_list: List of asset symbols
        p_min_holding_periods: Minimum bars to hold allocation before switch (default: 0)
        p_switch_threshold_pct: Relative % new ROC must exceed current to switch (default: 0.0)
        p_use_rsi_entry_filter: Enable RSI entry filter (default: False)
        p_rsi_entry_max: Maximum RSI value to allow entry (default: 30.0)
        p_use_rsi_entry_queue: Enable pending queue for RSI entry (default: False)
        p_use_rsi_diff_filter: Enable RSI difference filter (default: False)
        p_rsi_diff_threshold: Minimum RSI difference required to switch (default: 10.0)
        p_rsi_feature_id: RSI feature column suffix (default: "F_rsi_14_S_close_f32_f16")

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

    rsi_matrix = None
    if p_use_rsi_entry_filter or p_use_rsi_diff_filter:
        rsi_cols = {}
        for asset in p_asset_list:
            col_name = f"{asset}_{p_rsi_feature_id}"
            if col_name not in df.columns:
                raise ValueError(f"RSI feature column not found: {col_name}")
            rsi_cols[asset] = col_name
        rsi_matrix = pd.DataFrame({asset: df[col] for asset, col in rsi_cols.items()})
        rsi_matrix.index = df.index

    n_periods = len(df)
    n_assets = len(p_asset_list)

    allocations = np.zeros((n_periods, n_assets))
    using_safe_haven = np.zeros(n_periods, dtype=bool)
    rsi_entry_blocked = np.zeros(n_periods, dtype=bool)
    rsi_diff_blocked = np.zeros(n_periods, dtype=bool)

    prev_allocation = None
    periods_since_switch = 0
    prev_top_roc = None
    prev_top_asset_idx = None
    pending_queue = set()

    for t in range(n_periods):
        current_roc = roc_matrix.iloc[t].values

        valid_mask = ~np.isnan(current_roc)
        if not valid_mask.any():
            new_allocation = np.zeros(n_assets)
            new_allocation[p_default_asset_idx] = 1.0
            using_safe_haven[t] = True
            if prev_allocation is None or not np.array_equal(new_allocation, prev_allocation):
                periods_since_switch = 0
            allocations[t] = new_allocation
            prev_allocation = new_allocation.copy()
            continue

        abs_momentum_mask = current_roc > p_abs_momentum_threshold

        if not abs_momentum_mask.any():
            new_allocation = np.zeros(n_assets)
            new_allocation[p_default_asset_idx] = 1.0
            using_safe_haven[t] = True
            if prev_allocation is None or not np.array_equal(new_allocation, prev_allocation):
                periods_since_switch = 0
            allocations[t] = new_allocation
            prev_allocation = new_allocation.copy()
            continue

        ranked_indices = np.argsort(-current_roc)

        top_indices = []
        for idx in ranked_indices:
            if abs_momentum_mask[idx] and len(top_indices) < p_top_n:
                top_indices.append(idx)

        if not top_indices:
            new_allocation = np.zeros(n_assets)
            new_allocation[p_default_asset_idx] = 1.0
            using_safe_haven[t] = True
            if prev_allocation is None or not np.array_equal(new_allocation, prev_allocation):
                periods_since_switch = 0
            allocations[t] = new_allocation
            prev_allocation = new_allocation.copy()
            continue

        current_rsi = None
        if rsi_matrix is not None:
            current_rsi = rsi_matrix.iloc[t].values

        if p_use_rsi_entry_filter and current_rsi is not None:
            if p_use_rsi_entry_queue:
                immediate_candidates = []
                for idx in top_indices:
                    rsi_val = current_rsi[idx]
                    if not np.isnan(rsi_val):
                        if rsi_val < p_rsi_entry_max:
                            immediate_candidates.append(idx)
                        else:
                            pending_queue.add(idx)

                queue_candidates = []
                for idx in list(pending_queue):
                    rsi_val = current_rsi[idx]
                    if not np.isnan(rsi_val) and rsi_val < p_rsi_entry_max:
                        queue_candidates.append(idx)

                all_candidates = list(set(immediate_candidates + queue_candidates))

                if len(pending_queue) > 0 and len(all_candidates) == 0:
                    rsi_entry_blocked[t] = True

                top_indices = all_candidates
            else:
                filtered_indices = []
                for idx in top_indices:
                    rsi_val = current_rsi[idx]
                    if not np.isnan(rsi_val) and rsi_val < p_rsi_entry_max:
                        filtered_indices.append(idx)
                if len(filtered_indices) < len(top_indices):
                    rsi_entry_blocked[t] = True
                top_indices = filtered_indices

        if not top_indices:
            if prev_allocation is not None:
                allocations[t] = prev_allocation.copy()
                periods_since_switch += 1
            else:
                new_allocation = np.zeros(n_assets)
                new_allocation[p_default_asset_idx] = 1.0
                using_safe_haven[t] = True
                allocations[t] = new_allocation
                prev_allocation = new_allocation.copy()
            continue

        if p_use_rsi_entry_filter and current_rsi is not None:
            valid_candidates = [(idx, current_rsi[idx]) for idx in top_indices if not np.isnan(current_rsi[idx])]
            if valid_candidates:
                valid_candidates.sort(key=lambda x: x[1])
                top_indices = [valid_candidates[0][0]]

        new_top_roc = current_roc[top_indices[0]]
        new_top_asset_idx = top_indices[0]

        should_switch = True
        if prev_allocation is not None and p_min_holding_periods > 0:
            if periods_since_switch < p_min_holding_periods:
                should_switch = False

        if should_switch and prev_top_roc is not None and p_switch_threshold_pct > 0:
            threshold_value = prev_top_roc * (1 + p_switch_threshold_pct)
            if new_top_roc < threshold_value:
                should_switch = False

        if should_switch and p_use_rsi_diff_filter and current_rsi is not None and prev_top_asset_idx is not None:
            current_asset_rsi = current_rsi[prev_top_asset_idx]
            target_asset_rsi = current_rsi[new_top_asset_idx]
            if not np.isnan(current_asset_rsi) and not np.isnan(target_asset_rsi):
                rsi_diff = current_asset_rsi - target_asset_rsi
                if rsi_diff < p_rsi_diff_threshold:
                    should_switch = False
                    rsi_diff_blocked[t] = True

        if should_switch or prev_allocation is None:
            weight = 1.0 / len(top_indices)
            new_allocation = np.zeros(n_assets)
            for idx in top_indices:
                new_allocation[idx] = weight
            if prev_allocation is None or not np.array_equal(new_allocation, prev_allocation):
                periods_since_switch = 0
                prev_top_roc = new_top_roc
                prev_top_asset_idx = new_top_asset_idx
                pending_queue.clear()
            allocations[t] = new_allocation
            prev_allocation = new_allocation.copy()
        else:
            allocations[t] = prev_allocation.copy()
            periods_since_switch += 1

        if len(top_indices) > 0:
            rsi_entry_blocked[t] = rsi_entry_blocked[t] or False
            rsi_diff_blocked[t] = rsi_diff_blocked[t] or False

    for i, asset in enumerate(p_asset_list):
        df[f"A_{asset}_alloc"] = allocations[:, i]

    df["A_top_asset"] = df[[f"A_{asset}_alloc" for asset in p_asset_list]].idxmax(axis=1)
    df["A_top_asset"] = df["A_top_asset"].str.replace("A_", "").str.replace("_alloc", "")

    df["A_n_positive_momentum"] = (roc_matrix > p_abs_momentum_threshold).sum(axis=1)
    df["A_using_safe_haven"] = using_safe_haven
    df["A_safe_haven_asset"] = p_default_asset
    df["A_rsi_entry_blocked"] = rsi_entry_blocked
    df["A_rsi_diff_blocked"] = rsi_diff_blocked

    if p_use_rsi_entry_filter:
        for asset in p_asset_list:
            rsi_col = f"{asset}_{p_rsi_feature_id}"
            if rsi_col in df.columns:
                df[f"A_{asset}_rsi"] = df[rsi_col]

    roc_rank = roc_matrix.rank(axis=1, ascending=False)
    for asset in p_asset_list:
        df[f"A_{asset}_roc_rank"] = roc_rank[asset]

    return df


def get_current_allocation(p_df: pd.DataFrame) -> dict:
    """Get the most recent allocation."""
    last_row = p_df.iloc[-1]

    allocations = {}
    alloc_cols = [c for c in p_df.columns if c.endswith("_alloc") and c.startswith("A_")]
    for col in alloc_cols:
        asset = col.replace("A_", "").replace("_alloc", "")
        allocations[asset] = last_row[col]

    return allocations


def create_test_bundle_with_allocations(p_n_rows: int = 100) -> pd.DataFrame:
    """Create a test bundle with synthetic data for backtesting."""
    np.random.seed(42)
    data = {}

    for asset in ETF_LIST:
        data[f"{asset}_F_roc_4800_F_mid_f32_f16"] = np.random.randn(p_n_rows) * 0.1
        close_prices = 100 + np.cumsum(np.random.randn(p_n_rows) * 0.5)
        data[f"{asset}_S_close_f32"] = close_prices
        data[f"{asset}_S_open_f32"] = close_prices * (1 + np.random.randn(p_n_rows) * 0.01)
        data[f"{asset}_S_high_f32"] = close_prices * (1 + abs(np.random.randn(p_n_rows)) * 0.01)
        data[f"{asset}_S_low_f32"] = close_prices * (1 - abs(np.random.randn(p_n_rows)) * 0.01)
        data[f"{asset}_S_volume_f64"] = np.random.randint(1000, 10000, p_n_rows).astype(float)

    df = pd.DataFrame(data)
    df = compute_dual_momentum(p_df=df, p_feature_id="F_roc_4800_F_mid_f32_f16", p_default_asset_idx=2, p_top_n=1)
    return df


def create_test_bundle_with_multiple_assets(p_base_df: pd.DataFrame) -> pd.DataFrame:
    """Create a test bundle with multiple assets by copying QQQ data to other assets."""
    df = p_base_df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    qqq_cols = [c for c in numeric_cols if c.startswith("QQQ_")]

    for asset in ["SPY", "TLT", "GLD", "VWO"]:
        for col in qqq_cols:
            new_col = col.replace("QQQ_", f"{asset}_")
            multiplier = 0.8 + 0.4 * np.random.random()
            df[new_col] = df[col].astype(float) * multiplier

    return df



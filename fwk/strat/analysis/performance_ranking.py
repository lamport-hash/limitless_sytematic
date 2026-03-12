"""
Hourly performance ranking utilities for ETF bundles.

Computes performance rankings across multiple time horizons for each hour of the day.
"""

from typing import List, Optional

import numpy as np
import pandas as pd

from norm.norm_utils import forward_fill_df

ETF_LIST = ["QQQ", "SPY", "TLT", "GLD", "VWO"]
DEFAULT_HORIZONS = [1, 2, 3, 4, 8, 12, 24, 48, 240, 480]
CLOSE_COL = "S_close_f32"


def compute_hourly_performance_ranking(
    p_df: pd.DataFrame,
    p_etf_list: Optional[List[str]] = None,
    p_horizons: Optional[List[int]] = None,
    p_close_col: str = CLOSE_COL,
    p_n_ranks: int = 5,
) -> pd.DataFrame:
    """
    Compute performance rankings for ETFs by hour of day across multiple time horizons.

    For each hour (0-23), computes returns over specified horizons and ranks ETFs.
    Output columns follow pattern: bestperf_<rank>_<hours> (e.g., bestperf_1_48).

    Args:
        p_df: DataFrame with ETF bundle (prefixed columns like QQQ_S_close_f32)
        p_etf_list: List of ETF symbols to analyze (default: QQQ, SPY, TLT, GLD, VWO)
        p_horizons: List of time horizons in hours (default: 1, 2, 3, 4, 8, 12, 24, 48, 240, 480)
        p_close_col: Column name suffix for close price (default: S_close_f32)
        p_n_ranks: Number of ranks to output (default: 5 for all ETFs)

    Returns:
        DataFrame indexed by hour (0-23) with columns:
        - bestperf_<rank>_<hours>: ETF symbol at that rank for that horizon
        - ret_<rank>_<hours>: Return value at that rank for that horizon

    Example:
        >>> df = compute_hourly_performance_ranking(bundle_df)
        >>> df['bestperf_1_48']  # Best performer over 48 hours for each hour
        >>> df['bestperf_2_24']  # 2nd best performer over 24 hours
    """
    df = p_df.copy()
    etf_list = p_etf_list or ETF_LIST
    horizons = p_horizons or DEFAULT_HORIZONS

    close_cols = {etf: f"{etf}_{p_close_col}" for etf in etf_list}
    for etf, col in close_cols.items():
        if col not in df.columns:
            raise ValueError(f"Close column not found: {col}")

    df = forward_fill_df(df, p_columns=list(close_cols.values()))

    if isinstance(df.index, pd.DatetimeIndex):
        df["hour"] = df.index.hour
    else:
        try:
            index_values = df.index.values.astype(np.int64)
            index_values = index_values / 1e7
            reference_time = pd.Timestamp("2000-01-01 00:00:00")
            datetimes = reference_time + pd.to_timedelta(index_values, unit="s")
            df["hour"] = datetimes.hour
        except Exception as e:
            raise ValueError(f"Cannot extract hour from index: {e}")

    returns_by_hour = {h: {etf: [] for etf in etf_list} for h in range(24)}

    for horizon in horizons:
        for etf in etf_list:
            col = close_cols[etf]
            prices = np.asarray(df[col].values, dtype=np.float64)
            shifted = np.roll(prices, -horizon)
            with np.errstate(divide="ignore", invalid="ignore"):
                ret = (shifted - prices) / prices
            df[f"_ret_{etf}_{horizon}"] = ret

    for hour in range(24):
        hour_mask = df["hour"] == hour
        hour_df = df[hour_mask]

        for horizon in horizons:
            ret_cols = [f"_ret_{etf}_{horizon}" for etf in etf_list]
            ret_matrix = hour_df[ret_cols].values

            with np.errstate(all="ignore"):
                if ret_matrix.shape[0] > 0:
                    mean_returns = np.nanmean(ret_matrix, axis=0)
                else:
                    mean_returns = np.full(len(etf_list), np.nan)

            for i, etf in enumerate(etf_list):
                returns_by_hour[hour][etf].append(mean_returns[i])

    results = []

    for hour in range(24):
        row = {"hour": hour}

        for h_idx, horizon in enumerate(horizons):
            hour_returns = {
                etf: returns_by_hour[hour][etf][h_idx] for etf in etf_list
            }

            valid_returns = {
                etf: ret for etf, ret in hour_returns.items()
                if not np.isnan(ret)
            }

            sorted_etfs = sorted(
                valid_returns.items(),
                key=lambda x: (-x[1], etf_list.index(x[0]) if x[0] in etf_list else 999),
            )

            for rank in range(1, p_n_ranks + 1):
                col_name = f"bestperf_{rank}_{horizon}"
                ret_col_name = f"ret_{rank}_{horizon}"

                if rank <= len(sorted_etfs):
                    row[col_name] = sorted_etfs[rank - 1][0]
                    row[ret_col_name] = sorted_etfs[rank - 1][1]
                else:
                    row[col_name] = None
                    row[ret_col_name] = np.nan

        results.append(row)

    result_df = pd.DataFrame(results)
    result_df = result_df.set_index("hour")

    result_df = forward_fill_df(result_df)

    return result_df


def compute_rolling_performance_ranking(
    p_df: pd.DataFrame,
    p_etf_list: Optional[List[str]] = None,
    p_horizons: Optional[List[int]] = None,
    p_close_col: str = CLOSE_COL,
    p_n_ranks: int = 5,
) -> pd.DataFrame:
    """
    Compute performance rankings for each bar across multiple time horizons.

    Unlike compute_hourly_performance_ranking which aggregates by hour,
    this function computes rankings for every single bar in the dataframe.

    Args:
        p_df: DataFrame with ETF bundle (prefixed columns like QQQ_S_close_f32)
        p_etf_list: List of ETF symbols to analyze (default: QQQ, SPY, TLT, GLD, VWO)
        p_horizons: List of time horizons in hours (default: 1, 2, 3, 4, 8, 12, 24, 48, 240, 480)
        p_close_col: Column name suffix for close price (default: S_close_f32)
        p_n_ranks: Number of ranks to output (default: 5 for all ETFs)

    Returns:
        DataFrame with same index as input, with columns:
        - bestperf_<rank>_<hours>: ETF symbol at that rank for that horizon
        - ret_<rank>_<hours>: Return value at that rank for that horizon
    """
    df = p_df.copy()
    etf_list = p_etf_list or ETF_LIST
    horizons = p_horizons or DEFAULT_HORIZONS

    close_cols = {etf: f"{etf}_{p_close_col}" for etf in etf_list}
    for etf, col in close_cols.items():
        if col not in df.columns:
            raise ValueError(f"Close column not found: {col}")

    df = forward_fill_df(df, p_columns=list(close_cols.values()))

    n_bars = len(df)

    for horizon in horizons:
        returns_matrix = np.zeros((n_bars, len(etf_list)))

        for i, etf in enumerate(etf_list):
            col = close_cols[etf]
            prices = np.asarray(df[col].values, dtype=np.float64)
            shifted = np.roll(prices, -horizon)
            with np.errstate(divide="ignore", invalid="ignore"):
                ret = (shifted - prices) / prices
            returns_matrix[:, i] = ret

        for rank in range(1, p_n_ranks + 1):
            bestperf_col = f"bestperf_{rank}_{horizon}"
            ret_col = f"ret_{rank}_{horizon}"

            bestperf_values = np.empty(n_bars, dtype=object)
            ret_values = np.zeros(n_bars)

            for t in range(n_bars):
                returns_t = returns_matrix[t]

                valid_mask = ~np.isnan(returns_t)
                if not valid_mask.any():
                    bestperf_values[t] = None
                    ret_values[t] = np.nan
                    continue

                valid_indices = np.where(valid_mask)[0]
                valid_returns = returns_t[valid_indices]

                sorted_local_indices = np.argsort(-valid_returns, kind="stable")

                if rank <= len(sorted_local_indices):
                    best_local_idx = sorted_local_indices[rank - 1]
                    best_global_idx = valid_indices[best_local_idx]
                    bestperf_values[t] = etf_list[best_global_idx]
                    ret_values[t] = valid_returns[best_local_idx]
                else:
                    bestperf_values[t] = None
                    ret_values[t] = np.nan

            df[bestperf_col] = bestperf_values
            df[ret_col] = ret_values

    return df


def compute_weekly_performance_persistence(
    p_df: pd.DataFrame,
    p_etf_list: Optional[List[str]] = None,
    p_close_col: str = CLOSE_COL,
    p_datetime_col: str = None,
    p_lookback_weeks: int = 5,
    p_forward_weeks: int = 1,
) -> pd.DataFrame:
    """
    Compute weekly performance persistence analysis.

    For each week in history:
    - Look back 1 week: best performer per hour
    - Look back N weeks: best performer per (day_of_week, hour)
    - Look forward: actual performance per (day_of_week, hour)

    Args:
        p_df: DataFrame with ETF bundle (DatetimeIndex or datetime column required)
        p_etf_list: List of ETF symbols (default: QQQ, SPY, TLT, GLD, VWO)
        p_close_col: Column name suffix for close price
        p_datetime_col: Column name containing datetime (auto-detected if None)
        p_lookback_weeks: Number of weeks to look back for pattern (default: 5)
        p_forward_weeks: Number of weeks to look forward for performance (default: 1)

    Returns:
        DataFrame with columns:
        - week_start: start date of target week
        - hour: hour of day (0-23)
        - day_of_week: 0=Monday, 6=Sunday
        - best_1w: best performer over last 1 week for this hour
        - best_Nw: best performer over last N weeks for this (day, hour)
        - actual_best: actual best performer in forward period
        - actual_ret_<ETF>: actual return for each ETF in forward period
        - hit_1w: 1 if best_1w == actual_best, else 0
        - hit_Nw: 1 if best_Nw == actual_best, else 0
    """
    df = p_df.copy()
    etf_list = p_etf_list or ETF_LIST

    close_cols = {etf: f"{etf}_{p_close_col}" for etf in etf_list}
    for etf, col in close_cols.items():
        if col not in df.columns:
            raise ValueError(f"Close column not found: {col}")

    df = forward_fill_df(df, p_columns=list(close_cols.values()))

    if isinstance(df.index, pd.DatetimeIndex):
        pass
    else:
        datetime_col = p_datetime_col
        if datetime_col is None:
            for col in df.columns:
                col_lower = col.lower()
                if "open_time" in col_lower or or elif
                        if datetime_col and datetime_col in df.columns:
                    break
        
        if datetime_col and datetime_col in df.columns:
            df = df.set_index(datetime_col)
            df.index = pd.to_datetime(df.index)
        else:
            idx_values = df.index.values.astype(np.int64)
            if idx_values.max() > 100000:
                idx_values = idx_values / 60
            reference_time = pd.Timestamp("2000-01-01 00:00:00")
            df.index = reference_time + pd.to_timedelta(idx_values, unit="m")
    
    df["_hour"] = df.index.hour
    df["_day_of_week"] = df.index.dayofweek

    iso = df.index.isocalendar()
    df["_year"] = iso["year"].values
    df["_week"] = iso["week"].values

    weeks = df.groupby(["_year", "_week"]).size().reset_index()[["_year", "_week"]]
    weeks = weeks.sort_values(["_year", "_week"]).reset_index(drop=True)

    min_weeks_required = p_lookback_weeks + 1 + p_forward_weeks
    if len(weeks) < min_weeks_required:
        return pd.DataFrame()

    results = []

    lookback_bars = p_lookback_weeks * 7 * 24
    forward_bars = p_forward_weeks * 7 * 24

    for i in range(p_lookback_weeks + 1, len(weeks) - p_forward_weeks):
        target_year, target_week = weeks.iloc[i]["_year"], weeks.iloc[i]["_week"]

        target_mask = (df["_year"] == target_year) & (df["_week"] == target_week)
        target_df = df[target_mask]

        if len(target_df) == 0:
            continue

        lookback_start_idx = max(0, i - p_lookback_weeks)
        lookback_1w_idx = max(0, i - 1)

        lookback_Nw_weeks = weeks.iloc[lookback_start_idx:i]
        lookback_1w_weeks = weeks.iloc[lookback_1w_idx:i]

        forward_weeks = weeks.iloc[i : i + p_forward_weeks]

        lookback_Nw_mask = pd.Series([False] * len(df))
        lookback_1w_mask = pd.Series([False] * len(df))
        forward_mask = pd.Series([False] * len(df))

        for _, row in lookback_Nw_weeks.iterrows():
            lookback_Nw_mask |= (df["_year"] == row["_year"]) & (df["_week"] == row["_week"])

        for _, row in lookback_1w_weeks.iterrows():
            lookback_1w_mask |= (df["_year"] == row["_year"]) & (df["_week"] == row["_week"])

        for _, row in forward_weeks.iterrows():
            forward_mask |= (df["_year"] == row["_year"]) & (df["_week"] == row["_week"])

        lookback_Nw_df = df[lookback_Nw_mask]
        lookback_1w_df = df[lookback_1w_mask]
        forward_df = df[forward_mask]

        if len(target_df) == 0:
            continue

        week_start = target_df.index.min()

        for hour in range(24):
            for day in range(7):
                hour_1w_df = lookback_1w_df[lookback_1w_df["_hour"] == hour]

                hour_day_Nw_df = lookback_Nw_df[
                    (lookback_Nw_df["_hour"] == hour) & (lookback_Nw_df["_day_of_week"] == day)
                ]

                forward_hour_day_df = forward_df[
                    (forward_df["_hour"] == hour) & (forward_df["_day_of_week"] == day)
                ]

                avg_rets_1w = {}
                for etf in etf_list:
                    col = close_cols[etf]
                    if len(hour_1w_df) > 1:
                        prices = hour_1w_df[col].values
                        with np.errstate(divide="ignore", invalid="ignore"):
                            rets = np.diff(prices) / prices[:-1]
                        avg_rets_1w[etf] = np.nanmean(rets)
                    else:
                        avg_rets_1w[etf] = np.nan

                avg_rets_Nw = {}
                for etf in etf_list:
                    col = close_cols[etf]
                    if len(hour_day_Nw_df) > 1:
                        prices = hour_day_Nw_df[col].values
                        with np.errstate(divide="ignore", invalid="ignore"):
                            rets = np.diff(prices) / prices[:-1]
                        avg_rets_Nw[etf] = np.nanmean(rets)
                    else:
                        avg_rets_Nw[etf] = np.nan

                actual_rets = {}
                for etf in etf_list:
                    col = close_cols[etf]
                    if len(forward_hour_day_df) > 1:
                        prices = forward_hour_day_df[col].values
                        with np.errstate(divide="ignore", invalid="ignore"):
                            rets = np.diff(prices) / prices[:-1]
                        actual_rets[etf] = np.nanmean(rets)
                    else:
                        actual_rets[etf] = np.nan

                valid_1w = {k: v for k, v in avg_rets_1w.items() if not np.isnan(v)}
                best_1w = max(valid_1w, key=valid_1w.get) if valid_1w else None

                valid_Nw = {k: v for k, v in avg_rets_Nw.items() if not np.isnan(v)}
                best_Nw = max(valid_Nw, key=valid_Nw.get) if valid_Nw else None

                valid_actual = {k: v for k, v in actual_rets.items() if not np.isnan(v)}
                actual_best = max(valid_actual, key=valid_actual.get) if valid_actual else None

                row = {
                    "week_start": week_start,
                    "hour": hour,
                    "day_of_week": day,
                    "best_1w": best_1w,
                    "best_Nw": best_Nw,
                    "actual_best": actual_best,
                }

                for etf in etf_list:
                    row[f"actual_ret_{etf}"] = actual_rets.get(etf, np.nan)

                row["hit_1w"] = 1 if best_1w and actual_best and best_1w == actual_best else 0
                row["hit_Nw"] = 1 if best_Nw and actual_best and best_Nw == actual_best else 0

                results.append(row)

    result_df = pd.DataFrame(results)
    return result_dfiso = df.index.isocalendar()
    df["_year"] = iso["year"].values
    df["_week"] = iso["week"].values

    weeks = df.groupby(["_year", "_week"]).size().reset_index()[["_year", "_week"]]
    weeks = weeks.sort_values(["_year", "_week"]).reset_index(drop=True)

    min_weeks_required = p_lookback_weeks + 1 + p_forward_weeks
    if len(weeks) < min_weeks_required:
        return pd.DataFrame()

    results = []

    lookback_bars = p_lookback_weeks * 7 * 24
    forward_bars = p_forward_weeks * 7 * 24

    for i in range(p_lookback_weeks + 1, len(weeks) - p_forward_weeks):
        target_year, target_week = weeks.iloc[i]["_year"], weeks.iloc[i]["_week"]

        target_mask = (df["_year"] == target_year) & (df["_week"] == target_week)
        target_df = df[target_mask]

        if len(target_df) == 0:
            continue

        lookback_start_idx = max(0, i - p_lookback_weeks)
        lookback_1w_idx = max(0, i - 1)

        lookback_Nw_weeks = weeks.iloc[lookback_start_idx:i]
        lookback_1w_weeks = weeks.iloc[lookback_1w_idx:i]

        forward_weeks = weeks.iloc[i : i + p_forward_weeks]

        lookback_Nw_mask = pd.Series([False] * len(df))
        lookback_1w_mask = pd.Series([False] * len(df))
        forward_mask = pd.Series([False] * len(df))

        for _, row in lookback_Nw_weeks.iterrows():
            lookback_Nw_mask |= (df["_year"] == row["_year"]) & (df["_week"] == row["_week"])

        for _, row in lookback_1w_weeks.iterrows():
            lookback_1w_mask |= (df["_year"] == row["_year"]) & (df["_week"] == row["_week"])

        for _, row in forward_weeks.iterrows():
            forward_mask |= (df["_year"] == row["_year"]) & (df["_week"] == row["_week"])

        lookback_Nw_df = df[lookback_Nw_mask]
        lookback_1w_df = df[lookback_1w_mask]
        forward_df = df[forward_mask]

        if len(target_df) == 0:
            continue

        week_start = target_df.index.min()

        for hour in range(24):
            for day in range(7):
                hour_1w_df = lookback_1w_df[lookback_1w_df["_hour"] == hour]

                hour_day_Nw_df = lookback_Nw_df[
                    (lookback_Nw_df["_hour"] == hour) & (lookback_Nw_df["_day_of_week"] == day)
                ]

                forward_hour_day_df = forward_df[
                    (forward_df["_hour"] == hour) & (forward_df["_day_of_week"] == day)
                ]

                avg_rets_1w = {}
                for etf in etf_list:
                    col = close_cols[etf]
                    if len(hour_1w_df) > 1:
                        prices = hour_1w_df[col].values
                        with np.errstate(divide="ignore", invalid="ignore"):
                            rets = np.diff(prices) / prices[:-1]
                        avg_rets_1w[etf] = np.nanmean(rets)
                    else:
                        avg_rets_1w[etf] = np.nan

                avg_rets_Nw = {}
                for etf in etf_list:
                    col = close_cols[etf]
                    if len(hour_day_Nw_df) > 1:
                        prices = hour_day_Nw_df[col].values
                        with np.errstate(divide="ignore", invalid="ignore"):
                            rets = np.diff(prices) / prices[:-1]
                        avg_rets_Nw[etf] = np.nanmean(rets)
                    else:
                        avg_rets_Nw[etf] = np.nan

                actual_rets = {}
                for etf in etf_list:
                    col = close_cols[etf]
                    if len(forward_hour_day_df) > 1:
                        prices = forward_hour_day_df[col].values
                        with np.errstate(divide="ignore", invalid="ignore"):
                            rets = np.diff(prices) / prices[:-1]
                        actual_rets[etf] = np.nanmean(rets)
                    else:
                        actual_rets[etf] = np.nan

                valid_1w = {k: v for k, v in avg_rets_1w.items() if not np.isnan(v)}
                best_1w = max(valid_1w, key=valid_1w.get)

                valid_Nw = {k: v for k, v in avg_rets_Nw.items() if not np.isnan(v)}
                best_Nw = max(valid_Nw, key=valid_Nw.get)

                valid_actual = {k: v for k, v in actual_rets.items() if not np.isnan(v)}
                actual_best = max(valid_actual, key=valid_actual.get())

                row = {
                    "week_start": week_start,
                    "hour": hour,
                    "day_of_week": day,
                    "best_1w": best_1w,
                    "best_Nw": best_Nw,
                    "actual_best": actual_best,
                }

                for etf in etf_list:
                    row[f"actual_ret_{etf}"] = actual_rets.get(etf, np.nan)

                row["hit_1w"] = 1 if best_1w and actual_best and best_1w == actual_best else 0
                row["hit_Nw"] = 1 if best_Nw and actual_best and best_Nw == actual_best else 0

                results.append(row)

    result_df = pd.DataFrame(results)
    return result_df


def analyze_performance_persistence(
    p_persistence_df: pd.DataFrame,
    p_etf_list: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Analyze persistence of hourly/daily outperformance.

    Args:
        p_persistence_df: DataFrame from compute_weekly_performance_persistence()
        p_etf_list: List of ETF symbols

    Returns:
        DataFrame with persistence statistics:
        - hit_rate_1w: % of times 1-week lookback predicted correctly
        - hit_rate_Nw: % of times N-week lookback predicted correctly
        - by_hour: hit rates by hour of day
        - by_day: hit rates by day of week
        - by_hour_day: hit rates by (hour, day) combination
    """
    etf_list = p_etf_list or ETF_LIST
    df = p_persistence_df.copy()

    stats = {}

    overall_1w = df["hit_1w"].mean()
    overall_Nw = df["hit_Nw"].mean()

    stats["overall"] = {
        "hit_rate_1w": overall_1w,
        "hit_rate_Nw": overall_Nw,
        "n_samples": len(df),
    }

    by_hour = df.groupby("hour").agg(
        hit_rate_1w=("hit_1w", "mean"),
        hit_rate_Nw=("hit_Nw", "mean"),
        n_samples=("hit_1w", "count"),
    )
    stats["by_hour"] = by_hour

    by_day = df.groupby("day_of_week").agg(
        hit_rate_1w=("hit_1w", "mean"),
        hit_rate_Nw=("hit_Nw", "mean"),
        n_samples=("hit_1w", "count"),
    )
    stats["by_day"] = by_day

    by_hour_day = df.groupby(["hour", "day_of_week"]).agg(
        hit_rate_1w=("hit_1w", "mean"),
        hit_rate_Nw=("hit_Nw", "mean"),
        n_samples=("hit_1w", "count"),
    )
    stats["by_hour_day"] = by_hour_day

    best_predictions = {}
    for col in ["best_1w", "best_Nw"]:
        pred_counts = df[col].value_counts()
        best_predictions[col] = pred_counts.to_dict()
    stats["prediction_frequency"] = best_predictions

    actual_counts = df["actual_best"].value_counts()
    stats["actual_frequency"] = actual_counts.to_dict()

    avg_ret_by_pred = {}
    for etf in etf_list:
        mask_1w = df["best_1w"] == etf
        mask_Nw = df["best_Nw"] == etf

        if mask_1w.sum() > 0:
            avg_ret_1w = df.loc[mask_1w, f"actual_ret_{etf}"].mean()
            avg_ret_by_pred[f"pred_1w_{etf}_actual_ret"] = avg_ret_1w

        if mask_Nw.sum() > 0:
            avg_ret_Nw = df.loc[mask_Nw, f"actual_ret_{etf}"].mean()
            avg_ret_by_pred[f"pred_Nw_{etf}_actual_ret"] = avg_ret_Nw

    stats["avg_return_when_predicted"] = avg_ret_by_pred

    return stats

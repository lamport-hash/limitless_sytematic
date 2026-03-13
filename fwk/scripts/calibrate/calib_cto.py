#!/usr/bin/env python3
"""
Calibrate CTO Line parameters on forex data.

Performs grid search over CTO parameters (v1, m1, m2, v2) and generates
heatmaps to visualize parameter performance and find stable regions.

Usage:
    python scripts/calibrate/calib_cto.py
    python scripts/calibrate/calib_cto.py --data_pct 0.05
    python scripts/calibrate/calib_cto.py --data_pct 0.05 --v1_start 10 --v1_end 25
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple, List, Dict, Any
from itertools import product
import warnings

import pandas as pd
import numpy as np
import pandas_ta as ta
from backtesting import Backtest, Strategy
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from features.base_dataframe import BaseDataFrame
from features.features_utils import FeatureType
from norm.norm_utils import load_normalized_df
from core.enums import (
    g_open_col,
    g_high_col,
    g_low_col,
    g_close_col,
    g_volume_col,
    g_index_col,
)

warnings.filterwarnings('ignore')

DATA_DIR = Path("/home/brian/limitless_sytematic/fwk/data/normalised/candle_1min/firstrate_undefined/spot/EURUSD")


def get_data_path():
    """Get path to data file by scanning DATA_DIR for available .parquet files."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")
    
    parquet_files = list(DATA_DIR.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found in {DATA_DIR}")
    
    return parquet_files[0]


class CtoLineStrategy(Strategy):
    """CTO Line strategy for calibration."""
    
    atr_mult: float = 2.0
    rr: float = 2.0
    direction: str = "both"
    
    def init(self):
        pass
    
    def next(self):
        if self.position:
            return
        
        close = float(self.data.Close[-1])
        sig = int(self.data.df["signal"].iloc[-1])
        atr = float(self.data.df["ATR"].iloc[-1])
        
        if not np.isfinite(atr) or atr <= 0:
            return
        
        sl_dist = self.atr_mult * atr
        
        if sig == 1:
            sl = close - sl_dist
            tp = close + self.rr * sl_dist
            self.buy(sl=sl, tp=tp)
        
        elif sig == -1:
            sl = close + sl_dist
            tp = close - self.rr * sl_dist
            self.sell(sl=sl, tp=tp)


def load_data(p_data_pct: float = 0.05) -> pd.DataFrame:
    """Load first p_data_pct of data from first available currency pair."""
    data_path = get_data_path()
    
    try:
        df = load_normalized_df(str(data_path))
        n_rows = int(len(df) * p_data_pct)
        df = df.iloc[:n_rows].copy()
        return df
    except Exception as e:
        print(f"Warning: Could not print progress message: {e}")
        return df


def build_features_for_params(
    p_df: pd.DataFrame,
    p_params: Tuple[int, int, int, int],
    p_atr_len: int = 14,
) -> pd.DataFrame:
    """Build features with specific CTO parameters."""
    df = p_df.copy()
    
    bdf = BaseDataFrame(p_df=df)
    bdf.add_feature(FeatureType.CTO_LINE, params=p_params)
    df = bdf.get_dataframe()
    
    df["ATR"] = ta.atr(
        df[g_high_col],
        df[g_low_col],
        df[g_close_col],
        length=p_atr_len
    )
    
    df["signal"] = 0
    long_col = "F_cto_line_long_f16"
    short_col = "F_cto_line_short_f16"
    
    if long_col in df.columns:
        df.loc[df[long_col] == 1, "signal"] = 1
    if short_col in df.columns:
        df.loc[df[short_col] == 1, "signal"] = -1
    
    return df


def convert_to_ohlcv(p_df: pd.DataFrame) -> pd.DataFrame:
    """Convert to OHLCV format for backtesting."""
    ohlcv_df = pd.DataFrame(index=p_df.index)
    
    ohlcv_df["Open"] = p_df[g_open_col]
    ohlcv_df["High"] = p_df[g_high_col]
    ohlcv_df["Low"] = p_df[g_low_col]
    ohlcv_df["Close"] = p_df[g_close_col]
    ohlcv_df["Volume"] = p_df[g_volume_col]
    ohlcv_df["signal"] = p_df["signal"]
    ohlcv_df["ATR"] = p_df["ATR"]
    
    if g_index_col in p_df.columns:
        base = pd.Timestamp("2000-01-01")
        ohlcv_df.index = base + pd.to_timedelta(p_df[g_index_col], unit="m")
    
    ohlcv_df = ohlcv_df.dropna(subset=["Open", "High", "Low", "Close", "ATR"])
    return ohlcv_df


def run_single_backtest(
    p_df: pd.DataFrame,
    p_params: Tuple[int, int, int, int],
    p_cash: float = 100_000,
    p_commission: float = 0.0002,
    p_atr_mult: float = 2.0,
    p_rr: float = 2.0,
) -> Dict[str, float]:
    """Run single backtest and return metrics."""
    try:
        df = build_features_for_params(p_df, p_params)
        ohlcv_df = convert_to_ohlcv(df)
        
        if len(ohlcv_df) < 100:
            return {"return": -100, "sharpe": -10, "win_rate": 0, "trades": 0}
        
        bt = Backtest(
            ohlcv_df,
            CtoLineStrategy,
            cash=p_cash,
            commission=p_commission,
            trade_on_close=True,
            hedging=False,
            exclusive_orders=False,
        )
        
        stats = bt.run(atr_mult=p_atr_mult, rr=p_rr)
        
        return {
            "return": stats['Return [%]'],
            "sharpe": stats['Sharpe Ratio'] if np.isfinite(stats['Sharpe Ratio']) else -10,
            "win_rate": stats['Win Rate [%]'],
            "trades": stats['# Trades'],
            "max_dd": stats["Max. Drawdown [%]"],
            "profit_factor": stats['Profit Factor'] if np.isfinite(stats['Profit Factor']) else 0,
        }
    except Exception as e:
        return {"return": -100, "sharpe": -10, "win_rate": 0, "trades": 0}


def grid_search(
    p_df: pd.DataFrame,
    p_v1_range: Tuple[int, int, int] = (5, 30, 3),
    p_m1_range: Tuple[int, int, int] = (10, 40, 3),
    p_m2_range: Tuple[int, int, int] = (15, 50, 3),
    p_v2_range: Tuple[int, int, int] = (20, 60, 3),
    p_verbose: bool = True,
) -> pd.DataFrame:
    """
    Perform grid search over CTO parameters.
    
    Args:
        p_df: DataFrame with OHLCV data
        p_v1_range: (start, end, step) for v1
        p_m1_range: (start, end, step) for m1
        p_m2_range: (start, end, step) for m2
        p_v2_range: (start, end, step) for v2
    
    Returns:
        DataFrame with results
    """
    v1_vals = list(range(p_v1_range[0], p_v1_range[1] + 1, p_v1_range[2]))
    m1_vals = list(range(p_m1_range[0], p_m1_range[1] + 1, p_m1_range[2]))
    m2_vals = list(range(p_m2_range[0], p_m2_range[1] + 1, p_m2_range[2]))
    v2_vals = list(range(p_v2_range[0], p_v2_range[1] + 1, p_v2_range[2]))
    
    total_combos = len(v1_vals) * len(m1_vals) * len(m2_vals) * len(v2_vals)
    
    if p_verbose:
        print(f"\nGrid search: {total_combos} combinations")
        print(f"  v1: {v1_vals}")
        print(f"  m1: {m1_vals}")
        print(f"  m2: {m2_vals}")
        print(f"  v2: {v2_vals}")
    
    results = []
    
    for i, (v1, m1, m2, v2) in enumerate(product(v1_vals, m1_vals, m2_vals, v2_vals)):
        if not (v1 < m1 < m2 < v2):
            continue
        
        metrics = run_single_backtest(p_df, (v1, m1, m2, v2))
        
        results.append({
            "v1": v1, "m1": m1, "m2": m2, "v2": v2,
            **metrics
        })
        
        if p_verbose and (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{total_combos} ({100*(i+1)/total_combos:.1f}%)")
    
    return pd.DataFrame(results)


def create_heatmaps(p_results: pd.DataFrame, p_output_dir: Path, p_metric: str = "return"):
    """Create 2D heatmaps for parameter combinations."""
    p_output_dir.mkdir(parents=True, exist_ok=True)
    
    results = p_results.copy()
    
    if len(results) == 0:
        print("No results to plot")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'CTO Line Parameter Optimization - {p_metric.upper()}', fontsize=16)
    
    param_pairs = [
        ("v1", "m1"),
        ("v1", "m2"),
        ("v1", "v2"),
        ("m1", "m2"),
        ("m1", "v2"),
        ("m2", "v2"),
    ]
    
    for ax, (p1, p2) in zip(axes.flatten(), param_pairs):
        other_params = [p for p in ["v1", "m1", "m2", "v2"] if p not in [p1, p2]]
        
        median_vals = {p: results[p].median() for p in other_params}
        
        filtered = results.copy()
        for p in other_params:
            p_range = results[p].max() - results[p].min()
            if p_range > 0:
                tolerance = p_range * 0.15
                filtered = filtered[abs(filtered[p] - median_vals[p]) <= tolerance]
        
        if len(filtered) < 10:
            ax.text(0.5, 0.5, "Insufficient data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{p1} vs {p2}")
            continue
        
        pivot = filtered.pivot_table(
            values=p_metric,
            index=p2,
            columns=p1,
            aggfunc='mean'
        )
        
        sns.heatmap(pivot, ax=ax, cmap='RdYlGn', center=0, 
                    fmt='.1f', cbar_kws={'label': p_metric})
        ax.set_title(f"{p1} vs {p2}")
        ax.set_xlabel(p1)
        ax.set_ylabel(p2)
    
    plt.tight_layout()
    output_file = p_output_dir / f"heatmap_{p_metric}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def create_stability_analysis(p_results: pd.DataFrame, p_output_dir: Path):
    """Analyze parameter stability - find regions where small changes don't affect performance."""
    p_output_dir.mkdir(parents=True, exist_ok=True)
    
    results = p_results.copy()
    
    if len(results) == 0:
        print("No results for stability analysis")
        return results
    
    results["stability_score"] = 0.0
    
    for idx in results.index:
        row = results.loc[idx]
        
        neighbors = results[
            (abs(results["v1"] - row["v1"]) <= 3) &
            (abs(results["m1"] - row["m1"]) <= 3) &
            (abs(results["m2"] - row["m2"]) <= 3) &
            (abs(results["v2"] - row["v2"]) <= 3) &
            (results.index != idx)
        ]
        
        if len(neighbors) > 0:
            var_return = neighbors["return"].var() if neighbors["return"].var() > 0 else 0
            var_sharpe = neighbors["sharpe"].var() if neighbors["sharpe"].var() > 0 else 0
            stability = -var_return / 100 - var_sharpe
            results.loc[idx, "stability_score"] = stability
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax = axes[0]
    scatter = ax.scatter(
        results["return"], 
        results["stability_score"],
        c=results["sharpe"],
        cmap='viridis',
        alpha=0.6
    )
    ax.set_xlabel("Return [%]")
    ax.set_ylabel("Stability Score (higher = more stable)")
    ax.set_title("Return vs Stability")
    plt.colorbar(scatter, ax=ax, label='Sharpe')
    
    stability_75 = results["stability_score"].quantile(0.75)
    return_75 = results["return"].quantile(0.75)
    ax.axhline(y=stability_75, color='r', linestyle='--', label='75th percentile stability')
    ax.axvline(x=return_75, color='g', linestyle='--', label='75th percentile return')
    ax.legend()
    
    ax = axes[1]
    top_stable = results.nlargest(20, "stability_score")
    ax.barh(range(len(top_stable)), top_stable["stability_score"])
    ax.set_yticks(range(len(top_stable)))
    ax.set_yticklabels([f"({int(r.v1)},{int(r.m1)},{int(r.m2)},{int(r.v2)})" for _, r in top_stable.iterrows()])
    ax.set_xlabel("Stability Score")
    ax.set_title("Top 20 Most Stable Parameter Sets")
    
    plt.tight_layout()
    output_file = p_output_dir / "stability_analysis.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")
    
    return results


def create_summary_report(p_results: pd.DataFrame, p_output_dir: Path):
    """Create summary report of best parameters."""
    p_output_dir.mkdir(parents=True, exist_ok=True)
    
    results = p_results.copy()
    
    if len(results) == 0:
        print("No results for summary")
        return
    
    results["combined_score"] = (
        results["return"] / 100 + 
        results["sharpe"] * 0.5 + 
        results["win_rate"] / 100
    )
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax = axes[0, 0]
    top_return = results.nlargest(15, "return")
    ax.barh(range(len(top_return)), top_return["return"])
    ax.set_yticks(range(len(top_return)))
    ax.set_yticklabels([
        f"({r.v1},{r.m1},{r.m2},{r.v2})"
        for _, r in top_return.iterrows()
    ])
    ax.set_title("Top 15 by Return")
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    ax = axes[0, 1]
    top_sharpe = results.nlargest(15, "sharpe")
    ax.barh(range(len(top_sharpe)), top_sharpe["sharpe"])
    ax.set_yticks(range(len(top_sharpe)))
    ax.set_yticklabels([
        f"({r.v1},{r.m1},{r.m2},{r.v2})"
        for _, r in top_sharpe.iterrows()
    ])
    ax.set_xlabel("Sharpe Ratio")
    ax.set_title("Top 15 by Sharpe Ratio")
    
    ax = axes[1, 0]
    ax.scatter(results["trades"], results["return"], alpha=0.5, c=results["sharpe"], cmap='viridis')
    ax.set_xlabel("Number of Trades")
    ax.set_ylabel("Return [%]")
    ax.set_title("Return vs Number of Trades")
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    ax = axes[1, 1]
    ax.scatter(results["return"], results["max_dd"], alpha=0.5, c=results["sharpe"], cmap="viridis")
    ax.set_xlabel("Return [%]")
    ax.set_ylabel("Max Drawdown [%]")
    ax.set_title("Return vs Max Drawdown")
    ax.axhline(y=results["max_dd"].median(), color="r", linestyle="--", alpha=0.5)
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
    
    plt.tight_layout()
    output_file = p_output_dir / "summary_report.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")
    
    print("\n" + "=" * 80)
    print("TOP 10 PARAMETER SETS (by Combined Score)")
    print("=" * 80)
    top_combined = results.nlargest(10, "combined_score")
    print(top_combined[["v1", "m1", "m2", "v2", "return", "sharpe", "win_rate", "trades", "combined_score"]].to_string())
    
    results.to_csv(p_output_dir / "calibration_results.csv", index=False)
    print(f"\nSaved full results to: {p_output_dir / 'calibration_results.csv'}")
    
    return results


def main(
    p_data_pct: float = 0.05,
    p_v1_range: Tuple[int, int, int] = (5, 30, 3),
    p_m1_range: Tuple[int, int, int] = (10, 40, 3),
    p_m2_range: Tuple[int, int, int] = (15, 50, 3),
    p_v2_range: Tuple[int, int, int] = (20, 60, 3),
    p_output_dir: str = "output/calib_cto",
    p_verbose: bool = True,
):
    """Main calibration function."""
    
    output_dir = Path(p_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("CTO LINE PARAMETER CALIBRATION")
    print("=" * 80)
    
    df = load_data(p_data_pct)
    
    results = grid_search(
        p_df=df,
        p_v1_range=p_v1_range,
        p_m1_range=p_m1_range,
        p_m2_range=p_m2_range,
        p_v2_range=p_v2_range,
        p_verbose=p_verbose,
    )
    
    print(f"\nTotal valid combinations tested: {len(results)}")
    
    print("\n1. Creating return heatmaps...")
    create_heatmaps(results, output_dir, p_metric="return")
    
    print("\n2. Creating Sharpe heatmaps...")
    create_heatmaps(results, output_dir, p_metric="sharpe")
    
    print("\n3. Creating stability analysis...")
    results = create_stability_analysis(results, output_dir)
    
    print("\n4. Creating summary report...")
    results = create_summary_report(results, output_dir)
    
    print("\n" + "=" * 80)
    print("CALIBRATION COMPLETE")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate CTO Line parameters")
    parser.add_argument(
        "--data_pct",
        type=float,
        default=0.05,
        help="Percentage of data to use (default: 0.05 = 5%)")
    parser.add_argument(
        "--v1_start", type=int, default=5, help="v1 range start (default: 5)"
    )
    parser.add_argument(
        "--v1_end", type=int, default=30, help="v1 range end (default: 30)"
    )
    parser.add_argument(
        "--v1_step", type=int, default=3, help="v1 range step (default: 3)"
    )
    parser.add_argument(
        "--m1_start", type=int, default=10, help="m1 range start (default: 10)"
    )
    parser.add_argument(
        "--m1_end", type=int, default=40, help="m1 range end (default: 40)"
    )
    parser.add_argument(
        "--m1_step", type=int, default=3, help="m1 range step (default: 3)"
    )
    parser.add_argument(
        "--m2_start", type=int, default=15, help="m2 range start (default: 15)"
    )
    parser.add_argument(
        "--m2_end", type=int, default=50, help="m2 range end (default: 50)"
    )
    parser.add_argument(
        "--m2_step", type=int, default=3, help="m2 range step (default: 3)"
    )
    parser.add_argument(
        "--v2_start", type=int, default=20, help="v2 range start (default: 20)"
    )
    parser.add_argument(
        "--v2_end", type=int, default=60, help="v2 range end (default: 60)"
    )
    parser.add_argument(
        "--v2_step", type=int, default=3, help="v2 range step (default: 3)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/calib_cto",
        help="Output directory (default: output/calib_cto)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    results = main(
        p_data_pct=args.data_pct,
        p_v1_range=(args.v1_start, args.v1_end, args.v1_step),
        p_m1_range=(args.m1_start, args.m1_end, args.m1_step),
        p_m2_range=(args.m2_start, args.m2_end, args.m2_step),
        p_v2_range=(args.v2_start, args.v2_end, args.v2_step),
        p_output_dir=args.output,
        p_verbose=not args.quiet,
    )

#!/usr/bin/env python
"""
Trend Optimization Script

Runs multi-method trend parameter optimization on 10 major FX pairs.
Saves results to data_work/output/ with timestamp.

Usage:
    uv run python scripts/targets/run_trend_optimization.py
    uv run python scripts/targets/run_trend_optimization.py --n-iter 30
    uv run python scripts/targets/run_trend_optimization.py --methods sma zigzag
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from features.targets_trend_optimise import (
    TrendOptimizer,
    optimize_all_timeframes,
    get_param_grid_for_timeframe,
)
from features.targets_trend import (
    add_sma_trends,
    add_zigzag_trends,
    add_regression_trends,
    add_directional_trends,
)

from norm.norm_utils import load_normalized_df
from core.data_org import NORMALISED_DIR, WORK_DIR
from core.enums import g_close_col


MAJOR_FX_PAIRS = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "USDCHF",
    "AUDUSD",
    "NZDUSD",
    "USDCAD",
    "EURGBP",
    "EURJPY",
    "GBPJPY",
]

DATA_FREQ = "candle_1min"
SOURCE = "firstrate_undefined"
PRODUCT_TYPE = "spot"
METHOD_FUNCTIONS = {
    "sma": add_sma_trends,
    "zigzag": add_zigzag_trends,
    "regression": add_regression_trends,
    "directional": add_directional_trends,
}


def get_fx_data_path(symbol: str) -> Path:
    return NORMALISED_DIR / DATA_FREQ / SOURCE / PRODUCT_TYPE / symbol


def load_fx_pair(symbol: str, max_rows: int = None) -> pd.DataFrame:
    """Load a single FX pair."""
    pair_dir = get_fx_data_path(symbol)
    
    if not pair_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {pair_dir}")
    
    parquet_files = list(pair_dir.glob("*.df.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {pair_dir}")
    
    df = load_normalized_df(str(parquet_files[0]))
    
    if max_rows and len(df) > max_rows:
        df = df.tail(max_rows)
    
    return df


def align_dataframes(dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """Align multiple DataFrames on their common index."""
    if not dfs:
        return dfs
    
    common_index = dfs[0].index
    for df in dfs[1:]:
        common_index = common_index.intersection(df.index)
    
    return [df.loc[common_index].copy() for df in dfs]


def load_all_fx_pairs(max_rows: int = None) -> List[pd.DataFrame]:
    """Load and align all major FX pairs."""
    dfs = []
    loaded_symbols = []
    
    for symbol in MAJOR_FX_PAIRS:
        try:
            df = load_fx_pair(symbol, max_rows=max_rows)
            dfs.append(df)
            loaded_symbols.append(symbol)
            print(f"  Loaded {symbol}: {len(df)} rows")
        except FileNotFoundError as e:
            print(f"  Skipping {symbol}: {e}")
    
    if len(dfs) < 5:
        raise ValueError(f"Not enough FX pairs loaded (got {len(dfs)}, need at least 5)")
    
    print(f"\nAligning {len(dfs)} pairs...")
    aligned_dfs = align_dataframes(dfs)
    print(f"  Common rows: {len(aligned_dfs[0])}")
    
    return aligned_dfs, loaded_symbols


def run_single_method_optimization(
    dfs: List[pd.DataFrame],
    method: str,
    timeframe: str,
    n_iter: int,
    max_long: int,
    max_short: int,
    cost: float,
) -> Dict[str, Any]:
    """Run optimization for a single method."""
    print(f"\n{'='*60}")
    print(f"Method: {method.upper()} | Timeframe: {timeframe}")
    print(f"{'='*60}")
    
    trend_function = METHOD_FUNCTIONS[method]
    param_grid = get_param_grid_for_timeframe(method, timeframe)
    
    optimizer = TrendOptimizer(
        dfs=dfs,
        trend_function=trend_function,
        timeframe=timeframe,
        max_long=max_long,
        max_short=max_short,
        cost=cost,
    )
    
    results = optimizer.random_search(
        param_grid,
        n_iter=n_iter,
        n_jobs=-1,
        verbose=True,
    )
    
    backtest = optimizer.backtest(verbose=False)
    
    return {
        "method": method,
        "timeframe": timeframe,
        "best_params": results["best_params"],
        "best_return": float(results["best_return"]),
        "n_tested": results["n_tested"],
        "backtest": {
            "total_return": float(backtest["total_return"]),
            "sharpe_ratio": float(backtest["sharpe_ratio"]),
            "max_drawdown": float(backtest["max_drawdown"]),
            "win_rate": float(backtest["win_rate"]),
            "calmar_ratio": float(backtest["calmar_ratio"]),
            "volatility": float(backtest["volatility"]),
        },
        "statistics": results["statistics"],
    }


def save_results(results: Dict[str, Any], output_dir: Path) -> Path:
    """Save results to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"trend_optimization_{timestamp}.json"
    filepath = output_dir / filename
    
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    return filepath


def print_summary(all_results: List[Dict[str, Any]]):
    """Print summary comparison table."""
    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("=" * 80)
    
    summary_data = []
    for r in all_results:
        summary_data.append({
            "Method": r["method"],
            "Return": f"{r['best_return']:.2%}",
            "Sharpe": f"{r['backtest']['sharpe_ratio']:.2f}",
            "Drawdown": f"{r['backtest']['max_drawdown']:.2%}",
            "Win Rate": f"{r['backtest']['win_rate']:.1%}",
        })
    
    df = pd.DataFrame(summary_data)
    df = df.sort_values("Return", ascending=False)
    print(df.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("BEST OVERALL")
    print("=" * 80)
    best = max(all_results, key=lambda x: x["best_return"])
    print(f"  Method: {best['method']}")
    print(f"  Return: {best['best_return']:.2%}")
    print(f"  Sharpe: {best['backtest']['sharpe_ratio']:.2f}")
    print(f"  Params: {best['best_params']}")


def main():
    parser = argparse.ArgumentParser(description="Run trend optimization on FX pairs")
    parser.add_argument(
        "--n-iter", type=int, default=20,
        help="Number of iterations per method (default: 20)"
    )
    parser.add_argument(
        "--methods", nargs="+", default=["sma", "zigzag"],
        choices=["sma", "zigzag", "regression", "directional"],
        help="Methods to optimize (default: sma zigzag)"
    )
    parser.add_argument(
        "--max-rows", type=int, default=None,
        help="Maximum rows per asset (default: all)"
    )
    parser.add_argument(
        "--max-long", type=int, default=5,
        help="Maximum long positions (default: 5)"
    )
    parser.add_argument(
        "--max-short", type=int, default=2,
        help="Maximum short positions (default: 2)"
    )
    parser.add_argument(
        "--cost", type=float, default=0.0005,
        help="Trading cost per transaction (default: 0.0005 = 5bps)"
    )
    parser.add_argument(
        "--timeframe", type=str, default="1h",
        help="Timeframe for optimization (default: 1h)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("TREND OPTIMIZATION SCRIPT")
    print("=" * 80)
    print(f"Methods: {args.methods}")
    print(f"Iterations: {args.n_iter}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Max Long: {args.max_long}, Max Short: {args.max_short}")
    print(f"Cost: {args.cost:.4%}")
    
    print("\n" + "-" * 40)
    print("Loading FX data...")
    print("-" * 40)
    
    dfs, symbols = load_all_fx_pairs(max_rows=args.max_rows)
    print(f"\nLoaded {len(dfs)} assets: {symbols}")
    
    all_results = []
    for method in args.methods:
        result = run_single_method_optimization(
            dfs=dfs,
            method=method,
            timeframe=args.timeframe,
            n_iter=args.n_iter,
            max_long=args.max_long,
            max_short=args.max_short,
            cost=args.cost,
        )
        all_results.append(result)
    
    print_summary(all_results)
    
    output_dir = WORK_DIR / "output"
    output_file = save_results(
        {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "n_iter": args.n_iter,
                "methods": args.methods,
                "max_long": args.max_long,
                "max_short": args.max_short,
                "cost": args.cost,
                "timeframe": args.timeframe,
                "symbols": symbols,
                "n_assets": len(dfs),
                "n_periods": len(dfs[0]),
            },
            "results": all_results,
        },
        output_dir,
    )
    
    print(f"\nResults saved to: {output_file}")
    print("\nDone!")


if __name__ == "__main__":
    main()

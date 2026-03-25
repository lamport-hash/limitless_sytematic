#!/usr/bin/env python3
"""
Run ATR Open Range Breakout Strategy.

Usage:
    python scripts/strat/run_atr_breakout.py
    python scripts/strat/run_atr_breakout.py --direction long
    python scripts/strat/run_atr_breakout.py --atr_mult 0.3 --stop_mult 0.2
    python scripts/strat/run_atr_breakout.py --no-gap-bias --no-vol-filter
    python scripts/strat/run_atr_breakout.py --symbol GBPUSD
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strat.s_atr_breakout import main


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ATR Open Range Breakout Strategy"
    )
    
    parser.add_argument(
        "--symbol",
        type=str,
        default="EURUSD",
        help="Instrument symbol (default: EURUSD)",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="both",
        choices=["long", "short", "both"],
        help="Trading direction (default: both)",
    )
    parser.add_argument(
        "--atr_len",
        type=int,
        default=14,
        help="ATR calculation period (default: 14)",
    )
    parser.add_argument(
        "--atr_mult",
        type=float,
        default=0.5,
        help="ATR multiplier for breakout levels (default: 0.5)",
    )
    parser.add_argument(
        "--stop_mult",
        type=float,
        default=0.3,
        help="Stop loss in ATR units (default: 0.3)",
    )
    parser.add_argument(
        "--vol_ma_len",
        type=int,
        default=20,
        help="Volatility MA period (default: 20)",
    )
    parser.add_argument(
        "--gap_threshold",
        type=float,
        default=0.002,
        help="Minimum gap %% for directional bias (default: 0.002 = 0.2%%)",
    )
    parser.add_argument(
        "--no-gap-bias",
        action="store_true",
        help="Disable overnight gap directional bias",
    )
    parser.add_argument(
        "--no-vol-filter",
        action="store_true",
        help="Disable volatility filter",
    )
    parser.add_argument(
        "--no-exit-eod",
        action="store_true",
        help="Disable end-of-day exit (keep positions overnight)",
    )
    parser.add_argument(
        "--cash",
        type=float,
        default=100_000,
        help="Initial cash (default: 100000)",
    )
    parser.add_argument(
        "--commission",
        type=float,
        default=0.00002,
        help="Commission per trade (default: 0.00002)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date YYYYMMDD",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date YYYYMMDD",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output",
    )
    return parser.parse_args()


def main_cli():
    args = parse_args()
    
    df, stats = main(
        p_symbol=args.symbol,
        p_atr_len=args.atr_len,
        p_atr_mult=args.atr_mult,
        p_stop_mult=args.stop_mult,
        p_vol_ma_len=args.vol_ma_len,
        p_gap_threshold=args.gap_threshold,
        p_use_gap_bias=not args.no_gap_bias,
        p_use_vol_filter=not args.no_vol_filter,
        p_direction=args.direction,
        p_exit_eod=not args.no_exit_eod,
        p_cash=args.cash,
        p_commission=args.commission,
        p_verbose=not args.quiet,
        p_start=args.start,
        p_end=args.end,
    )
    
    if args.quiet:
        print(f"Return: {stats['Return [%]']:.2f}%")
        print(f"Sharpe: {stats['Sharpe Ratio']:.2f}")
        print(f"Win Rate: {stats['Win Rate [%]']:.2f}%")
        print(f"Profit Factor: {stats['Profit Factor']:.2f}")
        print(f"# Trades: {stats['# Trades']}")
    
    return df, stats


if __name__ == "__main__":
    main_cli()

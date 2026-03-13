#!/usr/bin/env python3
"""
Run Total Signal Pattern Strategy on QQQ ETF.

Usage:
    python scripts/strat/run_total_signal.py
    python scripts/strat/run_total_signal.py --direction long
    python scripts/strat/run_total_signal.py --direction short --atr_mult 2.5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strat.s_total_signal import main


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Total Signal Pattern Strategy on QQQ ETF"
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="both",
        choices=["long", "short", "both"],
        help="Trading direction (default: both)",
    )
    parser.add_argument(
        "--atr_mult",
        type=float,
        default=2.0,
        help="ATR multiplier for stop loss (default: 2.0)",
    )
    parser.add_argument(
        "--rr",
        type=float,
        default=2.0,
        help="Risk-reward ratio for take profit (default: 2.0)",
    )
    parser.add_argument(
        "--atr_len",
        type=int,
        default=14,
        help="ATR calculation period (default: 14)",
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
        default=0.0002,
        help="Commission per trade (default: 0.0002)",
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
        p_direction=args.direction,
        p_atr_mult=args.atr_mult,
        p_rr=args.rr,
        p_atr_len=args.atr_len,
        p_cash=args.cash,
        p_commission=args.commission,
        p_verbose=not args.quiet,
    )
    
    if args.quiet:
        print(f"Return: {stats['Return [%]']:.2f}%")
        print(f"Sharpe: {stats['Sharpe Ratio']:.2f}")
        print(f"Win Rate: {stats['Win Rate [%]']:.2f}%")
        print(f"Profit Factor: {stats['Profit Factor']:.2f}")
    
    return df, stats


if __name__ == "__main__":
    main_cli()

#!/usr/bin/env python3
"""Inspect i_minute_i values from a bundle file."""

import sys
import pandas as pd
import numpy as np
from pathlib import Path


def inspect_bundle(p_path: str):
    path = Path(p_path)
    if not path.exists():
        print(f"File not found: {path}")
        return

    df = pd.read_parquet(path)

    print(f"File: {path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nIndex type: {type(df.index)}")
    print(f"Index dtype: {df.index.dtype}")

    print("\n--- First 10 index values ---")
    for i, val in enumerate(df.index[:10]):
        print(f"  [{i}] raw={val} type={type(val)}")

    print("\n--- Last 10 index values ---")
    for i, val in enumerate(df.index[-10:]):
        print(f"  [{i}] raw={val} type={type(val)}")

    print("\n--- Trying different conversions on first value ---")
    first_val = df.index[0]
    print(f"Raw value: {first_val}")
    print(f"As int64: {np.int64(first_val)}")

    val_int = np.int64(first_val)

    print(f"\n/ 1e9 = {val_int / 1e9}")
    print(f"/ 1e7 = {val_int / 1e7}")
    print(f"/ 1e6 = {val_int / 1e6}")

    print(f"\n--- Assuming nanoseconds from 2000-01-01 ---")
    ts_ns = pd.Timestamp("2000-01-01") + pd.Timedelta(int(val_int), unit="ns")
    print(f"+ ns: {ts_ns}")

    ts_us = pd.Timestamp("2000-01-01") + pd.Timedelta(int(val_int), unit="us")
    print(f"+ us: {ts_us}")

    ts_ms = pd.Timestamp("2000-01-01") + pd.Timedelta(int(val_int), unit="ms")
    print(f"+ ms: {ts_ms}")

    ts_s = pd.Timestamp("2000-01-01") + pd.Timedelta(int(val_int), unit="s")
    print(f"+ s: {ts_s}")

    print(f"\n--- Dividing first ---")
    ts_div1e9 = pd.Timestamp("2000-01-01") + pd.Timedelta(val_int / 1e9, unit="s")
    print(f"val/1e9 as seconds: {ts_div1e9}")

    ts_div1e7 = pd.Timestamp("2000-01-01") + pd.Timedelta(val_int / 1e7, unit="s")
    print(f"val/1e7 as seconds: {ts_div1e7}")

    ts_div1e6 = pd.Timestamp("2000-01-01") + pd.Timedelta(val_int / 1e6, unit="s")
    print(f"val/1e6 as seconds: {ts_div1e6}")

    print(f"\n--- Unix epoch interpretation ---")
    ts_unix_ns = pd.Timestamp(int(val_int), unit="ns", tz="UTC")
    print(f"Unix ns: {ts_unix_ns}")

    ts_unix_us = pd.Timestamp(int(val_int), unit="us", tz="UTC")
    print(f"Unix us: {ts_unix_us}")

    ts_unix_ms = pd.Timestamp(int(val_int), unit="ms", tz="UTC")
    print(f"Unix ms: {ts_unix_ms}")

    ts_unix_s = pd.Timestamp(int(val_int), unit="s", tz="UTC")
    print(f"Unix s: {ts_unix_s}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_index.py <bundle_path>")
        print(
            "Example: python inspect_index.py /home/brian/sing/data/bundle/etf_features_bundle_momentum.parquet"
        )
        sys.exit(1)

    inspect_bundle(sys.argv[1])

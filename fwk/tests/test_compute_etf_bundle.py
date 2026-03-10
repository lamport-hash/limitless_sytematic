"""
Unit tests for ETF feature bundle computation.

Tests building features for ETFs and merging them into a bundle using the fwk modules.
"""

import pytest
import pandas as pd
from pathlib import Path
from typing import List

from features.base_dataframe import BaseDataFrame
from features.features_utils import FeatureType
from merger.merger_utils import merge_multiple_dataframes_from_parquet
from core.data_org import (
    BUNDLE_DIR,
    get_normalised_instrument_dir,
    MktDataFred,
    ExchangeNAME,
    ProductType,
)
from core.enums import g_index_col
from norm.norm_utils import load_normalized_df


DATA_FREQ = MktDataFred.CANDLE_1HOUR
SOURCE = ExchangeNAME.FIRSTRATE
PRODUCT_TYPE = ProductType.ETF

TEST_ETFS = ["QQQ"]

FEATURE_TYPES = [
    FeatureType.PRICE,
    FeatureType.RETURN,
    FeatureType.LOG_RETURN,
    FeatureType.VOLUME,
    FeatureType.HIST_VOLATILITY,
    FeatureType.LAG_DELTAS,
    FeatureType.RSI,
    FeatureType.EMA,
    FeatureType.SPREAD_REL_EMA,
    FeatureType.DIFF_REL_EMA_MID,
    FeatureType.ADI,
    FeatureType.ROC,
]


def compute_features_for_etf(etf_symbol: str) -> Path:
    """
    Compute features for a single ETF and save to bundle directory.

    Args:
        etf_symbol: ETF symbol (e.g., "QQQ", "SPY")

    Returns:
        Path to the saved feature parquet file
    """
    instrument_dir = get_normalised_instrument_dir(
        DATA_FREQ, SOURCE, PRODUCT_TYPE, etf_symbol
    )

    parquet_files = list(instrument_dir.glob("*.df.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {instrument_dir}")

    input_file = parquet_files[0]
    print(f"Loading: {input_file}")

    df = load_normalized_df(str(input_file))
    print(f"  Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    base_df = BaseDataFrame(
        p_df=df,
        p_valid_col_name="valid_row",
        p_scaling=-1,
        p_verbose=False,
    )

    for feature_type in FEATURE_TYPES:
        base_df.add_feature(feature_type)

    base_df.convert_f16_columns()

    result_df = base_df.get_dataframe()
    feature_cols = base_df.get_feature_columns()

    print(f"  Result: {result_df.shape[0]} rows, {len(feature_cols)} features")

    output_dir = BUNDLE_DIR / DATA_FREQ.value
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{etf_symbol}_features.parquet"

    result_df.to_parquet(output_file, index=False)
    print(f"  Saved: {output_file}")

    return output_file


def compute_etf_bundle(
    etf_list: List[str],
    output_prefix: str = "test_etf",
    output_dir: Path = BUNDLE_DIR,
) -> Path:
    """
    Compute features for multiple ETFs and merge into a single bundle.

    Args:
        etf_list: List of ETF symbols to process
        output_prefix: Prefix for output filename
        output_dir: Output directory (defaults to BUNDLE_DIR)

    Returns:
        Path to the merged bundle file
    """
    if output_dir is None:
        output_dir = BUNDLE_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    feature_files = []
    for etf in etf_list:
        print(f"\n{'='*60}")
        print(f"Processing: {etf}")
        print(f"{'='*60}")
        feature_file = compute_features_for_etf(etf)
        feature_files.append(feature_file)

    print(f"\n{'='*60}")
    print("Merging ETF feature files...")
    print(f"{'='*60}")

    merged_df = merge_multiple_dataframes_from_parquet(
        file_paths=[str(f) for f in feature_files],
        p_names=etf_list,
        p_cols_list=[],
        p_id_col=g_index_col,
        p_float_16=False,
    )

    print(f"\nMerged dataframe:")
    print(f"  Shape: {merged_df.shape}")

    feature_cols = [c for c in merged_df.columns if "_F_" in c]
    print(f"  Total feature columns: {len(feature_cols)}")

    output_path = output_dir / f"{output_prefix}_features_bundle.parquet"
    merged_df.to_parquet(output_path)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n{'='*60}")
    print(f"Bundle saved to: {output_path}")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"{'='*60}")

    return output_path


def test_compute_features_for_single_etf():
    """
    Test computing features for a single ETF (QQQ).
    """
    output_file = compute_features_for_etf("QQQ")

    print(f"\n{'='*60}")
    print(f"Output file: {output_file}")
    print(f"{'='*60}")

    assert output_file.exists(), f"Output file should exist: {output_file}"

    df = pd.read_parquet(output_file)
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()[:10]}...")

    assert df.shape[0] > 0, "DataFrame should have rows"
    
    feature_cols = [c for c in df.columns if c.startswith("F_")]
    assert len(feature_cols) > 0, "Should have feature columns"

    print(f"\n{'='*60}")
    print("TEST PASSED: Single ETF feature computation")
    print(f"{'='*60}")


def test_compute_etf_bundle_small():
    """
    Test computing and merging features for a small set of ETFs.
    """
    output_file = compute_etf_bundle(
        etf_list=TEST_ETFS,
        output_prefix="test_etf",
        output_dir=BUNDLE_DIR,
    )

    print(f"\n{'='*60}")
    print(f"Bundle file: {output_file}")
    print(f"{'='*60}")

    assert output_file.exists(), f"Bundle file should exist: {output_file}"

    df = pd.read_parquet(output_file)
    print(f"Shape: {df.shape}")
    print(f"Index: {df.index.name}")
    
    feature_cols = [c for c in df.columns if "_F_" in c]
    print(f"Feature columns: {len(feature_cols)}")

    assert df.shape[0] > 0, "DataFrame should have rows"
    assert len(feature_cols) > 0, "Should have feature columns"

    for etf in TEST_ETFS:
        etf_feature_cols = [c for c in feature_cols if c.startswith(f"{etf}_")]
        assert len(etf_feature_cols) > 0, f"Should have features for {etf}"

    print(f"\n{'='*60}")
    print("TEST PASSED: ETF bundle computation")
    print(f"{'='*60}")


def test_feature_types_list():
    """
    Verify that the feature types list contains expected features.
    """
    expected_features = [
        FeatureType.PRICE,
        FeatureType.RETURN,
        FeatureType.LOG_RETURN,
        FeatureType.VOLUME,
        FeatureType.HIST_VOLATILITY,
        FeatureType.LAG_DELTAS,
        FeatureType.RSI,
        FeatureType.EMA,
        FeatureType.SPREAD_REL_EMA,
        FeatureType.DIFF_REL_EMA_MID,
        FeatureType.ADI,
        FeatureType.ROC,
    ]

    assert FEATURE_TYPES == expected_features, "Feature types should match expected list"
    print(f"Feature types verified: {len(FEATURE_TYPES)} features")


if __name__ == "__main__":
    test_compute_features_for_single_etf()
    test_compute_etf_bundle_small()
    test_feature_types_list()

"""
Unit tests for ETF feature bundle computation.

Tests building features for ETFs and merging them into a bundle using the fwk modules.
"""

import pytest
import pandas as pd
from pathlib import Path
from typing import List, Optional

from features.features_utils import FeatureType
from core.data_org import (
    BUNDLE_DIR,
    MktDataTFreq,
    ExchangeNAME,
    ProductType,
)
from bundler.feature_bundler import (
    compute_features_for_asset,
    compute_asset_bundle,
    DEFAULT_FEATURE_TYPES,
)
from strat.strat_backtest import ETF_LIST


DATA_FREQ = MktDataTFreq.CANDLE_1HOUR
SOURCE = ExchangeNAME.FIRSTRATE
PRODUCT_TYPE = ProductType.ETF


def compute_features_for_etf(etf_symbol: str) -> Path:
    """
    Compute features for a single ETF and save to bundle directory.

    Args:
        etf_symbol: ETF symbol (e.g., "QQQ", "SPY")

    Returns:
        Path to the saved feature parquet file
    """
    return compute_features_for_asset(
        symbol=etf_symbol,
        freq=DATA_FREQ,
        source=SOURCE,
        product_type=PRODUCT_TYPE,
        feature_types=DEFAULT_FEATURE_TYPES,
        p_verbose=True,
    )


def compute_etf_bundle(
    etf_list: Optional[List[str]] = None,
    output_prefix: str = "test_etf",
    output_dir: Path = BUNDLE_DIR,
) -> Path:
    """
    Compute features for multiple ETFs and merge into a single bundle.

    Args:
        etf_list: List of ETF symbols to process (defaults to ETF_LIST from strat.dual_momentum)
        output_prefix: Prefix for output filename
        output_dir: Output directory (defaults to BUNDLE_DIR)

    Returns:
        Path to the merged bundle file
    """
    if etf_list is None:
        etf_list = ETF_LIST

    product_types = [PRODUCT_TYPE] * len(etf_list)

    return compute_asset_bundle(
        asset_list=etf_list,
        asset_product_type=product_types,
        freq=DATA_FREQ,
        source=SOURCE,
        feature_types=DEFAULT_FEATURE_TYPES,
        output_prefix=output_prefix,
        output_dir=output_dir,
        p_verbose=True,
    )


def test_compute_features_for_single_etf():
    """
    Test computing features for a single ETF (QQQ).
    """
    output_file = compute_features_for_etf("QQQ")

    print(f"\n{'=' * 60}")
    print(f"Output file: {output_file}")
    print(f"{'=' * 60}")

    assert output_file.exists(), f"Output file should exist: {output_file}"

    df = pd.read_parquet(output_file)
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()[:10]}...")

    assert df.shape[0] > 0, "DataFrame should have rows"

    feature_cols = [c for c in df.columns if c.startswith("F_")]
    assert len(feature_cols) > 0, "Should have feature columns"

    print(f"\n{'=' * 60}")
    print("TEST PASSED: Single ETF feature computation")
    print(f"{'=' * 60}")


def test_compute_etf_bundle_small():
    """
    Test computing and merging features for ETFs from ETF_LIST.
    """
    output_file = compute_etf_bundle(
        output_prefix="test_etf",
        output_dir=BUNDLE_DIR,
    )

    print(f"\n{'=' * 60}")
    print(f"Bundle file: {output_file}")
    print(f"{'=' * 60}")

    assert output_file.exists(), f"Bundle file should exist: {output_file}"

    df = pd.read_parquet(output_file)
    print(f"Shape: {df.shape}")
    print(f"Index: {df.index.name}")

    feature_cols = [c for c in df.columns if "_F_" in c]
    print(f"Feature columns: {len(feature_cols)}")

    assert df.shape[0] > 0, "DataFrame should have rows"
    assert len(feature_cols) > 0, "Should have feature columns"

    for etf in ETF_LIST:
        etf_feature_cols = [c for c in feature_cols if c.startswith(f"{etf}_")]
        assert len(etf_feature_cols) > 0, f"Should have features for {etf}"

    print(f"\n{'=' * 60}")
    print("TEST PASSED: ETF bundle computation")
    print(f"{'=' * 60}")


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

    assert DEFAULT_FEATURE_TYPES == expected_features, "Feature types should match expected list"
    print(f"Feature types verified: {len(DEFAULT_FEATURE_TYPES)} features")


if __name__ == "__main__":
    test_compute_features_for_single_etf()
    test_compute_etf_bundle_small()
    test_feature_types_list()

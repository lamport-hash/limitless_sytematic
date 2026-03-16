"""
Feature bundler for computing features across multiple assets and merging into bundles.

This module provides utilities for:
- Loading normalized data for any asset type
- Computing features using BaseDataFrame
- Bundling features from multiple assets into a single merged file
"""

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from features.base_dataframe import BaseDataFrame
from features.features_utils import FeatureType
from merger.merger_utils import merge_multiple_dataframes_from_parquet
from core.data_org import (
    BUNDLE_DIR,
    get_normalised_instrument_dir,
    MktDataTFreq,
    ExchangeNAME,
    ProductType,
)
from core.enums import g_index_col
from norm.norm_utils import load_normalized_df

logger = logging.getLogger(__name__)


DEFAULT_FEATURE_TYPES: List[FeatureType] = [
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


def get_base_df(
    freq: MktDataTFreq = MktDataTFreq.CANDLE_1HOUR,
    source: ExchangeNAME = ExchangeNAME.FIRSTRATE,
    product_type: ProductType = ProductType.ETF,
    symbol: str = "QQQ",
    p_verbose: bool = False,
) -> BaseDataFrame:
    """
    Load normalized data and create a BaseDataFrame for a single asset.

    Args:
        freq: Data frequency (default: CANDLE_1HOUR)
        source: Data source (default: FIRSTRATE)
        product_type: Product type (default: ETF)
        symbol: Asset symbol (default: "QQQ")
        p_verbose: Enable verbose logging (default: False)

    Returns:
        BaseDataFrame instance with loaded data

    Raises:
        FileNotFoundError: If no parquet files found for the asset
    """
    instrument_dir = get_normalised_instrument_dir(freq, source, product_type, symbol)

    parquet_files = list(instrument_dir.glob("*.df.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {instrument_dir}")

    input_file = parquet_files[0]
    if p_verbose:
        logger.info(f"Loading: {input_file}")

    df = load_normalized_df(str(input_file))
    base_df = BaseDataFrame(
        p_df=df,
        p_valid_col_name="valid_row",
        p_scaling=-1,
        p_verbose=p_verbose,
    )
    return base_df


def compute_features_for_asset(
    symbol: str,
    freq: MktDataTFreq = MktDataTFreq.CANDLE_1HOUR,
    source: ExchangeNAME = ExchangeNAME.FIRSTRATE,
    product_type: ProductType = ProductType.ETF,
    feature_types: Optional[List[FeatureType]] = None,
    output_dir: Optional[Path] = None,
    p_verbose: bool = False,
) -> Path:
    """
    Compute features for a single asset and save to bundle directory.

    Args:
        symbol: Asset symbol (e.g., "QQQ", "EURUSD")
        freq: Data frequency (default: CANDLE_1HOUR)
        source: Data source (default: FIRSTRATE)
        product_type: Product type (default: ETF)
        feature_types: List of feature types to compute (default: DEFAULT_FEATURE_TYPES)
        output_dir: Output directory (default: BUNDLE_DIR/<freq>)
        p_verbose: Enable verbose logging (default: False)

    Returns:
        Path to the saved feature parquet file

    Raises:
        FileNotFoundError: If no parquet files found for the asset
    """
    if feature_types is None:
        feature_types = DEFAULT_FEATURE_TYPES

    instrument_dir = get_normalised_instrument_dir(freq, source, product_type, symbol)

    parquet_files = list(instrument_dir.glob("*.df.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {instrument_dir}")

    input_file = parquet_files[0]
    if p_verbose:
        logger.info(f"Loading: {input_file}")

    df = load_normalized_df(str(input_file))
    if p_verbose:
        logger.info(f"  Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    base_df = BaseDataFrame(
        p_df=df,
        p_valid_col_name="valid_row",
        p_scaling=-1,
        p_verbose=p_verbose,
    )

    for feature_type in feature_types:
        base_df.add_feature(feature_type)

    base_df.convert_f16_columns()

    result_df = base_df.get_dataframe()
    feature_cols = base_df.get_feature_columns()

    if p_verbose:
        logger.info(f"  Result: {result_df.shape[0]} rows, {len(feature_cols)} features")

    if output_dir is None:
        output_dir = BUNDLE_DIR / freq.value
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{symbol}_features.parquet"

    result_df.to_parquet(output_file, index=False)
    if p_verbose:
        logger.info(f"  Saved: {output_file}")

    return output_file


def compute_asset_bundle(
    asset_list: List[str],
    asset_product_type: List[ProductType],
    freq: MktDataTFreq = MktDataTFreq.CANDLE_1HOUR,
    source: ExchangeNAME = ExchangeNAME.FIRSTRATE,
    feature_types: Optional[List[FeatureType]] = None,
    output_prefix: str = "asset_bundle",
    output_dir: Optional[Path] = None,
    p_verbose: bool = False,
) -> Path:
    """
    Compute features for multiple assets and merge into a single bundle.

    Args:
        asset_list: List of asset symbols (e.g., ["QQQ", "EURUSD"])
        asset_product_type: List of product types corresponding to each asset
        freq: Data frequency (default: CANDLE_1HOUR)
        source: Data source (default: FIRSTRATE)
        feature_types: List of feature types to compute (default: DEFAULT_FEATURE_TYPES)
        output_prefix: Prefix for output filename (default: "asset_bundle")
        output_dir: Output directory (default: BUNDLE_DIR)
        p_verbose: Enable verbose logging (default: False)

    Returns:
        Path to the merged bundle file

    Raises:
        ValueError: If asset_list and asset_product_type have different lengths
        FileNotFoundError: If no parquet files found for any asset
    """
    if len(asset_list) != len(asset_product_type):
        raise ValueError(
            f"asset_list and asset_product_type must have same length: "
            f"{len(asset_list)} != {len(asset_product_type)}"
        )

    if output_dir is None:
        output_dir = BUNDLE_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    feature_files: List[Path] = []
    for i, symbol in enumerate(asset_list):
        if p_verbose:
            logger.info(f"Processing: {symbol}")
        feature_file = compute_features_for_asset(
            symbol=symbol,
            freq=freq,
            source=source,
            product_type=asset_product_type[i],
            feature_types=feature_types,
            output_dir=output_dir / freq.value,
            p_verbose=p_verbose,
        )
        feature_files.append(feature_file)

    if p_verbose:
        logger.info("Merging all feature files...")

    merged_df = merge_multiple_dataframes_from_parquet(
        file_paths=[str(f) for f in feature_files],
        p_names=asset_list,
        p_cols_list=[],
        p_id_col=g_index_col,
        p_float_16=False,
    )

    if p_verbose:
        logger.info(f"Merged dataframe shape: {merged_df.shape}")

    feature_cols = [c for c in merged_df.columns if "_F_" in c]
    if p_verbose:
        logger.info(f"Total feature columns: {len(feature_cols)}")

    output_path = output_dir / f"{output_prefix}_features_bundle.parquet"
    merged_df.to_parquet(output_path)

    if p_verbose:
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"Bundle saved to: {output_path} ({file_size_mb:.2f} MB)")

    return output_path

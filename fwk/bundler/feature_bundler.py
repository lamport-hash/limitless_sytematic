"""
Feature bundler for computing features across multiple assets and merging into bundles.

This module provides utilities for:
- Loading normalized data for any asset type
- Computing features using BaseDataFrame
- Bundling features from multiple assets into a single merged file
"""

import logging
from pathlib import Path
from typing import List, Optional, Callable, Union

import pandas as pd
from ruamel.yaml import YAML

from features.base_dataframe import BaseDataFrame
from features.features_utils import FeatureType
from merger.merger_utils import merge_multiple_dataframes_from_parquet
from core.data_org import (
    BUNDLE_DIR,
    FEATURE_CONFIGS_DIR,
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


def load_feature_config_from_yaml(config_path: Union[str, Path]) -> List[dict]:
    """
    Load feature configuration from a YAML file.

    Args:
        config_path: Path to YAML feature config file (relative to FEATURE_CONFIGS_DIR or absolute).

    Returns:
        List of feature config dicts with 'type', optional 'periods', and optional 'kwargs'.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config is invalid.
    """
    full_path = FEATURE_CONFIGS_DIR / config_path if isinstance(config_path, str) else Path(config_path)
    if not full_path.exists():
        raise FileNotFoundError(f"Feature config file not found: {full_path}")

    yaml = YAML()
    with open(full_path, "r") as f:
        config = yaml.load(f)

    if not config or "features" not in config:
        raise ValueError(f"No 'features' key found in {full_path}")

    features = config["features"]
    if not features:
        raise ValueError(f"Empty features list in {full_path}")

    return features


def get_default_feature_config_path() -> Path:
    """Get the path to the default feature config file."""
    return FEATURE_CONFIGS_DIR / "default.yaml"


def list_available_feature_configs() -> List[Path]:
    """List all available feature config files."""
    if not FEATURE_CONFIGS_DIR.exists():
        return []
    return sorted(FEATURE_CONFIGS_DIR.glob("*.yaml"))


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
    feature_config_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Path] = None,
    p_verbose: bool = False,
    compute_features: bool = True,
) -> Path:
    """
    Compute features for a single asset and save to bundle directory.

    Args:
        symbol: Asset symbol (e.g., "QQQ", "EURUSD")
        freq: Data frequency (default: CANDLE_1HOUR)
        source: Data source (default: FIRSTRATE)
        product_type: Product type (default: ETF)
        feature_types: List of feature types to compute (legacy, default: DEFAULT_FEATURE_TYPES)
        feature_config_path: Path to YAML config file. Takes precedence over feature_types.
        output_dir: Output directory (default: BUNDLE_DIR/<freq>)
        p_verbose: Enable verbose logging (default: False)
        compute_features: Whether to compute features (default: True). If False, only raw OHLCV.

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

    if compute_features:
        if feature_config_path:
            _add_features_from_config(base_df, feature_config_path)
        elif feature_types:
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


def _add_features_from_config(base_df: BaseDataFrame, config_path: Union[str, Path]) -> None:
    """Load features from a YAML configuration file and add them to the BaseDataFrame."""
    full_path = FEATURE_CONFIGS_DIR / config_path if isinstance(config_path, str) else Path(config_path)
    if not full_path.exists():
        raise FileNotFoundError(f"Feature config file not found: {full_path}")

    yaml = YAML()
    with open(full_path, "r") as f:
        config = yaml.load(f)

    if not config or "features" not in config:
        raise ValueError(f"No 'features' key found in config file: {full_path}")

    features_config = config["features"]
    if not features_config:
        logger.warning(f"Empty features list in {full_path}")
        return

    for feature_config in features_config:
        if not isinstance(feature_config, dict) or "type" not in feature_config:
            logger.warning(f"Invalid feature config: {feature_config}")
            continue

        try:
            feature_type = FeatureType(feature_config["type"])
        except ValueError:
            logger.warning(f"Unknown FeatureType '{feature_config['type']}' in {full_path}")
            continue

        periods = feature_config.get("periods")
        kwargs = feature_config.get("kwargs", {})

        base_df.add_feature(feature_type, periods=periods, **kwargs)


def compute_asset_bundle(
    asset_list: List[str],
    asset_product_type: List[ProductType],
    freq: MktDataTFreq = MktDataTFreq.CANDLE_1HOUR,
    source: ExchangeNAME = ExchangeNAME.FIRSTRATE,
    feature_types: Optional[List[FeatureType]] = None,
    feature_config_path: Optional[Union[str, Path]] = None,
    output_prefix: str = "asset_bundle",
    output_dir: Optional[Path] = None,
    p_verbose: bool = False,
    compute_features: bool = True,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Path:
    """
    Compute features for multiple assets and merge into a single bundle.

    Args:
        asset_list: List of asset symbols (e.g., ["QQQ", "EURUSD"])
        asset_product_type: List of product types corresponding to each asset
        freq: Data frequency (default: CANDLE_1HOUR)
        source: Data source (default: FIRSTRATE)
        feature_types: List of feature types to compute (default: DEFAULT_FEATURE_TYPES)
        feature_config_path: Path to YAML config file. Takes precedence over feature_types.
        output_prefix: Prefix for output filename (default: "asset_bundle")
        output_dir: Output directory (default: BUNDLE_DIR)
        p_verbose: Enable verbose logging (default: False)
        compute_features: Whether to compute features (default: True). If False, only raw OHLCV.
        progress_callback: Optional callback(current, total, message) for progress updates.

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

    total_assets = len(asset_list)
    feature_files: List[Path] = []
    for i, symbol in enumerate(asset_list):
        if p_verbose:
            logger.info(f"Processing: {symbol}")
        if progress_callback:
            progress_callback(i, total_assets, f"Processing {symbol}")
        feature_file = compute_features_for_asset(
            symbol=symbol,
            freq=freq,
            source=source,
            product_type=asset_product_type[i],
            feature_types=feature_types,
            feature_config_path=feature_config_path,
            output_dir=output_dir / freq.value,
            p_verbose=p_verbose,
            compute_features=compute_features,
        )
        feature_files.append(feature_file)

    if progress_callback:
        progress_callback(total_assets, total_assets, "Merging feature files...")

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

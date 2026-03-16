"""
Create a merged multi-asset bundle from a YAML specification.

Builds BaseDataFrames with features for multiple assets, merges them,
and saves the combined bundle to parquet.

Example YAML config (multi_asset.yaml):
```yaml
assets:
  - symbol: EURUSD
    prefix: EUR
  - symbol: GBPUSD
    prefix: GBP
  - symbol: USDJPY
    prefix: JPY

data:
  data_freq: candle_1hour
  source: firstrate
  product_type: spot

features:
  - type: RSI
    periods: [14, 60, 240]
  - type: EMA
    periods: [15, 60]
  - type: HIST_VOLATILITY
    periods: [15, 60]
  - type: DAILY_SIGNAL
    kwargs:
      p_test_candles: 8
      p_exit_delay: 4

merge:
  id_col: minutes_since_2000
  float_16: false

output:
  path: data/bundles/fx_multi_asset_bundle.parquet
```

Usage:
    uv run python scripts/bundles/create_multiple_asset_bundle.py config.yaml
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from ruamel.yaml import YAML

from core.search_data import search_data
from core.enums import g_index_col
from norm.norm_utils import load_normalized_df
from features.base_dataframe import BaseDataFrame
from features.features_utils import FeatureType
from merger.merger_utils import merge_multiple_dataframes_from_parquet

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict[str, Any]:
    yaml = YAML()
    with open(config_path, "r") as f:
        return yaml.load(f)


def build_asset_dataframe(
    symbol: str,
    features_config: List[Dict[str, Any]],
    data_config: Dict[str, Any],
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    data_freq = data_config.get("data_freq")
    source = data_config.get("source")
    product_type = data_config.get("product_type")

    if verbose:
        logger.info(f"[{symbol}] Searching for data: data_freq={data_freq}, source={source}, product_type={product_type}")

    files = search_data(
        p_symbol=symbol,
        p_data_freq=data_freq,
        p_source=source,
        p_product_type=product_type,
    )

    if not files:
        raise ValueError(f"No data found for symbol={symbol}")

    data_file = files[0]
    if verbose:
        logger.info(f"[{symbol}] Loading: {data_file.path}")

    df = load_normalized_df(str(data_file.path))
    if verbose:
        logger.info(f"[{symbol}] Loaded {len(df)} rows, {len(df.columns)} columns")

    bdf = BaseDataFrame(
        p_df=df,
        p_valid_col_name="valid_row",
        p_scaling=-1,
        p_verbose=verbose,
    )

    for feature_def in features_config:
        if not isinstance(feature_def, dict) or "type" not in feature_def:
            logger.warning(f"[{symbol}] Invalid feature config: {feature_def}")
            continue

        try:
            feature_type = FeatureType(feature_def["type"])
        except ValueError:
            logger.warning(f"[{symbol}] Unknown FeatureType '{feature_def['type']}'")
            continue

        periods = feature_def.get("periods")
        kwargs = feature_def.get("kwargs", {})

        bdf.add_feature(feature_type, periods=periods, **kwargs)

    bdf.convert_f16_columns()

    result_df = bdf.get_dataframe()
    feature_cols = bdf.get_feature_columns()

    if verbose:
        logger.info(f"[{symbol}] Generated {len(feature_cols)} features")

    return result_df, feature_cols


def build_multi_asset_bundle(
    config: Dict[str, Any],
    verbose: bool = True,
) -> pd.DataFrame:
    assets_config = config.get("assets", [])
    data_config = config.get("data", {})
    features_config = config.get("features", [])
    merge_config = config.get("merge", {})

    if not assets_config:
        raise ValueError("YAML config must specify assets list")

    dfs: List[pd.DataFrame] = []
    prefixes: List[str] = []
    all_feature_cols: List[List[str]] = []

    for asset in assets_config:
        symbol = asset.get("symbol")
        prefix = asset.get("prefix", symbol[:3].upper() if symbol else "AST")

        if not symbol:
            raise ValueError("Each asset must specify a symbol")

        df, feature_cols = build_asset_dataframe(
            symbol=symbol,
            features_config=features_config,
            data_config=data_config,
            verbose=verbose,
        )

        dfs.append(df)
        prefixes.append(prefix)
        all_feature_cols.append(feature_cols)

    id_col = merge_config.get("id_col", g_index_col)
    float_16 = merge_config.get("float_16", False)

    if verbose:
        logger.info(f"Merging {len(dfs)} assets with id_col={id_col}")

    merged_df = merge_multiple_dataframes_from_parquet(
        file_paths=[],
        p_names=prefixes,
        p_cols_list=all_feature_cols,
        p_id_col=id_col,
        p_float_16=float_16,
        p_list_mode=True,
        p_list_df=dfs,
    )

    prefixed_feature_cols = []
    for prefix, cols in zip(prefixes, all_feature_cols):
        prefixed_feature_cols.extend([f"{prefix}_{col}" for col in cols])

    if verbose:
        logger.info(f"Merged shape: {merged_df.shape}")
        logger.info(f"Total prefixed features: {len(prefixed_feature_cols)}")

    return merged_df, prefixed_feature_cols, prefixes


def main():
    parser = argparse.ArgumentParser(description="Create a multi-asset bundle from YAML config")
    parser.add_argument("config", type=str, help="Path to YAML configuration file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    args = parser.parse_args()
    config_path = Path(args.config)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = load_config(config_path)

    merged_df, feature_cols, prefixes = build_multi_asset_bundle(config, verbose=args.verbose)

    output_config = config.get("output", {})
    output_path = Path(output_config.get("path", "data/bundles/multi_asset_bundle.parquet"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    merged_df.to_parquet(output_path)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Bundle saved to: {output_path}")
    logger.info(f"Shape: {merged_df.shape}")
    logger.info(f"Assets: {prefixes}")
    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"File size: {file_size_mb:.2f} MB")


if __name__ == "__main__":
    main()

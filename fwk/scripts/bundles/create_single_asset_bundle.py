"""
Create a single asset bundle from a YAML specification.

Builds a BaseDataFrame with features from normalized data and saves to parquet.

Example YAML config (single_asset.yaml):
```yaml
data:
  symbol: EURUSD
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

output:
  path: data/bundles/eurusd_features.parquet
  prefix_columns: true
  prefix: EUR
```

Usage:
    uv run python scripts/bundles/create_single_asset_bundle.py config.yaml
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from ruamel.yaml import YAML

from core.search_data import search_data
from core.enums import g_index_col
from norm.norm_utils import load_normalized_df
from features.base_dataframe import BaseDataFrame
from features.features_utils import FeatureType

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict[str, Any]:
    yaml = YAML()
    with open(config_path, "r") as f:
        return yaml.load(f)


def build_single_asset_bundle(
    config: Dict[str, Any],
    verbose: bool = True,
) -> pd.DataFrame:
    data_config = config.get("data", {})
    features_config = config.get("features", [])
    output_config = config.get("output", {})

    symbol = data_config.get("symbol")
    data_freq = data_config.get("data_freq")
    source = data_config.get("source")
    product_type = data_config.get("product_type")

    if not symbol:
        raise ValueError("YAML config must specify data.symbol")

    if verbose:
        logger.info(f"Searching for data: symbol={symbol}, data_freq={data_freq}, source={source}, product_type={product_type}")

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
        logger.info(f"Loading: {data_file.path}")

    df = load_normalized_df(str(data_file.path))
    if verbose:
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    bdf = BaseDataFrame(
        p_df=df,
        p_valid_col_name="valid_row",
        p_scaling=-1,
        p_verbose=verbose,
    )

    for feature_def in features_config:
        if not isinstance(feature_def, dict) or "type" not in feature_def:
            logger.warning(f"Invalid feature config: {feature_def}")
            continue

        try:
            feature_type = FeatureType(feature_def["type"])
        except ValueError:
            logger.warning(f"Unknown FeatureType '{feature_def['type']}'")
            continue

        periods = feature_def.get("periods")
        kwargs = feature_def.get("kwargs", {})

        bdf.add_feature(feature_type, periods=periods, **kwargs)

    bdf.convert_f16_columns()

    result_df = bdf.get_dataframe()
    feature_cols = bdf.get_feature_columns()

    if verbose:
        logger.info(f"Generated {len(feature_cols)} features")

    prefix_columns = output_config.get("prefix_columns", False)
    prefix = output_config.get("prefix", symbol[:3].upper())

    if prefix_columns and prefix:
        rename_map = {}
        for col in feature_cols:
            if not col.startswith(prefix + "_"):
                rename_map[col] = f"{prefix}_{col}"
        if rename_map:
            result_df = result_df.rename(columns=rename_map)
            feature_cols = [rename_map.get(c, c) for c in feature_cols]
            if verbose:
                logger.info(f"Prefixed {len(rename_map)} columns with '{prefix}'")

    return result_df, feature_cols


def main():
    parser = argparse.ArgumentParser(description="Create a single asset bundle from YAML config")
    parser.add_argument("config", type=str, help="Path to YAML configuration file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    args = parser.parse_args()
    config_path = Path(args.config)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = load_config(config_path)

    result_df, feature_cols = build_single_asset_bundle(config, verbose=args.verbose)

    output_config = config.get("output", {})
    output_path = Path(output_config.get("path", "data/bundles/single_asset_bundle.parquet"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result_df.to_parquet(output_path, index=False)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Bundle saved to: {output_path}")
    logger.info(f"Shape: {result_df.shape}")
    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"File size: {file_size_mb:.2f} MB")


if __name__ == "__main__":
    main()

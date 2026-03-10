"""
Compute ETF Feature Bundle.

This script loads normalized ETF data, builds features for each ETF,
and merges them into a single bundle file using the merger utilities.
"""

from pathlib import Path
from typing import List

from features.base_dataframe import BaseDataFrame
from features.features_utils import FeatureType
from merger.merger_utils import merge_multiple_dataframes_from_parquet
from core.data_org import BUNDLE_DIR, get_normalised_instrument_dir, MktDataFred, ExchangeNAME, ProductType
from core.enums import g_index_col
from norm.norm_utils import load_normalized_df


DATA_FREQ = MktDataFred.CANDLE_1HOUR
SOURCE = ExchangeNAME.FIRSTRATE
PRODUCT_TYPE = ProductType.ETF

ETF_LIST = ["QQQ", "SPY", "TLT", "GLD", "VWO"]

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
    etf_list: List[str] = ETF_LIST,
    output_prefix: str = "etf",
    output_dir: Path = None,
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
    print(f"  Index: {merged_df.index.name}")

    feature_cols = [c for c in merged_df.columns if "_F_" in c]
    print(f"  Total feature columns: {len(feature_cols)}")

    valid_col = [c for c in merged_df.columns if "valid_row" in c]
    print(f"  Validity columns: {valid_col}")

    output_path = output_dir / f"{output_prefix}_features_bundle.parquet"
    merged_df.to_parquet(output_path)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n{'='*60}")
    print(f"Bundle saved to: {output_path}")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"{'='*60}")

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute ETF Feature Bundle")
    parser.add_argument(
        "--etfs",
        type=str,
        nargs="+",
        default=ETF_LIST,
        help=f"ETF symbols to process (default: {' '.join(ETF_LIST)})",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="etf",
        help="Prefix for output filename (default: etf -> etf_features_bundle.parquet)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(BUNDLE_DIR),
        help="Output directory for bundle",
    )

    args = parser.parse_args()

    compute_etf_bundle(
        etf_list=args.etfs,
        output_prefix=args.prefix,
        output_dir=Path(args.output_dir),
    )

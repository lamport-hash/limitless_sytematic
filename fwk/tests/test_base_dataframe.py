"""
Unit test for BaseDataFrame that builds all features and Saves to bundle directory.
Loads QQQ data from candle_1hour/firstrate_undefined/etf and builds each feature type and saves output to data/bundle/candle_1hour.
"""

import pytest
import pandas as pd
from pathlib import Path

from features.base_dataframe import BaseDataFrame
from features.features_utils import FeatureType
from norm.norm_utils import load_normalized_df
from core.data_org import (
    get_normalised_file,
    BUNDLE_DIR,
    MktDataTFreq,
    ExchangeNAME,
    ProductType,
)


QQQ_FILE = get_normalised_file(
    MktDataTFreq.CANDLE_1HOUR, ExchangeNAME.FIRSTRATE, ProductType.ETF, "QQQ"
)

ALL_FEATURE_TYPES = [
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
    FeatureType.RITA,
    FeatureType.ROC,
]


def test_base_dataframe_qqq_all_features():
    """
    Test BaseDataFrame with QQQ data, building all features and saving to bundle.
    """
    df = load_normalized_df(QQQ_FILE)
    
    print(f"\n{'='*60}")
    print(f"Loaded QQQ data: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"{'='*60}\n")

    base_df = BaseDataFrame(
        p_df=df,
        p_valid_col_name="valid_row",
        p_scaling=-1,
        p_verbose=True,
    )

    for feature_type in ALL_FEATURE_TYPES:
        print(f"Adding feature: {feature_type}")
        base_df.add_feature(feature_type)

    base_df.convert_f16_columns()

    result_df = base_df.get_dataframe()
    features = base_df.get_features()
    feature_cols = base_df.get_feature_columns()

    print(f"\n{'='*60}")
    print(f"Result DataFrame shape: {result_df.shape}")
    print(f"Number of features: {len(features)}")
    print(f"Feature columns ({len(feature_cols)}): {feature_cols[:10]}...")
    print(f"{'='*60}\n")

    assert result_df.shape[0] > 0, "Result DataFrame should have rows"
    assert len(features) > 0, "Should have features registered"
    assert len(feature_cols) > 0, "Should have feature columns"

    output_dir = BUNDLE_DIR / "candle_1hour"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "QQQ_features.parquet"
    
    result_df.to_parquet(output_file, index=False)
    print(f"Saved to: {output_file}")

    assert output_file.exists(), f"Output file should exist: {output_file}"
    
    print(f"\n{'='*60}")
    print("FIRST 5 ROWS - Feature columns that exist:")
    print(f"{'='*60}")
    existing_feature_cols = [c for c in feature_cols if c in result_df.columns]
    print(result_df[existing_feature_cols].head().to_string())
    
    print(f"\n{'='*60}")
    print("TEST PASSED!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    test_base_dataframe_qqq_all_features()

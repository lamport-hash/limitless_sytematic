"""
Unit test for CTO Line feature.

Tests the feature calculation function and BaseDataFrame integration.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
import numpy as np

from features.base_dataframe import BaseDataFrame
from features.features_utils import FeatureType
from features.f_cto_line import smma_numba, feature_cto_line_signal
from norm.norm_utils import load_normalized_df


DATA_DIR = Path("/home/brian/sing/data/normalised/candle_1min/firstrate_undefined/spot/AUDNZD")
DATA_FILE = DATA_DIR / "AUDNZD_20100103_20260226_candle_1min.df.parquet"


def test_smma_numba():
    """Test SMMA calculation with simple input."""
    src = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    result = smma_numba(src, 3)
    
    assert len(result) == len(src), "SMMA output length should match input"
    assert result[0] == 1.0, "First SMMA value should be first input value"
    
    expected_second = (1.0 * 2 + 2.0) / 3
    assert abs(result[1] - expected_second) < 1e-10, f"Second SMMA value incorrect: {result[1]} vs {expected_second}"
    
    print(f"SMMA test passed. First 5 values: {result[:5]}")


def test_feature_cto_line_signal():
    """Test CTO Line signal generation with synthetic data."""
    n = 100
    df = pd.DataFrame({
        'S_high_f32': np.sin(np.linspace(0, 4*np.pi, n)) * 10 + 100,
        'S_low_f32': np.sin(np.linspace(0, 4*np.pi, n)) * 10 + 90,
    })
    
    long_signal, short_signal = feature_cto_line_signal(df, p_params=(15, 19, 25, 29))
    
    assert len(long_signal) == n, "Long signal length should match input"
    assert len(short_signal) == n, "Short signal length should match input"
    assert long_signal.dtype == np.int8, "Long signal should be int8"
    assert short_signal.dtype == np.int8, "Short signal should be int8"
    
    assert long_signal.isin([0, 1]).all(), "Long signal should only contain 0 or 1"
    assert short_signal.isin([0, 1]).all(), "Short signal should only contain 0 or 1"
    
    print(f"CTO Line signal test passed.")
    print(f"  Long signals: {long_signal.sum()}")
    print(f"  Short signals: {short_signal.sum()}")


def test_base_dataframe_cto_line_feature():
    """Test BaseDataFrame CTO Line feature integration with real data."""
    if not DATA_FILE.exists():
        pytest.skip(f"Data file not found: {DATA_FILE}")
    
    df = load_normalized_df(str(DATA_FILE))
    
    print(f"\n{'='*60}")
    print(f"Loaded AUDNZD data: {df.shape}")
    print(f"{'='*60}\n")
    
    base_df = BaseDataFrame(
        p_df=df,
        p_valid_col_name="valid_row",
        p_scaling=-1,
        p_verbose=True,
    )
    
    base_df.add_feature(FeatureType.CTO_LINE, params=(15, 19, 25, 29))
    
    result_df = base_df.get_dataframe()
    features = base_df.get_features()
    
    assert "F_cto_line_long_f16" in result_df.columns, "Long signal column should exist"
    assert "F_cto_line_short_f16" in result_df.columns, "Short signal column should exist"
    
    long_col = result_df["F_cto_line_long_f16"]
    short_col = result_df["F_cto_line_short_f16"]
    
    assert long_col.isin([0, 1]).all(), "Long signal should only contain 0 or 1"
    assert short_col.isin([0, 1]).all(), "Short signal should only contain 0 or 1"
    
    n_long = long_col.sum()
    n_short = short_col.sum()
    
    print(f"\n{'='*60}")
    print(f"Result DataFrame shape: {result_df.shape}")
    print(f"Number of features: {len(features)}")
    print(f"Long signals: {n_long}")
    print(f"Short signals: {n_short}")
    print(f"{'='*60}\n")
    
    assert n_long > 0, "Should have at least some long signals"
    assert n_short > 0, "Should have at least some short signals"
    
    print(f"\n{'='*60}")
    print("TEST PASSED!")
    print(f"{'='*60}\n")


def test_cto_line_with_custom_params():
    """Test CTO Line with custom parameters."""
    if not DATA_FILE.exists():
        pytest.skip(f"Data file not found: {DATA_FILE}")
    
    df = load_normalized_df(str(DATA_FILE))
    
    base_df = BaseDataFrame(p_df=df)
    
    custom_params = (10, 15, 20, 25)
    base_df.add_feature(FeatureType.CTO_LINE, params=custom_params)
    
    result_df = base_df.get_dataframe()
    
    assert "F_cto_line_long_f16" in result_df.columns
    assert "F_cto_line_short_f16" in result_df.columns
    
    print(f"Custom params test passed with params={custom_params}")


if __name__ == "__main__":
    test_smma_numba()
    test_feature_cto_line_signal()
    test_base_dataframe_cto_line_feature()
    test_cto_line_with_custom_params()

import pytest
import pandas as pd
import numpy as np
from fitting.fitting_feature_bundler import (
    FeatureBundler,
    LagBundler,
    RollingBundler,
    InteractionBundler,
    FeatureBundlerFactory
)


# =====================
# Fixtures
# =====================

@pytest.fixture
def sample_df():
    """Basic sample DataFrame with features and non-feature columns."""
    return pd.DataFrame({
        "date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"],
        "id": [1, 2, 3, 4, 5],
        "price": [100.0, 102.0, 104.0, 106.0, 108.0],
        "volume": [500, 520, 490, 510, 530],
        "target": [1, 0, 1, 0, 1]
    })


@pytest.fixture
def basecols():
    return ["price", "volume"]


# =====================
# Test LagBundler — EXPECTS TRUNCATION
# =====================

def test_lag_bundler_simple(sample_df, basecols):
    bundler = LagBundler(lags=[1, 2])
    result = bundler.bundle(sample_df, basecols)

    # Lag creates NaN in first 2 rows → dropna() removes them → output has 3 rows
    assert len(result) == 3

    # Original columns still present (no column dropped)
    assert all(col in result.columns for col in sample_df.columns)

    # So index 0 of result corresponds to original row2
    assert result["price_lag1"].iloc[0] == 102.0   # price at row2 (104) → lag1 = 102
    assert result["price_lag2"].iloc[0] == 100.0   # lag2 = 100

    # target should be preserved in surviving rows
    assert result["target"].tolist() == [1, 0, 1]   # original rows 2,3,4

    # Ensure no extra columns added beyond basecols derivations
    assert len(result.columns) == 9  # 5 original + 4 derived (price_lag1, price_lag2, volume_lag1, volume_lag2)


# =====================
# Test RollingBundler — EXPECTS TRUNCATION
# =====================

def test_rolling_bundler_simple(sample_df, basecols):
    bundler = RollingBundler(windows=[2], ops=["mean"])
    result = bundler.bundle(sample_df, basecols)

    # Rolling with window=2 → NaN in first row → dropna() removes it → 4 rows remain? Wait no:
    # Rolling mean: window=2 needs 2 values → NaN at row0, valid from row1
    # So after dropna: rows 1-4 remain → 4 rows

    assert len(result) == 4

    # Rolling mean at row1: (100+102)/2 = 101.0 → becomes index0 in result
    assert result["price_roll2_mean"].iloc[0] == 101.0

    # Rolling mean at row2: (102+104)/2 = 103.0 → index1
    assert result["price_roll2_mean"].iloc[1] == 103.0

    # target preserved in surviving rows
    assert result["target"].tolist() == [0, 1, 0, 1]   # original rows 1-4

    assert "price_roll2_mean" in result.columns
    assert all(col in result.columns for col in sample_df.columns)


# =====================
# Test InteractionBundler — EXPECTS TRUNCATION
# =====================

def test_interaction_bundler_simple(sample_df, basecols):
    bundler = InteractionBundler(max_interactions=2)
    result = bundler.bundle(sample_df, basecols)

    # Interaction: no NaNs created → so dropna() has NO effect → all 5 rows remain
    # BUT: Lag and Rolling *before* Interaction would drop rows → so in pipeline, interaction might see less data
    # However — this bundler does NOT create NaNs → so output should have 5 rows

    assert len(result) == 5

    # Interaction: price * volume at row0 = 100*500 = 50,000
    assert result["price_x_volume"].iloc[0] == 50000.0

    # target preserved
    assert result["target"].tolist() == [1, 0, 1, 0, 1]


# =====================
# Test FeatureBundlerFactory — EXPECTS TRUNCATION THROUGH PIPELINE
# =====================

def test_factory_apply_preserves_non_basecols(sample_df, basecols):
    config = {
        "lag": {"lags": [1]},
        "rolling": {"windows": [1], "ops": ["mean"]},
        "interaction": {}
    }
    factory = FeatureBundlerFactory.create_from_config(config)
    result = factory.apply(sample_df, basecols)

    # Lag drops first row → Rolling drops first row → Interaction runs on remaining 4 rows
    # So output has 4 rows

    assert len(result) == 4, f"Expected 4 rows after truncation, got {len(result)}"
    
    # Non-basecols (date, id, target) must still be in result — and preserved
    assert "date" in result.columns
    assert "id" in result.columns
    assert "target" in result.columns

    # target should have values from rows 1–4: [0, 1, 0, 1]
    assert result["target"].tolist() == [0, 1, 0, 1]

    # Ensure interaction still happened — it should be present
    assert "price_x_volume" in result.columns


def test_factory_multiple_bundlers_sequence(sample_df, basecols):
    config = {
        "lag": {"lags": [1]},
        "rolling": {"windows": [1], "ops": ["mean"]},
        "interaction": {}
    }
    factory = FeatureBundlerFactory.create_from_config(config)
    result = factory.apply(sample_df, basecols)

    # After lag + rolling → 4 rows remain  
    assert len(result) == 4

    # price_x_volume: computed on original values (but only for surviving rows)
    # row0 of result is original row1 → price=102, volume=520 → 102*520 = 53040
    assert result["price_x_volume"].iloc[0] == 102 * 520   # = 53040

    # price_lag1 at row0 (original row1) = 100
    assert result["price_lag1"].iloc[0] == 100.0


# =====================
# Test Config Validation
# =====================

def test_factory_create_from_config_unknown_bundler():
    with pytest.raises(ValueError, match="Unknown bundler type"):
        FeatureBundlerFactory.create_from_config({"fake": {}})


def test_factory_empty_config(sample_df, basecols):
    factory = FeatureBundlerFactory.create_from_config({})
    result = factory.apply(sample_df, basecols)
    assert len(result) == 5  # no changes → no truncation

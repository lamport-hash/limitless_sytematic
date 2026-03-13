"""
Tests for BaseDataFrame.add_features_from_md() method.
"""

import pytest
import tempfile
from pathlib import Path

from features.base_dataframe import BaseDataFrame
from features.features_utils import FeatureType


import pandas as pd
import numpy as np


def create_sample_df():
    n = 100
    return pd.DataFrame({
        "S_open_f32": np.random.randn(n).cumsum() + 100,
        "S_high_f32": np.random.randn(n).cumsum() + 101,
        "S_low_f32": np.random.randn(n).cumsum() + 99,
        "S_close_f32": np.random.randn(n).cumsum() + 100,
        "S_volume_f64": np.random.rand(n) * 1000,
        "S_close_time_i": range(1700000000000, 1700000000000 + n * 60000, 60000),
    })


def test_add_features_from_md_valid():
    """Test loading features from a valid markdown file with YAML block."""
    md_content = """# Test Feature Config

Some description here.

```yaml
features:
  - type: rsi
    periods: [14, 60]
  - type: hist_volatility
    periods: [15]
```
"""
    df = create_sample_df()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(md_content)
        f.flush()
        filepath = Path(f.name)

    try:
        bdf = BaseDataFrame(p_df=df)
        result = bdf.add_features_from_md(filepath)

        assert result is bdf

        features = bdf.get_features()
        feature_cols = bdf.get_feature_columns()

        assert len(features) > 0
        assert any("rsi" in col.lower() for col in feature_cols)
        assert any("vol" in col.lower() for col in feature_cols)
    finally:
        filepath.unlink()


def test_add_features_from_md_with_kwargs():
    """Test loading features with kwargs parameter."""
    md_content = """```yaml
features:
  - type: daily_signal
    kwargs:
      p_test_candles: 4
      p_exit_delay: 2
```
"""
    df = create_sample_df()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(md_content)
        f.flush()
        filepath = Path(f.name)

    try:
        bdf = BaseDataFrame(p_df=df)
        bdf.add_features_from_md(filepath)

        feature_cols = bdf.get_feature_columns()
        assert any("daily_signal" in col.lower() for col in feature_cols)
    finally:
        filepath.unlink()


def test_add_features_from_md_file_not_found():
    """Test that FileNotFoundError is raised for missing file."""
    df = create_sample_df()
    bdf = BaseDataFrame(p_df=df)

    with pytest.raises(FileNotFoundError):
        bdf.add_features_from_md("/nonexistent/path/features.md")


def test_add_features_from_md_no_yaml_block():
    """Test that ValueError is raised when no YAML block is found."""
    md_content = """# No YAML Here

Just regular markdown without any code blocks.
"""
    df = create_sample_df()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(md_content)
        f.flush()
        filepath = Path(f.name)

    try:
        bdf = BaseDataFrame(p_df=df)
        with pytest.raises(ValueError, match="No YAML code block found"):
            bdf.add_features_from_md(filepath)
    finally:
        filepath.unlink()


def test_add_features_from_md_no_features_key():
    """Test that ValueError is raised when 'features' key is missing."""
    md_content = """```yaml
other_key:
  - type: RSI
```
"""
    df = create_sample_df()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(md_content)
        f.flush()
        filepath = Path(f.name)

    try:
        bdf = BaseDataFrame(p_df=df)
        with pytest.raises(ValueError, match="No 'features' key found"):
            bdf.add_features_from_md(filepath)
    finally:
        filepath.unlink()


def test_add_features_from_md_empty_features():
    """Test that empty features list is handled gracefully."""
    md_content = """```yaml
features: []
```
"""
    df = create_sample_df()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(md_content)
        f.flush()
        filepath = Path(f.name)

    try:
        bdf = BaseDataFrame(p_df=df)
        result = bdf.add_features_from_md(filepath)

        assert result is bdf
    finally:
        filepath.unlink()


def test_add_features_from_md_invalid_feature_type():
    """Test that invalid feature types are skipped with a warning."""
    md_content = """```yaml
features:
  - type: rsi
    periods: [14]
  - type: invalid_feature_type
  - type: hist_volatility
    periods: [15]
```
"""
    df = create_sample_df()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(md_content)
        f.flush()
        filepath = Path(f.name)

    try:
        bdf = BaseDataFrame(p_df=df)
        bdf.add_features_from_md(filepath)

        feature_cols = bdf.get_feature_columns()
        assert any("rsi" in col.lower() for col in feature_cols)
        assert any("vol" in col.lower() for col in feature_cols)
    finally:
        filepath.unlink()


def test_add_features_from_md_chaining():
    """Test that method chaining works correctly."""
    md_content = """```yaml
features:
  - type: rsi
    periods: [14]
  - type: roc
    periods: [14]
```
"""
    df = create_sample_df()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(md_content)
        f.flush()
        filepath = Path(f.name)

    try:
        bdf = (
            BaseDataFrame(p_df=df)
            .add_features_from_md(filepath)
            .add_feature(FeatureType.HIST_VOLATILITY, periods=[15])
        )

        feature_cols = bdf.get_feature_columns()
        assert any("rsi" in col.lower() for col in feature_cols)
        assert any("roc" in col.lower() for col in feature_cols)
        assert any("vol" in col.lower() for col in feature_cols)
    finally:
        filepath.unlink()

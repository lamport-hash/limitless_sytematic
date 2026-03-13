"""
Tests for targets_generators.add_targets_from_md() function.
"""

import pytest
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np

from features.targets_generators import add_targets_from_md, setup_framework_indicators


def create_sample_ohlc_df():
    n = 100
    return pd.DataFrame({
        "S_open_f32": np.random.randn(n).cumsum() + 100,
        "S_high_f32": np.random.randn(n).cumsum() + 101,
        "S_low_f32": np.random.randn(n).cumsum() + 99,
        "S_close_f32": np.random.randn(n).cumsum() + 100,
        "S_volume_f64": np.random.rand(n) * 1000,
    })


def test_add_targets_from_md_file_not_found():
    """Test that FileNotFoundError is raised for missing file."""
    df = create_sample_ohlc_df()
    target_df = pd.DataFrame(index=df.index)

    with pytest.raises(FileNotFoundError):
        add_targets_from_md("/nonexistent/path/targets.md", df, target_df)


def test_add_targets_from_md_no_yaml_block():
    """Test that ValueError is raised when no YAML block is found."""
    md_content = """# No YAML Here

Just regular markdown without any code blocks.
"""
    df = create_sample_ohlc_df()
    target_df = pd.DataFrame(index=df.index)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(md_content)
        f.flush()
        filepath = Path(f.name)

    try:
        with pytest.raises(ValueError, match="No YAML code block found"):
            add_targets_from_md(filepath, df, target_df)
    finally:
        filepath.unlink()


def test_add_targets_from_md_empty_targets():
    """Test that empty t_classification section is handled gracefully."""
    md_content = """```yaml
t_classification: {}
```
"""
    df = create_sample_ohlc_df()
    target_df = pd.DataFrame(index=df.index)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(md_content)
        f.flush()
        filepath = Path(f.name)

    try:
        result = add_targets_from_md(filepath, df, target_df)
        assert result.shape[0] == target_df.shape[0]
    finally:
        filepath.unlink()


def test_add_targets_from_md_no_targets_key():
    """Test behavior when t_classification key is missing."""
    md_content = """```yaml
other_key:
  something: value
```
"""
    df = create_sample_ohlc_df()
    target_df = pd.DataFrame(index=df.index)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(md_content)
        f.flush()
        filepath = Path(f.name)

    try:
        result = add_targets_from_md(filepath, df, target_df)
        assert result.shape[0] == target_df.shape[0]
    finally:
        filepath.unlink()

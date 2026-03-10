
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import field
import pandas as pd
import numpy as np


class FeatureBundler(ABC):
    """Abstract base class for feature bundling strategies. Applied to specific basecols in a full DataFrame."""

    @abstractmethod
    def bundle(self, full_df: pd.DataFrame, basecols: List[str]) -> pd.DataFrame:
        """Transform only the specified basecols into derived features. Returns full_df with new cols added."""
        pass


class LagBundler(FeatureBundler):
    """Creates lagged versions of specified basecols."""

    def __init__(self, lags: List[int] = [1, 2]):
        self.lags = lags

    def bundle(self, full_df: pd.DataFrame, basecols: List[str]) -> pd.DataFrame:
        df = full_df.copy()
        for col in basecols:
            if col not in df.columns:
                raise ValueError(f"Basecol '{col}' not found in DataFrame")
            for lag in self.lags:
                df[f"{col}_lag{lag}"] = df[col].shift(lag)
        return df.dropna()  # drop rows with NaNs from lagging


class RollingBundler(FeatureBundler):
    """Creates rolling window features (mean, std, etc.) on specified basecols."""

    def __init__(self, windows: List[int] = [5, 10], ops: List[str] = ["mean", "std"]):
        self.windows = windows
        self.ops = ops

    def bundle(self, full_df: pd.DataFrame, basecols: List[str]) -> pd.DataFrame:
        df = full_df.copy()
        for col in basecols:
            if col not in df.columns:
                raise ValueError(f"Basecol '{col}' not found in DataFrame")
            for window in self.windows:
                for op in self.ops:
                    func = getattr(df[col].rolling(window), op)
                    df[f"{col}_roll{window}_{op}"] = func().ffill()
        return df.dropna()  # drop NaNs from rolling ops


class InteractionBundler(FeatureBundler):
    """Creates pairwise interactions ONLY between specified basecols."""

    def __init__(self, max_interactions: int = 2):
        self.max_interactions = max_interactions

    def bundle(self, full_df: pd.DataFrame, basecols: List[str]) -> pd.DataFrame:
        df = full_df.copy()
        if len(basecols) < 2:
            return df  # not enough cols to interact

        for i in range(len(basecols)):
            for j in range(i + 1, len(basecols)):
                if self.max_interactions >= 2:
                    col_i, col_j = basecols[i], basecols[j]
                    df[f"{col_i}_x_{col_j}"] = df[col_i] * df[col_j]
        return df


class FeatureBundlerFactory:
    """
    Factory to create and apply feature bundlers based on config.
    Uses 'basecols' explicitly to avoid data leakage or accidental transformations.
    """

    _bundler_registry = {
        "lag": LagBundler,
        "rolling": RollingBundler,
        "interaction": InteractionBundler
    }

    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> 'FeatureBundlerFactory':
        instance = cls()
        instance.bundlers = []
        
        for bundler_name, params in config.items():
            if bundler_name not in cls._bundler_registry:
                raise ValueError(f"Unknown bundler type: {bundler_name}. Valid options: {list(cls._bundler_registry.keys())}")
            bundler_class = cls._bundler_registry[bundler_name]
            instance.bundlers.append(bundler_class(**params))
        
        return instance

    def __init__(self):
        self.bundlers: List[FeatureBundler] = []

    def apply(self, full_df: pd.DataFrame, basecols: List[str]) -> pd.DataFrame:
        """
        Apply all bundlers sequentially to full_df using only the basecols.
        Returns DataFrame with new derived columns added, original cols preserved.
        """
        result = full_df.copy()
        for bundler in self.bundlers:
            result = bundler.bundle(result, basecols)
        return result

    def add_bundler(self, bundler: FeatureBundler):
        self.bundlers.append(bundler)

    def get_bundler_names(self) -> List[str]:
        return [type(b).__name__ for b in self.bundlers]



import pandas as pd

def align_target_with_features(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """
    Aligns target series `y` with feature DataFrame `X` by matching indices.
    
    After applying lag or rolling operations, X has fewer rows. 
    This function returns the subset of y that corresponds to the same index as X.
    
    Parameters:
        X (pd.DataFrame): Feature DataFrame, possibly reduced by lag/roll operations.
        y (pd.Series): Original target series with full length.
    
    Returns:
        pd.Series: Target values aligned to X's index (same length as X).
    
    Raises:
        ValueError: If X and y have no overlapping indices.
    """
    # Ensure both are pandas Series/DataFrame with index
    if not isinstance(X.index, pd.Index) or not isinstance(y.index, pd.Index):
        raise TypeError("X and y must have pandas Index objects.")

    # Find intersection of indices
    common_idx = X.index.intersection(y.index)
    
    if len(common_idx) == 0:
        raise ValueError("No overlapping indices between features and target.")
    
    # Return y values aligned to X's index
    y_aligned = y.loc[common_idx]
    
    # Optional: Ensure order matches X (in case index was shuffled during processing)
    y_aligned = y_aligned.reindex(X.index)
    
    # Check lengths match
    if len(y_aligned) != len(X):
        raise ValueError(f"Aligned target length ({len(y_aligned)}) "
                         f"does not match feature length ({len(X)}). "
                         "This suggests index mismatch beyond simple truncation.")
    
    return y_aligned

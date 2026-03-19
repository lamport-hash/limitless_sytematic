"""
Base Strategy Module

Provides abstract base class and common utilities for allocation strategies.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd


class BaseStrategy(ABC):
    """Abstract base class for allocation strategies."""
    
    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Return strategy name."""
        pass
    
    @property
    @abstractmethod
    def strategy_display_name(self) -> str:
        """Return human-readable strategy name for UI."""
        pass
    
    @abstractmethod
    def compute_allocations(
        self, 
        p_df: pd.DataFrame, 
        p_asset_list: List[str],
        **params
    ) -> pd.DataFrame:
        """
        Compute allocation columns.
        
        Must add A_{asset}_alloc columns to DataFrame.
        
        Args:
            p_df: DataFrame with asset data
            p_asset_list: List of asset symbols
            **params: Strategy-specific parameters
            
        Returns:
            DataFrame with A_{asset}_alloc columns added
        """
        pass
    
    @abstractmethod
    def get_required_features(self, **params) -> List[str]:
        """
        Return list of required feature column suffixes.
        
        These will be checked before running the strategy.
        """
        pass
    
    @abstractmethod
    def get_param_schema(self) -> Dict[str, Any]:
        """
        Return parameter schema for UI validation.
        
        Returns:
            Dict mapping param names to {type, default, description}
        """
        pass


def normalize_allocations(p_df: pd.DataFrame, p_asset_list: List[str]) -> pd.DataFrame:
    """
    Normalize allocations so they sum to 1.0 (or 0 if no allocations).
    
    Args:
        p_df: DataFrame with A_{asset}_alloc columns
        p_asset_list: List of asset symbols
        
    Returns:
            DataFrame with normalized allocations
    """
    df = p_df.copy()
    alloc_cols = [f"A_{asset}_alloc" for asset in p_asset_list if f"A_{asset}_alloc" in df.columns]
    
    if not alloc_cols:
        return df
    
    alloc_matrix = df[alloc_cols].values.astype(float)
    row_sums = alloc_matrix.sum(axis=1)
    
    mask = row_sums > 0
    if mask.any():
        df.loc[mask, alloc_cols] = alloc_matrix[mask] / row_sums[mask, np.newaxis]
    
    return df


def validate_allocations(p_df: pd.DataFrame, p_asset_list: List[str]) -> bool:
    for asset in p_asset_list:
        col = f"A_{asset}_alloc"
        if col not in p_df.columns:
            raise ValueError(f"Missing allocation column: {col}")
        
        nan_count = int(p_df[col].isna().sum())
        if nan_count > 0:
            raise ValueError(f"Allocation column {col} contains NaN values")
    
    alloc_cols = [f"A_{asset}_alloc" for asset in p_asset_list]
    row_sums = p_df[alloc_cols].sum(axis=1)
    
    min_sum = float(row_sums.min())
    if min_sum < -0.001:
        raise ValueError("Allocation columns contain negative sums")
    
    return True

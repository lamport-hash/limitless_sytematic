"""
QQQ Fitting Pipeline

Loads hourly QQQ data, computes features and targets, then fits XGB model.
Targets:
  - T_overnight_perf: overnight performance (close -> next open)
  - T_close_to_close: close to next close performance
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List

from core.search_data import search_data
from norm.norm_utils import load_normalized_df
from features.base_dataframe import BaseDataFrame
from features.features_utils import FeatureType
from fitting.fitting_models import TimeSeriesModelTrainer, TrainingConfig
from fitting.fitting_core import TrainingSplitType, TaskType
from fitting.models import ModelFactory, ModelType

from core.enums import (
    g_close_col,
    g_open_col,
    g_close_time_col,
    g_open_time_col,
    g_index_col,
)

MINUTES_PER_DAY = 1440
HOURS_PER_DAY = 24


def add_day_features(p_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Add day of week and lag returns over last 5 days (hourly)."""
    df = p_df.copy()
    new_features = []
    
    base = pd.Timestamp("2000-01-01")
    df["datetime"] = base + pd.to_timedelta(df[g_index_col], unit="m")
    df["F_day_of_week"] = df["datetime"].dt.dayofweek.astype(np.float32)
    new_features.append("F_day_of_week")
    
    for lag_hours in range(1, 5 * HOURS_PER_DAY + 1):
        lag_minutes = lag_hours * 60
        col_name = f"F_lag_ret_{lag_hours}h_f16"
        df[col_name] = (df[g_close_col] - df[g_close_col].shift(lag_hours)) / df[g_close_col].shift(lag_hours)
        df[col_name] = df[col_name].replace([np.inf, -np.inf], np.nan)
        new_features.append(col_name)
    
    print(f"[DAY_FEATURES] Added day_of_week + {len(new_features) - 1} hourly lag returns (5 days)")
    return df, new_features


def add_eod_flag(p_df: pd.DataFrame) -> pd.DataFrame:
    """Add flag for end-of-day candles (last candle of each trading day)."""
    df = p_df.copy()
    df["day_num"] = df[g_index_col] // MINUTES_PER_DAY
    df["next_day_num"] = df["day_num"].shift(-1)
    df["is_eod"] = df["day_num"] != df["next_day_num"]
    df.loc[df["is_eod"].isna(), "is_eod"] = False
    print(f"[EOD] Found {df['is_eod'].sum()} end-of-day candles out of {len(df)} total")
    return df


def load_qqq_data(p_data_freq: str = "candle_1hour") -> pd.DataFrame:
    """Load QQQ normalized data."""
    files = search_data(p_symbol="QQQ", p_data_freq=p_data_freq)
    if not files:
        raise ValueError("No QQQ data found")
    df = load_normalized_df(str(files[0].path))
    print(f"[QQQ] Loaded {len(df)} rows from {files[0].filename}")
    return df


def add_features(p_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Add features using BaseDataFrame."""
    bdf = BaseDataFrame(p_df=p_df, p_verbose=False)
    
    bdf.add_feature(FeatureType.RSI, periods=[14, 60, 240])
    bdf.add_feature(FeatureType.EMA, periods=[15, 60, 240])
    bdf.add_feature(FeatureType.SPREAD_REL_EMA, periods=[15, 60, 240])
    bdf.add_feature(FeatureType.DIFF_REL_EMA_MID, periods=[15, 60, 240])
    bdf.add_feature(FeatureType.HIST_VOLATILITY, periods=[15, 60, 240])
    bdf.add_feature(FeatureType.ROC, periods=[14, 60])
    bdf.add_feature(FeatureType.ADI)
    
    df_with_features = bdf.get_dataframe()
    feature_cols = bdf.get_feature_columns()
    
    print(f"[FEATURES] Generated {len(feature_cols)} features")
    return df_with_features, feature_cols


def compute_targets(p_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Compute target columns:
      - T_overnight_perf: (next_open - close) / close
      - T_close_to_close: (next_close - close) / close
    """
    df = p_df.copy()
    
    df["T_overnight_perf"] = (df[g_open_col].shift(-1) - df[g_close_col]) / df[g_close_col]
    df["T_close_to_close"] = (df[g_close_col].shift(-1) - df[g_close_col]) / df[g_close_col]
    
    df["T_overnight_perf"] = df["T_overnight_perf"].replace([np.inf, -np.inf], np.nan)
    df["T_close_to_close"] = df["T_close_to_close"].replace([np.inf, -np.inf], np.nan)
    
    target_cols = ["T_overnight_perf", "T_close_to_close"]
    print(f"[TARGETS] Generated {len(target_cols)} targets")
    
    return df, target_cols


def prepare_training_data(
    p_df: pd.DataFrame,
    p_feature_cols: List[str],
    p_target_col: str,
    p_eod_only: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare X and y arrays for training. If p_eod_only, only use end-of-day candles."""
    valid_mask = ~p_df[p_feature_cols].isna().any(axis=1)
    valid_mask &= ~p_df[p_target_col].isna()
    
    if p_eod_only:
        valid_mask &= p_df["is_eod"]
    
    X = p_df.loc[valid_mask, p_feature_cols].values.astype(np.float32)
    y = p_df.loc[valid_mask, p_target_col].values.astype(np.float32)
    
    print(f"[DATA] Prepared {X.shape[0]} samples with {X.shape[1]} features (eod_only={p_eod_only})")
    return X, y


def fit_xgb_model(
    X: np.ndarray,
    y: np.ndarray,
    p_n_estimators: int = 200,
    p_max_depth: int = 6,
    p_learning_rate: float = 0.1,
) -> Tuple[TimeSeriesModelTrainer, object]:
    """Fit XGB regression model."""
    config = TrainingConfig(
        mode=TrainingSplitType.TRAIN_VAL_TEST,
        train_ratio=0.6,
        val_ratio=0.2,
        normalization="standardize",
    )
    
    model = ModelFactory.create_model(
        model_type=ModelType.XGB,
        task_type=TaskType.REGRESSION,
        n_estimators=p_n_estimators,
        max_depth=p_max_depth,
        learning_rate=p_learning_rate,
    )
    
    trainer = TimeSeriesModelTrainer(
        model=model,
        config=config,
        task_type=TaskType.REGRESSION,
        verbose=False,
    )
    
    metrics = trainer.fit(X, y)
    return trainer, metrics


def main():
    print("=" * 60)
    print("QQQ FITTING PIPELINE")
    print("=" * 60)
    
    df = load_qqq_data(p_data_freq="candle_1hour")
    
    df, feature_cols = add_features(df)
    
    df, day_feature_cols = add_day_features(df)
    feature_cols = feature_cols + day_feature_cols
    
    df, target_cols = compute_targets(df)
    
    df = add_eod_flag(df)
    
    for target_col in target_cols:
        print(f"\n{'='*60}")
        print(f"TRAINING FOR TARGET: {target_col}")
        print("=" * 60)
        
        X, y = prepare_training_data(df, feature_cols, target_col, p_eod_only=True)
        
        trainer, metrics = fit_xgb_model(X, y)
        
        print(f"\n[RESULTS] Target: {target_col}")
        print(f"  Train RMSE: {metrics.train_score:.6f}")
        if metrics.val_score is not None:
            print(f"  Val RMSE:   {metrics.val_score:.6f}")
        print(f"  Test RMSE:  {metrics.test_score:.6f}")
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

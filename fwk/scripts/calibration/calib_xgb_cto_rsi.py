#!/usr/bin/env python3
"""
Calibrate XGBoost model using CTO_line and RSI features.

Loads normalized data, creates features (CTO_line and RSI), generates targets,
and trains an XGBoost classifier to predict market direction.

Usage:
    python scripts/calibration/calib_xgb_cto_rsi.py
    python scripts/calibration/calib_xgb_cto_rsi.py --data_pct 0.1
    python scripts/calibration/calib_xgb_cto_rsi.py --symbol EURUSD
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Tuple, Dict, Any

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from features.base_dataframe import BaseDataFrame
from features.features_utils import FeatureType
from features.targets_generators import add_targets_from_md
from norm.norm_utils import load_normalized_df
from core.data_org import get_normalised_file, MktDataFred, ExchangeNAME, ProductType
from core.enums import (
    g_open_col,
    g_high_col,
    g_low_col,
    g_close_col,
    g_volume_col,
    g_index_col,
)
from fitting.fitting_models import TrainingConfig, TimeSeriesModelTrainer
from fitting.fitting_core import TrainingSplitType, TaskType
from fitting.models import ModelFactory, ModelType
from fitting.fitting_judge import FittingJudge, ModelInfo, DataInfo

warnings.filterwarnings("ignore")

DEFAULT_CTO_PARAMS = (15, 19, 25, 29)
DEFAULT_RSI_PERIODS = [14, 60, 240]


def load_data(
    p_symbol: str = "EURUSD",
    p_data_pct: float = 1.0,
    p_start: str = "20100103",
    p_end: str = "20260226",
) -> pd.DataFrame:
    """Load normalized OHLCV data for specified symbol."""
    try:
        path = get_normalised_file(
            MktDataFred.CANDLE_1MIN,
            ExchangeNAME.FIRSTRATE,
            ProductType.SPOT,
            p_symbol,
            p_start=p_start,
            p_end=p_end,
        )
        df = load_normalized_df(str(path))
        if p_data_pct < 1.0:
            n_rows = int(len(df) * p_data_pct)
            df = df.iloc[:n_rows].copy()
        print(f"Loaded {len(df)} rows for {p_symbol}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def build_features(
    p_df: pd.DataFrame,
    p_cto_params: Tuple[int, int, int, int] = DEFAULT_CTO_PARAMS,
    p_rsi_periods: list[int] = None,
) -> Tuple[pd.DataFrame, list[str]]:
    """
    Build CTO_line and RSI features using BaseDataFrame.
    
    Returns:
        Tuple of (DataFrame with features, list of feature column names)
    """
    if p_rsi_periods is None:
        p_rsi_periods = DEFAULT_RSI_PERIODS
    
    bdf = BaseDataFrame(p_df=p_df.copy())
    
    bdf.add_feature(FeatureType.CTO_LINE, params=p_cto_params)
    bdf.add_feature(FeatureType.RSI, periods=p_rsi_periods)
    
    df = bdf.get_dataframe()
    feature_cols = bdf.get_feature_columns()
    
    cto_cols = [c for c in df.columns if "cto_line" in c.lower()]
    rsi_cols = [c for c in df.columns if "rsi" in c.lower() and c.startswith("F_")]
    feature_cols = cto_cols + rsi_cols
    
    print(f"Built {len(feature_cols)} features: {feature_cols}")
    return df, feature_cols


def create_target_config() -> str:
    """Create target configuration YAML for markdown file."""
    return """# Target Configuration
```yaml
t_classification:
  signal_class_60:
    type: single_asset
    asset: S
    function: gen_perfect_signal_class
    params:
      close_col: close_f32
      high_col: high_f32
      low_col: low_f32
      upstrong_val: 0.02
      downstrong_val: -0.02
      flat_val: 0.005
      N_periods: 60
```
"""


def build_targets_from_md(p_df: pd.DataFrame, p_config_str: str) -> Tuple[pd.DataFrame, str]:
    """
    Build targets from markdown config string.
    
    Returns:
        Tuple of (target DataFrame, target column name)
    """
    target_df = pd.DataFrame(index=p_df.index)
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(p_config_str)
        config_path = Path(f.name)
    
    try:
        target_df = add_targets_from_md(config_path, p_df, target_df)
    finally:
        config_path.unlink()
    
    target_cols = [c for c in target_df.columns if c.startswith("T_")]
    if not target_cols:
        raise ValueError("No target columns were generated")
    
    target_col = target_cols[0]
    print(f"Built target column: {target_col}")
    print(f"Target distribution:\n{target_df[target_col].value_counts()}")
    
    return target_df, target_col


def build_targets_direct(p_df: pd.DataFrame, p_n_periods: int = 60) -> Tuple[pd.DataFrame, str]:
    """
    Build classification target directly without markdown config.
    
    Returns:
        Tuple of (target DataFrame, target column name)
    """
    from features.targets_generators import gen_perfect_signal_class
    
    signal, signal_name = gen_perfect_signal_class(
        df=p_df,
        close_col=g_close_col,
        high_col=g_high_col,
        low_col=g_low_col,
        upstrong_val=0.02,
        downstrong_val=-0.02,
        flat_val=0.005,
        N_periods=p_n_periods,
    )
    
    target_df = pd.DataFrame(index=p_df.index)
    target_col = f"T_{signal_name}"
    target_df[target_col] = signal
    
    print(f"Built target column: {target_col}")
    print(f"Target distribution:\n{target_df[target_col].value_counts()}")
    
    return target_df, target_col


def prepare_xy(
    p_features_df: pd.DataFrame,
    p_feature_cols: list[str],
    p_target_df: pd.DataFrame,
    p_target_col: str,
    p_valid_col: str = "valid_row",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare X (features) and y (target) arrays for model training.
    
    Filters out rows with NaN values and invalid rows.
    """
    combined = pd.concat([p_features_df[p_feature_cols], p_target_df[[p_target_col]]], axis=1)
    
    if p_valid_col in p_features_df.columns:
        combined[p_valid_col] = p_features_df[p_valid_col]
        combined = combined[combined[p_valid_col] == True]
    
    combined = combined.dropna()
    
    X = combined[p_feature_cols].values.astype(np.float32)
    y = combined[p_target_col].values.astype(np.float32)
    
    print(f"Prepared X shape: {X.shape}, y shape: {y.shape}")
    print(f"Feature columns: {p_feature_cols}")
    
    return X, y


def train_xgb_model(
    X: np.ndarray,
    y: np.ndarray,
    p_feature_cols: list[str],
    p_target_col: str,
    p_task_type: TaskType = TaskType.CLASSIFICATION,
    p_train_ratio: float = 0.6,
    p_val_ratio: float = 0.2,
    p_verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train XGBoost model and return metrics.
    
    Args:
        X: Feature matrix
        y: Target vector
        p_feature_cols: List of feature column names
        p_target_col: Target column name
        p_task_type: Classification or Regression
        p_train_ratio: Ratio of data for training
        p_val_ratio: Ratio of data for validation
        p_verbose: Print progress
    
    Returns:
        Dictionary with model, trainer, metrics, and judge verdict
    """
    config = TrainingConfig(
        mode=TrainingSplitType.TRAIN_VAL_TEST,
        train_ratio=p_train_ratio,
        val_ratio=p_val_ratio,
        normalization="standardize",
    )
    
    model = ModelFactory.create_model(ModelType.XGB, p_task_type)
    trainer = TimeSeriesModelTrainer(model, config, p_task_type)
    
    metrics = trainer.fit(X, y)
    
    n_samples = len(X)
    model_info = ModelInfo(
        model_type="XGBoost",
        task_type=p_task_type,
        params=model.get_params() if hasattr(model, 'get_params') else {},
        model_class=ModelType.XGB.value,
    )
    data_info = DataInfo(
        n_samples=n_samples,
        n_features=len(p_feature_cols),
        feature_names=p_feature_cols,
        target_name=p_target_col,
        train_size=int(n_samples * p_train_ratio),
        val_size=int(n_samples * p_val_ratio),
        test_size=int(n_samples * (1 - p_train_ratio - p_val_ratio)),
    )
    
    judge = FittingJudge(metrics, model_info, data_info, config)
    verdict = judge.evaluate()
    
    if p_verbose:
        print(f"\n{'='*60}")
        print("XGBoost Training Results")
        print(f"{'='*60}")
        print(f"Train score: {metrics.train_score:.4f}")
        if metrics.val_score is not None:
            print(f"Val score:   {metrics.val_score:.4f}")
        if metrics.test_score is not None:
            print(f"Test score:  {metrics.test_score:.4f}")
        print(f"{'='*60}")
        print(f"\n{'='*60}")
        print("FITTING JUDGE VERDICT")
        print(f"{'='*60}")
        print(f"Rating: {verdict.rating.value.upper()}")
        print(f"Score:  {verdict.score:.2%}")
        print(f"\nSummary:")
        print(verdict.summary)
        if verdict.issues:
            print(f"\nIssues found: {len(verdict.issues)}")
            for issue in verdict.issues:
                print(f"  [{issue.severity.upper()}] {issue.message}")
        if verdict.recommendations:
            print(f"\nRecommendations:")
            for i, rec in enumerate(verdict.recommendations[:5], 1):
                print(f"  {i}. {rec}")
        print(f"{'='*60}\n")
    
    return {
        "model": model,
        "trainer": trainer,
        "metrics": metrics,
        "train_score": metrics.train_score,
        "val_score": metrics.val_score,
        "test_score": metrics.test_score,
        "verdict": verdict,
    }


def main(
    p_symbol: str = "EURUSD",
    p_data_pct: float = 1.0,
    p_cto_params: Tuple[int, int, int, int] = DEFAULT_CTO_PARAMS,
    p_rsi_periods: list[int] = None,
    p_target_periods: int = 60,
    p_train_ratio: float = 0.6,
    p_val_ratio: float = 0.2,
    p_verbose: bool = True,
):
    """Main calibration function."""
    print("=" * 80)
    print("XGBoost CALIBRATION: CTO_LINE + RSI FEATURES")
    print("=" * 80)
    
    print(f"\n1. Loading data for {p_symbol}...")
    df = load_data(p_symbol=p_symbol, p_data_pct=p_data_pct)
    
    print(f"\n2. Building features (CTO_line + RSI)...")
    features_df, feature_cols = build_features(
        p_df=df,
        p_cto_params=p_cto_params,
        p_rsi_periods=p_rsi_periods,
    )
    
    print(f"\n3. Building targets (N_periods={p_target_periods})...")
    target_df, target_col = build_targets_direct(
        p_df=features_df,
        p_n_periods=p_target_periods,
    )
    
    print(f"\n4. Preparing X/y matrices...")
    X, y = prepare_xy(
        p_features_df=features_df,
        p_feature_cols=feature_cols,
        p_target_df=target_df,
        p_target_col=target_col,
    )
    
    print(f"\n5. Training XGBoost model...")
    results = train_xgb_model(
        X=X,
        y=y,
        p_feature_cols=feature_cols,
        p_target_col=target_col,
        p_task_type=TaskType.CLASSIFICATION,
        p_train_ratio=p_train_ratio,
        p_val_ratio=p_val_ratio,
        p_verbose=p_verbose,
    )
    
    print("\n" + "=" * 80)
    print("CALIBRATION COMPLETE")
    print("=" * 80)
    
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate XGBoost with CTO_line and RSI features")
    parser.add_argument(
        "--symbol",
        type=str,
        default="EURUSD",
        help="Symbol to use (default: EURUSD)",
    )
    parser.add_argument(
        "--data_pct",
        type=float,
        default=1.0,
        help="Percentage of data to use (default: 1.0 = 100%)",
    )
    parser.add_argument(
        "--cto_v1",
        type=int,
        default=15,
        help="CTO v1 period (default: 15)",
    )
    parser.add_argument(
        "--cto_m1",
        type=int,
        default=19,
        help="CTO m1 period (default: 19)",
    )
    parser.add_argument(
        "--cto_m2",
        type=int,
        default=25,
        help="CTO m2 period (default: 25)",
    )
    parser.add_argument(
        "--cto_v2",
        type=int,
        default=29,
        help="CTO v2 period (default: 29)",
    )
    parser.add_argument(
        "--target_periods",
        type=int,
        default=60,
        help="Number of periods for target generation (default: 60)",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.6,
        help="Training data ratio (default: 0.6)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Validation data ratio (default: 0.2)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    rsi_periods = [14, 60, 240] if not args.quiet else [14, 60]
    
    results = main(
        p_symbol=args.symbol,
        p_data_pct=args.data_pct,
        p_cto_params=(args.cto_v1, args.cto_m1, args.cto_m2, args.cto_v2),
        p_rsi_periods=rsi_periods,
        p_target_periods=args.target_periods,
        p_train_ratio=args.train_ratio,
        p_val_ratio=args.val_ratio,
        p_verbose=not args.quiet,
    )

#!/usr/bin/env python3
"""
Fitting Judge Script

Evaluates ML model fitting quality using the FittingJudge module.
Can analyze existing fitting results or run a full fitting pipeline first.

Usage:
    # Analyze existing metrics
    python scripts/calibrate/fitting/judge_fitting.py --symbol EURUSD
    
    # Full pipeline with judgment
    python scripts/calibrate/fitting/judge_fitting.py --symbol EURUSD --data_pct 0.5
    
    # Generate AI prompt for analysis
    python scripts/calibrate/fitting/judge_fitting.py --symbol EURUSD --generate_prompt
    
    # Save results to file
    python scripts/calibrate/fitting/judge_fitting.py --symbol EURUSD --output results.md
"""

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from features.base_dataframe import BaseDataFrame
from features.features_utils import FeatureType
from features.targets_generators import gen_perfect_signal_class
from norm.norm_utils import load_normalized_df
from core.data_org import get_normalised_file, MktDataTFreq, ExchangeNAME, ProductType
from core.enums import g_close_col, g_high_col, g_low_col
from fitting.fitting_models import TrainingConfig, TimeSeriesModelTrainer
from fitting.fitting_core import TrainingSplitType, TaskType
from fitting.models import ModelFactory, ModelType
from fitting.fitting_judge import (
    FittingJudge,
    ModelInfo,
    DataInfo,
    generate_ai_analysis_prompt,
    quick_judge,
)

warnings.filterwarnings("ignore")


def load_data(
    p_symbol: str = "EURUSD",
    p_data_pct: float = 1.0,
    p_start: str = "20100103",
    p_end: str = "20260226",
) -> pd.DataFrame:
    """Load normalized OHLCV data for specified symbol."""
    try:
        path = get_normalised_file(
            MktDataTFreq.CANDLE_1MIN,
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


def build_features(p_df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """Build features using BaseDataFrame."""
    bdf = BaseDataFrame(p_df=p_df.copy())
    bdf.add_feature(FeatureType.RSI, periods=[14, 60])
    bdf.add_feature(FeatureType.EMA, periods=[15, 60])
    
    df = bdf.get_dataframe()
    feature_cols = [c for c in df.columns if c.startswith("F_")]
    
    print(f"Built {len(feature_cols)} features")
    return df, feature_cols


def build_targets(p_df: pd.DataFrame, p_n_periods: int = 60) -> Tuple[pd.DataFrame, str]:
    """Build classification targets."""
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
    
    print(f"Target distribution:\n{target_df[target_col].value_counts()}")
    return target_df, target_col


def prepare_xy(
    p_features_df: pd.DataFrame,
    p_feature_cols: list,
    p_target_df: pd.DataFrame,
    p_target_col: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare X and y arrays for training."""
    combined = pd.concat([p_features_df[p_feature_cols], p_target_df[[p_target_col]]], axis=1)
    combined = combined.dropna()
    
    X = combined[p_feature_cols].values.astype(np.float32)
    y = combined[p_target_col].values.astype(np.float32)
    
    print(f"Prepared X shape: {X.shape}, y shape: {y.shape}")
    return X, y


def run_fitting(
    X: np.ndarray,
    y: np.ndarray,
    model_type: ModelType = ModelType.XGB,
    task_type: TaskType = TaskType.CLASSIFICATION,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> Tuple[Any, Any, Any]:
    """Run model fitting and return results."""
    config = TrainingConfig(
        mode=TrainingSplitType.TRAIN_VAL_TEST,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        normalization="standardize",
    )
    
    model = ModelFactory.create_model(model_type, task_type)
    trainer = TimeSeriesModelTrainer(model, config, task_type)
    metrics = trainer.fit(X, y)
    
    return model, trainer, metrics


def print_verdict(verdict, verbose: bool = True):
    """Print fitting verdict in formatted output."""
    print("\n" + "=" * 70)
    print("FITTING JUDGMENT")
    print("=" * 70)
    
    print(f"\nRATING: {verdict.rating.value.upper()}")
    print(f"SCORE:  {verdict.score:.2%}")
    
    if verdict.issues:
        print(f"\nISSUES FOUND: {len(verdict.issues)}")
        print("-" * 40)
        for issue in verdict.issues:
            severity_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
            emoji = severity_emoji.get(issue.severity, "⚪")
            print(f"  {emoji} [{issue.severity.upper()}] {issue.type.value}")
            print(f"     {issue.message}")
    
    if verdict.recommendations:
        print(f"\nRECOMMENDATIONS:")
        print("-" * 40)
        for i, rec in enumerate(verdict.recommendations, 1):
            print(f"  {i}. {rec}")
    
    print(f"\nSUMMARY:")
    print("-" * 40)
    print(verdict.summary)
    
    print("\n" + "=" * 70)


def save_results(
    verdict,
    prompt: Optional[str],
    output_path: Path,
    model_info: ModelInfo,
    data_info: DataInfo,
):
    """Save judgment results to markdown file."""
    with open(output_path, "w") as f:
        f.write("# Fitting Judgment Report\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- **Rating**: {verdict.rating.value.upper()}\n")
        f.write(f"- **Score**: {verdict.score:.2%}\n\n")
        
        f.write("## Model Information\n\n")
        f.write(f"- **Model Type**: {model_info.model_type}\n")
        f.write(f"- **Task Type**: {model_info.task_type.value}\n")
        f.write(f"- **Parameters**: `{json.dumps(model_info.params)}`\n\n")
        
        f.write("## Data Information\n\n")
        f.write(f"- **Samples**: {data_info.n_samples}\n")
        f.write(f"- **Features**: {data_info.n_features}\n")
        f.write(f"- **Train/Val/Test**: {data_info.train_size}/{data_info.val_size}/{data_info.test_size}\n\n")
        
        if verdict.issues:
            f.write("## Issues\n\n")
            for issue in verdict.issues:
                f.write(f"### {issue.type.value} ({issue.severity})\n\n")
                f.write(f"{issue.message}\n\n")
                if issue.details:
                    f.write(f"```\n{json.dumps(issue.details, indent=2)}\n```\n\n")
        
        if verdict.recommendations:
            f.write("## Recommendations\n\n")
            for rec in verdict.recommendations:
                f.write(f"- {rec}\n")
            f.write("\n")
        
        if prompt:
            f.write("## AI Analysis Prompt\n\n")
            f.write("```markdown\n")
            f.write(prompt)
            f.write("\n```\n")
    
    print(f"Results saved to: {output_path}")


def main(
    p_symbol: str = "EURUSD",
    p_data_pct: float = 1.0,
    p_model_type: ModelType = ModelType.XGB,
    p_task_type: TaskType = TaskType.CLASSIFICATION,
    p_train_ratio: float = 0.6,
    p_val_ratio: float = 0.2,
    p_target_periods: int = 60,
    p_generate_prompt: bool = False,
    p_output: Optional[str] = None,
    p_verbose: bool = True,
):
    """Main fitting judgment function."""
    print("=" * 70)
    print("FITTING JUDGE")
    print("=" * 70)
    
    print(f"\n1. Loading data for {p_symbol}...")
    df = load_data(p_symbol=p_symbol, p_data_pct=p_data_pct)
    
    print(f"\n2. Building features...")
    features_df, feature_cols = build_features(df)
    
    print(f"\n3. Building targets...")
    target_df, target_col = build_targets(features_df, p_n_periods=p_target_periods)
    
    print(f"\n4. Preparing data...")
    X, y = prepare_xy(features_df, feature_cols, target_df, target_col)
    
    print(f"\n5. Running fitting...")
    model, trainer, metrics = run_fitting(
        X, y,
        model_type=p_model_type,
        task_type=p_task_type,
        train_ratio=p_train_ratio,
        val_ratio=p_val_ratio,
    )
    
    print(f"\n6. Evaluating fitting quality...")
    
    n_samples = X.shape[0]
    n_features = X.shape[1]
    train_size = int(n_samples * p_train_ratio)
    val_size = int(n_samples * p_val_ratio)
    test_size = n_samples - train_size - val_size
    
    model_info = ModelInfo(
        model_type=p_model_type.code,
        task_type=p_task_type,
        params=model.getParams(),
        model_class=type(model).__name__,
    )
    
    data_info = DataInfo(
        n_samples=n_samples,
        n_features=n_features,
        feature_names=feature_cols,
        target_name=target_col,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        has_retrain=False,
        retrain_events=0,
    )
    
    judge = FittingJudge(metrics, model_info, data_info)
    verdict = judge.evaluate()
    
    print_verdict(verdict, verbose=p_verbose)
    
    prompt = None
    if p_generate_prompt:
        print(f"\n7. Generating AI analysis prompt...")
        prompt = generate_ai_analysis_prompt(metrics, model_info, data_info)
        
        prompt_file = Path("data_work/logs") / f"fitting_prompt_{p_symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        with open(prompt_file, "w") as f:
            f.write(prompt)
        print(f"Prompt saved to: {prompt_file}")
    
    if p_output:
        output_path = Path(p_output)
        save_results(verdict, prompt, output_path, model_info, data_info)
    
    return {
        "verdict": verdict,
        "metrics": metrics,
        "model_info": model_info,
        "data_info": data_info,
        "prompt": prompt,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Judge ML model fitting quality")
    
    parser.add_argument(
        "--symbol",
        type=str,
        default="EURUSD",
        help="Symbol to use (default: EURUSD)",
    )
    parser.add_argument(
        "--data_pct",
        type=float,
        default=0.1,
        help="Percentage of data to use (default: 0.1 for faster testing)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="xgb",
        choices=["xgb", "rf", "dt", "mlp"],
        help="Model type (default: xgb)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="classification",
        choices=["classification", "regression"],
        help="Task type (default: classification)",
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
        "--target_periods",
        type=int,
        default=60,
        help="Target prediction periods (default: 60)",
    )
    parser.add_argument(
        "--generate_prompt",
        action="store_true",
        help="Generate AI analysis prompt",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for results (markdown)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output",
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    model_map = {
        "xgb": ModelType.XGB,
        "rf": ModelType.RF_SK,
        "dt": ModelType.DT_SK,
        "mlp": ModelType.MLP_TORCH,
    }
    
    task_map = {
        "classification": TaskType.CLASSIFICATION,
        "regression": TaskType.REGRESSION,
    }
    
    results = main(
        p_symbol=args.symbol,
        p_data_pct=args.data_pct,
        p_model_type=model_map[args.model],
        p_task_type=task_map[args.task],
        p_train_ratio=args.train_ratio,
        p_val_ratio=args.val_ratio,
        p_target_periods=args.target_periods,
        p_generate_prompt=args.generate_prompt,
        p_output=args.output,
        p_verbose=not args.quiet,
    )

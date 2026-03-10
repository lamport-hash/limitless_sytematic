"""
Time Series Model Training and Hyperparameter Tuning API

A modular framework for training, validating, and testing ML models on time series data
with support for both static and periodic retraining strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, TypeAlias, Literal, Dict, List
import numpy as np
from numpy.typing import NDArray
import os
import joblib
from datetime import datetime

from .fitting_core import Features, Targets, Predictions, BaseModel, TaskType, RetrainMode, TrainingSplitType
from .fitting_metrics import RegressionMetric, ClassificationMetric, MetricCalculator


@dataclass
class DataSplit:
    """Holds train/validation/test data splits."""
    
    X_train: Features
    y_train: Targets
    X_val: Features | None = None
    y_val: Targets | None = None
    X_test: Features | None = None
    y_test: Targets | None = None
    
    train_indices: tuple[int, int] = field(default=(0, 0))
    val_indices: tuple[int, int] | None = None
    test_indices: List[tuple[int, int]] = field(default_factory=list)  # ✅ FIXED! Not tuple, but List[tuple]



@dataclass
class Normalizer:
    """Handles feature normalization with expanding statistics."""
    
    method: Literal["standardize", "minmax", "none"] = "standardize"
    _min: Features | None = None
    _max: Features | None = None
    _mean: Features | None = None
    _std: Features | None = None
    
    def fit(self, X: Features) -> None:
        """Fit normalizer on data (only expands, never contracts stats)."""
        if self.method == "none":
            return
        
        if self.method == "minmax":
            new_min = np.min(X, axis=0)
            new_max = np.max(X, axis=0)
            
            if self._min is None:
                self._min = new_min
                self._max = new_max
            else:
                # Always expand the range, never contract
                self._min = np.minimum(self._min, new_min)
                self._max = np.maximum(self._max, new_max)
        
        elif self.method == "standardize":
            new_mean = np.mean(X, axis=0)
            new_std = np.std(X, axis=0)
            
            if self._mean is None:
                self._mean = new_mean
                self._std = new_std
            else:
                # Expand standard deviation range
                combined_data = np.vstack([
                    (X - new_mean) / (new_std + 1e-8),
                    np.zeros_like(self._mean)
                ])
                self._std = np.maximum(self._std, new_std)
                # Update mean with weighted average (keep expanding)
                self._mean = new_mean
    
    def transform(self, X: Features) -> Features:
        """Transform data using fitted statistics."""
        if self.method == "none":
            return X.copy()
        
        if self.method == "minmax":
            if self._min is None or self._max is None:
                raise ValueError("Normalizer not fitted yet")
            range_val = self._max - self._min
            range_val[range_val == 0] = 1.0  # Avoid division by zero
            return (X - self._min) / range_val
        
        elif self.method == "standardize":
            if self._mean is None or self._std is None:
                raise ValueError("Normalizer not fitted yet")
            std_safe = self._std.copy()
            std_safe[std_safe == 0] = 1.0
            return (X - self._mean) / std_safe
        
        return X
    
    def fit_transform(self, X: Features) -> Features:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)



@dataclass
class TrainingConfig:
    """Configuration for training procedure."""
    
    # Split configuration
    mode: TrainingSplitType = TrainingSplitType.TRAIN_VAL_TEST
    train_ratio: float = 0.6
    val_ratio: float = 0.2  # Only used in three_way mode
    
    # Periodic retraining config
    retrain_period: int | None = None  # P: retrain every P samples in test
    retrain_mode: RetrainMode = RetrainMode.EXPANDING
    sliding_window_size: int | None = None  # SW: size of sliding window
    
    # Normalization
    normalization: Literal["standardize", "minmax", "none"] = "standardize"
    normalize_targets: bool = False
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.mode == TrainingSplitType.TRAIN_VAL_TEST:
            if not (0 < self.train_ratio < 1):
                raise ValueError("train_ratio must be between 0 and 1")
            if not (0 < self.val_ratio < 1):
                raise ValueError("val_ratio must be between 0 and 1")
            if self.train_ratio + self.val_ratio >= 1:
                raise ValueError("train_ratio + val_ratio must be < 1")
        
        elif self.mode == TrainingSplitType.PERIODIC_RETRAIN:
            if self.retrain_period is None or self.retrain_period <= 0:
                raise ValueError(
                            "retrain_period must be a positive integer in periodic_retrain mode. "
                            f"Got: {self.retrain_period}"
                        )
            if self.retrain_mode == RetrainMode.SLIDING:
                if self.sliding_window_size is None or self.sliding_window_size <= 0:
                    raise ValueError("sliding_window_size required for sliding retrain mode"
                                     f"Got: {self.retrain_period}")

@dataclass
class ModelMetrics:
    """Stores evaluation metrics for a model."""
    
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float] | None = None
    test_metrics: Dict[str, float] | None = None
    predictions: Predictions | None = None
    retrain_history: list[dict[str, Any]] = field(default_factory=list)
    
    @property
    def train_score(self) -> float:
        """Convenience property: returns the default scoring metric ('score')."""
        return self.train_metrics.get("score", float("nan"))
    
    @property
    def val_score(self) -> float | None:
        """Convenience property: returns default scoring metric for validation."""
        return self.val_metrics.get("score", float("nan")) if self.val_metrics is not None else None
    
    @property
    def test_score(self) -> float | None:
        """Convenience property: returns default scoring metric for test."""
        return self.test_metrics.get("score", float("nan")) if self.test_metrics is not None else None
    
    def __repr__(self) -> str:
        """String representation of metrics."""
        lines = [
            f"Train Score: {self.train_score:.4f}",
            f"Test Score: {self.test_score:.4f}"
        ]
        
        if self.val_score is not None:
            lines.insert(1, f"Val Score: {self.val_score:.4f}")
        
        # Optionally show all metrics if verbose or for debugging
        # For now, keep it simple with just 'score' in main output.
        
        if self.retrain_history:
            lines.append(f"Retraining Events: {len(self.retrain_history)}")
        
        return "\n".join(lines)
    
    def get_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """Returns full dictionary of all metrics per split."""
        return {
            "train": self.train_metrics,
            "val": self.val_metrics or {},
            "test": self.test_metrics or {}
        }


class TimeSeriesModelTrainer:
    """Main class for training and evaluating time series models."""
    
    def __init__(
        self,
        model: BaseModel,
        config: TrainingConfig,
        task_type: TaskType = TaskType.REGRESSION,
        metric_calculator: MetricCalculator | None = None,
        verbose: bool = False  # 🆕 NEW: Verbose mode to print progress
    ):
        """
        Initialize trainer.
        
        Args:
            model: ML model following BaseModel
            config: Training configuration
            task_type: Type of task (classification or regression)
            metric_calculator: Custom metric calculator (optional)
            verbose: If True, print detailed progress messages during training (default: False)
        """
        self.model = model
        self.config = config
        self.task_type = task_type
        self.verbose = verbose  # 🆕 Store verbose flag
        
        # Set default metric if not provided
        if metric_calculator is None:
            if task_type == TaskType.REGRESSION:
                self.metric_calc = RegressionMetric()
            else:
                self.metric_calc = ClassificationMetric()
        else:
            self.metric_calc = metric_calculator
        
        # Normalizers
        self.X_normalizer = Normalizer(method=config.normalization)
        self.y_normalizer = Normalizer(method=config.normalization) if config.normalize_targets else None
    
    def _create_splits(self, X: Features, y: Targets) -> DataSplit:
        """Create train/val/test splits based on configuration."""
        n_samples = len(X)
        
        if self.verbose:
            print(f"Creating splits with {n_samples} samples...")
        
        if self.config.mode == TrainingSplitType.TRAIN_VAL_TEST:
            train_end = int(n_samples * self.config.train_ratio)
            val_end = int(n_samples * (self.config.train_ratio + self.config.val_ratio))
            
            if self.verbose:
                print(f"  Train: {train_end} samples ({self.config.train_ratio:.1%})")
                print(f"  Val: {val_end - train_end} samples ({self.config.val_ratio:.1%})")
                print(f"  Test: {n_samples - val_end} samples ({1 - self.config.train_ratio - self.config.val_ratio:.1%})")
            
            return DataSplit(
                X_train=X[:train_end],
                y_train=y[:train_end],
                X_val=X[train_end:val_end],
                y_val=y[train_end:val_end],
                X_test=X[val_end:],
                y_test=y[val_end:],
                train_indices=(0, train_end),
                val_indices=(train_end, val_end),
                test_indices=[(val_end, n_samples)]  # ✅ Single chunk
            )
        
        else:  # periodic_retrain
            train_end = int(n_samples * self.config.train_ratio)
            
            if self.verbose:
                print(f"  Train: {train_end} samples ({self.config.train_ratio:.1%})")
                print(f"  Test: {n_samples - train_end} samples ({1 - self.config.train_ratio:.1%})")
            
            # Generate test chunks for periodic retraining
            test_chunks = []
            period = self.config.retrain_period
            assert period is not None, "retrain_period must be set for periodic_retrain mode"

            start = train_end
            while start < n_samples:
                end = min(start + period, n_samples)
                test_chunks.append((start, end))
                start = end  # next chunk starts where current ends

            if self.verbose:
                print(f"  Generated {len(test_chunks)} test chunks of size ~{period} samples")

            return DataSplit(
                X_train=X[:train_end],
                y_train=y[:train_end],
                X_test=X[train_end:],
                y_test=y[train_end:],
                train_indices=(0, train_end),
                test_indices=test_chunks  # ✅ List of (start, end) tuples
            )

    def fit(self, X: Features, y: Targets) -> ModelMetrics:
        """
        Train model and evaluate on splits.
        
        Args:
            X: Input features (n_samples, n_features)
            y: Target values (n_samples,) or (n_samples, n_targets)
        
        Returns:
            ModelMetrics with training results
        """
        if self.verbose:
            print(f"Starting fit() with mode='{self.config.mode}'...")
        
        # Create splits
        splits = self._create_splits(X, y)
        
        # Normalize training data
        if self.config.mode == TrainingSplitType.TRAIN_VAL_TEST:
            X_train_norm = self.X_normalizer.fit_transform(splits.X_train)
            if self.verbose:
                print("✅ Normalized training features.")
            
            y_train_norm = splits.y_train
            if self.y_normalizer is not None:
                y_train_norm = self.y_normalizer.fit_transform(splits.y_train.reshape(-1, 1)).ravel()
                if self.verbose:
                    print("✅ Normalized training targets.")
            
            if self.verbose:
                print(f"Training model on {len(X_train_norm)} samples...")
            
            self.model.fit(X_train_norm, y_train_norm)
            
            # Static test eval
            test_pred, test_metrics = self._evaluate_test_static(splits)
            retrain_history = []
        
        else:  # periodic_retrain
            if self.verbose:
                print("🚀 Entering periodic retraining mode...")
            
            test_pred, test_metrics, retrain_history = self._evaluate_test_periodic(X, y, splits)
        
        # Evaluate train score after training
        X_train_norm = self.X_normalizer.transform(splits.X_train)
        train_pred_norm = self.model.predict(X_train_norm)
        train_pred = self._denormalize_predictions(train_pred_norm)
        train_metrics = self.metric_calc.calculate(splits.y_train, train_pred)  # 🚨 Now a dict!

        val_metrics = None
        if splits.X_val is not None and splits.y_val is not None:
            X_val_norm = self.X_normalizer.transform(splits.X_val)
            val_pred_norm = self.model.predict(X_val_norm)
            val_pred = self._denormalize_predictions(val_pred_norm)
            val_metrics = self.metric_calc.calculate(splits.y_val, val_pred)  # 🚨 Dict!
            if self.verbose:
                print(f"📊 Validation metrics: {', '.join([f'{k}: {v:.4f}' for k, v in val_metrics.items()])}")
        
        if self.verbose:
            print(f"📈 Final Results:")
            print(f"   Train Score: {train_metrics.get('score', float('nan')):.4f}")
            if val_metrics is not None:
                print(f"   Val Score: {val_metrics.get('score', float('nan')):.4f}")
            print(f"   Test Score: {test_metrics.get('score', float('nan')):.4f}")

        return ModelMetrics(
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            predictions=test_pred,
            retrain_history=retrain_history
        )

    def _evaluate_test_static(self, splits: DataSplit) -> tuple[Predictions, Dict[str, float]]:
        """Evaluate on test set without retraining."""
        if self.verbose:
            print("🔍 Evaluating static test set...")
            print(f"   Test set size: {len(splits.X_test)} samples")
            print(f"   Using trained model from training phase (no retraining).")

        X_test_norm = self.X_normalizer.transform(splits.X_test)
        if self.verbose:
            print(f"   ✅ Applied X-normalization to test set (mean: {self.X_normalizer.mean_[:3] if hasattr(self.X_normalizer, 'mean_') else 'N/A'}...)")

        test_pred_norm = self.model.predict(X_test_norm)
        if self.verbose:
            print(f"   ✅ Model prediction completed on test set (first 5 preds: {test_pred_norm[:5]})")

        test_pred = self._denormalize_predictions(test_pred_norm)
        if self.verbose and hasattr(self, 'y_normalizer') and self.y_normalizer is not None:
            print(f"   ✅ Applied y-denormalization (mean: {self.y_normalizer.mean_[0]:.4f}, std: {self.y_normalizer.scale_[0]:.4f})")

        test_metrics = self.metric_calc.calculate(splits.y_test, test_pred)
        
        if self.verbose:
            print(f"   📊 Test Scores: {', '.join([f'{k}: {v:.4f}' for k, v in test_metrics.items()])}")
            print(f"   📌 Static evaluation complete. No retraining performed.")

        return test_pred, test_metrics

    def _evaluate_test_periodic(
        self, 
        X: Features, 
        y: Targets, 
        splits: DataSplit
    ) -> tuple[Predictions, Dict[str, float], list[dict[str, Any]]]:
        """Evaluate on test set with periodic retraining using EXPANDING or SLIDING windows."""
        
        if self.verbose:
            print(f"🔄 Starting periodic retraining with period={self.config.retrain_period}...")
            print(f"   Retrain mode: {self.config.retrain_mode.value}")
            if self.config.retrain_mode == RetrainMode.SLIDING:
                print(f"   Sliding window size: {self.config.sliding_window_size}")

        test_predictions = []  # Will collect predictions for entire test set
        retrain_history = []   # Store metrics per retraining event

        test_start = splits.train_indices[1]  # End of training set → start of testing
        test_end = len(X)
        period = self.config.retrain_period
        assert period is not None, "retrain_period must be set for periodic mode"

        if self.verbose:
            print(f"   Test set: [{test_start}, {test_end}) with period={period}")

        # Initialize normalizers on initial training set
        if self.verbose:
            print(f"🏗️  Initializing normalizers on initial train set (samples: {test_start})")

        X_train_initial = X[:test_start]
        y_train_initial = y[:test_start]

        # Fit normalizers on initial training data
        X_train_norm = self.X_normalizer.fit_transform(X_train_initial)
        y_train_norm = y_train_initial
        if self.y_normalizer is not None:
            y_train_norm = self.y_normalizer.fit_transform(y_train_initial.reshape(-1, 1)).ravel()

        # Train initial model
        if self.verbose:
            print(f"🏗️  Initial training on {len(X_train_initial)} samples...")
        self.model.fit(X_train_norm, y_train_norm)

        # Generate retrain/test chunk boundaries
        retrain_indices = []
        for i in range(test_start, test_end, period):
            train_end = i  # Train up to this index (exclusive of it in slicing)
            test_chunk_start = i
            test_chunk_end = min(i + period, test_end)

            if test_chunk_start >= test_end:
                break

            retrain_indices.append({
                'train_end': train_end,
                'test_start': test_chunk_start,
                'test_end': test_chunk_end,
            })

        if self.verbose:
            print(f"📅 Planned retrain/test chunks: {len(retrain_indices)} events")

        # Iterate over each retrain point
        for chunk in retrain_indices:
            train_end = chunk['train_end']
            test_start_chunk = chunk['test_start']
            test_end_chunk = chunk['test_end']

            if self.config.retrain_mode == RetrainMode.EXPANDING:
                # Use all data from beginning to train_end
                start_idx = 0
            elif self.config.retrain_mode == RetrainMode.SLIDING:
                # Only use the most recent `sliding_window_size` samples
                start_idx = max(0, train_end - self.config.sliding_window_size)
            else:
                raise ValueError(f"Unknown retrain_mode: {self.config.retrain_mode}")

            # Extract training data for this window
            X_train_chunk = X[start_idx:train_end]
            y_train_chunk = y[start_idx:train_end]

            if self.verbose:
                print(f"\n🔁 [RETRAIN @ {train_end}] → Train: [{start_idx}, {train_end}), "
                    f"Test: [{test_start_chunk}, {test_end_chunk}) | Window size: {len(X_train_chunk)}")

            # Fit normalizers on current training window
            X_train_norm = self.X_normalizer.fit_transform(X_train_chunk)
            y_train_norm = y_train_chunk
            if self.y_normalizer is not None:
                y_train_norm = self.y_normalizer.fit_transform(y_train_chunk.reshape(-1, 1)).ravel()

            # Retrain model
            if self.verbose:
                print(f"   🔁 Retraining on {len(X_train_chunk)} samples...")
            self.model.fit(X_train_norm, y_train_norm)

            # Evaluate on training set (monitoring)
            X_train_eval = X[start_idx:train_end]
            y_train_eval = y[start_idx:train_end]
            X_train_norm_eval = self.X_normalizer.transform(X_train_eval)
            y_pred_train = self._denormalize_predictions(self.model.predict(X_train_norm_eval))
            train_metrics = self.metric_calc.calculate(y_train_eval, y_pred_train)

            # Evaluate on validation set (if available)
            val_metrics = None
            if splits.X_val is not None and splits.y_val is not None:
                X_val_norm = self.X_normalizer.transform(splits.X_val)
                y_pred_val = self._denormalize_predictions(self.model.predict(X_val_norm))
                val_metrics = self.metric_calc.calculate(splits.y_val, y_pred_val)

            # Evaluate on current test chunk
            X_test_chunk = X[test_start_chunk:test_end_chunk]
            y_test_chunk = y[test_start_chunk:test_end_chunk]
            X_test_norm = self.X_normalizer.transform(X_test_chunk)
            y_pred_chunk = self._denormalize_predictions(self.model.predict(X_test_norm))
            
            test_predictions.extend(y_pred_chunk)

            test_metrics_chunk = self.metric_calc.calculate(y_test_chunk, y_pred_chunk)

            if self.verbose:
                print(f"   📈 Train Metrics: {', '.join([f'{k}: {v:.4f}' for k, v in train_metrics.items()])}")
                if val_metrics:
                    print(f"   📉 Val Metrics: {', '.join([f'{k}: {v:.4f}' for k, v in val_metrics.items()])}")
                print(f"   📊 Test Chunk Metrics ({len(y_test_chunk)} samples): {', '.join([f'{k}: {v:.4f}' for k, v in test_metrics_chunk.items()])}")

            # Record this retraining event
            retrain_history.append({
                "index": train_end,
                "train_window": (start_idx, train_end),
                "test_chunk": (test_start_chunk, test_end_chunk),
                "train_size": len(X_train_chunk),
                "test_size": len(y_test_chunk),
                "train_metrics": dict(train_metrics),
                "val_metrics": dict(val_metrics) if val_metrics is not None else None,
                "test_chunk_metrics": dict(test_metrics_chunk),
            })

        # Final test set metrics (on entire test set)
        final_test_pred = np.array(test_predictions)
        final_test_metrics = self.metric_calc.calculate(y[test_start:test_end], final_test_pred)

        if self.verbose:
            print(f"\n🏁 Final Test Metrics (full test set):")
            print(f"   📈 Score: {final_test_metrics.get('score', float('nan')):.4f}")
            print(f"   📦 Total retrain events: {len(retrain_history)}")

        return final_test_pred, final_test_metrics, retrain_history




    
    def _denormalize_predictions(self, predictions: Predictions) -> Predictions:
        """Denormalize predictions if target normalization was used."""
        if self.y_normalizer is None or self.y_normalizer.method == "none":
            return predictions
        
        # Inverse transform
        pred_reshaped = predictions.reshape(-1, 1)
        if self.y_normalizer.method == "minmax":
            assert self.y_normalizer._min is not None
            assert self.y_normalizer._max is not None
            range_val = self.y_normalizer._max - self.y_normalizer._min
            return (pred_reshaped * range_val + self.y_normalizer._min).ravel()
        
        elif self.y_normalizer.method == "standardize":
            assert self.y_normalizer._mean is not None
            assert self.y_normalizer._std is not None
            return (pred_reshaped * self.y_normalizer._std + self.y_normalizer._mean).ravel()
        
        return predictions
    
    def predict(self, X: Features) -> Predictions:
        """Make predictions on new data."""
        if self.verbose:
            print(f"🔮 Making prediction on {len(X)} samples...")
        
        X_norm = self.X_normalizer.transform(X)
        pred_norm = self.model.predict(X_norm)
        return self._denormalize_predictions(pred_norm)
    
    def save_golden_batch(self, folder: str, model_id: str, X_test: Features, y_test: Targets, version: int = 0) -> None:
        """
        Save a "golden batch" — the true targets and model predictions on a fixed test set.
        Useful for regression testing, monitoring, and CI/CD.

        Args:
            folder: Directory to save files
            model_id: Identifier for this model (e.g., "lstm_v1")
            X_test: Test features (fixed batch)
            y_test: True test targets
            version: Version number for this golden batch
        """
        if self.verbose:
            print(f"💾 Saving golden batch (version {version})...")

        # Make predictions on the fixed test set
        X_test_norm = self.X_normalizer.transform(X_test)
        y_pred_norm = self.model.predict(X_test_norm)
        y_pred = self._denormalize_predictions(y_pred_norm)

        # Compute metrics for reference
        test_metrics = self.metric_calc.calculate(y_test, y_pred)

        # Package everything
        golden_payload = {
            "version": version,
            "model_id": model_id,
            "task_type": self.task_type,
            "created_at": datetime.now().isoformat(),
            "X_test": X_test,           # Raw features (save for reproducibility)
            "y_test": y_test,           # True labels
            "y_pred": y_pred,           # Model predictions (denormalized)
            "test_metrics": test_metrics,
            "config": self.config,      # Include config for full reproducibility
            "normalizer_state": {
                "X_normalizer": self.X_normalizer,
                "y_normalizer": self.y_normalizer,
            },
        }

        # Save as .joblib
        filename = os.path.join(folder, f"golden_batch_{model_id}_{version}.joblib")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        joblib.dump(golden_payload, filename)

        print(f"✅ Golden batch saved to: {filename}")
        print(f"   Predictions shape: {y_pred.shape}, Test metrics: {test_metrics}")

    @classmethod
    def load_golden_batch(cls, filename: str) -> dict:
        """
        Load a previously saved golden batch.

        Returns:
            Dict containing: X_test, y_test, y_pred, test_metrics, config, normalizer_state
        """
        payload = joblib.load(filename)

        print(f"✅ Golden batch loaded from: {filename}")
        print(f"   Version: {payload['version']}, Created: {payload['created_at']}")
        print(f"   Test metrics: {payload['test_metrics']}")

        return payload

    def compare_with_golden_batch(self, golden_file: str) -> dict:
        """
        Compare current model's predictions on the same X_test against golden batch.
        Returns drift metrics.

        Args:
            golden_file: Path to .joblib golden batch file

        Returns:
            Dict with comparison results
        """
        golden = self.load_golden_batch(golden_file)

        # Re-run current model on the same test data
        X_test = golden["X_test"]
        y_test = golden["y_test"]

        # Normalize using current normalizer state
        X_test_norm = self.X_normalizer.transform(X_test)
        y_pred_current = self._denormalize_predictions(self.model.predict(X_test_norm))

        # Recompute metrics
        current_metrics = self.metric_calc.calculate(y_test, y_pred_current)

        # Compare against golden metrics
        golden_metrics = golden["test_metrics"]
        metric_differences = {}
        for key in golden_metrics:
            diff = current_metrics.get(key, float("nan")) - golden_metrics[key]
            metric_differences[key] = diff

        # Return comparison
        result = {
            "golden_metrics": golden_metrics,
            "current_metrics": current_metrics,
            "metric_differences": metric_differences,
            "predictions_match": np.allclose(y_pred_current, golden["y_pred"], atol=1e-5),
            "predictions_diff_norm": np.linalg.norm(y_pred_current - golden["y_pred"]),
        }

        print("📊 Golden Batch Comparison:")
        for metric, diff in metric_differences.items():
            status = "⚠️ DRIFT" if abs(diff) > 0.01 else "✅ OK"
            print(f"   {metric}: Golden={golden_metrics[metric]:.6f} → Current={current_metrics.get(metric, 'N/A'):.6f} ({diff:+.6f}) {status}")

        if result["predictions_match"]:
            print("✅ Predictions are numerically identical.")
        else:
            print(f"⚠️ Predictions differ by L2 norm: {result['predictions_diff_norm']:.8f}")

        return result


    def save_model_and_golden_batch(self, folder: str, model_id: str, X_test: Features, y_test: Targets, version: int = 0) -> None:
        # Save model
        self.model.save_to_file(folder, model_id, version)
        
        # Save golden batch
        self.save_golden_batch(folder, model_id, X_test, y_test, version)

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict
from enum import Enum
from sklearn.metrics import precision_recall_fscore_support


class MetricType(Enum):
    # Regression
    MSE = "mse"
    RMSE = "rmse"
    MAE = "mae"
    R2 = "r2"
    AUC = "auc"
    
    # Classification
    ACCURACY = "accuracy"
    F1 = "f1"
    PRECISION = "precision"
    RECALL = "recall"

    @classmethod
    def from_str(cls, s: str) -> 'MetricType':
        """Safe string-to-enum converter with validation."""
        try:
            return cls(s)
        except ValueError:
            valid_options = [m.value for m in cls]
            raise ValueError(f"Invalid metric '{s}'. Valid options: {valid_options}")


class AveragingType(Enum):
    MICRO = "micro"
    MACRO = "macro"
    WEIGHTED = "weighted"
    BINARY = "binary"

    @classmethod
    def from_str(cls, s: str) -> 'AveragingType':
        """Safe string-to-enum converter with validation."""
        try:
            return cls(s)
        except ValueError:
            valid_options = [a.value for a in cls]
            raise ValueError(f"Invalid averaging '{s}'. Valid options: {valid_options}")


class MetricCalculator(ABC):
    """Abstract base class for metric calculation with multi-metric output."""

    @abstractmethod
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate all metrics and return as dict. Default scoring metric is key 'score'."""
        pass

class RegressionMetric(MetricCalculator):
    """Metrics for regression tasks. Computes ALL metrics and returns them as dict with 'score' as default."""

    def __init__(self, metric: MetricType = MetricType.R2):
        if isinstance(metric, str):
            self.metric = MetricType.from_str(metric)
        else:
            self.metric = metric

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate ALL regression metrics and return dictionary with 'score' as default."""
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        y_mean = np.mean(y_true)

        results: Dict[str, float] = {}

        # Compute all regression metrics unconditionally
        diff = y_true - y_pred

        # MSE
        mse_val = float(np.mean(diff ** 2))
        results[MetricType.MSE.value] = mse_val

        # RMSE
        rmse_val = float(np.sqrt(mse_val))
        results[MetricType.RMSE.value] = rmse_val

        # MAE
        mae_val = float(np.mean(np.abs(diff)))
        results[MetricType.MAE.value] = mae_val

        # R²
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_mean) ** 2)
        r2_val = float(1 - (ss_res / (ss_tot + 1e-8)))
        results[MetricType.R2.value] = r2_val

        # Compute baseline (null model) for score normalization
        baseline_mse = float(np.mean((y_true - y_mean) ** 2))
        baseline_mae = float(np.mean(np.abs(y_true - y_mean)))
        baseline_rmse = float(np.sqrt(baseline_mse))

        # Set 'score' to the requested metric's value, normalized if needed (except R2)
        base_metric_value = results[self.metric.value]

        if self.metric == MetricType.R2:
            score = base_metric_value  # R² is already in [0,1]
        else:
            # For MSE, RMSE, MAE: score = 1 - (metric / baseline)
            if self.metric == MetricType.MSE:
                score = 1.0 - (base_metric_value / (baseline_mse + 1e-8))
            elif self.metric == MetricType.RMSE:
                score = 1.0 - (base_metric_value / (baseline_rmse + 1e-8))
            elif self.metric == MetricType.MAE:
                score = 1.0 - (base_metric_value / (baseline_mae + 1e-8))
            else:
                raise ValueError(f"Unrecognized regression metric: {self.metric}")

        # Clamp score to [0, 1]
        score = max(0.0, min(1.0, score))
        results["score"] = float(score)

        return results


class ClassificationMetric(MetricCalculator):
    """Metrics for classification tasks. Returns all metrics as dict with 'score' as default."""

    def __init__(
        self,
        metric: str | MetricType = MetricType.ACCURACY,
        averaging: str | AveragingType = AveragingType.MACRO,
        num_classes: int = None
    ):
        # Validate and convert metric from string if needed
        if isinstance(metric, str):
            self.metric = MetricType.from_str(metric)  # Will raise ValueError on invalid
        else:
            self.metric = metric

        # Validate and convert averaging from string if needed
        if isinstance(averaging, str):
            self.averaging = AveragingType.from_str(averaging)  # Will raise ValueError
        else:
            self.averaging = averaging

        self.num_classes = int(num_classes) if num_classes is not None else None

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate ALL classification metrics and return dictionary."""
        y_true = np.asarray(y_true).astype(int)
        y_pred = np.asarray(y_pred).astype(int)

        # Determine number of classes if not provided
        if self.num_classes is None:
            n = max(np.max(y_true), np.max(y_pred)) + 1
            self.num_classes = n

        # Use precision_recall_fscore_support with averaging based on strategy
        # This returns scalars when average is not None, arrays when average=None (we don't want that)
        precisions, recalls, f1s, _ = precision_recall_fscore_support(
            y_true, y_pred,
            average=self.averaging.value,
            zero_division=0,
            labels=list(range(self.num_classes))
        )

        # Compute accuracy (always scalar)
        accuracy = float(np.mean(y_true == y_pred))

        # Check if average is None (shouldn't happen due to above) — defensive
        if isinstance(precisions, np.ndarray):
            raise RuntimeError("Expected scalar from precision_recall_fscore_support, got array. Check averaging parameter.")

        # Convert to scalars (they should be already)
        precision = float(precisions) if not np.isnan(precisions) else 0.0
        recall = float(recalls) if not np.isnan(recalls) else 0.0
        f1 = float(f1s) if not np.isnan(f1s) else 0.0

        # Build results dictionary with ALL metrics (as required)
        results: Dict[str, float] = {
            MetricType.ACCURACY.value: accuracy,
            MetricType.F1.value: f1,
            MetricType.PRECISION.value: precision,
            MetricType.RECALL.value: recall
        }

        # Set default scoring metric based on self.metric
        results["score"] = float(results[self.metric.value])

        return results


from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Any, Optional
from sklearn.metrics import confusion_matrix
import warnings

from .fitting_core import TaskType, Targets,Features
from .fitting_models import TimeSeriesModelTrainer, TrainingConfig, MetricCalculator


# Suppress minor matplotlib warnings during plotting
warnings.filterwarnings("ignore", category=UserWarning)

class MetricVisualiser:
    """
    Visualises model performance using confusion matrix (classification) or scatter plot (regression),
    alongside calculated metrics from the trainer's MetricCalculator.

    Integrates with TimeSeriesModelTrainer’s normalizers and task_type.
    """

    def __init__(self, trainer: "TimeSeriesModelTrainer"):
        """
        Initialize with a trained TimeSeriesModelTrainer.

        Args:
            trainer (TimeSeriesModelTrainer): Already fitted trainer with trained model, normalizers, and metric_calc.
        """
        self.trainer = trainer
        self.model = trainer.model
        self.task_type = trainer.task_type  # Auto-detect from trainer
        self.metric_calc = trainer.metric_calc
        self.X_normalizer = trainer.X_normalizer
        self.y_normalizer = trainer.y_normalizer

    def visualize(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        figsize: tuple[int, int] = (14, 6),
        save_path: Optional[str] = None,
        show_metrics: bool = True
    ) -> Dict[str, float]:
        """
        Generate and display visualizations + metrics.

        Args:
            X_test (np.ndarray): Test features, raw (will be normalized using trainer's normalizer)
            y_test (np.ndarray): True test targets
            figsize (tuple): Figure size for plots
            save_path (str, optional): Path to save figure. If None, shows plot.
            show_metrics (bool): Whether to print metrics to console.

        Returns:
            Dict[str, float]: Calculated metrics dictionary (same as metric_calc.calculate)
        """

        # Normalize test features
        X_test_norm = self.X_normalizer.transform(X_test)

        # Predict
        y_pred_norm = self.model.predict(X_test_norm)
        y_pred = self._denormalize_predictions(y_pred_norm)

        # Ensure shapes match
        if y_test.ndim == 2 and y_test.shape[1] == 1:
            y_test = y_test.ravel()
        if y_pred.ndim == 2 and y_pred.shape[1] == 1:
            y_pred = y_pred.ravel()

        # Compute metrics
        metrics = self.metric_calc.calculate(y_test, y_pred)

        if show_metrics:
            print("\n" + "=" * 60)
            print("📊 MODEL METRICS")
            print("=" * 60)
            for key, value in metrics.items():
                display_key = key.replace("_", " ").title()
                print(f"{display_key:>15}: {value:8.4f}")
            print("=" * 60)

        # Plot based on task type
        plt.figure(figsize=figsize)
        
        if self.task_type == TaskType.CLASSIFICATION:
            self._plot_confusion_matrix(y_test, y_pred, metrics)
        elif self.task_type == TaskType.REGRESSION:
            self._plot_scatter_regression(y_test, y_pred, metrics)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Chart saved to: {save_path}")
        else:
            plt.show()

        return metrics

    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, metrics: Dict[str, float]) -> None:
        """Plot confusion matrix for classification tasks."""
        cm = confusion_matrix(y_true, y_pred)
        classes = np.unique(np.concatenate([y_true, y_pred]))
        
        # Normalize to percentages
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=classes, yticklabels=classes)
        plt.title("Confusion Matrix (Counts)")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        plt.subplot(1, 2, 2)
        sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap="Blues", xticklabels=classes, yticklabels=classes)
        plt.title("Confusion Matrix (% by Row)")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        # Add accuracy as title info
        accuracy = metrics.get("accuracy", 0.0)
        plt.suptitle(f"Classification Performance | Accuracy: {accuracy:.4f}", fontsize=16, y=0.98)

    def _plot_scatter_regression(self, y_true: np.ndarray, y_pred: np.ndarray, metrics: Dict[str, float]) -> None:
        """Plot scatter plot with identity line and trendline for regression tasks."""
        plt.figure(figsize=(8, 6))
        
        # Scatter plot
        plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='w', linewidth=0.5)
        
        # Identity line (perfect predictions)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Prediction")

        # Trendline (linear regression)
        z = np.polyfit(y_true, y_pred, 1)
        p = np.poly1d(z)
        plt.plot(y_true, p(y_true), "g-", lw=2, label=f"Trendline (slope={z[0]:.3f})")

        # Labels and title
        plt.title(f"Regression Performance\nR²: {metrics.get('r2', 0.0):.4f}, RMSE: {metrics.get('rmse', 0.0):.4f}", fontsize=16)
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

    def _denormalize_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Denormalize predictions using trainer's y_normalizer."""
        if self.y_normalizer is None or self.y_normalizer.method == "none":
            return predictions
        
        pred_reshaped = predictions.reshape(-1, 1)
        
        if self.y_normalizer.method == "minmax":
            assert self.y_normalizer._min is not None, "Normalizer not fitted!"
            assert self.y_normalizer._max is not None, "Normalizer not fitted!"
            range_val = self.y_normalizer._max - self.y_normalizer._min
            return (pred_reshaped * range_val + self.y_normalizer._min).ravel()
        
        elif self.y_normalizer.method == "standardize":
            assert self.y_normalizer._mean is not None, "Normalizer not fitted!"
            assert self.y_normalizer._std is not None, "Normalizer not fitted!"
            return (pred_reshaped * self.y_normalizer._std + self.y_normalizer._mean).ravel()
        
        return predictions


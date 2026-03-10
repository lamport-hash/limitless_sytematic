# visualization.py

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Union
import pandas as pd

from .fitting_core import TaskType

from .fitting_models import ModelMetrics

from .fitting_metrics import MetricType



# Optional: Align with your metric names
DEFAULT_SCORING_METRIC = "score"

class TimeSeriesModelVisualizer:
    """
    A visualization class to plot and compare model performance on time series data.
    
    Supports:
        - Single model: train/val/test metrics
        - Two-model comparison across all metrics (bar plots, radar charts, etc.)
        - Retraining history plots for periodic retraining
    """

    def __init__(self, style: str = "whitegrid", palette: str = "Set2"):
        """
        Initialize the visualizer with plot style.
        
        Args:
            style (str): seaborn style to use ("whitegrid", "darkgrid", etc.)
            palette (str): color palette name
        """
        sns.set_style(style)
        self.palette = palette

    def plot_single_model(
        self,
        metrics: ModelMetrics,
        figsize: tuple = (14, 6),
        show_retrain_history: bool = True,
        metric_keys: Optional[List[str]] = None
    ) -> plt.Figure:
        """
        Plot performance metrics for a single model across train/val/test splits.

        Args:
            metrics: ModelMetrics object from TimeSeriesModelTrainer
            figsize: Figure size (width, height)
            show_retrain_history: Whether to plot retraining events (if available)
            metric_keys: List of specific metrics to plot (e.g., ['r2', 'rmse', 'accuracy']). 
                         If None, plots all available metrics.

        Returns:
            matplotlib Figure object
        """
        # Prepare data
        splits = ["Train"]
        if metrics.val_metrics is not None:
            splits.append("Validation")
        splits.append("Test")

        # Default to all available metrics if not specified
        if metric_keys is None:
            metric_keys = list(metrics.train_metrics.keys())
        
        # Remove 'score' if present – we'll use it as the main metric to highlight
        if DEFAULT_SCORING_METRIC in metric_keys:
            metric_keys.remove(DEFAULT_SCORING_METRIC)
        # Always include 'score' as the primary metric
        all_metrics = [DEFAULT_SCORING_METRIC] + metric_keys

        # Build DataFrame
        data = []
        for split in splits:
            metrics_dict = {
                "Train": metrics.train_metrics,
                "Validation": metrics.val_metrics,
                "Test": metrics.test_metrics
            }[split]
            
            for metric in all_metrics:
                if metric in metrics_dict:
                    data.append({
                        "Split": split,
                        "Metric": metric,
                        "Value": metrics_dict[metric]
                    })

        df = pd.DataFrame(data)

        if len(df) == 0:
            raise ValueError("No valid metrics to plot.")

        # Create figure with subplots
        fig, axes = plt.subplots(1, 2 if show_retrain_history and metrics.retrain_history else 1,
                                 figsize=figsize, gridspec_kw={"width_ratios": [2, 1] if show_retrain_history and metrics.retrain_history else [3]})
        
        if not show_retrain_history or not metrics.retrain_history:
            ax = axes
        else:
            ax, ax_retrain = axes[0], axes[1]

        # Plot metrics
        sns.barplot(data=df, x="Split", y="Value", hue="Metric", palette=self.palette, ax=ax)
        ax.set_title("Model Performance Across Splits")
        ax.set_ylabel("Metric Value")
        ax.legend(title="Metrics", bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=0)

        # Rotate x-labels if needed
        for label in ax.get_xticklabels():
            label.set_rotation(0)

        plt.tight_layout()

        # Plot retraining history if available
        if show_retrain_history and metrics.retrain_history:
            retrain_df = pd.DataFrame(metrics.retrain_history)
            ax_retrain.plot(retrain_df["index"], retrain_df["train_size"], 
                           marker='o', linestyle='-', color='darkblue', label="Training Window Size")
            ax_retrain.set_title("Retraining History (Window Size)")
            ax_retrain.set_xlabel("Test Sample Index")
            ax_retrain.set_ylabel("Training Window Size")
            ax_retrain.legend()
            ax_retrain.grid(True, linestyle='--', alpha=0.6)

        return fig

    def compare_two_models(
        self,
        metrics1: ModelMetrics,
        metrics2: ModelMetrics,
        model_names: List[str] = ["Model 1", "Model 2"],
        figsize: tuple = (16, 8),
        metric_keys: Optional[List[str]] = None,
        plot_type: str = "bar",  # Options: "bar", "radar"
    ) -> plt.Figure:
        """
        Compare performance of two models across train/val/test splits.

        Args:
            metrics1, metrics2: ModelMetrics objects for two models
            model_names: Labels for the models (default: "Model 1", "Model 2")
            figsize: Figure size
            metric_keys: List of metrics to compare. If None, uses all available.
            plot_type: "bar" (default) or "radar"

        Returns:
            matplotlib Figure object
        """
        if plot_type not in ["bar", "radar"]:
            raise ValueError("plot_type must be 'bar' or 'radar'")

        # Determine metrics to compare
        if metric_keys is None:
            all_metrics = set(metrics1.train_metrics.keys()) | set(metrics2.train_metrics.keys())
            if DEFAULT_SCORING_METRIC in all_metrics:
                metric_keys = [DEFAULT_SCORING_METRIC] + list(all_metrics - {DEFAULT_SCORING_METRIC})
            else:
                metric_keys = list(all_metrics)

        # Prepare data
        splits = ["Train", "Test"]
        if metrics1.val_metrics is not None:
            splits.insert(1, "Validation")

        data = []
        for split in splits:
            m1_metrics = {"Train": metrics1.train_metrics, "Validation": metrics1.val_metrics, "Test": metrics1.test_metrics}[split]
            m2_metrics = {"Train": metrics2.train_metrics, "Validation": metrics2.val_metrics, "Test": metrics2.test_metrics}[split]
            
            for metric in metric_keys:
                val1 = m1_metrics.get(metric, np.nan)
                val2 = m2_metrics.get(metric, np.nan)
                
                data.append({"Split": split, "Metric": metric, model_names[0]: val1, model_names[1]: val2})

        df = pd.DataFrame(data)
        
        # Melt for easier plotting
        melt_cols = ["Split", "Metric"] + model_names
        df_melted = df.melt(id_vars=["Split", "Metric"], value_vars=model_names, 
                            var_name="Model", value_name="Value")

        fig = plt.figure(figsize=figsize)

        if plot_type == "bar":
            ax = sns.barplot(data=df_melted, x="Split", y="Value", hue="Model", 
                             palette=self.palette, errorbar=None)
            ax.set_title(f"Model Comparison: {model_names[0]} vs {model_names[1]}")
            ax.set_ylabel("Metric Value")
            ax.legend(title="Model")
            plt.xticks(rotation=0)
        
        elif plot_type == "radar":
            from math import pi

            # Prepare radar chart data
            num_vars = len(metric_keys)
            angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
            angles += angles[:1]  # Close the loop

            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

            # Get values for each model
            def get_model_values(model_name, metric_keys):
                vals = []
                for m in metric_keys:
                    val = df[df["Metric"] == m][model_name].values[0]
                    vals.append(val if not np.isnan(val) else 0)
                vals += vals[:1]  # Close the loop
                return vals

            model_vals_0 = get_model_values(model_names[0], metric_keys)
            model_vals_1 = get_model_values(model_names[1], metric_keys)

            # Plot data
            ax.plot(angles, model_vals_0, 'o-', linewidth=2, label=model_names[0], color='blue')
            ax.fill(angles, model_vals_0, alpha=0.25, color='blue')

            ax.plot(angles, model_vals_1, 'o-', linewidth=2, label=model_names[1], color='orange')
            ax.fill(angles, model_vals_1, alpha=0.25, color='orange')

            # Draw labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metric_keys, fontsize=10)

            # Add legend
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))

            # Set title
            ax.set_title(f"Radar Chart: {model_names[0]} vs {model_names[1]}", size=16, y=1.1)

        plt.tight_layout()
        return fig

    def plot_retrain_history(
        self,
        metrics: ModelMetrics,
        figsize: tuple = (12, 6),
        metric_to_plot: str = "score",
        show_train_window_size: bool = True
    ) -> plt.Figure:
        """
        Plot evolution of a specific metric over retraining events.

        Useful for periodic_retrain mode to see how model improves with each retrain.

        Args:
            metrics: ModelMetrics object containing retrain_history
            figsize: Figure size
            metric_to_plot: Which metric to plot over time (e.g., "score", "rmse")
            show_train_window_size: Optionally overlay training window size

        Returns:
            matplotlib Figure
        """
        if not metrics.retrain_history:
            raise ValueError("No retraining history found. This plot requires periodic_retrain mode.")

        # Extract metrics over time
        retrain_df = pd.DataFrame(metrics.retrain_history)
        
        # Try to extract metric scores from each retrain event
        # We assume you store the test performance after each retraining in the history
        # If not, we'll just show window size.

        if "test_metrics" in metrics.retrain_history[0]:
            # We need to extract the metric values
            scores = [h.get("test_metrics", {}).get(metric_to_plot, np.nan) for h in metrics.retrain_history]
            retrain_df[metric_to_plot] = scores
        else:
            # Fallback: use train score or just show window size
            if metric_to_plot == "score":
                scores = [h.get("train_score", np.nan) for h in metrics.retrain_history]
                retrain_df[metric_to_plot] = scores
            else:
                raise ValueError(f"Metric '{metric_to_plot}' not found in retrain_history. "
                                 "Ensure test/train metrics are logged during retraining.")

        fig, ax1 = plt.subplots(figsize=figsize)

        # Plot metric over retrain indices
        ax1.plot(retrain_df["index"], retrain_df[metric_to_plot], 
                 marker='s', linestyle='-', color='green', linewidth=2, label=f"{metric_to_plot}")
        ax1.set_xlabel("Sample Index")
        ax1.set_ylabel(f"{metric_to_plot.capitalize()}", color='green')
        ax1.tick_params(axis='y', labelcolor='green')

        # Add training window size if requested
        if show_train_window_size:
            ax2 = ax1.twinx()
            ax2.plot(retrain_df["index"], retrain_df["train_size"],
                     marker='o', linestyle='--', color='purple', linewidth=1.5, label="Training Window Size")
            ax2.set_ylabel("Training Window Size", color='purple')
            ax2.tick_params(axis='y', labelcolor='purple')

        # Title and legend
        plt.title(f"Model Performance Over Retraining Events ({metric_to_plot})")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels() if show_train_window_size else ([], [])
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        return fig

    def plot_prediction_vs_actual(
        self,
        metrics: ModelMetrics,
        figsize: tuple = (10, 6),
        title_suffix: str = ""
    ) -> plt.Figure:
        """
        Plot actual vs predicted values for test set (useful for regression).

        For classification: plots confusion matrix.

        Args:
            metrics: ModelMetrics containing predictions and true values
            figsize: Figure size
            title_suffix: Optional suffix for plot title

        Returns:
            matplotlib Figure
        """
        if metrics.predictions is None or not hasattr(metrics, "test_indices"):
            raise ValueError("No predictions available. Run fit() first.")


        if self.task_type == TaskType.REGRESSION:
            # Regression: scatter plot
            y_true = metrics.y_test
            y_pred = metrics.predictions

            fig, ax = plt.subplots(figsize=figsize)
            ax.scatter(y_true, y_pred, alpha=0.6, color='blue')
            
            # Diagonal line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title(f"Predictions vs Actual (Test) {title_suffix}")
            plt.grid(True, linestyle='--', alpha=0.6)
        
        else:  # Classification
            from sklearn.metrics import confusion_matrix

            y_true = metrics.y_test
            y_pred = metrics.predictions
            classes = np.unique(y_true)

            cm = confusion_matrix(y_true, y_pred)
            
            fig, ax = plt.subplots(figsize=figsize)
            sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
            ax.set_title(f"Confusion Matrix {title_suffix}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

        plt.tight_layout()
        return fig

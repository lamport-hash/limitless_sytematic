"""
Examples using the Time Series ML API with XGBoost and MLP (Neural Networks)

This demonstrates:
1. XGBoost with three-way split and hyperparameter tuning
2. MLP with periodic retraining and sliding window
3. Both regression and classification tasks
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.neural_network import MLPRegressor, MLPClassifier
import xgboost as xgb

from fitting.fitting_core import BaseModel, TaskType, Targets, Predictions, RetrainMode, Features

from fitting.fitting_models import (
    TimeSeriesModelTrainer,
    TrainingConfig,
    TrainingSplitType
)

from fitting.fitting_metrics import MetricType, ClassificationMetric, RegressionMetric


from fitting.models.xgb_model import XGBoostModel
from fitting.models.mlp_model import MLPModel


# ===================================================================
# FIXTURES: Data & Configs (Centralized, Reusable)
# ===================================================================

@pytest.fixture
def regression_data():
    """Generate synthetic regression data with temporal patterns."""
    np.random.seed(42)
    n_samples = 500
    n_features = 10
    t = np.linspace(0, 10, n_samples)
    X = np.random.randn(n_samples, n_features)
    X[:, 0] = np.sin(t) + np.random.randn(n_samples) * 0.02
    X[:, 1] = np.cos(t) + np.random.randn(n_samples) * 0.02
    y = 2 * np.sin(t) + 0.5 * np.cos(2 * t) + 0.1 * t + np.random.randn(n_samples) * 0.1
    return X, y


@pytest.fixture
def classification_data():
    """Generate synthetic binary classification data with temporal patterns."""
    np.random.seed(456)
    n_samples = 500
    n_features = 6
    t = np.linspace(0, 15, n_samples)
    X = np.random.randn(n_samples, n_features)
    X[:, 0] = np.sin(t) + np.random.randn(n_samples) * 0.3
    X[:, 1] = np.cos(t) + np.random.randn(n_samples) * 0.3
    decision_boundary = X[:, 0] + 0.5 * X[:, 1]
    y = (decision_boundary > 0).astype(np.float32)
    # Add noise
    noise_idx = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
    y[noise_idx] = 1 - y[noise_idx]
    return X, y


@pytest.fixture
def multiclass_data():
    """Generate synthetic multi-class classification data."""
    np.random.seed(789)
    n_samples = 500
    n_features = 8
    X = np.random.randn(n_samples, n_features)
    y = np.zeros(n_samples)
    y[X[:, 0] + X[:, 1] > 1] = 1
    y[X[:, 0] - X[:, 1] > 1] = 2
    return X, y


@pytest.fixture
def regression_config_three_way():
    """Standard three-way split config for regression."""
    return TrainingConfig(
        mode=TrainingSplitType.TRAIN_VAL_TEST,
        train_ratio=0.6,
        val_ratio=0.2,
        normalization="standardize",
        normalize_targets=False
    )


@pytest.fixture
def regression_config_periodic_expanding():
    """Periodic retraining with expanding window for regression."""
    return TrainingConfig(
        mode=TrainingSplitType.PERIODIC_RETRAIN,
        train_ratio=0.5,
        retrain_period=100,
        retrain_mode=RetrainMode.EXPANDING,
        normalization="standardize",
        normalize_targets=True
    )


@pytest.fixture
def classification_config_sliding():
    """Periodic retraining with sliding window for classification."""
    return TrainingConfig(
        mode=TrainingSplitType.PERIODIC_RETRAIN,
        train_ratio=0.65,
        retrain_period=75,
        retrain_mode=RetrainMode.SLIDING,
        sliding_window_size=300,
        normalization="standardize",
        normalize_targets=False
    )

@pytest.fixture
def classification_config_periodic_expanding():
    """Periodic retraining with sliding window for classification."""
    return TrainingConfig(
        mode=TrainingSplitType.PERIODIC_RETRAIN,
        train_ratio=0.65,
        retrain_period=75,
        retrain_mode=RetrainMode.EXPANDING,
        normalization="standardize",
        normalize_targets=False
    )


@pytest.fixture
def classification_config_three_way():
    """Standard three-way split config for classification."""
    return TrainingConfig(
        mode=TrainingSplitType.TRAIN_VAL_TEST,
        train_ratio=0.6,
        val_ratio=0.2,
        normalization="standardize"
    )


@pytest.fixture
def xgboost_regression_model():
    """Factory for XGBoost regression model."""
    def _make():
        model = XGBoostModel(task_type=TaskType.REGRESSION)
        model.setParams(**model.getDefaultParams(task_type=TaskType.REGRESSION))
        return model
    return _make


@pytest.fixture
def xgboost_classification_model():
    """Factory for XGBoost classification model."""
    def _make():
        model = XGBoostModel(task_type=TaskType.CLASSIFICATION)
        model.setParams(**model.getDefaultParams(task_type=TaskType.CLASSIFICATION))
        return model
    return _make


@pytest.fixture
def mlp_regression_model():
    """Factory for MLP regression model."""
    def _make(n_features):
        return MLPModel(
            task_type=TaskType.REGRESSION,
            input_size=n_features,
            hidden_sizes=[128,64,32],
            output_size=1
        )
    return _make


@pytest.fixture
def mlp_classification_model():
    """Factory for MLP classification model."""
    def _make(n_features):
        return MLPModel(
            task_type=TaskType.CLASSIFICATION,
            input_size=n_features,
            hidden_sizes=[128,64],
            output_size=1
        )
    return _make


# ===================================================================
# TESTS: Clean, Focused, Parameterized
# ===================================================================


def test_example_xgboost_regression(regression_data, regression_config_three_way, xgboost_regression_model):
    """XGBoost for time series regression with three-way split."""
    print("=" * 80)
    print("EXAMPLE 1: XGBoost Regression with Three-Way Split")
    print("=" * 80)

    X, y = regression_data
    print(f"\nDataset: {len(X)} samples, {X.shape[1]} features")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")

    model = xgboost_regression_model()
    trainer = TimeSeriesModelTrainer(
        model=model,
        config=regression_config_three_way,
        task_type=TaskType.REGRESSION,
        metric_calculator=RegressionMetric()
    )

    metrics = trainer.fit(X, y)
    print(f"\n{metrics}")

    # Assert: RMSE should be reasonably low for synthetic data
    assert metrics.test_score > 0, "Test RMSE should be positive"
    assert isinstance(metrics.test_score, float)


def test_example_xgboost_regression(regression_data, regression_config_periodic_expanding, xgboost_regression_model):
    """XGBoost for time series regression with three-way split."""
    print("=" * 80)
    print("EXAMPLE 1: XGBoost Regression with Three-Way Split")
    print("=" * 80)

    X, y = regression_data
    print(f"\nDataset: {len(X)} samples, {X.shape[1]} features")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")

    print(regression_config_periodic_expanding.retrain_period)

    model = xgboost_regression_model()
    trainer = TimeSeriesModelTrainer(
        model=model,
        config=regression_config_periodic_expanding,
        task_type=TaskType.REGRESSION,
        metric_calculator=RegressionMetric()
    )

    metrics = trainer.fit(X, y)
    
    assert len(metrics.retrain_history) > 0, "Should have at least one retraining event"
    assert metrics.test_score > 0, "Test RMSE should be positive"

def test_mlp_regression_periodic(regression_data, regression_config_periodic_expanding, mlp_regression_model):
    """MLP Regression with periodic retraining (expanding window)."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: MLP Regression with Periodic Retraining (Expanding Window)")
    print("=" * 80)

    X, y = regression_data
    n_features = X.shape[1]
    print(f"\nDataset: {len(X)} samples, {n_features} features")
    print("Note: Concept drift simulated")

    model = mlp_regression_model(n_features)
    model.setParams(**model.getDefaultParams(task_type=TaskType.REGRESSION))

    trainer = TimeSeriesModelTrainer(
        model=model,
        config=regression_config_periodic_expanding,
        task_type=TaskType.REGRESSION,
        metric_calculator=RegressionMetric()
    )

    print("\n--- Training with Periodic Retraining ---")
    metrics = trainer.fit(X, y)
    print(f"\n{metrics}")
    print(f"Retrain Events: {len(metrics.retrain_history)}")

    assert len(metrics.retrain_history) > 0, "Should have at least one retraining event"
    assert metrics.test_score > 0

    # Compare with no retraining
    print("\n--- Comparison: Same Model WITHOUT Retraining ---")
    config_no_retrain = TrainingConfig(
        mode=TrainingSplitType.TRAIN_VAL_TEST,
        train_ratio=0.5,
        val_ratio=0.15,
        normalization="standardize",
        normalize_targets=True
    )

    model_no_retrain = mlp_regression_model(n_features)
    model_no_retrain.setParams(**model.getDefaultParams(task_type=TaskType.REGRESSION))

    trainer_no_retrain = TimeSeriesModelTrainer(
        model=model_no_retrain,
        config=config_no_retrain,
        task_type=TaskType.REGRESSION,
        metric_calculator=RegressionMetric()
    )

    metrics_no_retrain = trainer_no_retrain.fit(X, y)
    print(f"\n{metrics_no_retrain}")

    # Assert retraining improves performance
    improvement = metrics.test_score - metrics_no_retrain.test_score
    assert improvement >= 0, f"Retraining should not hurt performance; got {improvement:.4f} RMSE reduction"
    print(f"*** Performance Improvement with Retraining: {improvement:.4f} RMSE reduction ***")


def test_mlp_classification_sliding(classification_data, classification_config_sliding, mlp_classification_model):
    """MLP Classification with sliding window retraining."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: MLP Classification with Periodic Retraining (Sliding Window)")
    print("=" * 80)

    X, y = classification_data
    n_features = X.shape[1]
    print(f"\nDataset: {len(X)} samples, {n_features} features")
    print(f"Class distribution: Class 0: {(y == 0).sum()}, Class 1: {(y == 1).sum()}")

    model = mlp_classification_model(n_features)
    model.setParams(**model.getDefaultParams(task_type=TaskType.CLASSIFICATION))

    trainer = TimeSeriesModelTrainer(
        model=model,
        config=classification_config_sliding,
        task_type=TaskType.CLASSIFICATION,
        metric_calculator=ClassificationMetric()
    )

    print("\n--- Training with Sliding Window Retraining ---")
    metrics = trainer.fit(X, y)
    print(f"\n{metrics}")
    print(f"Retrain Events: {len(metrics.retrain_history)}")

    assert len(metrics.retrain_history) > 0
    assert 0 <= metrics.test_score <= 1.0, "Accuracy should be between 0 and 1"

    # Test additional metrics
    print("\n--- Additional Classification Metrics ---")
    for metric_name in [MetricType.PRECISION, MetricType.RECALL, MetricType.F1]:
        metric_calc = ClassificationMetric(metric_name)
        trainer_metric = TimeSeriesModelTrainer(
            model=model,
            config=classification_config_sliding,
            task_type=TaskType.CLASSIFICATION,
            metric_calculator=metric_calc
        )
        metrics_additional = trainer_metric.fit(X, y)
        print(f"{metric_name }: "
              f"Train={metrics_additional.train_score:.4f}, "
              f"Test={metrics_additional.test_score:.4f}")

        assert 0 <= metrics_additional.test_score <= 1.0


def test_xgboost_classification(multiclass_data, classification_config_three_way, xgboost_classification_model):
    """XGBoost for multiclass classification with hyperparameter tuning."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: XGBoost Classification with Hyperparameter Tuning")
    print("=" * 80)

    X, y = multiclass_data
    n_classes = len(np.unique(y))
    print(f"\nDataset: {len(X)} samples, {X.shape[1]} features, {n_classes} classes")
    for cls in range(n_classes):
        print(f"  Class {cls}: {(y == cls).sum()} samples")

    model = xgboost_classification_model()

    # Hyperparameter tuning (simplified)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
    }

    print("\n--- Hyperparameter Tuning ---")
    # Note: If your TimeSeriesModelTrainer supports param_grid, use it here.
    # For now, just fit with default params since tuning logic isn't implemented in the example
    trainer = TimeSeriesModelTrainer(
        model=model,
        config=classification_config_three_way,
        task_type=TaskType.CLASSIFICATION,
        metric_calculator=ClassificationMetric()
    )

    metrics = trainer.fit(X, y)
    print(f"\n{metrics}")

    assert 0 <= metrics.test_score <= 1.0, "Accuracy should be in [0, 1]"



def test_xgb_classification_sliding(classification_data, classification_config_sliding, xgboost_classification_model):
    """MLP Classification with sliding window retraining."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: MLP Classification with Periodic Retraining (Sliding Window)")
    print("=" * 80)

    X, y = classification_data
    n_features = X.shape[1]
    print(f"\nDataset: {len(X)} samples, {n_features} features")
    print(f"Class distribution: Class 0: {(y == 0).sum()}, Class 1: {(y == 1).sum()}")

    model = xgboost_classification_model()
    model.setParams(**model.getDefaultParams(task_type=TaskType.CLASSIFICATION))

    trainer = TimeSeriesModelTrainer(
        model=model,
        config=classification_config_sliding,
        task_type=TaskType.CLASSIFICATION,
        metric_calculator=ClassificationMetric()
    )

    print("\n--- Training with Sliding Window Retraining ---")
    metrics = trainer.fit(X, y)
    print(f"\n{metrics}")
    print(f"Retrain Events: {len(metrics.retrain_history)}")

    assert len(metrics.retrain_history) > 0
    assert 0 <= metrics.test_score <= 1.0, "Accuracy should be between 0 and 1"

    # Test additional metrics
    print("\n--- Additional Classification Metrics ---")
    for metric_name in [MetricType.PRECISION, MetricType.RECALL, MetricType.F1]:
        metric_calc = ClassificationMetric(metric_name)
        trainer_metric = TimeSeriesModelTrainer(
            model=model,
            config=classification_config_sliding,
            task_type=TaskType.CLASSIFICATION,
            metric_calculator=metric_calc
        )
        metrics_additional = trainer_metric.fit(X, y)
        print(f"{metric_name }: "
              f"Train={metrics_additional.train_score:.4f}, "
              f"Test={metrics_additional.test_score:.4f}")

        assert 0 <= metrics_additional.test_score <= 1.0




def test_xgb_classification_periodic(classification_data, classification_config_periodic_expanding, xgboost_classification_model):
    """XGB Classification with periodic window retraining."""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: XGB Classification with Periodic Retraining (Sliding Window)")
    print("=" * 80)

    X, y = classification_data
    n_features = X.shape[1]
    print(f"\nDataset: {len(X)} samples, {n_features} features")
    print(f"Class distribution: Class 0: {(y == 0).sum()}, Class 1: {(y == 1).sum()}")

    model = xgboost_classification_model()
    model.setParams(**model.getDefaultParams(task_type=TaskType.CLASSIFICATION))

    trainer = TimeSeriesModelTrainer(
        model=model,
        config=classification_config_periodic_expanding,
        task_type=TaskType.CLASSIFICATION,
        metric_calculator=ClassificationMetric()
    )

    print("\n--- Training with Sliding Window Retraining ---")
    metrics = trainer.fit(X, y)
    print(f"\n{metrics}")
    print(f"Retrain Events: {len(metrics.retrain_history)}")

    assert len(metrics.retrain_history) > 0
    assert 0 <= metrics.test_score <= 1.0, "Accuracy should be between 0 and 1"

    # Test additional metrics
    print("\n--- Additional Classification Metrics ---")
    for metric_name in [MetricType.PRECISION, MetricType.RECALL, MetricType.F1]:
        metric_calc = ClassificationMetric(metric_name)
        trainer_metric = TimeSeriesModelTrainer(
            model=model,
            config=classification_config_periodic_expanding,
            task_type=TaskType.CLASSIFICATION,
            metric_calculator=metric_calc
        )
        metrics_additional = trainer_metric.fit(X, y)
        print(f"{metric_name }: "
              f"Train={metrics_additional.train_score:.4f}, "
              f"Test={metrics_additional.test_score:.4f}")

        assert 0 <= metrics_additional.test_score <= 1.0
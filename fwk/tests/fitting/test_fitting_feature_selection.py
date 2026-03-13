import pytest
import numpy as np
import pandas as pd
from typing import List
from unittest.mock import MagicMock, patch

from fitting.fitting_feature_selection import (
    FeatureSelector,
    TaskType,
    FeatureSelectionMethod,
)

from fitting.fitting_metrics import RegressionMetric, ClassificationMetric
from fitting.fitting_models import TrainingConfig, TrainingSplitType
from fitting.models.xgb_model import XGBoostModel


# =======================
# Test FeatureSelector Core Logic
# =======================

@pytest.fixture
def sample_data():
    """Create a small, reproducible dataset with 4 features and target."""
    np.random.seed(42)
    n = 100
    data = pd.DataFrame({
        "f1": np.random.randn(n),
        "f2": np.sin(np.linspace(0, 4 * np.pi, n)) + 0.1 * np.random.randn(n),
        "f3": np.cumsum(np.random.randn(n)),
        "noise": np.random.rand(n),
    })
    target = data["f1"] + 0.5 * data["f2"] + np.random.randn(n) * 0.1
    return data, target

@pytest.fixture
def sample_classification_data(n_samples: int = 1000, n_features: int = 10, n_classes: int = 2):
    """
    Generate synthetic data for classification.
    
    Returns:
        features_df: pd.DataFrame with shape (n_samples, n_features)
        target_df: pd.Series with integer class labels [0, 1, ..., n_classes-1]
    """
    np.random.seed(42)  # for reproducibility

    # Generate random features
    features_df = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feat_{i}" for i in range(n_features)]
    )

    # Generate discrete class labels: 0 to n_classes-1
    target_labels = np.random.choice(range(n_classes), size=n_samples, replace=True)
    target_df = pd.Series(target_labels, name="target")

    return features_df, target_df

@pytest.fixture
def config():
    """Return a minimal TrainingConfig for testing."""
    return TrainingConfig(
        mode=TrainingSplitType.TRAIN_VAL_TEST,
        train_ratio=0.7,
        val_ratio=0.2,
        normalization="standardize"
    )


def test_feature_selector_initialization(sample_data, config):
    """Test that FeatureSelector initializes correctly."""
    features_df, target_df = sample_data
    model = XGBoostModel(TaskType.REGRESSION)
    metric = RegressionMetric()
    selector = FeatureSelector(
        model=model,
        config=config,
        features_df=features_df,
        target_df=target_df,
        max_features=3,
        metric_calculator=metric, 
        verbose=False
    )

    assert selector.model == model
    assert len(selector.features_df.columns) == 4


def test_feature_selector_fit_exhaustive(sample_data, config):
    """Test exhaustive feature selection works and produces results."""
    features_df, target_df = sample_data
    model = XGBoostModel(TaskType.REGRESSION)
    metric = RegressionMetric()
    selector = FeatureSelector(
        model=model,
        config=config,
        features_df=features_df,
        target_df=target_df,
        max_features=3,
        min_features=1,
        feature_selection_strategy = FeatureSelectionMethod.EXHAUSTIVE,
        metric_calculator=metric, 
        verbose=False
    )

    selector.fit()

    assert len(selector.results) > 0
    assert len(selector.best_features) >= selector.min_features
    assert selector.best_features == ['f1','f2']
    assert selector.best_score is not None

    # Test results DataFrame
    df = selector.get_results_df()
    assert set(df.columns) == {"features", "bundler", "score", "num_features"}
    assert len(df) > 0


def test_feature_selector_fit_greedy_rfe(sample_data, config):
    """Test greedy RFE strategy works."""
    features_df, target_df = sample_data
    model = XGBoostModel(TaskType.REGRESSION)
    metric = RegressionMetric()
    selector = FeatureSelector(
        model=model,
        config=config,
        features_df=features_df,
        target_df=target_df,
        max_features=3,
        min_features=1,
        feature_selection_strategy = FeatureSelectionMethod.GREEDY_RFE,
        metric_calculator=metric, 
        verbose=False
    )
    selector.fit()

    assert len(selector.best_features) >= 1
    assert len(selector.results) > 0
    assert selector.best_score is not None
    assert set(selector.best_features) == {'f1', 'f2'}  # This should be the best!

def test_feature_selector_predict(sample_data, config):
    """Test that predict() uses the same feature selection and bundling as fit()."""
    features_df, target_df = sample_data
    model = XGBoostModel(TaskType.REGRESSION)
    metric = RegressionMetric()
    selector = FeatureSelector(
        model=model,
        config=config,
        features_df=features_df,
        target_df=target_df,
        max_features=3,
        min_features=1,
        feature_selection_strategy = FeatureSelectionMethod.GREEDY_RFE,
        metric_calculator=metric, 
        verbose=False
    )
    selector.fit()

    # Predict on new data with same columns
    X_new = features_df.iloc[:3].copy()
    predictions = selector.predict(X_new)

    assert len(predictions) == 3
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (3,)


def test_feature_selector_with_classification(sample_classification_data, config):
    """Test classification task type with accuracy metric."""
    features_df, target_df = sample_classification_data
    model = XGBoostModel(TaskType.CLASSIFICATION)
    metric = ClassificationMetric()
    selector = FeatureSelector(
        model=model,
        config=config,
        features_df=features_df,
        target_df=target_df,
        max_features=3,
        min_features=1,
        feature_selection_strategy = FeatureSelectionMethod.GREEDY_RFE,
        metric_calculator=metric, 
        verbose=False
    )
    selector.fit()
    
    
    assert selector.task_type == TaskType.CLASSIFICATION
    assert isinstance(selector.metric_calc, ClassificationMetric)
    assert selector.best_score >= 0.0 and selector.best_score <= 1.0

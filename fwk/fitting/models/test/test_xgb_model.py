import numpy as np
import pytest
from xgboost import XGBClassifier, XGBRegressor

from fitting.fitting_core import TaskType
from fitting.models.xgb_model import XGBoostModel
from numpy.typing import NDArray


# Helper: Generate synthetic data
def generate_classification_data(n_samples=100, n_features=5):
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([0, 1], size=n_samples)
    return X.astype(np.float32), y.astype(np.float32)


def generate_regression_data(n_samples=100, n_features=5):
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.randn(n_samples) * 0.1  # linear target
    return X.astype(np.float32), y.astype(np.float32)


# -------------------- TEST: CLASSIFICATION MODE --------------------

def test_xgb_model_classification():
    """Test XGBoostModel in classification mode."""
    
    # Generate data
    X, y = generate_classification_data()

    task_type=TaskType.CLASSIFICATION

    # Initialize model in classification mode
    model = XGBoostModel(task_type=task_type)

    # Assert correct internal model type
    assert isinstance(model._model, XGBClassifier), "Model should be XGBClassifier in classification mode"

    # Test default parameters
    default_params = model.getDefaultParams(task_type=task_type)
    assert isinstance(default_params, dict)
    assert 'n_estimators' in default_params

    # Test parameter setting
    custom_params = {
        'n_estimators': 50,
        'max_depth': 3,
        'learning_rate': 0.2
    }
    model.setParams(**custom_params)
    current_params = model.getParams()
    for key, value in custom_params.items():
        assert current_params[key] == value

    # Test parameter grid
    param_grid = model.getParamsGrids(task_type=task_type)
    assert 'n_estimators' in param_grid
    assert len(param_grid['n_estimators']) > 0
    assert 'learning_rate' in param_grid

    # Test fit and predict
    model.fit(X, y)
    predictions = model.predict(X)

    # Assertions on prediction output
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (len(y),)  # 1D array
    assert np.issubdtype(predictions.dtype, np.floating)
    assert np.all(np.isfinite(predictions))  # No NaN or inf

    # Predictions should be class probabilities? Actually, XGBClassifier predict() returns class labels (not probs)
    # Check they are integers cast to float
    unique_preds = np.unique(predictions)
    assert set(unique_preds).issubset({0.0, 1.0}), "Classification predictions should be 0 or 1"


# -------------------- TEST: REGRESSION MODE --------------------

def test_xgb_model_regression():
    """Test XGBoostModel in regression mode."""
    
    # Generate data
    X, y = generate_regression_data()
    task_type=TaskType.REGRESSION

    # Initialize model in regression mode
    model = XGBoostModel(task_type=task_type)

    # Assert correct internal model type
    assert isinstance(model._model, XGBRegressor), "Model should be XGBRegressor in regression mode"

    # Test default parameters
    default_params = model.getDefaultParams(task_type=task_type)
    assert isinstance(default_params, dict)
    assert 'n_estimators' in default_params
    assert default_params['eval_metric'] == 'rmse'

    # Test parameter setting
    custom_params = {
        'n_estimators': 200,
        'max_depth': 9,
        'learning_rate': 0.05
    }
    model.setParams(**custom_params)
    current_params = model.getParams()
    for key, value in custom_params.items():
        assert current_params[key] == value

    # Test parameter grid
    param_grid = model.getParamsGrids(task_type=task_type)
    assert 'max_depth' in param_grid
    assert len(param_grid['max_depth']) == 3

    # Test fit and predict
    model.fit(X, y)
    predictions = model.predict(X)

    # Assertions on prediction output
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (len(y),)  # 1D array
    assert np.issubdtype(predictions.dtype, np.floating)
    assert np.all(np.isfinite(predictions))

    # Check that predictions are continuous (not discrete)
    unique_preds = np.unique(predictions)
    assert len(unique_preds) > 2, "Regression predictions should be continuous values"

    # Optional: check that RMSE is reasonable (just a sanity check)
    mse = np.mean((predictions - y) ** 2)
    assert mse < 1.0, f"Regression MSE too high: {mse}"


import pytest
import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import make_classification, make_regression

from fitting.models.rf_model import RandomForestModel
from fitting.fitting_core import TaskType

# =====================
# Test Case: Classification
# =====================

def test_random_forest_classification_initialization():
    """Test that RandomForestModel initializes correctly for classification."""
    model = RandomForestModel(task_type=TaskType.CLASSIFICATION)
    assert model.task_type == TaskType.CLASSIFICATION
    assert model._model.__class__.__name__ == "RandomForestClassifier"
    
    # Check default parameters
    defaults = model._params
    assert defaults["n_estimators"] == 100
    assert defaults["max_depth"] is None
    assert defaults["min_samples_split"] == 2
    assert defaults["min_samples_leaf"] == 1
    assert defaults["max_features"] == "sqrt"
    assert defaults["random_state"] == 42


def test_random_forest_classification_set_params():
    """Test that setParams correctly updates model parameters for classification."""
    model = RandomForestModel(task_type=TaskType.CLASSIFICATION)
    model.setParams(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="log2",
        random_state=123
    )
    
    params = model._params  # Should reflect updated values
    assert params["n_estimators"] == 200
    assert params["max_depth"] == 15
    assert params["min_samples_split"] == 5
    assert params["min_samples_leaf"] == 2
    assert params["max_features"] == "log2"
    assert params["random_state"] == 123
    
    # Confirm underlying model was re-initialized with new params
    assert model._model.n_estimators == 200
    assert model._model.max_depth == 15
    assert model._model.min_samples_split == 5
    assert model._model.min_samples_leaf == 2
    assert model._model.max_features == "log2"
    assert model._model.random_state == 123


def test_random_forest_classification_fit_predict():
    """Test that classification model fits and predicts correctly."""
    X, y = make_classification(
        n_samples=200,
        n_features=4,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        random_state=42
    )

    model = RandomForestModel(task_type=TaskType.CLASSIFICATION)
    model.fit(X, y)
    
    predictions = model.predict(X)
    
    # Check output type and shape
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (200,)
    
    # Predictions should be integer class labels (even if dtype=float64)
    unique_preds = np.unique(predictions)
    expected_classes = np.unique(y)
    
    # Ensure predicted values are discrete and match class labels
    assert set(unique_preds).issubset(set(expected_classes))
    
    # Check accuracy — should be high on training set
    accuracy = np.mean(predictions == y)
    assert accuracy > 0.85 


def test_random_forest_classification_predict_proba():
    """Test that predict_proba works for classification and raises error for regression."""
    X, y = make_classification(
        n_samples=200,
        n_features=4,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        random_state=42
    )

    model = RandomForestModel(task_type=TaskType.CLASSIFICATION)
    model.fit(X, y)

    proba = model.predict_proba(X)
    
    # Check shape: (n_samples, n_classes)
    assert proba.shape == (200, 2)
    # Check probabilities sum to 1 per sample
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

    # Test that predict_proba raises error for regression
    X_reg, y_reg = make_regression(n_samples=200, n_features=4, random_state=42)
    model_reg = RandomForestModel(task_type=TaskType.REGRESSION)
    model_reg.fit(X_reg, y_reg)

    with pytest.raises(ValueError, match="predict_proba is only available for classification tasks"):
        model_reg.predict_proba(X_reg)


def test_random_forest_classification_param_grid():
    """Test that parameter grid is correctly structured for classification."""
    task_type = TaskType.CLASSIFICATION
    model = RandomForestModel(task_type=task_type)
    param_grid = model.getParamsGrids(task_type=task_type)
    
    assert isinstance(param_grid, dict)
    assert "n_estimators" in param_grid
    assert "max_depth" in param_grid
    assert "min_samples_split" in param_grid
    assert "min_samples_leaf" in param_grid
    assert "max_features" in param_grid
    
    # Check values are lists
    assert isinstance(param_grid["n_estimators"], list)
    assert 50 in param_grid["n_estimators"]
    assert 100 in param_grid["n_estimators"]
    assert 200 in param_grid["n_estimators"]

    assert isinstance(param_grid["max_depth"], list)
    assert None in param_grid["max_depth"]
    assert 10 in param_grid["max_depth"]

    assert isinstance(param_grid["max_features"], list)
    assert "sqrt" in param_grid["max_features"]
    assert "log2" in param_grid["max_features"]
    assert None in param_grid["max_features"]


# =====================
# Test Case: Regression
# =====================

def test_random_forest_regression_initialization():
    """Test that RandomForestModel initializes correctly for regression."""
    model = RandomForestModel(task_type=TaskType.REGRESSION)
    assert model.task_type == TaskType.REGRESSION
    assert model._model.__class__.__name__ == "RandomForestRegressor"
    
    # Check default parameters
    defaults = model._params
    assert defaults["n_estimators"] == 100
    assert defaults["max_depth"] is None
    assert defaults["min_samples_split"] == 2
    assert defaults["min_samples_leaf"] == 1
    assert defaults["max_features"] == "sqrt"
    assert defaults["random_state"] == 42


def test_random_forest_regression_set_params():
    """Test that setParams correctly updates model parameters for regression."""
    model = RandomForestModel(task_type=TaskType.REGRESSION)
    model.setParams(
        n_estimators=150,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=3,
        max_features=None,
        random_state=99
    )
    
    params = model._params
    assert params["n_estimators"] == 150
    assert params["max_depth"] == 20
    assert params["min_samples_split"] == 10
    assert params["min_samples_leaf"] == 3
    assert params["max_features"] is None
    assert params["random_state"] == 99
    
    # Confirm underlying model was updated
    assert model._model.n_estimators == 150
    assert model._model.max_depth == 20
    assert model._model.min_samples_split == 10
    assert model._model.min_samples_leaf == 3
    assert model._model.max_features is None
    assert model._model.random_state == 99


def test_random_forest_regression_fit_predict():
    """Test that regression model fits and predicts correctly."""
    X, y = make_regression(
        n_samples=200,
        n_features=4,
        noise=0.1,
        random_state=42
    )

    model = RandomForestModel(task_type=TaskType.REGRESSION)
    model.fit(X, y)
    
    predictions = model.predict(X)
    
    # Check output type and shape
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (200,)
    
    # Check that predictions are float64 (as per contract)
    assert predictions.dtype == np.float64

    # Check R² score on training set (should be high)
    from sklearn.metrics import r2_score
    r2 = r2_score(y, predictions)
    assert r2 > 0.85  # RF should fit this simple regression well


def test_random_forest_regression_param_grid():
    """Test that parameter grid is correctly structured for regression."""
    task_type = TaskType.REGRESSION
    model = RandomForestModel(task_type=task_type)
    param_grid = model.getParamsGrids(task_type=task_type)
    
    assert isinstance(param_grid, dict)
    assert "n_estimators" in param_grid
    assert "max_depth" in param_grid
    assert "min_samples_split" in param_grid
    assert "min_samples_leaf" in param_grid
    assert "max_features" in param_grid
    
    # Check values are lists
    assert isinstance(param_grid["n_estimators"], list)
    assert 50 in param_grid["n_estimators"]
    assert 100 in param_grid["n_estimators"]
    assert 200 in param_grid["n_estimators"]

    assert isinstance(param_grid["max_depth"], list)
    assert None in param_grid["max_depth"]
    assert 10 in param_grid["max_depth"]

    assert isinstance(param_grid["max_features"], list)
    assert "sqrt" in param_grid["max_features"]
    assert "log2" in param_grid["max_features"]
    assert None in param_grid["max_features"]



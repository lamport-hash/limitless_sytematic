import pytest
import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import make_classification, make_regression

from fitting.models.dt_model import DecisionTreeModel 
from fitting.fitting_core import TaskType

# =====================
# Test Case: Classification
# =====================

def test_decision_tree_classification_initialization():
    """Test that DecisionTreeModel initializes correctly for classification."""
    model = DecisionTreeModel(task_type=TaskType.CLASSIFICATION)
    assert model.task_type == TaskType.CLASSIFICATION
    assert model._model.__class__.__name__ == "DecisionTreeClassifier"
    
    # Check default parameters
    defaults = model._params
    assert defaults["criterion"] == "gini"
    assert defaults["max_depth"] is None
    assert defaults["random_state"] == 42


def test_decision_tree_classification_set_params():
    """Test that setParams correctly updates model parameters."""
    model = DecisionTreeModel(task_type=TaskType.CLASSIFICATION)
    model.setParams(max_depth=5, min_samples_split=10, criterion="entropy")
    
    params = model._params  # Should reflect updated values
    assert params["max_depth"] == 5
    assert params["min_samples_split"] == 10
    assert params["criterion"] == "entropy"
    
    # Confirm underlying model was re-initialized
    assert model._model.max_depth == 5
    assert model._model.criterion == "entropy"


def test_decision_tree_classification_fit_predict():
    """Test that classification model fits and predicts correctly."""
    X, y = make_classification(
        n_samples=200,
        n_features=4,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        random_state=42
    )

    model = DecisionTreeModel(task_type=TaskType.CLASSIFICATION)
    model.fit(X, y)
    
    predictions = model.predict(X)
    
    # Check output type and shape
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (200,)
    assert all(isinstance(p, (int, np.integer)) for p in predictions)  # Class labels are integers

    # Check accuracy: should be decent on training set (no overfitting penalty here)
    accuracy = np.mean(predictions == y)
    assert accuracy > 0.8  # Should easily achieve >80% on this simple dataset


def test_decision_tree_classification_param_grid():
    """Test that parameter grid is correctly structured for classification."""
    task_type=TaskType.CLASSIFICATION
    model = DecisionTreeModel(task_type=task_type)
    param_grid = model.getParamsGrids(task_type=task_type)
    
    assert isinstance(param_grid, dict)
    assert "criterion" in param_grid
    assert "max_depth" in param_grid
    assert "min_samples_split" in param_grid
    assert "min_samples_leaf" in param_grid
    assert "max_features" in param_grid
    
    # Check values are lists
    assert isinstance(param_grid["criterion"], list)
    assert "gini" in param_grid["criterion"]
    assert "entropy" in param_grid["criterion"]
    
    # max_depth includes None
    assert None in param_grid["max_depth"]


# =====================
# Test Case: Regression
# =====================

def test_decision_tree_regression_initialization():
    """Test that DecisionTreeModel initializes correctly for regression."""
    task_type=TaskType.REGRESSION
    model = DecisionTreeModel(task_type=task_type)
    assert model.task_type == TaskType.REGRESSION
    assert model._model.__class__.__name__ == "DecisionTreeRegressor"
    
    # Check default parameters
    defaults = model.getDefaultParams(task_type=task_type)
    assert defaults["criterion"] == "squared_error"
    assert defaults["max_depth"] is None
    assert defaults["random_state"] == 42


def test_decision_tree_regression_set_params():
    """Test that setParams correctly updates model parameters for regression."""
    model = DecisionTreeModel(task_type=TaskType.REGRESSION)
    model.setParams(max_depth=8, min_samples_leaf=5, criterion="absolute_error")
    
    params = model._params
    assert params["max_depth"] == 8
    assert params["min_samples_leaf"] == 5
    assert params["criterion"] == "absolute_error"
    
    # Confirm underlying model was updated
    assert model._model.max_depth == 8
    assert model._model.criterion == "absolute_error"


def test_decision_tree_classification_fit_predict():
    """Test that classification model fits and predicts correctly."""
    X, y = make_classification(
        n_samples=200,
        n_features=4,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        random_state=42
    )

    model = DecisionTreeModel(task_type=TaskType.CLASSIFICATION)
    model.fit(X, y)
    
    predictions = model.predict(X)
    unique_vals = np.unique(predictions)
    expected_classes = np.unique(y)

    # Check output type and shape
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (200,)
    
    # Convert to integers for accuracy calculation to avoid floating point issues
    predictions_int = predictions.astype(int)

    # Check that we have integer class labels (0 and 1 for binary classification)
    assert set(predictions_int).issubset({0, 1})  # Should only contain class labels 0 and 1
    
    # Check that we have the right class labels (even if they're floats)
    assert set(unique_vals) == set(expected_classes), f"Expected classes {expected_classes}, got {unique_vals}"

    # Check accuracy
    accuracy = np.mean(predictions == y)
    assert accuracy > 0.8


def test_decision_tree_regression_param_grid():
    """Test that parameter grid is correctly structured for regression."""
    task_type=TaskType.REGRESSION
    model = DecisionTreeModel(task_type=task_type)
    param_grid = model.getParamsGrids(task_type=task_type)
    
    assert isinstance(param_grid, dict)
    assert "criterion" in param_grid
    assert "max_depth" in param_grid
    assert "min_samples_split" in param_grid
    assert "min_samples_leaf" in param_grid
    assert "max_features" in param_grid
    
    # Check values are lists
    assert isinstance(param_grid["criterion"], list)
    assert "squared_error" in param_grid["criterion"]
    assert "absolute_error" in param_grid["criterion"]
    assert "friedman_mse" in param_grid["criterion"]
    
    # max_depth includes None
    assert None in param_grid["max_depth"]


# =====================
# Edge Case: Invalid Task Type
# =====================

def test_decision_tree_invalid_task_type():
    """Test that invalid task type raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported task type:.*"):
        DecisionTreeModel(task_type="invalid_task")  # type: ignore


# =====================
# Run with: pytest test_decision_tree.py -v
# =====================

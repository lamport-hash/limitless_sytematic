import pytest
import numpy as np

from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

import torch
import torch.nn as nn

from fitting.models.mlp_model import MLPModel, MLP
from fitting.fitting_core import TaskType


from numpy.typing import NDArray
from sklearn.datasets import make_classification, make_regression


# =====================
# Test Case: Classification
# =====================

def test_mlp_classification_initialization():
    """Test that MLPModel initializes correctly for classification."""
    model = MLPModel(
        task_type=TaskType.CLASSIFICATION,
        input_size=4,
        hidden_sizes=[64],
        output_size=2
    )
    
    assert model.task_type == TaskType.CLASSIFICATION
    assert model._model is not None
    assert isinstance(model._model, MLP)
    
    # Check default parameters
    defaults = model._params
    assert defaults["batch_size"] == 32
    assert defaults["epochs"] == 100
    assert defaults["learning_rate"] == 0.001
    assert defaults["optimizer"] == "adam"
    assert defaults["dropout"] == 0.2
    assert defaults["patience"] == 20


def test_mlp_classification_set_params():
    """Test that setParams correctly updates model parameters for classification."""
    model = MLPModel(
        task_type=TaskType.CLASSIFICATION,
        input_size=4,
        hidden_sizes=[64],
        output_size=2
    )
    
    model.setParams(
        batch_size=64,
        epochs=200,
        learning_rate=0.01,
        optimizer="sgd",
        dropout=0.3,
        patience=15,
        hidden_sizes=[128, 64]  # This triggers model rebuild
    )
    
    print(model._model)
    
    params = model._params  # Should reflect updated values
    assert params["batch_size"] == 64
    assert params["epochs"] == 200
    assert params["learning_rate"] == 0.01
    assert params["optimizer"] == "sgd"
    assert params["dropout"] == 0.3
    assert params["patience"] == 15
    
    # Confirm underlying model was rebuilt with new hidden_sizes and dropout
    assert len(model._model.network) > 2  # Should be deeper now (linear + relu + dropout)
    assert model._model.network[0].in_features == 4
    assert model._model.network[0].out_features == 128
    assert model._model.network[3].out_features == 64
    assert model._model.network[6].out_features == 2
    assert isinstance(model._model.network[1], nn.ReLU)
    
    # Confirm optimizer was reset (we can't inspect its lr directly without training, but model rebuilds)
    # We check that the parameters are still attached — if rebuild failed, the model would be stale
    assert len(list(model._model.parameters())) > 0


def test_mlp_classification_fit_predict():
    """Test that classification model fits and predicts correctly."""
    X, y = make_classification(
        n_samples=200,
        n_features=4,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        random_state=42
    )

    model = MLPModel(
        task_type=TaskType.CLASSIFICATION,
        input_size=4,
        hidden_sizes=[32],
        output_size=2,
        epochs=50,  # Reduce for fast test
        learning_rate=0.01,
        batch_size=32
    )
    
    model.fit(X, y)
    
    predictions = model.predict(X)
    
    # Check output type and shape
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (200,)
    
    # Predictions should be discrete class labels
    unique_preds = np.unique(predictions)
    expected_classes = np.unique(y)
    
    # Ensure predicted values are discrete and match class labels
    assert set(unique_preds).issubset(set(expected_classes))
    
    # Check accuracy — should be decent even with small model
    accuracy = np.mean(predictions == y)
    assert accuracy > 0.75  # MLP with small hidden layers can still overfit this dataset


def test_mlp_classification_predict_proba():
    """Test that predict_proba works for classification and raises error for regression."""
    X, y = make_classification(
        n_samples=200,
        n_features=4,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        random_state=42
    )

    model = MLPModel(
        task_type=TaskType.CLASSIFICATION,
        input_size=4,
        hidden_sizes=[32],
        output_size=2,
        epochs=50,  # Fast training
        learning_rate=0.01
    )
    
    model.fit(X, y)
    
    proba = model.predict_proba(X)
    
    # Check shape: (n_samples, n_classes)
    assert proba.shape == (200, 2)
    
    # Check probabilities sum to 1 per sample
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)
    
    # Ensure probabilities are in [0,1]
    assert np.all((proba >= 0) & (proba <= 1))
    
    # Test that predict_proba raises error for regression
    X_reg, y_reg = make_regression(n_samples=200, n_features=4, noise=0.1, random_state=42)
    
    model_reg = MLPModel(
        task_type=TaskType.REGRESSION,
        input_size=4,
        hidden_sizes=[32],
        output_size=1,
        epochs=50,
        learning_rate=0.01
    )
    
    model_reg.fit(X_reg, y_reg)
    
    with pytest.raises(ValueError, match="predict_proba is only available for classification tasks"):
        model_reg.predict_proba(X_reg)


def test_mlp_classification_param_grid():
    """Test that parameter grid is correctly structured for classification."""
    task_type = TaskType.CLASSIFICATION
    model = MLPModel(
        task_type=task_type,
        input_size=4,
        hidden_sizes=[32],
        output_size=2
    )
    
    param_grid = model.getParamsGrids(task_type=task_type)
    
    assert isinstance(param_grid, dict)
    assert "hidden_sizes" in param_grid
    assert "learning_rate" in param_grid
    assert "batch_size" in param_grid
    assert "optimizer" in param_grid
    assert "dropout" in param_grid
    assert "epochs" in param_grid
    
    # Check values are lists
    assert isinstance(param_grid["hidden_sizes"], list)
    assert [64] in param_grid["hidden_sizes"]
    assert [128, 64] in param_grid["hidden_sizes"]

    assert isinstance(param_grid["learning_rate"], list)
    assert 0.001 in param_grid["learning_rate"]
    assert 0.01 in param_grid["learning_rate"]
    assert 0.0001 in param_grid["learning_rate"]

    assert isinstance(param_grid["batch_size"], list)
    assert 16 in param_grid["batch_size"]
    assert 32 in param_grid["batch_size"]
    assert 64 in param_grid["batch_size"]

    assert isinstance(param_grid["optimizer"], list)
    assert "adam" in param_grid["optimizer"]
    assert "sgd" in param_grid["optimizer"]

    assert isinstance(param_grid["dropout"], list)
    assert 0.0 in param_grid["dropout"]
    assert 0.2 in param_grid["dropout"]
    assert 0.5 in param_grid["dropout"]

    assert isinstance(param_grid["epochs"], list)
    assert 50 in param_grid["epochs"]
    assert 100 in param_grid["epochs"]
    assert 200 in param_grid["epochs"]


# =====================
# Test Case: Regression
# =====================

def test_mlp_regression_initialization():
    """Test that MLPModel initializes correctly for regression."""
    model = MLPModel(
        task_type=TaskType.REGRESSION,
        input_size=4,
        hidden_sizes=[64],
        output_size=1
    )
    
    assert model.task_type == TaskType.REGRESSION
    assert model._model is not None
    assert isinstance(model._model, MLP)
    
    # Check default parameters
    defaults = model._params
    assert defaults["batch_size"] == 32
    assert defaults["epochs"] == 100
    assert defaults["learning_rate"] == 0.001
    assert defaults["optimizer"] == "adam"
    assert defaults["dropout"] == 0.2
    assert defaults["patience"] == 20


def test_mlp_regression_set_params():
    """Test that setParams correctly updates model parameters for regression."""
    model = MLPModel(
        task_type=TaskType.REGRESSION,
        input_size=4,
        hidden_sizes=[64],
        output_size=1
    )
    
    model.setParams(
        batch_size=64,
        epochs=200,
        learning_rate=0.01,
        optimizer="sgd",
        dropout=0.4,
        patience=20,
        hidden_sizes=[128, 64]
    )
    
    params = model._params
    assert params["batch_size"] == 64
    assert params["epochs"] == 200
    assert params["learning_rate"] == 0.01
    assert params["optimizer"] == "sgd"
    assert params["dropout"] == 0.4
    assert params["patience"] == 20
    
    print(model._model)

    # Confirm model was rebuilt with new hidden_sizes and dropout
    assert len(model._model.network) > 3
    assert model._model.network[0].out_features == 128
    assert model._model.network[3].out_features == 64
    assert model._model.network[6].out_features == 1
    assert isinstance(model._model.network[-1], nn.Linear)  # Output layer unchanged


def test_mlp_regression_fit_predict():
    """Test that regression model fits and predicts correctly."""
    X, y = make_regression(
        n_samples=200,
        n_features=4,
        noise=0.1,
        random_state=42
    )

    model = MLPModel(
        task_type=TaskType.REGRESSION,
        input_size=4,
        hidden_sizes=[32],
        output_size=1,
        epochs=50,  # Reduce for fast test
        learning_rate=0.01,
        batch_size=32
    )
    
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
    assert r2 > 0.85  # MLP should fit this simple regression well


def test_mlp_regression_param_grid():
    """Test that parameter grid is correctly structured for regression."""
    task_type = TaskType.REGRESSION
    model = MLPModel(
        task_type=task_type,
        input_size=4,
        hidden_sizes=[32],
        output_size=1
    )
    
    param_grid = model.getParamsGrids(task_type=task_type)
    
    assert isinstance(param_grid, dict)
    assert "hidden_sizes" in param_grid
    assert "learning_rate" in param_grid
    assert "batch_size" in param_grid
    assert "optimizer" in param_grid
    assert "dropout" in param_grid
    assert "epochs" in param_grid
    
    # Check values are lists (identical to classification)
    assert isinstance(param_grid["hidden_sizes"], list)
    assert [64] in param_grid["hidden_sizes"]
    assert [128, 64] in param_grid["hidden_sizes"]

    assert isinstance(param_grid["learning_rate"], list)
    assert 0.001 in param_grid["learning_rate"]
    assert 0.01 in param_grid["learning_rate"]
    assert 0.0001 in param_grid["learning_rate"]

    assert isinstance(param_grid["batch_size"], list)
    assert 16 in param_grid["batch_size"]
    assert 32 in param_grid["batch_size"]
    assert 64 in param_grid["batch_size"]

    assert isinstance(param_grid["optimizer"], list)
    assert "adam" in param_grid["optimizer"]
    assert "sgd" in param_grid["optimizer"]

    assert isinstance(param_grid["dropout"], list)
    assert 0.0 in param_grid["dropout"]
    assert 0.2 in param_grid["dropout"]
    assert 0.5 in param_grid["dropout"]

    assert isinstance(param_grid["epochs"], list)
    assert 50 in param_grid["epochs"]
    assert 100 in param_grid["epochs"]
    assert 200 in param_grid["epochs"]




def test_mlp_predict_with_wrong_input_shape():
    """Test that predict handles incorrect feature dimensions gracefully (no crash)."""
    model = MLPModel(
        task_type=TaskType.CLASSIFICATION,
        input_size=4,
        hidden_sizes=[32],
        output_size=2
    )
    
    X, y = make_classification(n_samples=10, n_features=4, random_state=42)
    model.fit(X, y)

    # Invalid shape: 3 features instead of 4
    X_bad = np.random.rand(10, 3)
    
    with pytest.raises(RuntimeError):  # PyTorch will raise a shape error if input is wrong
        model.predict(X_bad)  # Should fail during forward pass — acceptable behavior


# =====================
# Test: Device Support
# =====================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_mlp_model_cuda_device():
    """Test that MLPModel can use CUDA if available."""
    model = MLPModel(
        task_type=TaskType.CLASSIFICATION,
        input_size=4,
        hidden_sizes=[32],
        output_size=2,
        device="cuda"
    )
    
    assert model.device.type == "cuda"
    assert next(model._model.parameters()).is_cuda  # Confirm param is on GPU

# Run the tests with: pytest -v test_models.py
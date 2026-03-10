# conftest.py
import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

@pytest.fixture(scope="session")
def classification_data():
    """Generate classification dataset for testing."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42
    )
    return X.astype(np.float32), y

@pytest.fixture(scope="session")
def regression_data():
    """Generate regression dataset for testing."""
    X, y = make_regression(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        noise=0.1,
        random_state=42
    )
    return X.astype(np.float32), y.astype(np.float32)

@pytest.fixture
def train_test_split_classification(classification_data):
    """Split classification data into train and test sets."""
    X, y = classification_data
    return train_test_split(X, y, test_size=0.2, random_state=42)

@pytest.fixture
def train_test_split_regression(regression_data):
    """Split regression data into train and test sets."""
    X, y = regression_data
    return train_test_split(X, y, test_size=0.2, random_state=42)
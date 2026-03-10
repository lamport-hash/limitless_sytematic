
from enum import Enum
from typing import Any, Dict, List
import numpy as np

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ..fitting_core import BaseModel, TaskType, Targets, Predictions, Features

class DecisionTreeModel(BaseModel):
    """Concrete implementation of ModelProtocol for sklearn's Decision Tree."""

    def __init__(self, task_type: TaskType, **kwargs):
        # Call parent constructor to initialize task_type and params
        super().__init__(task_type, **kwargs)
        self._model = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the underlying sklearn Decision Tree model based on task type."""
        if self.task_type == TaskType.CLASSIFICATION:
            self._model = DecisionTreeClassifier(**self._params)
        elif self.task_type == TaskType.REGRESSION:
            self._model = DecisionTreeRegressor(**self._params)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

    @classmethod
    def getDefaultParams(cls, task_type: TaskType) -> Dict[str, Any]:
        """Return default parameters for Decision Tree."""
        base_params = {
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': None,
            'random_state': 42
        }

        if task_type == TaskType.CLASSIFICATION:
            base_params.update({
                'criterion': 'gini',
            })
        elif task_type == TaskType.REGRESSION:
            base_params.update({
                'criterion': 'squared_error',
            })
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

        return base_params

    @classmethod
    def getParamsGrids(cls, task_type: TaskType) -> Dict[str, List[Any]]:
        """Return a parameter grid for hyperparameter tuning (e.g., GridSearchCV)."""
        if task_type == TaskType.CLASSIFICATION:
            return {
                'criterion': ['gini', 'entropy'],
                'max_depth': [3, 5, 7, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
        elif task_type == TaskType.REGRESSION:
            return {
                'criterion': ['squared_error', 'absolute_error', 'friedman_mse'],
                'max_depth': [3, 5, 7, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

    def setParams(self, **kwargs) -> None:
        """Set model parameters and re-initialize the underlying model."""
        super().setParams(**kwargs)  # Updates self._params via parent
        if self._model is not None:
            self._initialize_model()  # Rebuild model with new params

    def fit(self, X: Features, y: Targets) -> None:
        """Train the model on features X and targets y."""
        if hasattr(y, 'shape') and len(y.shape) > 1:
            y = y.ravel()
        self._model.fit(X, y)

    def predict(self, X: Features) -> Predictions:
        """Generate predictions for features X."""
        predictions = self._model.predict(X)
        return np.asarray(predictions, dtype=np.float64)  # Ensure float64 consistency

    def predict_proba(self, X: Features) -> np.ndarray:
        """Generate probability predictions for classification tasks."""
        if self.task_type != TaskType.CLASSIFICATION:
            raise ValueError("predict_proba is only available for classification tasks")
        return self._model.predict_proba(X)
    
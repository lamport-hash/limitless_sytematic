

from typing import Any, Dict, List
import numpy as np

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


from ..fitting_core import BaseModel, TaskType, Targets, Predictions, Features


class RandomForestModel(BaseModel):
    """Concrete implementation of ModelProtocol for sklearn's Random Forest."""

    def __init__(self, task_type: TaskType, **kwargs):
        # Call parent constructor to initialize task_type and _params
        super().__init__(task_type, **kwargs)
        self._model = None  # Explicitly reset (since dataclass fields are initialized before __init__)
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the underlying sklearn Random Forest model based on task type."""
        if self.task_type == TaskType.CLASSIFICATION:
            self._model = RandomForestClassifier( **self._params)
        elif self.task_type == TaskType.REGRESSION:
            self._model = RandomForestRegressor( **self._params)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

    @classmethod
    def getDefaultParams(cls, task_type: TaskType) -> Dict[str, Any]:
        """Get default parameters for Random Forest."""
        return {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'random_state': 42
        }

    @classmethod
    def getParamsGrids(cls, task_type: TaskType) -> Dict[str, List[Any]]:
        """Get parameter grids for hyperparameter tuning (e.g., GridSearchCV)."""
        return {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }

    def setParams(self, **kwargs) -> None:
        """Set model parameters and update underlying sklearn model."""
        super().setParams(**kwargs)  # Updates self._params via BaseModel
        if self._model is not None:
            self._initialize_model()  # Rebuild model with new params

    def fit(self, X: Features, y: Targets) -> None:
        """Train the Random Forest model."""
        if hasattr(y, 'shape') and len(y.shape) > 1:
            y = y.ravel()
        self._model.fit(X, y)

    def predict(self, X: Features) -> Predictions:
        """Generate predictions using the trained Random Forest."""
        predictions = self._model.predict(X)
        return np.asarray(predictions, dtype=np.float64)

    def predict_proba(self, X: Features) -> np.ndarray:
        """Generate probability predictions for classification tasks."""
        if self.task_type != TaskType.CLASSIFICATION:
            raise ValueError("predict_proba is only available for classification tasks")
        return self._model.predict_proba(X)

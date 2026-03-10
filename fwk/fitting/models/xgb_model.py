
from typing import Any, Dict
import numpy as np

import xgboost as xgb

from ..fitting_core import BaseModel, TaskType, Targets, Predictions, Features


class XGBoostModel(BaseModel):
    """Concrete implementation of BaseModel for XGBoost."""
    
    def __init__(self, task_type: TaskType, **kwargs):
        # Call parent constructor
        super().__init__(task_type, **kwargs)
        self._model = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the underlying XGBoost model."""
        if self.task_type == TaskType.CLASSIFICATION:
            self._model = xgb.XGBClassifier()
        elif self.task_type == TaskType.REGRESSION:
            self._model = xgb.XGBRegressor()
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
        
        # Apply current parameters
        self._model.set_params(**self._params)

    @classmethod
    def getDefaultParams(cls, task_type: TaskType) -> Dict[str, Any]:
        base_params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42
        }
        
        if task_type == TaskType.CLASSIFICATION:
            base_params.update({
                "objective": "binary:logistic",
                "eval_metric": "logloss",
            })
        elif task_type == TaskType.REGRESSION:
            base_params.update({
                "objective": "reg:squarederror",
                "eval_metric": "rmse"
            })
            
        return base_params

    @classmethod
    def getParamsGrids(cls, task_type: TaskType) -> Dict[str, Any]:
        return {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'min_child_weight': [1, 3, 5]
        }

    def setParams(self, **kwargs) -> None:
        """Set model parameters and update underlying XGBoost model."""
        super().setParams(**kwargs)
        if self._model is not None:
            self._model.set_params(**self._params)

    def fit(self, X: Features, y: Targets) -> None:
        if hasattr(y, 'shape') and len(y.shape) > 1:
            y = y.ravel()
        self._model.fit(X, y)

    def predict(self, X: Features) -> Predictions:
        predictions = self._model.predict(X)
        return predictions.astype(np.float64)
    
    def predict_proba(self, X: Features) -> np.ndarray:
        """Generate probability predictions for classification tasks."""
        if self.task_type != TaskType.CLASSIFICATION:
            raise ValueError("predict_proba is only available for classification tasks")
        return self._model.predict_proba(X)
   
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Protocol, TypeAlias, Dict
import numpy as np
from numpy.typing import NDArray
import joblib
import datetime
import os

# Type aliases for clarity
Features: TypeAlias = NDArray[np.floating]
Targets: TypeAlias = NDArray[np.floating]
Predictions: TypeAlias = NDArray[np.floating]

class RetrainMode(Enum):
    """Retraining strategy for test period."""
    EXPANDING = "expanding"  # Use all historical data
    SLIDING = "sliding"      # Use only sliding window

class TrainingSplitType(Enum):
    TRAIN_VAL_TEST = "train_val_test"
    PERIODIC_RETRAIN = "periodic_retrain"

class TaskType(Enum):
    """Type of ML task."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


# Abstract base class for concrete implementations
class BaseModel(ABC):
    """Abstract base class providing common functionality."""

    def __init__(self, task_type: TaskType, **kwargs):
        self.task_type = task_type
        # Initialize with default params for the specific task type
        self._params = self.__class__.getDefaultParams(task_type)
        self._params.update(kwargs)
        self._model = None
    
    @classmethod
    @abstractmethod
    def getDefaultParams(cls, task_type: TaskType) -> Dict[str, Any]:
        """Get default parameters for this model type and task type."""
        pass
    
    @classmethod
    @abstractmethod
    def getParamsGrids(cls, task_type: TaskType) -> Dict[str, Any]:
        """Get parameter grids for hyperparameter tuning."""
        pass
    
    def getParams(self) -> Dict[str, Any]:
        """Get current model parameters, merging stored params with actual model params."""
        # Start with our stored parameters
        params = self._params.copy()
        
        return params
    
    def setParams(self, **kwargs) -> None:
        self._params.update(kwargs)


    @abstractmethod
    def fit(self, X: Features, y: Targets) -> None:
        """Train the model on features X and targets y."""
    
    @abstractmethod
    def predict(self, X: Features) -> Predictions:
        """Generate predictions for features X."""
    
    @abstractmethod
    def predict_proba(self, X: Features) -> np.ndarray:
        """Generate predictions for features X."""



    # todo save golden batch => check if inference is the same
    def save_to_file(self, folder:str, model_id:str, version:int = 0) -> None:
        payload = {
            "version": version,
            "task_type": self.task_type,
            "params":self._params,
            "class": self.__class__,
            "model": self._model,
            "class_name": self.__class__.__name__,
            "created_at": datetime.datetime.now().isoformat(),
        }

        # Build full path with proper separator
        filename = os.path.join(folder, f"basemodel_{model_id}_{version}.joblib")

        # Ensure directory exists (creates if needed)
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        joblib.dump(payload, filename )

    @classmethod
    def load_from_file(cls, filename:str)-> BaseModel:

        payload = joblib.load(filename)
        
        # Retrieve the concrete class that saved this model
        model_class = payload["class"]
        
        if not issubclass(model_class, BaseModel):
            raise ValueError(f"Saved model class {model_class} is not a subclass of BaseModel")

        # Reconstruct the instance using the correct class
        model_instance = model_class(
            task_type=payload["task_type"],
            **payload["params"]
        )
        
        # Restore the trained model object
        model_instance._model = payload["model"]
        
        print(f"✅ Model loaded from: {filename}")
        print(f"   Class: {model_class.__name__}, Version: {payload['version']}, "
              f"Created: {payload['created_at']}")

        return model_instance
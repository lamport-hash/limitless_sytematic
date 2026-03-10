from .xgb_model import XGBoostModel
from .dt_model import DecisionTreeModel
from .rf_model import RandomForestModel
from .mlp_model import MLPModel

from ..fitting_core import BaseModel,TaskType
from enum import Enum
from typing import Dict, Type

class ModelType(Enum):
    XGB = ("xgb", XGBoostModel, True, True)      # xgb xgboost
    RF_SK = ("rf_sk", RandomForestModel, True, True)  # random forest sklearn
    DT_SK = ("dt_sk", DecisionTreeModel, True, True)  # dt sklearn
    MLP_TORCH = ("mlp_torch", MLPModel, True, True)   # mlp torch both
    #BNN_GAUTO = ("bnn_gauto", BNNAutoModel, True, True)  # google bnn auto

    def __init__(self, code: str, model_class: Type[BaseModel], supports_classification: bool, supports_regression: bool):
        self.code = code
        self.model_class = model_class
        self.supports_classification = supports_classification
        self.supports_regression = supports_regression

    @classmethod
    def from_code(cls, code: str) -> "ModelType":
        """Look up ModelType by its string code (e.g., 'xgb')"""
        for member in cls:
            if member.code == code:
                return member
        raise ValueError(f"No ModelType with code '{code}'")

    @classmethod
    def get_model_class(cls, model_type: "ModelType") -> Type[BaseModel]:
        """Get the associated class for a ModelType."""
        if not isinstance(model_type, cls):
            raise ValueError(f"Expected {cls.__name__}, got {type(model_type)}")
        return model_type.model_class

    @classmethod
    def list_supported(cls) -> Dict[str, Type[BaseModel]]:
        """Return all supported model types and their classes."""
        return {mt.code: mt.model_class for mt in cls}

    @classmethod
    def get_supported_by_task(cls, task_type: TaskType) -> list["ModelType"]:
        """Return all model types that support the given task."""
        if task_type == TaskType.CLASSIFICATION:
            return [mt for mt in cls if mt.supports_classification]
        elif task_type == TaskType.REGRESSION:
            return [mt for mt in cls if mt.supports_regression]
        else:
            raise ValueError(f"Unknown task type: {task_type}")
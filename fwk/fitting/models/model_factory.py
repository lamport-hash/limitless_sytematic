
from typing import Any, Dict
import os
from typing import Any, Protocol, TypeAlias, Literal, Dict, List,Type, Optional
import random


from .xgb_model import XGBoostModel
from .dt_model import DecisionTreeModel
from .rf_model import RandomForestModel
from .mlp_model import MLPModel
from .model_register import ModelType

from ..fitting_core import BaseModel, TaskType, Targets, Predictions, Features



class ModelFactory:
    """Factory class for creating model instances with task-specific configurations."""

    @classmethod
    def create_model(cls, 
                    model_type: ModelType,
                    task_type: TaskType,
                    **kwargs) -> BaseModel:
        """Create a model instance of the specified type and task."""
        if not cls._is_task_supported(model_type, task_type):
            raise ValueError(
                f"Model type {model_type.name} does not support task {task_type.value}"
            )
        model_class = ModelType.get_model_class(model_type)
        return model_class(task_type, **kwargs)

    @classmethod
    def get_default_params(cls, 
                          model_type: ModelType,
                          task_type: TaskType) -> Dict[str, Any]:
        """Get default parameters for a model type and task without creating an instance."""
        if not cls._is_task_supported(model_type, task_type):
            raise ValueError(
                f"Model type {model_type.name} does not support task {task_type.value}"
            )
        model_class = ModelType.get_model_class(model_type)
        return model_class.getDefaultParams(task_type)

    @classmethod
    def get_params_grids(cls, 
                        model_type: ModelType,
                        task_type: TaskType) -> Dict[str, Any]:
        """Get parameter grids for a model type and task without creating an instance."""
        if not cls._is_task_supported(model_type, task_type):
            raise ValueError(
                f"Model type {model_type.name} does not support task {task_type.value}"
            )
        model_class = ModelType.get_model_class(model_type)
        return model_class.getParamsGrids(task_type)


    @classmethod
    def get_random_param(cls,
                        model_type: ModelType,
                        task_type: TaskType,
                        n_samples: int = 1) -> Dict[str, Any] | list[Dict[str, Any]]:
        """
        Randomly sample one (or more) parameter combinations from the model's parameter grid.

        Args:
            model_type: ModelType enum member
            task_type: TaskType (regression/classification)
            n_samples: number of random parameter sets to sample. If > 1, returns list.

        Returns:
            Single dict (if n_samples=1) or list of dicts with randomly sampled parameters.
        """
        if not cls._is_task_supported(model_type, task_type):
            raise ValueError(
                f"Model type {model_type.name} does not support task {task_type.value}"
            )

        model_class = ModelType.get_model_class(model_type)
        param_grid = model_class.getParamsGrids(task_type)

        if not isinstance(param_grid, dict):
            raise TypeError(f"getParamsGrids must return a dict, got {type(param_grid)}")

        if not param_grid:
            # Return empty config if grid is empty
            return {} if n_samples == 1 else [{}]

        # Generate one or more random samples
        def sample_one():
            return {
                param: random.choice(values)
                for param, values in param_grid.items()
                if isinstance(values, list) and len(values) > 0
            }

        if n_samples == 1:
            return sample_one()
        else:
            return [sample_one() for _ in range(n_samples)]

    @classmethod
    def _is_task_supported(cls, model_type: ModelType, task_type: TaskType) -> bool:
        """Helper to check if model supports the given task."""
        if not isinstance(model_type, ModelType):
            raise ValueError(f"Expected ModelType, got {type(model_type)}")
        if task_type == TaskType.CLASSIFICATION:
            return model_type.supports_classification
        elif task_type == TaskType.REGRESSION:
            return model_type.supports_regression
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
    @classmethod
    def load_from_file(cls, filepath:str) -> Optional[BaseModel]:
        filename_with_ext = os.path.basename(filepath)
        print(f"Filename with extension: {filename_with_ext}")

        # Split filename and extension
        filename, extension = os.path.splitext(filename_with_ext)

        if filename[:10] == 'basemodel_':
            return BaseModel.load_from_file(filepath)
        elif filename[:10] == 'mplmodel_':
            return MLPModel.load_from_file(filepath)
        else:
            return None


import pytest
import random
from typing import Dict, Any

from fitting.fitting_core import TaskType
from fitting.models.model_register import ModelType
from fitting.models.model_factory import ModelFactory



# Fixture to generate random parameters for any model
@pytest.fixture
def random_params():
    def _random_params(model_type: ModelType, task_type: TaskType) -> Dict[str, Any]:
        default_params = ModelFactory.get_default_params(model_type, task_type)
        random_params_dict = {}
        for key, value in default_params.items():
            if isinstance(value, int):
                random_params_dict[key] = random.randint(max(1, value - 5), value + 5)
            elif isinstance(value, float):
                random_params_dict[key] = random.uniform(value * 0.5, value * 1.5)
            elif isinstance(value, str):
                random_params_dict[key] = value + "_random"
            elif isinstance(value, bool):
                random_params_dict[key] = not value
            else:
                # Fallback: use original if type unknown
                random_params_dict[key] = value
        return random_params_dict
    return _random_params


# Fixture to get grid params (for each model type, we expect a dict with lists)
@pytest.fixture
def grid_params():
    def _grid_params(model_type: ModelType, task_type: TaskType) -> Dict[str, Any]:
        grids = ModelFactory.get_params_grids(model_type, task_type)

        # Ensure it's a dict with list values (typical for grid search)
        assert isinstance(grids, dict), f"Grid params must be a dict, got {type(grids)}"
        for key, val in grids.items():
            assert isinstance(val, list), f"Grid param {key} must be a list, got {type(val)}"
            assert len(val) > 0, f"Grid param {key} must have at least one value"
        return grids
    return _grid_params


# ----------------------------
# Test: Model Creation with Default Params
# ----------------------------
@pytest.mark.parametrize("model_type", list(ModelType))  # ✅ Fixed: Use ModelType members directly
@pytest.mark.parametrize("task_type", TaskType)
def test_create_model_with_default_params(model_type, task_type):
    model = ModelFactory.create_model(model_type, task_type)
    assert model is not None
    # Ensure it's an instance of the correct class
    expected_class = model_type.model_class  # ✅ Use ModelType.model_class directly!
    assert isinstance(model, expected_class)


# ----------------------------
# Test: Model Creation with Random Params
# ----------------------------
@pytest.mark.parametrize("model_type", list(ModelType))  # ✅ Fixed: Use ModelType members
@pytest.mark.parametrize("task_type", TaskType)
def test_create_model_with_random_params(model_type, task_type, random_params):
    # Generate random params based on default
    rand_params = random_params(model_type, task_type)
    model = ModelFactory.create_model(model_type, task_type, **rand_params)
    assert model is not None
    # Ensure we can access the params (optional: check if they were passed correctly)
    assert model.__class__ == model_type.model_class  # ✅ Fixed: Use model_type.model_class


# ----------------------------
# Test: Model Creation with Grid Params (use first value from each list)
# ----------------------------
@pytest.mark.parametrize("model_type", list(ModelType))  # ✅ Fixed: Use ModelType members
@pytest.mark.parametrize("task_type", TaskType)
def test_create_model_with_grid_params(model_type, task_type, grid_params):
    grids = grid_params(model_type, task_type)
    # Extract first value from each list to use as hyperparameter
    grid_params_sample = {key: values[0] for key, values in grids.items()}
    
    model = ModelFactory.create_model(model_type, task_type, **grid_params_sample)
    assert model is not None
    assert isinstance(model, model_type.model_class)  # ✅ Fixed


# ----------------------------
# Test: Default Params Retrieved Correctly (Structure + Type)
# ----------------------------
@pytest.mark.parametrize("model_type", list(ModelType))  # ✅ Fixed
@pytest.mark.parametrize("task_type", TaskType)
def test_get_default_params_structure(model_type, task_type):
    default_params = ModelFactory.get_default_params(model_type, task_type)
    assert isinstance(default_params, dict), "Default params must be a dict"
    assert len(default_params) > 0, f"No default parameters found for {model_type} with {task_type}"


# ----------------------------
# Test: Grid Params Retrieved Correctly (Structure + Type)
# ----------------------------
@pytest.mark.parametrize("task_type", TaskType)
def test_get_params_grids_structure(task_type):
    model_type_list = ModelType.get_supported_by_task(task_type)  # ✅ This is correct
    for model_type in model_type_list:  # ✅ Loop over each supported model type!
        grid_params = ModelFactory.get_params_grids(model_type, task_type)
        assert isinstance(grid_params, dict), "Grid params must be a dict"
        assert len(grid_params) > 0, f"No parameter grids found for {model_type} with {task_type}"
        # Ensure all values are lists
        for key, val in grid_params.items():
            assert isinstance(val, list), f"Grid value for {key} is not a list"
            assert len(val) > 0, f"Grid for {key} is empty"

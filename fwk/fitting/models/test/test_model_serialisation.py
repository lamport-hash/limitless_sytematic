import pytest


from fitting.models.model_factory import ModelFactory
from fitting.models.model_register import ModelType
from fitting.models.mlp_model import MLPModel

from fitting.fitting_core import BaseModel, TaskType, Targets, Predictions, Features


@pytest.mark.parametrize("model_type", list(ModelType))
@pytest.mark.parametrize("task_type", TaskType)
def test_save_and_load_roundtrip(model_type, task_type, tmp_path):
    """
    Test that any model created by ModelFactory can be saved to disk and reloaded
    with identical parameters and class.
    
    Uses tmp_path for auto-cleanup. No files left behind.
    """

    # Arrange: Create model via factory
    model = ModelFactory.create_model(model_type, task_type=task_type)
    
    # Validate creation
    assert model is not None, f"Failed to create model for {model_type} with task type {task_type}"
    assert isinstance(model, BaseModel), f"Model is not a subclass of BaseModel: {type(model)}"

    # Arrange: Define filename using your standard format
    model_id = "test_model"  # ← MATCHES your hardcoded filename!
    version = 0
    filepath = tmp_path / f"basemodel_{model_id}_{version}.joblib"
    filepath_mlp = tmp_path / f"mlpmodel_{model_id}_{version}.pt"

    # Act: Save to file
    model.save_to_file(tmp_path, model_id, version)  # ← This creates "model_test_model_0.joblib"

    print(f"Saving model to: {filepath} or {filepath_mlp}")

    # Verify file was created
    file_exist = (filepath.exists() or filepath_mlp.exists())

    assert file_exist, f"Model file not saved: {filepath} neither {filepath_mlp}"

    # Act: Load model from file
    if filepath_mlp.exists() :
        loaded_model = MLPModel.load_from_file(str(filepath_mlp))    
    else :
        loaded_model = BaseModel.load_from_file(str(filepath))

    # Assert: Model class matches
    assert type(model) == type(loaded_model), f"Class mismatch: {type(model)} vs {type(loaded_model)}"
    
    # Assert: Parameters match exactly
    assert model.getParams() == loaded_model.getParams(), "Loaded parameters do not match original"

    # Assert: Task type preserved
    assert model.task_type == loaded_model.task_type, "Task type not preserved"

    print(f"✅ Test passed: {model_type} ({task_type}) -> saved and loaded from {filepath}")
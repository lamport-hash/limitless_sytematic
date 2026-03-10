from abc import ABC
from enum import Enum
from typing import Any, Dict, Protocol, TypeAlias, Literal, List, Optional
import time
import numpy as np
from numpy.typing import NDArray

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add logging
import logging
import os
import datetime

# Configure logging (adjust level as needed: DEBUG, INFO, WARNING)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


from ..fitting_core import BaseModel, TaskType, Targets, Predictions, Features


class MLP(nn.Module):
    """Multi-Layer Perceptron PyTorch model."""
    
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int,
                 task_type: TaskType, dropout: float = 0.0):
        super(MLP, self).__init__()
        self.task_type = task_type

        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MLPModel(BaseModel):
    """Concrete implementation of ModelProtocol for PyTorch-based MLP."""
    
    def __init__(self, task_type: TaskType, **kwargs):
        # Initialize parent (ModelProtocol) — handles task_type and _params
        super().__init__(task_type, **kwargs)

        # NEVER hardcode these as instance attributes!
        # They are now purely derived from self._params — fully dynamic.
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        # Initialize model and training components
        self._model: Optional[MLP] = None
        self._optimizer: Optional[optim.Optimizer] = None
        self._criterion: Optional[nn.Module] = None
        self._scheduler: Optional[ReduceLROnPlateau] = None

        # Initialize model based on current params
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize MLP model, optimizer, and loss function based on current _params."""
        
        # ✅ ALL architecture params MUST come from self._params (never hard-coded!)
        input_size = self._params.get("input_size",10)
        hidden_sizes = self._params.get("hidden_sizes", [])
        output_size = self._params.get("output_size",1)
        dropout = self._params.get("dropout", 0.0)

        # Validate required params
        if input_size is None or output_size is None:
            raise ValueError("input_size and output_size must be provided in params.")

        if not isinstance(hidden_sizes, list):
            raise TypeError("hidden_sizes must be a list of integers.")

        # Log model architecture
        logger.info(f"Initializing MLP with: input_size={input_size}, hidden_sizes={hidden_sizes}, output_size={output_size}, dropout={dropout}")
        logger.info(f"Device: {self.device}")

        # Build model
        self._model = MLP(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            task_type=self.task_type,
            dropout=dropout
        ).to(self.device)

        # Initialize optimizer
        optimizer_name = self._params.get("optimizer", "adam").lower()
        lr = self._params.get("learning_rate", 0.001)
        
        if optimizer_name == "adam":
            self._optimizer = optim.Adam(self._model.parameters(), lr=lr)
        elif optimizer_name == "sgd":
            self._optimizer = optim.SGD(self._model.parameters(), lr=lr)
        else:
            logger.warning(f"Unknown optimizer '{optimizer_name}', defaulting to 'adam'")
            self._optimizer = optim.Adam(self._model.parameters(), lr=lr)

        logger.info(f"Optimizer: {optimizer_name}, Learning Rate: {lr}")

        # Initialize loss function
        if self.task_type == TaskType.CLASSIFICATION:
            self._criterion = nn.CrossEntropyLoss()
            logger.info("Using CrossEntropyLoss for classification")
        else:  # REGRESSION
            self._criterion = nn.MSELoss()
            logger.info("Using MSELoss for regression")

        # Initialize scheduler
        # todo: get patience/mode/factor from parameters
        patience = self._params.get('patience', 10)
        factor = self._params.get('scheduler_factor', 0.1)
        mode = self._params.get('scheduler_mode', 'min')
        
        self._scheduler = ReduceLROnPlateau(
            self._optimizer,
            mode=mode,
            factor=factor,
            patience=patience
        )
        logger.info(f"ReduceLROnPlateau scheduler initialized: mode={mode}, factor={factor}, patience={patience}")

    @classmethod
    def getDefaultParams(cls, task_type: TaskType) -> Dict[str, Any]:
        """Return default hyperparameters for MLP training."""
        return {
            'batch_size': 32,
            'epochs': 100,
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'dropout': 0.2,
            'weight_decay':1e-4,
            'patience': 20,  # early stopping
        }

    @classmethod
    def getParamsGrids(cls, task_type: TaskType) -> Dict[str, List[Any]]:
        """Return a parameter grid for hyperparameter tuning."""
        return {
            'hidden_sizes': [[64], [128], [64, 32], [128, 64]],
            'learning_rate': [0.001, 0.01, 0.0001],
            'batch_size': [16, 32, 64],
            'optimizer': ['adam', 'sgd'],
            'dropout': [0.0, 0.2, 0.5],
            'weight_decay':[1e-4,0],
            'epochs': [50, 100, 200]
        }

    def setParams(self, **kwargs) -> None:
        """Set model parameters and re-initialize model if architecture or training params change."""
        logger.debug(f"Updating model params: {kwargs}")
        super().setParams(**kwargs)  # Updates self._params via base class
        logger.info("Reinitializing model due to parameter update.")
        self._initialize_model()
        

    def fit(self, X: Features, y: Targets) -> None:
        if self._model is None:
            raise RuntimeError("Model not initialized. Call setParams() or __init__ first.")
        
        torch.manual_seed(self._params.get('random_state', 42))
        logger.info(f"Random seed set to: {self._params.get('random_state', 42)}")

        self._model.train()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        if self.task_type == TaskType.CLASSIFICATION:
            y_tensor = torch.LongTensor(y).to(self.device)
        else:
            y_tensor = torch.FloatTensor(y).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        
        val_fraction = self._params.get('validation_fraction', 0.1)
        batch_size = self._params.get('batch_size', 32)
        train_size = int((1 - val_fraction) * len(dataset))
        
        # ✅ Sequential split: first train_size samples = train, rest = val
        train_dataset = Subset(dataset, range(train_size))
        val_dataset = Subset(dataset, range(train_size, len(dataset)))

        logger.info(f"Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)} (validation fraction = {val_fraction})")
        logger.info(f"Training batch size: {batch_size}")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

        epochs = self._params.get('epochs', 50)
        patience = self._params.get('patience', 5)
        
        best_val_loss = float('inf')
        patience_counter = 0

        total_epochs = epochs
        start_time = time.time()
        epoch_times: List[float] = []  # Record each epoch's duration

        logger.info(f"Starting training for {epochs} epochs with batch_size={batch_size}, patience={patience}")

        for epoch in range(1, epochs + 1):  # Start from 1 for readability
            epoch_start = time.time()

            # Training loop
            self._model.train()
            total_train_loss = 0.0
            for batch_X, batch_y in train_loader:
                self._optimizer.zero_grad()
                outputs = self._model(batch_X)

                if self.task_type == TaskType.CLASSIFICATION:
                    loss = self._criterion(outputs, batch_y)
                else:
                    loss = self._criterion(outputs.squeeze(), batch_y)

                loss.backward()
                self._optimizer.step()
                total_train_loss += loss.item()

            # Validation loop
            self._model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self._model(batch_X)
                    if self.task_type == TaskType.CLASSIFICATION:
                        loss = self._criterion(outputs, batch_y)
                    else:
                        loss = self._criterion(outputs.squeeze(), batch_y)
                    total_val_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            val_loss = total_val_loss / len(val_loader)

            # Scheduler step (using validation loss)
            prev_lr = self._optimizer.param_groups[0]['lr']
            self._scheduler.step(val_loss)
            current_lr = self._optimizer.param_groups[0]['lr']

            if current_lr != prev_lr:
                logger.info(f"Learning rate reduced from {prev_lr:.2e} to {current_lr:.2e} at epoch {epoch}")

            # Track epoch time
            epoch_duration = time.time() - epoch_start
            epoch_times.append(epoch_duration)

            # ✅ ETA Estimation
            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            remaining_epochs = total_epochs - epoch
            estimated_total_time = avg_epoch_time * total_epochs
            remaining_time = avg_epoch_time * remaining_epochs

            # ✅ Log progress with timing and ETA
            logger.info(
                f"Epoch [{epoch:3d}/{total_epochs}] "
                f"| Train Loss: {avg_train_loss:.6f} "
                f"| Val Loss: {val_loss:.6f} "
                f"| LR: {current_lr:.2e} "
                f"| Time: {epoch_duration:.2f}s "
                f"| ETA: {remaining_time:.1f}s"
            )

            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                logger.debug(f"New best validation loss: {best_val_loss:.6f}")
            else:
                patience_counter += 1
                logger.debug(f"Patience counter: {patience_counter}/{patience}")

            if patience_counter >= patience:
                logger.warning(f"Early stopping triggered at epoch {epoch}")
                break

        # Final summary
        total_time = time.time() - start_time
        logger.info(f"\n✅ Training completed in {total_time:.2f}s ({total_time/60:.1f} min)")
        logger.info(f"Best validation loss: {best_val_loss:.6f}")
        logger.info(f"Final learning rate: {current_lr:.2e}")

    def predict(self, X: Features) -> Predictions:
        """Generate predictions using the trained MLP."""
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        logger.info(f"Predicting on {len(X)} samples")
        self._model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self._model(X_tensor)

            if self.task_type == TaskType.CLASSIFICATION:
                _, predictions = torch.max(outputs, dim=1)
            else:  # REGRESSION
                predictions = outputs.squeeze()

        result = predictions.cpu().numpy().astype(np.float64)
        logger.debug(f"Prediction shape: {result.shape}")
        return result

    def predict_proba(self, X: Features) -> np.ndarray:
        """Generate probability predictions for classification tasks."""
        if self.task_type != TaskType.CLASSIFICATION:
            raise ValueError("predict_proba is only available for classification tasks")
        
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        logger.info(f"Predicting probabilities for {len(X)} samples")
        
        self._model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            probabilities = torch.softmax(self._model(X_tensor), dim=1)

        result = probabilities.cpu().numpy()
        logger.debug(f"Probability shape: {result.shape}, first 5 probs: {result[0][:5]}")
        return result


    def save_to_file(self, folder: str, model_id: str, version: int = 0) -> None:
        """Save model state_dict and architecture params (not the full model object)."""
        
        if self._model is None:
            raise RuntimeError("Cannot save: Model not initialized.")

        # Extract serializable metadata
        payload = {
            "version": version,
            "task_type": self.task_type.value,  # ✅ Convert Enum to str for JSON safety
            "params": self._params.copy(),      # ✅ Copy param dict (no references)
            "class_name": self.__class__.__name__,
            "created_at": datetime.datetime.now().isoformat(),
            # ✅ Save ONLY the state_dict (lightweight, serializable)
            "model_state_dict": self._model.state_dict(),
            # ✅ Save architecture params needed to reconstruct model
            "input_size": self._params.get("input_size"),
            "hidden_sizes": self._params.get("hidden_sizes", []),
            "output_size": self._params.get("output_size"),
            "dropout": self._params.get("dropout", 0.0),
        }

        # Build filename
        filename = os.path.join(folder, f"mlpmodel_{model_id}_{version}.pt")  # ✅ Use .pt for PyTorch

        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Save with torch.save (best for PyTorch)
        torch.save(payload, filename)
        logger.info(f"✅ Model saved to: {filename}")

    @classmethod
    def load_from_file(cls, filename: str) -> "MLPModel":
        """Load model from PyTorch file. Reconstructs architecture and loads state_dict."""
        
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Model file not found: {filename}")

        payload = torch.load(filename, map_location='cpu')  # Load on CPU first for safety

        # Validate required fields
        required_keys = {"version", "task_type", "params", "class_name", 
                         "model_state_dict", "input_size", "hidden_sizes", "output_size", "dropout"}
        missing = required_keys - set(payload.keys())
        if missing:
            raise ValueError(f"Missing keys in saved model: {missing}")

        # Reconstruct class
        if payload["class_name"] != cls.__name__:
            raise ValueError(f"Expected class {cls.__name__}, got {payload['class_name']}")

        # Reconstruct task_type (from str back to Enum)
        try:
            task_type = TaskType(payload["task_type"])
        except ValueError:
            raise ValueError(f"Invalid task_type: {payload['task_type']}")

        # Reconstruct model instance using params
        model_instance = cls(
            task_type=task_type,
            **payload["params"]  # This includes input_size, hidden_sizes, etc.
        )

        # ✅ Manually reconstruct model architecture (since params already set)
        model_instance._initialize_model()  # This builds the MLP with correct arch

        # ✅ Load the state_dict into the newly constructed model
        if model_instance._model is None:
            raise RuntimeError("Model failed to initialize during load.")

        # Load state dict
        model_instance._model.load_state_dict(payload["model_state_dict"])
        
        # Move to appropriate device (retain original)
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        model_instance._model.to(device)
        
        # NOTE: No need to restore optimizer/scheduler — they will be re-initialized on first .fit()
        #       You could save them if needed, but it's rarely useful after training.

        logger.info(f"✅ Model loaded from: {filename}")
        logger.info(f"   Class: {payload['class_name']}, Version: {payload['version']}, "
                    f"Created: {payload['created_at']}")

        return model_instance
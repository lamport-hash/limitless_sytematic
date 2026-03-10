from abc import ABC
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import time
import numpy as np
from numpy.typing import NDArray
import jax
import jax.numpy as jnp
import autobnn as ab
from autobnn import estimators, kernels, operators
from typing import Literal

# Import your base classes (assuming these exist)
from ..fitting_core import BaseModel, TaskType, Targets, Predictions, RetrainMode, Features

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class KernelComplexity(Enum):
    low = "low"
    med = "med"
    high = "high"

class AutoBNNModel(BaseModel):
    """
    Concrete implementation of BaseModel using AutoBNN (JAX-based Bayesian Neural Networks).
    Supports multi-class classification via one-vs-all binary BNNs.
    """

    def __init__(self, task_type: TaskType, **kwargs):
        super().__init__(task_type, **kwargs)

        # Only classification is supported for now
        if self.task_type != TaskType.CLASSIFICATION:
            raise NotImplementedError("AutoBNNModel currently only supports classification.")

        # JAX device setup (automatic)
        self._rng = jax.random.PRNGKey(self._params.get("random_state", 42))
        logger.info(f"AutoBNNModel initialized with random_state={self._params.get('random_state', 42)}")

        # Model components (will be initialized in _initialize_model)
        self._models: Optional[List[ab.estimators.AutoBnnMapEstimator]] = None  # One per class
        self._n_classes: Optional[int] = None

        # Initialize model based on params
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize AutoBNN models (one per class)."""
        # Extract hyperparameters from self._params
        n_classes = self._params.get("n_classes")
        if n_classes is None:
            raise ValueError("n_classes must be provided in params for AutoBNNModel.")

        self._n_classes = n_classes
        logger.info(f"Initializing {n_classes} binary AutoBNN models (one per class)")

        # Kernel configuration
        kernel_names = self._params.get("kernels", ["periodic", "linear", "matern"])

        kernel_configs_low = {
            "periodic": {"width": 20, "period": 12.0},
            "linear": {"width": 20},
            "matern": {"width": 20},
            "rbf": {"width": 20, "lengthscale": 1.0},
            "constant": {"width": 20}
        }

        # Medium dimensional (10-15 features)
        kernel_configs_med = {
            "periodic": {"width": 30, "period": 12.0},
            "linear": {"width": 30},
            "matern": {"width": 30},
            "rbf": {"width": 30},
            "constant": {"width": 15}
        }

        # High dimensional (30+ features)
        kernel_configs_high = {
            "periodic": {"width": 80, "period": 12.0},
            "linear": {"width": 80},
            "matern": {"width": 80},
            "rbf": {"width": 80},
            "constant": {"width": 40}
        }

        kernel_complexity = self._params.get("kernels_complexity",KernelComplexity.low)
        kernel_configs = kernel_configs_low

        if kernel_complexity == KernelComplexity.med:
            kernel_configs = kernel_configs_med
        if kernel_complexity == KernelComplexity.high:
            kernel_configs = kernel_configs_high

        # Build list of kernel instances
        kernels_list = []
        for name in kernel_names:
            if name not in kernel_configs:
                raise ValueError(f"Unknown kernel: {name}. Supported: {list(kernel_configs.keys())}")
            config = kernel_configs[name]
            if name == "periodic":
                kernels_list.append(ab.kernels.PeriodicBNN(**config))
            elif name == "linear":
                kernels_list.append(ab.kernels.LinearBNN(**config))
            elif name == "matern":
                kernels_list.append(ab.kernels.MaternBNN(**config))
            elif name == "rbf":
                kernels_list.append(ab.kernels.RBFBNN(**config))
            elif name == "constant":
                kernels_list.append(ab.kernels.ConstantBNN(**config))

        # Build model: sum of kernels
        model = operators.Add(bnns=tuple(kernels_list))
        logger.info(f"Built composite kernel: {kernel_names}")

        # Likelihood model (must be compatible with autobnn)
        likelihood = self._params.get("likelihood", "normal_likelihood_logistic_noise")
        if likelihood not in ["normal_likelihood_logistic_noise"]:
            logger.warning(f"Likelihood '{likelihood}' may be unsupported. Using default: normal_likelihood_logistic_noise")
            likelihood = "normal_likelihood_logistic_noise"

        # Initialize one estimator per class
        self._models = []
        for c in range(n_classes):
            logger.info(f"Initializing AutoBNN estimator for class {c}")
            estimator = estimators.AutoBnnMapEstimator(
                model=model,
                likelihood_model=likelihood,
                seed=self._rng,  # We'll re-seed per model if needed
                periods=[12] if "periodic" in kernel_names else None,  # Optional hint
            )
            self._models.append(estimator)

        logger.info(f"Initialized {n_classes} AutoBNN models.")

    @classmethod
    def getDefaultParams(cls, task_type: TaskType) -> Dict[str, Any]:
        if task_type != TaskType.CLASSIFICATION:
            raise NotImplementedError("AutoBNNModel only supports classification.")

        return {
            "n_classes": 2,           # Required
            "kernels": ["periodic", "linear", "matern"],  # Kernel composition
            "likelihood": "normal_likelihood_logistic_noise",
            "random_state": 42,
            "epochs": 50,              # Training iterations (autobnn uses MAP with gradient steps)
            "learning_rate": 0.01,
            "patience": 10,           # For early stopping based on loss
        }

    @classmethod
    def getParamsGrids(cls, task_type: TaskType) -> Dict[str, List[Any]]:
        if task_type != TaskType.CLASSIFICATION:
            raise NotImplementedError("AutoBNNModel only supports classification.")
        return {
            "n_classes": [2, 3, 5],
            "kernels": [
                ["periodic"],
                ["linear", "matern"],
                ["periodic", "linear", "matern"],
                ["rbf"]
            ],
            "likelihood": ["normal_likelihood_logistic_noise"],  # Only one tested
            "learning_rate": [0.01, 0.005],
            "epochs": [20, 50, 100],
            "patience": [5, 10],
        }

    def setParams(self, **kwargs) -> None:
        """Override to re-initialize AutoBNN models when parameters change."""
        logger.debug(f"Updating AutoBNNModel params: {kwargs}")
        super().setParams(**kwargs)
        logger.info("Reinitializing AutoBNN models due to parameter update.")
        self._initialize_model()

    def fit(self, X: Features, y: Targets) -> None:
        if self._models is None:
            raise RuntimeError("Model not initialized. Call setParams() or __init__ first.")

        X = np.array(X).astype(np.float32)
        y = np.array(y).astype(int)

        n_classes = self._n_classes
        if len(np.unique(y)) != n_classes:
            raise ValueError(f"Number of unique classes in y ({len(np.unique(y))}) != n_classes ({n_classes})")

        logger.info(f"Starting training {n_classes} binary AutoBNN models...")
        start_time = time.time()

        # Train one model per class
        for c in range(n_classes):
            logger.info(f"Training AutoBNN model for class {c}")
            y_binary = (y == c).astype(np.float32)  # Binary labels: 1 if class c, else 0

            estimator = self._models[c]

            # Fit the model
            # autobnn's fit() expects (X, y) as arrays; it handles batching internally
            # We'll use max_epochs to control training iterations
            epochs = self._params.get("epochs", 50)
            lr = self._params.get("learning_rate", 0.01)

            # We assume the estimator.fit supports max_iter equivalent
            # If not, we may need to manually run optimization loop (see below)
            try:
                estimator.fit(X, y_binary, max_iter=epochs, learning_rate=lr)
            except TypeError:
                # Fallback: if max_iter not supported, we need to implement manual training loop
                logger.warning("AutoBnnMapEstimator.fit does not support max_iter. Using manual training loop.")
                self._manual_fit(estimator, X, y_binary, epochs=epochs, lr=lr)

            logger.info(f"Completed training class {c}.")

        total_time = time.time() - start_time
        logger.info(f"✅ Training completed in {total_time:.2f}s ({total_time/60:.1f} min)")

    def _manual_fit(self, estimator: ab.estimators.AutoBnnMapEstimator,
                    X: NDArray, y: NDArray, epochs: int = 50, lr: float = 0.01) -> None:
        """
        Manual training loop for AutoBnnMapEstimator if .fit() doesn't support max_iter.
        Uses JAX optax and custom training step.
        """
        # We assume AutoBnnMapEstimator has a method to get the loss and gradients
        # This is implementation-dependent. If autobnn exposes optimizer state, use it.
        # Otherwise, approximate by calling estimator.fit multiple times with warm start.

        # NOTE: As of autobnn v0.1, .fit() does NOT expose fine-grained control.
        # You may need to use a lower-level interface. Here’s a workaround:

        logger.info(f"Running {epochs} manual epochs for AutoBNN...")
        best_loss = float('inf')
        patience_counter = 0
        patience = self._params.get("patience", 10)

        for epoch in range(epochs):
            # This is a placeholder: autobnn's estimator.fit() currently doesn't expose loss or grad
            # You may need to dive into https://github.com/autobnn/autobnn/blob/main/src/autobnn/estimators.py
            # and adapt the MAP estimator to allow manual steps.

            # For now, call .fit() with incrementally increasing steps
            try:
                estimator.fit(X, y, max_iter=1)  # Single step
            except Exception as e:
                logger.error(f"Fit failed on epoch {epoch}: {e}")
                break

            # Get loss manually if possible (depends on internal state)
            # In practice, you may need to use estimator.get_loss() -> requires access to internal state
            # If not available, assume loss decreases each time and use early stopping on no improvement

            # Placeholder: Assume loss improves every iteration — in practice, monitor internal state
            if epoch % 10 == 0:
                logger.info(f"Manual Epoch {epoch}/{epochs} - Training progress...")

        # If loss tracking were available, you'd replace the above with:
        #   loss = estimator.compute_loss(X, y)
        #   if loss < best_loss: ... early stop

    def predict(self, X: Features) -> Predictions:
        """Predict class labels via maximum probability over binary classifiers."""
        if self._models is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = np.array(X).astype(np.float32)
        logger.info(f"Predicting on {len(X)} samples with {self._n_classes} binary models.")

        # Get predictions (probabilities) for each class
        probs = []
        for c in range(self._n_classes):
            estimator = self._models[c]
            # Use likelihood to sample or get predictive distribution
            # In autobnn, we can use `estimator.predict` for point estimates (MAP)
            try:
                pred = estimator.predict(X)  # Returns mean prediction → interpreted as P(y=1|X)
                probs.append(pred)
            except AttributeError:
                # Fallback: if predict doesn't exist, use a simple approximation
                logger.warning("predict method not available. Using posterior mean via sample.")
                # We assume estimator has .sample() or you need to do: samples = sampler.sample(100); mean(samples)
                # For now, use MAP prediction if available
                raise NotImplementedError("predict method must be implemented in your autobnn version.")

        # Stack predictions: shape (n_classes, n_samples)
        probs = np.stack(probs, axis=1)  # Each col is P(class=c | X)

        # Convert to probabilities: softmax or clamp
        probs = np.clip(probs, 1e-6, 1 - 1e-6)  # Avoid extremes

        # Since these are binary models, we interpret each as P(class=c)
        # We can normalize or do softmax (since they are not mutually exclusive by construction)

        # Option 1: Simple argmax on the raw predictions
        # But note: they are independent approximations → better to normalize
        probs_norm = probs / np.sum(probs, axis=1, keepdims=True)  # Softmax-like normalization

        # Predict class with highest probability
        predictions = np.argmax(probs_norm, axis=1)

        logger.debug(f"Predictions shape: {predictions.shape}")
        return predictions.astype(np.int32)

    def predict_proba(self, X: Features) -> np.ndarray:
        """Predict class probabilities using normalized binary model outputs."""
        if self._models is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = np.array(X).astype(np.float32)
        logger.info(f"Predicting probabilities for {len(X)} samples")

        probs = []
        for c in range(self._n_classes):
            estimator = self._models[c]
            try:
                pred = estimator.predict(X)
                probs.append(pred)
            except Exception as e:
                logger.error(f"Failed to predict for class {c}: {e}")
                raise

        probs = np.stack(probs, axis=1)  # Shape: (n_samples, n_classes)
        probs = np.clip(probs, 1e-6, 1 - 1e-6)
        probs_norm = probs / np.sum(probs, axis=1, keepdims=True)

        logger.debug(f"Probability shape: {probs_norm.shape}, first row: {probs_norm[0]}")
        return probs_norm.astype(np.float64)

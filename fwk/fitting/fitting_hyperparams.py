
from .fitting_core import BaseModel, TaskType, Targets, Predictions, Features
from .fitting_models import TimeSeriesModelTrainer, Features,TrainingConfig, MetricCalculator

from .models.model_factory import ModelFactory

from .models.model_register import (
ModelType
)

from typing import Any, Dict
from enum import Enum

class ParamSearchStrat(Enum):
    GRID = "grid"
    RANDOM = "random"


class HyperparameterTuner:
    """Hyperparameter tuning with grid or random search."""
    
    def __init__(
        self,
        model_factory: type,
        model_type: ModelType,
        param_grid: dict[str, list[Any]],
        config: TrainingConfig,
        task_type: TaskType = TaskType.REGRESSION,
        metric_calculator: MetricCalculator | None = None,
        search_strategy: ParamSearchStrat = ParamSearchStrat.RANDOM,
        n_random_samples: int = 10
    ):
        """
        Initialize hyperparameter tuner.
        
        Args:
            model_factory: Class that creates model instances
            param_grid: Dictionary of parameter names to lists of values
            config: Training configuration
            task_type: Type of task
            metric_calculator: Metric for evaluation
            search_strategy: 'grid' for exhaustive, 'random' for random sampling
            n_random_samples: Number of random samples if search_strategy='random'
        """
        self.model_factory = model_factory
        self.model_type = model_type
        self.param_grid = param_grid
        self.config = config
        self.task_type = task_type
        self.metric_calculator = metric_calculator
        self.search_strategy = search_strategy
        self.n_random_samples = n_random_samples
        
        self.best_params_: dict[str, Any] | None = None
        self.best_score_: float | None = None
        self.results_: list[dict[str, Any]] = []
    
    def _generate_param_combinations(self) -> list[dict[str, Any]]:
        """Generate parameter combinations based on search strategy."""
        if self.search_strategy == "grid":
            # Exhaustive grid search
            import itertools
            keys = self.param_grid.keys()
            values = self.param_grid.values()
            combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
            return combinations
        
        else:  # random
            # Random sampling
            import random
            combinations = []
            keys = list(self.param_grid.keys())
            
            for _ in range(self.n_random_samples):
                combo = {k: random.choice(self.param_grid[k]) for k in keys}
                combinations.append(combo)
            
            return combinations
    
    def fit(self, X: Features, y: Targets) -> dict[str, Any]:
        """
        Run hyperparameter search.
        
        Args:
            X: Input features
            y: Target values
        
        Returns:
            Dictionary with best parameters and score
        """
        combinations = self._generate_param_combinations()
        best_score = float('inf') if self.task_type == TaskType.REGRESSION else float('-inf')
        best_params = None
        
        for params in combinations:
            # Create model with current parameters
            model = self.model_factory.create_model(model_type = self.model_type, task_type = self.task_type,**params)
            
            # Train and evaluate
            trainer = TimeSeriesModelTrainer(
                model=model,
                config=self.config,
                task_type=self.task_type,
                metric_calculator=self.metric_calculator
            )
            
            metrics = trainer.fit(X, y)
            
            # Use validation score if available, otherwise test score
            score = metrics.val_score if metrics.val_score is not None else metrics.test_score
            
            # Track results
            self.results_.append({
                "params": params,
                "train_score": metrics.train_score,
                "val_score": metrics.val_score,
                "test_score": metrics.test_score
            })
            
            # Update best (minimize for regression, maximize for classification)
            is_better = (
                score < best_score if self.task_type == TaskType.REGRESSION
                else score > best_score
            )
            
            if is_better:
                best_score = score
                best_params = params
        
        self.best_params_ = best_params
        self.best_score_ = best_score
        
        return {
            "best_params": best_params,
            "best_score": best_score,
            "all_results": self.results_
        }
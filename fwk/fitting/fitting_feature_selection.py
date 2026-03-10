from __future__ import annotations

import warnings
import gc
import psutil  # Add this dependency if not already installed
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import field
import pandas as pd
import numpy as np
from enum import Enum
import random

# --- ADD THIS ---
def get_system_memory() -> float:
    """Returns current memory usage in GB"""
    process = psutil.Process(os.getpid())
    mem_gb = process.memory_info().rss / 1024**3  # bytes to GB
    return mem_gb


from .fitting_core import Features, Targets, BaseModel, TaskType, RetrainMode
from .fitting_metrics import MetricCalculator
from .fitting_models import TimeSeriesModelTrainer, TrainingConfig, DataSplit, Normalizer, ModelMetrics


# =======================
# FeatureSelectionMethod Enum (UPDATED)
# =======================
class FeatureSelectionMethod(Enum):
    EXHAUSTIVE = "exhaustive"
    GREEDY_RFE = "greedy_rfe"
    GREEDY_CLEVER = "greedy_clever"  # <-- NEW STRATEGY


# =======================
# FeatureSelector Class (UPDATED)
# =======================
class FeatureSelector:
    """
    Performs feature selection over combinations of features in time series data.
    
    Iterates through different subsets of (already-bundled) features,
    trains a model using TimeSeriesModelTrainer, and selects the best combination
    based on validation/test performance.
    
    Supports:
        - Exhaustive or greedy search over feature subsets
        - Configurable model and metric
        - Periodic/retrain mode compatibility
    
    MEMORY SAFETY: Each trainer is freshly instantiated and explicitly garbage-collected.
    """

    def __init__(
        self,
        model: BaseModel,
        config: TrainingConfig,
        features_df: pd.DataFrame, 
        target_df: pd.Series,
        metric_calculator: MetricCalculator,
        feature_selection_strategy: FeatureSelectionMethod = FeatureSelectionMethod.GREEDY_RFE,
        max_features: int = 100,
        min_features: int = 10,
        n_combinations_to_test: Optional[int] = None,  # For exhaustive: limit search space
        verbose: bool = True,
        # --- NEW FOR GREEDY_CLEVER ---
        greedy_clever_p: int = 10,     # Number of random batches
        greedy_clever_n: int = 50,     # Features per batch
        greedy_clever_m: int = 10,     # Return top/bottom M features
    ):
        """
        Initialize the FeatureSelector.

        Args:
            model: BaseModel instance (e.g., LinearRegression, RandomForest).
            config: TrainingConfig defining splits and retraining strategy.
            features_df: DataFrame with **already-bundled** feature columns (n_samples x n_features).
            target_df: Series with target values (n_samples,).
            metric_calculator: Custom MetricCalculator
            feature_selection_strategy: "exhaustive", "greedy_rfe", or "greedy_clever"
            max_features: Max number of features to consider (for efficiency)
            min_features: Min number of features to include
            n_combinations_to_test: Limit exhaustive search for large feature sets.
            verbose: Print progress during selection.

            # --- NEW FOR GREEDY_CLEVER ---
            greedy_clever_p: Number of random batches to generate (default: 10)
            greedy_clever_n: Number of features per batch (default: 50) — must be >= min_features
            greedy_clever_m: Number of top/bottom features to return (default: 10)
        """
        self.model = model
        self.config = config
        self.features_df = features_df.copy()
        self.target_df = target_df.copy()
        self.task_type = model.task_type
        self.metric_calc = metric_calculator
        
        self.feature_selection_strategy = feature_selection_strategy
        self.max_features = max_features or len(features_df.columns)
        self.min_features = min_features
        self.n_combinations_to_test = n_combinations_to_test
        self.verbose = verbose

        # --- NEW FOR GREEDY_CLEVER ---
        self.greedy_clever_p = greedy_clever_p
        self.greedy_clever_n = max(greedy_clever_n, min_features)  # Ensure >= min_features
        self.greedy_clever_m = greedy_clever_m

        # Results storage
        self.results: List[Dict[str, Any]] = []
        self.best_features: List[str] = []
        self.best_metrics: ModelMetrics | None = None
        self.best_score: float = float('-inf') if model.task_type == TaskType.REGRESSION else 0.0

        # Feature importance tracking for GREEDY_CLEVER
        self.feature_scores: Dict[str, List[float]] = {}  # feature -> list of scores it achieved in subsets

        # Log initial memory
        #if self.verbose:
        #    print(f"[MEMORY] Initial system memory: {get_system_memory():.2f} GB")

    def _generate_feature_combinations(self, all_features: List[str]) -> List[List[str]]:
        """
        Generate feature subsets using exhaustive, greedy_rfe, or greedy_clever strategy.
        Strictly limits exhaustive combinations to n_combinations_to_test.
        """
        if self.feature_selection_strategy == FeatureSelectionMethod.EXHAUSTIVE:
            from itertools import combinations
            all_combinations = []
            start = max(self.min_features, 1)
            end = min(self.max_features + 1, len(all_features) + 1)

            total_combos = sum(len(list(combinations(all_features, r))) for r in range(start, end))
            if self.verbose:
                print(f"[MEMORY] Total possible combinations: {total_combos}")

            # STRICT LIMIT: Do not exceed n_combinations_to_test
            if self.n_combinations_to_test and total_combos > self.n_combinations_to_test:
                warnings.warn(f"Too many combinations ({total_combos}). Limiting to {self.n_combinations_to_test}")
                from itertools import islice
                for r in range(start, end):
                    combs = list(combinations(all_features, r))
                    # Sample up to n_combinations_to_test total across all r
                    remaining = self.n_combinations_to_test - len(all_combinations)
                    if remaining <= 0:
                        break
                    sampled = islice(combs, min(len(combs), remaining))
                    all_combinations.extend(list(sampled))
            else:
                for r in range(start, end):
                    all_combinations.extend(combinations(all_features, r))
            
            #if self.verbose:
            #    print(f"[MEMORY] Actual combinations to test: {len(all_combinations)}")
            return [list(c) for c in all_combinations]

        elif self.feature_selection_strategy == FeatureSelectionMethod.GREEDY_RFE:
            # GREEDY RFE: Start with all features, remove worst feature one-by-one
            current_features = set(all_features)
            subsets = []  # Will store [full, full-1, full-2, ... min_features]

            while len(current_features) > self.min_features:
                subsets.append(list(sorted(current_features)))

                # Evaluate impact of removing each feature
                scores = []
                for f in list(current_features):
                    subset_without_f = [x for x in current_features if x != f]
                    score = self._evaluate_feature_subset(subset_without_f)
                    scores.append((f, score))

                # Remove the feature whose removal causes LEAST drop (i.e., least important)
                scores.sort(key=lambda x: x[1], reverse=True)  # Highest score after removal = least important
                worst_feature, _ = scores[0]
                current_features.remove(worst_feature)

            # Add final subset
            subsets.append(list(sorted(current_features)))

            #if self.verbose:
            #    print(f"[MEMORY] RFE path: {[len(s) for s in subsets]} features per step")

            return subsets

        elif self.feature_selection_strategy == FeatureSelectionMethod.GREEDY_CLEVER:
            # GREEDY CLEVER: Generate P batches of N random features, then run greedy RFE on each
            if self.greedy_clever_n > len(all_features):
                warnings.warn(f"greedy_clever_n ({self.greedy_clever_n}) > total features ({len(all_features)}). Using all.")
                self.greedy_clever_n = len(all_features)

            subsets = []
            if self.verbose:
                print(f"[GREEDY_CLEVER] Generating {self.greedy_clever_p} batches of {self.greedy_clever_n} features each...")

            for batch_idx in range(self.greedy_clever_p):
                # Randomly sample N features
                sampled_features = random.sample(all_features, self.greedy_clever_n)
                current_features = set(sampled_features)

                
                print(f"[BATCH {batch_idx+1}/{self.greedy_clever_p}] Starting with {len(current_features)} features")

                # Perform greedy RFE on this batch until min_features reached
                while len(current_features) > self.min_features:
                    subset = list(sorted(current_features))
                    subsets.append(subset)

                    # Evaluate impact of removing each feature
                    scores = []
                    for f in list(current_features):
                        subset_without_f = [x for x in current_features if x != f]
                        score = self._evaluate_feature_subset(subset_without_f)
                        scores.append((f, score))

                    # Remove the feature whose removal causes LEAST drop (least important)
                    scores.sort(key=lambda x: x[1], reverse=True)  # Higher score after removal = less important
                    worst_feature, _ = scores[0]
                    current_features.remove(worst_feature)

                # Add final subset
                subsets.append(list(sorted(current_features)))

            if self.verbose:
                print(f"[MEMORY] Greedy Clever generated {len(subsets)} subsets from {self.greedy_clever_p} batches.")

            return subsets

        else:
            raise ValueError(f"Unsupported strategy: {self.feature_selection_strategy}")

    def _evaluate_feature_subset(self, feature_subset: List[str]) -> float:
        """
        Evaluate a specific feature subset.
        Each evaluation uses a NEW trainer. Memory is cleaned up after.
        """
        if not feature_subset:
            return float('-inf') if self.task_type == TaskType.REGRESSION else 0.0

        # Subset features
        X_subset = self.features_df[feature_subset].copy()
        
        # Clean: remove non-numeric, constant, or all-NaN columns
        X_subset = X_subset.select_dtypes(include=[np.number])
        X_subset = X_subset.loc[:, (X_subset != 0).any(axis=0)]
        X_subset = X_subset.dropna(axis=1, how='all')

        if len(X_subset.columns) == 0:
            return float('-inf') if self.task_type == TaskType.REGRESSION else 0.0

        aligned_target = self.target_df.loc[X_subset.index]

        if len(X_subset) != len(aligned_target):
            raise ValueError(
                f"Mismatch between feature and target lengths: "
                f"X has {len(X_subset)} rows, y has {len(aligned_target)} rows."
            )

        X_np = X_subset.values.astype(np.float32)
        y_np = aligned_target.values.astype(np.float32)

        # --- LOG MEMORY BEFORE TRAINER ---
        if self.verbose:
            print(f"[MEMORY] Before training {len(feature_subset)} features: {get_system_memory():.2f} GB")

        trainer = TimeSeriesModelTrainer(
            model=self.model,
            config=self.config,
            task_type=self.task_type,
            metric_calculator=self.metric_calc
        )

        try:
            metrics = trainer.fit(X_np, y_np)
        except Exception as e:
            print(f"[ERROR] Trainer failed for features {feature_subset}: {e}")
            return float('-inf') if self.task_type == TaskType.REGRESSION else 0.0

        score = metrics.val_score if metrics.val_score is not None else metrics.test_score
        if score is None:
            print(f"[WARNING] No valid score for {feature_subset}")
            return float('-inf') if self.task_type == TaskType.REGRESSION else 0.0

        # --- LOG MEMORY AFTER TRAINING AND BEFORE CLEANUP ---
        if self.verbose:
            print(f"[MEMORY] After training {len(feature_subset)} features: {get_system_memory():.2f} GB")
            print(f"[EVAL] Features: {len(feature_subset)} → Val Score: {score:.4f}")

        # --- TRACK FEATURE IMPORTANCE FOR GREEDY_CLEVER ---
        if self.feature_selection_strategy == FeatureSelectionMethod.GREEDY_CLEVER:
            for f in feature_subset:
                if f not in self.feature_scores:
                    self.feature_scores[f] = []
                self.feature_scores[f].append(score)

        # --- CRITICAL: CLEAN UP TRAINER TO FREE MEMORY ---
        del trainer
        del metrics
        gc.collect()  # Force garbage collection

        # --- LOG MEMORY AFTER CLEANUP ---
        if self.verbose:
            print(f"[MEMORY] After cleanup: {get_system_memory():.2f} GB")

        return score

    def fit(self) -> None:
        """
        Run feature selection over all combinations.
        Updates best_features and best_metrics in-place.
        """
        all_features = list(self.features_df.columns)
        
        combinations = self._generate_feature_combinations(all_features)

        if self.verbose:
            print(f"[MEMORY] Total combinations to test: {len(combinations)}")
            print("[START] Starting feature selection...")

        for i, subset in enumerate(combinations):
            if self.verbose:
                print(f"[STEP {i+1}/{len(combinations)}] Testing features: {subset[:5]}... ({len(subset)} total)")

            score = self._evaluate_feature_subset(subset)
            result = {
                "features": subset,
                "bundler": "N/A",
                "score": score,
                "num_features": len(subset),
            }
            
            if score > self.best_score:
                self.best_score = score
                self.best_features = subset.copy()
                if self.verbose:
                    print(f"[BEST] Found better combination: {subset} → Score: {score:.4f}")

            self.results.append(result)

        # Train final model on best features using full data — ONE TIME ONLY
        if self.best_features:
            print(f"[FINAL] Training final model on best features: {self.best_features}")
            X_final = self.features_df[self.best_features].copy()
            X_np = X_final.values.astype(np.float32)
            y_np = self.target_df.values.astype(np.float32)

            final_trainer = TimeSeriesModelTrainer(
                model=self.model,
                config=self.config,
                task_type=self.task_type,
                metric_calculator=self.metric_calc
            )

            self.best_metrics = final_trainer.fit(X_np, y_np)

            # Cleanup final trainer if not needed later
            del final_trainer
            gc.collect()

        else:
            raise RuntimeError("No valid feature combination found.")

        if self.verbose:
            print(f"[MEMORY] Final memory after fit: {get_system_memory():.2f} GB")

    def get_best_features(self) -> List[str]:
        """Return the best selected feature subset."""
        return self.best_features

    def get_best_bundler(self) -> None:
        """Return None — bundling is done externally. This method exists for backward compatibility."""
        warnings.warn("FeatureBundler is now external. Use your pre-bundled features directly.")
        return None

    def get_best_model_metrics(self) -> ModelMetrics:
        """Return the metrics for the best model."""
        if self.best_metrics is None:
            raise RuntimeError("No best metrics found. Run .fit() first.")
        return self.best_metrics

    def get_results_df(self) -> pd.DataFrame:
        """Return all test results as a DataFrame."""
        return pd.DataFrame(self.results)

    def predict(self, X_new: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the best selected feature subset.

        Args:
            X_new: New input features — must contain all columns from original `features_df`
                   and be pre-bundled (same transformation as training data).

        Returns:
            Predictions array
        """
        if not self.best_features:
            raise RuntimeError("Model must be fitted before prediction.")

        X_subset = X_new[self.best_features].copy()
        
        # Clean: drop non-numeric, constant, or all-NaN columns
        X_subset = X_subset.select_dtypes(include=[np.number])
        X_subset = X_subset.loc[:, (X_subset != 0).any(axis=0)]
        X_subset = X_subset.dropna(axis=1, how='all')

        if len(X_subset.columns) == 0:
            raise ValueError("No valid features after cleaning.")

        X_np = X_subset.values.astype(np.float32)

        # Re-train on full original data to ensure normalization state (do this once)
        X_orig = self.features_df[self.best_features].copy()
        X_orig_np = X_orig.values.astype(np.float32)
        y_np = self.target_df.values.astype(np.float32)

        final_trainer = TimeSeriesModelTrainer(
            model=self.model,
            config=self.config,
            task_type=self.task_type,
            metric_calculator=self.metric_calc
        )

        final_trainer.fit(X_orig_np, y_np)
        predictions = final_trainer.predict(X_np)

        # Clean up
        del final_trainer
        gc.collect()

        return predictions

    # --- NEW METHODS FOR GREEDY_CLEVER ---
    def get_top_m_features(self, m: Optional[int] = None) -> List[str]:
        """
        Returns the top M features based on average performance across all batches.
        Only valid when using GREEDY_CLEVER strategy.

        Args:
            m: Number of top features to return. Defaults to greedy_clever_m.
        """
        if self.feature_selection_strategy != FeatureSelectionMethod.GREEDY_CLEVER:
            raise ValueError("get_top_m_features() is only available for GREEDY_CLEVER strategy.")

        if not self.feature_scores:
            raise RuntimeError("No feature scoring data available. Run .fit() first.")

        m = m or self.greedy_clever_m
        avg_scores = {feature: np.mean(scores) for feature, scores in self.feature_scores.items()}
        top_features = sorted(avg_scores.keys(), key=lambda x: avg_scores[x], reverse=True)[:m]
        return top_features

    def get_bottom_m_features(self, m: Optional[int] = None) -> List[str]:
        """
        Returns the bottom M features based on average performance across all batches.
        Only valid when using GREEDY_CLEVER strategy.

        Args:
            m: Number of bottom features to return. Defaults to greedy_clever_m.
        """
        if self.feature_selection_strategy != FeatureSelectionMethod.GREEDY_CLEVER:
            raise ValueError("get_bottom_m_features() is only available for GREEDY_CLEVER strategy.")

        if not self.feature_scores:
            raise RuntimeError("No feature scoring data available. Run .fit() first.")

        m = m or self.greedy_clever_m
        avg_scores = {feature: np.mean(scores) for feature, scores in self.feature_scores.items()}
        bottom_features = sorted(avg_scores.keys(), key=lambda x: avg_scores[x])[:m]
        return bottom_features

    def get_feature_importance_df(self) -> pd.DataFrame:
        """
        Returns a DataFrame with feature names and their average scores across all evaluations.
        Only valid for GREEDY_CLEVER strategy.

        Returns:
            pd.DataFrame with columns: ['feature', 'avg_score', 'num_evaluations']
        """
        if self.feature_selection_strategy != FeatureSelectionMethod.GREEDY_CLEVER:
            raise ValueError("get_feature_importance_df() is only available for GREEDY_CLEVER strategy.")

        if not self.feature_scores:
            raise RuntimeError("No feature scoring data available. Run .fit() first.")

        df = pd.DataFrame([
            {
                "feature": f,
                "avg_score": np.mean(scores),
                "num_evaluations": len(scores)
            }
            for f, scores in self.feature_scores.items()
        ])
        return df.sort_values("avg_score", ascending=False)

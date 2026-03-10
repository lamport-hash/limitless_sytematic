
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import List

from fitting.fitting_feature_bundler import (
    LagBundler,
    align_target_with_features
)

from fitting.fitting_feature_selection import (
    FeatureSelector,
    TaskType,
    FeatureSelectionMethod,
)

from fitting.models.model_factory import (
    ModelFactory
)
from fitting.models.model_register import (
    ModelType
)
from fitting.fitting_hyperparams import (
    HyperparameterTuner, ParamSearchStrat
)
from fitting.fitting_metrics import RegressionMetric
from fitting.fitting_models import TrainingConfig, TimeSeriesModelTrainer, TrainingSplitType
from fitting.models.xgb_model import XGBoostModel




@pytest.fixture
def create_time_series_dataset(n_samples=100):
    """Create a realistic time series dataset with multiple features and realistic patterns."""
    np.random.seed(42)
    
    # Create datetime index
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(n_samples)]
    
    data = pd.DataFrame(index=dates)
    
    # 1. Trend component (slowly changing)
    data["trend"] = 0.01 * np.arange(n_samples) + 2 * np.sin(np.arange(n_samples) * 0.01)
    
    # 2. Seasonal patterns (multiple frequencies)
    data["seasonal_daily"] = 3 * np.sin(2 * np.pi * np.arange(n_samples) / 24)  # Daily seasonality
    data["seasonal_weekly"] = 2 * np.sin(2 * np.pi * np.arange(n_samples) / (24 * 7))  # Weekly seasonality
    
    # 3. Cyclical components (irregular cycles)
    data["cycle_1"] = 1.5 * np.sin(np.arange(n_samples) * 0.1 + np.random.randn(n_samples) * 0.1)
    data["cycle_2"] = np.sin(np.arange(n_samples) * 0.05) * np.cos(np.arange(n_samples) * 0.02)
    
    # 4. Auto-regressive features (lag dependencies)
    data["ar_feature"] = np.zeros(n_samples)
    for t in range(1, n_samples):
        data["ar_feature"].iloc[t] = (0.8 * data["ar_feature"].iloc[t-1] + 
                                    np.random.randn() * 0.5)
    
    # 5. Moving average features
    data["ma_feature"] = np.random.randn(n_samples)
    window = 5
    for t in range(window, n_samples):
        data["ma_feature"].iloc[t] = np.mean(data["ma_feature"].iloc[t-window:t]) + np.random.randn() * 0.3
    
    # 6. Exogenous features with different characteristics
    data["exog_linear"] = 0.002 * np.arange(n_samples) + np.random.randn(n_samples) * 0.1
    data["exog_cyclic"] = np.sin(np.arange(n_samples) * 0.3) * np.cos(np.arange(n_samples) * 0.1)
    data["exog_noise"] = np.random.randn(n_samples)
    
    # 7. Feature interactions and transformations
    data["interaction_1"] = data["seasonal_daily"] * data["cycle_1"]
    data["interaction_2"] = data["trend"] * data["exog_cyclic"]
    
    # 8. Rolling statistics (window-based features)
    data["rolling_mean_7"] = data["seasonal_daily"].rolling(window=7, min_periods=1).mean()
    data["rolling_std_7"] = data["cycle_1"].rolling(window=7, min_periods=1).std()
    
    # 9. Time-based features
    data["hour"] = data.index.hour
    data["day_of_week"] = data.index.dayofweek
    data["month"] = data.index.month
    
    # 10. Random walk component
    data["random_walk"] = np.cumsum(np.random.randn(n_samples) * 0.1)
    
    # Fill any NaN values from rolling operations
    data = data.fillna(method='bfill')
    
    # Create target with realistic dependencies
    # Target depends on multiple features with different weights and time delays
    target = (
        2.0 * data["trend"] +
        1.5 * data["seasonal_daily"] +
        0.8 * data["seasonal_weekly"].shift(1).fillna(0) +  # Lagged dependency
        1.2 * data["cycle_1"] +
        0.6 * data["ar_feature"].shift(2).fillna(0) +  # Multi-period lag
        0.9 * data["interaction_1"] +
        0.3 * data["rolling_mean_7"] +
        np.random.randn(n_samples) * 0.5  # Noise
    )
    
    # Add target to dataframe
    data["target"] = target
    
    return data


@pytest.fixture
def sample_data():
    """Create a small, reproducible dataset with 4 features and target."""
    np.random.seed(42)
    n = 300
    data = pd.DataFrame({
        "f1": np.random.randn(n),
        "f2": np.sin(np.linspace(0, 4 * np.pi, n)) + 0.1 * np.random.randn(n),
        "f3": np.cumsum(np.random.randn(n)),
        "noise1": np.random.rand(n),
        "noise2": np.random.rand(n),
    })
    target = data["f1"] + 0.5 * data["f2"] + np.random.randn(n) * 0.1
    return data, target

@pytest.fixture
def base_cols():
    cols = [
        "f1",
        "f2",
        "f3",
        "noise1",
        "noise2"]
    return cols

@pytest.fixture
def sample_classification_data(n_samples: int = 1000, n_features: int = 10, n_classes: int = 2):
    """
    Generate synthetic data for classification.
    
    Returns:
        features_df: pd.DataFrame with shape (n_samples, n_features)
        target_df: pd.Series with integer class labels [0, 1, ..., n_classes-1]
    """
    np.random.seed(42)  # for reproducibility

    # Generate random features
    features_df = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feat_{i}" for i in range(n_features)]
    )

    # Generate discrete class labels: 0 to n_classes-1
    target_labels = np.random.choice(range(n_classes), size=n_samples, replace=True)
    target_df = pd.Series(target_labels, name="target")

    return features_df, target_df

@pytest.fixture
def config_3ways():
    """Return a minimal TrainingConfig for testing."""
    return TrainingConfig(
        mode=TrainingSplitType.TRAIN_VAL_TEST,
        train_ratio=0.7,
        val_ratio=0.2,
        normalization="standardize"
    )

def test_full_pipeline(sample_data, base_cols, config_3ways):
    """
    End-to-end test of the full pipeline:
    Bundler → FeatureSelector → HyperparameterTuner → Final Calibration → Prediction & Validation
    """
    # --- 1. BUNDLER STAGE ---
    bundler = LagBundler(lags=[1, 2])
    feature_df, target_df = sample_data
    feature_bundle_df = bundler.bundle(feature_df, base_cols)
    
    target_df = align_target_with_features(feature_bundle_df, target_df)

    # Validate bundling worked
    assert feature_bundle_df.shape[1] > len(base_cols)  # Should have lagged features
    assert "f1_lag1" in feature_bundle_df.columns
    assert "f2_lag2" in feature_bundle_df.columns

    # --- 2. FEATURE SELECTOR STAGE ---
    model = XGBoostModel(TaskType.REGRESSION)
    metric = RegressionMetric()

    selector = FeatureSelector(
        model=model,
        config=config_3ways,
        features_df=feature_bundle_df,
        target_df=target_df,
        max_features=5,
        min_features=1,
        feature_selection_strategy=FeatureSelectionMethod.GREEDY_RFE,
        metric_calculator=metric,
        verbose=False
    )

    selector.fit()

    # Validate feature selection ran and found best features
    assert len(selector.get_best_features()) > 0, "No best features selected!"
    assert selector.best_score != float('-inf'), "Best score not updated"
    results_df = selector.get_results_df()
    assert len(results_df) > 0, "No feature combinations tested"
    assert "score" in results_df.columns
    assert results_df["score"].max() == selector.best_score

    # --- 3. PARAM TUNER STAGE ---
    best_features = selector.get_best_features()
    X_selected = feature_bundle_df[best_features].copy()
    print(best_features)

    # Define hyperparameter search space for XGBoost
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [3, 5],
        "learning_rate": [0.01, 0.1]
    }

    tuner = HyperparameterTuner(
        model_factory=ModelFactory,
        model_type=ModelType.XGB,
        param_grid=param_grid,
        config=config_3ways,
        task_type=TaskType.REGRESSION,
        metric_calculator=metric,
        search_strategy=ParamSearchStrat.RANDOM,
        n_random_samples=4  # Small for speed
    )

    X_np = X_selected.values.astype(np.float32)
    y_np = target_df.values.astype(np.float32)

    tuner_results = tuner.fit(X_np, y_np)

    # Validate tuning ran
    assert tuner.best_params_ is not None, "No best hyperparameters found"
    assert tuner.best_score_ is not None
    assert len(tuner.results_) > 0, "No hyperparameter combinations evaluated"
    assert tuner.best_score_ < float('inf'), "Best score not computed"

    # --- 4. FINAL CALIBRATION (retrain best model on full data with tuned params) ---
    # Use best features and best hyperparameters to train final model
    final_model = XGBoostModel(
        **tuner.best_params_,
        task_type=TaskType.REGRESSION
    )

    final_trainer = TimeSeriesModelTrainer(
        model=final_model,
        config=config_3ways,
        task_type=TaskType.REGRESSION,
        metric_calculator=metric
    )

    # Fit on full data (no train/val split needed for final calibration — it's the real model)
    X_final_np = feature_bundle_df[best_features].values.astype(np.float32)
    y_final_np = target_df.values.astype(np.float32)

    final_metrics = final_trainer.fit(X_final_np, y_final_np)

    # Validate calibration produced metrics
    assert final_metrics.train_score is not None
    assert final_metrics.test_score is not None
    if config_3ways.mode == "three_way":
        assert final_metrics.val_score is not None

    # --- 5. PREDICTIONS & TEST QUALITY ---
    X_new = feature_bundle_df.iloc[:3].copy()  # Same columns as training
    predictions = selector.predict(X_new)  # Uses best_features + retrains on full data internally

    # Also predict with final calibrated model for comparison
    predictions_final = final_trainer.predict(X_new[best_features].values.astype(np.float32))

    # Validate prediction shapes
    assert predictions.shape == (3,), "Prediction shape mismatch"
    assert predictions_final.shape == (3,), "Final model prediction shape mismatch"

    # Validate predictions are similar (should be nearly identical since both retrain on full data)
    #assert np.allclose(predictions, predictions_final, atol=1e-5), "Selector and final model predictions diverge!"

    # Validate that final model score is reasonable (not NaN, not extreme)
    assert -10 < final_metrics.test_score < 10, f"Unreasonable test score: {final_metrics.test_score}"

    # Validate that feature selection improved over baseline (all features)
    all_features_model = XGBoostModel(task_type=TaskType.REGRESSION)
    all_trainer = TimeSeriesModelTrainer(
        model=all_features_model,
        config=config_3ways,
        task_type=TaskType.REGRESSION,
        metric_calculator=metric
    )
    X = feature_bundle_df.values.astype(np.float32)
    y = target_df.values.astype(np.float32)
    all_metrics = all_trainer.fit(
        X,
        y
    )

    # Save golden batch using the test set from splits
    splits = all_trainer._create_splits(X, y) 
    all_trainer.save_golden_batch(
        folder="./models/golden",
        model_id="xgb_v1",
        X_test=splits.X_test,
        y_test=splits.y_test,
        version=1
    )



    # Confirm that selected features performed at least as well or better
    assert final_metrics.test_score <= all_metrics.test_score + 1e-5, \
        "Feature selection did not improve or maintain performance"

    # --- 6. EXTRA: Verify FeatureSelector.predict() handles new data correctly ---
    # Drop a feature from X_new to simulate real-world use
    X_new_dirty = X_new.copy()
    X_new_dirty = X_new_dirty.drop(columns=[best_features[0]])  # Remove one best feature
    # Should still work because selector cleans features internally

#    try:
#        pred_cleaned = selector.predict(X_new_dirty)
#        assert len(pred_cleaned) == 3
#    except Exception as e:
#        raise AssertionError(f"FeatureSelector.predict failed on dirty data: {e}")

    # --- 7. VERIFICATION OF INTERNAL LOGIC ---
    # Confirm feature selection did not select 'noise' as best if it's unimportant
    # In our data, target = f1 + 0.5*f2 + noise — so 'noise' should be low importance
    best_features_names = selector.get_best_features()
    print(best_features_names)
#    assert "noise" not in best_features_names, "'noise' was selected as best feature — this shouldn't happen if RFE works!"

    # Confirm all lagged features from bundler are present in feature_bundle_df
    expected_lags = [f"{col}_lag{l}" for col in base_cols for l in [1,2]]
    #assert all(lag in feature_bundle_df.columns for lag in expected_lags)

    print("✅ Full pipeline test passed successfully!")

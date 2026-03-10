
import pytest
import pandas as pd
import numpy as np
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

from core.enums import (
    g_close_col,
    g_open_col,
    g_high_col,
    g_low_col,
    g_volume_col,
    g_open_time_col,
    g_close_time_col,
    g_index_col,
)

DATA_PATH = "data/bundle/candle_1hour/QQQ_features.parquet"
SKIP_HYPERPARAMETER_TUNER = True


@pytest.fixture
def real_data():
    df = pd.read_parquet(DATA_PATH)
    if "target" not in df.columns:
        df["target"] = df[g_close_col].pct_change().shift(-1)
    df = df.dropna()
    return df


@pytest.fixture
def feature_columns(real_data):
    exclude_cols = ["target"]
    if g_close_col in real_data.columns:
        exclude_cols.append(g_close_col)
    cols = [c for c in real_data.columns if c not in exclude_cols]
    return cols


@pytest.fixture
def config_3ways():
    return TrainingConfig(
        mode=TrainingSplitType.TRAIN_VAL_TEST,
        train_ratio=0.7,
        val_ratio=0.2,
        normalization="standardize"
    )


def test_full_pipeline_realdata(real_data, feature_columns, config_3ways):
    real_data = real_data.head(10000).reset_index(drop=True)
    feature_columns = feature_columns[::30]
    print(f"\n[1/6] Initial data: {real_data.shape[0]} rows, {real_data.shape[1]} columns")
    print(f"      Feature columns (1 every 30): {len(feature_columns)}")

    bundler = LagBundler(lags=[1, 2])
    feature_df = real_data[feature_columns].copy()
    target_df = real_data["target"].copy()
    
    feature_bundle_df = bundler.bundle(feature_df, feature_columns)
    target_df = align_target_with_features(feature_bundle_df, target_df)
    print(f"\n[2/6] After bundling: {feature_bundle_df.shape[0]} rows, {feature_bundle_df.shape[1]} features")

    assert feature_bundle_df.shape[1] > len(feature_columns)
    
    model = XGBoostModel(TaskType.REGRESSION)
    metric = RegressionMetric()

    selector = FeatureSelector(
        model=model,
        config=config_3ways,
        features_df=feature_bundle_df,
        target_df=target_df,
        max_features=10,
        min_features=3,
        feature_selection_strategy=FeatureSelectionMethod.GREEDY_RFE,
        metric_calculator=metric,
        verbose=True
    )

    print("\n[3/6] Running feature selection...")
    selector.fit()

    assert len(selector.get_best_features()) > 0
    assert selector.best_score != float('-inf')
    
    best_features = selector.get_best_features()
    print(f"\n[4/6] Selected {len(best_features)} features: {best_features}")

    X_selected = feature_bundle_df[best_features].copy()

    X_np = X_selected.values.astype(np.float32)
    y_np = target_df.values.astype(np.float32)

    if SKIP_HYPERPARAMETER_TUNER:
        print("\n[5/6] Skipping HyperparameterTuner (SKIP_HYPERPARAMETER_TUNER=True)")
        best_params = {}
    else:
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
            n_random_samples=4
        )

        print("\n[5/6] Running hyperparameter tuning...")
        tuner_results = tuner.fit(X_np, y_np)
        best_params = tuner.best_params_
        assert best_params is not None

    final_model = XGBoostModel(
        **best_params,
        task_type=TaskType.REGRESSION
    )

    final_trainer = TimeSeriesModelTrainer(
        model=final_model,
        config=config_3ways,
        task_type=TaskType.REGRESSION,
        metric_calculator=metric
    )

    X_final_np = feature_bundle_df[best_features].values.astype(np.float32)
    y_final_np = target_df.values.astype(np.float32)

    print(f"\n[6/6] Training final model on {X_final_np.shape[0]} samples...")
    final_metrics = final_trainer.fit(X_final_np, y_final_np)

    assert final_metrics.train_score is not None
    assert final_metrics.test_score is not None

    X_new = feature_bundle_df.iloc[:3].copy()
    predictions = selector.predict(X_new)
    predictions_final = final_trainer.predict(X_new[best_features].values.astype(np.float32))

    assert predictions.shape == (3,)
    assert predictions_final.shape == (3,)

    print(f"\n=== Results ===")
    print(f"Best features: {best_features}")
    print(f"Best params: {best_params}")
    print(f"Test score: {final_metrics.test_score}")
    print("Full pipeline test on real data passed!")

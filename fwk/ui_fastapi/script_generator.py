from __future__ import annotations

from typing import Any

from schemas import (
    FittingConfig,
    MetricType,
    ModelType,
    TaskType,
    TrainingSplitType,
)


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    elif isinstance(value, str):
        return f'"{value}"'
    elif isinstance(value, list):
        return "[" + ", ".join(_format_value(v) for v in value) + "]"
    elif value is None:
        return "None"
    else:
        return str(value)


def _format_dict(d: dict[str, Any], indent: int = 0) -> str:
    lines = []
    prefix = "    " * indent
    for key, value in d.items():
        formatted_value = _format_value(value)
        lines.append(f'{prefix}    "{key}": {formatted_value},')
    return "\n".join(lines)


def generate_script(config: FittingConfig) -> str:
    lines = [
        '"""',
        "Auto-generated Fitting Script",
        f"Task: {config.task_type.value}",
        f"Model: {config.model_type.value}",
        '"""',
        "",
        "import os",
        "import sys",
        "",
        "import numpy as np",
        "import pandas as pd",
        "",
        "from fitting.fitting_core import (",
        "    TaskType,",
        "    TrainingSplitType,",
        "    RetrainMode,",
        "    TrainingConfig,",
        "    Normalizer,",
        ")",
        "from fitting.fitting_metrics import MetricType, AveragingType, RegressionMetric, ClassificationMetric",
        "from fitting.fitting_models import TimeSeriesModelTrainer",
        "from fitting.fitting_hyperparams import HyperparameterTuner, ParamSearchStrat",
        "from fitting.fitting_feature_selection import FeatureSelector, FeatureSelectionMethod",
        "from fitting.fitting_feature_bundler import FeatureBundlerFactory",
        "from fitting.models.model_factory import ModelFactory",
        "from fitting.models.model_register import ModelType",
        "",
        "",
        "def main():",
    ]

    if config.data_source.use_synthetic:
        lines.extend(
            [
                f'    print("Generating synthetic data...")',
                f"    np.random.seed(42)",
                f"    n_samples = {config.data_source.synthetic_n_samples}",
                f"    n_features = {config.data_source.synthetic_n_features}",
                f"    X = np.random.randn(n_samples, n_features)",
            ]
        )
        if config.task_type == TaskType.REGRESSION:
            lines.extend(
                [
                    f"    y = np.random.randn(n_samples)",
                ]
            )
        else:
            lines.extend(
                [
                    f"    y = np.random.randint(0, 2, n_samples)",
                ]
            )
        lines.extend(
            [
                f'    feature_names = [f"feature_{{i}}" for i in range(n_features)]',
                f"    features_df = pd.DataFrame(X, columns=feature_names)",
                f'    target_df = pd.Series(y, name="target")',
            ]
        )
    else:
        lines.extend(
            [
                f'    print("Loading data from file...")',
                f'    data_path = "{config.data_source.file_path}"',
                f'    df = pd.read_csv(data_path) if data_path.endswith(".csv") else pd.read_parquet(data_path)',
                f'    target_col = "{config.data_source.target_column}"',
            ]
        )
        if config.data_source.feature_columns:
            lines.extend(
                [
                    f"    feature_cols = {config.data_source.feature_columns}",
                    f"    features_df = df[feature_cols]",
                ]
            )
        else:
            lines.extend(
                [
                    f"    feature_cols = [c for c in df.columns if c != target_col]",
                    f"    features_df = df[feature_cols]",
                ]
            )
        lines.extend(
            [
                f"    target_df = df[target_col]",
                f"    X = features_df.to_numpy()",
                f"    y = target_df.to_numpy()",
            ]
        )

    lines.extend(
        [
            "",
            "    # Task type",
            f"    task_type = TaskType.{config.task_type.value.upper()}",
            "",
            "    # Training config",
        ]
    )

    tc = config.training
    lines.extend(
        [
            f"    training_config = TrainingConfig(",
            f"        mode=TrainingSplitType.{tc.mode.value.upper()},",
            f"        train_ratio={tc.train_ratio},",
            f"        val_ratio={tc.val_ratio},",
        ]
    )
    if tc.retrain_period:
        lines.append(f"        retrain_period={tc.retrain_period},")
    lines.append(f"        retrain_mode=RetrainMode.{tc.retrain_mode.value.upper()},")
    if tc.sliding_window_size:
        lines.append(f"        sliding_window_size={tc.sliding_window_size},")
    lines.extend(
        [
            f'        normalization="{tc.normalization}",',
            f"        normalize_targets={tc.normalize_targets},",
            f"    )",
            "",
        ]
    )

    if (
        config.feature_bundling.lag.enabled
        or config.feature_bundling.rolling.enabled
        or config.feature_bundling.interaction.enabled
    ):
        lines.extend(
            [
                "    # Feature bundling",
                "    bundler_config = {}",
            ]
        )
        if config.feature_bundling.lag.enabled:
            lb = config.feature_bundling.lag
            lines.append(f'    bundler_config["lag"] = {{"lags": {lb.lags}}}')
        if config.feature_bundling.rolling.enabled:
            rb = config.feature_bundling.rolling
            lines.append(
                f'    bundler_config["rolling"] = {{"windows": {rb.windows}, "ops": {rb.ops}}}'
            )
        if config.feature_bundling.interaction.enabled:
            ib = config.feature_bundling.interaction
            lines.append(
                f'    bundler_config["interaction"] = {{"max_interactions": {ib.max_interactions}}}'
            )
        lines.extend(
            [
                "    bundler = FeatureBundlerFactory.create_from_config(bundler_config)",
                "    basecols = list(features_df.columns[:5])  # Use first 5 columns as base",
                "    features_df = bundler.apply(features_df, basecols=basecols)",
                "    X = features_df.to_numpy()",
                "",
            ]
        )

    lines.extend(
        [
            "    # Metric calculator",
        ]
    )
    if config.task_type == TaskType.REGRESSION:
        lines.append(
            f"    metric_calculator = RegressionMetric(metric=MetricType.{config.metrics.primary_metric.value.upper()})"
        )
    else:
        lines.append(
            f"    metric_calculator = ClassificationMetric(metric=MetricType.{config.metrics.primary_metric.value.upper()}, averaging=AveragingType.{config.metrics.averaging.value.upper()})"
        )

    lines.extend(
        [
            "",
            "    # Model",
            f"    model_type = ModelType.{config.model_type.value.upper()}",
        ]
    )

    if config.model_params:
        lines.extend(
            [
                "    model_params = {",
            ]
        )
        for key, value in config.model_params.items():
            lines.append(f'        "{key}": {_format_value(value)},')
        lines.extend(
            [
                "    }",
                "    model = ModelFactory.create_model(model_type, task_type, **model_params)",
            ]
        )
    else:
        lines.append("    model = ModelFactory.create_model(model_type, task_type)")

    lines.extend(
        [
            "",
        ]
    )

    if config.hyperparameter_tuning.enabled:
        lines.extend(
            [
                "    # Hyperparameter tuning",
                f"    search_strategy = ParamSearchStrat.{config.hyperparameter_tuning.strategy.value.upper()}",
                "    param_grid = ModelFactory.get_params_grids(model_type, task_type)",
                "",
                "    tuner = HyperparameterTuner(",
                "        model_factory=ModelFactory,",
                "        model_type=model_type,",
                "        param_grid=param_grid,",
                "        config=training_config,",
                "        task_type=task_type,",
                "        metric_calculator=metric_calculator,",
                "        search_strategy=search_strategy,",
                f"        n_random_samples={config.hyperparameter_tuning.n_random_samples},",
                "    )",
                '    print("Starting hyperparameter tuning...")',
                "    tuning_result = tuner.fit(X, y)",
                "    print(f\"Best params: {{tuning_result['best_params']}}\")",
                "    print(f\"Best score: {{tuning_result['best_score']:.4f}}\")",
                "",
                "    # Re-create model with best params",
                "    model = ModelFactory.create_model(model_type, task_type, **tuning_result['best_params'])",
                "",
            ]
        )

    if config.feature_selection.enabled:
        lines.extend(
            [
                "    # Feature selection",
                f"    fs_method = FeatureSelectionMethod.{config.feature_selection.method.value.upper().replace('GREEDY_', 'GREEDY_')}",
                "",
                "    selector = FeatureSelector(",
                "        model=model,",
                "        config=training_config,",
                "        features_df=features_df,",
                "        target_df=target_df,",
                "        metric_calculator=metric_calculator,",
                "        feature_selection_strategy=fs_method,",
                f"        max_features={config.feature_selection.max_features},",
                f"        min_features={config.feature_selection.min_features},",
            ]
        )
        if config.feature_selection.n_combinations_to_test:
            lines.append(
                f"        n_combinations_to_test={config.feature_selection.n_combinations_to_test},"
            )
        if config.feature_selection.method.value == "greedy_clever":
            lines.extend(
                [
                    f"        greedy_clever_p={config.feature_selection.greedy_clever_p},",
                    f"        greedy_clever_n={config.feature_selection.greedy_clever_n},",
                    f"        greedy_clever_m={config.feature_selection.greedy_clever_m},",
                ]
            )
        lines.extend(
            [
                f"        verbose={config.verbose},",
                "    )",
                '    print("Starting feature selection...")',
                "    selector.fit()",
                "    best_features = selector.get_best_features()",
                '    print(f"Selected {{len(best_features)}} features: {{best_features}}")',
                "",
                "    # Update features",
                "    features_df = features_df[best_features]",
                "    X = features_df.to_numpy()",
                "    model = ModelFactory.create_model(model_type, task_type)",
                "",
            ]
        )

    lines.extend(
        [
            "    # Train model",
            "    trainer = TimeSeriesModelTrainer(",
            "        model=model,",
            "        config=training_config,",
            "        task_type=task_type,",
            "        metric_calculator=metric_calculator,",
            f"        verbose={config.verbose},",
            "    )",
            "",
            '    print("Training model...")',
            "    metrics = trainer.fit(X, y)",
            "",
            "    # Print results",
            '    print("\\n" + "=" * 50)',
            '    print("TRAINING RESULTS")',
            '    print("=" * 50)',
            '    print(f"Train {{config.metrics.primary_metric.value}}: {{metrics.train_score:.4f}}")',
        ]
    )
    if config.training.mode == TrainingSplitType.TRAIN_VAL_TEST:
        lines.extend(
            [
                "    if metrics.val_score is not None:",
                '        print(f"Val {{config.metrics.primary_metric.value}}: {{metrics.val_score:.4f}}")',
            ]
        )
    lines.extend(
        [
            "    if metrics.test_score is not None:",
            f'        print(f"Test {config.metrics.primary_metric.value}: {{metrics.test_score:.4f}}")',
            "",
        ]
    )

    if config.save_model:
        lines.extend(
            [
                "    # Save model",
                f'    save_path = "{config.model_save_path}"',
                f'    model_id = "{config.model_id}"',
                "    os.makedirs(save_path, exist_ok=True)",
                "    model.save_to_file(save_path, model_id, version=0)",
                f'    print(f"Model saved to {{save_path}}/{{model_id}}")',
                "",
            ]
        )

    lines.extend(
        [
            '    print("\\nDone!")',
            "    return metrics",
            "",
            "",
            'if __name__ == "__main__":',
            "    main()",
        ]
    )

    return "\n".join(lines)

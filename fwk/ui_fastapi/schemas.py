from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"


class TrainingSplitType(str, Enum):
    TRAIN_VAL_TEST = "train_val_test"
    PERIODIC_RETRAIN = "periodic_retrain"


class RetrainMode(str, Enum):
    EXPANDING = "expanding"
    SLIDING = "sliding"


class MetricType(str, Enum):
    MSE = "mse"
    RMSE = "rmse"
    MAE = "mae"
    R2 = "r2"
    ACCURACY = "accuracy"
    F1 = "f1"
    PRECISION = "precision"
    RECALL = "recall"


class AveragingType(str, Enum):
    MICRO = "micro"
    MACRO = "macro"
    WEIGHTED = "weighted"
    BINARY = "binary"


class ParamSearchStrat(str, Enum):
    GRID = "grid"
    RANDOM = "random"


class FeatureSelectionMethod(str, Enum):
    EXHAUSTIVE = "exhaustive"
    GREEDY_RFE = "greedy_rfe"
    GREEDY_CLEVER = "greedy_clever"


class ModelType(str, Enum):
    XGB = "xgb"
    RF_SK = "rf_sk"
    DT_SK = "dt_sk"
    MLP_TORCH = "mlp_torch"
    BNN_GAUTO = "bnn_gauto"


class DataSourceType(str, Enum):
    SYNTHETIC = "synthetic"
    FILE = "file"
    NORMALIZED = "normalized"


class XGBoostParams(BaseModel):
    n_estimators: int = Field(default=100, ge=1)
    max_depth: int = Field(default=6, ge=1)
    learning_rate: float = Field(default=0.1, gt=0)
    subsample: float = Field(default=1.0, ge=0.1, le=1.0)
    colsample_bytree: float = Field(default=1.0, ge=0.1, le=1.0)
    min_child_weight: int = Field(default=1, ge=1)


class RandomForestParams(BaseModel):
    n_estimators: int = Field(default=100, ge=1)
    max_depth: Optional[int] = Field(default=None)
    min_samples_split: int = Field(default=2, ge=2)
    min_samples_leaf: int = Field(default=1, ge=1)
    max_features: Optional[str] = Field(default="sqrt")


class DecisionTreeParams(BaseModel):
    max_depth: Optional[int] = Field(default=None)
    min_samples_split: int = Field(default=2, ge=2)
    min_samples_leaf: int = Field(default=1, ge=1)
    max_features: Optional[str] = Field(default=None)
    criterion: str = Field(default="squared_error")


class MLPParams(BaseModel):
    hidden_sizes: list[int] = Field(default_factory=lambda: [128, 64])
    batch_size: int = Field(default=32, ge=1)
    epochs: int = Field(default=100, ge=1)
    learning_rate: float = Field(default=0.001, gt=0)
    optimizer: str = Field(default="adam")
    dropout: float = Field(default=0.2, ge=0, le=1)
    weight_decay: float = Field(default=1e-4, ge=0)
    patience: int = Field(default=20, ge=1)


class AutoBNNParams(BaseModel):
    n_classes: int = Field(default=2, ge=2)
    kernels: list[str] = Field(default_factory=lambda: ["periodic", "linear", "matern"])
    epochs: int = Field(default=50, ge=1)
    learning_rate: float = Field(default=0.01, gt=0)
    patience: int = Field(default=10, ge=1)


class TrainingConfigSchema(BaseModel):
    mode: TrainingSplitType = Field(default=TrainingSplitType.TRAIN_VAL_TEST)
    train_ratio: float = Field(default=0.6, gt=0, lt=1)
    val_ratio: float = Field(default=0.2, gt=0, lt=1)
    retrain_period: Optional[int] = Field(default=None, ge=1)
    retrain_mode: RetrainMode = Field(default=RetrainMode.EXPANDING)
    sliding_window_size: Optional[int] = Field(default=None, ge=1)
    normalization: Literal["standardize", "minmax", "none"] = Field(
        default="standardize"
    )
    normalize_targets: bool = Field(default=False)


class MetricsConfig(BaseModel):
    primary_metric: MetricType = Field(default=MetricType.R2)
    averaging: AveragingType = Field(default=AveragingType.MACRO)


class HyperparameterTuningConfig(BaseModel):
    enabled: bool = Field(default=False)
    strategy: ParamSearchStrat = Field(default=ParamSearchStrat.RANDOM)
    n_random_samples: int = Field(default=10, ge=1)


class FeatureSelectionConfig(BaseModel):
    enabled: bool = Field(default=False)
    method: FeatureSelectionMethod = Field(default=FeatureSelectionMethod.GREEDY_RFE)
    max_features: int = Field(default=100, ge=1)
    min_features: int = Field(default=10, ge=1)
    n_combinations_to_test: Optional[int] = Field(default=None)
    greedy_clever_p: int = Field(default=10, ge=1)
    greedy_clever_n: int = Field(default=50, ge=1)
    greedy_clever_m: int = Field(default=10, ge=1)


class LagBundlerConfig(BaseModel):
    enabled: bool = Field(default=False)
    lags: list[int] = Field(default_factory=lambda: [1, 2, 3])


class RollingBundlerConfig(BaseModel):
    enabled: bool = Field(default=False)
    windows: list[int] = Field(default_factory=lambda: [5, 10])
    ops: list[str] = Field(default_factory=lambda: ["mean", "std"])


class InteractionBundlerConfig(BaseModel):
    enabled: bool = Field(default=False)
    max_interactions: int = Field(default=2, ge=1)


class FeatureBundlingConfig(BaseModel):
    lag: LagBundlerConfig = Field(default_factory=LagBundlerConfig)
    rolling: RollingBundlerConfig = Field(default_factory=RollingBundlerConfig)
    interaction: InteractionBundlerConfig = Field(
        default_factory=InteractionBundlerConfig
    )


class DataSourceConfig(BaseModel):
    source_type: DataSourceType = Field(default=DataSourceType.SYNTHETIC)
    file_path: Optional[str] = Field(default=None)
    target_column: str = Field(default="target")
    feature_columns: Optional[List[str]] = Field(default=None)
    use_synthetic: bool = Field(default=True)
    synthetic_n_samples: int = Field(default=1000, ge=100)
    synthetic_n_features: int = Field(default=10, ge=1)
    symbol: Optional[str] = Field(default=None)
    data_freq: Optional[str] = Field(default=None)
    source: Optional[str] = Field(default=None)
    product_type: Optional[str] = Field(default=None)
    start: Optional[str] = Field(default=None)
    end: Optional[str] = Field(default=None)


class FeatureItem(BaseModel):
    feature_type: str
    periods: Optional[List[int]] = Field(default=None)
    kwargs: Optional[Dict[str, Any]] = Field(default=None)


class FeaturesConfig(BaseModel):
    enabled: bool = Field(default=False)
    features: List[FeatureItem] = Field(default_factory=list)
    config_file: Optional[str] = Field(default=None)


class TargetItem(BaseModel):
    function: str
    params: Dict[str, Any] = Field(default_factory=dict)


class TargetsConfig(BaseModel):
    enabled: bool = Field(default=False)
    asset: Optional[str] = Field(default=None)
    targets: List[TargetItem] = Field(default_factory=list)
    config_file: Optional[str] = Field(default=None)


class JudgeConfig(BaseModel):
    enabled: bool = Field(default=True)
    generate_prompt: bool = Field(default=False)


class FittingConfig(BaseModel):
    task_type: TaskType = Field(default=TaskType.REGRESSION)
    model_type: ModelType = Field(default=ModelType.XGB)
    model_params: Dict[str, Any] = Field(default_factory=dict)
    training: TrainingConfigSchema = Field(default_factory=TrainingConfigSchema)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    hyperparameter_tuning: HyperparameterTuningConfig = Field(
        default_factory=HyperparameterTuningConfig
    )
    feature_selection: FeatureSelectionConfig = Field(
        default_factory=FeatureSelectionConfig
    )
    feature_bundling: FeatureBundlingConfig = Field(
        default_factory=FeatureBundlingConfig
    )
    data_source: DataSourceConfig = Field(default_factory=DataSourceConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    targets: TargetsConfig = Field(default_factory=TargetsConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    verbose: bool = Field(default=True)
    save_model: bool = Field(default=False)
    model_save_path: str = Field(default="./output")
    model_id: str = Field(default="model_001")


MODEL_PARAMS_MAP: dict[ModelType, type[BaseModel]] = {
    ModelType.XGB: XGBoostParams,
    ModelType.RF_SK: RandomForestParams,
    ModelType.DT_SK: DecisionTreeParams,
    ModelType.MLP_TORCH: MLPParams,
    ModelType.BNN_GAUTO: AutoBNNParams,
}

DEFAULT_PARAMS: dict[ModelType, dict[str, Any]] = {
    ModelType.XGB: XGBoostParams().model_dump(),
    ModelType.RF_SK: RandomForestParams().model_dump(),
    ModelType.DT_SK: DecisionTreeParams().model_dump(),
    ModelType.MLP_TORCH: MLPParams().model_dump(),
    ModelType.BNN_GAUTO: AutoBNNParams().model_dump(),
}

REGRESSION_METRICS = [MetricType.MSE, MetricType.RMSE, MetricType.MAE, MetricType.R2]
CLASSIFICATION_METRICS = [
    MetricType.ACCURACY,
    MetricType.F1,
    MetricType.PRECISION,
    MetricType.RECALL,
]

"""
Tests for fitting_judge.py module.
"""

import pytest
import numpy as np

from fitting.fitting_judge import (
    FittingJudge,
    FittingVerdict,
    ModelInfo,
    DataInfo,
    Issue,
    IssueType,
    QualityRating,
    generate_ai_analysis_prompt,
    quick_judge,
)
from fitting.fitting_models import ModelMetrics
from fitting.fitting_core import TaskType


def create_sample_metrics(
    train_score: float = 0.85,
    val_score: float = 0.80,
    test_score: float = 0.75,
    train_metrics: dict | None = None,
    val_metrics: dict | None = None,
    test_metrics: dict | None = None,
    retrain_history: list | None = None,
) -> ModelMetrics:
    """Create sample ModelMetrics for testing."""
    if train_metrics is None:
        train_metrics = {"score": train_score, "mse": 0.1, "r2": train_score}
    if val_metrics is None:
        val_metrics = {"score": val_score, "mse": 0.12, "r2": val_score} if val_score else None
    if test_metrics is None:
        test_metrics = {"score": test_score, "mse": 0.15, "r2": test_score} if test_score else None
    
    return ModelMetrics(
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        predictions=np.random.randn(100) if test_score else None,
        retrain_history=retrain_history if retrain_history is not None else [],
    )


def create_sample_model_info(
    model_type: str = "XGBoost",
    task_type: TaskType = TaskType.CLASSIFICATION,
) -> ModelInfo:
    """Create sample ModelInfo for testing."""
    return ModelInfo(
        model_type=model_type,
        task_type=task_type,
        params={"n_estimators": 100, "max_depth": 6},
        model_class="XGBoostModel",
    )


def create_sample_data_info(
    n_samples: int = 10000,
    n_features: int = 25,
    train_size: int = 6000,
    val_size: int = 2000,
    test_size: int = 2000,
) -> DataInfo:
    """Create sample DataInfo for testing."""
    return DataInfo(
        n_samples=n_samples,
        n_features=n_features,
        feature_names=[f"feature_{i}" for i in range(n_features)],
        target_name="T_target",
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        has_retrain=False,
        retrain_events=0,
    )


class TestFittingJudge:
    """Tests for FittingJudge class."""
    
    def test_init(self):
        """Test FittingJudge initialization."""
        metrics = create_sample_metrics()
        model_info = create_sample_model_info()
        data_info = create_sample_data_info()
        
        judge = FittingJudge(metrics, model_info, data_info)
        
        assert judge.metrics is metrics
        assert judge.model_info is model_info
        assert judge.data_info is data_info
    
    def test_evaluate_excellent(self):
        """Test evaluation of excellent fitting."""
        metrics = create_sample_metrics(train_score=0.95, val_score=0.90, test_score=0.88)
        model_info = create_sample_model_info()
        data_info = create_sample_data_info()
        
        judge = FittingJudge(metrics, model_info, data_info)
        verdict = judge.evaluate()
        
        assert verdict.rating in [QualityRating.EXCELLENT, QualityRating.GOOD]
        assert verdict.score >= 0.70
    
    def test_evaluate_overfitting(self):
        """Test detection of overfitting."""
        metrics = create_sample_metrics(train_score=0.95, val_score=0.70, test_score=0.65)
        model_info = create_sample_model_info()
        data_info = create_sample_data_info()
        
        judge = FittingJudge(metrics, model_info, data_info)
        verdict = judge.evaluate()
        
        overfitting_issues = [i for i in verdict.issues if i.type == IssueType.OVERFITTING]
        assert len(overfitting_issues) > 0
        assert any("overfitting" in i.message.lower() for i in overfitting_issues)
    
    def test_evaluate_underfitting(self):
        """Test detection of underfitting."""
        metrics = create_sample_metrics(train_score=0.40, val_score=0.38, test_score=0.35)
        model_info = create_sample_model_info()
        data_info = create_sample_data_info()
        
        judge = FittingJudge(metrics, model_info, data_info)
        verdict = judge.evaluate()
        
        underfitting_issues = [i for i in verdict.issues if i.type == IssueType.UNDERFITTING]
        assert len(underfitting_issues) > 0
    
    def test_evaluate_low_score(self):
        """Test detection of low scores."""
        metrics = create_sample_metrics(train_score=0.55, val_score=0.30, test_score=0.25)
        model_info = create_sample_model_info()
        data_info = create_sample_data_info()
        
        judge = FittingJudge(metrics, model_info, data_info)
        verdict = judge.evaluate()
        
        assert verdict.rating in [QualityRating.POOR, QualityRating.FAILED]
        low_score_issues = [i for i in verdict.issues if i.type == IssueType.LOW_SCORE]
        assert len(low_score_issues) > 0
    
    def test_evaluate_small_training_data(self):
        """Test detection of small training data issues."""
        metrics = create_sample_metrics(train_score=0.80, val_score=0.75, test_score=0.70)
        model_info = create_sample_model_info()
        data_info = DataInfo(
            n_samples=150,
            n_features=25,
            train_size=90,
            val_size=30,
            test_size=30,
        )
        
        judge = FittingJudge(metrics, model_info, data_info)
        verdict = judge.evaluate()
        
        unstable_issues = [i for i in verdict.issues if i.type == IssueType.UNSTABLE]
        assert len(unstable_issues) > 0
    
    def test_evaluate_high_feature_ratio(self):
        """Test detection of high feature-to-sample ratio."""
        metrics = create_sample_metrics(train_score=0.85, val_score=0.80, test_score=0.75)
        model_info = create_sample_model_info()
        data_info = DataInfo(
            n_samples=500,
            n_features=100,
            train_size=300,
            val_size=100,
            test_size=100,
        )
        
        judge = FittingJudge(metrics, model_info, data_info)
        verdict = judge.evaluate()
        
        overfit_issues = [i for i in verdict.issues if i.type == IssueType.OVERFITTING]
        assert len(overfit_issues) > 0
    
    def test_evaluate_with_retrain_history(self):
        """Test evaluation with retraining history."""
        retrain_history = [
            {"train_size": 1000, "test_chunk_metrics": {"score": 0.75}},
            {"train_size": 1000, "test_chunk_metrics": {"score": 0.78}},
            {"train_size": 1000, "test_chunk_metrics": {"score": 0.72}},
            {"train_size": 1000, "test_chunk_metrics": {"score": 0.80}},
        ]
        metrics = create_sample_metrics(
            train_score=0.85,
            val_score=0.80,
            test_score=0.75,
            retrain_history=retrain_history,
        )
        data_info = DataInfo(
            n_samples=5000,
            n_features=25,
            train_size=3000,
            val_size=1000,
            test_size=1000,
            has_retrain=True,
            retrain_events=4,
        )
        model_info = create_sample_model_info()
        
        judge = FittingJudge(metrics, model_info, data_info)
        verdict = judge.evaluate()
        
        assert isinstance(verdict, FittingVerdict)
        assert verdict.score >= 0
    
    def test_recommendations_generated(self):
        """Test that recommendations are generated for issues."""
        metrics = create_sample_metrics(train_score=0.95, val_score=0.65, test_score=0.60)
        model_info = create_sample_model_info()
        data_info = create_sample_data_info()
        
        judge = FittingJudge(metrics, model_info, data_info)
        verdict = judge.evaluate()
        
        assert len(verdict.recommendations) > 0
        assert any("regular" in r.lower() or "complexity" in r.lower() for r in verdict.recommendations)
    
    def test_summary_generated(self):
        """Test that summary is generated."""
        metrics = create_sample_metrics()
        model_info = create_sample_model_info()
        data_info = create_sample_data_info()
        
        judge = FittingJudge(metrics, model_info, data_info)
        verdict = judge.evaluate()
        
        assert len(verdict.summary) > 0


class TestQuickJudge:
    """Tests for quick_judge convenience function."""
    
    def test_quick_judge_basic(self):
        """Test quick_judge with basic parameters."""
        metrics = create_sample_metrics(train_score=0.80, val_score=0.75, test_score=0.70)
        
        verdict = quick_judge(
            metrics=metrics,
            model_type="XGBoost",
            task_type=TaskType.CLASSIFICATION,
            n_samples=10000,
            n_features=25,
        )
        
        assert isinstance(verdict, FittingVerdict)
        assert verdict.rating in QualityRating
        assert 0.0 <= verdict.score <= 1.0


class TestGenerateAIAnalysisPrompt:
    """Tests for generate_ai_analysis_prompt function."""
    
    def test_generates_prompt(self):
        """Test that prompt is generated."""
        metrics = create_sample_metrics()
        model_info = create_sample_model_info()
        data_info = create_sample_data_info()
        
        prompt = generate_ai_analysis_prompt(metrics, model_info, data_info)
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "ML Model Fitting Analysis" in prompt
        assert "XGBoost" in prompt
        assert "classification" in prompt.lower()
    
    def test_includes_metrics(self):
        """Test that metrics are included in prompt."""
        metrics = create_sample_metrics(train_score=0.85, val_score=0.80, test_score=0.75)
        model_info = create_sample_model_info()
        data_info = create_sample_data_info()
        
        prompt = generate_ai_analysis_prompt(metrics, model_info, data_info)
        
        assert "Training Set" in prompt
        assert "Validation Set" in prompt
        assert "Test Set" in prompt
    
    def test_includes_model_params(self):
        """Test that model parameters are included."""
        metrics = create_sample_metrics()
        model_info = ModelInfo(
            model_type="RandomForest",
            task_type=TaskType.REGRESSION,
            params={"n_estimators": 200, "max_depth": 10},
        )
        data_info = create_sample_data_info()
        
        prompt = generate_ai_analysis_prompt(metrics, model_info, data_info)
        
        assert "n_estimators" in prompt
        assert "200" in prompt
    
    def test_includes_data_info(self):
        """Test that data info is included."""
        metrics = create_sample_metrics()
        model_info = create_sample_model_info()
        data_info = DataInfo(
            n_samples=50000,
            n_features=100,
            feature_names=[f"feat_{i}" for i in range(100)],
            target_name="T_my_target",
            train_size=30000,
            val_size=10000,
            test_size=10000,
        )
        
        prompt = generate_ai_analysis_prompt(metrics, model_info, data_info)
        
        assert "50000" in prompt
        assert "100" in prompt
        assert "T_my_target" in prompt
    
    def test_includes_retrain_history(self):
        """Test that retrain history is included when present."""
        retrain_history = [
            {"train_size": 1000, "test_chunk_metrics": {"score": 0.75}},
            {"train_size": 1000, "test_chunk_metrics": {"score": 0.78}},
            {"train_size": 1000, "test_chunk_metrics": {"score": 0.80}},
        ]
        metrics = create_sample_metrics(retrain_history=retrain_history)
        model_info = create_sample_model_info()
        data_info = DataInfo(
            n_samples=5000,
            n_features=25,
            train_size=3000,
            val_size=1000,
            test_size=1000,
            has_retrain=True,
            retrain_events=3,
        )
        
        prompt = generate_ai_analysis_prompt(metrics, model_info, data_info)
        
        assert "Retraining History" in prompt
    
    def test_includes_analysis_template(self):
        """Test that analysis template is included."""
        metrics = create_sample_metrics()
        model_info = create_sample_model_info()
        data_info = create_sample_data_info()
        
        prompt = generate_ai_analysis_prompt(metrics, model_info, data_info)
        
        assert "Analysis Template" in prompt or "Overall Assessment" in prompt
        assert "Recommendations" in prompt


class TestDataclasses:
    """Tests for dataclasses."""
    
    def test_issue_dataclass(self):
        """Test Issue dataclass."""
        issue = Issue(
            type=IssueType.OVERFITTING,
            severity="high",
            message="Training-validation gap indicates overfitting",
            details={"gap": 0.25},
        )
        
        assert issue.type == IssueType.OVERFITTING
        assert issue.severity == "high"
        assert "overfitting" in issue.message.lower()
    
    def test_fitting_verdict_dataclass(self):
        """Test FittingVerdict dataclass."""
        verdict = FittingVerdict(
            rating=QualityRating.GOOD,
            score=0.75,
            issues=[],
            recommendations=["Increase training data"],
            summary="Good fitting quality",
        )
        
        assert verdict.rating == QualityRating.GOOD
        assert verdict.score == 0.75
        assert len(verdict.recommendations) == 1
    
    def test_model_info_dataclass(self):
        """Test ModelInfo dataclass."""
        info = ModelInfo(
            model_type="MLP",
            task_type=TaskType.REGRESSION,
            params={"hidden_layers": [64, 32]},
        )
        
        assert info.model_type == "MLP"
        assert info.task_type == TaskType.REGRESSION
    
    def test_data_info_dataclass(self):
        """Test DataInfo dataclass."""
        info = DataInfo(
            n_samples=1000,
            n_features=10,
            feature_names=["f1", "f2"],
            target_name="target",
            train_size=600,
            val_size=200,
            test_size=200,
        )
        
        assert info.n_samples == 1000
        assert info.n_features == 10
        assert info.train_size == 600


class TestIssueTypes:
    """Tests for IssueType enum values."""
    
    def test_issue_type_values(self):
        """Test that all expected issue types exist."""
        assert IssueType.OVERFITTING.value == "overfitting"
        assert IssueType.UNDERFITTING.value == "underfitting"
        assert IssueType.UNSTABLE.value == "unstable"
        assert IssueType.LOW_SCORE.value == "low_score"
        assert IssueType.HIGH_VARIANCE.value == "high_variance"
        assert IssueType.RETRAIN_DEGRADATION.value == "retrain_degradation"


class TestQualityRating:
    """Tests for QualityRating enum values."""
    
    def test_quality_rating_values(self):
        """Test that all expected quality ratings exist."""
        assert QualityRating.EXCELLENT.value == "excellent"
        assert QualityRating.GOOD.value == "good"
        assert QualityRating.ACCEPTABLE.value == "acceptable"
        assert QualityRating.POOR.value == "poor"
        assert QualityRating.FAILED.value == "failed"

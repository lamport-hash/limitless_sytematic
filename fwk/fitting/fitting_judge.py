"""
Fitting Judge Module

Provides tools to evaluate the quality of model fitting results:
1. Local Python-based analysis (FittingJudge class)
2. AI prompt generation for external analysis (generate_ai_analysis_prompt)

Usage:
    from fitting.fitting_judge import FittingJudge, generate_ai_analysis_prompt
    
    # Local analysis
    judge = FittingJudge(metrics, model_info, data_info)
    verdict = judge.evaluate()
    
    # Generate AI prompt
    prompt = generate_ai_analysis_prompt(metrics, model_info, data_info)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np

from .fitting_core import TaskType
from .fitting_models import ModelMetrics, TrainingConfig, TrainingSplitType


class QualityRating(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    FAILED = "failed"


class IssueType(Enum):
    OVERFITTING = "overfitting"
    UNDERFITTING = "underfitting"
    UNSTABLE = "unstable"
    DATA_LEAKAGE = "data_leakage"
    IMBALANCED = "imbalanced"
    LOW_SCORE = "low_score"
    HIGH_VARIANCE = "high_variance"
    RETRAIN_DEGRADATION = "retrain_degradation"
    OVERFITTING_RISK = "overfitting_risk"


@dataclass
class Issue:
    type: IssueType
    severity: str  # "low", "medium", "high", "critical"
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FittingVerdict:
    rating: QualityRating
    score: float  # 0.0 to 1.0
    issues: List[Issue]
    recommendations: List[str]
    summary: str


@dataclass
class ModelInfo:
    model_type: str
    task_type: TaskType
    params: Dict[str, Any]
    model_class: Optional[str] = None


@dataclass
class DataInfo:
    n_samples: int
    n_features: int
    feature_names: Optional[List[str]] = None
    target_name: Optional[str] = None
    train_size: int = 0
    val_size: int = 0
    test_size: int = 0
    has_retrain: bool = False
    retrain_events: int = 0


class FittingJudge:
    """
    Evaluates model fitting quality based on metrics, model info, and data info.
    
    Provides both a quantitative score and qualitative assessment with
    specific issues and recommendations.
    """
    
    THRESHOLDS = {
        "excellent": 0.85,
        "good": 0.70,
        "acceptable": 0.55,
        "poor": 0.40,
    }
    
    OVERFITTING_THRESHOLD = 0.15
    UNDERFITTING_THRESHOLD = 0.50
    HIGH_VARIANCE_THRESHOLD = 0.10
    
    def __init__(
        self,
        metrics: ModelMetrics,
        model_info: ModelInfo,
        data_info: DataInfo,
        config: Optional[TrainingConfig] = None,
        custom_thresholds: Optional[Dict[str, float]] = None,
    ):
        self.metrics = metrics
        self.model_info = model_info
        self.data_info = data_info
        self.config = config
        self.issues: List[Issue] = []
        
        if custom_thresholds:
            self.THRESHOLDS.update(custom_thresholds)
    
    def evaluate(self) -> FittingVerdict:
        issues: List[Issue] = []
        
        issues.extend(self._check_overfitting())
        issues.extend(self._check_underfitting())
        issues.extend(self._check_score_levels())
        issues.extend(self._check_stability())
        issues.extend(self._check_data_quality())
        issues.extend(self._check_retrain_quality())
        
        score = self._calculate_overall_score(issues)
        rating = self._determine_rating(score)
        recommendations = self._generate_recommendations(issues)
        summary = self._generate_summary(rating, score, issues)
        
        return FittingVerdict(
            rating=rating,
            score=score,
            issues=issues,
            recommendations=recommendations,
            summary=summary
        )
    
    def _check_overfitting(self) -> List[Issue]:
        issues = []
        train_score = self.metrics.train_score
        val_score = self.metrics.val_score
        test_score = self.metrics.test_score
        
        if val_score is not None:
            gap = train_score - val_score
            if gap > self.OVERFITTING_THRESHOLD:
                severity = "critical" if gap > 0.30 else "high" if gap > 0.20 else "medium"
                issues.append(Issue(
                    type=IssueType.OVERFITTING,
                    severity=severity,
                    message=f"Training-validation gap ({gap:.2%}) indicates overfitting",
                    details={"train_val_gap": gap, "train_score": train_score, "val_score": val_score}
                ))
        
        if test_score is not None:
            gap = train_score - test_score
            if gap > self.OVERFITTING_THRESHOLD:
                severity = "critical" if gap > 0.30 else "high" if gap > 0.20 else "medium"
                issues.append(Issue(
                    type=IssueType.OVERFITTING,
                    severity=severity,
                    message=f"Training-test gap ({gap:.2%}) indicates overfitting",
                    details={"train_test_gap": gap, "train_score": train_score, "test_score": test_score}
                ))
        
        return issues
    
    def _check_underfitting(self) -> List[Issue]:
        issues = []
        train_score = self.metrics.train_score
        
        if train_score < self.UNDERFITTING_THRESHOLD:
            issues.append(Issue(
                type=IssueType.UNDERFITTING,
                severity="high" if train_score < 0.30 else "medium",
                message=f"Low training score ({train_score:.2%}) indicates underfitting",
                details={"train_score": train_score}
            ))
        
        return issues
    
    def _check_score_levels(self) -> List[Issue]:
        issues = []
        test_score = self.metrics.test_score
        val_score = self.metrics.val_score
        
        if test_score is not None and test_score < self.THRESHOLDS["poor"]:
            issues.append(Issue(
                type=IssueType.LOW_SCORE,
                severity="critical" if test_score < 0.25 else "high",
                message=f"Test score ({test_score:.2%}) is too low",
                details={"test_score": test_score}
            ))
        
        if val_score is not None and val_score < self.THRESHOLDS["poor"]:
            issues.append(Issue(
                type=IssueType.LOW_SCORE,
                severity="high" if val_score < 0.30 else "medium",
                message=f"Validation score ({val_score:.2%}) is too low",
                details={"val_score": val_score}
            ))
        
        return issues
    
    def _check_stability(self) -> List[Issue]:
        issues = []
        
        if not self.metrics.retrain_history:
            return issues
        
        scores = []
        for event in self.metrics.retrain_history:
            test_chunk = event.get("test_chunk_metrics", {})
            if "score" in test_chunk:
                scores.append(test_chunk["score"])
        
        if len(scores) < 2:
            return issues
        
        variance = np.var(scores)
        std = np.std(scores)
        mean_score = np.mean(scores)
        
        if std > self.HIGH_VARIANCE_THRESHOLD:
            issues.append(Issue(
                type=IssueType.HIGH_VARIANCE,
                severity="medium" if std < 0.15 else "high",
                message=f"High variance in retrain scores (std={std:.4f})",
                details={"variance": variance, "std": std, "mean": mean_score, "scores": scores}
            ))
        
        if len(scores) > 3:
            recent_scores = scores[-3:]
            early_scores = scores[:3]
            recent_mean = np.mean(recent_scores)
            early_mean = np.mean(early_scores)
            
            if recent_mean < early_mean - 0.05:
                issues.append(Issue(
                    type=IssueType.RETRAIN_DEGRADATION,
                    severity="medium",
                    message="Model performance degrading over retraining events",
                    details={"early_mean": early_mean, "recent_mean": recent_mean, "degradation": early_mean - recent_mean}
                ))
        
        return issues
    
    def _check_data_quality(self) -> List[Issue]:
        issues = []
        
        if self.data_info.n_features < 3:
            issues.append(Issue(
                type=IssueType.LOW_SCORE,
                severity="low",
                message=f"Very few features ({self.data_info.n_features}) may limit model capacity",
                details={"n_features": self.data_info.n_features}
            ))
        
        if self.data_info.train_size < 100:
            issues.append(Issue(
                type=IssueType.UNSTABLE,
                severity="high",
                message=f"Small training set ({self.data_info.train_size} samples) may cause instability",
                details={"train_size": self.data_info.train_size}
            ))
        
        if self.data_info.n_features > self.data_info.train_size // 10:
            issues.append(Issue(
                type=IssueType.OVERFITTING,
                severity="medium",
                message="High feature-to-sample ratio may cause overfitting",
                details={"n_features": self.data_info.n_features, "train_size": self.data_info.train_size}
            ))
        
        return issues
    
    def _check_retrain_quality(self) -> List[Issue]:
        issues = []
        
        if not self.metrics.retrain_history:
            return issues
        
        window_sizes = [e.get("train_size", 0) for e in self.metrics.retrain_history]
        
        if window_sizes and self.config:
            if self.config.retrain_mode.value == "sliding":
                expected = self.config.sliding_window_size
                if expected and not all(s == expected for s in window_sizes):
                    issues.append(Issue(
                        type=IssueType.UNSTABLE,
                        severity="low",
                        message="Inconsistent sliding window sizes during retraining",
                        details={"expected": expected, "actual": window_sizes}
                    ))
        
        return issues
    
    def _calculate_overall_score(self, issues: List[Issue]) -> float:
        base_score = self.metrics.test_score or self.metrics.val_score or self.metrics.train_score
        
        penalties = {
            "critical": 0.25,
            "high": 0.15,
            "medium": 0.08,
            "low": 0.03,
        }
        
        total_penalty = 0.0
        for issue in issues:
            total_penalty += penalties.get(issue.severity, 0)
        
        return max(0.0, min(1.0, base_score - total_penalty))
    
    def _determine_rating(self, score: float) -> QualityRating:
        if score >= self.THRESHOLDS["excellent"]:
            return QualityRating.EXCELLENT
        elif score >= self.THRESHOLDS["good"]:
            return QualityRating.GOOD
        elif score >= self.THRESHOLDS["acceptable"]:
            return QualityRating.ACCEPTABLE
        elif score >= self.THRESHOLDS["poor"]:
            return QualityRating.POOR
        else:
            return QualityRating.FAILED
    
    def _generate_recommendations(self, issues: List[Issue]) -> List[str]:
        recommendations = []
        issue_types = {i.type for i in issues}
        
        if IssueType.OVERFITTING in issue_types:
            recommendations.extend([
                "Reduce model complexity (fewer trees, lower depth)",
                "Increase regularization (higher min_samples_leaf, lower learning_rate)",
                "Add dropout or use early stopping",
                "Reduce number of features or use feature selection",
                "Increase training data size if possible",
            ])
        
        if IssueType.UNDERFITTING in issue_types:
            recommendations.extend([
                "Increase model complexity (more trees, higher depth)",
                "Reduce regularization",
                "Add more features or create feature interactions",
                "Try a more powerful model type",
                "Check for data quality issues",
            ])
        
        if IssueType.HIGH_VARIANCE in issue_types:
            recommendations.extend([
                "Use expanding window instead of sliding window",
                "Increase retrain period to reduce noise",
                "Ensemble multiple models",
                "Add more stable features",
            ])
        
        if IssueType.LOW_SCORE in issue_types:
            recommendations.extend([
                "Review target definition - is it predictable?",
                "Check for data leakage or label errors",
                "Try different feature sets",
                "Consider if the task is achievable",
            ])
        
        if IssueType.RETRAIN_DEGRADATION in issue_types:
            recommendations.extend([
                "Investigate concept drift in the data",
                "Use adaptive learning rates",
                "Consider online learning approaches",
            ])
        
        for issue in issues:
            if issue.severity == "critical" and "Small training set" in issue.message:
                recommendations.append("Collect more training data before proceeding")
        
        seen = set()
        return [r for r in recommendations if not (r in seen or seen.add(r))]
    
    def _generate_summary(self, rating: QualityRating, score: float, issues: List[Issue]) -> str:
        critical = [i for i in issues if i.severity == "critical"]
        high = [i for i in issues if i.severity == "high"]
        
        parts = [f"Fitting Quality: {rating.value.upper()} (score: {score:.2%})"]
        
        if critical:
            parts.append(f"CRITICAL ISSUES: {len(critical)}")
            for issue in critical[:2]:
                parts.append(f"  - {issue.message}")
        
        if high:
            parts.append(f"HIGH SEVERITY ISSUES: {len(high)}")
            for issue in high[:2]:
                parts.append(f"  - {issue.message}")
        
        if not critical and not high:
            parts.append("No critical or high severity issues detected.")
        
        return "\n".join(parts)


def generate_ai_analysis_prompt(
    metrics: ModelMetrics,
    model_info: ModelInfo,
    data_info: DataInfo,
    config: Optional[TrainingConfig] = None,
    additional_context: Optional[str] = None,
) -> str:
    """
    Generate a detailed prompt for AI analysis of fitting results.
    
    Args:
        metrics: ModelMetrics from training
        model_info: ModelInfo with model details
        data_info: DataInfo with dataset details
        config: Optional TrainingConfig
        additional_context: Optional additional context to include
    
    Returns:
        Formatted prompt string for AI analysis
    """
    
    all_metrics = metrics.get_all_metrics()
    
    prompt_parts = [
        "# ML Model Fitting Analysis Request",
        "",
        "Please analyze the following model fitting results and provide:",
        "1. **Quality Assessment**: Overall rating (Excellent/Good/Acceptable/Poor/Failed)",
        "2. **Issue Detection**: Identify any overfitting, underfitting, or stability problems",
        "3. **Root Cause Analysis**: What is likely causing any issues?",
        "4. **Recommendations**: Specific, actionable suggestions to improve the model",
        "5. **Risk Assessment**: Any concerns about using this model in production?",
        "",
        "---",
        "",
        "## Model Information",
        "",
        f"- **Model Type**: {model_info.model_type}",
        f"- **Task Type**: {model_info.task_type.value}",
        f"- **Model Class**: {model_info.model_class or 'N/A'}",
        "",
        "### Model Parameters",
        "```",
    ]
    
    for key, value in model_info.params.items():
        prompt_parts.append(f"{key}: {value}")
    
    prompt_parts.extend([
        "```",
        "",
        "## Data Information",
        "",
        f"- **Total Samples**: {data_info.n_samples}",
        f"- **Number of Features**: {data_info.n_features}",
        f"- **Training Size**: {data_info.train_size}",
        f"- **Validation Size**: {data_info.val_size}",
        f"- **Test Size**: {data_info.test_size}",
        f"- **Has Retraining**: {data_info.has_retrain}",
        f"- **Retrain Events**: {data_info.retrain_events}",
    ])
    
    if data_info.target_name:
        prompt_parts.append(f"- **Target**: {data_info.target_name}")
    
    if data_info.feature_names and len(data_info.feature_names) <= 20:
        prompt_parts.append("")
        prompt_parts.append("### Feature Names")
        prompt_parts.append("```")
        prompt_parts.extend(data_info.feature_names)
        prompt_parts.append("```")
    
    prompt_parts.extend([
        "",
        "## Training Configuration",
        "",
    ])
    
    if config:
        prompt_parts.extend([
            f"- **Mode**: {config.mode.value}",
            f"- **Train Ratio**: {config.train_ratio}",
            f"- **Val Ratio**: {config.val_ratio if hasattr(config, 'val_ratio') else 'N/A'}",
            f"- **Normalization**: {config.normalization}",
            f"- **Retrain Period**: {config.retrain_period or 'N/A'}",
            f"- **Retrain Mode**: {config.retrain_mode.value if hasattr(config, 'retrain_mode') else 'N/A'}",
        ])
    else:
        prompt_parts.append("_No configuration provided_")
    
    prompt_parts.extend([
        "",
        "## Metrics Summary",
        "",
        "### Training Set",
        "```",
    ])
    
    for key, value in all_metrics["train"].items():
        prompt_parts.append(f"{key}: {value:.6f}")
    
    prompt_parts.extend(["```", ""])
    
    if all_metrics["val"]:
        prompt_parts.extend([
            "### Validation Set",
            "```",
        ])
        for key, value in all_metrics["val"].items():
            prompt_parts.append(f"{key}: {value:.6f}")
        prompt_parts.extend(["```", ""])
    
    prompt_parts.extend([
        "### Test Set",
        "```",
    ])
    
    for key, value in all_metrics["test"].items():
        prompt_parts.append(f"{key}: {value:.6f}")
    
    prompt_parts.extend(["```", ""])
    
    if metrics.retrain_history:
        prompt_parts.extend([
            "## Retraining History",
            "",
            f"Total retraining events: {len(metrics.retrain_history)}",
            "",
            "### Score Progression",
            "| Event | Train Size | Test Score |",
            "|-------|------------|------------|",
        ])
        
        for i, event in enumerate(metrics.retrain_history[:20]):
            test_score = event.get("test_chunk_metrics", {}).get("score", "N/A")
            if isinstance(test_score, float):
                test_score = f"{test_score:.4f}"
            prompt_parts.append(f"| {i+1} | {event.get('train_size', 'N/A')} | {test_score} |")
        
        if len(metrics.retrain_history) > 20:
            prompt_parts.append(f"| ... | ({len(metrics.retrain_history) - 20} more events) | |")
    
    if additional_context:
        prompt_parts.extend([
            "",
            "## Additional Context",
            "",
            additional_context,
        ])
    
    prompt_parts.extend([
        "",
        "---",
        "",
        "## Analysis Template",
        "",
        "Please structure your response as follows:",
        "",
        "### 1. Overall Assessment",
        "- **Rating**: [Excellent/Good/Acceptable/Poor/Failed]",
        "- **Confidence**: [High/Medium/Low]",
        "- **Summary**: [1-2 sentences]",
        "",
        "### 2. Detailed Analysis",
        "",
        "#### Training Dynamics",
        "- Analysis of train/val/test gap",
        "- Signs of overfitting or underfitting",
        "",
        "#### Model Stability",
        "- Analysis of retraining performance (if applicable)",
        "- Variance concerns",
        "",
        "#### Data & Features",
        "- Feature quality assessment",
        "- Data sufficiency concerns",
        "",
        "### 3. Recommendations",
        "1. [Priority recommendation]",
        "2. [Secondary recommendation]",
        "3. [Optional improvement]",
        "",
        "### 4. Production Readiness",
        "- **Ready for Production**: [Yes/No/With caveats]",
        "- **Caveats**: [List any concerns]",
        "- **Monitoring Suggestions**: [What to watch for]",
    ])
    
    return "\n".join(prompt_parts)


def quick_judge(
    metrics: ModelMetrics,
    model_type: str,
    task_type: TaskType,
    n_samples: int,
    n_features: int,
) -> FittingVerdict:
    """
    Quick evaluation with minimal configuration.
    
    Convenience function for simple fitting quality checks.
    """
    model_info = ModelInfo(
        model_type=model_type,
        task_type=task_type,
        params={}
    )
    
    data_info = DataInfo(
        n_samples=n_samples,
        n_features=n_features,
        train_size=int(n_samples * 0.6),
        val_size=int(n_samples * 0.2),
        test_size=int(n_samples * 0.2),
    )
    
    judge = FittingJudge(metrics, model_info, data_info)
    return judge.evaluate()

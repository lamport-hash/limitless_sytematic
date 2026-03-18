from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Union
import csv
from datetime import date
from pathlib import Path
from enum import StrEnum, IntEnum


# ========== ENUMS ==========
class FeatureType(StrEnum):
    PRICE = "price"
    RETURN = "return"
    LOG_RETURN = "log_return"
    VOLUME = "volume"
    HIST_VOLATILITY = "hist_volatility"
    LAG_DELTAS = "lagged_deltas"
    RSI = "rsi"
    EMA = "ema"
    SPREAD_REL_EMA = "spread_rel_ema"
    DIFF_REL_EMA_MID = "diff_rel_ema_mid"
    ADI = "adi"
    RITA = "rita"
    V_USD = "volume_dollar"
    V_OBV = "volume_obv"
    T_BUY_RATIO_B = "trade_buy_ratio_base"
    T_BUY_RATIO_Q = "trade_buy_ratio_quote"
    T_AVG_TS_USD = "trade_avg_trade_size_dollar"
    T_NET_TAKER_FLOW = "trade_net_taker_flow"
    V_MA_RATIO = "volume_ma_ratio"
    V_USD_MA_RATIO = "volume_dollar_ma_ratio"
    T_BUY_MA_RATIO_B = "trade_buy_pressure_ma_ratio_base"
    T_BUY_MA_RATIO_Q = "trade_buy_pressure_ma_ratio_quote"
    T_AVG_TS_USD_MA_RATIO = "trade_avg_trade_size_dollar_ma_ratio"
    T_TC_MA_RATIO = "trade_trade_count_ma_ratio"
    ROC = "roc"
    ROC_CORRECT_MIN = "roc_correct_min"
    DAILY_SIGNAL = "daily_signal"
    TOTAL_SIGNAL = "total_signal"
    CTO_LINE = "cto_line"


class NormalisationType(StrEnum):
    NONE = "none"
    MINMAX = "minmax"
    ZSCORE = "zscore"
    ROBUST = "robust"
    RSI_FORMULA_50_2 = "rsi_formula_50_2"


class NormalisationPeriodType(IntEnum):
    Y2 = 730
    Y1 = 365
    M6 = 180
    M3 = 90
    M1 = 30
    W1 = 7
    D1 = 1


FEATURE_TYPE_TO_NORMALISATION: Dict[
    FeatureType, Tuple[NormalisationType, NormalisationPeriodType, NormalisationPeriodType]
] = {
    FeatureType.PRICE: (
        NormalisationType.MINMAX,
        NormalisationPeriodType.Y1,
        NormalisationPeriodType.W1,
    ),
    FeatureType.RETURN: (
        NormalisationType.ZSCORE,
        NormalisationPeriodType.Y1,
        NormalisationPeriodType.W1,
    ),
    FeatureType.LOG_RETURN: (
        NormalisationType.ZSCORE,
        NormalisationPeriodType.Y1,
        NormalisationPeriodType.W1,
    ),
    FeatureType.VOLUME: (
        NormalisationType.ZSCORE,
        NormalisationPeriodType.Y1,
        NormalisationPeriodType.W1,
    ),
    FeatureType.HIST_VOLATILITY: (
        NormalisationType.ROBUST,
        NormalisationPeriodType.Y1,
        NormalisationPeriodType.W1,
    ),
    FeatureType.LAG_DELTAS: (
        NormalisationType.ZSCORE,
        NormalisationPeriodType.M6,
        NormalisationPeriodType.D1,
    ),
    FeatureType.RSI: (
        NormalisationType.RSI_FORMULA_50_2,
        NormalisationPeriodType.Y2,
        NormalisationPeriodType.D1,
    ),
    FeatureType.EMA: (
        NormalisationType.ZSCORE,
        NormalisationPeriodType.M6,
        NormalisationPeriodType.W1,
    ),
    FeatureType.SPREAD_REL_EMA: (
        NormalisationType.ZSCORE,
        NormalisationPeriodType.M6,
        NormalisationPeriodType.W1,
    ),
    FeatureType.DIFF_REL_EMA_MID: (
        NormalisationType.ZSCORE,
        NormalisationPeriodType.M6,
        NormalisationPeriodType.W1,
    ),
    FeatureType.ADI: (
        NormalisationType.MINMAX,
        NormalisationPeriodType.Y1,
        NormalisationPeriodType.W1,
    ),
    FeatureType.RITA: (
        NormalisationType.ROBUST,
        NormalisationPeriodType.Y1,
        NormalisationPeriodType.W1,
    ),
    FeatureType.V_USD: (
        NormalisationType.ZSCORE,
        NormalisationPeriodType.Y1,
        NormalisationPeriodType.W1,
    ),  # Volume in USD
    FeatureType.V_OBV: (
        NormalisationType.ZSCORE,
        NormalisationPeriodType.Y1,
        NormalisationPeriodType.W1,
    ),  # OBV is cumulative, but Z-score works well
    FeatureType.T_BUY_RATIO_B: (
        NormalisationType.MINMAX,
        NormalisationPeriodType.M6,
        NormalisationPeriodType.D1,
    ),  # Ratio between 0-1 → minmax
    FeatureType.T_BUY_RATIO_Q: (
        NormalisationType.MINMAX,
        NormalisationPeriodType.M6,
        NormalisationPeriodType.D1,
    ),  # Same logic
    FeatureType.T_AVG_TS_USD: (
        NormalisationType.ROBUST,
        NormalisationPeriodType.M6,
        NormalisationPeriodType.W1,
    ),  # Heavy tails in trade size → robust
    FeatureType.T_NET_TAKER_FLOW: (
        NormalisationType.ZSCORE,
        NormalisationPeriodType.M6,
        NormalisationPeriodType.W1,
    ),  # Mean-centered flow
    FeatureType.V_MA_RATIO: (
        NormalisationType.ZSCORE,
        NormalisationPeriodType.M6,
        NormalisationPeriodType.W1,
    ),  # Ratio of volume to MA → zscore appropriate
    FeatureType.V_USD_MA_RATIO: (
        NormalisationType.ZSCORE,
        NormalisationPeriodType.M6,
        NormalisationPeriodType.W1,
    ),  # Same as above
    FeatureType.T_BUY_MA_RATIO_B: (
        NormalisationType.ZSCORE,
        NormalisationPeriodType.M6,
        NormalisationPeriodType.W1,
    ),  # Buy pressure ratio → zscore
    FeatureType.T_BUY_MA_RATIO_Q: (
        NormalisationType.ZSCORE,
        NormalisationPeriodType.M6,
        NormalisationPeriodType.W1,
    ),  # Buy pressure ratio → zscore
    FeatureType.T_AVG_TS_USD_MA_RATIO: (
        NormalisationType.ZSCORE,
        NormalisationPeriodType.M6,
        NormalisationPeriodType.W1,
    ),  # Moving ratio → zscore
    FeatureType.T_TC_MA_RATIO: (
        NormalisationType.ZSCORE,
        NormalisationPeriodType.M6,
        NormalisationPeriodType.W1,
    ),  # Trade count ratio → zscore
    FeatureType.ROC: (
        NormalisationType.NONE,
        NormalisationPeriodType.Y1,
        NormalisationPeriodType.W1,
    ),
    FeatureType.ROC_CORRECT_MIN: (
        NormalisationType.NONE,
        NormalisationPeriodType.Y1,
        NormalisationPeriodType.W1,
    ),
    FeatureType.DAILY_SIGNAL: (
        NormalisationType.NONE,
        NormalisationPeriodType.Y1,
        NormalisationPeriodType.W1,
    ),
    FeatureType.TOTAL_SIGNAL: (
        NormalisationType.NONE,
        NormalisationPeriodType.Y1,
        NormalisationPeriodType.W1,
    ),
    FeatureType.CTO_LINE: (
        NormalisationType.NONE,
        NormalisationPeriodType.Y1,
        NormalisationPeriodType.W1,
    ),
}


# ========== DATA CLASS ==========
@dataclass
class Feature:
    """
    Represents a feature with normalization metadata.
    Automatically sets normalisation_type, start period and rescale frequency
    based on feature_type if not provided.
    """

    feature_id: str
    activated: bool
    feature_type: FeatureType

    # Optional fields for overrides — populated from defaults if not provided
    normalisation_type: Optional[NormalisationType] = None
    normalisation_first_period: Optional[NormalisationPeriodType] = None
    normalisation_rescale_frequency: Optional[NormalisationPeriodType] = None

    def __post_init__(self):
        """Auto-populate normalization metadata based on feature_type if not provided."""
        if self.feature_type not in FEATURE_TYPE_TO_NORMALISATION:
            raise ValueError(
                f"Unknown feature_type: {self.feature_type}. "
                f"Supported types: {list(FEATURE_TYPE_TO_NORMALISATION.keys())}"
            )

        default_norm, default_start, default_freq = FEATURE_TYPE_TO_NORMALISATION[self.feature_type]

        # Use provided values if not None, else use defaults
        self.normalisation_type = self.normalisation_type or default_norm
        self.normalisation_first_period = self.normalisation_first_period or default_start
        self.normalisation_rescale_frequency = self.normalisation_rescale_frequency or default_freq

    def __repr__(self):
        return (
            f"Feature("
            f"id='{self.feature_id}', "
            f"activated={self.activated}, "
            f"type='{self.feature_type}', "
            f"norm_type='{self.normalisation_type}', "
            f"start_period={self.normalisation_first_period}, "
            f"rescale_freq={self.normalisation_rescale_frequency})"
        )

    @classmethod
    def from_csv(cls, csv_str: str) -> List["Feature"]:
        """
        Deserialize a list of Feature objects from CSV string.
        Header expected: feature_id,activated,feature_type,normalisation_type,normalisation_first_period,normalisation_rescale_frequency
        """
        reader = csv.DictReader(csv_str.splitlines())
        features = []
        for row in reader:
            feature_type = FeatureType(row["feature_type"])
            activated = row["activated"].lower() in ("true", "1", "yes", "on")
            normalisation_type = (
                NormalisationType(row["normalisation_type"]) if row["normalisation_type"] else None
            )
            normalisation_first_period = (
                NormalisationPeriodType(int(row["normalisation_first_period"]))
                if row["normalisation_first_period"]
                else None
            )
            normalisation_rescale_frequency = (
                NormalisationPeriodType(int(row["normalisation_rescale_frequency"]))
                if row["normalisation_rescale_frequency"]
                else None
            )

            feature = cls(
                feature_id=row["feature_id"],
                activated=activated,
                feature_type=feature_type,
                normalisation_type=normalisation_type,
                normalisation_first_period=normalisation_first_period,
                normalisation_rescale_frequency=normalisation_rescale_frequency,
            )
            features.append(feature)
        return features

    @classmethod
    def from_file(cls, filepath: Union[str, Path]) -> List["Feature"]:
        """Load Feature objects from a CSV file."""
        with open(filepath, "r", newline="", encoding="utf-8") as f:
            return cls.from_csv(f.read())

    def to_dict(self) -> Dict[str, Union[str, bool, int]]:
        """Convert Feature to dict for CSV writing."""
        return {
            "feature_id": self.feature_id,
            "activated": str(self.activated).lower(),
            "feature_type": self.feature_type.value,
            "normalisation_type": self.normalisation_type.value if self.normalisation_type else "",
            "normalisation_first_period": str(int(self.normalisation_first_period))
            if self.normalisation_first_period
            else "",
            "normalisation_rescale_frequency": str(int(self.normalisation_rescale_frequency))
            if self.normalisation_rescale_frequency
            else "",
        }

    @classmethod
    def to_csv(cls, features: List["Feature"], filepath: Union[str, Path]) -> None:
        """Serialize a list of Feature objects to CSV file."""
        if not features:
            return

        fieldnames = [
            "feature_id",
            "activated",
            "feature_type",
            "normalisation_type",
            "normalisation_first_period",
            "normalisation_rescale_frequency",
        ]

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for feature in features:
                writer.writerow(feature.to_dict())

    def to_csv_row(self) -> str:
        """Return a single feature as CSV row string (for manual use)."""
        return ",".join(self.to_dict().values())

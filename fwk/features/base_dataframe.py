import pandas as pd
import numpy as np
import logging
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union

from ruamel.yaml import YAML

from features.features_utils import Feature, FeatureType
from features.feature_ta_utils import (
    numba_rsi,
    numba_ema,
    rolling_std,
    add_accumulation_distribution_index,
    numba_roc,
)
from features.f_order_flow import (
    feature_volume_dollar,
    feature_volume_obv,
    feature_trade_buy_ratio_base,
    feature_trade_buy_ratio_quote,
    feature_trade_avg_trade_size_dollar,
    feature_trade_net_taker_flow,
    feature_volume_ma_ratio,
    feature_volume_dollar_ma_ratio,
    feature_trade_buy_pressure_ma_ratio_base,
    feature_trade_buy_pressure_ma_ratio_quote,
    feature_trade_avg_trade_size_dollar_ma_ratio,
    feature_trade_trade_count_ma_ratio,
)
from features.f_daily_signal import (
    feature_daily_signal,
    feature_daily_signal_with_exit,
    feature_pointpos,
)
from features.f_total_signal import feature_total_signal
from features.f_cto_line import feature_cto_line_signal
from merger.merger_utils import (
    convert_df_cols_float_to_float,
    scale_dataframe_columns,
)
from norm.norm_utils import add_minutes_since_2000

from core.enums import (
    g_close_col,
    g_open_col,
    g_high_col,
    g_low_col,
    g_volume_col,
    g_mid_col,
    g_mid2_col,
    g_close_time_col,
    g_open_time_col,
    g_index_col,
    g_qa_vol_col,
    g_nt_col,
    g_la_vol_col,
    g_lqa_vol_col,
    g_precision,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

G_SCALING_DICT = {"BTC": 30, "SOL": 3, "ETH": 12.69, "ADA": 0.01, "DOT": 0.1, "LINK": 0.1, "BNB": 5}


def resample_ohlcv(df: pd.DataFrame, n: int, p_valid_col_name: str) -> pd.DataFrame:
    base = pd.Timestamp("2000-01-01")
    df = df.copy()
    df["datetime"] = base + pd.to_timedelta(df[g_index_col], unit="m")
    df.set_index("datetime", inplace=True)

    if g_close_col in df.columns:
        ohlcv_dict = {
            g_open_col: "first",
            g_high_col: "max",
            g_low_col: "min",
            g_close_col: "last",
            g_close_time_col: "last",
            g_open_time_col: "first",
            g_volume_col: "sum",
            g_qa_vol_col: "sum",
            g_nt_col: "sum",
            g_la_vol_col: "sum",
            g_lqa_vol_col: "sum",
            p_valid_col_name: "max",
            g_index_col: "last",
        }
    else:
        ohlcv_dict = {
            g_open_col: "first",
            g_high_col: "max",
            g_low_col: "min",
            g_close_col: "last",
            g_volume_col: "sum",
            g_index_col: "last",
            g_close_time_col: "last",
        }

    resampled_df = df.resample(f"{n}min").apply(ohlcv_dict).dropna(how="any")
    resampled_df.reset_index(inplace=True)
    resampled_df[g_index_col] = resampled_df[g_index_col].astype("Int64")
    return resampled_df


def convert_columns_to_float(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for column in columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            continue
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


class BaseDataFrame:
    """
    A class to build and manage a base dataframe with OHLCV data and feature additions.

    This class provides a modular approach to adding features based on FeatureType enum.
    """

    def __init__(
        self,
        p_df: pd.DataFrame,
        p_valid_col_name: str = "valid_row",
        p_last_n_rows: int = -1,
        p_resampling_period: int = 1,
        p_scaling: float = -1,
        p_scaling_fixed: bool = False,
        p_scaling_asset: str = "",
        p_verbose: bool = False,
    ):
        self.df: pd.DataFrame = p_df.copy()
        self.valid_col_name: str = p_valid_col_name
        self.last_n_rows: int = p_last_n_rows
        self.resampling_period: int = p_resampling_period
        self.scaling: float = p_scaling
        self.scaling_fixed: bool = p_scaling_fixed
        self.scaling_asset: str = p_scaling_asset
        self.verbose: bool = p_verbose
        self.factor: float = 1.0
        self.mid_col: str = g_mid_col

        self.selected_columns: List[str] = []
        self.selected_features: List[Feature] = []
        self.selected_utils_columns: List[str] = []

        self.prepare_base_dataframe()

    def prepare_base_dataframe(self) -> None:
        """Prepare the base dataframe with gap detection, scaling, and base features."""
        logger.info("******")
        logger.info("STARTING BaseDataFrame preparation")

        if self.last_n_rows != -1:
            self.df = self.df[-self.last_n_rows :].copy()

        if g_index_col not in self.df.columns:
            self.df = add_minutes_since_2000(self.df, g_close_time_col, g_index_col)
        self.df["minute_diff"] = self.df[g_index_col].diff().fillna(1)
        self.df["gap_flag"] = self.df["minute_diff"] > 1
        self.df[self.valid_col_name] = self.df["gap_flag"] == False
        self.selected_utils_columns = [self.valid_col_name]

        volume_cols = [g_volume_col, g_qa_vol_col, g_lqa_vol_col, g_la_vol_col]
        existing_vol_cols = [c for c in volume_cols if c in self.df.columns]
        if existing_vol_cols:
            self.df = convert_columns_to_float(self.df, existing_vol_cols)

        if self.resampling_period != 1:
            self.df = resample_ohlcv(self.df, self.resampling_period, self.valid_col_name)

        self._scale_ohlcv_columns()
        self._compute_base_features()

    def _scale_ohlcv_columns(self) -> None:
        average_val = self.df[g_close_col][:1000].mean()
        self.factor = max(round(float(average_val / 100.0), 3), 0.001)

        if self.scaling_fixed:
            self.factor = G_SCALING_DICT[self.scaling_asset]

        if self.scaling != -1:
            print(f"using scaling for {self.scaling_asset} scaling is :{self.scaling}")
            self.factor = self.scaling

        self.df["factor_f32"] = self.factor

        cols_to_scale = [g_open_col, g_high_col, g_low_col, g_close_col]
        self.df = scale_dataframe_columns(self.df, cols_to_scale, float(1.0 / self.factor))
        f32_cols = [col for col in self.df.columns if col.endswith("_f32")]
        self.df = convert_df_cols_float_to_float(self.df, f32_cols, "32", [])

        if g_lqa_vol_col in self.df.columns:
            self.df[g_lqa_vol_col] = self.df[g_lqa_vol_col] / self.factor
        if g_qa_vol_col in self.df.columns:
            self.df[g_qa_vol_col] = self.df[g_qa_vol_col] / self.factor

    def _compute_base_features(self) -> None:
        self.df[g_mid_col] = (self.df[g_high_col] + self.df[g_low_col]) / 2.0
        self.df[g_mid2_col] = (self.df[g_open_col] + self.df[g_close_col]) / 2.0

        self.df["F_LR_mid_f16"] = np.log(self.df[g_mid_col] / self.df[g_mid_col].shift(1))
        self.df["F_LR_close_f16"] = np.log(self.df[g_close_col] / self.df[g_close_col].shift(1))
        self.df["F_LR_mid2_f16"] = np.log(self.df[g_mid2_col] / self.df[g_mid2_col].shift(1))
        self.df["F_range_f16"] = (self.df[g_high_col] - self.df[g_low_col]) / self.df[g_mid_col]

        # Replace inf/-inf with nan to prevent JSON serialization issues
        for col in ["F_LR_mid_f16", "F_LR_close_f16", "F_LR_mid2_f16", "F_range_f16"]:
            self.df[col] = self.df[col].replace([np.inf, -np.inf], np.nan)

        self.selected_columns += ["F_LR_close_f16", "F_LR_mid_f16", "F_LR_mid2_f16", "F_range_f16"]

    def add_feature(
        self,
        feature_type: FeatureType,
        periods: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> "BaseDataFrame":
        """
        Add a feature of the specified type to the dataframe.

        Args:
            feature_type (FeatureType): The type of feature to add.
            periods (Optional[List[int]]): Periods for features that require them.
            **kwargs: Additional parameters for specific features.

        Returns:
            BaseDataFrame: Self for method chaining.
        """
        feature_methods = {
            FeatureType.PRICE: self._add_price,
            FeatureType.RETURN: self._add_return,
            FeatureType.LOG_RETURN: self._add_log_return,
            FeatureType.VOLUME: self._add_volume,
            FeatureType.HIST_VOLATILITY: self._add_hist_volatility,
            FeatureType.LAG_DELTAS: self._add_lag_deltas,
            FeatureType.RSI: self._add_rsi,
            FeatureType.EMA: self._add_ema,
            FeatureType.SPREAD_REL_EMA: self._add_spread_rel_ema,
            FeatureType.DIFF_REL_EMA_MID: self._add_diff_rel_ema_mid,
            FeatureType.ADI: self._add_adi,
            FeatureType.RITA: self._add_rita,
            FeatureType.V_USD: self._add_volume_dollar,
            FeatureType.V_OBV: self._add_volume_obv,
            FeatureType.T_BUY_RATIO_B: self._add_trade_buy_ratio_base,
            FeatureType.T_BUY_RATIO_Q: self._add_trade_buy_ratio_quote,
            FeatureType.T_AVG_TS_USD: self._add_trade_avg_trade_size_dollar,
            FeatureType.T_NET_TAKER_FLOW: self._add_trade_net_taker_flow,
            FeatureType.V_MA_RATIO: self._add_volume_ma_ratio,
            FeatureType.V_USD_MA_RATIO: self._add_volume_dollar_ma_ratio,
            FeatureType.T_BUY_MA_RATIO_B: self._add_trade_buy_pressure_ma_ratio_base,
            FeatureType.T_BUY_MA_RATIO_Q: self._add_trade_buy_pressure_ma_ratio_quote,
            FeatureType.T_AVG_TS_USD_MA_RATIO: self._add_trade_avg_trade_size_dollar_ma_ratio,
            FeatureType.T_TC_MA_RATIO: self._add_trade_trade_count_ma_ratio,
            FeatureType.ROC: self._add_roc,
            FeatureType.DAILY_SIGNAL: self._add_daily_signal,
            FeatureType.TOTAL_SIGNAL: self._add_total_signal,
            FeatureType.CTO_LINE: self._add_cto_line,
        }

        method = feature_methods.get(feature_type)
        if method:
            method(periods=periods, **kwargs)
        else:
            logger.warning(f"Feature type {feature_type} not implemented yet")

        return self

    def add_features(
        self,
        feature_types: List[FeatureType],
        periods: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> "BaseDataFrame":
        """
        Add multiple features to the dataframe.

        Args:
            feature_types (List[FeatureType]): List of feature types to add.
            periods (Optional[List[int]]): Periods for features that require them.
            **kwargs: Additional parameters for specific features.

        Returns:
            BaseDataFrame: Self for method chaining.
        """
        for feature_type in feature_types:
            self.add_feature(feature_type, periods=periods, **kwargs)
        return self

    def add_features_from_md(self, filepath: Union[str, Path]) -> "BaseDataFrame":
        """
        Add features defined in a markdown file containing a YAML code block.

        The markdown file should contain a YAML code block like:

        ```yaml
        features:
          - type: RSI
            periods: [14, 60, 240]
          - type: EMA
            periods: [15, 60]
          - type: HIST_VOLATILITY
            periods: [15, 60]
          - type: DAILY_SIGNAL
            kwargs:
              p_test_candles: 8
              p_exit_delay: 4
        ```

        Args:
            filepath: Path to .md file with YAML feature configuration.

        Returns:
            BaseDataFrame: Self for method chaining.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If no YAML block found or invalid configuration.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Feature config file not found: {filepath}")

        content = filepath.read_text(encoding="utf-8")

        yaml_block_match = re.search(
            r"```yaml\s*\n(.*?)\n```", content, re.DOTALL | re.IGNORECASE
        )
        if not yaml_block_match:
            raise ValueError(f"No YAML code block found in {filepath}")

        yaml_content = yaml_block_match.group(1)

        yaml = YAML()
        config = yaml.load(yaml_content)

        if not config or "features" not in config:
            raise ValueError(f"No 'features' key found in YAML config in {filepath}")

        features_config = config["features"]
        if not features_config:
            logger.warning(f"Empty features list in {filepath}")
            return self

        for feature_config in features_config:
            if not isinstance(feature_config, dict) or "type" not in feature_config:
                logger.warning(f"Invalid feature config: {feature_config}")
                continue

            try:
                feature_type = FeatureType(feature_config["type"])
            except ValueError:
                logger.warning(
                    f"Unknown FeatureType '{feature_config['type']}' in {filepath}"
                )
                continue

            periods = feature_config.get("periods")
            kwargs = feature_config.get("kwargs", {})

            self.add_feature(feature_type, periods=periods, **kwargs)

        return self

    def _register_feature(
        self, feature_id: str, feature_type: FeatureType, activated: bool = True
    ) -> None:
        """Register a feature in the selected_features list."""
        feature_obj = Feature(
            feature_id=feature_id,
            activated=activated,
            feature_type=feature_type,
        )
        self.selected_features.append(feature_obj)
        self.selected_columns.append(feature_id)

    def _add_price(self, periods: Optional[List[int]] = None, **kwargs: Any) -> None:
        """Price features - reference to source columns."""
        for col in [g_open_col, g_high_col, g_low_col, g_close_col, g_mid_col, g_mid2_col]:
            self._register_feature(col, FeatureType.PRICE)

    def _add_return(self, periods: Optional[List[int]] = None, **kwargs: Any) -> None:
        """Simple returns."""
        name = "F_return_close_f16"
        self.df[name] = self.df[g_close_col].pct_change()
        self._register_feature(name, FeatureType.RETURN)

    def _add_volume(self, periods: Optional[List[int]] = None, **kwargs: Any) -> None:
        """Volume features - reference to source columns."""
        for col in [g_volume_col, g_qa_vol_col, g_la_vol_col, g_lqa_vol_col]:
            self._register_feature(col, FeatureType.VOLUME)

    def _add_log_return(self, periods: Optional[List[int]] = None, **kwargs: Any) -> None:
        """Log returns are already computed in base features."""
        pass

    def _add_hist_volatility(self, periods: Optional[List[int]] = None, **kwargs: Any) -> None:
        periods = periods or [15, 60]
        new_cols = {}
        for period in periods:
            name = f"F_vol_{self.mid_col}_{period}_f16"
            new_cols[name] = rolling_std(self.df[self.mid_col].to_numpy(), period)
            self._register_feature(name, FeatureType.HIST_VOLATILITY)
        self.df = pd.concat([self.df, pd.DataFrame(new_cols, index=self.df.index)], axis=1)

    def _add_lag_deltas(
        self,
        periods: Optional[List[int]] = None,
        n_lags: int = 72,
        n_minutes: int = 60,
        **kwargs: Any,
    ) -> None:
        new_cols = {}
        for lag in range(1, n_lags + 1):
            lag_minutes = n_minutes * lag
            feature_id = f"F_delta_rel_{self.mid_col}_{lag_minutes}_f16"
            new_cols[feature_id] = (
                self.df[self.mid_col] - self.df[self.mid_col].shift(lag_minutes)
            ) / self.df[self.mid_col].shift(lag_minutes)
            self._register_feature(feature_id, FeatureType.LAG_DELTAS)
        self.df = pd.concat([self.df, pd.DataFrame(new_cols, index=self.df.index)], axis=1)

    def _add_rsi(self, periods: Optional[List[int]] = None, **kwargs: Any) -> None:
        periods = periods or [2, 15, 60, 240, 500, 1440, 2880]
        new_cols = {}
        for period in periods:
            name = f"F_rsi_{period}_{self.mid_col}_f16"
            new_cols[name] = numba_rsi(self.df[self.mid_col].to_numpy(), period)
            self._register_feature(name, FeatureType.RSI)
        self.df = pd.concat([self.df, pd.DataFrame(new_cols, index=self.df.index)], axis=1)

    def _add_roc(self, periods: Optional[List[int]] = None, **kwargs: Any) -> None:
        periods = periods or [14, 60, 240]
        new_cols = {}
        for period in periods:
            name = f"F_roc_{period}_{self.mid_col}_f16"
            new_cols[name] = numba_roc(self.df[self.mid_col].to_numpy(), period)
            self._register_feature(name, FeatureType.ROC)
        self.df = pd.concat([self.df, pd.DataFrame(new_cols, index=self.df.index)], axis=1)

    def _add_ema(self, periods: Optional[List[int]] = None, **kwargs: Any) -> None:
        periods = periods or [2, 15, 60, 240, 500, 1440, 2880]
        new_cols = {}
        for period in periods:
            ema_id = f"F_ema_{period}_{self.mid_col}_f32"
            new_cols[ema_id] = numba_ema(self.df[self.mid_col].to_numpy(), period)
            self._register_feature(ema_id, FeatureType.EMA, activated=False)
        self.df = pd.concat([self.df, pd.DataFrame(new_cols, index=self.df.index)], axis=1)

    def _add_spread_rel_ema(self, periods: Optional[List[int]] = None, **kwargs: Any) -> None:
        periods = periods or [2, 15, 60, 240, 500, 1440, 2880]
        epsilon = 0.0001

        ema_names = [f"F_ema_{period}_{self.mid_col}_f32" for period in periods]
        missing_ema_cols = {}
        for col in ema_names:
            if col not in self.df.columns:
                period = int(col.split("_")[2])
                missing_ema_cols[col] = numba_ema(self.df[self.mid_col].to_numpy(), period)
        if missing_ema_cols:
            self.df = pd.concat([self.df, pd.DataFrame(missing_ema_cols, index=self.df.index)], axis=1)

        new_cols = {}
        for i in range(len(periods) - 1):
            name_id = f"F_delta_rel_ema_{periods[i]}_{periods[i + 1]}_{self.mid_col}_f16"
            new_cols[name_id] = (self.df[ema_names[i + 1]] - self.df[ema_names[i]]) / (
                self.df[ema_names[i]] + epsilon
            )
            self._register_feature(name_id, FeatureType.SPREAD_REL_EMA)
        self.df = pd.concat([self.df, pd.DataFrame(new_cols, index=self.df.index)], axis=1)

    def _add_diff_rel_ema_mid(self, periods: Optional[List[int]] = None, **kwargs: Any) -> None:
        periods = periods or [2, 15, 60, 240, 500, 1440, 2880]
        epsilon = 0.0001

        ema_names = [f"F_ema_{period}_{self.mid_col}_f32" for period in periods]
        missing_ema_cols = {}
        for col in ema_names:
            if col not in self.df.columns:
                period = int(col.split("_")[2])
                missing_ema_cols[col] = numba_ema(self.df[self.mid_col].to_numpy(), period)
        if missing_ema_cols:
            self.df = pd.concat([self.df, pd.DataFrame(missing_ema_cols, index=self.df.index)], axis=1)

        new_cols = {}
        for i in range(len(periods) - 1):
            name_id = f"F_diff_rel_ema_2_{self.mid_col}_{periods[i]}_f16"
            new_cols[name_id] = (self.df[ema_names[i + 1]] - self.df[self.mid_col]) / (
                self.df[self.mid_col] + epsilon
            )
            self._register_feature(name_id, FeatureType.DIFF_REL_EMA_MID)
        self.df = pd.concat([self.df, pd.DataFrame(new_cols, index=self.df.index)], axis=1)

    def _add_adi(self, periods: Optional[List[int]] = None, **kwargs: Any) -> None:
        self.df, colname = add_accumulation_distribution_index(self.df)
        self._register_feature(colname, FeatureType.ADI)

    def _add_rita(self, periods: Optional[List[int]] = None, **kwargs: Any) -> None:
        if g_qa_vol_col not in self.df.columns or g_lqa_vol_col not in self.df.columns:
            logger.warning(f"RITA feature requires {g_qa_vol_col} and {g_lqa_vol_col} columns")
            return
        given_volume = pd.to_numeric(self.df[g_qa_vol_col]) - pd.to_numeric(self.df[g_lqa_vol_col])
        rita_colname = "F_RITA_f64"
        self.df[rita_colname] = (pd.to_numeric(self.df[g_lqa_vol_col]) - given_volume).cumsum()
        self._register_feature(rita_colname, FeatureType.RITA)

    def _add_volume_dollar(self, periods: Optional[List[int]] = None, **kwargs: Any) -> None:
        colname = "F_volume_dollar_f64"
        self.df[colname] = feature_volume_dollar(self.df)
        self._register_feature(colname, FeatureType.V_USD)

    def _add_volume_obv(self, periods: Optional[List[int]] = None, **kwargs: Any) -> None:
        colname = "F_volume_obv_f64"
        self.df[colname] = feature_volume_obv(self.df)
        self._register_feature(colname, FeatureType.V_OBV)

    def _add_trade_buy_ratio_base(self, periods: Optional[List[int]] = None, **kwargs: Any) -> None:
        colname = "F_trade_buy_ratio_base_f64"
        self.df[colname] = feature_trade_buy_ratio_base(self.df)
        self._register_feature(colname, FeatureType.T_BUY_RATIO_B)

    def _add_trade_buy_ratio_quote(
        self, periods: Optional[List[int]] = None, **kwargs: Any
    ) -> None:
        colname = "F_trade_buy_ratio_quote_f64"
        self.df[colname] = feature_trade_buy_ratio_quote(self.df)
        self._register_feature(colname, FeatureType.T_BUY_RATIO_Q)

    def _add_trade_avg_trade_size_dollar(
        self, periods: Optional[List[int]] = None, **kwargs: Any
    ) -> None:
        colname = "F_trade_avg_trade_size_dollar_f64"
        self.df[colname] = feature_trade_avg_trade_size_dollar(self.df)
        self._register_feature(colname, FeatureType.T_AVG_TS_USD)

    def _add_trade_net_taker_flow(self, periods: Optional[List[int]] = None, **kwargs: Any) -> None:
        colname = "F_trade_net_taker_flow_f64"
        self.df[colname] = feature_trade_net_taker_flow(self.df)
        self._register_feature(colname, FeatureType.T_NET_TAKER_FLOW)

    def _add_volume_ma_ratio(self, periods: Optional[List[int]] = None, **kwargs: Any) -> None:
        periods = periods or [15, 120]
        for window in periods:
            colname = f"F_volume_ma_ratio_{window}_f64"
            self.df[colname] = feature_volume_ma_ratio(self.df, p_window=window)
            self._register_feature(colname, FeatureType.V_MA_RATIO)

    def _add_volume_dollar_ma_ratio(
        self, periods: Optional[List[int]] = None, **kwargs: Any
    ) -> None:
        periods = periods or [15, 120]
        for window in periods:
            colname = f"F_volume_dollar_ma_ratio_{window}_f64"
            self.df[colname] = feature_volume_dollar_ma_ratio(self.df, p_window=window)
            self._register_feature(colname, FeatureType.V_USD_MA_RATIO)

    def _add_trade_buy_pressure_ma_ratio_base(
        self, periods: Optional[List[int]] = None, **kwargs: Any
    ) -> None:
        periods = periods or [15, 120]
        for window in periods:
            colname = f"F_trade_buy_pressure_ma_ratio_base_{window}_f64"
            self.df[colname] = feature_trade_buy_pressure_ma_ratio_base(self.df, p_window=window)
            self._register_feature(colname, FeatureType.T_BUY_MA_RATIO_B)

    def _add_trade_buy_pressure_ma_ratio_quote(
        self, periods: Optional[List[int]] = None, **kwargs: Any
    ) -> None:
        periods = periods or [15, 120]
        for window in periods:
            colname = f"F_trade_buy_pressure_ma_ratio_quote_{window}_f64"
            self.df[colname] = feature_trade_buy_pressure_ma_ratio_quote(self.df, p_window=window)
            self._register_feature(colname, FeatureType.T_BUY_MA_RATIO_Q)

    def _add_trade_avg_trade_size_dollar_ma_ratio(
        self, periods: Optional[List[int]] = None, **kwargs: Any
    ) -> None:
        periods = periods or [15, 120]
        for window in periods:
            colname = f"F_trade_avg_trade_size_dollar_ma_ratio_{window}_f64"
            self.df[colname] = feature_trade_avg_trade_size_dollar_ma_ratio(
                self.df, p_window=window
            )
            self._register_feature(colname, FeatureType.T_AVG_TS_USD_MA_RATIO)

    def _add_trade_trade_count_ma_ratio(
        self, periods: Optional[List[int]] = None, **kwargs: Any
    ) -> None:
        periods = periods or [15, 120]
        for window in periods:
            colname = f"F_trade_trade_count_ma_ratio_{window}_f64"
            self.df[colname] = feature_trade_trade_count_ma_ratio(self.df, p_window=window)
            self._register_feature(colname, FeatureType.T_TC_MA_RATIO)

    def _add_daily_signal(
        self,
        periods: Optional[List[int]] = None,
        p_test_candles: int = 8,
        p_exit_delay: int = 4,
        **kwargs: Any,
    ) -> None:
        signal_col = "F_daily_signal_f16"
        stop_col = "F_daily_stop_price_f64"
        exit_col = "F_daily_exit_f16"
        
        self.df[signal_col] = feature_daily_signal(self.df)
        
        stop_price, exit_signal = feature_daily_signal_with_exit(
            self.df,
            p_test_candles=p_test_candles,
            p_exit_delay=p_exit_delay,
        )
        self.df[stop_col] = stop_price
        self.df[exit_col] = exit_signal
        
        self._register_feature(signal_col, FeatureType.DAILY_SIGNAL)
        self._register_feature(stop_col, FeatureType.DAILY_SIGNAL)
        self._register_feature(exit_col, FeatureType.DAILY_SIGNAL)

    def _add_total_signal(self, periods: Optional[List[int]] = None, **kwargs: Any) -> None:
        long_col = "F_total_signal_long_f16"
        short_col = "F_total_signal_short_f16"
        long_signal, short_signal = feature_total_signal(self.df)
        self.df[long_col] = long_signal
        self.df[short_col] = short_signal
        self._register_feature(long_col, FeatureType.TOTAL_SIGNAL)
        self._register_feature(short_col, FeatureType.TOTAL_SIGNAL)

    def _add_cto_line(
        self,
        periods: Optional[List[int]] = None,
        params: Tuple[int, int, int, int] = (15, 19, 25, 29),
        **kwargs: Any,
    ) -> None:
        long_col = "F_cto_line_long_f16"
        short_col = "F_cto_line_short_f16"
        v1_rel_dist_col = "F_cto_line_v1_rel_dist_f16"
        v2_rel_dist_col = "F_cto_line_v2_rel_dist_f16"
        long_signal, short_signal, v1_rel_dist, v2_rel_dist = feature_cto_line_signal(self.df, p_params=params)
        self.df[long_col] = long_signal
        self.df[short_col] = short_signal
        self.df[v1_rel_dist_col] = v1_rel_dist
        self.df[v2_rel_dist_col] = v2_rel_dist
        self._register_feature(long_col, FeatureType.CTO_LINE)
        self._register_feature(short_col, FeatureType.CTO_LINE)
        self._register_feature(v1_rel_dist_col, FeatureType.CTO_LINE)
        self._register_feature(v2_rel_dist_col, FeatureType.CTO_LINE)

    def convert_f16_columns(self) -> "BaseDataFrame":
        """Convert all _f16 columns to the target precision."""
        f16_cols = [col for col in self.df.columns if col.endswith("_f16")]
        self.df = convert_df_cols_float_to_float(self.df, f16_cols, g_precision, [])
        return self

    def get_dataframe(self) -> pd.DataFrame:
        """Return the dataframe."""
        return self.df

    def get_features(self) -> List[Feature]:
        """Return the list of features."""
        return self.selected_features

    def get_feature_columns(self) -> List[str]:
        """Return the list of feature column names."""
        return self.selected_columns

    def desc(self) -> Dict[str, Any]:
        """Return a description of the dataframe with size, dates, and columns."""
        if g_close_time_col in self.df.columns:
            start_date = pd.to_datetime(self.df[g_close_time_col].iloc[0], unit="ms")
            end_date = pd.to_datetime(self.df[g_close_time_col].iloc[-1], unit="ms")
        else:
            start_date = None
            end_date = None
        return {
            "size": len(self.df),
            "start_date": start_date,
            "end_date": end_date,
            "columns": list(self.df.columns),
        }

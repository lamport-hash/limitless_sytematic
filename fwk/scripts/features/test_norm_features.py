import sys
from pathlib import Path

import pandas as pd

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent
sys.path.insert(0, str(_project_root / "data_norm_features"))

from features.base_dataframe import BaseDataFrame
from features.features_utils import FeatureType
from dataframe_utils.df_utils import read_parquet_to_f16


_docker_data_path = Path("/data/normalised/candle_1hour/firstrate/etf/QQQ")
_local_data_path = (
    _project_root / "docker_data/data/normalised/candle_1hour/firstrate/etf/QQQ"
)
DATA_PATH = _docker_data_path if _docker_data_path.exists() else _local_data_path


def get_parquet_files(p_symbol: str) -> list[Path]:
    files = list(DATA_PATH.glob(f"*{p_symbol}*.parquet"))
    return sorted(files)


def test_basic_feature_generation(p_symbol: str = "QQQ") -> pd.DataFrame:
    files = get_parquet_files(p_symbol)
    if not files:
        raise FileNotFoundError(f"No parquet files found for {p_symbol} in {DATA_PATH}")

    filepath = files[0]
    print(f"Loading: {filepath}")

    df = read_parquet_to_f16(filepath)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {df.columns.tolist()}")

    bdf = BaseDataFrame(
        p_df=df,
        p_verbose=True,
    )

    feature_types = [
        FeatureType.RSI,
        FeatureType.HIST_VOLATILITY,
        FeatureType.EMA,
        FeatureType.SPREAD_REL_EMA,
    ]
    periods = [14, 60, 240]

    bdf.add_features(feature_types, periods=periods)

    result_df = bdf.get_dataframe()
    features = bdf.get_feature_columns()

    print(f"\nGenerated {len(features)} feature columns:")
    for f in features[:20]:
        print(f"  - {f}")
    if len(features) > 20:
        print(f"  ... and {len(features) - 20} more")

    print(f"\nResult DataFrame shape: {result_df.shape}")
    print(f"Sample of last 5 rows:\n{result_df.tail(5)}")

    return result_df


if __name__ == "__main__":
    test_basic_feature_generation()

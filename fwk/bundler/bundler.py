import gzip
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, cast

import pandas as pd
from tqdm import tqdm

from dataframe_utils.df_utils import (
    g_index_col,
    read_parquet_to_f16,
)
from features.feature_build_features_targets import prepare_base_dataframe

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DataBundler:
    """
    Loads a range of candle data files for a single symbol (either all CANDLE or all CANDLE_BINANCE), merges them with gap detection,
    and saves as compressed pickle.

    # todo use new data/normalised/candle_* folders
    # output is always data/bundle

    File naming convention expected:
        {DATA_TYPE}_{SYMBOL}_{START_DATE}_{END_DATE}_1min.df.parquet
        Example: {DATA_TYPE}_BTCUSDT_20241215_20241216_1min.df.parquet
    """

    DEFAULT_SOURCE_DIR = "data/normalised/candle_1min/binance/spot"
    DEFAULT_BUNDLE_DIR = "data/bundle"

    def __init__(
        self,
        p_source_dir: Optional[str] = None,
        p_bundle_dir: Optional[str] = None,
        p_verbose: bool = False,
        p_float_precision: str = "16",
    ):
        """
        Initialize DataBundler.

        Args:
            p_source_dir: Source directory for parquet files.
                          Default: data/candles/binance/spot/
            p_bundle_dir: Base bundle directory.
                          Default: data/bundle/ (subfolder structure mirrors source)
            p_verbose: Enable verbose logging.
            p_float_precision: Float precision for data ("16" or "32"). Default: "16".
        """
        self.source_dir = Path(p_source_dir or self.DEFAULT_SOURCE_DIR)
        self.bundle_dir = Path(p_bundle_dir or self.DEFAULT_BUNDLE_DIR)
        self.verbose = p_verbose
        self.float_precision = p_float_precision

        if self.verbose:
            logger.setLevel(logging.DEBUG)

    def _parse_filename(self, p_filename: str) -> Optional[Tuple[str, datetime, datetime]]:
        """
        Parse a filename to extract symbol and date range.

        Args:
            p_filename: Filename to parse.

        Returns:
            Tuple of (symbol, start_date, end_date) or None if parsing fails.
        """
        pattern = r"binance_(\w+)_(\d{8})_(\d{8})_1min\.df\.parquet"
        match = re.match(pattern, p_filename)
        if not match:
            return None

        symbol = match.group(1)
        start_date = datetime.strptime(match.group(2), "%Y%m%d")
        end_date = datetime.strptime(match.group(3), "%Y%m%d")

        return symbol, start_date, end_date

    def _get_output_path(
        self,
        p_symbol: str,
        p_start_date: datetime,
        p_end_date: datetime,
        p_subfolder: str,
    ) -> Path:
        """
        Generate output path for the bundled file.

        Args:
            p_symbol: Trading symbol (e.g., 'BTCUSDT').
            p_start_date: Start date of the bundle.
            p_end_date: End date of the bundle.
            p_subfolder: Subfolder path relative to bundle dir (mirrors source structure).

        Returns:
            Full path for the output file.
        """
        start_str = p_start_date.strftime("%Y%m%d")
        end_str = p_end_date.strftime("%Y%m%d")
        filename = f"{p_symbol}_{start_str}_{end_str}.pkl.gz"

        output_dir = self.bundle_dir / p_subfolder
        output_dir.mkdir(parents=True, exist_ok=True)

        return output_dir / filename

    def get_files_for_range(
        self,
        p_symbol: str,
        p_start_date: str,
        p_end_date: str,
    ) -> List[Path]:
        """
        Find matching parquet files for the given symbol and date range.

        Args:
            p_symbol: Trading symbol (e.g., 'BTCUSDT').
            p_start_date: Start date string (YYYY-MM-DD format).
            p_end_date: End date string (YYYY-MM-DD format).

        Returns:
            List of Path objects for matching files, sorted by date.
        """
        start_dt = datetime.strptime(p_start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(p_end_date, "%Y-%m-%d")

        matching_files = []

        if not self.source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {self.source_dir}")

        for file_path in self.source_dir.glob("*.parquet"):
            parsed = self._parse_filename(file_path.name)
            if parsed is None:
                continue

            symbol, file_start, file_end = parsed

            if symbol != p_symbol:
                continue

            if file_start <= end_dt and file_end >= start_dt:
                matching_files.append(file_path)

        matching_files.sort(key=lambda x: x.name)

        if self.verbose:
            logger.info(f"Found {len(matching_files)} files for {p_symbol}")

        return matching_files

    def load_and_merge(
        self,
        p_files: List[Path],
        p_add_validity: bool = True,
    ) -> pd.DataFrame:
        """
        Load multiple parquet files and merge them into a single DataFrame.

        Args:
            p_files: List of parquet file paths.
            p_add_validity: Whether to add gap detection and validity columns.

        Returns:
            Merged DataFrame with all data.
        """
        if not p_files:
            raise ValueError("No files provided for merging")

        dfs = []

        for file_path in tqdm(p_files, desc="Loading files", disable=not self.verbose):
            try:
                df = read_parquet_to_f16(
                    file_path,
                    P_col_exception_list=PRICE_COLS_RAW,# this need to be fixed
                    p_float_precision=self.float_precision,
                )
                dfs.append(df)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                raise

        merged_df = pd.concat(dfs, ignore_index=True)

        if self.verbose:
            logger.info(f"Merged {len(dfs)} files, total rows: {len(merged_df)}")

        if p_add_validity:
            merged_df, _, _, _ = prepare_base_dataframe(
                merged_df,
                p_verbose=self.verbose,
            )

        merged_df = merged_df.copy()

        return merged_df

    def bundle(
        self,
        p_symbol: str,
        p_start_date: str,
        p_end_date: str,
        p_output_name: Optional[str] = None,
        p_add_validity: bool = True,
    ) -> Tuple[pd.DataFrame, Path]:
        """
        Full pipeline: find files, load, merge, and save as compressed pickle.

        Args:
            p_symbol: Trading symbol (e.g., 'BTCUSDT').
            p_start_date: Start date string (YYYY-MM-DD format).
            p_end_date: End date string (YYYY-MM-DD format).
            p_output_name: Custom output filename (without extension).
                          If None, uses '{symbol}_{start}_{end}'.
            p_add_validity: Whether to add gap detection and validity columns.

        Returns:
            Tuple of (merged DataFrame, output file path).
        """
        files = self.get_files_for_range(p_symbol, p_start_date, p_end_date)

        if not files:
            raise FileNotFoundError(
                f"No files found for {p_symbol} between {p_start_date} and {p_end_date}"
            )

        merged_df = self.load_and_merge(files, p_add_validity=p_add_validity)

        start_dt = datetime.strptime(p_start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(p_end_date, "%Y-%m-%d")

        subfolder_parts = list(self.source_dir.parts)
        if subfolder_parts and subfolder_parts[0] == "data":
            subfolder_parts = subfolder_parts[1:]
        subfolder = str(Path(*subfolder_parts)) if subfolder_parts else ""
        output_path = self._get_output_path(p_symbol, start_dt, end_dt, subfolder)

        if p_output_name:
            output_path = output_path.parent / f"{p_output_name}.pkl.gz"

        merged_df.to_pickle(output_path, compression="gzip")

        if self.verbose:
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"Saved bundle to {output_path} ({file_size_mb:.2f} MB)")

        return merged_df, output_path

    @staticmethod
    def load_bundle(p_filepath: str) -> pd.DataFrame:
        """
        Load a compressed pickle bundle.

        Args:
            p_filepath: Path to the .pkl.gz file.

        Returns:
            DataFrame containing the bundled data.
        """
        filepath = Path(p_filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Bundle file not found: {filepath}")

        df = cast(pd.DataFrame, pd.read_pickle(filepath, compression="gzip"))
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(df)}")
        return df

    def list_available_symbols(self) -> List[str]:
        """
        List all unique symbols available in the source directory.

        Returns:
            List of symbol strings.
        """
        symbols = set()

        for file_path in self.source_dir.glob("*.parquet"):
            parsed = self._parse_filename(file_path.name)
            if parsed:
                symbols.add(parsed[0])

        return sorted(list(symbols))

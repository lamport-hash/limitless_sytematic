import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import pyarrow.parquet as pq

BUNDLE_DIR = Path(os.environ.get("BUNDLE_DIR", "/home/brian/sing/data/bundle"))


class ParquetReader:
    def __init__(self, bundle_dir: Path = BUNDLE_DIR):
        self.bundle_dir = Path(bundle_dir)
        if not self.bundle_dir.exists():
            raise ValueError(f"Bundle directory does not exist: {self.bundle_dir}")

    def list_bundle_files(self) -> List[Dict[str, Any]]:
        files = []
        for root, _, filenames in os.walk(self.bundle_dir):
            for filename in filenames:
                if filename.endswith(".parquet"):
                    filepath = Path(root) / filename
                    rel_path = filepath.relative_to(self.bundle_dir)
                    try:
                        parquet_file = pq.ParquetFile(filepath)
                        metadata = parquet_file.metadata
                        files.append({
                            "filename": filename,
                            "path": str(rel_path),
                            "full_path": str(filepath),
                            "num_rows": metadata.num_rows,
                            "num_columns": metadata.num_columns,
                            "num_row_groups": metadata.num_row_groups,
                        })
                    except Exception as e:
                        files.append({
                            "filename": filename,
                            "path": str(rel_path),
                            "full_path": str(filepath),
                            "error": str(e),
                        })
        return files

    def get_columns(self, filename: str) -> List[str]:
        filepath = self._resolve_path(filename)
        df = pd.read_parquet(filepath, columns=None)
        return df.columns.tolist()

    def get_column_types(self, filename: str) -> Dict[str, str]:
        filepath = self._resolve_path(filename)
        df = pd.read_parquet(filepath, columns=None)
        return {col: str(dtype) for col, dtype in df.dtypes.items()}

    def load_dataframe(self, filename: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
        filepath = self._resolve_path(filename)
        df = pd.read_parquet(filepath, columns=columns)
        return df

    def get_data(
        self,
        filename: str,
        columns: Optional[List[str]] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> Dict[str, Any]:
        filepath = self._resolve_path(filename)
        df = pd.read_parquet(filepath, columns=columns)

        if offset > 0:
            df = df.iloc[offset:]
        if limit is not None:
            df = df.iloc[:limit]

        result = {
            "columns": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "data": df.to_dict(orient="records"),
        }
        return result

    def get_chart_data(
        self,
        filename: str,
        column: str,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        filepath = self._resolve_path(filename)
        df = pd.read_parquet(filepath, columns=[column])

        if limit is not None:
            df = df.tail(limit)

        index_col = None
        for col in ["i_minute_i", "index", "date", "datetime", "timestamp"]:
            if col in df.columns:
                index_col = col
                break

        if index_col:
            x_data = df[index_col].tolist()
        else:
            x_data = list(range(len(df)))

        return {
            "x": x_data,
            "y": df[column].tolist(),
            "column": column,
            "num_points": len(df),
        }

    def _resolve_path(self, filename: str) -> Path:
        filepath = self.bundle_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        return filepath


_reader: Optional[ParquetReader] = None


def get_reader() -> ParquetReader:
    global _reader
    if _reader is None:
        _reader = ParquetReader()
    return _reader


def list_bundle_files() -> List[Dict[str, Any]]:
    return get_reader().list_bundle_files()


def get_columns(filename: str) -> List[str]:
    return get_reader().get_columns(filename)


def get_column_types(filename: str) -> Dict[str, str]:
    return get_reader().get_column_types(filename)


def get_data(
    filename: str,
    columns: Optional[List[str]] = None,
    limit: Optional[int] = None,
    offset: int = 0,
) -> Dict[str, Any]:
    return get_reader().get_data(filename, columns, limit, offset)


def get_chart_data(
    filename: str,
    column: str,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    return get_reader().get_chart_data(filename, column, limit)

"""
Search utilities for finding normalized data files.

Provides flexible search capabilities to locate parquet files in data/normalised/
based on various criteria such as symbol, frequency, product type, source, etc.

Directory structure:
    data/normalised/<data_freq>/<source_exchange>/<product_type>/<instrument>/<file>.df.parquet
"""

from pathlib import Path
from typing import List, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum

from core.data_org import (
    NORMALISED_DIR,
    MktDataTFreq,
    ExchangeNAME,
    ProductType,
)


@dataclass
class DataFile:
    """Represents a found data file with parsed metadata."""

    path: Path
    data_freq: str
    source: str
    exchange: str
    product_type: str
    instrument: str
    filename: str

    def __str__(self) -> str:
        return str(self.path)

    def __repr__(self) -> str:
        return f"DataFile({self.instrument} @ {self.data_freq}/{self.product_type})"


def search_data(
    p_symbol: Optional[str] = None,
    p_data_freq: Optional[Union[str, MktDataTFreq]] = None,
    p_source: Optional[Union[str, ExchangeNAME]] = None,
    p_exchange: Optional[Union[str, ExchangeNAME]] = None,
    p_product_type: Optional[Union[str, ProductType]] = None,
    p_start: Optional[str] = None,
    p_end: Optional[str] = None,
    p_base_dir: Optional[Path] = None,
) -> List[DataFile]:
    """
    Search for normalized data files based on criteria.

    All parameters are optional - omitting a parameter means "match any".
    Parameters are case-insensitive for string matching.

    Args:
        p_symbol: Instrument/symbol to match (e.g., "EURUSD", "qqq")
        p_data_freq: Data frequency (e.g., "candle_1min", MktDataTFreq.CANDLE_1HOUR)
        p_source: Data source name (e.g., "firstrate", ExchangeNAME.FIRSTRATE)
        p_exchange: Exchange name (e.g., "undefined", ExchangeNAME.UNDEFINED)
        p_product_type: Product type (e.g., "spot", ProductType.ETF)
        p_start: Start date filter (e.g., "20200101")
        p_end: End date filter (e.g., "20201231")
        p_base_dir: Base directory to search (default: NORMALISED_DIR)

    Returns:
        List of DataFile objects matching the criteria.

    Examples:
        # Find all EURUSD data
        >>> files = search_data(p_symbol="EURUSD")

        # Find all 1-hour candle data for spot products
        >>> files = search_data(p_data_freq="candle_1hour", p_product_type="spot")

        # Find specific combination
        >>> files = search_data(
        ...     p_symbol="QQQ",
        ...     p_data_freq=MktDataTFreq.CANDLE_1HOUR,
        ...     p_source=ExchangeNAME.FIRSTRATE,
        ...     p_product_type=ProductType.ETF
        ... )

        # Find data from a specific date range
        >>> files = search_data(p_symbol="EURUSD", p_start="20200101", p_end="20201231")
    """
    base_dir = p_base_dir or NORMALISED_DIR

    if not base_dir.exists():
        return []

    data_freq = _enum_to_str(p_data_freq)
    source = _enum_to_str(p_source)
    exchange = _enum_to_str(p_exchange)
    product_type = _enum_to_str(p_product_type)
    symbol = p_symbol.lower() if p_symbol else None
    start = p_start.lower() if p_start else None
    end = p_end.lower() if p_end else None

    results: List[DataFile] = []

    for freq_dir in base_dir.iterdir():
        if not freq_dir.is_dir():
            continue
        if data_freq and freq_dir.name != data_freq:
            continue

        for src_ex_dir in freq_dir.iterdir():
            if not src_ex_dir.is_dir():
                continue

            dir_parts = src_ex_dir.name.split("_")
            dir_source = dir_parts[0] if len(dir_parts) >= 1 else ""
            dir_exchange = dir_parts[1] if len(dir_parts) >= 2 else ""

            if source and dir_source != source:
                continue
            if exchange and dir_exchange != exchange:
                continue

            for prod_dir in src_ex_dir.iterdir():
                if not prod_dir.is_dir():
                    continue
                if product_type and prod_dir.name != product_type:
                    continue

                for instr_dir in prod_dir.iterdir():
                    if not instr_dir.is_dir():
                        continue
                    if symbol and instr_dir.name.lower() != symbol:
                        continue

                    for file_path in instr_dir.glob("*.df.parquet"):
                        file_start, file_end = _parse_dates_from_filename(file_path.name)

                        if start and file_start and file_start < start:
                            continue
                        if end and file_end and file_end > end:
                            continue

                        results.append(
                            DataFile(
                                path=file_path,
                                data_freq=freq_dir.name,
                                source=dir_source,
                                exchange=dir_exchange,
                                product_type=prod_dir.name,
                                instrument=instr_dir.name,
                                filename=file_path.name,
                            )
                        )

    return sorted(results, key=lambda x: x.path)


def search_data_paths(**kwargs) -> List[Path]:
    """
    Search for data files and return only paths.

    Convenience wrapper around search_data() that returns Path objects.

    Args:
        **kwargs: Same arguments as search_data()

    Returns:
        List of Path objects matching the criteria.
    """
    return [df.path for df in search_data(**kwargs)]


def list_available(
    p_data_freq: Optional[Union[str, MktDataTFreq]] = None,
    p_source: Optional[Union[str, ExchangeNAME]] = None,
    p_product_type: Optional[Union[str, ProductType]] = None,
    p_base_dir: Optional[Path] = None,
) -> dict:
    """
    List available data organized by category.

    Useful for exploring what data is available in the normalized directory.

    Args:
        p_data_freq: Filter by data frequency
        p_source: Filter by source
        p_product_type: Filter by product type
        p_base_dir: Base directory to search

    Returns:
        Dict with keys: 'frequencies', 'sources', 'product_types', 'instruments'

    Example:
        >>> info = list_available(p_product_type="spot")
        >>> print(info['instruments'][:5])
        ['AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD', 'AUDUSD']
    """
    files = search_data(
        p_data_freq=p_data_freq,
        p_source=p_source,
        p_product_type=p_product_type,
        p_base_dir=p_base_dir,
    )

    return {
        "frequencies": sorted(set(f.data_freq for f in files)),
        "sources": sorted(set(f.source for f in files)),
        "product_types": sorted(set(f.product_type for f in files)),
        "instruments": sorted(set(f.instrument for f in files)),
        "count": len(files),
    }


def _enum_to_str(value: Optional[Union[str, Enum]]) -> Optional[str]:
    """Convert enum to string value, or pass through string."""
    if value is None:
        return None
    if isinstance(value, str):
        return value.lower()
    if isinstance(value, Enum):
        return str(value.value).lower()
    return str(value).lower()


def _parse_dates_from_filename(filename: str) -> tuple:
    """
    Parse start and end dates from filename.

    Expected format: <INSTRUMENT>_<YYYYMMDD>_<YYYYMMDD>_<freq>.df.parquet

    Returns:
        Tuple of (start_date, end_date) as strings, or (None, None) if not parseable.
    """
    parts = filename.replace(".df.parquet", "").split("_")
    if len(parts) >= 4:
        return parts[1], parts[2]
    return None, None

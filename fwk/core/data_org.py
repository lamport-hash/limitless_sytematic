"""
Data organization utilities for managing file paths and locations.

This module provides utilities for locating and managing data files
from various sources using the standardized directory structure:

Directory Tree:
    data/                          # ROOT: Read-Only for Agents
    ├── import/                    # RAW: Immutable Golden Source
    │   ├── <source_name>/         # e.g., firstrate, ibkr, binance, listener
    │   │   ├── <subtype>/         # fx, etf, index
    │   │   │   └── *.txt          # Vendor specific formats
    │
    └── normalized/                # PROCESSED: Immutable Golden Source
        ├── <data_type_freq>/      # candles_1sec | candle_1min | candle_1hour | tick_ba | tick_trades
        │   ├── <source_name_exchange_name>/   # e.g., binance_binance, kraken_kraken
        │   │   ├── <product_type>/ # future, spot, option, etf
        │   │   │   └── <instrument>/
        │   │   │       └── *.df.parquet  # Format: <instrument>_<start>_<end>_<freq>.df.parquet

Enums used:
    - MktDataFred: CANDLE_1MIN, CANDLE_1HOUR
    - MktDataType: ORDERBOOK, TRADE, CANDLE, CANDLE_BINANCE
    - ExchangeNAME: BINANCE, FIRSTRATE, UNDEFINED
    - ConnectorTYPE: CCXT_WS, CCXT_REST, etc.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List
from enum import Enum

from core.enums import MktDataFred, ExchangeNAME


DATA_DIR = Path(__file__).parent.parent / "data"
IMPORT_DIR = DATA_DIR / "import"
NORMALISED_DIR = DATA_DIR / "normalised"
BUNDLE_DIR = DATA_DIR / "bundle"
WORK_DIR = DATA_DIR / "work"


class ProductType(Enum):
    """Product type for normalized data."""

    FUTURE = "future"
    SPOT = "spot"
    OPTION = "option"
    ETF = "etf"


def get_data_dir() -> Path:
    """Get the root data directory."""
    return DATA_DIR


def get_import_dir() -> Path:
    """Get the import directory."""
    return IMPORT_DIR


def get_normalised_dir() -> Path:
    """Get the normalised directory."""
    return NORMALISED_DIR


def get_import_source_dir(p_source: ExchangeNAME, p_subtype: str = "") -> Path:
    """
    Get the import directory for a specific source.

    Args:
        p_source: Source enum (e.g., ExchangeNAME.FIRSTRATE, ExchangeNAME.BINANCE).
        p_subtype: Optional subtype (e.g., "etf", "fx", "index").

    Returns:
        Path to the source import directory.

    Examples:
        >>> get_import_source_dir(ExchangeNAME.FIRSTRATE, "etf")
        Path('/home/brian/sing/data/import/firstrate/etf')

        >>> get_import_source_dir(ExchangeNAME.BINANCE)
        Path('/home/brian/sing/data/import/binance')
    """
    source_name = p_source.value if isinstance(p_source, ExchangeNAME) else p_source
    if p_subtype:
        return IMPORT_DIR / source_name / p_subtype
    return IMPORT_DIR / source_name


def get_import_file(
    p_source: ExchangeNAME, p_symbol: str, p_subtype: str = "", p_extension: str = "txt"
) -> Path:
    """
    Get the full path to an import file.

    Args:
        p_source: Source enum.
        p_symbol: Symbol/ticker (e.g., "AAA", "BTC").
        p_subtype: Optional subtype (e.g., "etf", "fx", "etf_1h").
        p_extension: File extension (default: "txt").

    Returns:
        Path to the import file.

    Examples:
        >>> get_import_file(ExchangeNAME.FIRSTRATE, "AAA", "etf_1h")
        Path('/home/brian/sing/data/import/firstrate/etf_1h/AAA_full_1hour_adjsplitdiv.txt')
    """
    source_dir = get_import_source_dir(p_source, p_subtype)

    if p_source == ExchangeNAME.FIRSTRATE:
        if p_subtype and "1h" in p_subtype:
            filename = f"{p_symbol}_full_1hour_adjsplitdiv.{p_extension}"
        elif p_subtype == "etf":
            filename = f"{p_symbol}_full_1hour_adjsplitdiv.{p_extension}"
        else:
            filename = f"{p_symbol}_full_1hour_adjsplitdiv.{p_extension}"
    else:
        filename = f"{p_symbol}.{p_extension}"

    return source_dir / filename


def get_normalised_data_freq_dir(
    p_data_freq: MktDataFred,
) -> Path:
    """
    Get the normalised directory for a specific data frequency.

    Args:
        p_data_freq: Data frequency enum (e.g., MktDataFred.CANDLE_1MIN).

    Returns:
        Path to the data frequency directory.

    Examples:
        >>> get_normalised_data_freq_dir(MktDataFred.CANDLE_1HOUR)
        Path('/home/brian/sing/data/normalised/candle_1hour')
    """
    return NORMALISED_DIR / p_data_freq.value


def get_normalised_source_dir(
    p_data_freq: MktDataFred,
    p_source: ExchangeNAME,
    p_exchange: Optional[ExchangeNAME] = None,
) -> Path:
    """
    Get the normalised directory for a specific source.

    The directory format is: <data_freq>/<source_name>_<exchange_name>/

    Args:
        p_data_freq: Data frequency enum.
        p_source: Source enum (e.g., ExchangeNAME.FIRSTRATE).
        p_exchange: Optional exchange enum. If None, uses p_source.

    Returns:
        Path to the source normalised directory.

    Examples:
        >>> get_normalised_source_dir(MktDataFred.CANDLE_1HOUR, ExchangeNAME.FIRSTRATE)
        Path('/home/brian/sing/data/normalised/candle_1hour/firstrate_undefined')

        >>> get_normalised_source_dir(MktDataFred.CANDLE_1MIN, ExchangeNAME.BINANCE, ExchangeNAME.BINANCE)
        Path('/home/brian/sing/data/normalised/candle_1min/binance_binance')
    """
    data_freq_dir = get_normalised_data_freq_dir(p_data_freq)

    source_name = p_source.value if isinstance(p_source, ExchangeNAME) else p_source
    exchange_name = (
        (p_exchange.value if isinstance(p_exchange, ExchangeNAME) else p_exchange)
        if p_exchange
        else "undefined"
    )

    return data_freq_dir / f"{source_name}_{exchange_name}"


def get_normalised_product_dir(
    p_data_freq: MktDataFred,
    p_source: ExchangeNAME,
    p_product_type: ProductType,
    p_exchange: Optional[ExchangeNAME] = None,
) -> Path:
    """
    Get the normalised directory for a specific product type.

    Args:
        p_data_freq: Data frequency enum.
        p_source: Source enum.
        p_product_type: Product type enum (e.g., ProductType.ETF, ProductType.SPOT).
        p_exchange: Optional exchange enum.

    Returns:
        Path to the product normalised directory.

    Examples:
        >>> get_normalised_product_dir(MktDataFred.CANDLE_1HOUR, ExchangeNAME.FIRSTRATE, ProductType.ETF)
        Path('/home/brian/sing/data/normalised/candle_1hour/firstrate_undefined/etf')
    """
    source_dir = get_normalised_source_dir(p_data_freq, p_source, p_exchange)
    return source_dir / p_product_type.value


def get_normalised_instrument_dir(
    p_data_freq: MktDataFred,
    p_source: ExchangeNAME,
    p_product_type: ProductType,
    p_instrument: str,
    p_exchange: Optional[ExchangeNAME] = None,
) -> Path:
    """
    Get the normalised directory for a specific instrument.

    Args:
        p_data_freq: Data frequency enum.
        p_source: Source enum.
        p_product_type: Product type enum.
        p_instrument: Instrument/symbol (e.g., "AAA", "BTCUSDT").
        p_exchange: Optional exchange enum.

    Returns:
        Path to the instrument normalised directory.

    Examples:
        >>> get_normalised_instrument_dir(MktDataFred.CANDLE_1HOUR, ExchangeNAME.FIRSTRATE, ProductType.ETF, "AAA")
        Path('/home/brian/sing/data/normalised/candle_1hour/firstrate_undefined/etf/AAA')
    """
    product_dir = get_normalised_product_dir(p_data_freq, p_source, p_product_type, p_exchange)
    return product_dir / p_instrument


def get_normalised_file(
    p_data_freq: MktDataFred,
    p_source: ExchangeNAME,
    p_product_type: ProductType,
    p_instrument: str,
    p_start: Optional[str] = None,
    p_end: Optional[str] = None,
    p_exchange: Optional[ExchangeNAME] = None,
) -> Path:
    """
    Get the full path to a normalised parquet file.

    Filename format: <instrument>_<start>_<end>_<freq>.df.parquet

    Args:
        p_data_freq: Data frequency enum.
        p_source: Source enum.
        p_product_type: Product type enum.
        p_instrument: Instrument/symbol (e.g., "AAA", "BTCUSDT").
        p_start: Start date string (e.g., "20200101"). If None, uses "*".
        p_end: End date string (e.g., "20201231"). If None, uses "*".
        p_exchange: Optional exchange enum.

    Returns:
        Path to the normalised parquet file.

    Examples:
        >>> get_normalised_file(MktDataFred.CANDLE_1HOUR, ExchangeNAME.FIRSTRATE, ProductType.ETF, "AAA")
        Path('/home/brian/sing/data/normalised/candle_1hour/firstrate_undefined/etf/AAA/AAA_*_*_candle_1hour.df.parquet')

        >>> get_normalised_file(MktDataFred.CANDLE_1HOUR, ExchangeNAME.FIRSTRATE, ProductType.ETF, "AAA", "20200101", "20201231")
        Path('/home/brian/sing/data/normalised/candle_1hour/firstrate_undefined/etf/AAA/AAA_20200101_20201231_candle_1hour.df.parquet')
    """
    instrument_dir = get_normalised_instrument_dir(
        p_data_freq, p_source, p_product_type, p_instrument, p_exchange
    )

    start_str = p_start if p_start else "*"
    end_str = p_end if p_end else "*"
    freq_str = p_data_freq.value

    filename = f"{p_instrument}_{start_str}_{end_str}_{freq_str}.df.parquet"

    return instrument_dir / filename


def list_instruments(
    p_data_freq: MktDataFred,
    p_source: ExchangeNAME,
    p_product_type: ProductType,
    p_exchange: Optional[ExchangeNAME] = None,
) -> List[str]:
    """
    List all available instruments for a given configuration.

    Args:
        p_data_freq: Data frequency enum.
        p_source: Source enum.
        p_product_type: Product type enum.
        p_exchange: Optional exchange enum.

    Returns:
        List of instrument symbols.

    Examples:
        >>> list_instruments(MktDataFred.CANDLE_1HOUR, ExchangeNAME.FIRSTRATE, ProductType.ETF)
        ['AAA', 'AAAU', 'AADR', ...]
    """
    product_dir = get_normalised_product_dir(p_data_freq, p_source, p_product_type, p_exchange)

    if not product_dir.exists():
        return []

    instruments = []
    for item in product_dir.iterdir():
        if item.is_dir():
            instruments.append(item.name)

    return sorted(instruments)


def create_normalised_dirs(
    p_data_freq: MktDataFred,
    p_source: ExchangeNAME,
    p_product_type: ProductType,
    p_instrument: str,
    p_exchange: Optional[ExchangeNAME] = None,
) -> Path:
    """
    Create all necessary directories for a normalised file and return the path.

    Args:
        p_data_freq: Data frequency enum.
        p_source: Source enum.
        p_product_type: Product type enum.
        p_instrument: Instrument/symbol.
        p_exchange: Optional exchange enum.

    Returns:
        Path to the instrument directory that was created.

    Examples:
        >>> create_normalised_dirs(MktDataFred.CANDLE_1HOUR, ExchangeNAME.FIRSTRATE, ProductType.ETF, "AAA")
        Path('/home/brian/sing/data/normalised/candle_1hour/firstrate_undefined/etf/AAA')
    """
    instrument_dir = get_normalised_instrument_dir(
        p_data_freq, p_source, p_product_type, p_instrument, p_exchange
    )
    instrument_dir.mkdir(parents=True, exist_ok=True)
    return instrument_dir

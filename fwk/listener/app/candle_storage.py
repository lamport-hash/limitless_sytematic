import os
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def gen_filename_download(
    exchange: str,
    pair: str,
    start_time: datetime,
    end_time: datetime,
    timeframe: str = "1min",
) -> str:
    """Generate filename for downloaded candle data"""
    return f"{exchange}_{pair}_{start_time.strftime('%Y%m%d')}_{end_time.strftime('%Y%m%d')}_{timeframe}.df.parquet"


def get_folder_location(
    p_folder: str, p_exchange: str = "binance", p_market_type: str = "spot"
) -> str:
    """Get folder location for storing candle data"""
    # Ensure p_folder ends with exactly one slash
    if not p_folder.endswith("/"):
        p_folder = p_folder + "/"
    elif p_folder.endswith("//"):
        p_folder = p_folder.rstrip("/") + "/"

    folder_name = f"{p_folder}{p_exchange}/{p_market_type}/"
    return folder_name


def get_create_folder_location(
    p_folder: str, p_exchange: str = "binance", p_market_type: str = "spot"
) -> str:
    """Create folder if it doesn't exist and return path"""
    folder_name = get_folder_location(p_folder, p_exchange, p_market_type)
    os.makedirs(folder_name, exist_ok=True)
    return folder_name


def create_placeholder_file(filepath: str, reason: str = "no_data") -> None:
    """Create empty placeholder file to mark failed/no-data download"""
    placeholder_path = filepath.replace(".parquet", ".placeholder")
    try:
        with open(placeholder_path, "w") as f:
            f.write(f"{reason}\n{datetime.now().isoformat()}")
        logger.info(f"Created placeholder: {placeholder_path} ({reason})")
    except Exception as e:
        logger.error(f"Failed to create placeholder {placeholder_path}: {e}")


def is_file_or_placeholder(filepath: str) -> bool:
    """Check if data file or placeholder exists"""
    data_file_exists = os.path.exists(filepath)
    placeholder_exists = os.path.exists(filepath.replace(".parquet", ".placeholder"))
    return data_file_exists or placeholder_exists


def get_placeholder_reason(filepath: str) -> Optional[str]:
    """Get reason from placeholder file if it exists"""
    placeholder_path = filepath.replace(".parquet", ".placeholder")
    if os.path.exists(placeholder_path):
        try:
            with open(placeholder_path, "r") as f:
                lines = f.readlines()
                return lines[0].strip() if lines else "unknown"
        except Exception as e:
            logger.error(f"Failed to read placeholder {placeholder_path}: {e}")
            return "unknown"
    return None


def get_file_status(filepath: str) -> str:
    """Get status of file: exists, placeholder, or missing"""
    if os.path.exists(filepath):
        return "exists"
    elif os.path.exists(filepath.replace(".parquet", ".placeholder")):
        return "placeholder"
    else:
        return "missing"


def candles_to_dataframe(candles: List) -> pd.DataFrame:
    """Convert list of candle objects to pandas DataFrame"""
    if not candles:
        return pd.DataFrame()

    data = []
    for candle in candles:
        # Check if this is an extended candle
        if hasattr(candle, "quote_volume"):
            # Extended candle
            data.append(
                {
                    "timestamp": candle.timestamp,
                    "open": candle.open,
                    "high": candle.high,
                    "low": candle.low,
                    "close": candle.close,
                    "volume": candle.volume,
                    "quote_volume": candle.quote_volume,
                    "num_trades": candle.num_trades,
                    "taker_buy_volume": candle.taker_buy_volume,
                    "taker_buy_quote_volume": candle.taker_buy_quote_volume,
                }
            )
        else:
            # Basic candle
            data.append(
                {
                    "timestamp": candle.timestamp,
                    "open": candle.open,
                    "high": candle.high,
                    "low": candle.low,
                    "close": candle.close,
                    "volume": candle.volume,
                }
            )

    df = pd.DataFrame(data)
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def save_candles_to_parquet(candles: List, filepath: str) -> bool:
    """Save candles to parquet file"""
    try:
        df = candles_to_dataframe(candles)
        if df.empty:
            create_placeholder_file(filepath, "no_data")
            return False

        df.to_parquet(filepath, index=False)
        logger.info(f"Saved {len(candles)} candles to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to save candles to {filepath}: {e}")
        create_placeholder_file(filepath, "save_error")
        return False


def save_candles_extended_to_parquet(candles: List, filepath: str) -> bool:
    """Save extended candles to parquet file with _extended suffix"""
    try:
        df = candles_to_dataframe(candles)
        if df.empty:
            create_placeholder_file(filepath, "no_data")
            return False

        # Add _extended suffix to filename before extension
        extended_filepath = filepath.replace(".parquet", "_extended.parquet")
        df.to_parquet(extended_filepath, index=False)
        logger.info(f"Saved {len(candles)} extended candles to {extended_filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to save extended candles to {filepath}: {e}")
        create_placeholder_file(filepath, "save_error")
        return False


def scan_exchange_folder(
    base_folder: str, exchange: str, market_type: str
) -> List[dict]:
    """Scan folder for existing candle files and return metadata"""
    folder_path = get_folder_location(base_folder, exchange, market_type)
    if not os.path.exists(folder_path):
        return []

    files_info = []

    try:
        for filename in os.listdir(folder_path):
            if filename.endswith(".parquet") or filename.endswith(".placeholder"):
                filepath = os.path.join(folder_path, filename)

                # Parse filename to extract symbol and dates
                parts = (
                    filename.replace(".placeholder", "")
                    .replace(".parquet", "")
                    .split("_")
                )
                if len(parts) >= 4:
                    symbol = parts[1]
                    start_date_str = parts[2]
                    end_date_str = parts[3]

                    try:
                        start_date = datetime.strptime(start_date_str, "%Y%m%d")
                        end_date = datetime.strptime(end_date_str, "%Y%m%d")

                        files_info.append(
                            {
                                "symbol": symbol,
                                "filename": filename,
                                "filepath": filepath,
                                "start_date": start_date,
                                "end_date": end_date,
                                "status": "exists"
                                if filename.endswith(".parquet")
                                else "placeholder",
                                "reason": get_placeholder_reason(filepath)
                                if filename.endswith(".placeholder")
                                else None,
                            }
                        )
                    except ValueError as e:
                        logger.warning(
                            f"Failed to parse date from filename {filename}: {e}"
                        )
                        continue
    except Exception as e:
        logger.error(f"Error scanning folder {folder_path}: {e}")

    return files_info


def get_symbol_file_metadata(
    base_folder: str,
    exchange: str,
    market_type: str,
    symbol: str,
    start_date=None,
    end_date=None,
) -> dict:
    """Get metadata for all files of a specific symbol"""
    files = scan_exchange_folder(base_folder, exchange, market_type)
    symbol_files = [f for f in files if f["symbol"] == symbol]

    if not symbol_files:
        return {
            "symbol": symbol,
            "nb_days_on_drive": 0,
            "first_date": None,
            "last_date": None,
            "files": [],
        }

    # Filter by date range if provided
    if start_date and end_date:
        start_date_only = (
            start_date.date() if hasattr(start_date, "date") else start_date
        )
        end_date_only = end_date.date() if hasattr(end_date, "date") else end_date
        symbol_files = [
            f
            for f in symbol_files
            if start_date_only <= f["start_date"].date() <= end_date_only
        ]

    # Sort by date
    symbol_files.sort(key=lambda x: x["start_date"])

    # Count BOTH data files AND placeholders as completed days
    completed_files = [
        f for f in symbol_files if f["status"] in ["exists", "placeholder"]
    ]

    first_date = symbol_files[0]["start_date"] if symbol_files else None
    last_date = symbol_files[-1]["end_date"] if symbol_files else None

    return {
        "symbol": symbol,
        "nb_days_on_drive": len(completed_files),
        "first_date": first_date,
        "last_date": last_date,
        "files": symbol_files,
    }


def calculate_missing_days(
    base_folder: str, exchange: str, market_type: str, symbol: str
) -> int:
    """Calculate missing days between first_available and yesterday"""
    metadata = get_symbol_file_metadata(base_folder, exchange, market_type, symbol)

    if not metadata["first_date"]:
        return 0  # No files at all

    first_date = metadata["first_date"].date()
    yesterday = (datetime.now() - timedelta(days=1)).date()

    # Generate all expected dates
    expected_dates = set()
    current_date = first_date
    while current_date <= yesterday:
        expected_dates.add(current_date)
        current_date += timedelta(days=1)

    # Find actual dates from existing files (exclude placeholders)
    actual_dates = set()
    for file_info in metadata["files"]:
        if (
            file_info["status"] == "exists"
            and "start_date" in file_info
            and "end_date" in file_info
        ):
            if (
                file_info["start_date"] is not None
                and file_info["end_date"] is not None
            ):
                current_date = file_info["start_date"].date()
                end_date = file_info["end_date"].date()
                while current_date <= end_date:
                    actual_dates.add(current_date)
                    current_date += timedelta(days=1)

    missing_days = len(expected_dates - actual_dates)
    return missing_days


def load_candles_from_parquet(filepath: str) -> pd.DataFrame:
    """Load candles from parquet file"""
    try:
        if not os.path.exists(filepath):
            logger.warning(f"File not found: {filepath}")
            return pd.DataFrame()

        df = pd.read_parquet(filepath)
        if df.empty:
            logger.warning(f"Empty data in file: {filepath}")
            return pd.DataFrame()

        # Ensure datetime column exists
        if "datetime" not in df.columns and "timestamp" in df.columns:
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")

        # Sort by timestamp
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp").reset_index(drop=True)

        logger.info(f"Loaded {len(df)} candles from {filepath}")
        return df

    except Exception as e:
        logger.error(f"Failed to load candles from {filepath}: {e}")
        return pd.DataFrame()


def find_candle_files_for_date_range(
    base_folder: str,
    exchange: str,
    market_type: str,
    symbol: str,
    start_date: datetime,
    end_date: datetime,
) -> List[str]:
    """Find all parquet files that contain data for the given date range"""
    folder_path = get_folder_location(base_folder, exchange, market_type)
    if not os.path.exists(folder_path):
        return []

    matching_files = []

    try:
        for filename in os.listdir(folder_path):
            if not filename.endswith(".parquet"):
                continue

            # Parse filename to extract symbol and dates
            parts = filename.replace(".parquet", "").split("_")
            if len(parts) < 4:
                continue

            file_symbol = parts[1]
            if file_symbol != symbol:
                continue

            try:
                file_start_date = datetime.strptime(parts[2], "%Y%m%d")
                file_end_date = datetime.strptime(parts[3], "%Y%m%d")

                # Check if file date range overlaps with requested range
                if (file_start_date <= end_date) and (file_end_date >= start_date):
                    filepath = os.path.join(folder_path, filename)
                    if os.path.exists(filepath):
                        matching_files.append(filepath)

            except ValueError as e:
                logger.warning(f"Failed to parse date from filename {filename}: {e}")
                continue

    except Exception as e:
        logger.error(f"Error scanning folder {folder_path}: {e}")

    # Sort files by date
    matching_files.sort()
    return matching_files


def get_candles_for_date_range(
    base_folder: str,
    exchange: str,
    market_type: str,
    symbol: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """Load candles for a date range from parquet files"""
    # Get symbol metadata to find available date range if not specified
    if start_date is None or end_date is None:
        metadata = get_symbol_file_metadata(base_folder, exchange, market_type, symbol)

        # Default to last available date as end_date
        if end_date is None and metadata["last_date"]:
            end_date = metadata["last_date"]

        # Default to 30 days before end_date as start_date
        if start_date is None and end_date:
            start_date = end_date - timedelta(days=30)

            # If there's not 30 days of data, use the earliest available date
            if metadata["first_date"] and metadata["first_date"] > start_date:
                start_date = metadata["first_date"]

        # Still None means no data available
        if start_date is None or end_date is None:
            logger.warning(f"No data available for {exchange}:{symbol}")
            return pd.DataFrame()

    files = find_candle_files_for_date_range(
        base_folder, exchange, market_type, symbol, start_date, end_date
    )

    if not files:
        logger.warning(
            f"No files found for {exchange}:{symbol} from {start_date.date()} to {end_date.date()}"
        )
        return pd.DataFrame()

    all_data = []

    for filepath in files:
        df = load_candles_from_parquet(filepath)
        if df.empty:
            continue

        # Filter by date range
        if "datetime" in df.columns:
            mask = (df["datetime"] >= start_date) & (df["datetime"] <= end_date)
            filtered_df = df[mask]
        elif "timestamp" in df.columns:
            start_ts = int(start_date.timestamp() * 1000)
            end_ts = int(end_date.timestamp() * 1000)
            mask = (df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)
            filtered_df = df[mask]
        else:
            continue

        if not filtered_df.empty:
            all_data.append(filtered_df)

    if not all_data:
        logger.warning(f"No data found for {exchange}:{symbol} in specified date range")
        return pd.DataFrame()

    # Combine all data
    result_df = pd.concat(all_data, ignore_index=True)

    # Sort by timestamp and remove duplicates
    if "timestamp" in result_df.columns:
        result_df = (
            result_df.sort_values("timestamp")
            .drop_duplicates(subset=["timestamp"])
            .reset_index(drop=True)
        )

    logger.info(
        f"Loaded {len(result_df)} candles for {exchange}:{symbol} from {start_date.date()} to {end_date.date()}"
    )
    return result_df


def get_all_symbols_metadata(base_folder: str) -> List[dict]:
    """Get metadata for all symbols across all exchanges"""
    all_metadata = []

    if not os.path.exists(base_folder):
        return all_metadata

    try:
        for exchange_name in os.listdir(base_folder):
            exchange_path = os.path.join(base_folder, exchange_name)
            if os.path.isdir(exchange_path):
                for market_type in os.listdir(exchange_path):
                    market_path = os.path.join(exchange_path, market_type)
                    if os.path.isdir(market_path):
                        # Get unique symbols in this exchange/market
                        files = scan_exchange_folder(
                            base_folder, exchange_name, market_type
                        )
                        symbols = set(f["symbol"] for f in files)

                        for symbol in symbols:
                            metadata = get_symbol_file_metadata(
                                base_folder, exchange_name, market_type, symbol
                            )
                            metadata["exchange"] = exchange_name
                            metadata["market_type"] = market_type
                            metadata["nb_missings"] = calculate_missing_days(
                                base_folder, exchange_name, market_type, symbol
                            )
                            all_metadata.append(metadata)
    except Exception as e:
        logger.error(f"Error scanning base folder {base_folder}: {e}")

    return all_metadata

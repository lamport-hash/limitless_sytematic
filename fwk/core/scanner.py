import os
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
import logging

try:
    import pyarrow.parquet as pq

    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

logger = logging.getLogger(__name__)


def parse_filename_parts(filename: str) -> Optional[Dict]:
    """
    Parse filename into components.

    Pattern: {exchange}_{symbol}_{start_date}_{end_date}_{timeframe}.df.parquet

    Returns dict with keys:
        - exchange: str
        - symbol: str
        - start_date: datetime
        - end_date: datetime
        - timeframe: str
    """
    try:
        base_name = (
            filename.replace(".parquet", "")
            .replace(".placeholder", "")
            .replace(".df", "")
        )
        parts = base_name.split("_")

        if len(parts) < 5:
            return None

        exchange = parts[0]
        symbol = parts[1]
        start_date_str = parts[2]
        end_date_str = parts[3]
        timeframe = parts[4]

        start_date = datetime.strptime(start_date_str, "%Y%m%d")
        end_date = datetime.strptime(end_date_str, "%Y%m%d")

        return {
            "exchange": exchange,
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "timeframe": timeframe,
        }
    except Exception as e:
        logger.warning(f"Failed to parse filename {filename}: {e}")
        return None


def get_candle_type_from_file(filepath: str) -> str:
    """
    Detect candle type from parquet file columns.

    Returns: 'ohlcv' or 'binance_kline'
    """
    try:
        if PYARROW_AVAILABLE:
            parquet_file = pq.ParquetFile(filepath)
            columns = parquet_file.schema_arrow.names
        else:
            df = pd.read_parquet(filepath)
            columns = df.columns.tolist()

        extended_columns = [
            "timestamp_close",
            "quote_volume",
            "num_trades",
            "taker_buy_volume",
            "taker_buy_quote_volume",
        ]

        if any(col in columns for col in extended_columns):
            return "binance_kline"
        else:
            return "ohlcv"
    except Exception as e:
        logger.warning(f"Failed to detect candle type for {filepath}: {e}")
        return "ohlcv"


def get_data_inventory(base_folder: str) -> List[Dict]:
    """
    Get inventory of all available candle data.

    Returns list of dicts with fields:
        - exchange: str
        - market_type: str
        - symbol: str
        - candle_type: str (ohlcv/binance_kline)
        - timeframe: str (1min/1h/etc)
        - nb_files: int
        - first_date: str (YYYY-MM-DD)
        - last_date: str (YYYY-MM-DD)
    """
    inventory = []

    if not os.path.exists(base_folder):
        logger.warning(f"Base folder {base_folder} does not exist")
        return inventory

    try:
        for exchange_name in os.listdir(base_folder):
            exchange_path = os.path.join(base_folder, exchange_name)
            if not os.path.isdir(exchange_path):
                continue

            for market_type in os.listdir(exchange_path):
                market_path = os.path.join(exchange_path, market_type)
                if not os.path.isdir(market_path):
                    continue

                for filename in os.listdir(market_path):
                    if not filename.endswith(".parquet"):
                        continue

                    filepath = os.path.join(market_path, filename)
                    parts = parse_filename_parts(filename)

                    if not parts:
                        continue

                    candle_type = get_candle_type_from_file(filepath)

                    inventory.append(
                        {
                            "exchange": exchange_name,
                            "market_type": market_type,
                            "symbol": parts["symbol"],
                            "candle_type": candle_type,
                            "timeframe": parts["timeframe"],
                            "nb_files": 1,
                            "first_date": parts["start_date"].strftime("%Y-%m-%d"),
                            "last_date": parts["end_date"].strftime("%Y-%m-%d"),
                            "filepath": filepath,
                        }
                    )
    except Exception as e:
        logger.error(f"Error building inventory: {e}")

    return inventory


def aggregate_inventory_by_symbol(inventory: List[Dict]) -> List[Dict]:
    """
    Aggregate inventory to one row per symbol (combining all files).
    """
    symbol_map = {}

    for item in inventory:
        key = (item["exchange"], item["market_type"], item["symbol"])

        if key not in symbol_map:
            symbol_map[key] = {
                "exchange": item["exchange"],
                "market_type": item["market_type"],
                "symbol": item["symbol"],
                "candle_type": item["candle_type"],
                "timeframe": item["timeframe"],
                "nb_files": 0,
                "first_date": None,
                "last_date": None,
            }

        symbol_item = symbol_map[key]
        symbol_item["nb_files"] += 1

        if (
            symbol_item["first_date"] is None
            or item["first_date"] < symbol_item["first_date"]
        ):
            symbol_item["first_date"] = item["first_date"]

        if (
            symbol_item["last_date"] is None
            or item["last_date"] > symbol_item["last_date"]
        ):
            symbol_item["last_date"] = item["last_date"]

    return list(symbol_map.values())

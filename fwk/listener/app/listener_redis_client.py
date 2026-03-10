import json
import redis
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta, timezone, date
from collections import OrderedDict

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import os
import yaml
import logging

from pathlib import Path

from enum import Enum

from core.classes import (
    NormalizedOrderBook,
    NormalizedTrade,
    NormalizedCandle,
    NormalizedCandleBinance,
)

from core.enums import (
    MktDataType,
    ConnectorTYPE,
    ConnectorCapacity,
    CandleType,
)

default_priority_exchange_symbol = """
BTCUSDT:
  - exchange: binance
    listener: listener_prod_binance_rest
    connectortype: ccxt-rest-candle-binance
  - exchange: binanceusdm
    listener: listener_prod_binance_rest_usdm
    connectortype: ccxt-rest
  - exchange: okx
    listener: listener_okx_rest
    connectortype: ccxt-rest
ETHUSDT:
  - exchange: binance
    listener: listener_prod_binance_rest
    connectortype: ccxt-rest-candle-binance
  - exchange: binanceusdm
    listener: listener_prod_binance_rest_usdm
    connectortype: ccxt-rest
  - exchange: okx
    listener: listener_okx_rest
    connectortype: ccxt-rest
"""

PRIORITY_CONFIG_PATH = Path(
    os.path.join(
        os.path.dirname(__file__), "..", "config", "priority_exchange_symbol.yaml"
    )
)


def load_priority_config() -> Dict[str, List[Dict[str, str]]]:
    """Load priority config from yaml file"""
    if not PRIORITY_CONFIG_PATH.exists():
        config = yaml.safe_load(default_priority_exchange_symbol)
        return config
    try:
        with open(PRIORITY_CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
        return config or {}
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML priority config file: {e}")
        return {}


# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
extra_log = False


class RedisKeyBuilder:
    """Centralized key generation with consistent patterns"""

    @staticmethod
    def generate_stream_key(
        listener_id: str,
        connector_type: str,
        exchange: str,
        symbol: str,
        data_type: MktDataType,
    ) -> str:
        """Generate key for Redis Streams (trades and orderbooks)"""
        return f"{listener_id}:{connector_type}:{data_type.value}:{exchange}:{symbol}"

    @staticmethod
    def generate_candle_key(
        listener_id: str,
        connector_type: str,
        exchange: str,
        symbol: str,
        data_type: MktDataType,
        timestamp: int,
    ) -> str:
        """Generate key for candle storage (historical data)"""
        return f"{listener_id}:{connector_type}:{data_type.value}:{exchange}:{symbol}:{timestamp}"

    @staticmethod
    def generate_timestamp_key(
        listener_id: str,
        connector_type: str,
        exchange: str,
        symbol: str,
        data_type: MktDataType,
    ) -> str:
        """Generate key for sorted sets storing candle timestamps"""
        return f"{listener_id}:{connector_type}:{data_type.value}_timestamps:{exchange}:{symbol}"

    @staticmethod
    def generate_last_processed_key(
        listener_id: str,
        connector_type: str,
        exchange: str,
        symbol: str,
        data_type: MktDataType,
    ) -> str:
        """Generate key for storing last processed timestamp"""
        return (
            f"{listener_id}:{connector_type}:last_{data_type.value}:{exchange}:{symbol}"
        )

    @staticmethod
    def generate_latest_orderbook_key(
        listener_id: str,
        connector_type: str,
        exchange: str,
        symbol: str,
    ) -> str:
        """Generate key for latest orderbook snapshot"""
        return f"{listener_id}:{connector_type}:latest_orderbook:{exchange}:{symbol}"


class RedisClient:
    def __init__(
        self,
        db: int = 0,
        listener_config: Dict = {},
        priority_config: Dict = {},
        orderbook_stream_maxlen: int = 100,  # Keep last 100 orderbooks
        trade_stream_maxlen: int = 10000,  # Keep last 10k trades
    ):
        self.host = os.getenv("REDIS_HOST")
        self.port = int(os.getenv("REDIS_PORT"))
        self.password = os.getenv("REDIS_PASS")

        # Initialize Redis connection with password
        self.redis = redis.Redis(
            host=self.host,
            port=self.port,
            db=db,
            password=self.password,
            decode_responses=False,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
        )

        # Test connection
        try:
            self.redis.ping()
            logger.info(f"Successfully connected to Redis at {self.host}:{self.port}")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis at {self.host}:{self.port}: {e}")
            raise

        self._listeners_config = listener_config

        self.key_builder = RedisKeyBuilder()
        self.orderbook_stream_maxlen = orderbook_stream_maxlen
        self.trade_stream_maxlen = trade_stream_maxlen

        if priority_config == {}:
            self._priority_config = load_priority_config()
        else:
            self._priority_config = priority_config

    # --- Simplified Stream Storage ---
    def store_market_data(
        self,
        listener_id: str,
        connector_type: str,
        exchange: str,
        symbol: str,
        data_type: MktDataType,
        data: Union[NormalizedOrderBook, NormalizedTrade, NormalizedCandle, List],
    ) -> bool:
        """
        Unified method to store any market data type
        """
        try:
            if data_type in [MktDataType.ORDERBOOK, MktDataType.TRADE]:
                return self._store_in_stream(
                    listener_id, connector_type, exchange, symbol, data_type, data
                )

            elif data_type in [MktDataType.CANDLE, MktDataType.CANDLE_BINANCE]:
                if isinstance(data, list):
                    success = True
                    for item in data:
                        if not self._store_candle(
                            listener_id,
                            connector_type,
                            exchange,
                            symbol,
                            data_type,
                            item,
                        ):
                            success = False
                    return success
                else:
                    return self._store_candle(
                        listener_id, connector_type, exchange, symbol, data_type, data
                    )

            else:
                logger.warning(f"Unsupported data type: {data_type}")
                return False

        except Exception as e:
            logger.error(
                f"Error storing {data_type.value} for {exchange}:{symbol}: {e}"
            )
            return False

    def _store_in_stream(
        self,
        listener_id: str,
        connector_type: str,
        exchange: str,
        symbol: str,
        data_type: MktDataType,
        data: Union[NormalizedOrderBook, NormalizedTrade, List],
    ) -> bool:
        """Store data in Redis Stream - simplified version"""
        if not data:
            return True

        # Convert single item to list
        if not isinstance(data, list):
            data = [data]

        stream_key = self.key_builder.generate_stream_key(
            listener_id, connector_type, exchange, symbol, data_type
        )

        # Set maxlen based on data type
        maxlen = (
            self.orderbook_stream_maxlen
            if data_type == MktDataType.ORDERBOOK
            else self.trade_stream_maxlen
        )

        pipe = self.redis.pipeline()

        for item in data:
            # Store raw msgpack bytes in 'data' field
            pipe.xadd(
                stream_key,
                {"data": item.to_msgpack()},  # Direct binary storage
                maxlen=maxlen,
                approximate=True,
            )

        pipe.execute()

        # Update last processed timestamp
        if data:
            last_item = data[-1]
            last_key = self.key_builder.generate_last_processed_key(
                listener_id, connector_type, exchange, symbol, data_type
            )
            self.redis.set(last_key, str(last_item.timestamp))

        logger.debug(f"Stored {len(data)} {data_type.value} in stream {stream_key}")
        return True

    def _store_candle(
        self,
        listener_id: str,
        connector_type: str,
        exchange: str,
        symbol: str,
        data_type: MktDataType,
        candle: Union[NormalizedCandle, NormalizedCandleBinance],
    ) -> bool:
        """Store candle data"""
        key = self.key_builder.generate_candle_key(
            listener_id, connector_type, exchange, symbol, data_type, candle.timestamp
        )

        if self.redis.exists(key):
            logger.debug(
                f"Candle already exists for {exchange}:{symbol} at {candle.timestamp}"
            )
            return False

        self.redis.set(key, candle.to_msgpack())

        # Add to timestamp index
        ts_key = self.key_builder.generate_timestamp_key(
            listener_id, connector_type, exchange, symbol, data_type
        )
        self.redis.zadd(ts_key, {str(candle.timestamp): candle.timestamp})

        # Update last processed
        last_key = self.key_builder.generate_last_processed_key(
            listener_id, connector_type, exchange, symbol, data_type
        )
        self.redis.set(last_key, str(candle.timestamp))

        return True

    # --- Simplified Data Retrieval ---

    def get_market_data(
        self,
        listener_id: str,
        connector_type: str,
        exchange: str,
        symbol: str,
        data_type: MktDataType,
        n_last: int = 1,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> Optional[Union[List, Any]]:
        """
        Unified method to retrieve market data
        """
        try:
            if data_type in [MktDataType.ORDERBOOK, MktDataType.TRADE]:
                return self._get_from_stream(
                    listener_id, connector_type, exchange, symbol, data_type, n_last
                )

            elif data_type in [MktDataType.CANDLE, MktDataType.CANDLE_BINANCE]:
                if start_time is not None or end_time is not None:
                    return self._get_candles_range(
                        listener_id,
                        connector_type,
                        exchange,
                        symbol,
                        data_type,
                        start_time,
                        end_time,
                    )
                else:
                    return self._get_candles_recent(
                        listener_id, connector_type, exchange, symbol, data_type, n_last
                    )

            else:
                logger.warning(f"Unsupported data type: {data_type}")
                return None

        except Exception as e:
            logger.error(
                f"Error retrieving {data_type.value} for {exchange}:{symbol}: {e}"
            )
            return None

    def _get_from_stream(
        self,
        listener_id: str,
        connector_type: str,
        exchange: str,
        symbol: str,
        data_type: MktDataType,
        n_last: int,
    ) -> Optional[List]:
        """Get data from stream - simplified version"""
        stream_key = self.key_builder.generate_stream_key(
            listener_id, connector_type, exchange, symbol, data_type
        )

        entries = self.redis.xrevrange(stream_key, count=n_last)
        if not entries:
            return None

        items = []
        for _, fields in entries:
            raw_bytes = fields[b"data"]

            # Deserialize based on data type
            if data_type == MktDataType.ORDERBOOK:
                item = NormalizedOrderBook.from_msgpack(raw_bytes)
            elif data_type == MktDataType.TRADE:
                item = NormalizedTrade.from_msgpack(raw_bytes)
            else:
                continue

            if item:
                items.append(item)

        return items

    def _get_candles_recent(
        self,
        listener_id: str,
        connector_type: str,
        exchange: str,
        symbol: str,
        data_type: MktDataType,
        n_last: int,
    ) -> Optional[List]:
        """Get most recent candles"""
        ts_key = self.key_builder.generate_timestamp_key(
            listener_id, connector_type, exchange, symbol, data_type
        )

        recent_timestamps = self.redis.zrevrange(ts_key, 0, n_last - 1)
        if not recent_timestamps:
            return None

        candles = []
        for ts_bytes in recent_timestamps:
            timestamp = int(ts_bytes.decode())

            key = self.key_builder.generate_candle_key(
                listener_id, connector_type, exchange, symbol, data_type, timestamp
            )

            data = self.redis.get(key)
            if data:
                # Map data type to class
                type_map = {
                    MktDataType.CANDLE: NormalizedCandle,
                    MktDataType.CANDLE_BINANCE: NormalizedCandleBinance,
                }
                cls = type_map.get(data_type)
                if cls:
                    candles.append(cls.from_msgpack(data))

        return candles

    def _get_candles_range(
        self,
        listener_id: str,
        connector_type: str,
        exchange: str,
        symbol: str,
        data_type: MktDataType,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> Optional[List]:
        """Get candles within time range"""
        ts_key = self.key_builder.generate_timestamp_key(
            listener_id, connector_type, exchange, symbol, data_type
        )

        start = start_time or "-inf"
        end = end_time or "+inf"

        timestamps = self.redis.zrangebyscore(ts_key, start, end)
        if not timestamps:
            return None

        candles = []
        for ts_bytes in timestamps:
            timestamp = int(ts_bytes.decode())

            key = self.key_builder.generate_candle_key(
                listener_id, connector_type, exchange, symbol, data_type, timestamp
            )

            data = self.redis.get(key)
            if data:
                type_map = {
                    MktDataType.CANDLE: NormalizedCandle,
                    MktDataType.CANDLE_BINANCE: NormalizedCandleBinance,
                }
                cls = type_map.get(data_type)
                if cls:
                    candles.append(cls.from_msgpack(data))

        return candles

    def _to_millis(self, dt: datetime) -> int:
        """Convert datetime to milliseconds timestamp"""
        return int(dt.timestamp() * 1000)

    def _from_millis(self, ts: int) -> datetime:
        """Convert milliseconds timestamp to datetime"""
        return datetime.fromtimestamp(ts / 1000, tz=timezone.utc)

    def get_alive_listeners_for_symbol_for_ob(
        self, symbol: str
    ) -> List[Tuple[str, str]]:
        """Get list of (listener_id, exchange_name) that are alive for the symbol"""
        alive = []
        current_time = datetime.now().timestamp() * 1000  # in ms
        threshold = 120000  # 2min

        symbols = self.get_all_symbols()
        for exchange, sym, listener_id, connector_type in symbols:
            if sym == symbol and listener_id in self._listeners_config:
                ts = self.get_last_update_ts(
                    listener_id,
                    connector_type,
                    exchange,
                    sym,
                    data_type=MktDataType.ORDERBOOK,
                )
                if ts and (current_time - ts) < threshold:
                    alive.append((listener_id, exchange, connector_type))
        return alive

    def get_alive_listeners_for_symbol_for_candles(
        self, symbol: str
    ) -> List[Tuple[str, str]]:
        """Get list of (listener_id, exchange_name) that are alive for the symbol"""
        alive = []
        current_time = datetime.now().timestamp() * 1000  # in ms
        threshold = 120000  # 2min

        symbols = self.get_all_symbols_candles()
        for exchange, sym, listener_id, connector_type in symbols:
            if sym == symbol and listener_id in self._listeners_config:
                ts = self.get_last_update_ts(
                    listener_id,
                    connector_type,
                    exchange,
                    sym,
                    data_type=MktDataType.CANDLE,
                )
                if ts and (current_time - ts) < threshold:
                    alive.append((listener_id, exchange, connector_type))
        return alive

    def get_candle_data_fallback(
        self,
        symbol: str,
        n_last: int = 1,
    ):
        get_all_symbols_candles
        alive_listeners = self.get_alive_listeners_for_symbol(symbol)

        if not alive_listeners:
            logger.warning(f"No alive listeners for symbol {symbol}")
            print("no alive listener")
            return None

        selected_listener_id = None
        selected_exchange = None
        selected_connectortype = None

        priority_config = self._priority_config
        if symbol in priority_config:
            priorities = priority_config[symbol]
            for pri in priorities:
                # Validate required keys
                if not {"exchange", "listener", "connectortype"}.issubset(pri.keys()):
                    logger.warning(f"Malformed priority entry for {symbol}: {pri}")
                    continue

                exchange = pri["exchange"]
                listener = pri["listener"]
                connectortype = pri["connectortype"]

                # `alive_listeners` is a list of (listener_id, exchange)
                if (listener, exchange, connectortype) in alive_listeners:
                    selected_listener_id = listener
                    selected_exchange = exchange
                    selected_connectortype = connectortype
                    break

        if not selected_listener_id:
            # alive_listeners is [(lid, exch), …]
            selected_listener_id, selected_exchange = alive_listeners[0]
            # we do **not** have a connector_type from the priority file,
            # so we fetch it from the config that originally created the listener.
            # The easiest way is to look it up in the original listener config:
            for lid, exch, ct in self._listeners_config.get(symbol, []):
                if lid == selected_listener_id and exch == selected_exchange:
                    selected_connectortype = ct
                    break
            # If we still cannot find a connector_type, default to "ccxt-rest"
            if not selected_connectortype:
                selected_connectortype = "ccxt-rest"

        print(
            selected_listener_id
            + " "
            + selected_connectortype
            + " "
            + selected_exchange
            + " "
            + symbol
            + " "
            + data_type
            + " "
            + str(n_last)
        )

        return self.get_market_data(
            listener_id=selected_listener_id,
            connector_type=selected_connectortype,
            exchange=selected_exchange,
            symbol=symbol,
            data_type=data_type,
            n_last=n_last,
        )

    def get_candles(
        self,
        listener_id: str,
        connector_type: str,
        exchange: str,
        symbol: str,
        data_type: MktDataType,
        n_last: int,
    ) -> Optional[List[NormalizedCandle]]:
        return self._get_candles_recent(
            listener_id, connector_type, exchange, symbol, data_type, n_last
        )

    def get_daily_candle_counts(
        self,
        listener_id: str,
        connector_type: str,
        exchange: str,
        symbol: str,
        datatype: MktDataType,
        n_days: int,
    ) -> OrderedDict[str, int]:
        ts_key = self.key_builder.generate_timestamp_key(
            listener_id, connector_type, exchange, symbol, datatype
        )

        now = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        counts = OrderedDict()

        for i in range(n_days):
            day_start = now - timedelta(days=i)
            day_end = day_start + timedelta(days=1) - timedelta(milliseconds=1)

            start_ts = int(day_start.timestamp() * 1000)
            end_ts = int(day_end.timestamp() * 1000)

            count = self.redis.zcount(ts_key, start_ts, end_ts)
            counts[day_start.strftime("%Y/%m/%d")] = count

        return counts

    def get_all_symbols(self) -> List[tuple]:
        pattern = f"*:latest_orderbook:*"
        keys = self.redis.keys(pattern)
        symbols = []
        for key in keys:
            parts = key.decode("utf-8").split(":")
            if len(parts) >= 5:
                listener_id = parts[0]
                connector_type = parts[1]
                exchange = parts[3]
                symbol = parts[4]
                symbols.append((exchange, symbol, listener_id, connector_type))
        return symbols

    def get_all_symbols_candles(self) -> List[tuple]:
        pattern = f"*:candle_time:*"
        keys = self.redis.keys(pattern)
        symbols = []
        for key in keys:
            parts = key.decode("utf-8").split(":")
            if len(parts) >= 5:
                listener_id = parts[0]
                connector_type = parts[1]
                exchange = parts[3]
                symbol = parts[4]
                symbols.append((exchange, symbol, listener_id, connector_type))
        return symbols

    def get_last_candle_time(
        self, listener_id: str, connector_type: str, exchange: str, symbol: str
    ) -> Optional[int]:
        """Get the timestamp of the last stored candle"""
        key = f"{listener_id}:{connector_type}:candle_time:{exchange}:{symbol}"
        time_str = self.redis.get(key)
        if time_str:
            try:
                return int(time_str)
            except (ValueError, TypeError):
                return None
        return None

    def set_last_candle_time(
        self,
        listener_id: str,
        connector_type: str,
        exchange: str,
        symbol: str,
        timestamp: int,
    ):
        """Store timestamp of last processed candle"""
        key = f"{listener_id}:{connector_type}:candle_time:{exchange}:{symbol}"
        self.redis.set(key, str(timestamp))

    def get_last_stream_timestamp(
        self,
        listener_id: str,
        connector_type: str,
        exchange: str,
        symbol: str,
        data_type: MktDataType,
    ) -> Optional[int]:
        """Get timestamp of last entry in a Redis stream (for trades and orderbooks)"""
        stream_key = self.key_builder.generate_stream_key(
            listener_id, connector_type, exchange, symbol, data_type
        )

        # Get most recent entry (count=1)
        entries = self.redis.xrevrange(stream_key, count=1)
        if not entries:
            return None

        # Stream ID format: "timestamp-sequence" (both parts are integers)
        stream_id = entries[0][0]
        try:
            timestamp_str = stream_id.decode("utf-8")
            timestamp = int(timestamp_str.split("-")[0])
            return timestamp
        except (ValueError, IndexError, AttributeError):
            return None

    def get_daily_candles(
        self,
        listener_id: str,
        connector_type: str,
        exchange: str,
        symbol: str,
        datatype: MktDataType,
        day: datetime,
        intraday: bool,
    ) -> Optional[List[NormalizedCandle]]:
        # Calculate day boundaries
        start_time = day.replace(hour=0, minute=0, second=0, microsecond=0)
        end_time = (day + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        # Convert to timestamps for range query
        start_ts = self._to_millis(start_time)
        end_ts = self._to_millis(end_time)

        ts_key = self.key_builder.generate_timestamp_key(
            listener_id, connector_type, exchange, symbol, datatype
        )

        # Get all timestamps within the day's range
        daily_timestamps = self.redis.zrangebyscore(ts_key, start_ts, end_ts)

        candles = []
        for ts_bytes in daily_timestamps:
            try:
                timestamp = int(ts_bytes.decode())
                candle_key = self.key_builder.generate_candle_key(
                    listener_id, connector_type, exchange, symbol, datatype, timestamp
                )

                data = self.redis.get(candle_key)
                if data:
                    type_map = {
                        MktDataType.CANDLE: NormalizedCandle,
                        MktDataType.CANDLE_BINANCE: NormalizedCandleBinance,
                    }
                    cls = type_map.get(datatype)
                    if cls:
                        candles.append(cls.from_msgpack(data))

            except (ValueError, IndexError) as e:
                logger.error(f"Error processing timestamp {ts_bytes}: {e}")

        return candles

    def get_total_count(
        self, listener_id: str, connector_type: str, data_type: MktDataType
    ) -> int:
        """
        Get the total count of all data points for a given data type across all exchanges and symbols.
        """
        pattern = f"{listener_id}:{connector_type}:{data_type.value}_timestamps:*"
        keys = self.redis.keys(pattern)
        total_count = 0
        for key in keys:
            total_count += self.redis.zcard(key)
        return total_count

    def get_redis_memory_usage(self) -> Dict[str, str]:
        """
        Get Redis memory usage information.
        """
        info = self.redis.info("memory")
        return {
            "used_memory": info.get("used_memory_human", "N/A"),
            "used_memory_peak": info.get("used_memory_peak_human", "N/A"),
            "total_system_memory": info.get("total_system_memory_human", "N/A"),
        }

    def get_nb_candles(
        self,
        listener_id: str,
        connector_type: str,
        exchange: str,
        symbol: str,
        datatype: MktDataType,
        date_str: str,
    ) -> int:
        """
        Get the number of candles for a specific exchange, symbol, and date.
        Date format: YYYY/MM/DD
        """
        try:
            # Parse date string to date object
            date_obj = datetime.strptime(date_str, "%Y/%m/%d").date()
            start_of_day = datetime.combine(date_obj, datetime.min.time())
            end_of_day = start_of_day + timedelta(days=1)

            start_ts = self._to_millis(start_of_day)
            end_ts = self._to_millis(end_of_day)

            ts_key = self.key_builder.generate_timestamp_key(
                listener_id, connector_type, exchange, symbol, datatype
            )
            return self.redis.zcount(ts_key, start_ts, end_ts)
        except Exception as e:
            logger.error(
                f"Error getting candle count for {exchange}:{symbol} on {date_str}: {e}"
            )
            return 0

    def get_total_candle_count(
        self,
        listener_id: str,
        connector_type: str,
        exchange: str,
        datatype: MktDataType,
        symbol: str,
    ) -> int:
        """
        Get the total number of candles stored for a specific exchange and symbol.
        """
        ts_key = self.key_builder.generate_timestamp_key(
            listener_id, connector_type, exchange, symbol, datatype
        )
        return self.redis.zcard(ts_key)

    def get_stream_count(
        self,
        listener_id: str,
        connector_type: str,
        exchange: str,
        data_type: MktDataType,
        symbol: str,
    ) -> int:
        """
        Get the number of entries in the stream for a symbol.
        """
        stream_key = self.key_builder.generate_stream_key(
            listener_id, connector_type, exchange, symbol, data_type
        )
        return self.redis.xlen(stream_key)

    def get_last_update_ts(
        self,
        listener_id: str,
        connector_type: str,
        exchange: str,
        symbol: str,
        data_type: MktDataType = MktDataType.CANDLE_BINANCE,
    ) -> Optional[int]:
        timestamps = []

        if data_type == MktDataType.CANDLE_BINANCE or data_type == MktDataType.CANDLE:
            # Candles - check both CANDLE and CANDLE_BINANCE types
            for candle_type in [MktDataType.CANDLE, MktDataType.CANDLE_BINANCE]:
                ts_key = self.key_builder.generate_timestamp_key(
                    listener_id, connector_type, exchange, symbol, candle_type
                )
                latest = self.redis.zrevrange(ts_key, 0, 0)
                if latest:
                    timestamps.append(int(latest[0]))

        if data_type == MktDataType.TRADE:
            # Trades — from stream
            trade_stream_key = self.key_builder.generate_stream_key(
                listener_id, connector_type, exchange, symbol, MktDataType.TRADE
            )
            latest_entry = self.redis.xrevrange(trade_stream_key, count=1)
            if latest_entry:
                fields = latest_entry[0][1]
                raw_bytes = fields[b"data"]
                trade = NormalizedTrade.from_msgpack(raw_bytes)
                if trade:
                    timestamps.append(trade.timestamp)

        # Orderbook — from stream
        if data_type == MktDataType.ORDERBOOK:
            ob_stream_key = self.key_builder.generate_stream_key(
                listener_id, connector_type, exchange, symbol, MktDataType.ORDERBOOK
            )
            latest_ob_entry = self.redis.xrevrange(ob_stream_key, count=1)
            if latest_ob_entry:
                fields = latest_ob_entry[0][1]
                raw_bytes = fields[b"data"]
                ob = NormalizedOrderBook.from_msgpack(raw_bytes)
                if ob:
                    timestamps.append(ob.timestamp)

        return max(timestamps) if timestamps else None

    def get_last_trades(
        self, exchange: str, symbol: str, n: int = 100
    ) -> List[NormalizedTrade]:
        # Note: This method needs listener_id and connector_type to be provided
        # For backward compatibility, we need to get all listener_ids
        symbols = self.get_all_symbols()

        trades = []
        for exch, sym, listener_id, connector_type in symbols:
            if exch == exchange and sym == symbol:
                stream_key = self.key_builder.generate_stream_key(
                    listener_id, connector_type, exchange, symbol, MktDataType.TRADE
                )
                entries = self.redis.xrevrange(stream_key, count=n)
                for _, fields in entries:
                    raw_bytes = fields[b"data"]
                    trade = NormalizedTrade.from_msgpack(raw_bytes)
                    if trade:
                        trades.append(trade)

        # Sort by timestamp descending and take first n
        trades.sort(key=lambda x: x.timestamp, reverse=True)
        return trades[:n]

    def get_last_trades_quick(
        self,
        exchange: str,
        symbol: str,
        n: int = 100,
        # Optional: provide listener_id and connector_type if known
        listener_id: Optional[str] = None,
        connector_type: Optional[str] = None,
    ) -> List[NormalizedTrade]:
        """
        Efficiently get last n trades for a specific exchange and symbol.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            n: Number of trades to return
            listener_id: Optional - if known, directly access the stream
            connector_type: Optional - if known, directly access the stream
        """

        # If listener_id and connector_type are provided, use direct lookup
        if listener_id and connector_type:
            stream_key = self.key_builder.generate_stream_key(
                listener_id, connector_type, exchange, symbol, MktDataType.TRADE
            )
            return self._get_trades_from_stream(stream_key, n)

        # Otherwise, use pattern matching for efficiency
        # Option 1: Use Redis SCAN with pattern to avoid loading all keys
        pattern = f"*:{MktDataType.TRADE.value}:{exchange}:{symbol}"
        trades = []

        # Use SCAN to efficiently find matching keys
        cursor = 0
        while True:
            cursor, keys = self.redis.scan(cursor=cursor, match=pattern, count=100)

            for key in keys:
                # Get trades from this stream
                stream_trades = self._get_trades_from_stream(key.decode(), n)
                trades.extend(stream_trades)

            if cursor == 0:
                break

        # Alternative Option 2: Maintain an index (more efficient for frequent queries)
        # You could maintain a Redis hash/set that maps (exchange, symbol) -> [stream_keys]

        # Sort by timestamp descending and take first n
        trades.sort(key=lambda x: x.timestamp, reverse=True)
        return trades[:n]

    def _get_trades_from_stream(self, stream_key: str, n: int) -> List[NormalizedTrade]:
        """Helper method to get trades from a specific stream key"""
        entries = self.redis.xrevrange(stream_key, count=n)
        trades = []

        for _, fields in entries:
            raw_bytes = fields[b"data"]
            trade = NormalizedTrade.from_msgpack(raw_bytes)
            if trade:
                trades.append(trade)

        return trades

    def get_monitoring_data(self) -> List[Dict[str, Any]]:
        """
        Get monitoring data for all symbols: symbol, exchange, last_update_ts, nb_candles, nb_trades, nb_ob, nb_ob_stream, ob_snapshot.
        """
        symbols = self.get_all_symbols()
        data = []
        for exchange, symbol, listener_id, connector_type in symbols:
            last_ts = self.get_last_update_ts(
                listener_id,
                connector_type,
                exchange,
                symbol,
                data_type=MktDataType.CANDLE,
            )
            last_ts_ob = self.get_last_update_ts(
                listener_id,
                connector_type,
                exchange,
                symbol,
                data_type=MktDataType.ORDERBOOK,
            )
            last_update = (
                datetime.fromtimestamp(last_ts / 1000).isoformat()
                if last_ts or last_ts_ob
                else "Never"
            )

            # Count candles (both types)
            total_candles = 0
            for candle_type in [MktDataType.CANDLE, MktDataType.CANDLE_BINANCE]:
                total_candles += self.get_total_candle_count(
                    listener_id, connector_type, exchange, candle_type, symbol
                )

            trades = self.get_stream_count(
                listener_id, connector_type, exchange, MktDataType.TRADE, symbol
            )
            ob_count = self.get_stream_count(
                listener_id, connector_type, exchange, MktDataType.ORDERBOOK, symbol
            )
            ob_stream_count = ob_count  # In this implementation, they're the same

            # Get orderbook snapshot from stream
            ob_snapshot = "N/A"
            if ob_count > 0:
                ob_stream_key = self.key_builder.generate_stream_key(
                    listener_id, connector_type, exchange, symbol, MktDataType.ORDERBOOK
                )
                latest_entry = self.redis.xrevrange(ob_stream_key, count=1)
                if latest_entry:
                    fields = latest_entry[0][1]
                    raw_bytes = fields[b"data"]
                    ob = NormalizedOrderBook.from_msgpack(raw_bytes)
                    if ob:
                        bids = ob.bids[:3] if len(ob.bids) > 3 else ob.bids
                        asks = ob.asks[:3] if len(ob.asks) > 3 else ob.asks
                        bids_str = ", ".join(f"{b[0]}:{b[1]}" for b in bids)
                        asks_str = ", ".join(f"{a[0]}:{a[1]}" for a in asks)
                        ob_snapshot = f"Bids: [{bids_str}] Asks: [{asks_str}]"

            data.append(
                {
                    "symbol": symbol,
                    "exchange": exchange,
                    "listener_id": listener_id,
                    "connector_type": connector_type,
                    "last_update_ts": last_update,
                    "nb_candles": total_candles,
                    "nb_trades": trades,
                    "nb_ob": ob_count,
                    "nb_ob_stream": ob_stream_count,
                    "ob_snapshot": ob_snapshot,
                }
            )
        return data


redis_client = RedisClient()

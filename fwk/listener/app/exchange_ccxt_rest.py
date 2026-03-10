import time
import os
from datetime import datetime, timedelta, timezone
from typing import List, Dict
import logging
from pathlib import Path

import ccxt
import yaml

from core.classes import (
    NormalizedOrderBook,
    NormalizedTrade,
    NormalizedCandle,
    NormalizedCandleBinance,
)

from core.enums import (
    ExchangeConnector_REST,
    ConnectorCapacity,
    CandleType,
)

from .ccxt_normalisation import (
    ccxt_normalize_orderbook,
    ccxt_normalize_trades,
    ccxt_normalize_candles,
    ccxt_normalize_candles_binance,
)

logger = logging.getLogger(__name__)
exchange_cache = {}


def load_ccxt_config() -> Dict[str, Dict[str, str]]:
    """Load CCXT API keys from ccxt_keys.conf"""
    try:
        config_dir = os.getenv("CONFIG_PATH", "/app/config")
        path = Path(config_dir) / "ccxt_keys.conf"
        if not os.path.isfile(path):
            logger.warning(f"CCXT config not found: {path}")
            return {}

        with open(path, "r") as f:
            config = yaml.safe_load(f)
            return config.get("exchanges", {})
    except Exception as e:
        logger.error(f"Failed to load CCXT config: {e}")
        return {}


def get_exchange(exchange_name):
    if exchange_name not in exchange_cache:
        exchange_class = getattr(ccxt, exchange_name.lower())
        ccxt_config = load_ccxt_config()

        exchange_kwargs = {}
        if exchange_name.lower() in ccxt_config:
            keys = ccxt_config[exchange_name.lower()]
            exchange_kwargs = {
                "apiKey": keys.get("api_key", ""),
                "secret": keys.get("secret", ""),
            }

        exchange_cache[exchange_name] = exchange_class(exchange_kwargs)
    return exchange_cache[exchange_name]


# Implementation 1
class CCXT_ExchangeConnector_REST(ExchangeConnector_REST):
    def __init__(self):
        return

    def get_capacity(self) -> ConnectorCapacity:
        return ConnectorCapacity.ALL

    def get_candle_type(self) -> CandleType:
        return CandleType.OHLCV

    def get_last_orderbook(
        self, exchange_name: str, symbol: str
    ) -> NormalizedOrderBook:
        try:
            if exchange_name == "oanda":
                raise ValueError("exchange_name cannot be oanda")
            if not symbol:
                raise ValueError("Symbol must be provided")

            exchange = get_exchange(exchange_name)
            orderbook = exchange.fetch_order_book(symbol)
            return ccxt_normalize_orderbook(exchange_name, orderbook)
        except Exception as e:
            print(f"Error orderbook: {exchange_name} {symbol} {e}")
            return NormalizedOrderBook(bids=[], asks=[], timestamp=0)

    def get_last_n_trades(
        self, exchange_name: str, symbol: str, n: int
    ) -> List[NormalizedTrade]:
        try:
            if exchange_name == "oanda":
                raise ValueError("exchange_name cannot be oanda")
            if not symbol:
                raise ValueError("Symbol must be provided")

            exchange = get_exchange(exchange_name)
            trades = exchange.fetch_trades(symbol, limit=n)
            return ccxt_normalize_trades(exchange_name, trades)
        except Exception as e:
            print(f"Error trades:  {exchange_name} {symbol} {e}")
            return []

    def get_last_n_candles(
        self, exchange_name: str, symbol: str, n: int, timeframe: str
    ) -> List[NormalizedCandle]:
        try:
            if exchange_name == "oanda":
                raise ValueError("exchange_name cannot be oanda")
            if not symbol:
                raise ValueError("Symbol must be provided")

            exchange = get_exchange(exchange_name)
            # Fetch candles (CCXT returns newest first)
            candles = exchange.fetch_ohlcv(symbol, timeframe, limit=n)

            # print(f"DEBUG: candles response type: {type(candles)}")
            # print(f"DEBUG: first candle: {candles[0] if candles else 'None'}")
            # print(
            #    f"DEBUG: first candle element types: {[type(x) for x in candles[0]] if candles else 'None'}"
            # )

            # Convert to NormalizedCandle objects
            normalized = ccxt_normalize_candles(exchange_name, candles)
            return normalized
        except Exception as e:
            print(f"Error fetching candles from  {exchange_name} {symbol} : {e}")
            return []

    def get_all_1min_candles_for_day(
        self, exchange_name: str, symbol: str, day: datetime, intraday: bool = False
    ) -> List[NormalizedCandle]:
        """Get all normalized 1-minute candles for a specific day using CCXT

        Args:
            exchange_name: Name of the exchange
            symbol: Trading symbol
            day: Date to fetch candles for
            include_current: Whether to include the currently forming candle
        """
        if exchange_name == "oanda":
            raise ValueError("exchange_name cannot be oanda")
        if not symbol:
            raise ValueError("Symbol must be provided")

        try:
            exchange = get_exchange(exchange_name)
            start_time = datetime(day.year, day.month, day.day)

            if intraday:
                end_time = datetime.now(
                    timezone.utc
                )  # Use now() and specify UTC timezone
            else:
                end_time = start_time + timedelta(
                    days=1
                )  # End of day is tomorrow at the same time as start_time

            print("downloading until :" + str(end_time))
            since = int(start_time.timestamp() * 1000)
            until = int(end_time.timestamp() * 1000)

            all_candles = []
            current_since = since

            while current_since < until:
                try:
                    candles = exchange.fetch_ohlcv(
                        symbol=symbol, timeframe="1m", since=current_since, limit=1440
                    )

                    if not candles:
                        break

                    print(f"DEBUG: candles response type: {type(candles)}")
                    print(f"DEBUG: first candle: {candles[0] if candles else 'None'}")
                    print(
                        f"DEBUG: first candle element types: {[type(x) for x in candles[0]] if candles else 'None'}"
                    )

                    normalized = ccxt_normalize_candles(exchange_name, candles)
                    all_candles.extend(normalized)

                    current_since = candles[-1][0] + 60000
                    time.sleep(exchange.rateLimit / 1000)

                    if candles[-1][0] >= until:
                        break

                except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                    print(f"Error during pagination: {e}")
                    break

            # Filter candles to only include the requested range
            day_candles = [c for c in all_candles if since <= c.timestamp < until]

            if len(day_candles) > 0:
                expected_minutes = int((until - since) / 60000)
                if len(day_candles) < expected_minutes:
                    print(
                        f"Warning: Got {len(day_candles)} candles (expected ~{expected_minutes})"
                    )

            return day_candles

        except AttributeError:
            print(f"Exchange {exchange_name} not supported by CCXT")
            return []
        except Exception as e:
            print(f"Unexpected error: {e}")
            return []

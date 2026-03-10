import time
from datetime import datetime, timedelta, timezone
from typing import List
import logging
import array

import ccxt

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


def get_exchange(exchange_name):
    if exchange_name not in exchange_cache:
        exchange_class = getattr(ccxt, exchange_name.lower())
        exchange_cache[exchange_name] = exchange_class()
    return exchange_cache[exchange_name]


# Implementation 1
class CCXT_ExchangeConnector_REST_candle_binance(ExchangeConnector_REST):
    def __init__(self):
        return

    def get_capacity(self) -> ConnectorCapacity:
        return ConnectorCapacity.ALL

    def get_candle_type(self) -> CandleType:
        return CandleType.BINANCE_KLINE

    def get_last_orderbook(
        self, exchange_name: str, symbol: str
    ) -> NormalizedOrderBook:
        try:
            if exchange_name != "binance":
                raise ValueError("exchange_name cannot be other than binance")
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
            if exchange_name != "binance":
                raise ValueError("exchange_name cannot be other than binance")
            if not symbol:
                raise ValueError("Symbol must be provided")

            exchange = get_exchange(exchange_name)
            trades = exchange.fetch_trades(symbol, limit=n)
            return ccxt_normalize_trades(exchange_name, trades)
        except Exception as e:
            print(f"Error trades:  {exchange_name} {symbol} {e}")
            return []

    def cast_candles(self, candles):
        casted_candles = []

        for candle in candles:
            casted_candle = array.array(
                "d",
                [
                    int(candle[0]),  # timestamp as float (or use 'l' for long)
                    float(candle[1]),  # open
                    float(candle[2]),  # high
                    float(candle[3]),  # low
                    float(candle[4]),  # close
                    float(candle[5]),  # volume
                    int(candle[6]),  # volume
                    float(candle[7]),  # volume
                    float(candle[8]),  # volume
                    float(candle[9]),  # volume
                    float(candle[10]),  # volume
                ],
            )

            casted_candles.append(casted_candle)

        return casted_candles

    def get_last_n_candles(
        self, exchange_name: str, symbol: str, n: int, timeframe: str
    ) -> List[NormalizedCandleBinance]:
        try:
            if exchange_name != "binance":
                raise ValueError("exchange_name cannot be other than binance")
            if not symbol:
                raise ValueError("Symbol must be provided")

            exchange = get_exchange(exchange_name)
            # Fetch candles (CCXT returns newest first)
            # candles = exchange.fetch_ohlcv(symbol, timeframe, limit=n)
            raw_candles = exchange.public_get_klines(
                {"symbol": symbol, "interval": "1m", "limit": n}
            )
            candles = self.cast_candles(raw_candles)

            # print(f"DEBUG: candles response type: {type(candles)}")
            # print(f"DEBUG: first candle: {candles[0] if candles else 'None'}")
            # print(
            #    f"DEBUG: first candle element types: {[type(x) for x in candles[0]] if candles else 'None'}"
            # )

            # Convert to NormalizedCandle objects
            normalized = ccxt_normalize_candles_binance(exchange_name, candles)
            return normalized
        except Exception as e:
            print(f"Error fetching candles from bn  {exchange_name} {symbol} : {e}")
            return []

    def get_all_1min_candles_for_day(
        self, exchange_name: str, symbol: str, day: datetime, intraday: bool = False
    ) -> List[NormalizedCandleBinance]:
        """Get all normalized 1-minute candles for a specific day using CCXT

        Args:
            exchange_name: Name of the exchange
            symbol: Trading symbol
            day: Date to fetch candles for
            include_current: Whether to include the currently forming candle
        """
        if exchange_name != "binance":
            raise ValueError("exchange_name cannot be other than binance")
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

            print(f"downloading until: {end_time}")
            since = int(start_time.timestamp() * 1000)
            until = int(end_time.timestamp() * 1000)

            all_candles = []
            current_since = since

            while current_since < until:
                try:
                    # Debug: Print symbol type and value
                    print(f"DEBUG: symbol={symbol}, type={type(symbol)}")

                    raw_candles = exchange.public_get_klines(
                        {"symbol": str(symbol), "interval": "1m", "limit": 1440}
                    )

                    if not raw_candles:
                        break

                    candles = self.cast_candles(raw_candles)

                    print(f"DEBUG: candles response type: {type(candles)}")
                    print(f"DEBUG: first candle: {candles[0] if candles else 'None'}")
                    print(
                        f"DEBUG: first candle element types: {[type(x) for x in candles[0]] if candles else 'None'}"
                    )

                    normalized = ccxt_normalize_candles_binance(exchange_name, candles)
                    all_candles.extend(normalized)

                    current_since = int(candles[-1][0]) + 60000
                    time.sleep(exchange.rateLimit / 1000)

                    if int(candles[-1][0]) >= until:
                        break

                except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                    print(f"Error during pagination: {e}")
                    break

            # Filter candles to only include the requested range
            day_candles = [c for c in all_candles if since <= int(c.timestamp) < until]

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

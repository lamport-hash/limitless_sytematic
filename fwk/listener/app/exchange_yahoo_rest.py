from datetime import datetime, timedelta
from typing import List
import logging
import pandas as pd

import yfinance as yf

from core.classes import (
    NormalizedOrderBook,
    NormalizedTrade,
    NormalizedCandle,
)
from core.enums import (
    ExchangeConnector_REST,
    ConnectorCapacity,
    CandleType,
)


logger = logging.getLogger(__name__)


class YahooFinance_ExchangeConnector_REST(ExchangeConnector_REST):
    def __init__(self):
        return

    def get_capacity(self) -> ConnectorCapacity:
        return ConnectorCapacity.CANDLES

    def get_candle_type(self) -> CandleType:
        return CandleType.OHLCV

    def get_last_orderbook(
        self, exchange_name: str, symbol: str
    ) -> NormalizedOrderBook:
        logger.warning("Yahoo Finance does not support orderbook data")
        return NormalizedOrderBook(bids=[], asks=[], timestamp=0)

    def get_last_n_trades(
        self, exchange_name: str, symbol: str, n: int
    ) -> List[NormalizedTrade]:
        logger.warning("Yahoo Finance does not support trade data")
        return []

    def get_last_n_candles(
        self, exchange_name: str, symbol: str, n: int, timeframe: str
    ) -> List[NormalizedCandle]:
        if not symbol:
            raise ValueError("Symbol must be provided")

        try:
            interval = self._map_timeframe_to_yahoo_interval(timeframe)
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="max", interval=interval)

            if data.empty:
                logger.warning(
                    f"No data available for {symbol} with timeframe {timeframe}"
                )
                return []

            candles = self._convert_yahoo_data_to_candles(data, symbol)
            return candles[-n:] if len(candles) > n else candles
        except Exception as e:
            logger.error(f"Error fetching candles from Yahoo Finance for {symbol}: {e}")
            return []

    def get_all_1min_candles_for_day(
        self,
        exchange_name: str,
        symbol: str,
        day: datetime,
        intraday: bool = False,
        timeframe: str = "1h",
    ) -> List[NormalizedCandle]:
        if not symbol:
            raise ValueError("Symbol must be provided")

        start_time = datetime(day.year, day.month, day.day)

        if intraday:
            end_time = datetime.utcnow()
        else:
            end_time = start_time + timedelta(days=1)

        try:
            interval = self._map_timeframe_to_yahoo_interval(timeframe)
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_time.strftime("%Y-%m-%d"),
                end=end_time.strftime("%Y-%m-%d"),
                interval=interval,
            )

            if data.empty:
                logger.warning(
                    f"No data available for {symbol} on {day.strftime('%Y-%m-%d')} with timeframe {timeframe}"
                )
                return []

            candles = self._convert_yahoo_data_to_candles(data, symbol)
            return candles
        except Exception as e:
            logger.error(
                f"Error fetching candles for {symbol} on {day.strftime('%Y-%m-%d')}: {e}"
            )
            return []

    def _map_timeframe_to_yahoo_interval(self, timeframe: str) -> str:
        timeframe_map = {
            "1m": "1m",
            "2m": "2m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "4h": "1h",
            "1d": "1d",
            "1wk": "1wk",
            "1mo": "1mo",
            "3mo": "3mo",
        }
        return timeframe_map.get(timeframe, "1d")

    def _convert_yahoo_data_to_candles(
        self, data, symbol: str
    ) -> List[NormalizedCandle]:
        candles = []

        for index, row in data.iterrows():
            timestamp = int(index.timestamp() * 1000)

            candle = NormalizedCandle(
                timestamp=timestamp,
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row["Volume"])
                if "Volume" in row and pd.notna(row["Volume"])
                else 0.0,
            )
            candles.append(candle)

        return candles

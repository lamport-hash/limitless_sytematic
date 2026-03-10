import os
import time
import lzma
import struct
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging

from core.classes import (
    NormalizedOrderBook,
    NormalizedTrade,
    NormalizedCandle,
)
from core.enums import ExchangeConnector_REST, ConnectorCapacity, CandleType

logger = logging.getLogger(__name__)


class Dukascopy_ExchangeConnector_REST(ExchangeConnector_REST):
    """Dukascopy connector for downloading historical forex data"""

    BASE_URL = "https://datafeed.dukascopy.com/datafeed"

    def __init__(self, base_output_dir: str = "/data/candles"):
        """
        Initialize Dukascopy connector

        Args:
            base_output_dir: Base directory for storing downloaded data
        """
        self.base_output_dir = base_output_dir
        self.rate_limit_delay = 0.1  # Be polite to Dukascopy servers

    def get_capacity(self) -> ConnectorCapacity:
        return ConnectorCapacity.CANDLES

    def get_candle_type(self) -> CandleType:
        return CandleType.OHLCV

    def get_last_orderbook(
        self, exchange_name: str, symbol: str
    ) -> NormalizedOrderBook:
        """Dukascopy doesn't support real-time orderbook data"""
        raise NotImplementedError(
            "Dukascopy only provides historical data, not real-time orderbook"
        )

    def get_last_n_trades(
        self, exchange_name: str, symbol: str, n: int
    ) -> List[NormalizedTrade]:
        """Dukascopy doesn't support real-time trade data"""
        raise NotImplementedError(
            "Dukascopy only provides historical data, not real-time trades"
        )

    def get_last_n_candles(
        self, exchange_name: str, symbol: str, n: int, timeframe: str
    ) -> List[NormalizedCandle]:
        """Get recent candles (not implemented for Dukascopy historical data)"""
        raise NotImplementedError(
            "Dukascopy is designed for bulk historical downloads, not recent candles"
        )

    def get_all_1min_candles_for_day(
        self, exchange_name: str, symbol: str, day: datetime, intraday: bool = False
    ) -> List[NormalizedCandle]:
        """
        Get all 1-minute candles for a specific day from Dukascopy

        Args:
            exchange_name: Should be "dukascopy"
            symbol: Forex symbol (e.g., "EURUSD")
            day: Date to fetch candles for
            intraday: Whether to include current day (not used for Dukascopy)

        Returns:
            List of NormalizedCandle objects
        """
        if exchange_name.lower() != "dukascopy":
            raise ValueError("exchange_name must be 'dukascopy' for this connector")

        if not symbol:
            raise ValueError("Symbol must be provided")

        try:
            # Download tick data for the entire day
            ticks = self._download_ticks_for_day(symbol.upper(), day)

            if not ticks:
                logger.warning(
                    f"No tick data for {symbol} on {day.strftime('%Y-%m-%d')}"
                )
                return []

            # Convert ticks to 1-minute OHLCV candles
            candles = self._ticks_to_1min_candles(ticks)

            logger.info(
                f"Generated {len(candles)} 1-minute candles for {symbol} on {day.strftime('%Y-%m-%d')}"
            )
            return candles

        except Exception as e:
            logger.error(
                f"Error fetching Dukascopy candles for {symbol} on {day.strftime('%Y-%m-%d')}: {e}"
            )
            return []

    def _download_ticks_for_day(self, symbol: str, date: datetime) -> List[tuple]:
        """
        Download tick data for a specific day

        Args:
            symbol: Forex symbol (e.g., "EURUSD")
            date: Date to download

        Returns:
            List of (timestamp, price, volume) tuples
        """
        all_ticks = []

        for hour in range(24):
            try:
                url = f"{self.BASE_URL}/{symbol}/{date.year:04d}/{date.month - 1:02d}/{date.day:02d}/{hour:02d}h_ticks.bi5"

                response = requests.get(url, timeout=15)
                time.sleep(self.rate_limit_delay)  # Rate limiting

                if response.status_code != 200:
                    logger.debug(f"HTTP {response.status_code} for {url}")
                    continue

                if len(response.content) < 100:
                    logger.debug(f"File too small: {url}")
                    continue

                # Decompress and parse tick data
                raw = lzma.decompress(response.content)
                hour_ticks = self._parse_bi5_data(raw, date, hour)
                all_ticks.extend(hour_ticks)

            except Exception as e:
                logger.debug(
                    f"Error downloading hour {hour:02d} for {symbol} on {date.strftime('%Y-%m-%d')}: {e}"
                )
                continue

        return all_ticks

    def _parse_bi5_data(self, raw: bytes, date: datetime, hour: int) -> List[tuple]:
        """
        Parse Dukascopy .bi5 tick data format

        Args:
            raw: Raw decompressed data
            date: Date for the data
            hour: Hour of the day

        Returns:
            List of (timestamp, price, volume) tuples
        """
        ticks = []

        for i in range(0, len(raw), 20):
            try:
                # Dukascopy format: >iii ff (big endian)
                ms, bid_i, ask_i, bid_vol, ask_vol = struct.unpack(
                    ">iii ff", raw[i : i + 20]
                )

                # Calculate timestamp
                timestamp = datetime(date.year, date.month, date.day, hour) + timedelta(
                    milliseconds=ms
                )

                # Calculate mid price and volume
                price = (bid_i + ask_i) / 2 / 1e5  # Convert from integer to price
                volume = bid_vol + ask_vol

                ticks.append((timestamp, price, volume))

            except struct.error:
                # Skip malformed data
                continue

        return ticks

    def _ticks_to_1min_candles(self, ticks: List[tuple]) -> List[NormalizedCandle]:
        """
        Convert tick data to 1-minute OHLCV candles

        Args:
            ticks: List of (timestamp, price, volume) tuples

        Returns:
            List of NormalizedCandle objects
        """
        if not ticks:
            return []

        # Create DataFrame for easier resampling
        df = pd.DataFrame(ticks, columns=["timestamp", "price", "volume"])
        df.set_index("timestamp", inplace=True)

        # Resample to 1-minute OHLCV
        ohlc = df["price"].resample("1min").ohlc()
        volume = df["volume"].resample("1min").sum()

        # Combine and clean
        candles_df = pd.concat([ohlc, volume], axis=1)
        candles_df.dropna(inplace=True)

        # Convert to NormalizedCandle objects
        candles = []
        for timestamp, row in candles_df.iterrows():
            candle = NormalizedCandle(
                timestamp=int(timestamp.timestamp() * 1000),  # Convert to milliseconds
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
            candles.append(candle)

        return candles

    def get_supported_symbols(self) -> List[str]:
        """
        Get list of commonly supported Dukascopy symbols

        Returns:
            List of forex symbols
        """
        return [
            "EURUSD",
            "GBPUSD",
            "USDJPY",
            "USDCHF",
            "USDCAD",
            "AUDUSD",
            "NZDUSD",
            "EURGBP",
            "EURJPY",
            "GBPJPY",
            "EURCHF",
            "EURAUD",
            "EURCAD",
            "EURNZD",
            "GBPCHF",
            "GBPAUD",
            "GBPCAD",
            "GBPNZD",
            "CHFJPY",
            "CADJPY",
            "AUDJPY",
            "NZDJPY",
            "AUDCHF",
            "CADCHF",
            "NZDCHF",
            "AUDCAD",
            "AUDNZD",
            "CADNZD",
        ]

    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if symbol is supported by Dukascopy

        Args:
            symbol: Forex symbol to validate

        Returns:
            True if symbol is likely supported
        """
        supported = self.get_supported_symbols()
        return symbol.upper() in supported

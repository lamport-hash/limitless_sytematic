import time
from datetime import datetime, timedelta, timezone
from typing import List
import logging

import requests

from core.classes import (
    NormalizedOrderBook,
    NormalizedTrade,
    NormalizedCandle,
)
from core.enums import ExchangeConnector_REST, ConnectorCapacity, CandleType


logger = logging.getLogger(__name__)


class Oanda_ExchangeConnector_REST(ExchangeConnector_REST):
    def __init__(self, api_token: str, account_id: str = "", demo: bool = True):
        self.api_token = api_token
        self.account_id = account_id
        self.base_url = (
            "https://api-fxtrade.oanda.com/v3"
            if not demo
            else "https://api-fxpractice.oanda.com/v3"
        )
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_token}",
        }
        logger.warning(self.headers)

    def get_capacity(self) -> ConnectorCapacity:
        return ConnectorCapacity.ALL

    def get_candle_type(self) -> CandleType:
        return CandleType.OHLCV

    def get_last_orderbook(
        self, exchange_name: str, symbol: str
    ) -> NormalizedOrderBook:
        """
        Get top of book (order book snapshot) from Oanda API

        Args:
            exchange_name: Always 'Oanda' for this implementation
            symbol: Instrument symbol (e.g., 'EUR_USD')

        Returns:
            NormalizedOrderBook object with top bid/ask
        """
        if exchange_name != "oanda":
            raise ValueError("exchange_name must be oanda")

        if not symbol:
            raise ValueError("Symbol must be provided")

        url = f"{self.base_url}/instruments/{symbol}/candles"
        params = {"count": 3, "price": "B", "granularity": "S5"}

        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        # Parse the response
        data_bid = response.json()

        url = f"{self.base_url}/instruments/{symbol}/candles"
        params = {"count": 3, "price": "A", "granularity": "S5"}

        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        # Parse the response
        data_ask = response.json()

        # candles = [] # in
        bid = None
        ask = None
        timestamp_o = None

        for candle_data in data_bid["candles"]:
            # if candle_data['complete']:
            mid = candle_data["bid"]
            timestamp_o = (
                int(datetime.fromisoformat(candle_data["time"]).timestamp()) * 1000
            )
            # candles.append(NormalizedCandle(
            #    timestamp=timestamp_o,
            #    open=float(mid['o']),
            #    high=float(mid['h']),
            #    low=float(mid['l']),
            #    close=float(mid['c']),
            #    volume=float(candle_data['volume'])
            # ))
            bid = float(mid["c"])

        for candle_data in data_ask["candles"]:
            # if candle_data['complete']:
            mid = candle_data["ask"]
            timestamp_o = (
                int(datetime.fromisoformat(candle_data["time"]).timestamp()) * 1000
            )
            # candles.append(NormalizedCandle(
            #    timestamp=timestamp_o,
            #    open=float(mid['o']),
            #    high=float(mid['h']),
            #    low=float(mid['l']),
            #    close=float(mid['c']),
            #    volume=float(candle_data['volume'])
            # ))
            ask = float(mid["c"])

        if bid is None or ask is None or timestamp_o is None:
            raise ValueError("No orderbook data available")

        tob = NormalizedOrderBook([[bid, 1]], [[ask, 1]], timestamp_o)

        return tob

    def get_last_n_trades(
        self, exchange_name: str, symbol: str, n: int
    ) -> List[NormalizedTrade]:
        if exchange_name != "oanda":
            raise ValueError("exchange_name must be oanda")

        return []

    def get_last_n_candles(
        self, exchange_name: str, symbol: str, n: int, timeframe: str
    ) -> List[NormalizedCandle]:
        """
        Get last N candles from oanda API

        Args:
            exchange_name: Always 'oanda' for this implementation
            symbol: Instrument symbol (e.g., 'EUR_USD')
            n: Number of candles to retrieve
            timeframe: Timeframe for candles (e.g., 'M1', 'M5', 'H1', etc.)

        Returns:
            List of NormalizedCandle objects
        """
        if exchange_name != "oanda":
            raise ValueError("exchange_name must be oanda")

        if not symbol:
            raise ValueError("Symbol must be provided")

        # Map the timeframe to Oanda's granularity parameter
        granularity = "M1"

        # Make the API request
        url = f"{self.base_url}/instruments/{symbol}/candles"
        params = {
            "count": n,
            "price": "M",  # Midpoint candles
            "granularity": granularity,
        }
        # logger.warning(self.headers)

        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()

        # Parse the response
        data = response.json()
        # logger.warning(data)
        candles = []

        for candle_data in data["candles"]:
            # if candle_data['complete']:
            mid = candle_data["mid"]
            timestamp_o = (
                int(datetime.fromisoformat(candle_data["time"]).timestamp()) * 1000
            )
            candles.append(
                NormalizedCandle(
                    timestamp=timestamp_o,
                    open=float(mid["o"]),
                    high=float(mid["h"]),
                    low=float(mid["l"]),
                    close=float(mid["c"]),
                    volume=float(candle_data["volume"]),
                )
            )

        # logger.warning(' retreived nb candles : ' + str(len(candles)))
        return candles

    def get_all_1min_candles_for_day(
        self, exchange_name: str, symbol: str, day: datetime, intraday: bool = False
    ) -> List[NormalizedCandle]:
        if exchange_name != "oanda":
            raise ValueError("exchange_name must be oanda")

        if not symbol:
            raise ValueError("Symbol must be provided")

        start_time = datetime(day.year, day.month, day.day)

        if intraday:
            end_time = datetime.now(timezone.utc)  # Use now() and specify UTC timezone
        else:
            end_time = start_time + timedelta(
                days=1
            )  # End of day is tomorrow at the same time as start_time

        # Map the timeframe to Oanda's granularity parameter
        granularity = "M1"

        # Make the API request
        url = f"{self.base_url}/instruments/{symbol}/candles"
        params = {
            "count": 1440,
            "price": "M",  # Midpoint candles
            "granularity": granularity,
        }
        # logger.warning(self.headers)

        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()

        # Parse the response
        data = response.json()
        # logger.warning(data)
        candles = []

        for candle_data in data["candles"]:
            if candle_data["complete"]:
                mid = candle_data["mid"]
                timestamp_o = (
                    int(datetime.fromisoformat(candle_data["time"]).timestamp()) * 1000
                )
                candles.append(
                    NormalizedCandle(
                        timestamp=timestamp_o,
                        open=float(mid["o"]),
                        high=float(mid["h"]),
                        low=float(mid["l"]),
                        close=float(mid["c"]),
                        volume=float(candle_data["volume"]),
                    )
                )

        # logger.warning(' retreived nb candles : ' + str(len(candles)))
        return candles

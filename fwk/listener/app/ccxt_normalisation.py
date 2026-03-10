import time
from datetime import datetime, timedelta, timezone
from typing import List
import logging


from core.classes import (
    NormalizedOrderBook,
    NormalizedTrade,
    NormalizedCandle,
    NormalizedCandleBinance,
)

logger = logging.getLogger(__name__)


def ccxt_normalize_orderbook(
    exchange_name: str, orderbook: dict, top_n: int = 5
) -> NormalizedOrderBook:
    """Convert CCXT orderbook to normalized format"""
    # CCXT standard format:
    # {
    #   'bids': [[price, amount], ...],
    #   'asks': [[price, amount], ...],
    #   'timestamp': 1642345678901,
    #   'datetime': '2022-01-16T12:34:56.789Z'
    # }
    bids = [[float(b[0]), float(b[1])] for b in orderbook["bids"][:top_n]]
    asks = [[float(a[0]), float(a[1])] for a in orderbook["asks"][:top_n]]
    timestamp = orderbook.get("timestamp", int(time.time() * 1000))

    if timestamp is None:
        # logger.info(f'Timestamp missing from {exchange_name} orderbook, using current time')
        timestamp = int(time.time() * 1000)
    else:
        timestamp = int(timestamp)  # Ensure it's an integer

    return NormalizedOrderBook(bids, asks, timestamp)


def ccxt_normalize_trades(exchange_name: str, trades: list) -> List[NormalizedTrade]:
    """Convert CCXT trades to normalized format"""
    # CCXT standard trade format:
    # {
    #   'id': '12345',
    #   'timestamp': 1642345678901,
    #   'datetime': '2022-01-16T12:34:56.789Z',
    #   'symbol': 'BTC/USDT',
    #   'order': '67890',
    #   'type': 'limit',
    #   'side': 'buy',
    #   'takerOrMaker': 'taker',
    #   'price': 42000.0,
    #   'amount': 0.01,
    #   'cost': 420.0,
    #   'fee': { 'currency': 'USDT', 'cost': 0.42 }
    # }
    normalized = []

    for trade in trades:
        norm_trade = NormalizedTrade(
            trade_id=str(trade["id"]),
            price=float(trade["price"]),
            amount=float(trade["amount"]),
            timestamp=trade["timestamp"],
            side=trade["side"].lower(),
        )
        normalized.append(norm_trade)

    return normalized


def ccxt_normalize_candles(
    exchange_name: str, candles: List[List]
) -> List[NormalizedCandle]:
    """Convert CCXT candles to NormalizedCandle objects (newest first)"""
    normalized = []
    for candle in candles:
        try:
            nc = NormalizedCandle(
                timestamp=candle[0],  # Unix timestamp in milliseconds
                open=candle[1],
                high=candle[2],
                low=candle[3],
                close=candle[4],
                volume=candle[5],
            )
            normalized.append(nc)
        except Exception as e:
            print(f"Error normalizing candle: {e}")
    return normalized


def ccxt_normalize_candles_binance(
    exchange_name: str, candles: List[List]
) -> List[NormalizedCandleBinance]:
    """Convert Binance candles to NormalizedCandleBinance objects (newest first)

    Binance klines API returns 12 fields:
    [0] Open time
    [1] Open price
    [2] High price
    [3] Low price
    [4] Close price
    [5] Volume
    [6] Close time
    [7] Quote asset volume
    [8] Number of trades
    [9] Taker buy base asset volume
    [10] Taker buy quote asset volume
    [11] Ignore
    """
    normalized = []
    for candle in candles:
        try:
            # Validate that we have all 12 fields
            ncb = NormalizedCandleBinance(
                timestamp=int(candle[0]),  # Open time
                open=float(candle[1]),  # Open price
                high=float(candle[2]),  # High price
                low=float(candle[3]),  # Low price
                close=float(candle[4]),  # Close price
                volume=float(candle[5]),  # close time
                timestamp_close=int(candle[6]),  # Open time
                quote_volume=float(candle[7]),  # Quote asset volume
                num_trades=float(candle[8]),  # Number of trades
                taker_buy_volume=float(candle[9]),  # Taker buy base asset volume
                taker_buy_quote_volume=float(
                    candle[10]
                ),  # Taker buy quote asset volume
            )
            normalized.append(ncb)
        except Exception as e:
            logger.error(f"Error normalizing binance candle: {e}")
    return normalized

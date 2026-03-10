from dataclasses import dataclass
from typing import List, Union
import json
import msgpack
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NormalizedOrderBook:
    bids: List[List[float]]
    asks: List[List[float]]
    timestamp: int

    def to_dict(self):
        return {"bids": self.bids, "asks": self.asks, "timestamp": self.timestamp}

    def to_dict_np(self):
        return {
            "bids": np.array(self.bids, dtype=np.float32),
            "asks": np.array(self.asks, dtype=np.float32),
            "timestamp": self.timestamp,
        }

    def to_json(self):
        return json.dumps(self.to_dict())

    def to_msgpack(self) -> bytes:
        """Optimized MessagePack serialization (no intermediate dict)"""
        return msgpack.packb((self.timestamp, self.bids, self.asks), use_bin_type=True)

    @classmethod
    def from_msgpack(cls, data: bytes) -> Union["NormalizedOrderBook", None]:
        """More efficient than going through dict"""
        try:
            unpacked = msgpack.unpackb(data, raw=False)
            return cls(timestamp=unpacked[0], bids=unpacked[1], asks=unpacked[2])
        except Exception as e:
            logger.exception("Failed to unpack NormalizedOrderBook msgpack data")
            return None


@dataclass
class NormalizedTrade:
    trade_id: str
    price: float
    amount: float
    timestamp: int
    side: str

    def to_dict(self):
        return {
            "trade_id": self.trade_id,
            "price": self.price,
            "amount": self.amount,
            "timestamp": self.timestamp,
            "side": self.side,
        }

    def to_json(self):
        return json.dumps(self.to_dict())

    def to_msgpack(self) -> bytes:
        """Optimized serialization using tuple"""
        return msgpack.packb(
            (self.trade_id, self.price, self.amount, self.timestamp, self.side),
            use_bin_type=True,
        )

    @classmethod
    def from_msgpack(cls, data: bytes) -> Union["NormalizedTrade", None]:
        try:
            unpacked = msgpack.unpackb(data, raw=False)
            return cls(
                trade_id=unpacked[0],
                price=unpacked[1],
                amount=unpacked[2],
                timestamp=unpacked[3],
                side=unpacked[4],
            )
        except Exception as e:
            logger.exception("Failed to unpack NormalizedTrade msgpack data")
            return None


@dataclass
class NormalizedCandle:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }

    def to_json(self):
        return json.dumps(self.to_dict())

    def to_msgpack(self) -> bytes:
        """Most compact representation using tuple"""
        return msgpack.packb(
            (self.timestamp, self.open, self.high, self.low, self.close, self.volume)
        )

    @classmethod
    def from_msgpack(cls, data: bytes) -> Union["NormalizedCandle", None]:
        try:
            unpacked = msgpack.unpackb(data)
            return cls(
                timestamp=unpacked[0],
                open=unpacked[1],
                high=unpacked[2],
                low=unpacked[3],
                close=unpacked[4],
                volume=unpacked[5],
            )
        except Exception as e:
            logger.exception("Failed to unpack NormalizedCandle msgpack data")
            return None


@dataclass
class NormalizedCandleBinance:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp_close: int
    quote_volume: float
    num_trades: float
    taker_buy_volume: float
    taker_buy_quote_volume: float

    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "timestamp_close": self.timestamp_close,
            "quote_volume": self.quote_volume,
            "num_trades": self.num_trades,
            "taker_buy_volume": self.taker_buy_volume,
            "taker_buy_quote_volume": self.taker_buy_quote_volume,
        }

    def to_json(self):
        return json.dumps(self.to_dict())

    def to_msgpack(self) -> bytes:
        """Most compact representation using tuple"""
        return msgpack.packb(
            (
                self.timestamp,
                self.open,
                self.high,
                self.low,
                self.close,
                self.volume,
                self.timestamp_close,
                self.quote_volume,
                self.num_trades,
                self.taker_buy_volume,
                self.taker_buy_quote_volume,
            )
        )

    @classmethod
    def from_msgpack(cls, data: bytes) -> Union["NormalizedCandleBinance", None]:
        try:
            unpacked = msgpack.unpackb(data)
            return cls(
                timestamp=unpacked[0],
                open=unpacked[1],
                high=unpacked[2],
                low=unpacked[3],
                close=unpacked[4],
                volume=unpacked[5],
                timestamp_close=unpacked[6],
                quote_volume=unpacked[7],
                num_trades=unpacked[8],
                taker_buy_volume=unpacked[9],
                taker_buy_quote_volume=unpacked[10],
            )
        except Exception as e:
            logger.exception("Failed to unpack NormalizedCandleBinance msgpack data")
            return None

from abc import ABC, abstractmethod
from typing import List
from datetime import datetime
from enum import Enum

from .classes import NormalizedOrderBook, NormalizedTrade, NormalizedCandle


#normalised col names
##################################################

#standard candle columns
g_index_col = "i_minute_i" # normalised index it is a nub of minutes since 01/01/2000
g_close_col = "S_close_f32"
g_open_col = "S_open_f32"
g_high_col = "S_high_f32"
g_low_col = "S_low_f32"
g_volume_col = "S_volume_f64"
g_open_time_col = "S_open_time_i"
g_close_time_col = "S_close_time_i"

#extended binance candle columns
g_qa_vol_col = "S_qa_volume_f64"
g_nt_col = "S_ntrades_i"
g_la_vol_col = "S_lift_asset_volume_f64"
g_lqa_vol_col = "S_lift_qa_volume_f64"

# mid cols
g_mid_col = "F_mid_f32"
g_mid2_col = "F_mid2_f32"




# binance specific additional cols


# default precision
g_precision = "32"


class MktDataTFreq(Enum):
    CANDLE_1MIN="candle_1min"
    CANDLE_1HOUR="candle_1hour"
    CANDLE_1DAY="candle_1day"


class MktDataType(Enum):
    ORDERBOOK = "orderbook"
    TRADE = "trade"
    CANDLE = "candle"
    CANDLE_BINANCE = "candle_binance"


class ConnectorTYPE(Enum):
    CCXT_WS = "ccxt-ws"
    CCXT_REST = "ccxt-rest"
    CCXT_REST_CANDLE_BINANCE = "ccxt-rest-candle-binance"
    OANDA_REST = "oanda-rest"
    DUKASCOPY_REST = "dukascopy-rest"
    YAHOO_REST = "yahoo-rest"
    DOWNLOADER = "downloader"


class ExchangeNAME(Enum):
    BINANCE = "binance"
    FIRSTRATE = "firstrate"
    UNDEFINED = "undefined"


class ConnectorCapacity(Enum):
    CANDLES = "candles"
    TRADES = "trades"
    OB = "ob"
    OB_TRADES = "ob_trades"
    ALL = "all"


class CandleType(Enum):
    OHLCV = "ohlcv"
    BINANCE_KLINE = "binance_kline"


class ExchangeConnector_REST(ABC):
    @abstractmethod
    def get_capacity(self) -> ConnectorCapacity:
        pass

    @abstractmethod
    def get_candle_type(self) -> CandleType:
        pass

    @abstractmethod
    def get_last_orderbook(
        self, exchange_name: str, symbol: str
    ) -> NormalizedOrderBook:
        pass

    @abstractmethod
    def get_last_n_trades(
        self, exchange_name: str, symbol: str, n: int
    ) -> List[NormalizedTrade]:
        pass

    @abstractmethod
    def get_last_n_candles(
        self, exchange_name: str, symbol: str, n: int, timeframe: str
    ) -> List[NormalizedCandle]:
        pass

    @abstractmethod
    def get_all_1min_candles_for_day(
        self, exchange_name: str, symbol: str, day: datetime, intraday: bool = False
    ) -> List[NormalizedCandle]:
        pass

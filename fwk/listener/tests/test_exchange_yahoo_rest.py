import pytest
from datetime import datetime
from app.exchange_yahoo_rest import YahooFinance_ExchangeConnector_REST
from core.classes import (
    NormalizedOrderBook,
    NormalizedTrade,
    NormalizedCandle,
)
from core.enums import ConnectorCapacity, CandleType


class TestYahooFinanceExchangeConnectorREST:
    def setup_method(self):
        self.connector = YahooFinance_ExchangeConnector_REST()

    def test_init(self):
        connector = YahooFinance_ExchangeConnector_REST()
        assert connector is not None

    def test_get_capacity(self):
        assert self.connector.get_capacity() == ConnectorCapacity.CANDLES

    def test_get_candle_type(self):
        assert self.connector.get_candle_type() == CandleType.OHLCV

    def test_get_last_orderbook(self):
        result = self.connector.get_last_orderbook("yahoo", "BTC-USD")
        assert isinstance(result, NormalizedOrderBook)
        assert result.bids == []
        assert result.asks == []
        assert result.timestamp == 0

    def test_get_last_orderbook_wrong_exchange(self):
        result = self.connector.get_last_orderbook("binance", "BTC-USD")
        assert isinstance(result, NormalizedOrderBook)
        assert result.bids == []
        assert result.asks == []

    def test_get_last_orderbook_no_symbol(self):
        result = self.connector.get_last_orderbook("yahoo", "")
        assert isinstance(result, NormalizedOrderBook)
        assert result.bids == []
        assert result.asks == []

    def test_get_last_n_trades(self):
        result = self.connector.get_last_n_trades("yahoo", "BTC-USD", 10)
        assert result == []

    def test_get_last_n_trades_wrong_exchange(self):
        result = self.connector.get_last_n_trades("binance", "BTC-USD", 10)
        assert result == []

    def test_get_last_n_candles_wrong_exchange(self):
        with pytest.raises(ValueError, match="exchange_name must be yahoo"):
            self.connector.get_last_n_candles("binance", "BTC-USD", 10, "1h")

    def test_get_last_n_candles_no_symbol(self):
        with pytest.raises(ValueError, match="Symbol must be provided"):
            self.connector.get_last_n_candles("yahoo", "", 10, "1h")

    def test_get_all_1min_candles_for_day_wrong_exchange(self):
        day = datetime(2024, 1, 1)
        with pytest.raises(ValueError, match="exchange_name must be yahoo"):
            self.connector.get_all_1min_candles_for_day("binance", "BTC-USD", day)

    def test_get_all_1min_candles_for_day_no_symbol(self):
        day = datetime(2024, 1, 1)
        with pytest.raises(ValueError, match="Symbol must be provided"):
            self.connector.get_all_1min_candles_for_day("yahoo", "", day)

    def test_map_timeframe_to_yahoo_interval(self):
        assert self.connector._map_timeframe_to_yahoo_interval("1m") == "1m"
        assert self.connector._map_timeframe_to_yahoo_interval("1h") == "1h"
        assert self.connector._map_timeframe_to_yahoo_interval("4h") == "1h"
        assert self.connector._map_timeframe_to_yahoo_interval("1d") == "1d"
        assert self.connector._map_timeframe_to_yahoo_interval("1wk") == "1wk"
        assert self.connector._map_timeframe_to_yahoo_interval("1mo") == "1mo"
        assert self.connector._map_timeframe_to_yahoo_interval("3mo") == "3mo"
        assert self.connector._map_timeframe_to_yahoo_interval("invalid") == "1h"

    def test_get_all_1min_candles_for_day_default_timeframe(self, caplog):
        day = datetime(2024, 1, 1)
        result = self.connector.get_all_1min_candles_for_day("yahoo", "BTC-USD", day)
        assert isinstance(result, list)
        assert all(isinstance(c, NormalizedCandle) for c in result) if result else True

    def test_get_all_1min_candles_for_day_with_timeframe(self, caplog):
        day = datetime(2024, 1, 1)
        result = self.connector.get_all_1min_candles_for_day(
            "yahoo", "BTC-USD", day, timeframe="1d"
        )
        assert isinstance(result, list)
        assert all(isinstance(c, NormalizedCandle) for c in result) if result else True

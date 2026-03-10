import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from app.exchange_ccxt_rest import (
    ccxt_normalize_orderbook,
    ccxt_normalize_trades,
    ccxt_normalize_candles,
    CCXT_ExchangeConnector_REST,
    get_exchange,
)
from core.classes import (
    NormalizedOrderBook,
    NormalizedTrade,
    NormalizedCandle,
)


class TestNormalizationFunctions:
    def test_ccxt_normalize_orderbook(self):
        orderbook_data = {
            "bids": [[100.0, 1.0], [99.5, 2.0]],
            "asks": [[100.5, 1.5], [101.0, 0.5]],
            "timestamp": 1640995200000,
        }
        result = ccxt_normalize_orderbook("binance", orderbook_data)
        assert isinstance(result, NormalizedOrderBook)
        assert result.bids == [[100.0, 1.0], [99.5, 2.0]]
        assert result.asks == [[100.5, 1.5], [101.0, 0.5]]
        assert result.timestamp == 1640995200000

    def test_ccxt_normalize_orderbook_no_timestamp(self):
        orderbook_data = {
            "bids": [[100.0, 1.0]],
            "asks": [[100.5, 1.5]],
            "timestamp": None,
        }
        with patch("time.time", return_value=1640995200.0):
            result = ccxt_normalize_orderbook("binance", orderbook_data)
            assert result.timestamp == 1640995200000

    def test_ccxt_normalize_trades(self):
        trades_data = [
            {
                "id": "123",
                "price": 100.0,
                "amount": 1.0,
                "timestamp": 1640995200000,
                "side": "buy",
            },
            {
                "id": "124",
                "price": 101.0,
                "amount": 0.5,
                "timestamp": 1640995260000,
                "side": "sell",
            },
        ]
        result = ccxt_normalize_trades("binance", trades_data)
        assert len(result) == 2
        assert isinstance(result[0], NormalizedTrade)
        assert result[0].trade_id == "123"
        assert result[0].price == 100.0
        assert result[0].amount == 1.0
        assert result[0].timestamp == 1640995200000
        assert result[0].side == "buy"

    def test_ccxt_normalize_candles(self):
        candles_data = [
            [1640995200000, 100.0, 105.0, 95.0, 102.0, 1000.0],
            [1640995260000, 102.0, 107.0, 98.0, 104.0, 1200.0],
        ]
        result = ccxt_normalize_candles("binance", candles_data)
        assert len(result) == 2
        assert isinstance(result[0], NormalizedCandle)
        assert result[0].timestamp == 1640995200000
        assert result[0].open == 100.0
        assert result[0].high == 105.0
        assert result[0].low == 95.0
        assert result[0].close == 102.0
        assert result[0].volume == 1000.0


class TestCCXTExchangeConnectorREST:
    def setup_method(self):
        self.connector = CCXT_ExchangeConnector_REST()

    @patch("app.exchange_ccxt_rest.get_exchange")
    def test_get_last_orderbook_success(self, mock_get_exchange):
        mock_exchange = Mock()
        mock_exchange.fetch_order_book.return_value = {
            "bids": [[100.0, 1.0]],
            "asks": [[101.0, 1.0]],
            "timestamp": 1640995200000,
        }
        mock_get_exchange.return_value = mock_exchange

        result = self.connector.get_last_orderbook("binance", "BTCUSDT")
        assert isinstance(result, NormalizedOrderBook)
        assert result.bids == [[100.0, 1.0]]
        assert result.asks == [[101.0, 1.0]]

    def test_get_last_orderbook_oanda_error(self):
        result = self.connector.get_last_orderbook("oanda", "BTCUSDT")
        assert isinstance(result, NormalizedOrderBook)
        assert result.bids == []
        assert result.asks == []
        assert result.timestamp == 0

    def test_get_last_orderbook_no_symbol(self):
        result = self.connector.get_last_orderbook("binance", "")
        assert isinstance(result, NormalizedOrderBook)
        assert result.bids == []
        assert result.asks == []
        assert result.timestamp == 0

    @patch("app.exchange_ccxt_rest.get_exchange")
    def test_get_last_orderbook_exception(self, mock_get_exchange):
        mock_exchange = Mock()
        mock_exchange.fetch_order_book.side_effect = Exception("API error")
        mock_get_exchange.return_value = mock_exchange

        result = self.connector.get_last_orderbook("binance", "BTCUSDT")
        assert isinstance(result, NormalizedOrderBook)
        assert result.bids == []
        assert result.asks == []
        assert result.timestamp == 0

    @patch("app.exchange_ccxt_rest.get_exchange")
    def test_get_last_n_trades_success(self, mock_get_exchange):
        mock_exchange = Mock()
        mock_exchange.fetch_trades.return_value = [
            {
                "id": "123",
                "price": 100.0,
                "amount": 1.0,
                "timestamp": 1640995200000,
                "side": "buy",
            }
        ]
        mock_get_exchange.return_value = mock_exchange

        result = self.connector.get_last_n_trades("binance", "BTCUSDT", 10)
        assert len(result) == 1
        assert isinstance(result[0], NormalizedTrade)

    def test_get_last_n_trades_oanda_error(self):
        result = self.connector.get_last_n_trades("oanda", "BTCUSDT", 10)
        assert result == []

    @patch("app.exchange_ccxt_rest.get_exchange")
    def test_get_last_n_trades_exception(self, mock_get_exchange):
        mock_exchange = Mock()
        mock_exchange.fetch_trades.side_effect = Exception("API error")
        mock_get_exchange.return_value = mock_exchange

        result = self.connector.get_last_n_trades("binance", "BTCUSDT", 10)
        assert result == []

    @patch("app.exchange_ccxt_rest.get_exchange")
    def test_get_last_n_candles_success(self, mock_get_exchange):
        mock_exchange = Mock()
        mock_exchange.fetch_ohlcv.return_value = [
            [1640995200000, 100.0, 105.0, 95.0, 102.0, 1000.0]
        ]
        mock_get_exchange.return_value = mock_exchange

        result = self.connector.get_last_n_candles("binance", "BTCUSDT", 10, "1m")
        assert len(result) == 1
        assert isinstance(result[0], NormalizedCandle)

    def test_get_last_n_candles_oanda_error(self):
        result = self.connector.get_last_n_candles("oanda", "BTCUSDT", 10, "1m")
        assert result == []

    @patch("app.exchange_ccxt_rest.get_exchange")
    def test_get_last_n_candles_exception(self, mock_get_exchange):
        mock_exchange = Mock()
        mock_exchange.fetch_ohlcv.side_effect = Exception("API error")
        mock_get_exchange.return_value = mock_exchange

        result = self.connector.get_last_n_candles("binance", "BTCUSDT", 10, "1m")
        assert result == []

    def test_get_all_1min_candles_for_day_oanda_error(self):
        day = datetime(2022, 1, 1)
        with pytest.raises(ValueError, match="exchange_name cannot be oanda"):
            self.connector.get_all_1min_candles_for_day("oanda", "BTCUSDT", day)

    def test_get_all_1min_candles_for_day_no_symbol(self):
        day = datetime(2022, 1, 1)
        with pytest.raises(ValueError, match="Symbol must be provided"):
            self.connector.get_all_1min_candles_for_day("binance", "", day)


class TestGetExchange:
    @patch("app.exchange_ccxt_rest.exchange_cache", {})
    @patch("ccxt.binance")
    def test_get_exchange_new(self, mock_ccxt_class):
        mock_exchange = Mock()
        mock_ccxt_class.return_value = mock_exchange

        result = get_exchange("binance")
        assert result == mock_exchange
        mock_ccxt_class.assert_called_once()

    @patch("app.exchange_ccxt_rest.exchange_cache")
    def test_get_exchange_cached(self, mock_cache):
        mock_exchange = Mock()
        mock_cache.__getitem__.return_value = mock_exchange
        mock_cache.__contains__.return_value = True

        result = get_exchange("binance")
        assert result == mock_exchange

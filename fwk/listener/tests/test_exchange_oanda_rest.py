import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from app.exchange_oanda_rest import Oanda_ExchangeConnector_REST
from core.classes import (
    NormalizedOrderBook,
    NormalizedTrade,
    NormalizedCandle,
)


class TestOandaExchangeConnectorREST:
    def setup_method(self):
        self.api_token = "test_token"
        self.account_id = "test_account"
        self.connector = Oanda_ExchangeConnector_REST(
            self.api_token, self.account_id, demo=True
        )

    def test_init_demo(self):
        connector = Oanda_ExchangeConnector_REST("token", "acc", demo=True)
        assert connector.api_token == "token"
        assert connector.account_id == "acc"
        assert connector.base_url == "https://api-fxpractice.oanda.com/v3"
        assert connector.headers == {
            "Content-Type": "application/json",
            "Authorization": "Bearer token",
        }

    def test_init_live(self):
        connector = Oanda_ExchangeConnector_REST("token", "acc", demo=False)
        assert connector.base_url == "https://api-fxtrade.oanda.com/v3"

    @patch("requests.get")
    def test_get_last_orderbook_success(self, mock_get):
        # Mock responses for bid and ask
        bid_response = Mock()
        bid_response.json.return_value = {
            "candles": [
                {"time": "2022-01-01T00:00:00.000000000Z", "bid": {"c": "1.0500"}}
            ]
        }
        ask_response = Mock()
        ask_response.json.return_value = {
            "candles": [
                {"time": "2022-01-01T00:00:00.000000000Z", "ask": {"c": "1.0510"}}
            ]
        }
        mock_get.side_effect = [bid_response, ask_response]

        result = self.connector.get_last_orderbook("oanda", "EUR_USD")
        assert isinstance(result, NormalizedOrderBook)
        assert result.bids == [[1.05, 1]]
        assert result.asks == [[1.051, 1]]
        assert result.timestamp == 1640995200000

    def test_get_last_orderbook_wrong_exchange(self):
        with pytest.raises(ValueError, match="exchange_name must be oanda"):
            self.connector.get_last_orderbook("binance", "EUR_USD")

    def test_get_last_orderbook_no_symbol(self):
        with pytest.raises(ValueError, match="Symbol must be provided"):
            self.connector.get_last_orderbook("oanda", "")

    @patch("requests.get")
    def test_get_last_orderbook_request_exception(self, mock_get):
        mock_get.side_effect = Exception("API error")
        with pytest.raises(Exception):
            self.connector.get_last_orderbook("oanda", "EUR_USD")

    def test_get_last_n_trades(self):
        result = self.connector.get_last_n_trades("oanda", "EUR_USD", 10)
        assert result == []

    def test_get_last_n_trades_wrong_exchange(self):
        with pytest.raises(ValueError, match="exchange_name must be oanda"):
            self.connector.get_last_n_trades("binance", "EUR_USD", 10)

    @patch("requests.get")
    def test_get_last_n_candles_success(self, mock_get):
        response = Mock()
        response.json.return_value = {
            "candles": [
                {
                    "time": "2022-01-01T00:00:00.000000000Z",
                    "mid": {"o": "1.0500", "h": "1.0520", "l": "1.0480", "c": "1.0510"},
                    "volume": 100,
                }
            ]
        }
        mock_get.return_value = response

        result = self.connector.get_last_n_candles("oanda", "EUR_USD", 1, "M1")
        assert len(result) == 1
        assert isinstance(result[0], NormalizedCandle)
        assert result[0].open == 1.05
        assert result[0].high == 1.052
        assert result[0].low == 1.048
        assert result[0].close == 1.051
        assert result[0].volume == 100
        assert result[0].timestamp == 1640995200000

    def test_get_last_n_candles_wrong_exchange(self):
        with pytest.raises(ValueError, match="exchange_name must be oanda"):
            self.connector.get_last_n_candles("binance", "EUR_USD", 1, "M1")

    def test_get_last_n_candles_no_symbol(self):
        with pytest.raises(ValueError, match="Symbol must be provided"):
            self.connector.get_last_n_candles("oanda", "", 1, "M1")

    @patch("requests.get")
    def test_get_last_n_candles_request_exception(self, mock_get):
        mock_get.side_effect = Exception("API error")
        with pytest.raises(Exception):
            self.connector.get_last_n_candles("oanda", "EUR_USD", 1, "M1")

    @patch("requests.get")
    def test_get_all_1min_candles_for_day_success(self, mock_get):
        response = Mock()
        response.json.return_value = {
            "candles": [
                {
                    "time": "2022-01-01T00:00:00.000000000Z",
                    "complete": True,
                    "mid": {"o": "1.0500", "h": "1.0520", "l": "1.0480", "c": "1.0510"},
                    "volume": 100,
                }
            ]
        }
        mock_get.return_value = response

        day = datetime(2022, 1, 1)
        result = self.connector.get_all_1min_candles_for_day("oanda", "EUR_USD", day)
        assert len(result) == 1
        assert isinstance(result[0], NormalizedCandle)

    def test_get_all_1min_candles_for_day_wrong_exchange(self):
        day = datetime(2022, 1, 1)
        with pytest.raises(ValueError, match="exchange_name must be oanda"):
            self.connector.get_all_1min_candles_for_day("binance", "EUR_USD", day)

    def test_get_all_1min_candles_for_day_no_symbol(self):
        day = datetime(2022, 1, 1)
        with pytest.raises(ValueError, match="Symbol must be provided"):
            self.connector.get_all_1min_candles_for_day("oanda", "", day)

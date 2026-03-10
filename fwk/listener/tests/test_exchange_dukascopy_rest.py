import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import pandas as pd

from app.exchange_dukascopy_rest import Dukascopy_ExchangeConnector_REST
from core.classes import NormalizedCandle


class TestDukascopyConnector:
    """Test cases for Dukascopy connector"""

    def setup_method(self):
        """Setup test fixtures"""
        self.connector = Dukascopy_ExchangeConnector_REST(base_output_dir="/test/data")

    def test_initialization(self):
        """Test connector initialization"""
        assert self.connector.base_output_dir == "/test/data"
        assert self.connector.rate_limit_delay == 0.1

    def test_get_last_orderbook_not_implemented(self):
        """Test that orderbook method raises NotImplementedError"""
        with pytest.raises(
            NotImplementedError, match="Dukascopy only provides historical data"
        ):
            self.connector.get_last_orderbook("dukascopy", "EURUSD")

    def test_get_last_n_trades_not_implemented(self):
        """Test that trades method raises NotImplementedError"""
        with pytest.raises(
            NotImplementedError, match="Dukascopy only provides historical data"
        ):
            self.connector.get_last_n_trades("dukascopy", "EURUSD", 10)

    def test_get_last_n_candles_not_implemented(self):
        """Test that recent candles method raises NotImplementedError"""
        with pytest.raises(
            NotImplementedError,
            match="Dukascopy is designed for bulk historical downloads",
        ):
            self.connector.get_last_n_candles("dukascopy", "EURUSD", 10, "1m")

    def test_invalid_exchange_name(self):
        """Test validation of exchange name"""
        with pytest.raises(ValueError, match="exchange_name must be 'dukascopy'"):
            self.connector.get_all_1min_candles_for_day(
                "binance", "EURUSD", datetime.now()
            )

    def test_missing_symbol(self):
        """Test validation of symbol parameter"""
        with pytest.raises(ValueError, match="Symbol must be provided"):
            self.connector.get_all_1min_candles_for_day("dukascopy", "", datetime.now())

    @patch("app.exchange_dukascopy_rest.requests.get")
    @patch("app.exchange_dukascopy_rest.lzma.decompress")
    def test_download_ticks_for_day_success(self, mock_decompress, mock_get):
        """Test successful tick data download"""
        # Mock HTTP response - make content long enough to pass size check
        long_content = b"x" * 200  # 200 bytes > 100 bytes threshold
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = long_content
        mock_get.return_value = mock_response

        # Mock decompressed data (20 bytes per tick, 2 ticks)
        import struct

        mock_decompressed_data = (
            struct.pack(">i", 0)  # ms (4 bytes, big endian int)
            + struct.pack(">i", 1)  # bid_i (4 bytes, big endian int)
            + struct.pack(">i", 2)  # ask_i (4 bytes, big endian int)
            + struct.pack(">f", 1.0)  # bid_vol (4 bytes, big endian float)
            + struct.pack(">f", 2.0)  # ask_vol (4 bytes, big endian float)
            + struct.pack(">i", 1000)  # ms (4 bytes)
            + struct.pack(">i", 3)  # bid_i (4 bytes)
            + struct.pack(">i", 4)  # ask_i (4 bytes)
            + struct.pack(">f", 1.5)  # bid_vol (4 bytes)
            + struct.pack(">f", 2.5)  # ask_vol (4 bytes)
        )
        mock_decompress.return_value = mock_decompressed_data

        # Test
        date = datetime(2024, 1, 1)
        ticks = self.connector._download_ticks_for_day("EURUSD", date)

        # Debug output
        print(f"Mock decompress called: {mock_decompress.called}")
        print(f"Mock get called: {mock_get.called}")
        print(f"Mock get call count: {mock_get.call_count}")
        print(f"Mock decompress call count: {mock_decompress.call_count}")
        if mock_decompress.called:
            print(f"Mock decompress args: {mock_decompress.call_args}")

        # Verify - we should get some ticks (2 per hour for each successful hour)
        assert len(ticks) > 0
        assert all(isinstance(tick, tuple) and len(tick) == 3 for tick in ticks)
        assert mock_get.call_count == 24  # 24 hours

    @patch("app.exchange_dukascopy_rest.requests.get")
    def test_download_ticks_for_day_http_error(self, mock_get):
        """Test handling of HTTP errors"""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        date = datetime(2024, 1, 1)
        ticks = self.connector._download_ticks_for_day("EURUSD", date)

        assert ticks == []

    @patch("app.exchange_dukascopy_rest.requests.get")
    def test_download_ticks_for_day_small_file(self, mock_get):
        """Test handling of files that are too small"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"too_small"
        mock_get.return_value = mock_response

        date = datetime(2024, 1, 1)
        ticks = self.connector._download_ticks_for_day("EURUSD", date)

        assert ticks == []

    def test_parse_bi5_data(self):
        """Test parsing of .bi5 data format"""
        # Create mock data for 2 ticks
        raw_data = (
            b"\x00\x00\x00\x00\x00\x00\x00\x00"  # ms (8 bytes)
            b"\x00\x00\x00\x01"  # bid_i (4 bytes)
            b"\x00\x00\x00\x02"  # ask_i (4 bytes)
            b"\x3f\x80\x00\x00"  # bid_vol (4 bytes float)
            b"\x40\x00\x00\x00"  # ask_vol (4 bytes float)
            b"\x00\x00\x00\x00\x00\x00\x00\x00"  # ms (8 bytes)
            b"\x00\x00\x00\x03"  # bid_i (4 bytes)
            b"\x00\x00\x00\x04"  # ask_i (4 bytes)
            b"\x3f\x81\x00\x00"  # bid_vol (4 bytes float)
            b"\x40\x01\x00\x00"  # ask_vol (4 bytes float)
        )

        date = datetime(2024, 1, 1)
        hour = 10
        ticks = self.connector._parse_bi5_data(raw_data, date, hour)

        assert len(ticks) == 2
        assert all(isinstance(tick, tuple) and len(tick) == 3 for tick in ticks)
        assert all(isinstance(tick[0], datetime) for tick in ticks)
        assert all(isinstance(tick[1], (int, float)) for tick in ticks)
        assert all(isinstance(tick[2], (int, float)) for tick in ticks)

    def test_parse_bi5_data_malformed(self):
        """Test handling of malformed .bi5 data"""
        raw_data = b"malformed_data"

        date = datetime(2024, 1, 1)
        hour = 10
        ticks = self.connector._parse_bi5_data(raw_data, date, hour)

        assert ticks == []

    def test_ticks_to_1min_candles_empty(self):
        """Test conversion of empty tick list"""
        candles = self.connector._ticks_to_1min_candles([])
        assert candles == []

    def test_ticks_to_1min_candles(self):
        """Test conversion of ticks to 1-minute candles"""
        # Create test ticks for 2 minutes
        base_time = datetime(2024, 1, 1, 10, 0, 0)
        ticks = [
            (base_time, 1.1000, 100),  # Minute 1
            (base_time + timedelta(seconds=30), 1.1005, 200),
            (base_time + timedelta(minutes=1), 1.1010, 150),  # Minute 2
            (base_time + timedelta(minutes=1, seconds=30), 1.1008, 180),
        ]

        candles = self.connector._ticks_to_1min_candles(ticks)

        assert len(candles) == 2
        assert all(isinstance(candle, NormalizedCandle) for candle in candles)

        # Check first candle
        first_candle = candles[0]
        assert first_candle.open == 1.1000
        assert first_candle.high == 1.1005
        assert first_candle.low == 1.1000
        assert first_candle.close == 1.1005
        assert first_candle.volume == 300  # 100 + 200

    def test_get_supported_symbols(self):
        """Test getting supported symbols"""
        symbols = self.connector.get_supported_symbols()

        assert isinstance(symbols, list)
        assert len(symbols) > 0
        assert "EURUSD" in symbols
        assert "GBPUSD" in symbols
        assert all(isinstance(symbol, str) for symbol in symbols)

    def test_validate_symbol_valid(self):
        """Test validation of valid symbols"""
        assert self.connector.validate_symbol("EURUSD") == True
        assert self.connector.validate_symbol("GBPUSD") == True
        assert self.connector.validate_symbol("eurusd") == True  # Case insensitive

    def test_validate_symbol_invalid(self):
        """Test validation of invalid symbols"""
        assert self.connector.validate_symbol("INVALID") == False
        assert self.connector.validate_symbol("") == False
        assert self.connector.validate_symbol("BTCUSDT") == False  # Not a forex pair

    @patch(
        "app.exchange_dukascopy_rest.Dukascopy_ExchangeConnector_REST._download_ticks_for_day"
    )
    @patch(
        "app.exchange_dukascopy_rest.Dukascopy_ExchangeConnector_REST._ticks_to_1min_candles"
    )
    def test_get_all_1min_candles_for_day_success(self, mock_convert, mock_download):
        """Test successful candle retrieval"""
        # Mock tick data
        mock_ticks = [(datetime.now(), 1.1000, 100)]
        mock_download.return_value = mock_ticks

        # Mock candle conversion
        mock_candles = [
            NormalizedCandle(
                timestamp=int(datetime.now().timestamp() * 1000),
                open=1.1000,
                high=1.1005,
                low=1.0995,
                close=1.1002,
                volume=100,
            )
        ]
        mock_convert.return_value = mock_candles

        # Test
        date = datetime(2024, 1, 1)
        candles = self.connector.get_all_1min_candles_for_day(
            "dukascopy", "EURUSD", date
        )

        # Verify
        assert len(candles) == 1
        assert isinstance(candles[0], NormalizedCandle)
        mock_download.assert_called_once_with("EURUSD", date)
        mock_convert.assert_called_once_with(mock_ticks)

    @patch(
        "app.exchange_dukascopy_rest.Dukascopy_ExchangeConnector_REST._download_ticks_for_day"
    )
    def test_get_all_1min_candles_for_day_no_data(self, mock_download):
        """Test handling of no data scenario"""
        mock_download.return_value = []

        date = datetime(2024, 1, 1)
        candles = self.connector.get_all_1min_candles_for_day(
            "dukascopy", "EURUSD", date
        )

        assert candles == []

    @patch(
        "app.exchange_dukascopy_rest.Dukascopy_ExchangeConnector_REST._download_ticks_for_day"
    )
    def test_get_all_1min_candles_for_day_error(self, mock_download):
        """Test handling of download errors"""
        mock_download.side_effect = Exception("Network error")

        date = datetime(2024, 1, 1)
        candles = self.connector.get_all_1min_candles_for_day(
            "dukascopy", "EURUSD", date
        )

        assert candles == []


class TestDukascopyIntegration:
    """Integration tests for Dukascopy connector"""

    def test_connector_with_real_data_structure(self):
        """Test connector with realistic data structures"""
        connector = Dukascopy_ExchangeConnector_REST()

        # Test supported symbols
        symbols = connector.get_supported_symbols()
        assert len(symbols) > 20  # Should have many forex pairs

        # Test symbol validation
        for symbol in ["EURUSD", "GBPUSD", "USDJPY"]:
            assert connector.validate_symbol(symbol) == True

        # Test invalid symbols
        for symbol in ["INVALID", "BTCUSDT", ""]:
            assert connector.validate_symbol(symbol) == False

import pytest
from core.scanner import (
    parse_filename_parts,
    get_data_inventory,
    aggregate_inventory_by_symbol,
)
from app.screener_app import app


def test_parse_filename_parts_basic():
    result = parse_filename_parts("binance_BTCUSDT_20240101_20240102_1min.df.parquet")
    assert result is not None
    assert result["exchange"] == "binance"
    assert result["symbol"] == "BTCUSDT"
    assert result["timeframe"] == "1min"


def test_parse_filename_parts_with_placeholder():
    result = parse_filename_parts(
        "binance_BTCUSDT_20240101_20240102_1min.df.placeholder"
    )
    assert result is not None
    assert result["timeframe"] == "1min"


def test_parse_filename_parts_hourly():
    result = parse_filename_parts("sp500_AAPL_20240101_20240102_1h.df.parquet")
    assert result is not None
    assert result["exchange"] == "sp500"
    assert result["symbol"] == "AAPL"
    assert result["timeframe"] == "1h"


def test_screener_app_creation():
    assert app is not None
    assert app.title == "Candle Data Screener"
    assert len(app.routes) >= 4


def test_screener_routes_exist():
    routes = [route.path for route in app.routes]
    assert "/screener/data_inventory" in routes
    assert "/screener/data" in routes
    assert "/screener/dashboard_screener" in routes
    assert "/screener/health" in routes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

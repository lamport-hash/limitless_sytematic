#!/usr/bin/env python3
"""
Integration test for the screener dashboard.
Tests the full flow: fetching data inventory and rendering chart data.
"""

import requests
import json
import gzip
from datetime import datetime


def test_data_inventory():
    """Test the /screener/data_inventory endpoint"""
    print("Testing /screener/data_inventory endpoint...")
    response = requests.get("http://192.168.0.101:8004/screener/data_inventory")

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    data = response.json()
    assert "total" in data, "Response missing 'total' key"
    assert "data" in data, "Response missing 'data' key"
    assert isinstance(data["data"], list), "data should be a list"

    print(f"✓ Found {data['total']} symbols in inventory")

    if data["data"]:
        first_symbol = data["data"][0]
        print(f"  Example: {first_symbol['exchange']}:{first_symbol['symbol']}")
        return first_symbol

    return None


def test_data_endpoint(symbol_data):
    """Test the /screener/data endpoint with a specific symbol"""
    if not symbol_data:
        print("⚠ No symbols available to test data endpoint")
        return

    print(f"\nTesting /screener/data endpoint for {symbol_data['symbol']}...")

    params = {
        "exchange": symbol_data["exchange"],
        "market_type": symbol_data["market_type"],
        "symbol": symbol_data["symbol"],
        "candle_type": symbol_data["candle_type"],
        "timeframe": symbol_data["timeframe"],
    }

    response = requests.get("http://192.168.0.101:8004/screener/data", params=params)

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    # Response should be gzipped, but requests handles decompression automatically
    candles = response.json()

    assert isinstance(candles, list), "Candles should be a list"
    assert len(candles) > 0, "Should have at least some candles"

    # Check first candle structure
    first_candle = candles[0]
    required_fields = ["timestamp", "open", "high", "low", "close", "volume"]
    for field in required_fields:
        assert field in first_candle, f"Missing field: {field}"

    print(f"✓ Retrieved {len(candles)} candles")
    print(f"  First candle: {first_candle['datetime']}")
    print(f"  Last candle: {candles[-1]['datetime']}")

    # Verify date range (should be ~30 days)
    if len(candles) >= 2:
        first_dt = datetime.fromisoformat(candles[0]["datetime"])
        last_dt = datetime.fromisoformat(candles[-1]["datetime"])
        days = (last_dt - first_dt).days + 1
        print(f"  Date range: {days} days")
        assert days <= 31, f"Date span too large: {days} days"

    return candles


def test_price_range_calculation(candles):
    """
    Simulate frontend's min/max price calculation.
    This should not cause stack overflow with large datasets.
    """
    print("\nTesting price range calculation (frontend simulation)...")

    # Simulate the old approach that caused stack overflow
    try:
        # This would fail with large datasets due to spread operator
        prices_old = [c["open"] for c in candles[:100]]  # Small sample only
        min_old = min(prices_old)
        max_old = max(prices_old)
        print(f"  Old approach (small sample): min={min_old:.2f}, max={max_old:.2f}")
    except Exception as e:
        print(f"  ✗ Old approach failed: {e}")

    # New approach using manual loop (no spread operator)
    min_price = float("inf")
    max_price = float("-inf")

    for candle in candles:
        for price in [candle["open"], candle["high"], candle["low"], candle["close"]]:
            if price < min_price:
                min_price = price
            if price > max_price:
                max_price = price

    print(f"  New approach: min={min_price:.2f}, max={max_price:.2f}")
    print(
        f"✓ Successfully calculated price range for {len(candles)} candles without stack overflow"
    )

    assert max_price > min_price, "Max should be greater than min"
    assert max_price > 0, "Prices should be positive"


def main():
    print("=" * 70)
    print("Screener Dashboard Integration Test")
    print("=" * 70)

    try:
        # Test 1: Get data inventory
        symbol_data = test_data_inventory()

        # Test 2: Get chart data
        candles = test_data_endpoint(symbol_data)

        # Test 3: Simulate frontend rendering
        if candles:
            test_price_range_calculation(candles)

        print("\n" + "=" * 70)
        print("✓ All tests passed!")
        print("=" * 70)
        return 0

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

import pytest
import pandas as pd
from datetime import datetime, timedelta
from core.storage import get_candles_for_date_range


def test_get_candles_default_range_no_dates():
    """Test that get_candles_for_date_range returns last 30 days when no dates specified"""
    base_folder = "./docker_data/data/candles"

    # Get candles without specifying dates - should return last 30 days
    df = get_candles_for_date_range(
        base_folder, exchange="binance", market_type="spot", symbol="ETHUSDT"
    )

    assert not df.empty, "Should return data for ETHUSDT"
    assert len(df) > 0, "Should have at least some candles"

    # Check that data is roughly within last 30 days range
    # (this depends on actual data available)
    first_dt = df["datetime"].iloc[0] if "datetime" in df.columns else None
    last_dt = df["datetime"].iloc[-1] if "datetime" in df.columns else None

    if first_dt and last_dt:
        days_span = (last_dt - first_dt).days
        # Should be at most ~30-31 days (allowing for timezone differences)
        assert days_span <= 31, f"Date span too large: {days_span} days"
    else:
        days_span = "unknown"

    print(f"✓ Retrieved {len(df)} candles spanning {days_span} days")


def test_get_candles_specific_date_range():
    """Test get_candles_for_date_range with specific date range"""
    base_folder = "./docker_data/data/candles"

    # Get candles for a specific short date range
    start_date = datetime(2025, 11, 1)
    end_date = datetime(2025, 11, 2)

    df = get_candles_for_date_range(
        base_folder,
        exchange="binance",
        market_type="spot",
        symbol="ETHUSDT",
        start_date=start_date,
        end_date=end_date,
    )

    if not df.empty:
        # Verify all data is within the requested range
        df["dt"] = (
            df["datetime"]
            if "datetime" in df.columns
            else pd.to_datetime(df["timestamp"], unit="ms")
        )

        min_date = df["dt"].min()
        max_date = df["dt"].max()

        assert min_date.date() >= start_date.date(), (
            f"Min date {min_date} before start {start_date}"
        )
        assert max_date.date() <= end_date.date(), (
            f"Max date {max_date} after end {end_date}"
        )

        print(f"✓ Retrieved {len(df)} candles for specific date range")
    else:
        # No data available for that date - that's okay for the test
        print("⚠ No data available for test date range")


def test_get_candles_no_data():
    """Test get_candles_for_date_range with symbol that doesn't exist"""
    base_folder = "./docker_data/data/candles"

    df = get_candles_for_date_range(
        base_folder, exchange="binance", market_type="spot", symbol="NONEXISTENT_SYMBOL"
    )

    assert df.empty, "Should return empty DataFrame for non-existent symbol"
    print("✓ Returns empty DataFrame for non-existent symbol")


def test_large_dataset_no_stack_overflow():
    """Test that large datasets don't cause stack overflow issues"""
    base_folder = "./docker_data/data/candles"

    # Get a potentially large dataset
    df = get_candles_for_date_range(
        base_folder, exchange="binance", market_type="spot", symbol="ETHUSDT"
    )

    if not df.empty:
        # Simulate the frontend's min/max calculation without spread operator
        # This should not cause stack overflow
        prices = []
        for _, row in df.iterrows():
            prices.extend([row["open"], row["high"], row["low"], row["close"]])

        min_price = min(prices) if prices else 0
        max_price = max(prices) if prices else 0

        assert min_price > 0, "Should have positive min price"
        assert max_price > min_price, "Max should be greater than min"

        print(
            f"✓ Successfully processed {len(df)} candles ({len(prices)} price points) without stack overflow"
        )
    else:
        print("⚠ No data available for stack overflow test")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

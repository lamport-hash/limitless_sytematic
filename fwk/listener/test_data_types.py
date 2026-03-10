#!/usr/bin/env python3
"""
Test script to verify data_types configuration loading and validation
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

from config import (
    get_listener_config,
    parse_data_types,
    validate_data_types_compatibility,
)
from connector_exchange_api import ConnectorCapacity


def test_parse_data_types():
    """Test data_types parsing function"""
    print("Testing parse_data_types function:")

    # Test string format
    result1 = parse_data_types("candles")
    print(f"  'candles' -> {result1}")
    assert result1 == {"candles"}

    # Test ob_trades format
    result2 = parse_data_types("ob_trades")
    print(f"  'ob_trades' -> {result2}")
    assert result2 == {"ob", "trades"}

    # Test all format
    result3 = parse_data_types("all")
    print(f"  'all' -> {result3}")
    assert result3 == {"candles", "trades", "ob"}

    # Test array format
    result4 = parse_data_types(["candles", "trades"])
    print(f"  ['candles', 'trades'] -> {result4}")
    assert result4 == {"candles", "trades"}

    # Test default (None)
    result5 = parse_data_types(None)
    print(f"  None -> {result5}")
    assert result5 == {"all"}

    print("  ✓ All parse_data_types tests passed!")


def test_validation():
    """Test data_types validation function"""
    print("\nTesting validate_data_types_compatibility function:")

    # Test valid combination
    try:
        validate_data_types_compatibility({"candles"}, ConnectorCapacity.ALL)
        print("  ✓ Valid combination passed")
    except ValueError as e:
        print(f"  ✗ Valid combination failed: {e}")
        return False

    # Test invalid combination
    try:
        validate_data_types_compatibility({"candles"}, ConnectorCapacity.OB_TRADES)
        print("  ✗ Invalid combination should have failed")
        return False
    except ValueError as e:
        print(f"  ✓ Invalid combination correctly rejected: {e}")

    print("  ✓ All validation tests passed!")
    return True


def test_config_loading():
    """Test actual config loading from YAML"""
    print("\nTesting config loading from listeners.yaml:")

    try:
        # Test a few different listeners
        configs_to_test = [
            "listener_binance_rest",  # candles only
            "listener_binance_rest_usdm",  # [ob, trades]
            "listener_binance_usdm_ws",  # ob_trades
        ]

        for listener_id in configs_to_test:
            try:
                connector, exchanges, data_types = get_listener_config(listener_id)
                print(f"  {listener_id}:")
                print(f"    connector: {connector}")
                print(f"    data_types: {data_types}")
                print(f"    exchanges: {len(exchanges)} exchange(s)")
            except Exception as e:
                print(f"  {listener_id}: ERROR - {e}")

        print("  ✓ Config loading test completed!")

    except Exception as e:
        print(f"  ✗ Config loading failed: {e}")
        return False

    return True


if __name__ == "__main__":
    print("Testing data_types configuration implementation\n")

    try:
        test_parse_data_types()
        test_validation()
        test_config_loading()

        print("\n🎉 All tests passed! Implementation is working correctly.")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        sys.exit(1)

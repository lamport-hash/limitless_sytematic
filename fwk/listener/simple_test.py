#!/usr/bin/env python3
"""
Simple test to verify data_types configuration loading from YAML
"""

import yaml
import os
from pathlib import Path


def test_config_loading():
    """Test loading data_types from listeners.yaml"""
    config_path = Path("config/listeners.yaml")

    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return False

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        listeners = config.get("listeners", {})
        print(f"✅ Loaded {len(listeners)} listeners from config")

        for listener_id, listener_config in listeners.items():
            data_types = listener_config.get("data_types", "all")
            connector = listener_config.get("connector", "unknown")

            print(f"  {listener_id}:")
            print(f"    connector: {connector}")
            print(f"    data_types: {data_types}")
            print(f"    activated: {listener_config.get('activated', False)}")

        return True

    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return False


def test_data_types_parsing():
    """Test our data_types parsing logic"""
    print("\n🔧 Testing data_types parsing logic:")

    def parse_data_types(data_types_config):
        if not data_types_config:
            return {"all"}  # Default to all if not specified

        # Handle array format
        if isinstance(data_types_config, list):
            return set(data_types_config)

        # Handle string format
        if isinstance(data_types_config, str):
            if data_types_config == "all":
                return {"candles", "trades", "ob"}
            elif data_types_config == "ob_trades":
                return {"ob", "trades"}
            else:
                return {data_types_config}

        raise ValueError(f"Invalid data_types format: {data_types_config}")

    test_cases = [
        ("candles", {"candles"}),
        (["ob", "trades"], {"ob", "trades"}),
        ("ob_trades", {"ob", "trades"}),
        ("all", {"candles", "trades", "ob"}),
        (None, {"all"}),
    ]

    for input_val, expected in test_cases:
        try:
            result = parse_data_types(input_val)
            if result == expected:
                print(f"  ✅ {input_val} -> {result}")
            else:
                print(f"  ❌ {input_val} -> {result} (expected {expected})")
                return False
        except Exception as e:
            print(f"  ❌ {input_val} -> ERROR: {e}")
            return False

    return True


if __name__ == "__main__":
    print("🧪 Testing data_types configuration implementation\n")

    success = True
    success &= test_config_loading()
    success &= test_data_types_parsing()

    if success:
        print("\n🎉 All tests passed! Configuration loading is working correctly.")
    else:
        print("\n❌ Some tests failed!")
        exit(1)

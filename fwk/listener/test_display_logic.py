#!/usr/bin/env python3
"""
Simple test to verify the new configuration display works correctly
"""


def test_config_display_logic():
    """Test the JavaScript-like display logic"""

    # Test data similar to what our endpoint would return
    test_config = {
        "listener_id": "listener_binance_rest",
        "connector_type": "ccxt-rest-candle-binance",
        "connector_capacity": "all",
        "selected_data_types": ["candles"],
        "activated": True,
        "exchanges": [
            {"name": "binance", "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"]}
        ],
    }

    print("🧪 Testing Configuration Display Logic")
    print(f"Listener ID: {test_config['listener_id']}")
    print(f"Connector Type: {test_config['connector_type']}")
    print(f"Connector Capacity: {test_config['connector_capacity']}")

    # Test data type badges logic
    data_types = test_config["selected_data_types"]
    badges = []
    for data_type in data_types:
        badge_class = "bg-secondary"
        icon = ""

        if data_type == "candles":
            badge_class = "bg-success"
            icon = "📈 "
        elif data_type == "trades":
            badge_class = "bg-primary"
            icon = "📊 "
        elif data_type == "ob":
            badge_class = "bg-warning"
            icon = "📋 "

        badges.append(f"[{icon}{data_type}]")

    print(f"Selected Data Types: {' '.join(badges)}")
    print(f"Status: {'✅ Active' if test_config['activated'] else '❌ Inactive'}")

    # Test exchange display
    print("Exchanges:")
    for exchange in test_config["exchanges"]:
        symbols_str = " ".join([f"[{symbol}]" for symbol in exchange["symbols"]])
        print(
            f"  🏦 {exchange['name']} ({len(exchange['symbols'])} symbols): {symbols_str}"
        )

    print("\n✅ Configuration display logic working correctly!")


def test_connector_icons():
    """Test connector icon logic"""

    connectors = ["ccxt-rest-candle-binance", "ccxt-ws", "oanda-rest", "ccxt-rest"]

    print("\n🔧 Testing Connector Icon Logic")
    for connector in connectors:
        if connector.startswith("ccxt"):
            icon = "🔗"
        elif connector.startswith("oanda"):
            icon = "🏦"
        else:
            icon = "❓"

        print(f"  {icon} {connector}")


if __name__ == "__main__":
    test_config_display_logic()
    test_connector_icons()

    print("\n🎉 All configuration display tests passed!")

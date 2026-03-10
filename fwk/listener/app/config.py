import os
import yaml
from typing import List, Tuple, Dict, Set, Optional
from pathlib import Path
from core.enums import ConnectorTYPE, ConnectorCapacity

g_config_path = Path(
    os.path.join(os.path.dirname(__file__), "..", "config", "listeners.yaml")
)


def get_listeners_from_yaml():
    if not g_config_path.exists():
        raise FileNotFoundError(f"Config file not found at {g_config_path}")

    try:
        with open(g_config_path, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML config file: {str(e)}")

    # Extract and return the keys (listener names) as a list
    listeners = list(config.get("listeners", {}).keys())
    return listeners


def parse_data_types(data_types_config) -> Set[str]:
    """Convert data_types config to normalized set of data types"""
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


def validate_data_types_compatibility(
    data_types: Set[str], connector_capacity: ConnectorCapacity
) -> bool:
    """Check if requested data types are supported by the connector"""
    # Map ConnectorCapacity to supported data types
    capacity_mapping = {
        ConnectorCapacity.ALL: {"candles", "trades", "ob"},
        ConnectorCapacity.CANDLES: {"candles"},
        ConnectorCapacity.TRADES: {"trades"},
        ConnectorCapacity.OB: {"ob"},
        ConnectorCapacity.OB_TRADES: {"ob", "trades"},
    }

    supported_types = capacity_mapping.get(connector_capacity, set())

    # Check if all requested types are supported
    unsupported_types = data_types - supported_types
    if unsupported_types:
        raise ValueError(
            f"Connector with capacity '{connector_capacity.value}' does not support "
            f"requested data types: {', '.join(unsupported_types)}. "
            f"Supported types: {', '.join(supported_types)}"
        )

    return True


def get_listener_config(
    listener_id="",
) -> Tuple[str, List[Dict[str, List[str]]], Set[str]]:
    if not g_config_path.exists():
        raise FileNotFoundError(f"Config file not found at {g_config_path}")

    try:
        with open(g_config_path, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML config file: {str(e)}")

    if listener_id == "":
        listener_id = os.getenv("LISTENER_ID")
    if not listener_id:
        raise ValueError("LISTENER_ID environment variable not set")

    if not config or "listeners" not in config:
        raise ValueError("Invalid config format: missing 'listeners' key")

    if listener_id not in config["listeners"]:
        raise ValueError(f"Listener ID '{listener_id}' not found in config file")

    listener_config = config["listeners"][listener_id]
    connector = listener_config["connector"]
    exchanges = listener_config["exchanges"]

    # Parse data_types with default to "all"
    data_types_config = listener_config.get("data_types", "all")
    data_types = parse_data_types(data_types_config)

    supported_connectors = [
        ConnectorTYPE.CCXT_REST.value,
        ConnectorTYPE.CCXT_REST_CANDLE_BINANCE.value,
        ConnectorTYPE.CCXT_WS.value,
        ConnectorTYPE.OANDA_REST.value,
    ]

    if connector not in supported_connectors:
        raise ValueError(f"Not supported connector '{connector}'")

    if not exchanges:
        raise ValueError(f"No valid symbols configured for listener '{listener_id}'")

    return connector, exchanges, data_types


def get_all_listeners_config() -> Dict[str, Dict]:
    """Parse listeners.yaml and return all listeners with their full configuration"""
    if not g_config_path.exists():
        raise FileNotFoundError(f"Config file not found at {g_config_path}")

    try:
        with open(g_config_path, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML config file: {str(e)}")

    if not config or "listeners" not in config:
        raise ValueError("Invalid config format: missing 'listeners' key")

    return config["listeners"]

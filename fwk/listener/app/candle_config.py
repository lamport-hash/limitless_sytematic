import os
import yaml
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Valid connector-exchange combinations
VALID_CONNECTOR_EXCHANGE = {
    "ccxt-rest": ["binance", "okx", "bybit", "coinbase", "kraken"],
    "oanda-rest": ["oanda"],
    "dukascopy-rest": ["dukascopy"],
    "yahoo-rest": ["yahoo", "sp500", "nasdaq100"],
}

# Valid market types
VALID_MARKET_TYPES = ["spot", "futures", "forex"]


class DownloaderConfig:
    """Configuration manager for candle downloader"""

    def __init__(self, config_path: str = "/app/config/downloader.yaml"):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            if not os.path.exists(self.config_path):
                logger.warning(
                    f"Config file {self.config_path} not found, creating default"
                )
                self.create_default_config()

            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)

            # Validate configuration
            self.validate_config(config)
            logger.info(f"Loaded config from {self.config_path}")
            return config

        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise

    def create_default_config(self) -> None:
        """Create default configuration file"""
        default_config = {
            "downloader": {
                "base_folder": "/data/import/downloader/candle_1min",
                "nb_days_to_download": 365,
                "exchanges": [
                    {
                        "name": "binance",
                        "connector": "ccxt-rest",
                        "market_type": "spot",
                        "symbols": ["BTCUSDT", "ETHUSDT"],
                        "start_date": "2024-01-01",
                    }
                ],
            },
            "sleep_settings": {
                "enable_sleep_mode": False,
                "wake_minute_past_hour": 15,
                "priority_new_symbols": True,
            },
            "priority_config": {
                "new_symbol_priority": 0,  # Highest
                "daily_update_priority": 1,  # Medium
                "gap_fill_priority": 2,  # Lowest
            },
        }

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

        with open(self.config_path, "w") as f:
            yaml.dump(default_config, f, default_flow_style=False)

        logger.info(f"Created default config at {self.config_path}")

    def validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration structure and values"""
        if "downloader" not in config:
            raise ValueError("Missing 'downloader' section in config")

        downloader_config = config["downloader"]

        # Validate required fields
        if "base_folder" not in downloader_config:
            raise ValueError("Missing 'base_folder' in downloader config")

        if "nb_days_to_download" not in downloader_config:
            raise ValueError("Missing 'nb_days_to_download' in downloader config")

        if "exchanges" not in downloader_config:
            raise ValueError("Missing 'exchanges' in downloader config")

        # Validate exchanges
        for exchange in downloader_config["exchanges"]:
            self.validate_exchange_config(exchange)

    def validate_exchange_config(self, exchange: Dict[str, Any]) -> None:
        """Validate individual exchange configuration"""
        required_fields = ["name", "connector", "market_type", "symbols", "start_date"]

        for field in required_fields:
            if field not in exchange:
                raise ValueError(f"Missing '{field}' in exchange config")

        # Validate connector-exchange combination
        connector = exchange["connector"]
        exchange_name = exchange["name"]

        if connector not in VALID_CONNECTOR_EXCHANGE:
            raise ValueError(f"Invalid connector: {connector}")

        if exchange_name not in VALID_CONNECTOR_EXCHANGE[connector]:
            raise ValueError(
                f"Exchange {exchange_name} not supported by connector {connector}"
            )

        # Validate market type
        if exchange["market_type"] not in VALID_MARKET_TYPES:
            raise ValueError(f"Invalid market_type: {exchange['market_type']}")

        # Validate symbols list
        if not isinstance(exchange["symbols"], list) or not exchange["symbols"]:
            raise ValueError("Symbols must be a non-empty list")

        # Validate start_date format
        try:
            datetime.strptime(exchange["start_date"], "%Y-%m-%d")
        except ValueError:
            raise ValueError(
                f"Invalid start_date format: {exchange['start_date']}. Use YYYY-MM-DD"
            )

    def save_config_with_backup(self, new_config: Dict[str, Any]) -> str:
        """Save new configuration with backup"""
        try:
            # Create backup filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{self.config_path}.backup.{timestamp}"

            # Create backup if original file exists
            if os.path.exists(self.config_path):
                shutil.copy2(self.config_path, backup_path)
                logger.info(f"Created backup: {backup_path}")

            # Validate new config
            self.validate_config(new_config)

            # Save new config
            with open(self.config_path, "w") as f:
                yaml.dump(new_config, f, default_flow_style=False)

            # Update internal config
            self.config = new_config

            logger.info(f"Saved new config to {self.config_path}")
            return backup_path

        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            raise

    def get_base_folder(self) -> str:
        """Get base folder for candle storage"""
        return self.config["downloader"]["base_folder"]

    def get_nb_days_to_download(self) -> int:
        """Get default number of days to download for new symbols"""
        return self.config["downloader"]["nb_days_to_download"]

    def get_exchanges_config(self) -> List[Dict[str, Any]]:
        """Get list of exchange configurations"""
        return self.config["downloader"]["exchanges"]

    def get_exchange_config(self, exchange_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for specific exchange"""
        for exchange in self.config["downloader"]["exchanges"]:
            if exchange["name"] == exchange_name:
                return exchange
        return None

    def add_symbol_to_config(
        self, connector: str, exchange: str, market_type: str, symbol: str
    ) -> bool:
        """Add new symbol to configuration"""
        try:
            # Validate inputs
            if connector not in VALID_CONNECTOR_EXCHANGE:
                raise ValueError(f"Invalid connector: {connector}")

            if exchange not in VALID_CONNECTOR_EXCHANGE[connector]:
                raise ValueError(
                    f"Exchange {exchange} not supported by connector {connector}"
                )

            if market_type not in VALID_MARKET_TYPES:
                raise ValueError(f"Invalid market_type: {market_type}")

            # Find or create exchange config
            exchange_config = None
            for exc in self.config["downloader"]["exchanges"]:
                if (
                    exc["name"] == exchange
                    and exc["connector"] == connector
                    and exc["market_type"] == market_type
                ):
                    exchange_config = exc
                    break

            if exchange_config is None:
                # Create new exchange config
                start_date = (
                    datetime.now() - timedelta(days=self.get_nb_days_to_download())
                ).strftime("%Y-%m-%d")
                exchange_config = {
                    "name": exchange,
                    "connector": connector,
                    "market_type": market_type,
                    "symbols": [symbol],
                    "start_date": start_date,
                }
                self.config["downloader"]["exchanges"].append(exchange_config)
            else:
                # Add symbol to existing exchange config
                if symbol not in exchange_config["symbols"]:
                    exchange_config["symbols"].append(symbol)
                else:
                    logger.warning(f"Symbol {symbol} already exists in config")
                    return False

            return True

        except Exception as e:
            logger.error(f"Failed to add symbol to config: {e}")
            return False

    def remove_symbol_from_config(self, exchange: str, symbol: str) -> bool:
        """Remove symbol from configuration"""
        try:
            for exc in self.config["downloader"]["exchanges"]:
                if exc["name"] == exchange:
                    if symbol in exc["symbols"]:
                        exc["symbols"].remove(symbol)
                        logger.info(f"Removed symbol {symbol} from exchange {exchange}")
                        return True
                    else:
                        logger.warning(
                            f"Symbol {symbol} not found in exchange {exchange}"
                        )
                        return False

            logger.warning(f"Exchange {exchange} not found in config")
            return False

        except Exception as e:
            logger.error(f"Failed to remove symbol from config: {e}")
            return False

    def get_all_symbols(self) -> List[Dict[str, str]]:
        """Get all symbols from configuration"""
        symbols = []
        for exchange in self.config["downloader"]["exchanges"]:
            for symbol in exchange["symbols"]:
                symbols.append(
                    {
                        "exchange": exchange["name"],
                        "symbol": symbol,
                        "connector": exchange["connector"],
                        "market_type": exchange["market_type"],
                        "start_date": exchange["start_date"],
                    }
                )
        return symbols

    def calculate_start_date_for_new_symbol(self) -> str:
        """Calculate start date for new symbol based on nb_days_to_download"""
        days = self.get_nb_days_to_download()
        start_date = datetime.now() - timedelta(days=days)
        return start_date.strftime("%Y-%m-%d")

    def get_sleep_settings(self) -> Dict[str, Any]:
        """Get sleep settings configuration"""
        return self.config.get(
            "sleep_settings",
            {
                "enable_sleep_mode": False,
                "wake_minute_past_hour": 15,
                "priority_new_symbols": True,
            },
        )

    def get_priority_config(self) -> Dict[str, Any]:
        """Get priority configuration"""
        return self.config.get(
            "priority_config",
            {
                "new_symbol_priority": 0,
                "daily_update_priority": 1,
                "gap_fill_priority": 2,
            },
        )


def validate_connector_exchange(connector: str, exchange: str) -> bool:
    """Validate connector-exchange combination"""
    if connector not in VALID_CONNECTOR_EXCHANGE:
        return False
    return exchange in VALID_CONNECTOR_EXCHANGE[connector]


def get_valid_exchanges_for_connector(connector: str) -> List[str]:
    """Get list of valid exchanges for a connector"""
    return VALID_CONNECTOR_EXCHANGE.get(connector, [])


def get_valid_connectors() -> List[str]:
    """Get list of valid connectors"""
    return list(VALID_CONNECTOR_EXCHANGE.keys())


def get_valid_market_types() -> List[str]:
    """Get list of valid market types"""
    return VALID_MARKET_TYPES

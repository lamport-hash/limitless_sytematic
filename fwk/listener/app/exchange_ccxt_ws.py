import asyncio
import os
import time
import ccxt.pro as ccxtpro
import yaml
from pathlib import Path

from .ccxt_normalisation import (
    ccxt_normalize_orderbook,
    ccxt_normalize_trades,
    ccxt_normalize_candles,
    ccxt_normalize_candles_binance,
)

from .listener_redis_client import redis_client

import logging
from typing import Dict, Any

from core.enums import ConnectorCapacity, MktDataType
from .exchange_ccxt_rest import CCXT_ExchangeConnector_REST

logger = logging.getLogger(__name__)
tasks = []
stop_event = asyncio.Event()


def load_ccxt_config() -> Dict[str, Dict[str, str]]:
    """Load CCXT API keys from ccxt_keys.conf"""
    try:
        config_dir = os.getenv("CONFIG_PATH", "/app/config")
        path = Path(config_dir) / "ccxt_keys.conf"
        if not os.path.isfile(path):
            logger.warning(f"CCXT config not found: {path}")
            return {}

        with open(path, "r") as f:
            config = yaml.safe_load(f)
            return config.get("exchanges", {})
    except Exception as e:
        logger.error(f"Failed to load CCXT config: {e}")
        return {}


def validate_ws_data_types(data_types, exchange_name: str):
    """Validate data types for WebSocket connector"""
    # WS connectors support OB and TRADES (similar to ConnectorCapacity.OB_TRADES)
    supported_types = {"ob", "trades"}

    if data_types is None:
        return  # Default to all supported

    unsupported_types = data_types - supported_types
    if unsupported_types:
        raise ValueError(
            f"WebSocket connector for '{exchange_name}' does not support "
            f"requested data types: {', '.join(unsupported_types)}. "
            f"Supported types: {', '.join(supported_types)}"
        )


async def stream_trades(
    exchange,
    api_symbol: str,
    redis_client,
    listener_id: str,
    connector_type: str,
    redis_symbol: str,
    exchange_name: str,
    verbose=False,
    store_to_redis=True,
):
    if verbose:
        print(f"Starting trade stream for {api_symbol}")

    retry_count = 0
    max_retries = 5

    while not stop_event.is_set() and retry_count < max_retries:
        try:
            trades = await exchange.watch_trades(api_symbol)
            normalized = ccxt_normalize_trades(exchange_name, trades)

            if verbose:
                print(f"Received {len(normalized)} trades for {api_symbol}")
                for trade in normalized:
                    print(f"  Trade: {trade.price} @ {trade.amount} ({trade.side})")

            if store_to_redis:
                redis_client.store_market_data(
                    listener_id,
                    connector_type,
                    exchange_name,
                    redis_symbol,
                    MktDataType.TRADE,
                    normalized,
                )
            retry_count = 0  # Reset retry count on success

        except Exception as e:
            retry_count += 1
            if verbose:
                print(
                    f"Trade stream error for {api_symbol}: {e}, retry {retry_count}/{max_retries}"
                )

            if retry_count < max_retries:
                await asyncio.sleep(min(2**retry_count, 30))  # Exponential backoff
            else:
                if verbose:
                    print(f"Max retries exceeded for {api_symbol} trades")
                break


async def stream_orderbook(
    exchange,
    api_symbol: str,
    redis_client,
    listener_id: str,
    connector_type: str,
    redis_symbol: str,
    exchange_name: str,
    verbose=False,
    store_to_redis=True,
):
    if verbose:
        print(f"Starting orderbook stream for {api_symbol}")

    retry_count = 0
    max_retries = 5

    while not stop_event.is_set() and retry_count < max_retries:
        try:
            ob = await exchange.watch_order_book(api_symbol)
            normalized = ccxt_normalize_orderbook(exchange_name, ob)

            if verbose:
                print(f"Received orderbook for {api_symbol}")
                print(f"  Top 5 Bids: {normalized.bids[:5]}")
                print(f"  Top 5 Asks: {normalized.asks[:5]}")

            if store_to_redis:
                redis_client.store_market_data(
                    listener_id,
                    connector_type,
                    exchange_name,
                    redis_symbol,
                    MktDataType.ORDERBOOK,
                    normalized,
                )
            retry_count = 0

        except Exception as e:
            retry_count += 1
            if verbose:
                print(
                    f"Orderbook stream error for {api_symbol}: {e}, retry {retry_count}/{max_retries}"
                )

            if retry_count < max_retries:
                await asyncio.sleep(min(2**retry_count, 30))
            else:
                if verbose:
                    print(f"Max retries exceeded for {api_symbol} orderbook")
                break


async def stream_all(
    exchange_name: str,
    symbols: list,
    redis_client,
    listener_id: str,
    connector_type: str,
    data_types=None,
    verbose=False,
    store_to_redis=True,
):
    if verbose:
        print(f"Starting stream_all for {exchange_name}")

    # Validate data types for WebSocket connector
    validate_ws_data_types(data_types, exchange_name)

    exchange = None
    tasks = []
    try:
        exchange_class = getattr(ccxtpro, exchange_name.lower())
        ccxt_config = load_ccxt_config()

        exchange_kwargs: Dict[str, Any] = {"enableRateLimit": True}

        if exchange_name.lower() in ccxt_config:
            keys = ccxt_config[exchange_name.lower()]
            exchange_kwargs["apiKey"] = keys.get("api_key", "")
            exchange_kwargs["secret"] = keys.get("secret", "")

        exchange = exchange_class(exchange_kwargs)

        await exchange.load_markets()
        for symbol_tuple in symbols:
            symbol_exchange, symbol, connector, exsymbol = symbol_tuple
            # Create tasks only for configured data types
            if data_types is None or "trades" in data_types:
                trade_task = asyncio.create_task(
                    stream_trades(
                        exchange,
                        exsymbol,
                        redis_client,
                        listener_id,
                        connector_type,
                        symbol,
                        exchange_name,
                        verbose,
                        store_to_redis,
                    )
                )
                tasks.append(trade_task)

            if data_types is None or "ob" in data_types:
                ob_task = asyncio.create_task(
                    stream_orderbook(
                        exchange,
                        exsymbol,
                        redis_client,
                        listener_id,
                        connector_type,
                        symbol,
                        exchange_name,
                        verbose,
                        store_to_redis,
                    )
                )
                tasks.append(ob_task)

        # Wait for stop signal or any task to complete/fail
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        # If we get here, either stop_event was set or a task failed
        if not stop_event.is_set():
            print("A stream task ended unexpectedly")

    except Exception as e:
        print(f"Error in stream_all: {e}")
    finally:
        # Cleanup
        stop_event.set()

        # Cancel all tasks
        for task in tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to be cancelled
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # Close exchange
        if exchange is not None:
            await exchange.close()

        if verbose:
            print(f"Streaming stopped for {exchange_name}")


# Function to stop streaming
def stop_streaming():
    stop_event.set()

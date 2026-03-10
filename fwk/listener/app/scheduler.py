import os
import threading
import time
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional
import logging

from .listener_redis_client import RedisClient
from .exchanges import get_mapped_symbol
from .exchange_ccxt_rest import CCXT_ExchangeConnector_REST
from .exchange_ccxt_rest_candle_binance import (
    CCXT_ExchangeConnector_REST_candle_binance,
)
from .exchange_oanda_rest import Oanda_ExchangeConnector_REST
from core.enums import (
    ConnectorTYPE,
    CandleType,
    MktDataType,
    ConnectorCapacity,
)
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StaticsOrigin(Enum):
    LISTENER = "listener"
    CORE_HTTP = "core_http"
    CORE_API = "core_api"


class DataScheduler:
    def __init__(
        self, redis_client: RedisClient, oanda_config: Dict, listener_id: str = ""
    ):
        self.redis = redis_client
        self.listener_id = listener_id
        self.connector = ""
        self.data_types = {"all"}  # Default to all data types
        self.candle_type = ""
        self.candle_MktDataType = ""
        self.symbols: List[
            Tuple[str, str, str, str]
        ] = []  # List of (exchange, symbol, connector, exsymbol)
        self.lock = threading.Lock()
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.initial_load_complete = False  # Add this flag
        self.ccxt_rest_connector = CCXT_ExchangeConnector_REST()
        self.ccxt_rest_connector_binance = CCXT_ExchangeConnector_REST_candle_binance()
        self.ex_connector = None
        self.ws_task: Optional[asyncio.Task] = None
        self.statics_origin = StaticsOrigin.LISTENER.value
        self.statics_url = os.getenv("STATICS_URL", "http://localhost:8093")
        token_oanda = oanda_config.get("api-key")
        account_id_oanada = oanda_config.get("account-id")
        demo_oanda = oanda_config.get("demo")
        if token_oanda:
            logger.warning("Oanda connector is activated !")
            self.oanda_rest_connector = Oanda_ExchangeConnector_REST(
                token_oanda, account_id_oanada, demo_oanda
            )

    def get_symbols(self) -> List[Tuple[str, str, str, str]]:
        with self.lock:
            return self.symbols.copy()

    def load_initial_candles(self):
        """Load all today's candles on startup"""
        today = datetime.utcnow().date()
        for exchange, symbol, connector, exsymbol in self.get_symbols():
            try:
                print(
                    f"trying to load historical candles for {exchange}:{symbol}:{exsymbol}"
                )
                candles = self.ex_connector.get_all_1min_candles_for_day(
                    exchange,
                    exsymbol,
                    today,
                    intraday=True,  # load only up to now
                )
                if candles:
                    # Store all candles in Redis (with appropriate keys)
                    for candle in candles:
                        # Store each candle with its timestamp in the key
                        self.redis.store_market_data(
                            self.listener_id,
                            self.connector,
                            exchange,
                            symbol,
                            self.candle_MktDataType,
                            candle,
                        )

                    print(
                        f"Loaded {len(candles)} historical candles for {exchange}:{symbol}:{exsymbol}"
                    )
            except Exception as e:
                print(
                    f"Error loading initial candles for load_initial_candles {exchange}:{symbol}:{exsymbol} {e}"
                )

        self.initial_load_complete = True

    def load_initial_candles_for_days(self, n_days: int):
        for exchange, symbol, connector, exsymbol in self.get_symbols():
            self.ex_connector.get_candle_type()
            print(
                f"trying to load historical candles for {exchange}:{symbol}:{exsymbol}"
            )

            # For now, just load candles for the last n_days without count check
            for i in range(n_days):
                day_start = datetime.now(timezone.utc).date() - timedelta(days=i)
                day_start_dt = datetime.combine(day_start, datetime.min.time())

                try:
                    candles = self.ex_connector.get_all_1min_candles_for_day(
                        exchange,
                        exsymbol,
                        day_start_dt,
                        intraday=False,  # load full day
                    )
                    if candles:
                        # Store all candles in Redis (with appropriate keys)
                        for candle in candles:
                            # Store each candle with its timestamp in the key
                            self.redis.store_market_data(
                                self.listener_id,
                                self.connector,
                                exchange,
                                symbol,
                                self.candle_MktDataType,
                                candle,
                            )

                        print(
                            f"Loaded {len(candles)} historical candles for {exchange}:{symbol}:{exsymbol} on {day_start}"
                        )
                except Exception as e:
                    print(
                        f"Error loading initial candles for load_initial_candles_for_days {exchange}:{symbol}:{exsymbol} {e}"
                    )

        self.initial_load_complete = True

    def add_symbol(self, exchange: str, symbol: str):
        with self.lock:
            if self.statics_origin == StaticsOrigin.CORE_HTTP.value:
                import httpx

                try:
                    response = httpx.get(
                        f"{self.statics_url}/v1/refdata/venues/{exchange}/instruments/{symbol}",
                        timeout=5.0,
                    )
                    response.raise_for_status()
                    exsymbol = response.json()["venue_symbol"]
                    print(exsymbol)
                except httpx.ConnectError as e:
                    raise RuntimeError(
                        f"Failed to connect to refdata service at {self.statics_url}. "
                        f"Ensure the service is running and accessible. Error: {str(e)}"
                    )
                except httpx.HTTPStatusError as e:
                    raise RuntimeError(
                        f"HTTP error from refdata service for {exchange}/{symbol}: {str(e)}"
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Unexpected error fetching refdata for {exchange}/{symbol}: {str(e)}"
                    )
            else:
                exsymbol = get_mapped_symbol(exchange, symbol)
                print(exsymbol)
            if (exchange, symbol, self.connector, exsymbol) not in self.symbols:
                self.symbols.append((exchange, symbol, self.connector, exsymbol))

    def remove_symbol(self, exchange: str, symbol: str):
        with self.lock:
            exsymbol = get_mapped_symbol(exchange, symbol)
            if (exchange, symbol, self.connector, exsymbol) in self.symbols:
                self.symbols.remove((exchange, symbol, self.connector, exsymbol))

    def fetch_all_data(self):
        """Fetch data for all symbols based on configured data_types"""
        symbols = self.get_symbols()
        for exchange, symbol, connector, exsymbol in symbols:
            try:
                # Fetch order book if configured
                if "ob" in self.data_types:
                    orderbook = self.ex_connector.get_last_orderbook(exchange, exsymbol)
                    # if exchange == 'binance':
                    #    logger.info('binance' + orderbook.to_json())
                    self.redis.store_market_data(
                        self.listener_id,
                        self.connector,
                        exchange,
                        symbol,
                        MktDataType.ORDERBOOK,
                        orderbook,
                    )

                # Fetch trades if configured
                if "trades" in self.data_types:
                    trades = self.ex_connector.get_last_n_trades(
                        exchange, exsymbol, 100
                    )
                    self.redis.store_market_data(
                        self.listener_id,
                        self.connector,
                        exchange,
                        symbol,
                        MktDataType.TRADE,
                        trades,
                    )
            except Exception as e:
                print(f"Error fetching data for {exchange}:{symbol}:{exsymbol} {e}")

    def fetch_candle_data(self):
        """Fetch candle data and store only when a new candle is detected"""
        # Skip candle fetching if not configured
        if "candles" not in self.data_types:
            return

        symbols = self.get_symbols()

        for exchange, symbol, connector, exsymbol in symbols:
            try:
                last_processed_time = self.redis.get_last_candle_time(
                    self.listener_id, self.connector, exchange, symbol
                )
                # Convert to int if it's a string
                if last_processed_time is not None:
                    last_processed_time = int(last_processed_time)
                print(
                    f"last_processed_time candle for {exchange}:{symbol} is at {last_processed_time}"
                )

                # Fetch the last 2 candles
                candles = self.ex_connector.get_last_n_candles(
                    exchange, exsymbol, 2, "1m"
                )  # need 3 candles for oanda likely 2 for ccxt
                if not candles:
                    logger.warning(
                        f"no candles for {exchange}:{symbol}:{self.connector}"
                    )
                    continue

                current_candle = candles[0]
                forming_candle = candles[1]
                # last_forming_candle = candles[2]
                now_time = datetime.now(timezone.utc)
                # logger.info(
                #    f"Stored candle for {current_candle.timestamp} and {forming_candle.timestamp} localtime {now_time}"
                # )

                # If we haven't processed any candle yet, or if this is a new candle
                if last_processed_time is None or current_candle.timestamp > int(
                    last_processed_time if last_processed_time else 0
                ):
                    # The previous candle is complete and new so store it
                    self.redis.store_market_data(
                        self.listener_id,
                        self.connector,
                        exchange,
                        symbol,
                        self.candle_MktDataType,
                        current_candle,
                    )
                    logger.info(
                        f"Stored completed candle for {exchange}:{symbol} at {current_candle.timestamp}"
                    )

                    # Update our tracking timestamp
                    self.redis.set_last_candle_time(
                        self.listener_id,
                        self.connector,
                        exchange,
                        symbol,
                        current_candle.timestamp,
                    )
                    logger.info(
                        f"Stored completed candle for {exchange}:{symbol} at {current_candle.timestamp}"
                    )

                    # Update our tracking timestamp
                    self.redis.set_last_candle_time(
                        self.listener_id,
                        self.connector,
                        exchange,
                        symbol,
                        current_candle.timestamp,
                    )
                # else:
                # logger.info(
                #    f"No completed candle stored as too old for {exchange}:{symbol} at {current_candle.timestamp}"
                # )

            except Exception as e:
                print(
                    f"Error processing candle data for {exchange}:{symbol}:{exsymbol} {e}"
                )

    def start(self):
        """Start the scheduler thread"""
        if self.running:
            return

        self.running = True

        if self.connector == ConnectorTYPE.CCXT_REST.value:
            self.ex_connector = self.ccxt_rest_connector
        elif self.connector == ConnectorTYPE.CCXT_REST_CANDLE_BINANCE.value:
            self.ex_connector = self.ccxt_rest_connector_binance
        elif self.connector == ConnectorTYPE.OANDA_REST.value:
            self.ex_connector = self.oanda_rest_connector
        elif self.connector == ConnectorTYPE.CCXT_WS.value:
            # WebSocket connector - run async task
            # Validate data_types for WS
            from .exchange_ccxt_ws import validate_ws_data_types

            validate_ws_data_types(self.data_types, self.connector)

            # Get event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Start streaming task
            symbols = self.get_symbols()
            if symbols:
                exchange_name = symbols[0][
                    0
                ]  # Get exchange name from first symbol tuple
                self.ws_task = asyncio.create_task(
                    self.run_task(exchange_name, symbols)
                )
            return
        else:
            print(f"Connector not valid {self.connector}")
            return

        # Validate data_types compatibility with connector capacity
        connector_capacity = self.ex_connector.get_capacity()
        from .config import validate_data_types_compatibility

        validate_data_types_compatibility(self.data_types, connector_capacity)

        self.candle_type = self.ex_connector.get_candle_type()
        if CandleType.BINANCE_KLINE == self.candle_type:
            self.candle_MktDataType = MktDataType.CANDLE_BINANCE
        else:  # default is OHLCV hence CANDLE
            self.candle_MktDataType = MktDataType.CANDLE

        # Load initial data before starting the thread (only if candles are configured)
        if "candles" in self.data_types:
            self.load_initial_candles()

        def run():
            last_candle_fetch = time.time()
            while self.running:
                try:
                    # Normal operation continues here...
                    self.fetch_all_data()

                    current_time = time.time()
                    if current_time - last_candle_fetch >= 20:
                        self.fetch_candle_data()
                        last_candle_fetch = current_time
                    if self.connector == "oanda-rest":
                        time.sleep(5)
                    time.sleep(5)

                except Exception as e:
                    print(f"Error in scheduler thread: {e}")

        self.thread = threading.Thread(target=run)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """Stop the scheduler thread"""
        self.running = False
        if self.thread:
            self.thread.join()

        # Stop WebSocket task if running
        if self.connector == ConnectorTYPE.CCXT_WS.value:
            from .exchange_ccxt_ws import stop_streaming

            if self.ws_task and not self.ws_task.done():
                stop_streaming()
                self.ws_task.cancel()

    async def run_task(self, exchange_name: str, symbols: list):
        """
        Run WebSocket streaming task with restart on error.

        Args:
            exchange_name: Name of the exchange
            symbols: List of (exchange, symbol, connector, exsymbol) tuples
        """
        from .exchange_ccxt_ws import stream_all, stop_event

        max_retries = 5
        retry_count = 0

        while retry_count < max_retries and not stop_event.is_set():
            try:
                await stream_all(
                    exchange_name,
                    symbols,
                    self.redis,
                    self.listener_id,
                    self.connector,
                    self.data_types,
                )
                # If stream_all returns normally, reset retry count
                retry_count = 0
            except Exception as e:
                retry_count += 1
                print(
                    f"WebSocket stream error (attempt {retry_count}/{max_retries}): {e}"
                )
                if retry_count < max_retries and not stop_event.is_set():
                    await asyncio.sleep(min(2**retry_count, 30))  # Exponential backoff
                else:
                    print("Max retries reached or stop event set")
                    break

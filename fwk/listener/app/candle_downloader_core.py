import os
import threading
import queue
import time
import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from core.storage import (
    gen_filename_download,
    get_create_folder_location,
    save_candles_to_parquet,
    is_file_or_placeholder,
    get_all_symbols_metadata,
    calculate_missing_days,
    get_symbol_file_metadata,
    create_placeholder_file,
)
from core.scanner import (
    get_data_inventory,
    aggregate_inventory_by_symbol,
    parse_filename_parts,
)
from .candle_config import DownloaderConfig
from .exchange_ccxt_rest import CCXT_ExchangeConnector_REST
from .exchange_oanda_rest import Oanda_ExchangeConnector_REST
from .exchange_dukascopy_rest import Dukascopy_ExchangeConnector_REST
from .exchange_yahoo_rest import YahooFinance_ExchangeConnector_REST


logger = logging.getLogger(__name__)


class DownloadStatus(Enum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    COMPLETED = "completed"
    ERROR = "error"
    PAUSED = "paused"
    SLEEPING = "sleeping"
    WAKING = "waking"


@dataclass
class DownloadTask:
    priority: int  # 0=highest (new symbols), 1=normal (gap filling)
    exchange: str
    symbol: str
    connector: str
    market_type: str
    target_date: datetime
    total_days: int = 0
    completed_days: int = 0
    retry_count: int = 0
    max_retries: int = 3

    def __lt__(self, other):
        return self.priority < other.priority


@dataclass
class SymbolStatus:
    exchange: str
    symbol: str
    connector: str
    market_type: str
    status: DownloadStatus
    start_date: Optional[datetime] = None
    current_date: Optional[datetime] = None
    total_days: int = 0
    completed_days: int = 0
    error_message: Optional[str] = None
    last_update: datetime = field(default_factory=datetime.now)


class DownloadProgress:
    """Thread-safe progress tracking"""

    def __init__(self):
        self.current_symbol = None
        self.current_date = None
        self.total_days = 0
        self.completed_days = 0
        self.errors = []
        self.lock = threading.Lock()

    def update_progress(
        self, symbol: str, current_date: datetime, total: int, completed: int
    ):
        with self.lock:
            self.current_symbol = symbol
            self.current_date = current_date
            self.total_days = total
            self.completed_days = completed

    def add_error(self, error: str):
        with self.lock:
            self.errors.append(f"{datetime.now().isoformat()}: {error}")

    def get_progress(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "current_symbol": self.current_symbol,
                "current_date": self.current_date.isoformat()
                if self.current_date
                else None,
                "total_days": self.total_days,
                "completed_days": self.completed_days,
                "errors": self.errors[-10:] if self.errors else [],  # Last 10 errors
            }


class DownloadThread(threading.Thread):
    """Download thread for a specific exchange"""

    def __init__(
        self,
        exchange: str,
        connector: str,
        download_queue: queue.PriorityQueue,
        progress: DownloadProgress,
        config: DownloaderConfig,
        downloader: "CandleDownloaderCore",  # Reference to parent downloader
    ):
        super().__init__(daemon=True)
        self.exchange = exchange
        self.connector = connector
        self.download_queue = download_queue
        self.progress = progress
        self.config = config
        self.downloader = downloader  # Store reference to downloader
        self.running = True
        self.rate_limit_delay = 1.0  # Base delay in seconds

        # Initialize connector
        if connector == "ccxt-rest":
            self.exchange_connector = CCXT_ExchangeConnector_REST()
        elif connector == "oanda-rest":
            # Load Oanda config if needed
            oanda_config = self._load_oanda_config()
            self.exchange_connector = Oanda_ExchangeConnector_REST(
                api_token=oanda_config.get("api_token", ""),
                account_id=oanda_config.get("account_id", ""),
                demo=oanda_config.get("demo", True),
            )
        elif connector == "dukascopy-rest":
            # Initialize Dukascopy connector with base output directory
            base_output_dir = self.config.get_base_folder()
            self.exchange_connector = Dukascopy_ExchangeConnector_REST(
                base_output_dir=base_output_dir
            )
        elif connector == "yahoo-rest":
            self.exchange_connector = YahooFinance_ExchangeConnector_REST()
        else:
            raise ValueError(f"Unsupported connector: {connector}")

    def _load_oanda_config(self) -> Dict[str, Any]:
        """Load Oanda configuration from oanda.conf"""
        try:
            import yaml
            from pathlib import Path

            config_dir = os.getenv("CONFIG_PATH", "/app/config")
            path = Path(config_dir) / "oanda.conf"
            if not os.path.isfile(path):
                logger.warning(f"Oanda config not found: {path}")
                return {}

            with open(path, "r") as f:
                config = yaml.safe_load(f)
                return config or {}
        except Exception as e:
            logger.error(f"Failed to load Oanda config: {e}")
            return {}

    def run(self):
        """Main download loop"""
        logger.info(f"Download thread started for {self.exchange} ({self.connector})")

        # Initialize wake event for this exchange
        if self.exchange not in self.downloader.wake_events:
            self.downloader.wake_events[self.exchange] = threading.Event()

        while self.running:
            try:
                # Check if this exchange should sleep
                if self.downloader._should_sleep(self.exchange):
                    self._enter_sleep_mode()
                    continue

                # Try to get task with shorter timeout when active
                try:
                    task = self.download_queue.get(timeout=30.0)
                except queue.Empty:
                    # Check if we should enter sleep mode after timeout
                    if self.downloader._should_sleep(self.exchange):
                        self._enter_sleep_mode()
                    continue

                # Check if task is for this exchange
                if task.exchange != self.exchange:
                    self.download_queue.put(task)  # Put back for other thread
                    time.sleep(0.1)
                    continue

                # Process task
                self._process_task(task)

                # Mark task as done
                self.download_queue.task_done()

                # Rate limiting
                time.sleep(self.rate_limit_delay)

            except Exception as e:
                logger.error(f"Error in download thread for {self.exchange}: {e}")
                self.progress.add_error(f"Thread error: {e}")
                time.sleep(5.0)  # Wait before retrying

        logger.info(f"Download thread stopped for {self.exchange} ({self.connector})")

    def _enter_sleep_mode(self):
        """Enter sleep mode for this exchange"""
        with self.downloader.lock:
            self.downloader.per_exchange_sleep[self.exchange] = True
            self.downloader.last_completion_time[self.exchange] = datetime.now()
            self.downloader.next_wake_time[self.exchange] = (
                self.downloader._calculate_next_wake_time(datetime.now())
            )

        logger.info(
            f"Exchange {self.exchange} entering sleep mode until {self.downloader.next_wake_time[self.exchange]}"
        )

        # Sleep until wake event or scheduled time
        wake_event = self.downloader.wake_events[self.exchange]
        sleep_duration = (
            self.downloader.next_wake_time[self.exchange] - datetime.now()
        ).total_seconds()

        # Clear any previous wake signal
        wake_event.clear()

        if wake_event.wait(timeout=sleep_duration):
            logger.info(f"Exchange {self.exchange} woken up by new symbol/event")
        else:
            logger.info(f"Exchange {self.exchange} waking up for scheduled check")
            self.downloader._check_for_new_day(self.exchange)

        # Exit sleep mode
        with self.downloader.lock:
            self.downloader.per_exchange_sleep[self.exchange] = False

    def _process_task(self, task: DownloadTask):
        """Process a single download task"""
        try:
            timeframe = "1min"
            if self.connector == "yahoo-rest":
                timeframe = "1d"

            # Update progress
            self.progress.update_progress(
                f"{task.exchange}:{task.symbol}",
                task.target_date,
                task.total_days,
                task.completed_days,
            )

            # Generate file path
            end_time = task.target_date + timedelta(days=1)
            filename = gen_filename_download(
                task.exchange, task.symbol, task.target_date, end_time, timeframe
            )
            folder_path = get_create_folder_location(
                self.config.get_base_folder(), task.exchange, task.market_type
            )
            filepath = folder_path + filename

            # Check if file already exists
            if is_file_or_placeholder(filepath):
                logger.debug(f"File already exists: {filepath}")
                # DON'T update progress for already existing files
                return

            # Download candles for day
            logger.info(
                f"Downloading {task.exchange}:{task.symbol} for {task.target_date.strftime('%Y-%m-%d')} ({timeframe})"
            )

            # Pass timeframe to connector if supported
            kwargs = {}
            if self.connector == "yahoo-rest":
                kwargs["timeframe"] = timeframe

            candles = self.exchange_connector.get_all_1min_candles_for_day(
                task.exchange, task.symbol, task.target_date, intraday=False, **kwargs
            )

            if not candles:
                logger.warning(
                    f"No data for {task.exchange}:{task.symbol} on {task.target_date.strftime('%Y-%m-%d')}"
                )
                create_placeholder_file(filepath, "no_data")
                # Update symbol status even for no data days
                self.downloader._update_symbol_progress(task)
                return

            # Save to parquet
            success = save_candles_to_parquet(candles, filepath)
            if success:
                logger.info(f"Saved {len(candles)} candles to {filepath}")
                # Update symbol status on successful download
                self.downloader._update_symbol_progress(task)
            else:
                logger.error(f"Failed to save candles to {filepath}")

        except Exception as e:
            logger.error(
                f"Error processing task {task.exchange}:{task.symbol} for {task.target_date}: {e}"
            )
            self.progress.add_error(f"Task error: {e}")

            # Retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                self.download_queue.put(task)
                logger.info(
                    f"Retrying task (attempt {task.retry_count}/{task.max_retries})"
                )
            else:
                logger.error(
                    f"Max retries exceeded for {task.exchange}:{task.symbol} on {task.target_date}"
                )

                folder_path = get_create_folder_location(
                    self.config.get_base_folder(), task.exchange, task.market_type
                )
                timeframe = "1min" if self.connector != "yahoo-rest" else "1h"
                filename = gen_filename_download(
                    task.exchange,
                    task.symbol,
                    task.target_date,
                    task.target_date + timedelta(days=1),
                    timeframe,
                )
                filepath = folder_path + filename
                create_placeholder_file(filepath, "max_retries_exceeded")

    def stop(self):
        """Stop the download thread"""
        self.running = False


class CandleDownloaderCore:
    """Main candle downloader engine"""

    def __init__(self, config_path: str = "/app/config/downloader.yaml"):
        self.config = DownloaderConfig(config_path)
        self.download_queue = queue.PriorityQueue()
        self.progress = DownloadProgress()
        self.download_threads = {}
        self.state = {}  # {(exchange, symbol, connector, market_type): SymbolStatus}
        self.lock = threading.Lock()
        self.running = False
        self.scanning = False

        # Sleep/wake state management
        self.per_exchange_sleep = {}  # per-exchange sleep state
        self.last_completion_time = {}  # per-exchange completion time
        self.next_wake_time = {}  # per-exchange next wake time
        self.wake_events = {}  # per-exchange wake events

        # Load initial state from filesystem in background thread
        scan_thread = threading.Thread(target=self._scan_filesystem, daemon=True)
        scan_thread.start()

    def _scan_filesystem(self):
        """Scan filesystem to build initial state"""
        self.scanning = True
        try:
            all_metadata = get_all_symbols_metadata(self.config.get_base_folder())

            for metadata in all_metadata:
                key = (
                    metadata["exchange"],
                    metadata["symbol"],
                    metadata["connector"],
                    metadata["market_type"],
                )

                with self.lock:
                    self.state[key] = SymbolStatus(
                        exchange=metadata["exchange"],
                        symbol=metadata["symbol"],
                        connector=metadata["connector"],
                        market_type=metadata["market_type"],
                        status=DownloadStatus.COMPLETED,
                        start_date=metadata["first_date"],
                        total_days=metadata["nb_days_on_drive"],
                        completed_days=metadata["nb_days_on_drive"],
                    )

            logger.info(f"Scanned filesystem: {len(all_metadata)} symbols found")

        except Exception as e:
            logger.error(f"Error scanning filesystem: {e}")
        finally:
            self.scanning = False

    def start(self):
        """Start the downloader"""
        if self.running:
            logger.warning("Downloader already running")
            return

        self.running = True

        # Start download threads (one per exchange)
        exchanges_config = self.config.get_exchanges_config()
        for exchange_config in exchanges_config:
            exchange_name = exchange_config["name"]
            connector = exchange_config["connector"]

            if exchange_name not in self.download_threads:
                thread = DownloadThread(
                    exchange_name,
                    connector,
                    self.download_queue,
                    self.progress,
                    self.config,
                    self,  # Pass self as downloader reference
                )
                thread.start()
                self.download_threads[exchange_name] = thread
                logger.info(f"Started download thread for {exchange_name}")

        logger.info("Candle downloader started")

        queue_thread = threading.Thread(
            target=self._queue_initial_downloads, daemon=True
        )
        queue_thread.start()

    def stop(self):
        """Stop the downloader"""
        if not self.running:
            return

        self.running = False

        # Stop all threads
        for thread in self.download_threads.values():
            thread.stop()

        # Wait for threads to finish
        for thread in self.download_threads.values():
            thread.join(timeout=10.0)

        self.download_threads.clear()
        logger.info("Candle downloader stopped")

    def _queue_initial_downloads(self):
        """Queue initial download tasks based on configuration"""
        exchanges_config = self.config.get_exchanges_config()

        for exchange_config in exchanges_config:
            exchange_name = exchange_config["name"]
            connector = exchange_config["connector"]
            market_type = exchange_config["market_type"]
            symbols = exchange_config["symbols"]
            start_date = datetime.strptime(exchange_config["start_date"], "%Y-%m-%d")

            for symbol in symbols:
                self._queue_symbol_downloads(
                    exchange_name,
                    symbol,
                    connector,
                    market_type,
                    start_date,
                    priority=1,
                )

    def _update_symbol_progress(self, task: DownloadTask):
        """Update symbol progress in state after completing a task"""
        key = (task.exchange, task.symbol, task.connector, task.market_type)

        with self.lock:
            if key in self.state:
                symbol_status = self.state[key]
                symbol_status.completed_days += 1
                symbol_status.last_update = datetime.now()

                # Update current date for progress tracking
                symbol_status.current_date = task.target_date

                # Check if download is complete
                if symbol_status.completed_days >= symbol_status.total_days:
                    symbol_status.status = DownloadStatus.COMPLETED
                    logger.info(f"Download completed for {task.exchange}:{task.symbol}")

                logger.info(
                    f"Updated progress for {task.exchange}:{task.symbol}: {symbol_status.completed_days}/{symbol_status.total_days}"
                )
            else:
                # Create new status entry if not exists
                symbol_status = SymbolStatus(
                    exchange=task.exchange,
                    symbol=task.symbol,
                    connector=task.connector,
                    market_type=task.market_type,
                    status=DownloadStatus.DOWNLOADING,
                    start_date=task.target_date,
                    current_date=task.target_date,
                    total_days=task.total_days,
                    completed_days=1,
                    last_update=datetime.now(),
                )
                self.state[key] = symbol_status
                logger.info(
                    f"Created new progress entry for {task.exchange}:{task.symbol}: {symbol_status.completed_days}/{symbol_status.total_days}"
                )

    def _queue_symbol_downloads(
        self,
        exchange: str,
        symbol: str,
        connector: str,
        market_type: str,
        start_date: datetime,
        priority: int = 1,
    ):
        """Queue download tasks for a symbol from start_date to yesterday"""
        yesterday = (datetime.now() - timedelta(days=1)).date()
        current_date = yesterday
        start_date_only = start_date.date()

        # Calculate total days for progress tracking
        total_days = (yesterday - start_date_only).days + 1

        # Download from yesterday backwards to start_date
        day_count = 0
        while current_date >= start_date_only:
            task = DownloadTask(
                priority=priority,
                exchange=exchange,
                symbol=symbol,
                connector=connector,
                market_type=market_type,
                target_date=datetime.combine(current_date, datetime.min.time()),
                total_days=total_days,
                completed_days=day_count,
            )
            self.download_queue.put(task)
            current_date -= timedelta(days=1)
            day_count += 1

        # Update state
        key = (exchange, symbol, connector, market_type)
        with self.lock:
            # Get existing file metadata filtered by expected date range
            metadata = get_symbol_file_metadata(
                self.config.get_base_folder(),
                exchange,
                market_type,
                symbol,
                start_date=start_date_only,
                end_date=yesterday,
            )

            if key not in self.state:
                self.state[key] = SymbolStatus(
                    exchange=exchange,
                    symbol=symbol,
                    connector=connector,
                    market_type=market_type,
                    status=DownloadStatus.DOWNLOADING,
                    start_date=datetime.combine(start_date, datetime.min.time()),
                    total_days=(yesterday - start_date_only).days + 1,
                    completed_days=metadata["nb_days_on_drive"],
                )
            else:
                self.state[key].status = DownloadStatus.DOWNLOADING
                # Only update if file count is higher and within range
                if self.state[key].completed_days < metadata["nb_days_on_drive"]:
                    self.state[key].completed_days = metadata["nb_days_on_drive"]

        logger.info(
            f"Queued downloads for {exchange}:{symbol} from {start_date} to {yesterday}"
        )

    def add_symbol(
        self, connector: str, exchange: str, market_type: str, symbol: str
    ) -> bool:
        """Add new symbol for downloading"""
        try:
            # Calculate start date for new symbol
            start_date_str = self.config.calculate_start_date_for_new_symbol()
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")

            # Add to configuration
            if not self.config.add_symbol_to_config(
                connector, exchange, market_type, symbol
            ):
                return False

            # Start download thread if needed
            if exchange not in self.download_threads and self.running:
                thread = DownloadThread(
                    exchange,
                    connector,
                    self.download_queue,
                    self.progress,
                    self.config,
                    self,
                )
                thread.start()
                self.download_threads[exchange] = thread
                logger.info(f"Started download thread for new exchange {exchange}")

            # Wake up the exchange if it's sleeping
            self.wake_exchange(exchange)

            # Queue downloads with configurable priority
            priority_config = self.config.get_priority_config()
            priority = priority_config.get("new_symbol_priority", 0)
            self._queue_symbol_downloads(
                exchange, symbol, connector, market_type, start_date, priority
            )

            logger.info(
                f"Added new symbol {exchange}:{symbol} with start date {start_date} and woke up exchange thread"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to add symbol {exchange}:{symbol}: {e}")
            return False

    def remove_symbol(self, exchange: str, symbol: str) -> bool:
        """Remove symbol from active downloading (keep files)"""
        try:
            # Remove from configuration
            if not self.config.remove_symbol_from_config(exchange, symbol):
                return False

            # Remove from state (find matching entries)
            keys_to_remove = []
            with self.lock:
                for key in self.state.keys():
                    if key[0] == exchange and key[1] == symbol:
                        keys_to_remove.append(key)

                for key in keys_to_remove:
                    del self.state[key]

            logger.info(f"Removed symbol {exchange}:{symbol} from active downloading")
            return True

        except Exception as e:
            logger.error(f"Failed to remove symbol {exchange}:{symbol}: {e}")
            return False

    def get_download_status(self) -> List[Dict[str, Any]]:
        """Get current download status for all symbols"""
        status_list = []

        # Get progress info
        progress_info = self.progress.get_progress()

        with self.lock:
            for key, symbol_status in self.state.items():
                exchange, symbol, connector, market_type = key

                # Get file metadata
                metadata = get_symbol_file_metadata(
                    self.config.get_base_folder(), exchange, market_type, symbol
                )

                # Calculate missing days
                missing_days = calculate_missing_days(
                    self.config.get_base_folder(), exchange, market_type, symbol
                )

                # Determine current status
                status = symbol_status.status.value
                if (
                    progress_info["current_symbol"] == f"{exchange}:{symbol}"
                    and status == DownloadStatus.DOWNLOADING.value
                ):
                    status = f"downloading ({progress_info['current_date']})"

                status_dict = {
                    "exchange": exchange,
                    "symbol": symbol,
                    "connector": connector,
                    "market_type": market_type,
                    "status": status,
                    "nbdays_on_drive": metadata["nb_days_on_drive"],
                    "first_date": metadata["first_date"].strftime("%Y-%m-%d")
                    if metadata["first_date"]
                    else None,
                    "last_date": metadata["last_date"].strftime("%Y-%m-%d")
                    if metadata["last_date"]
                    else None,
                    "nb_missings": missing_days,
                    "start_date": symbol_status.start_date.strftime("%Y-%m-%d")
                    if symbol_status.start_date
                    else None,
                    "total_days": symbol_status.total_days,
                    "completed_days": symbol_status.completed_days,
                    "last_update": symbol_status.last_update.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                }

                status_list.append(status_dict)

        return status_list

    def get_queue_size(self) -> int:
        """Get current download queue size"""
        return self.download_queue.qsize()

    def is_running(self) -> bool:
        """Check if downloader is running"""
        return self.running

    def _is_weekend(self) -> bool:
        """Check if current time is during weekend (Saturday or Sunday)"""
        return datetime.now().weekday() >= 5  # 5=Saturday, 6=Sunday

    def _calculate_next_wake_time(self, current_time: datetime) -> datetime:
        """Calculate next wake time at configured minutes past the hour"""
        sleep_settings = self.config.get_sleep_settings()
        wake_minute = sleep_settings.get("wake_minute_past_hour", 15)

        next_wake = current_time.replace(minute=wake_minute, second=0, microsecond=0)
        if current_time >= next_wake:
            next_wake += timedelta(hours=1)
        return next_wake

    def _is_queue_empty_for_exchange(self, exchange: str) -> bool:
        """Check if queue has any tasks for this exchange"""
        if self.download_queue.empty():
            return True

        # Check if any tasks in queue are for this exchange
        temp_queue = queue.PriorityQueue()
        has_exchange_tasks = False

        try:
            while not self.download_queue.empty():
                task = self.download_queue.get_nowait()
                if task.exchange == exchange:
                    has_exchange_tasks = True
                temp_queue.put(task)
        except queue.Empty:
            pass

        # Restore queue
        while not temp_queue.empty():
            self.download_queue.put(temp_queue.get())

        return not has_exchange_tasks

    def _are_all_symbols_current(self, exchange: str) -> bool:
        """Check if all symbols for this exchange are current (have yesterday's data)"""
        yesterday = (datetime.now() - timedelta(days=1)).date()

        for key, symbol_status in self.state.items():
            if key[0] == exchange:  # key[0] is exchange
                # Get last download date for this symbol
                metadata = get_symbol_file_metadata(
                    self.config.get_base_folder(),
                    key[0],  # exchange
                    key[3],  # market_type
                    key[1],  # symbol
                )

                if (
                    metadata["last_date"] is None
                    or metadata["last_date"].date() < yesterday
                ):
                    return False

        return True

    def _should_sleep(self, exchange: str) -> bool:
        """Check if this exchange should enter sleep mode"""
        sleep_settings = self.config.get_sleep_settings()

        # Check if sleep mode is enabled
        if not sleep_settings.get("enable_sleep_mode", False):
            return False

        # Don't sleep on weekends
        if self._is_weekend():
            return False

        # Check if queue is empty for this exchange
        if not self._is_queue_empty_for_exchange(exchange):
            return False

        # Check if all symbols for this exchange are current
        return self._are_all_symbols_current(exchange)

    def _get_symbols_for_exchange(self, exchange: str) -> List[Dict[str, Any]]:
        """Get all symbols configured for this exchange"""
        exchanges_config = self.config.get_exchanges_config()
        symbols = []

        for exchange_config in exchanges_config:
            if exchange_config["name"] == exchange:
                for symbol in exchange_config["symbols"]:
                    symbols.append(
                        {
                            "symbol": symbol,
                            "connector": exchange_config["connector"],
                            "market_type": exchange_config["market_type"],
                        }
                    )
                break

        return symbols

    def _get_last_download_date(self, exchange: str, symbol: str) -> Optional[datetime]:
        """Get last download date for a symbol"""
        for key, symbol_status in self.state.items():
            if key[0] == exchange and key[1] == symbol:
                metadata = get_symbol_file_metadata(
                    self.config.get_base_folder(),
                    key[0],  # exchange
                    key[3],  # market_type
                    key[1],  # symbol
                )
                return metadata["last_date"]
        return None

    def _queue_single_day_download(
        self,
        exchange: str,
        symbol_info: Dict[str, Any],
        target_date: date,
        priority: int,
    ):
        """Queue a single day download for a symbol"""
        task = DownloadTask(
            priority=priority,
            exchange=exchange,
            symbol=symbol_info["symbol"],
            connector=symbol_info["connector"],
            market_type=symbol_info["market_type"],
            target_date=datetime.combine(target_date, datetime.min.time()),
            total_days=1,
            completed_days=0,
        )
        self.download_queue.put(task)

    def _check_for_new_day(self, exchange: str):
        """Check if new day has passed and queue downloads for this exchange"""
        today = datetime.now().date()
        yesterday = today - timedelta(days=1)

        # Get symbols for this exchange
        exchange_symbols = self._get_symbols_for_exchange(exchange)

        for symbol_info in exchange_symbols:
            last_date = self._get_last_download_date(exchange, symbol_info["symbol"])

            # If we don't have yesterday's data, queue it
            if last_date is None or last_date.date() < yesterday:
                priority_config = self.config.get_priority_config()
                priority = priority_config.get("daily_update_priority", 1)
                self._queue_single_day_download(
                    exchange, symbol_info, yesterday, priority
                )
                logger.info(
                    f"Queued daily update for {exchange}:{symbol_info['symbol']} on {yesterday}"
                )

    def wake_exchange(self, exchange: str):
        """Wake up a specific exchange thread"""
        if exchange in self.wake_events:
            self.wake_events[exchange].set()
            logger.info(f"Wake signal sent to exchange {exchange}")

    def get_sleep_status(self) -> Dict[str, Any]:
        """Get current sleep status for all exchanges"""
        status = {}

        for exchange_name in self.download_threads.keys():
            status[exchange_name] = {
                "sleeping": self.per_exchange_sleep.get(exchange_name, False),
                "last_completion": self.last_completion_time.get(exchange_name),
                "next_wake": self.next_wake_time.get(exchange_name),
                "is_weekend": self._is_weekend(),
                "queue_size": self._get_queue_size_for_exchange(exchange_name),
                "symbols_current": self._are_all_symbols_current(exchange_name),
            }

        return status

    def _get_queue_size_for_exchange(self, exchange: str) -> int:
        """Get queue size for specific exchange"""
        if self.download_queue.empty():
            return 0

        count = 0
        temp_queue = queue.PriorityQueue()

        try:
            while not self.download_queue.empty():
                task = self.download_queue.get_nowait()
                if task.exchange == exchange:
                    count += 1
                temp_queue.put(task)
        except queue.Empty:
            pass

        # Restore queue
        while not temp_queue.empty():
            self.download_queue.put(temp_queue.get())

        return count

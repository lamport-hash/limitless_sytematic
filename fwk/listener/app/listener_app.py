from fastapi import FastAPI, HTTPException, Request, Response, Query, Depends
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from datetime import datetime, timedelta

import asyncio
import json
from typing import List, Dict, Optional
import os
import threading
import signal
import uvloop
import yaml
from pathlib import Path

from starlette.responses import StreamingResponse

from .listener_redis_client import redis_client, MktDataType
from .scheduler import DataScheduler
from .config import get_listener_config
from .exchange_ccxt_ws import stop_event, stream_all
from core.enums import ConnectorTYPE, CandleType, ConnectorCapacity
from .portal_client import (
    router,
    get_db,
    get_user_status_html,
    verify_portal_service_token,
    SERVICE_NAME,
)


app = FastAPI()

# Include router for portal
app.include_router(router, prefix="/clientportal")


import logging

logging.basicConfig(level=logging.INFO)

oanda_dict = {}


def get_data_status(
    last_timestamp: Optional[int], stale_threshold_ms: int = 300000
) -> str:
    """
    Determine data status based on last timestamp.

    Args:
        last_timestamp: Last data timestamp in milliseconds (None if no data)
        stale_threshold_ms: Time threshold in ms to consider data stale (default: 5 minutes)

    Returns:
        "receiving", "no_data_yet", or "stale"
    """
    if last_timestamp is None:
        return "no_data_yet"

    current_time = int(datetime.utcnow().timestamp() * 1000)
    age_ms = current_time - last_timestamp

    if age_ms <= stale_threshold_ms:
        return "receiving"
    else:
        return "stale"


def get_connector_capacity_from_type(connector_type: str) -> str:
    """Determine connector capacity based on connector type string"""
    capacity_map = {
        ConnectorTYPE.CCXT_WS.value: ConnectorCapacity.OB_TRADES.value,
        ConnectorTYPE.CCXT_REST.value: ConnectorCapacity.ALL.value,
        ConnectorTYPE.CCXT_REST_CANDLE_BINANCE.value: ConnectorCapacity.ALL.value,
        ConnectorTYPE.OANDA_REST.value: ConnectorCapacity.ALL.value,
        ConnectorTYPE.DUKASCOPY_REST.value: ConnectorCapacity.CANDLES.value,
    }
    return capacity_map.get(connector_type, "unknown")


def load_oanda_config() -> dict:
    config_dir = os.getenv("CONFIG_PATH", "/app/config")
    path = Path(config_dir) / "oanda.conf"
    if not os.path.isfile(path):
        logging.error(f"Config file not found: {path}")
        return {}
    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
            logging.info(f"Loaded config from: {path}")
            return config
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        return {}


# Initialize Redis client and scheduler
listener_id = os.environ.get("LISTENER_ID", "")
scheduler = DataScheduler(redis_client, load_oanda_config(), listener_id)


def handle_sigterm():
    print("Stopping...")
    stop_event.set()


asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

signal.signal(signal.SIGINT, lambda s, f: handle_sigterm())
signal.signal(signal.SIGTERM, lambda s, f: handle_sigterm())


# Load configuration on startup
@app.on_event("startup")
async def startup_event():
    try:
        listener_id = os.environ.get("LISTENER_ID")
        if not listener_id:
            raise ValueError("LISTENER_ID environment variable not set")

        # Set port mapping in Redis
        config_path = os.path.join(
            os.getenv("CONFIG_PATH", "/app/config"), "listeners.yaml"
        )
        with open(config_path, "r") as f:
            listeners_data = yaml.safe_load(f)
        port = listeners_data["listeners"][listener_id]["port"]
        data = {"port": port, "datetime": datetime.now().isoformat()}
        redis_client.redis.hset("listener_port_map", listener_id, json.dumps(data))

        connector, exchanges, data_types = get_listener_config(listener_id)

        scheduler.connector = connector
        scheduler.data_types = data_types
        scheduler.statics_origin = listeners_data["listeners"][listener_id][
            "statics_origin"
        ]
        scheduler.statics_url = listeners_data["listeners"][listener_id].get(
            "statics_url", os.getenv("STATICS_URL", "http://localhost:8093")
        )

        if connector == ConnectorTYPE.CCXT_REST.value:
            for exchange in exchanges:
                for symbol in exchange["symbols"]:
                    scheduler.add_symbol(exchange["name"], symbol)
            scheduler.start()
        elif connector == ConnectorTYPE.CCXT_REST_CANDLE_BINANCE.value:
            for exchange in exchanges:
                for symbol in exchange["symbols"]:
                    scheduler.add_symbol(exchange["name"], symbol)
            scheduler.start()
        elif connector == ConnectorTYPE.OANDA_REST.value:
            for exchange in exchanges:
                for symbol in exchange["symbols"]:
                    scheduler.add_symbol(exchange["name"], symbol)
            scheduler.start()
        elif connector == ConnectorTYPE.CCXT_WS.value:
            for exchange in exchanges:
                for symbol in exchange["symbols"]:
                    scheduler.add_symbol(exchange["name"], symbol)
            scheduler.start()
        else:
            raise RuntimeError(f"Unknown connector: {str(connector)}")

    except Exception as e:
        raise RuntimeError(f"Failed to initialize listener: {str(e)}")


@app.on_event("shutdown")
def shutdown_event():
    if scheduler.connector == ConnectorTYPE.CCXT_REST.value:
        scheduler.stop()
    if scheduler.connector == ConnectorTYPE.CCXT_REST_CANDLE_BINANCE.value:
        scheduler.stop()
    if scheduler.connector == ConnectorTYPE.OANDA_REST.value:
        scheduler.stop()
    if scheduler.connector == ConnectorTYPE.CCXT_WS.value:
        scheduler.stop()


@app.get("/list_symbols")
def list_symbols():
    """List all currently monitored symbols"""
    return {"symbols": scheduler.get_symbols()}


@app.get("/config_info")
def get_config_info():
    """Get listener configuration information"""
    try:
        # Get current listener configuration
        connector, exchanges, data_types = get_listener_config(os.getenv("LISTENER_ID"))

        # Get connector capacity from the appropriate connector instance
        if scheduler.ex_connector:
            connector_capacity = scheduler.ex_connector.get_capacity().value
        else:
            # For CCXT_WS and other connectors that don't call scheduler.start()
            connector_capacity = get_connector_capacity_from_type(connector)

        # Format exchanges and symbols for display
        exchanges_info = []
        for exchange in exchanges:
            exchanges_info.append(
                {"name": exchange["name"], "symbols": exchange["symbols"]}
            )

        return {
            "listener_id": os.getenv("LISTENER_ID"),
            "connector_type": connector,
            "connector_capacity": connector_capacity,
            "selected_data_types": list(data_types),
            "exchanges": exchanges_info,
            "activated": True,
        }
    except Exception as e:
        return {
            "error": str(e),
            "listener_id": os.getenv("LISTENER_ID"),
            "connector_type": "unknown",
            "connector_capacity": "unknown",
            "selected_data_types": [],
            "exchanges": [],
            "activated": False,
        }


@app.post("/add_symbol")
def add_symbol(
    exchange: str, symbol: str, user_data: dict = Depends(verify_portal_service_token)
):
    """Add a new symbol to monitor"""
    scheduler.add_symbol(exchange, symbol)
    return {"status": "success", "message": f"Added {exchange}:{symbol}"}


@app.post("/remove_symbol")
def remove_symbol(
    exchange: str, symbol: str, user_data: dict = Depends(verify_portal_service_token)
):
    """Remove a symbol from monitoring"""
    scheduler.remove_symbol(exchange, symbol)
    return {"status": "success", "message": f"Removed {exchange}:{symbol}"}


@app.get("/candles/count")
def get_nb_candles(
    exchange: str = Query(...),
    symbol: str = Query(...),
    date: str = Query(..., regex=r"^\d{4}/\d{2}/\d{2}$"),
) -> int:
    return redis_client.get_nb_candles(
        scheduler.listener_id, scheduler.connector, exchange, symbol, date
    )


@app.get("/candles/daily_counts")
def get_daily_candle_counts(
    exchange: str = Query(...), symbol: str = Query(...), n_days: int = Query(7, gt=0)
) -> Dict[str, int]:
    # Return as dict (FastAPI will convert OrderedDict fine)
    return redis_client.get_daily_candle_counts(
        scheduler.listener_id, scheduler.connector, exchange, symbol, n_days
    )


# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/listener_ui", response_class=HTMLResponse)
async def get_listener_ui(request: Request):
    """Render monitoring listener_ui"""
    return templates.TemplateResponse(
        "listener_ui.html",
        {"request": request, "listener_id": os.getenv("LISTENER_ID")},
    )


# Add this to your monitoring endpoint
@app.get("/candle_status")
def candle_status():
    """Get candle processing status for all symbols"""
    symbols = scheduler.get_symbols()
    status = {}

    # Only include candle status if candles are configured
    if "candles" not in scheduler.data_types:
        return status

    for exchange, symbol, connector, exsymbol in symbols:
        last_time = redis_client.get_last_candle_time(
            scheduler.listener_id, scheduler.connector, exchange, symbol
        )
        status[f"{exchange}:{symbol}"] = {
            "last_timestamp": last_time,
            "current_time": int(datetime.utcnow().timestamp() * 1000),
            "status": get_data_status(last_time),
        }
    return status


@app.get("/monitor_data")
def monitor_data():
    """Get the latest data for all monitored symbols"""
    symbols = scheduler.get_symbols()
    result = {}

    for exchange, symbol, connector, exsymbol in symbols:
        orderbook = redis_client.get_market_data(
            scheduler.listener_id,
            scheduler.connector,
            exchange,
            symbol,
            MktDataType.ORDERBOOK,
        )
        trades = redis_client.get_market_data(
            scheduler.listener_id,
            scheduler.connector,
            exchange,
            symbol,
            MktDataType.TRADE,
            5,
        )

        candle = None
        if scheduler.candle_MktDataType and "candles" in scheduler.data_types:
            candles = redis_client.get_candles(
                scheduler.listener_id,
                scheduler.connector,
                exchange,
                symbol,
                scheduler.candle_MktDataType,
                1,
            )
            candle = candles[0] if candles else None

        result[f"{exchange}:{symbol}"] = {
            "orderbook": orderbook[0].to_dict() if orderbook else None,
            "trades": [trade.to_dict() for trade in trades] if trades else None,
            "candle": candle.to_dict() if candle else None,
        }

    result["data_types"] = list(scheduler.data_types)

    return JSONResponse(content=result)


@app.get("/monitor_updates")
async def monitor_updates():
    """Server-Sent Events endpoint for real-time updates"""

    async def event_generator():
        while True:
            try:
                # Get the latest data
                symbols = scheduler.get_symbols()
                result = {}

                for exchange, symbol, connector, exsymbol in symbols:
                    orderbook = redis_client.get_market_data(
                        scheduler.listener_id,
                        scheduler.connector,
                        exchange,
                        symbol,
                        MktDataType.ORDERBOOK,
                    )
                    trades = redis_client.get_market_data(
                        scheduler.listener_id,
                        scheduler.connector,
                        exchange,
                        symbol,
                        MktDataType.TRADE,
                        5,
                    )

                    candle = None
                    if (
                        scheduler.candle_MktDataType
                        and "candles" in scheduler.data_types
                    ):
                        candles = redis_client.get_candles(
                            scheduler.listener_id,
                            scheduler.connector,
                            exchange,
                            symbol,
                            scheduler.candle_MktDataType,
                            1,
                        )
                        candle = candles[0] if candles else None

                    result[f"{exchange}:{symbol}"] = {
                        "orderbook": orderbook[0].to_dict() if orderbook else None,
                        "trades": [trade.to_dict() for trade in trades]
                        if trades
                        else None,
                        "candle": candle.to_dict() if candle else None,
                    }

                result["data_types"] = list(scheduler.data_types)

                # Format as SSE message
                yield f"data: {json.dumps(result)}\n\n"

            except Exception as e:
                print(f"Error in SSE generator: {e}")
                yield "event: error\ndata: An error occurred\n\n"

            # Wait before sending the next update
            await asyncio.sleep(5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# Update health check endpoint
@app.get("/health")
def health_check():
    """Health check endpoint"""
    try:
        # Simple Redis ping check
        redis_client.redis.ping()
        redis_status = "healthy"
    except:
        redis_status = "unhealthy"

    return {
        "status": "healthy",
        "listener_id": os.getenv("LISTENER_ID"),
        "redis_status": redis_status,
    }


@app.get("/symbol-timestamps")
def symbol_timestamps():
    """Get last timestamp for each configured data type (candle/trade/orderbook) for all symbols"""
    symbols = scheduler.get_symbols()
    result = {}

    for exchange, symbol, connector, exsymbol in symbols:
        status = {
            "exchange": exchange,
            "symbol": symbol,
            "connector": connector,
            "data_types": {},
        }

        # Get last trade timestamp if trades are configured
        if "trades" in scheduler.data_types:
            last_trade_time = redis_client.get_last_stream_timestamp(
                scheduler.listener_id,
                scheduler.connector,
                exchange,
                symbol,
                MktDataType.TRADE,
            )
            status["data_types"]["trade"] = {
                "last_timestamp": last_trade_time,
                "status": get_data_status(last_trade_time),
            }

        # Get last orderbook timestamp if orderbook is configured
        if "ob" in scheduler.data_types:
            last_ob_time = redis_client.get_last_stream_timestamp(
                scheduler.listener_id,
                scheduler.connector,
                exchange,
                symbol,
                MktDataType.ORDERBOOK,
            )
            status["data_types"]["orderbook"] = {
                "last_timestamp": last_ob_time,
                "status": get_data_status(last_ob_time),
            }

        # Get last candle timestamp if candles are configured
        if "candles" in scheduler.data_types:
            last_candle_time = redis_client.get_last_candle_time(
                scheduler.listener_id, scheduler.connector, exchange, symbol
            )
            status["data_types"]["candle"] = {
                "last_timestamp": last_candle_time,
                "status": get_data_status(last_candle_time),
            }

        result[f"{exchange}:{symbol}"] = status

    return JSONResponse(content=result)

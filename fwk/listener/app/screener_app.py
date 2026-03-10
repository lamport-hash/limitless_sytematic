from fastapi import FastAPI, HTTPException, Query, Response
import gzip
import json
import os
from datetime import datetime, timedelta
from typing import Optional, List
import logging

from core.scanner import (
    get_data_inventory as get_data_inventory_impl,
    aggregate_inventory_by_symbol,
)
from core.storage import get_candles_for_date_range
from .candle_config import DownloaderConfig

logger = logging.getLogger(__name__)

app = FastAPI(title="Candle Data Screener", version="1.0.0")

import os

config_path = os.environ.get("CONFIG_PATH", "/app/config/downloader.yaml")
if os.path.isabs(config_path):
    config_path = os.environ.get("CONFIG_PATH", "config/downloader.yaml")

config = DownloaderConfig(config_path)
base_folder = config.get_base_folder()


@app.get("/screener/data_inventory")
async def get_data_inventory(
    exchange: Optional[str] = None,
    market_type: Optional[str] = None,
    candle_type: Optional[str] = None,
    timeframe: Optional[str] = None,
):
    """
    Get inventory of all available candle data (one row per symbol).

    Query Parameters:
        - exchange: Filter by exchange name (e.g., binance, oanda, sp500, nasdaq100)
        - market_type: Filter by market type (spot, futures, forex)
        - candle_type: Filter by candle type (ohlcv, binance_kline)
        - timeframe: Filter by timeframe (1min, 1h, 4h, 1d)

    Returns:
        JSON with total count and data array
    """
    try:
        inventory = get_data_inventory_impl(base_folder)
        aggregated = aggregate_inventory_by_symbol(inventory)

        if exchange:
            aggregated = [d for d in aggregated if d["exchange"] == exchange]

        if market_type:
            aggregated = [d for d in aggregated if d["market_type"] == market_type]

        if candle_type:
            aggregated = [d for d in aggregated if d["candle_type"] == candle_type]

        if timeframe:
            aggregated = [d for d in aggregated if d["timeframe"] == timeframe]

        return {"total": len(aggregated), "data": aggregated}
    except Exception as e:
        logger.error(f"Error getting data inventory: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/screener/data")
async def get_data(
    exchange: str = Query(..., description="Exchange name"),
    market_type: str = Query(..., description="Market type (spot, futures, forex)"),
    symbol: str = Query(..., description="Trading symbol"),
    candle_type: Optional[str] = None,
    timeframe: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """
    Get candle data for specific parameters (gzip-compressed JSON).

    Query Parameters:
        - exchange: Exchange name (required)
        - market_type: Market type (required)
        - symbol: Trading symbol (required)
        - candle_type: Candle type (optional)
        - timeframe: Timeframe (optional)
        - start_date: Start date in YYYY-MM-DD format (optional)
        - end_date: End date in YYYY-MM-DD format (optional)

    Returns:
        Gzip-compressed JSON array of candles
    """
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None

        df = get_candles_for_date_range(
            base_folder, exchange, market_type, symbol, start_dt, end_dt
        )

        if df.empty:
            raise HTTPException(
                status_code=404, detail="No data found for the specified parameters"
            )

        candles = df.to_dict(orient="records")

        for candle in candles:
            for key, value in candle.items():
                if hasattr(value, "isoformat"):
                    candle[key] = value.isoformat()

        json_str = json.dumps(candles)

        compressed = gzip.compress(json_str.encode("utf-8"))

        return Response(
            content=compressed,
            media_type="application/json",
            headers={
                "Content-Encoding": "gzip",
                "Content-Type": "application/json",
                "X-Original-Size": str(len(json_str)),
                "X-Compressed-Size": str(len(compressed)),
            },
        )
    except ValueError as e:
        logger.error(f"Date parsing error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid date format: {str(e)}")
    except Exception as e:
        logger.error(f"Error getting candle data: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/screener/dashboard_screener")
async def get_dashboard():
    """
    Serve the screener dashboard frontend.
    """
    try:
        from fastapi.staticfiles import StaticFiles

        static_dir = "/app/app/static"
        if not os.path.exists(static_dir):
            static_dir = "app/static"

        with open(os.path.join(static_dir, "screener.html"), "r") as f:
            html_content = f.read()

        return Response(content=html_content, media_type="text/html")
    except Exception as e:
        logger.error(f"Error serving dashboard: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to load dashboard: {str(e)}"
        )


@app.get("/screener/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "base_folder": base_folder,
        "timestamp": datetime.utcnow().isoformat(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

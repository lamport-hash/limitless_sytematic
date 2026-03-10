import os
import signal
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .candle_downloader_core import CandleDownloaderCore, DownloadStatus
from .candle_config import (
    DownloaderConfig,
    validate_connector_exchange,
    get_valid_exchanges_for_connector,
    get_valid_connectors,
    get_valid_market_types,
)
from core.storage import get_candles_for_date_range
from .portal_client import router, verify_portal_service_token

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Candle Downloader Service", version="1.0.0")

# Include portal router
app.include_router(router, prefix="/clientportal")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global downloader instance
downloader: Optional[CandleDownloaderCore] = None


# Pydantic models for API
class AddSymbolRequest(BaseModel):
    connector: str
    exchange: str
    market_type: str
    symbol: str


class RemoveSymbolRequest(BaseModel):
    exchange: str
    symbol: str


class SaveConfigRequest(BaseModel):
    config: Dict[str, Any]


class CandleDataRequest(BaseModel):
    connector: str
    exchange: str
    market_type: str
    symbol: str
    startdate: str
    enddate: str


class CandleDataResponse(BaseModel):
    status: str
    data: List[Dict[str, Any]]
    symbol: str
    exchange: str
    start_date: str
    end_date: str
    count: int
    message: Optional[str] = None


def get_downloader() -> CandleDownloaderCore:
    """Get or create downloader instance"""
    global downloader
    if downloader is None:
        config_path = os.path.join(
            os.getenv("CONFIG_PATH", "/app/config"), "downloader.yaml"
        )
        downloader = CandleDownloaderCore(config_path)
    return downloader


def handle_sigterm():
    """Handle shutdown signals"""
    logger.info("Shutting down candle downloader...")
    if downloader:
        downloader.stop()


# Register signal handlers
signal.signal(signal.SIGINT, lambda s, f: handle_sigterm())
signal.signal(signal.SIGTERM, lambda s, f: handle_sigterm())


@app.on_event("startup")
async def startup_event():
    """Initialize downloader on startup"""
    try:
        downloader_instance = get_downloader()
        downloader_instance.start()
        logger.info("Candle downloader service started successfully")
    except Exception as e:
        logger.error(f"Failed to start candle downloader: {e}")
        raise


@app.on_event("shutdown")
def shutdown_event():
    """Cleanup on shutdown"""
    handle_sigterm()


# API Endpoints
@app.get("/ohlcv")
def get_ohlcv_data(
    connector: str,
    exchange: str,
    market_type: str,
    symbol: str,
    startdate: str,
    enddate: str,
):
    """Get OHLCV data for a specific symbol and date range"""
    try:
        # Validate connector-exchange combination
        if not validate_connector_exchange(connector, exchange):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid connector-exchange combination: {connector}-{exchange}",
            )

        # Validate market type
        if market_type not in get_valid_market_types():
            raise HTTPException(
                status_code=400, detail=f"Invalid market type: {market_type}"
            )

        # Parse dates
        try:
            start_date = datetime.strptime(startdate, "%Y-%m-%d")
            end_date = datetime.strptime(enddate, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=400, detail="Invalid date format. Use YYYY-MM-DD format"
            )

        # Validate date range
        if start_date > end_date:
            raise HTTPException(
                status_code=400, detail="Start date cannot be after end date"
            )

        # Limit date range to prevent excessive data loading
        max_days = 365
        if (end_date - start_date).days > max_days:
            raise HTTPException(
                status_code=400, detail=f"Date range cannot exceed {max_days} days"
            )

        downloader_instance = get_downloader()
        base_folder = downloader_instance.config.get_base_folder()

        # Get candle data
        df = get_candles_for_date_range(
            base_folder, exchange, market_type, symbol, start_date, end_date
        )

        if df.empty:
            return CandleDataResponse(
                status="success",
                data=[],
                symbol=symbol,
                exchange=exchange,
                start_date=startdate,
                end_date=enddate,
                count=0,
                message="No data found for the specified parameters",
            )

        # Convert DataFrame to list of dictionaries
        candle_data = []
        for _, row in df.iterrows():
            candle_data.append(
                {
                    "timestamp": int(row["timestamp"]),
                    "datetime": row["datetime"].isoformat()
                    if "datetime" in row
                    else None,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                }
            )

        return CandleDataResponse(
            status="success",
            data=candle_data,
            symbol=symbol,
            exchange=exchange,
            start_date=startdate,
            end_date=enddate,
            count=len(candle_data),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get OHLCV data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    """Health check endpoint"""
    try:
        downloader_instance = get_downloader()
        return {
            "status": "healthy",
            "service": "candle-downloader",
            "running": downloader_instance.is_running(),
            "queue_size": downloader_instance.get_queue_size(),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            },
        )


@app.get("/download_status")
def get_download_status():
    """Get current download status for all symbols"""
    try:
        downloader_instance = get_downloader()
        status_list = downloader_instance.get_download_status()

        return {
            "status": "success",
            "data": status_list,
            "queue_size": downloader_instance.get_queue_size(),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Failed to get download status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/add_symbol")
def add_symbol(
    request: AddSymbolRequest, user_data: dict = Depends(verify_portal_service_token)
):
    """Add new symbol for downloading"""
    try:
        # Validate connector-exchange combination
        if not validate_connector_exchange(request.connector, request.exchange):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid connector-exchange combination: {request.connector}-{request.exchange}",
            )

        # Validate market type
        if request.market_type not in get_valid_market_types():
            raise HTTPException(
                status_code=400, detail=f"Invalid market type: {request.market_type}"
            )

        downloader_instance = get_downloader()
        success = downloader_instance.add_symbol(
            request.connector, request.exchange, request.market_type, request.symbol
        )

        if success:
            logger.info(f"Added symbol {request.exchange}:{request.symbol}")
            return {
                "status": "success",
                "message": f"Added {request.exchange}:{request.symbol} for downloading",
            }
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to add symbol {request.exchange}:{request.symbol}",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add symbol: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/remove_symbol")
def remove_symbol(
    request: RemoveSymbolRequest, user_data: dict = Depends(verify_portal_service_token)
):
    """Remove symbol from active downloading"""
    try:
        downloader_instance = get_downloader()
        success = downloader_instance.remove_symbol(request.exchange, request.symbol)

        if success:
            logger.info(f"Removed symbol {request.exchange}:{request.symbol}")
            return {
                "status": "success",
                "message": f"Removed {request.exchange}:{request.symbol} from active downloading",
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Symbol {request.exchange}:{request.symbol} not found",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove symbol: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/save_config")
def save_config(
    request: SaveConfigRequest, user_data: dict = Depends(verify_portal_service_token)
):
    """Save new configuration with backup"""
    try:
        downloader_instance = get_downloader()
        backup_path = downloader_instance.config.save_config_with_backup(request.config)

        logger.info(f"Configuration saved, backup created: {backup_path}")
        return {
            "status": "success",
            "message": "Configuration saved successfully",
            "backup_path": backup_path,
            "note": "Service restart required for changes to take effect",
        }

    except Exception as e:
        logger.error(f"Failed to save config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config")
def get_config():
    """Get current configuration"""
    try:
        downloader_instance = get_downloader()
        config = downloader_instance.config.config

        return {
            "status": "success",
            "data": config,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to get config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/valid_options")
def get_valid_options():
    """Get valid options for form dropdowns"""
    try:
        return {
            "status": "success",
            "data": {
                "connectors": get_valid_connectors(),
                "market_types": get_valid_market_types(),
                "connector_exchanges": {
                    connector: get_valid_exchanges_for_connector(connector)
                    for connector in get_valid_connectors()
                },
            },
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to get valid options: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download_progress")
def get_download_progress():
    """Get real-time download progress"""
    try:
        downloader_instance = get_downloader()
        progress_info = downloader_instance.progress.get_progress()

        return {
            "status": "success",
            "data": progress_info,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to get download progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Dashboard UI
@app.get("/candle_downloader_ui", response_class=HTMLResponse)
async def get_candle_downloader_ui(request: Request):
    """Render candle downloader dashboard"""
    return templates.TemplateResponse(
        "candle_downloader_ui.html",
        {
            "request": request,
            "service_name": os.getenv("SERVICE_NAME", "candle-downloader"),
        },
    )


# Management endpoints
@app.post("/start_downloader")
def start_downloader(user_data: dict = Depends(verify_portal_service_token)):
    """Start the downloader"""
    try:
        downloader_instance = get_downloader()
        if downloader_instance.is_running():
            return {"status": "success", "message": "Downloader already running"}

        downloader_instance.start()
        logger.info("Downloader started via API")
        return {"status": "success", "message": "Downloader started"}

    except Exception as e:
        logger.error(f"Failed to start downloader: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stop_downloader")
def stop_downloader(user_data: dict = Depends(verify_portal_service_token)):
    """Stop the downloader"""
    try:
        downloader_instance = get_downloader()
        if not downloader_instance.is_running():
            return {"status": "success", "message": "Downloader already stopped"}

        downloader_instance.stop()
        logger.info("Downloader stopped via API")
        return {"status": "success", "message": "Downloader stopped"}

    except Exception as e:
        logger.error(f"Failed to stop downloader: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/system_info")
def get_system_info():
    """Get system information"""
    try:
        downloader_instance = get_downloader()

        return {
            "status": "success",
            "data": {
                "service_name": os.getenv("SERVICE_NAME", "candle-downloader"),
                "config_path": downloader_instance.config.config_path,
                "base_folder": downloader_instance.config.get_base_folder(),
                "nb_days_to_download": downloader_instance.config.get_nb_days_to_download(),
                "running": downloader_instance.is_running(),
                "queue_size": downloader_instance.get_queue_size(),
                "active_threads": len(downloader_instance.download_threads),
                "tracked_symbols": len(downloader_instance.state),
            },
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to get system info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sleep_status")
def get_sleep_status():
    """Get current sleep status for all exchanges"""
    try:
        downloader_instance = get_downloader()
        sleep_status = downloader_instance.get_sleep_status()

        return {
            "status": "success",
            "data": sleep_status,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to get sleep status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/force_wake_exchange")
def force_wake_exchange(
    exchange: str, user_data: dict = Depends(verify_portal_service_token)
):
    """Force wake specific exchange"""
    try:
        downloader_instance = get_downloader()

        if exchange not in downloader_instance.download_threads:
            raise HTTPException(
                status_code=404, detail=f"Exchange {exchange} not found or not running"
            )

        downloader_instance.wake_exchange(exchange)

        logger.info(f"Force woke exchange {exchange} via API")
        return {
            "status": "success",
            "message": f"Exchange {exchange} woken up",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to wake exchange {exchange}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sleep_config")
def get_sleep_config():
    """Get current sleep configuration"""
    try:
        downloader_instance = get_downloader()
        sleep_settings = downloader_instance.config.get_sleep_settings()
        priority_config = downloader_instance.config.get_priority_config()

        return {
            "status": "success",
            "data": {
                "sleep_settings": sleep_settings,
                "priority_config": priority_config,
            },
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to get sleep config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

"""
Dual Momentum Strategy Backtest UI - FastAPI Application.
"""

import os
import sys
import uuid
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ui_strat.backtest_runner import (
    load_parquet_file,
    run_backtest,
    compute_trades_from_orders,
)

warnings.filterwarnings('ignore')

BACKTEST_CACHE: Dict = {}

app = FastAPI(title="Dual Momentum Strategy Backtest")

BASE_DIR = Path(__file__).resolve().parent
BUNDLE_DIR = Path(os.getenv("BUNDLE_DIR", str(BASE_DIR.parent / "data" / "bundle")))
CHARTS_DIR = BASE_DIR / "static" / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static/charts", StaticFiles(directory=str(CHARTS_DIR)), name="charts")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


class BacktestRequest(BaseModel):
    filename: str
    selected_assets: List[str]
    lookback: int = 3500
    default_asset: str
    top_n: int
    abs_momentum_threshold: float
    transaction_cost_pct: float = 0.01
    min_holding_periods: int = 240
    switch_threshold_pct: float = 0.0
    rsi_period: int = 14
    use_rsi_entry_filter: bool = False
    rsi_entry_max: float = 30.0
    use_rsi_entry_queue: bool = False
    use_rsi_diff_filter: bool = False
    rsi_diff_threshold: float = 10.0


class BacktestResponse(BaseModel):
    run_id: str
    metrics: Dict
    asset_metrics: Dict
    charts: Dict[str, str]
    orders_count: int


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/files")
async def list_parquet_files():
    try:
        if not BUNDLE_DIR.exists():
            return {"files": [], "error": f"Bundle directory not found: {BUNDLE_DIR}"}
        
        files = []
        for f in BUNDLE_DIR.iterdir():
            if f.is_file() and f.suffix == ".parquet":
                files.append({
                    "name": f.name,
                    "path": str(f),
                    "size": f.stat().st_size
                })
        
        files.sort(key=lambda x: x["name"])
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/files/{filename}/assets")
async def get_file_assets(filename: str):
    try:
        filepath = BUNDLE_DIR / filename
        if not filepath.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        _, assets = load_parquet_file(str(filepath))
        return {"assets": assets, "filename": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/run-backtest", response_model=BacktestResponse)
async def api_run_backtest(request: BacktestRequest):
    try:
        filepath = BUNDLE_DIR / request.filename
        if not filepath.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        run_id = str(uuid.uuid4())[:8]
        
        result = run_backtest(
            filepath=str(filepath),
            selected_assets=request.selected_assets,
            lookback=request.lookback,
            default_asset=request.default_asset,
            top_n=request.top_n,
            abs_momentum_threshold=request.abs_momentum_threshold,
            transaction_cost_pct=request.transaction_cost_pct,
            min_holding_periods=request.min_holding_periods,
            switch_threshold_pct=request.switch_threshold_pct,
            rsi_period=request.rsi_period,
            use_rsi_entry_filter=request.use_rsi_entry_filter,
            rsi_entry_max=request.rsi_entry_max,
            use_rsi_entry_queue=request.use_rsi_entry_queue,
            use_rsi_diff_filter=request.use_rsi_diff_filter,
            rsi_diff_threshold=request.rsi_diff_threshold,
            run_id=run_id
        )
        
        BACKTEST_CACHE[run_id] = {
            'orders_df': result.get('orders_df'),
            'p_df': result.get('p_df'),
            'assets': request.selected_assets
        }
        
        return BacktestResponse(
            run_id=run_id,
            metrics=result["metrics"],
            asset_metrics=result["asset_metrics"],
            charts=result["charts"],
            orders_count=result["orders_count"]
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/run-backtest/{run_id}/trades")
async def get_backtest_trades(run_id: str):
    try:
        if run_id not in BACKTEST_CACHE:
            raise HTTPException(status_code=404, detail="Run ID not found or expired")
        
        cache_entry = BACKTEST_CACHE[run_id]
        orders_df = cache_entry.get('orders_df')
        p_df = cache_entry.get('p_df')
        assets = cache_entry.get('assets', [])
        
        trades = compute_trades_from_orders(orders_df, p_df, assets)
        
        return {"trades": trades, "count": len(trades)}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def startup():
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8500)

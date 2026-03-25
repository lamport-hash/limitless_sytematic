"""
Seasonality analysis routes - API endpoints for financial seasonality analysis.
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel

from ui_data.services.seasonality_service import (
    search_assets_for_seasonality,
    get_full_analysis,
    load_asset_data,
    analyze_intraday,
    analyze_weekday,
    analyze_month,
    calculate_summary_stats,
    generate_calendar_heatmap,
    to_python_type,
)

router = APIRouter(prefix="/api/seasonality", tags=["seasonality"])


class AnalysisRequest(BaseModel):
    symbol: str
    freq: str = "candle_1hour"
    metric: str = "returns"
    interval_minutes: int = 60
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    filter_weekday: Optional[int] = None
    filter_month: Optional[int] = None
    custom_window: Optional[Dict[str, str]] = None


class DrilldownRequest(BaseModel):
    symbol: str
    freq: str = "candle_1hour"
    metric: str = "returns"
    interval_minutes: int = 60
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    filter_type: str
    filter_value: int


@router.get("/assets/search")
def search_assets(
    q: str = Query("", description="Search query for asset symbol"),
    limit: int = Query(20, description="Maximum number of results"),
) -> List[Dict[str, Any]]:
    """Search for assets available for seasonality analysis."""
    return search_assets_for_seasonality(q, limit)


@router.post("/analyze")
def analyze_asset(request: AnalysisRequest) -> Dict[str, Any]:
    """Run full seasonality analysis for an asset."""
    valid_metrics = ["returns", "volatility", "rsi"]
    if request.metric not in valid_metrics:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid metric. Must be one of: {valid_metrics}"
        )
    
    valid_intervals = [5, 15, 30, 60, 240]
    if request.interval_minutes not in valid_intervals:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid interval. Must be one of: {valid_intervals}"
        )
    
    result = get_full_analysis(
        symbol=request.symbol,
        freq=request.freq,
        metric=request.metric,
        interval_minutes=request.interval_minutes,
        start_date=request.start_date,
        end_date=request.end_date,
        filter_weekday=request.filter_weekday,
        filter_month=request.filter_month,
        custom_window=request.custom_window,
    )
    
    if 'error' in result:
        raise HTTPException(status_code=404, detail=result['error'])
    
    return result


@router.post("/drilldown")
def drilldown_analysis(request: DrilldownRequest) -> Dict[str, Any]:
    """Perform drill-down analysis based on selected period."""
    df = load_asset_data(
        symbol=request.symbol,
        freq=request.freq,
        start_date=request.start_date,
        end_date=request.end_date,
    )
    
    if df is None or len(df) == 0:
        raise HTTPException(status_code=404, detail=f"No data found for {request.symbol}")
    
    filter_weekday = None
    filter_month = None
    
    if request.filter_type == "weekday":
        filter_weekday = request.filter_value
    elif request.filter_type == "month":
        filter_month = request.filter_value
    
    intraday = analyze_intraday(
        df,
        metric=request.metric,
        interval_minutes=request.interval_minutes,
        filter_weekday=filter_weekday,
        filter_month=filter_month,
    )
    
    summary = calculate_summary_stats(
        df,
        metric=request.metric,
        filter_weekday=filter_weekday,
        filter_month=filter_month,
    )
    
    return to_python_type({
        'symbol': request.symbol,
        'filter_type': request.filter_type,
        'filter_value': request.filter_value,
        'intraday': intraday,
        'summary': summary,
    })


@router.get("/data/{symbol}")
def get_asset_data(
    symbol: str,
    freq: str = Query("candle_1hour", description="Data frequency"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
) -> Dict[str, Any]:
    """Get raw data info for an asset."""
    df = load_asset_data(symbol, freq, start_date, end_date)
    
    if df is None or len(df) == 0:
        raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
    
    return to_python_type({
        'symbol': symbol,
        'freq': freq,
        'total_rows': len(df),
        'columns': list(df.columns),
        'date_range': {
            'start': str(df['datetime'].min()),
            'end': str(df['datetime'].max()),
        },
    })


@router.get("/calendar/{symbol}")
def get_calendar_heatmap(
    symbol: str,
    freq: str = Query("candle_1hour", description="Data frequency"),
    year: Optional[int] = Query(None, description="Year for heatmap"),
    start_date: Optional[str] = Query(None, description="Start date"),
    end_date: Optional[str] = Query(None, description="End date"),
) -> Dict[str, Any]:
    """Get calendar heatmap data for an asset."""
    df = load_asset_data(symbol, freq, start_date, end_date)
    
    if df is None or len(df) == 0:
        raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
    
    return to_python_type(generate_calendar_heatmap(df, year))

"""
Asset browsing routes - list available frequencies, product types, and assets.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Query

from core.data_org import MktDataTFreq, ProductType
from core.search_data import search_data, list_available

router = APIRouter(prefix="/api/assets", tags=["assets"])


@router.get("/frequencies")
def get_frequencies() -> List[Dict[str, str]]:
    """List available data frequencies."""
    return [
        {"value": MktDataTFreq.CANDLE_1DAY.value, "label": "1 Day Candles"},
        {"value": MktDataTFreq.CANDLE_1HOUR.value, "label": "1 Hour Candles"},
        {"value": MktDataTFreq.CANDLE_1MIN.value, "label": "1 Minute Candles"},
    ]


@router.get("/product-types")
def get_product_types() -> List[Dict[str, str]]:
    """List available product types."""
    return [
        {"value": ProductType.SPOT.value, "label": "Crypto"},
        {"value": ProductType.SPOT.value, "label": "Spot (FX)"},
        {"value": ProductType.ETF.value, "label": "ETF"},
        {"value": ProductType.FUTURE.value, "label": "Future"},
        {"value": ProductType.OPTION.value, "label": "Option"},
    ]


@router.get("/list")
def list_assets(
    freq: Optional[str] = Query(None, description="Data frequency filter"),
    product_type: Optional[str] = Query(None, description="Product type filter"),
) -> Dict[str, Any]:
    """
    List available assets based on filters.
    
    Returns instruments grouped by product type with file counts.
    """
    freq_enum = None
    if freq:
        try:
            freq_enum = MktDataTFreq(freq)
        except ValueError:
            pass

    pt_enum = None
    if product_type:
        try:
            pt_enum = ProductType(product_type)
        except ValueError:
            pass

    available = list_available(
        p_data_freq=freq_enum,
        p_product_type=pt_enum,
    )

    files = search_data(
        p_data_freq=freq_enum,
        p_product_type=pt_enum,
    )

    assets_by_type: Dict[str, List[Dict[str, Any]]] = {}
    for f in files:
        if f.product_type not in assets_by_type:
            assets_by_type[f.product_type] = []
        
        existing = next(
            (a for a in assets_by_type[f.product_type] if a["symbol"] == f.instrument),
            None
        )
        if existing:
            existing["file_count"] += 1
        else:
            pt_value = f.product_type
            try:
                ProductType(pt_value)
            except ValueError:
                pt_value = "spot"
            
            assets_by_type[f.product_type].append({
                "symbol": f.instrument,
                "product_type": pt_value,
                "file_count": 1,
                "data_freq": f.data_freq,
            })

    for pt in assets_by_type:
        assets_by_type[pt].sort(key=lambda x: x["symbol"])

    return {
        "frequencies": available.get("frequencies", []),
        "product_types": available.get("product_types", []),
        "total_instruments": available.get("count", 0),
        "assets_by_type": assets_by_type,
    }

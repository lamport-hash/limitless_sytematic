from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query

from visualiser_app.services import parquet_reader

router = APIRouter(prefix="/api/chart", tags=["chart"])


@router.get("/single/{filename:path}")
def get_single_column_chart(
    filename: str,
    column: str = Query(..., description="Column to chart"),
    limit: Optional[int] = Query(None, description="Limit number of data points"),
) -> Dict[str, Any]:
    try:
        return parquet_reader.get_chart_data(filename, column, limit)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Column not found: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{filename:path}")
def get_chart_data(
    filename: str,
    col1: str = Query(..., description="First column to chart"),
    col2: Optional[str] = Query(None, description="Second column to chart"),
    col3: Optional[str] = Query(None, description="Third column to chart"),
    limit: Optional[int] = Query(None, description="Limit number of data points"),
) -> Dict[str, Any]:
    try:
        datasets = []

        for col in [col1, col2, col3]:
            if col:
                data = parquet_reader.get_chart_data(filename, col, limit)
                datasets.append({
                    "column": col,
                    "x": data["x"],
                    "y": data["y"],
                    "num_points": data["num_points"],
                })

        return {
            "filename": filename,
            "datasets": datasets,
            "num_datasets": len(datasets),
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Column not found: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

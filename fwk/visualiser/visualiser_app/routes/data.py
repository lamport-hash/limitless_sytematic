from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query

from visualiser_app.services import parquet_reader

router = APIRouter(prefix="/api", tags=["data"])


@router.get("/files")
def list_files() -> List[Dict[str, Any]]:
    return parquet_reader.list_bundle_files()


@router.get("/columns/{filename:path}")
def get_columns(filename: str) -> Dict[str, Any]:
    try:
        columns = parquet_reader.get_columns(filename)
        column_types = parquet_reader.get_column_types(filename)
        return {
            "filename": filename,
            "columns": columns,
            "column_types": column_types,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data/{filename:path}")
def get_data(
    filename: str,
    columns: Optional[str] = Query(None, description="Comma-separated list of columns"),
    limit: Optional[int] = Query(None, description="Limit number of rows"),
    offset: int = Query(0, description="Offset for pagination"),
) -> Dict[str, Any]:
    try:
        cols = columns.split(",") if columns else None
        return parquet_reader.get_data(filename, cols, limit, offset)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

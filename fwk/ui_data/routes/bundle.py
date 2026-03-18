"""
Bundle creation routes - start, monitor, and retrieve bundle creation tasks.
"""

import uuid
import shutil
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from core.data_org import MktDataTFreq, ExchangeNAME, ProductType, BUNDLE_DIR, FEATURE_CONFIGS_DIR
from ui_data.services.bundle_service import BundleTaskManager, BundleTaskStatus

router = APIRouter(prefix="/api/bundle", tags=["bundle"])

task_manager = BundleTaskManager()


class BundleCreateRequest(BaseModel):
    assets: List[str]
    product_types: List[str]
    freq: str = "candle_1hour"
    output_prefix: str = "custom_bundle"
    compute_features: bool = True
    feature_config_path: Optional[str] = None
    folder: Optional[str] = None


def _get_freq_suffix(freq: str) -> str:
    if "1hour" in freq or "1_hour" in freq:
        return "_1hour"
    elif "1min" in freq or "1_min" in freq:
        return "_1min"
    elif "5min" in freq or "5_min" in freq:
        return "_5min"
    elif "15min" in freq or "15_min" in freq:
        return "_15min"
    return f"_{freq}"


@router.post("/create")
def create_bundle(
    background_tasks: BackgroundTasks,
    request: BundleCreateRequest,
) -> Dict[str, Any]:
    assets = request.assets
    product_types = request.product_types
    freq = request.freq
    output_prefix = request.output_prefix
    compute_features = request.compute_features
    feature_config_path = request.feature_config_path
    folder = request.folder
    
    if len(assets) != len(product_types):
        raise HTTPException(
            status_code=400,
            detail=f"assets and product_types must have same length: {len(assets)} != {len(product_types)}"
        )

    if not assets:
        raise HTTPException(status_code=400, detail="assets list cannot be empty")

    try:
        freq_enum = MktDataTFreq(freq)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid frequency: {freq}")

    product_type_enums = []
    for pt in product_types:
        try:
            product_type_enums.append(ProductType(pt))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid product type: {pt}")

    freq_suffix = _get_freq_suffix(freq)
    final_prefix = f"{output_prefix}{freq_suffix}"
    
    if folder:
        folder = folder.strip("/").strip("\\")
        output_dir = BUNDLE_DIR / folder
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = BUNDLE_DIR
    
    task_id = str(uuid.uuid4())[:8]
    
    task_manager.create_task(
        task_id=task_id,
        assets=assets,
        product_types=product_type_enums,
        freq=freq_enum,
        output_prefix=final_prefix,
        compute_features=compute_features,
        feature_config_path=feature_config_path,
        output_dir=output_dir,
        folder=folder,
    )

    background_tasks.add_task(
        task_manager.run_task,
        task_id,
    )

    final_filename = f"{final_prefix}_bundle.parquet"
    relative_path = f"{folder}/{final_filename}" if folder else final_filename

    return {
        "task_id": task_id,
        "status": "pending",
        "message": f"Bundle creation started for {len(assets)} assets",
        "assets": assets,
        "freq": freq,
        "compute_features": compute_features,
        "output_path": relative_path,
    }


@router.get("/list")
def list_bundles() -> List[Dict[str, Any]]:
    bundles = []
    
    if not BUNDLE_DIR.exists():
        return bundles
    
    for bundle_file in BUNDLE_DIR.rglob("*_bundle.parquet"):
        stat = bundle_file.stat()
        relative_path = bundle_file.relative_to(BUNDLE_DIR)
        folder = str(relative_path.parent) if relative_path.parent != Path(".") else ""
        
        bundles.append({
            "filename": bundle_file.name,
            "folder": folder,
            "path": str(relative_path),
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        })
    
    return sorted(bundles, key=lambda x: (x["folder"], x["modified"]), reverse=True)


@router.get("/folders")
def list_folders() -> List[Dict[str, Any]]:
    folders = []
    
    if not BUNDLE_DIR.exists():
        return folders
    
    for item in BUNDLE_DIR.rglob("*"):
        if item.is_dir():
            relative = item.relative_to(BUNDLE_DIR)
            bundle_count = len(list(item.glob("*_bundle.parquet")))
            folders.append({
                "path": str(relative),
                "bundle_count": bundle_count,
            })
    
    return sorted(folders, key=lambda x: x["path"])


@router.get("/feature-configs")
def list_feature_configs() -> List[Dict[str, Any]]:
    configs = []
    
    if not FEATURE_CONFIGS_DIR.exists():
        return configs
    
    for config_file in FEATURE_CONFIGS_DIR.glob("*.yaml"):
        configs.append({
            "filename": config_file.name,
            "description": _get_config_description(config_file),
            "modified": datetime.fromtimestamp(config_file.stat().st_mtime).isoformat(),
        })
    
    return sorted(configs, key=lambda x: x["modified"], reverse=True)


def _get_config_description(config_path: Path) -> str:
    try:
        import yaml
        with open(config_path) as f:
            content = yaml.safe_load(f)
            if isinstance(content, dict):
                return content.get("description", "No description")
    except Exception:
        pass
    return "No description"


@router.get("/by-path/{filepath:path}/details")
def get_bundle_details(filepath: str) -> Dict[str, Any]:
    import pyarrow.parquet as pq
    import re
    
    bundle_path = BUNDLE_DIR / filepath
    
    if not bundle_path.exists():
        raise HTTPException(status_code=404, detail=f"Bundle not found: {filepath}")
    
    if not bundle_path.is_file():
        raise HTTPException(status_code=400, detail=f"Not a file: {filepath}")
    
    try:
        table = pq.read_table(bundle_path)
        columns = table.column_names
        df = table.to_pandas()
        
        assets = set()
        features = set()
        
        asset_pattern = re.compile(r'^([A-Z0-9]+)_')
        feature_pattern = re.compile(r'^[A-Z0-9]+_(.+)$')
        
        for col in columns:
            match = asset_pattern.match(col)
            if match:
                assets.add(match.group(1))
            
            feat_match = feature_pattern.match(col)
            if feat_match:
                features.add(feat_match.group(1))
        
        timestamp_col = None
        for col in columns:
            if 'time' in col.lower() or 'date' in col.lower():
                timestamp_col = col
                break
        
        start_date = None
        end_date = None
        if timestamp_col and timestamp_col in df.columns:
            try:
                ts = df[timestamp_col]
                if ts.dtype == 'object':
                    ts = pd.to_datetime(ts)
                start_date = str(ts.min())
                end_date = str(ts.max())
            except Exception:
                pass
        
        if df.index.name and ('time' in df.index.name.lower() or 'date' in df.index.name.lower()):
            try:
                start_date = str(df.index.min())
                end_date = str(df.index.max())
            except Exception:
                pass
        
        folder = str(bundle_path.parent.relative_to(BUNDLE_DIR)) if bundle_path.parent != BUNDLE_DIR else ""
        
        return {
            "filename": bundle_path.name,
            "folder": folder,
            "path": filepath,
            "total_rows": len(df),
            "total_columns": len(columns),
            "assets": sorted(list(assets)),
            "features": sorted(list(features)),
            "start_date": start_date,
            "end_date": end_date,
            "file_size_mb": round(bundle_path.stat().st_size / (1024 * 1024), 2),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading bundle: {str(e)}")


@router.delete("/by-path/{filepath:path}")
def delete_bundle(filepath: str) -> Dict[str, Any]:
    bundle_path = BUNDLE_DIR / filepath
    
    if not bundle_path.exists():
        raise HTTPException(status_code=404, detail=f"Bundle not found: {filepath}")
    
    if not bundle_path.is_file():
        raise HTTPException(status_code=400, detail=f"Not a file: {filepath}")
    
    try:
        bundle_path.unlink()
        
        parent = bundle_path.parent
        while parent != BUNDLE_DIR and parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent
        
        return {"success": True, "message": f"Deleted: {filepath}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting bundle: {str(e)}")


@router.get("/task/{task_id}/status")
def get_task_status(task_id: str) -> Dict[str, Any]:
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    
    return task.to_dict()


@router.get("/task/{task_id}/result")
def get_task_result(task_id: str) -> Dict[str, Any]:
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    
    if task.status == BundleTaskStatus.PENDING:
        raise HTTPException(status_code=400, detail="Task has not started yet")
    
    if task.status == BundleTaskStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Task is still running")
    
    if task.status == BundleTaskStatus.FAILED:
        return {
            "success": False,
            "error": task.error,
            "task_id": task_id,
        }
    
    output_path_str = str(task.output_path) if task.output_path else None
    relative_path = None
    if output_path_str:
        try:
            relative_path = str(Path(output_path_str).relative_to(BUNDLE_DIR))
        except ValueError:
            relative_path = output_path_str
    
    return {
        "success": True,
        "task_id": task_id,
        "output_path": output_path_str,
        "relative_path": relative_path,
        "file_size_mb": task.file_size_mb,
        "total_rows": task.total_rows,
        "feature_count": task.feature_count,
        "assets_processed": task.assets_processed,
        "completed_at": task.completed_at,
    }

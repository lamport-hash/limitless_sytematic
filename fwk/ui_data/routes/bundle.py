"""
Bundle creation routes - start, monitor, and retrieve bundle creation tasks.
"""

import uuid
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

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


@router.post("/create")
def create_bundle(
    background_tasks: BackgroundTasks,
    request: BundleCreateRequest,
) -> Dict[str, Any]:
    """Start a bundle creation task."""
    assets = request.assets
    product_types = request.product_types
    freq = request.freq
    output_prefix = request.output_prefix
    compute_features = request.compute_features
    
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

    task_id = str(uuid.uuid4())[:8]
    
    task_manager.create_task(
        task_id=task_id,
        assets=assets,
        product_types=product_type_enums,
        freq=freq_enum,
        output_prefix=output_prefix,
        compute_features=compute_features,
        feature_config_path=feature_config_path,
    )

    background_tasks.add_task(
        task_manager.run_task,
        task_id,
    )

    return {
        "task_id": task_id,
        "status": "pending",
        "message": f"Bundle creation started for {len(assets)} assets",
        "assets": assets,
        "freq": freq,
        "compute_features": compute_features,
    }


@router.get("/{task_id}/status")
def get_task_status(task_id: str) -> Dict[str, Any]:
    """Get the current status of a bundle creation task."""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    
    return task.to_dict()


@router.get("/{task_id}/result")
def get_task_result(task_id: str) -> Dict[str, Any]:
    """Get the final result of a completed bundle creation task."""
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
    
    return {
        "success": True,
        "task_id": task_id,
        "output_path": str(task.output_path) if task.output_path else None,
        "file_size_mb": task.file_size_mb,
        "total_rows": task.total_rows,
        "feature_count": task.feature_count,
        "assets_processed": task.assets_processed,
        "completed_at": task.completed_at,
    }


@router.get("/list")
def list_bundles() -> List[Dict[str, Any]:
    """List all existing bundle files."""
    bundles = []
    
    if not BUNDLE_DIR.exists():
        return bundles
    
    for bundle_file in BUNDLE_DIR.glob("*_bundle.parquet"):
        stat = bundle_file.stat()
        bundles.append({
            "filename": bundle_file.name,
            "path": str(bundle_file),
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        })
    
    return sorted(bundles, key=lambda x: x["modified"], reverse=True)


@router.get("/feature-configs")
def list_feature_configs() -> List[Dict[str, Any]]:
    """List all available feature config files."""
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

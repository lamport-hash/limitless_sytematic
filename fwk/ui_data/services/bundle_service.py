"""
Bundle creation service - manages background bundle creation tasks.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Callable
from pathlib import Path

import pandas as pd

from core.data_org import MktDataTFreq, ExchangeNAME, ProductType, BUNDLE_DIR
from bundler.feature_bundler import compute_asset_bundle

logger = logging.getLogger(__name__)


class BundleTaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BundleTask:
    """Represents a bundle creation task."""
    
    task_id: str
    assets: List[str]
    product_types: List[ProductType]
    freq: MktDataTFreq
    output_prefix: str
    compute_features: bool
    feature_config_path: Optional[str] = None
    
    status: BundleTaskStatus = BundleTaskStatus.PENDING
    progress: int = 0
    total: int = 0
    message: str = "Task created"
    
    output_path: Optional[Path] = None
    file_size_mb: Optional[float] = None
    total_rows: Optional[int] = None
    feature_count: Optional[int] = None
    assets_processed: int = 0
    
    error: Optional[str] = None
    
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for API response."""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "progress": self.progress,
            "total": self.total,
            "message": self.message,
            "assets": self.assets,
            "freq": self.freq.value,
            "compute_features": self.compute_features,
            "feature_config_path": self.feature_config_path,
            "assets_processed": self.assets_processed,
            "output_path": str(self.output_path) if self.output_path else None,
            "file_size_mb": self.file_size_mb,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


class BundleTaskManager:
    """Manages bundle creation tasks."""
    
    def __init__(self):
        self._tasks: Dict[str, BundleTask] = {}
    
    def create_task(
        self,
        task_id: str,
        assets: List[str],
        product_types: List[ProductType],
        freq: MktDataTFreq,
        output_prefix: str,
        compute_features: bool,
        feature_config_path: Optional[str] = None,
    ) -> BundleTask:
        """Create a new bundle task."""
        task = BundleTask(
            task_id=task_id,
            assets=assets,
            product_types=product_types,
            freq=freq,
            output_prefix=output_prefix,
            compute_features=compute_features,
            total=len(assets),
        )
        self._tasks[task_id] = task
        return task
    
    def get_task(self, task_id: str) -> Optional[BundleTask]:
        """Get a task by ID."""
        return self._tasks.get(task_id)
    
    def run_task(self, task_id: str) -> None:
        """Run a bundle creation task (called as background task)."""
        task = self._tasks.get(task_id)
        if not task:
            logger.error(f"Task not found: {task_id}")
            return
        
        try:
            task.status = BundleTaskStatus.RUNNING
            task.started_at = datetime.now().isoformat()
            task.message = "Starting bundle creation..."
            
            def progress_callback(current: int, total: int, message: str) -> None:
                task.progress = current
                task.message = message
                task.assets_processed = current
                logger.info(f"Task {task_id}: {message}")
            
            output_path = compute_asset_bundle(
                asset_list=task.assets,
                asset_product_type=task.product_types,
                freq=task.freq,
                source=ExchangeNAME.FIRSTRATE,
                output_prefix=task.output_prefix,
                output_dir=BUNDLE_DIR,
                p_verbose=True,
                compute_features=task.compute_features,
                feature_config_path=task.feature_config_path,
                progress_callback=progress_callback,
            )
            
            task.output_path = output_path
            task.status = BundleTaskStatus.COMPLETED
            task.completed_at = datetime.now().isoformat()
            task.message = "Bundle created successfully"
            task.progress = task.total
            
            if output_path.exists():
                stat = output_path.stat()
                task.file_size_mb = round(stat.st_size / (1024 * 1024), 2)
                
                df = pd.read_parquet(output_path)
                task.total_rows = len(df)
                feature_cols = [c for c in df.columns if "_F_" in c]
                task.feature_count = len(feature_cols)
            
            logger.info(f"Task {task_id} completed: {output_path}")
            
        except Exception as e:
            task.status = BundleTaskStatus.FAILED
            task.error = str(e)
            task.message = f"Error: {e}"
            logger.error(f"Task {task_id} failed: {e}", exc_info=True)

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class JobPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkerStatus(str, Enum):
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"


class WorkerRegister(BaseModel):
    name: str
    label: str = ""
    comment: str = ""
    host: str
    port: int
    max_concurrent_jobs: int = Field(default=1, ge=1)


class WorkerHeartbeat(BaseModel):
    current_jobs: int = 0


class Worker(BaseModel):
    worker_id: str
    name: str
    label: str
    comment: str
    host: str
    port: int
    status: WorkerStatus = WorkerStatus.IDLE
    max_concurrent_jobs: int = 1
    current_jobs: int = 0
    last_heartbeat: Optional[datetime] = None
    jobs_completed: int = 0
    jobs_failed: int = 0
    created_at: Optional[datetime] = None

    @property
    def is_available(self) -> bool:
        return (
            self.status != WorkerStatus.OFFLINE
            and self.current_jobs < self.max_concurrent_jobs
        )

    @property
    def load_ratio(self) -> float:
        if self.max_concurrent_jobs == 0:
            return 1.0
        return self.current_jobs / self.max_concurrent_jobs


class JobSubmit(BaseModel):
    name: str
    config: dict[str, Any]
    script: str
    priority: JobPriority = JobPriority.NORMAL
    dispatch_mode: str = "auto"
    target_worker_id: Optional[str] = None
    submitted_by: str = "anonymous"


class JobStatusUpdate(BaseModel):
    status: JobStatus
    error_message: Optional[str] = None
    return_code: Optional[int] = None


class JobOutputChunk(BaseModel):
    output: str
    is_final: bool = False


class Job(BaseModel):
    job_id: str
    name: str
    status: JobStatus = JobStatus.QUEUED
    priority: JobPriority = JobPriority.NORMAL
    script: str
    config: dict[str, Any]
    worker_id: Optional[str] = None
    worker_name: Optional[str] = None
    dispatch_mode: str = "auto"
    target_worker_id: Optional[str] = None
    submitted_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    output: Optional[str] = None
    return_code: Optional[int] = None
    error_message: Optional[str] = None
    submitted_by: str = "anonymous"

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class JobSummary(BaseModel):
    job_id: str
    name: str
    status: JobStatus
    priority: JobPriority
    worker_name: Optional[str]
    submitted_at: Optional[datetime]
    completed_at: Optional[datetime]
    duration_seconds: Optional[float]


class WorkerSummary(BaseModel):
    worker_id: str
    name: str
    label: str
    status: WorkerStatus
    current_jobs: int
    max_concurrent_jobs: int
    jobs_completed: int
    jobs_failed: int


class ManagerStats(BaseModel):
    total_workers: int
    online_workers: int
    idle_workers: int
    busy_workers: int
    offline_workers: int
    total_jobs: int
    queued_jobs: int
    running_jobs: int
    completed_jobs: int
    failed_jobs: int
    jobs_today: int


PRIORITY_ORDER = {
    JobPriority.URGENT: 0,
    JobPriority.HIGH: 1,
    JobPriority.NORMAL: 2,
    JobPriority.LOW: 3,
}

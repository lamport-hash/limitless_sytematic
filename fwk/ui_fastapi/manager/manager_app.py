from __future__ import annotations

import asyncio
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .auth import AuthConfig, create_auth_dependency
from .config import (
    DATABASE_PATH,
    JOB_OUTPUT_DIR,
    settings,
)
from .dispatcher import Dispatcher
from .schemas import (
    Job,
    JobOutputChunk,
    JobPriority,
    JobStatus,
    JobStatusUpdate,
    JobSubmit,
    JobSummary,
    ManagerStats,
    Worker,
    WorkerHeartbeat,
    WorkerRegister,
    WorkerStatus,
    WorkerSummary,
)
from .storage import Storage

BASE_DIR = Path(__file__).resolve().parent

storage = Storage(DATABASE_PATH)
dispatcher = Dispatcher(storage)

auth_config = AuthConfig(settings.get_api_keys())
require_admin = create_auth_dependency(auth_config, ["admin"])
require_worker = create_auth_dependency(auth_config, ["worker"])
require_client = create_auth_dependency(auth_config, ["client", "admin"])
require_any = create_auth_dependency(auth_config, ["client", "worker", "admin"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(JOB_OUTPUT_DIR, exist_ok=True)
    checker = threading.Thread(target=check_offline_workers, daemon=True)
    checker.start()
    yield


app = FastAPI(title="Fitting Manager", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


def check_offline_workers():
    while True:
        try:
            threshold = datetime.utcnow() - timedelta(
                seconds=settings.worker_offline_timeout
            )
            workers = storage.get_all_workers()
            for worker in workers:
                if (
                    worker.status != WorkerStatus.OFFLINE
                    and worker.last_heartbeat
                    and worker.last_heartbeat < threshold
                ):
                    storage.set_worker_offline(worker.worker_id)
        except Exception:
            pass
        threading.Event().wait(15)


@app.get("/", response_class=HTMLResponse)
async def ui(request: Request):
    return templates.TemplateResponse("manager.html", {"request": request})


@app.get("/api/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.get("/api/stats", response_model=ManagerStats)
async def get_stats(api_key: str = Depends(require_client)):
    return storage.get_stats()


# === WORKERS ===


@app.post("/api/workers/register", response_model=Worker)
async def register_worker(data: WorkerRegister, api_key: str = Depends(require_worker)):
    worker_id = f"worker-{data.name}"
    worker = storage.register_worker(
        {
            "worker_id": worker_id,
            "name": data.name,
            "label": data.label,
            "comment": data.comment,
            "host": data.host,
            "port": data.port,
            "max_concurrent_jobs": data.max_concurrent_jobs,
        }
    )
    return worker


@app.post("/api/workers/heartbeat", response_model=Worker)
async def worker_heartbeat(
    data: WorkerHeartbeat, api_key: str = Depends(require_worker)
):
    worker_name = getattr(api_key, "worker_name", None)
    if not worker_name:
        raise HTTPException(status_code=400, detail="Worker not identified")
    worker = storage.get_worker_by_name(worker_name)
    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")
    return storage.update_worker_heartbeat(worker.worker_id, data.current_jobs)


@app.get("/api/workers", response_model=list[WorkerSummary])
async def list_workers(api_key: str = Depends(require_client)):
    workers = storage.get_all_workers()
    return [
        WorkerSummary(
            worker_id=w.worker_id,
            name=w.name,
            label=w.label,
            status=w.status,
            current_jobs=w.current_jobs,
            max_concurrent_jobs=w.max_concurrent_jobs,
            jobs_completed=w.jobs_completed,
            jobs_failed=w.jobs_failed,
        )
        for w in workers
    ]


@app.get("/api/workers/{worker_id}", response_model=Worker)
async def get_worker(worker_id: str, api_key: str = Depends(require_client)):
    worker = storage.get_worker(worker_id)
    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")
    return worker


@app.delete("/api/workers/{worker_id}")
async def delete_worker(worker_id: str, api_key: str = Depends(require_admin)):
    if not storage.delete_worker(worker_id):
        raise HTTPException(status_code=404, detail="Worker not found")
    return {"success": True, "message": f"Worker {worker_id} deleted"}


# === JOBS ===


@app.post("/api/jobs", response_model=Job)
async def submit_job(data: JobSubmit, api_key: str = Depends(require_client)):
    return dispatcher.submit_job(data)


@app.get("/api/jobs", response_model=list[JobSummary])
async def list_jobs(
    status: Optional[JobStatus] = None,
    worker_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    api_key: str = Depends(require_client),
):
    jobs = storage.get_jobs(
        status=status, worker_id=worker_id, limit=limit, offset=offset
    )
    return [
        JobSummary(
            job_id=j.job_id,
            name=j.name,
            status=j.status,
            priority=j.priority,
            worker_name=j.worker_name,
            submitted_at=j.submitted_at,
            completed_at=j.completed_at,
            duration_seconds=j.duration_seconds,
        )
        for j in jobs
    ]


@app.get("/api/jobs/queued", response_model=list[Job])
async def list_queued_jobs(api_key: str = Depends(require_client)):
    return storage.get_queued_jobs()


@app.get("/api/jobs/{job_id}", response_model=Job)
async def get_job(job_id: str, api_key: str = Depends(require_client)):
    job = storage.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str, api_key: str = Depends(require_client)):
    job = storage.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status == JobStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Cannot delete running job")
    if not storage.delete_job(job_id):
        raise HTTPException(status_code=500, detail="Failed to delete job")
    return {"success": True, "message": f"Job {job_id} deleted"}


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, api_key: str = Depends(require_client)):
    job = dispatcher.cancel_job(job_id)
    if not job:
        raise HTTPException(
            status_code=404, detail="Job not found or cannot be cancelled"
        )
    return {"success": True, "job": job}


@app.post("/api/jobs/{job_id}/rerun", response_model=Job)
async def rerun_job(job_id: str, api_key: str = Depends(require_client)):
    job = dispatcher.rerun_job(job_id, submitted_by=api_key)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/jobs/{job_id}/output")
async def get_job_output(job_id: str, api_key: str = Depends(require_client)):
    job = storage.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"output": job.output or "", "status": job.status}


# === INTERNAL WORKER API ===


@app.get("/internal/workers/{worker_id}/job", response_model=Optional[Job])
async def get_next_job(worker_id: str, api_key: str = Depends(require_worker)):
    job = dispatcher.get_next_job_for_worker(worker_id)
    return job


@app.post("/internal/jobs/{job_id}/status", response_model=Job)
async def update_job_status(
    job_id: str,
    data: JobStatusUpdate,
    api_key: str = Depends(require_worker),
):
    job = storage.update_job_status(
        job_id,
        data.status,
        return_code=data.return_code,
        error_message=data.error_message,
    )
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.worker_id:
        completed = data.status == JobStatus.COMPLETED
        failed = data.status == JobStatus.FAILED
        storage.update_worker_job_stats(
            job.worker_id, completed=completed, failed=failed
        )

    return job


@app.post("/internal/jobs/{job_id}/output")
async def append_job_output(
    job_id: str,
    data: JobOutputChunk,
    api_key: str = Depends(require_worker),
):
    storage.append_job_output(job_id, data.output)
    return {"success": True}


# === HISTORY ===


@app.get("/api/history", response_model=list[Job])
async def get_history(
    days: int = 7,
    status: Optional[JobStatus] = None,
    limit: int = 200,
    api_key: str = Depends(require_client),
):
    return storage.get_jobs(status=status, limit=limit)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.manager_host, port=settings.manager_port)

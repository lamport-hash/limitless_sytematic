from __future__ import annotations

import os
import threading
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .config import WorkerConfig
from .executor import JobExecutor

config = WorkerConfig()

worker_id: str | None = None
executor: JobExecutor | None = None


def get_manager_client():
    return httpx.Client(
        base_url=config.manager_url,
        headers={"X-API-Key": config.api_key},
        timeout=30.0,
    )


def register_with_manager() -> str | None:
    global worker_id
    try:
        with get_manager_client() as client:
            response = client.post("/api/workers/register", json=config.to_dict())
            if response.status_code == 200:
                data = response.json()
                worker_id = data.get("worker_id")
                print(f"Registered with manager as {worker_id}")
                return worker_id
            else:
                print(f"Failed to register: {response.text}")
                return None
    except Exception as e:
        print(f"Error registering with manager: {e}")
        return None


def send_heartbeat(current_jobs: int):
    global worker_id
    if not worker_id:
        return
    try:
        with get_manager_client() as client:
            client.post("/api/workers/heartbeat", json={"current_jobs": current_jobs})
    except Exception as e:
        print(f"Heartbeat error: {e}")


def on_job_output(job_id: str, output: str):
    if not worker_id:
        return
    try:
        with get_manager_client() as client:
            client.post(f"/internal/jobs/{job_id}/output", json={"output": output})
    except Exception as e:
        print(f"Error sending output: {e}")


def on_job_complete(job_id: str, return_code: int, output: str):
    if not worker_id:
        return
    try:
        with get_manager_client() as client:
            status = "completed" if return_code == 0 else "failed"
            client.post(
                f"/internal/jobs/{job_id}/status",
                json={"status": status, "return_code": return_code},
            )
    except Exception as e:
        print(f"Error sending completion: {e}")


def poll_for_jobs():
    global worker_id, executor
    while True:
        if not worker_id:
            time.sleep(5)
            register_with_manager()
            continue

        if executor and executor.active_count < config.max_concurrent_jobs:
            try:
                with get_manager_client() as client:
                    response = client.get(f"/internal/workers/{worker_id}/job")
                    if response.status_code == 200:
                        job = response.json()
                        if job:
                            print(f"Received job: {job.get('job_id')}")
                            executor.execute(job["job_id"], job["script"])
            except Exception as e:
                print(f"Error polling for jobs: {e}")

        if executor:
            send_heartbeat(executor.active_count)

        time.sleep(config.poll_interval)


def heartbeat_loop():
    while True:
        if worker_id and executor:
            send_heartbeat(executor.active_count)
        time.sleep(config.heartbeat_interval)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global executor

    register_with_manager()

    executor = JobExecutor(on_output=on_job_output, on_complete=on_job_complete)

    poll_thread = threading.Thread(target=poll_for_jobs, daemon=True)
    poll_thread.start()

    yield


app = FastAPI(title="Fitting Worker", lifespan=lifespan)


@app.get("/api/status")
async def status():
    return {
        "worker_id": worker_id,
        "config": config.to_dict(),
        "active_jobs": executor.active_count if executor else 0,
        "max_jobs": config.max_concurrent_jobs,
    }


@app.get("/api/health")
async def health():
    return {"status": "healthy", "worker_id": worker_id}


@app.post("/api/stop/{job_id}")
async def stop_job(job_id: str):
    if not executor:
        raise HTTPException(status_code=500, detail="Executor not initialized")
    if executor.stop(job_id):
        return {"success": True, "message": f"Job {job_id} stopped"}
    raise HTTPException(status_code=404, detail="Job not found or already completed")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=config.port)

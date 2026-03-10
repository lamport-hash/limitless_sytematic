from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from runner import process_manager
from schemas import (
    CLASSIFICATION_METRICS,
    DEFAULT_PARAMS,
    FittingConfig,
    MetricType,
    ModelType,
    REGRESSION_METRICS,
    TaskType,
)
from script_generator import generate_script

app = FastAPI(title="Fitting API UI")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

MANAGER_URL = os.getenv("MANAGER_URL", "http://localhost:8888")
MANAGER_API_KEY = os.getenv("MANAGER_API_KEY", "client-key-001")

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/data-files")
async def list_data_files():
    files = []
    if DATA_DIR.exists():
        for f in DATA_DIR.iterdir():
            if f.is_file() and f.suffix in [".csv", ".parquet", ".pkl"]:
                files.append({"name": f.name, "path": str(f)})
    return {"files": files}


@app.get("/api/models")
async def list_models():
    return {
        "models": [
            {
                "type": ModelType.XGB.value,
                "name": "XGBoost",
                "supports": ["regression", "classification"],
            },
            {
                "type": ModelType.RF_SK.value,
                "name": "Random Forest",
                "supports": ["regression", "classification"],
            },
            {
                "type": ModelType.DT_SK.value,
                "name": "Decision Tree",
                "supports": ["regression", "classification"],
            },
            {
                "type": ModelType.MLP_TORCH.value,
                "name": "MLP (PyTorch)",
                "supports": ["regression", "classification"],
            },
            {
                "type": ModelType.BNN_GAUTO.value,
                "name": "AutoBNN",
                "supports": ["classification"],
            },
        ]
    }


@app.get("/api/models/{model_type}/params")
async def get_model_params(model_type: str):
    try:
        mt = ModelType(model_type)
        return {"params": DEFAULT_PARAMS.get(mt, {})}
    except ValueError:
        raise HTTPException(status_code=404, detail="Model type not found")


@app.get("/api/metrics/{task_type}")
async def get_metrics(task_type: str):
    if task_type == "regression":
        return {"metrics": [m.value for m in REGRESSION_METRICS]}
    elif task_type == "classification":
        return {"metrics": [m.value for m in CLASSIFICATION_METRICS]}
    else:
        raise HTTPException(status_code=400, detail="Invalid task type")


@app.post("/api/generate-script")
async def api_generate_script(config: FittingConfig):
    try:
        script = generate_script(config)
        return {"script": script, "success": True}
    except Exception as e:
        return {"script": "", "success": False, "error": str(e)}


@app.post("/api/run")
async def run_script(config: FittingConfig):
    try:
        script = generate_script(config)
        pid = process_manager.start_process(script)
        return {"pid": pid, "success": True}
    except Exception as e:
        return {"pid": None, "success": False, "error": str(e)}


@app.websocket("/ws/stream/{pid}")
async def websocket_stream(websocket: WebSocket, pid: int):
    await websocket.accept()
    last_line = 0

    try:
        while True:
            new_lines, is_complete = process_manager.get_output(pid, last_line)

            for line in new_lines:
                await websocket.send_json({"type": "output", "data": line})

            last_line += len(new_lines)

            if is_complete:
                status = process_manager.get_status(pid)
                await websocket.send_json(
                    {
                        "type": "complete",
                        "status": status.get("status") if status else "unknown",
                        "return_code": status.get("return_code") if status else None,
                    }
                )
                break

            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({"type": "error", "data": str(e)})


@app.post("/api/stop/{pid}")
async def stop_process(pid: int):
    success = process_manager.stop_process(pid)
    if success:
        return {"success": True, "message": f"Process {pid} stopped"}
    else:
        return {"success": False, "message": f"Process {pid} not found"}


@app.get("/api/status/{pid}")
async def get_status(pid: int):
    status = process_manager.get_status(pid)
    if status:
        return status
    else:
        raise HTTPException(status_code=404, detail="Process not found")


@app.get("/api/manager/workers")
async def get_manager_workers():
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                f"{MANAGER_URL}/api/workers",
                headers={"X-API-Key": MANAGER_API_KEY},
            )
            if response.status_code == 200:
                return {"workers": response.json(), "success": True}
            return {"workers": [], "success": False, "error": response.text}
    except Exception as e:
        return {"workers": [], "success": False, "error": str(e)}


@app.get("/api/manager/status")
async def get_manager_status():
    return {
        "manager_url": MANAGER_URL,
        "configured": True,
    }


@app.post("/api/manager/submit")
async def submit_to_manager(
    config: FittingConfig,
    name: Optional[str] = None,
    priority: str = "normal",
    dispatch_mode: str = "auto",
    target_worker_id: Optional[str] = None,
):
    try:
        script = generate_script(config)
        job_name = name or f"{config.model_type}_{config.task_type}"

        payload = {
            "name": job_name,
            "script": script,
            "config": config.model_dump(),
            "priority": priority,
            "dispatch_mode": dispatch_mode,
        }

        if dispatch_mode == "manual" and target_worker_id:
            payload["target_worker_id"] = target_worker_id

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{MANAGER_URL}/api/jobs",
                headers={
                    "X-API-Key": MANAGER_API_KEY,
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            if response.status_code == 200:
                return {"job": response.json(), "success": True}
            return {"job": None, "success": False, "error": response.text}
    except Exception as e:
        return {"job": None, "success": False, "error": str(e)}


@app.on_event("startup")
async def startup():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    manager_port: int = 8888
    manager_host: str = "0.0.0.0"
    database_path: str = "./data/manager.db"
    worker_offline_timeout: int = 60
    job_output_dir: str = "./data/output"
    api_keys_str: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def get_api_keys(self) -> dict[str, dict]:
        if self.api_keys_str:
            import json

            try:
                return json.loads(self.api_keys_str)
            except:
                pass
        return {
            "mgr-admin-001": {"role": "admin"},
            "worker-key-001": {"role": "worker"},
            "client-key-001": {"role": "client"},
        }


settings = Settings()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UI_FASTAPI_DIR = os.path.dirname(BASE_DIR)
DATABASE_PATH = os.path.join(UI_FASTAPI_DIR, settings.database_path)
JOB_OUTPUT_DIR = os.path.join(UI_FASTAPI_DIR, settings.job_output_dir)

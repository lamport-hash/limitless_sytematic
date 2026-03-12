from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml


class WorkerConfig:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._find_config()
        self._raw = self._load_config()

    def _find_config(self) -> str:
        candidates = [
            "worker_config.yaml",
            "worker_config.yml",
            os.path.join(os.path.dirname(__file__), "worker_config.yaml"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return candidates[0]

    def _load_config(self) -> dict:
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

    @property
    def name(self) -> str:
        return self._raw.get("worker", {}).get("name", f"worker-{os.uname().nodename}")

    @property
    def label(self) -> str:
        return self._raw.get("worker", {}).get("label", "")

    @property
    def comment(self) -> str:
        return self._raw.get("worker", {}).get("comment", "")

    @property
    def port(self) -> int:
        return self._raw.get("worker", {}).get("port", 8890)

    @property
    def host(self) -> str:
        import socket

        return self._raw.get("worker", {}).get(
            "host", socket.gethostbyname(socket.gethostname())
        )

    @property
    def max_concurrent_jobs(self) -> int:
        return self._raw.get("worker", {}).get("max_concurrent_jobs", 1)

    @property
    def manager_url(self) -> str:
        return self._raw.get("manager", {}).get("url", "http://localhost:8888")

    @property
    def api_key(self) -> str:
        return self._raw.get("manager", {}).get("api_key", "worker-key-001")

    @property
    def heartbeat_interval(self) -> int:
        return self._raw.get("manager", {}).get("heartbeat_interval", 15)

    @property
    def poll_interval(self) -> int:
        return self._raw.get("manager", {}).get("poll_interval", 3)

    @property
    def output_dir(self) -> str:
        path = self._raw.get("paths", {}).get("output_dir", "../data/output")
        if not os.path.isabs(path):
            base = os.path.dirname(os.path.dirname(__file__))
            path = os.path.join(base, path)
        return path

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "comment": self.comment,
            "host": self.host,
            "port": self.port,
            "max_concurrent_jobs": self.max_concurrent_jobs,
        }

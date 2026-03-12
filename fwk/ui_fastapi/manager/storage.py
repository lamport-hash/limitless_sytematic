from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Optional

from .schemas import (
    Job,
    JobPriority,
    JobStatus,
    JobSummary,
    JobSubmit,
    ManagerStats,
    Worker,
    WorkerStatus,
    WorkerSummary,
)


class Storage:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    @contextmanager
    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS workers (
                    worker_id TEXT PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    label TEXT DEFAULT '',
                    comment TEXT DEFAULT '',
                    host TEXT NOT NULL,
                    port INTEGER NOT NULL,
                    status TEXT DEFAULT 'idle',
                    max_concurrent_jobs INTEGER DEFAULT 1,
                    current_jobs INTEGER DEFAULT 0,
                    last_heartbeat TEXT,
                    jobs_completed INTEGER DEFAULT 0,
                    jobs_failed INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT DEFAULT 'queued',
                    priority TEXT DEFAULT 'normal',
                    script TEXT NOT NULL,
                    config TEXT,
                    worker_id TEXT,
                    worker_name TEXT,
                    dispatch_mode TEXT DEFAULT 'auto',
                    target_worker_id TEXT,
                    submitted_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    started_at TEXT,
                    completed_at TEXT,
                    output TEXT,
                    return_code INTEGER,
                    error_message TEXT,
                    submitted_by TEXT DEFAULT 'anonymous',
                    FOREIGN KEY (worker_id) REFERENCES workers(worker_id)
                );

                CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
                CREATE INDEX IF NOT EXISTS idx_jobs_priority ON jobs(priority);
                CREATE INDEX IF NOT EXISTS idx_workers_status ON workers(status);
                CREATE INDEX IF NOT EXISTS idx_jobs_submitted ON jobs(submitted_at);
            """)
            conn.commit()

    def register_worker(self, data: dict) -> Worker:
        with self._lock:
            with self._get_conn() as conn:
                now = datetime.utcnow().isoformat()
                cursor = conn.execute(
                    """
                    INSERT INTO workers (worker_id, name, label, comment, host, port, 
                                         max_concurrent_jobs, status, last_heartbeat, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'idle', ?, ?)
                    ON CONFLICT(name) DO UPDATE SET
                        label = excluded.label,
                        comment = excluded.comment,
                        host = excluded.host,
                        port = excluded.port,
                        max_concurrent_jobs = excluded.max_concurrent_jobs,
                        status = 'idle',
                        last_heartbeat = excluded.last_heartbeat
                    """,
                    (
                        data.get("worker_id") or f"worker-{data['name']}",
                        data["name"],
                        data.get("label", ""),
                        data.get("comment", ""),
                        data["host"],
                        data["port"],
                        data.get("max_concurrent_jobs", 1),
                        now,
                        now,
                    ),
                )
                conn.commit()
                worker_id = cursor.lastrowid or f"worker-{data['name']}"
                return self.get_worker_by_name(data["name"])

    def get_worker(self, worker_id: str) -> Optional[Worker]:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM workers WHERE worker_id = ?", (worker_id,)
            ).fetchone()
            return self._row_to_worker(row) if row else None

    def get_worker_by_name(self, name: str) -> Optional[Worker]:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM workers WHERE name = ?", (name,)
            ).fetchone()
            return self._row_to_worker(row) if row else None

    def get_all_workers(self) -> list[Worker]:
        with self._get_conn() as conn:
            rows = conn.execute("SELECT * FROM workers ORDER BY name").fetchall()
            return [self._row_to_worker(r) for r in rows]

    def get_available_workers(self) -> list[Worker]:
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM workers 
                WHERE status != 'offline' 
                AND current_jobs < max_concurrent_jobs
                ORDER BY current_jobs * 1.0 / max_concurrent_jobs ASC
                """
            ).fetchall()
            return [self._row_to_worker(r) for r in rows]

    def update_worker_heartbeat(
        self, worker_id: str, current_jobs: int
    ) -> Optional[Worker]:
        with self._lock:
            with self._get_conn() as conn:
                now = datetime.utcnow().isoformat()
                status = "busy" if current_jobs > 0 else "idle"
                conn.execute(
                    """
                    UPDATE workers 
                    SET last_heartbeat = ?, current_jobs = ?, status = ?
                    WHERE worker_id = ?
                    """,
                    (now, current_jobs, status, worker_id),
                )
                conn.commit()
                return self.get_worker(worker_id)

    def update_worker_job_stats(
        self, worker_id: str, completed: bool = False, failed: bool = False
    ):
        with self._lock:
            with self._get_conn() as conn:
                if completed:
                    conn.execute(
                        "UPDATE workers SET jobs_completed = jobs_completed + 1 WHERE worker_id = ?",
                        (worker_id,),
                    )
                if failed:
                    conn.execute(
                        "UPDATE workers SET jobs_failed = jobs_failed + 1 WHERE worker_id = ?",
                        (worker_id,),
                    )
                conn.commit()

    def set_worker_offline(self, worker_id: str):
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    "UPDATE workers SET status = 'offline' WHERE worker_id = ?",
                    (worker_id,),
                )
                conn.commit()

    def delete_worker(self, worker_id: str) -> bool:
        with self._lock:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    "DELETE FROM workers WHERE worker_id = ?", (worker_id,)
                )
                conn.commit()
                return cursor.rowcount > 0

    def create_job(self, job_id: str, data: JobSubmit) -> Job:
        with self._lock:
            with self._get_conn() as conn:
                now = datetime.utcnow().isoformat()
                conn.execute(
                    """
                    INSERT INTO jobs (job_id, name, status, priority, script, config, 
                                     dispatch_mode, target_worker_id, submitted_at, submitted_by)
                    VALUES (?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        data.name,
                        data.priority.value,
                        data.script,
                        json.dumps(data.config),
                        data.dispatch_mode,
                        data.target_worker_id,
                        now,
                        data.submitted_by,
                    ),
                )
                conn.commit()
                return self.get_job(job_id)

    def get_job(self, job_id: str) -> Optional[Job]:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            return self._row_to_job(row) if row else None

    def get_jobs(
        self,
        status: Optional[JobStatus] = None,
        worker_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Job]:
        with self._get_conn() as conn:
            query = "SELECT * FROM jobs WHERE 1=1"
            params: list[Any] = []

            if status:
                query += " AND status = ?"
                params.append(status.value)
            if worker_id:
                query += " AND worker_id = ?"
                params.append(worker_id)

            query += " ORDER BY submitted_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            rows = conn.execute(query, params).fetchall()
            return [self._row_to_job(r) for r in rows]

    def get_queued_jobs(self, limit: int = 100) -> list[Job]:
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM jobs 
                WHERE status = 'queued' 
                ORDER BY 
                    CASE priority 
                        WHEN 'urgent' THEN 0 
                        WHEN 'high' THEN 1 
                        WHEN 'normal' THEN 2 
                        WHEN 'low' THEN 3 
                    END,
                    submitted_at ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [self._row_to_job(r) for r in rows]

    def assign_job(
        self, job_id: str, worker_id: str, worker_name: str
    ) -> Optional[Job]:
        with self._lock:
            with self._get_conn() as conn:
                now = datetime.utcnow().isoformat()
                conn.execute(
                    """
                    UPDATE jobs 
                    SET status = 'running', worker_id = ?, worker_name = ?, started_at = ?
                    WHERE job_id = ? AND status = 'queued'
                    """,
                    (worker_id, worker_name, now, job_id),
                )
                conn.commit()
                return self.get_job(job_id)

    def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        output: Optional[str] = None,
        return_code: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> Optional[Job]:
        with self._lock:
            with self._get_conn() as conn:
                now = datetime.utcnow().isoformat()
                sets = ["status = ?"]
                params: list[Any] = [status.value]

                if status in (
                    JobStatus.COMPLETED,
                    JobStatus.FAILED,
                    JobStatus.CANCELLED,
                ):
                    sets.append("completed_at = ?")
                    params.append(now)

                if output is not None:
                    sets.append("output = ?")
                    params.append(output)

                if return_code is not None:
                    sets.append("return_code = ?")
                    params.append(return_code)

                if error_message is not None:
                    sets.append("error_message = ?")
                    params.append(error_message)

                params.append(job_id)
                conn.execute(
                    f"UPDATE jobs SET {', '.join(sets)} WHERE job_id = ?",
                    params,
                )
                conn.commit()
                return self.get_job(job_id)

    def append_job_output(self, job_id: str, output: str):
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    "UPDATE jobs SET output = COALESCE(output, '') || ? WHERE job_id = ?",
                    (output, job_id),
                )
                conn.commit()

    def delete_job(self, job_id: str) -> bool:
        with self._lock:
            with self._get_conn() as conn:
                cursor = conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
                conn.commit()
                return cursor.rowcount > 0

    def get_stats(self) -> ManagerStats:
        with self._get_conn() as conn:
            workers = conn.execute(
                "SELECT status, COUNT(*) as cnt FROM workers GROUP BY status"
            ).fetchall()
            worker_stats = {r["status"]: r["cnt"] for r in workers}

            jobs = conn.execute(
                "SELECT status, COUNT(*) as cnt FROM jobs GROUP BY status"
            ).fetchall()
            job_stats = {r["status"]: r["cnt"] for r in jobs}

            today = datetime.utcnow().date().isoformat()
            jobs_today = conn.execute(
                "SELECT COUNT(*) as cnt FROM jobs WHERE date(submitted_at) = ?",
                (today,),
            ).fetchone()["cnt"]

            return ManagerStats(
                total_workers=sum(worker_stats.values()),
                online_workers=worker_stats.get("idle", 0)
                + worker_stats.get("busy", 0),
                idle_workers=worker_stats.get("idle", 0),
                busy_workers=worker_stats.get("busy", 0),
                offline_workers=worker_stats.get("offline", 0),
                total_jobs=sum(job_stats.values()),
                queued_jobs=job_stats.get("queued", 0),
                running_jobs=job_stats.get("running", 0),
                completed_jobs=job_stats.get("completed", 0),
                failed_jobs=job_stats.get("failed", 0),
                jobs_today=jobs_today,
            )

    def _row_to_worker(self, row: sqlite3.Row) -> Worker:
        return Worker(
            worker_id=row["worker_id"],
            name=row["name"],
            label=row["label"] or "",
            comment=row["comment"] or "",
            host=row["host"],
            port=row["port"],
            status=WorkerStatus(row["status"]),
            max_concurrent_jobs=row["max_concurrent_jobs"],
            current_jobs=row["current_jobs"],
            last_heartbeat=datetime.fromisoformat(row["last_heartbeat"])
            if row["last_heartbeat"]
            else None,
            jobs_completed=row["jobs_completed"],
            jobs_failed=row["jobs_failed"],
            created_at=datetime.fromisoformat(row["created_at"])
            if row["created_at"]
            else None,
        )

    def _row_to_job(self, row: sqlite3.Row) -> Job:
        return Job(
            job_id=row["job_id"],
            name=row["name"],
            status=JobStatus(row["status"]),
            priority=JobPriority(row["priority"]),
            script=row["script"],
            config=json.loads(row["config"]) if row["config"] else {},
            worker_id=row["worker_id"],
            worker_name=row["worker_name"],
            dispatch_mode=row["dispatch_mode"],
            target_worker_id=row["target_worker_id"],
            submitted_at=datetime.fromisoformat(row["submitted_at"])
            if row["submitted_at"]
            else None,
            started_at=datetime.fromisoformat(row["started_at"])
            if row["started_at"]
            else None,
            completed_at=datetime.fromisoformat(row["completed_at"])
            if row["completed_at"]
            else None,
            output=row["output"],
            return_code=row["return_code"],
            error_message=row["error_message"],
            submitted_by=row["submitted_by"] or "anonymous",
        )

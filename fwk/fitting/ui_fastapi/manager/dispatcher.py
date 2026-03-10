from __future__ import annotations

import uuid
from typing import Optional

from .schemas import Job, JobPriority, JobSubmit, Worker
from .storage import Storage


class Dispatcher:
    def __init__(self, storage: Storage):
        self.storage = storage

    def submit_job(self, data: JobSubmit) -> Job:
        job_id = f"job-{uuid.uuid4().hex[:8]}"
        return self.storage.create_job(job_id, data)

    def get_next_job_for_worker(self, worker_id: str) -> Optional[Job]:
        worker = self.storage.get_worker(worker_id)
        if not worker or not worker.is_available:
            return None

        queued_jobs = self.storage.get_queued_jobs(limit=50)

        for job in queued_jobs:
            if job.dispatch_mode == "auto":
                assigned = self.storage.assign_job(
                    job.job_id, worker.worker_id, worker.name
                )
                if assigned:
                    return assigned

            elif job.dispatch_mode == "manual" and job.target_worker_id == worker_id:
                assigned = self.storage.assign_job(
                    job.job_id, worker.worker_id, worker.name
                )
                if assigned:
                    return assigned

        return None

    def dispatch_manual_job(self, job_id: str, worker_id: str) -> Optional[Job]:
        job = self.storage.get_job(job_id)
        if not job or job.status != "queued":
            return None

        worker = self.storage.get_worker(worker_id)
        if not worker or not worker.is_available:
            return None

        return self.storage.assign_job(job_id, worker.worker_id, worker.name)

    def cancel_job(self, job_id: str) -> Optional[Job]:
        job = self.storage.get_job(job_id)
        if not job:
            return None

        if job.status in ("queued", "running"):
            from .schemas import JobStatus

            return self.storage.update_job_status(job_id, JobStatus.CANCELLED)
        return job

    def rerun_job(self, job_id: str, submitted_by: str = "system") -> Optional[Job]:
        original = self.storage.get_job(job_id)
        if not original:
            return None

        new_job_id = f"job-{uuid.uuid4().hex[:8]}"
        new_submit = JobSubmit(
            name=f"Rerun: {original.name}",
            config=original.config,
            script=original.script,
            priority=original.priority,
            dispatch_mode=original.dispatch_mode,
            target_worker_id=original.worker_id,
            submitted_by=submitted_by,
        )
        return self.storage.create_job(new_job_id, new_submit)

    def get_best_worker(self) -> Optional[Worker]:
        workers = self.storage.get_available_workers()
        if not workers:
            return None

        return min(workers, key=lambda w: w.load_ratio)

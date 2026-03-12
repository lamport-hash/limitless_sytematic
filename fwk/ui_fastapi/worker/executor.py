from __future__ import annotations

import subprocess
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RunningJob:
    job_id: str
    process: subprocess.Popen
    script_path: str
    output_buffer: list[str] = field(default_factory=list)
    is_complete: bool = False
    return_code: Optional[int] = None


class JobExecutor:
    def __init__(self, on_output, on_complete):
        self.on_output = on_output
        self.on_complete = on_complete
        self._running_jobs: dict[str, RunningJob] = {}
        self._lock = threading.Lock()

    @property
    def active_count(self) -> int:
        with self._lock:
            return len([j for j in self._running_jobs.values() if not j.is_complete])

    def execute(self, job_id: str, script: str) -> bool:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script)
            script_path = f.name

        process = subprocess.Popen(
            ["python", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        job = RunningJob(
            job_id=job_id,
            process=process,
            script_path=script_path,
        )

        with self._lock:
            self._running_jobs[job_id] = job

        threading.Thread(
            target=self._read_output,
            args=(job,),
            daemon=True,
        ).start()

        return True

    def _read_output(self, job: RunningJob):
        try:
            for line in iter(job.process.stdout.readline, ""):
                if line:
                    job.output_buffer.append(line)
                    if self.on_output:
                        self.on_output(job.job_id, line)
        except Exception:
            pass
        finally:
            job.process.wait()
            job.return_code = job.process.returncode
            job.is_complete = True

            try:
                import os

                os.unlink(job.script_path)
            except Exception:
                pass

            if self.on_complete:
                self.on_complete(
                    job.job_id, job.return_code, "".join(job.output_buffer)
                )

    def stop(self, job_id: str) -> bool:
        with self._lock:
            job = self._running_jobs.get(job_id)
            if not job or job.is_complete:
                return False

        try:
            job.process.terminate()
            job.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            job.process.kill()
        except Exception:
            pass

        return True

    def get_status(self, job_id: str) -> Optional[dict]:
        with self._lock:
            job = self._running_jobs.get(job_id)
            if not job:
                return None
            return {
                "job_id": job.job_id,
                "is_complete": job.is_complete,
                "return_code": job.return_code,
                "output_lines": len(job.output_buffer),
            }

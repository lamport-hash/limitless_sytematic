from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ProcessInfo:
    pid: int
    process: subprocess.Popen
    output_file: str
    status: str = "running"
    return_code: Optional[int] = None


class ProcessManager:
    _instance: Optional["ProcessManager"] = None

    def __new__(cls) -> "ProcessManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._processes: dict[int, ProcessInfo] = {}
            cls._instance._output_buffers: dict[int, list[str]] = defaultdict(list)
            cls._instance._lock = threading.Lock()
        return cls._instance

    def start_process(self, script_content: str) -> int:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script_content)
            script_path = f.name

        process = subprocess.Popen(
            ["python", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=os.getcwd(),
        )

        pid = process.pid
        with self._lock:
            self._processes[pid] = ProcessInfo(
                pid=pid,
                process=process,
                output_file=script_path,
            )
            self._output_buffers[pid] = []

        threading.Thread(
            target=self._read_output,
            args=(pid,),
            daemon=True,
        ).start()

        return pid

    def _read_output(self, pid: int) -> None:
        with self._lock:
            if pid not in self._processes:
                return
            process_info = self._processes[pid]
            process = process_info.process

        try:
            for line in iter(process.stdout.readline, ""):
                if line:
                    with self._lock:
                        self._output_buffers[pid].append(line)
        except Exception:
            pass
        finally:
            process.wait()
            with self._lock:
                if pid in self._processes:
                    self._processes[pid].status = "completed"
                    self._processes[pid].return_code = process.returncode

    def get_output(self, pid: int, from_line: int = 0) -> tuple[list[str], bool]:
        with self._lock:
            if pid not in self._processes:
                return [], True

            output = self._output_buffers.get(pid, [])
            new_output = output[from_line:]
            is_complete = self._processes[pid].status == "completed"
            return new_output, is_complete

    def stop_process(self, pid: int) -> bool:
        with self._lock:
            if pid not in self._processes:
                return False
            process_info = self._processes[pid]

        try:
            process_info.process.terminate()
            process_info.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process_info.process.kill()
        except Exception:
            pass

        with self._lock:
            if pid in self._processes:
                self._processes[pid].status = "stopped"
        return True

    def get_status(self, pid: int) -> Optional[dict]:
        with self._lock:
            if pid not in self._processes:
                return None
            info = self._processes[pid]
            return {
                "pid": info.pid,
                "status": info.status,
                "return_code": info.return_code,
                "output_lines": len(self._output_buffers.get(pid, [])),
            }

    def cleanup_process(self, pid: int) -> None:
        with self._lock:
            if pid in self._processes:
                try:
                    os.unlink(self._processes[pid].output_file)
                except OSError:
                    pass
                del self._processes[pid]
            if pid in self._output_buffers:
                del self._output_buffers[pid]


process_manager = ProcessManager()

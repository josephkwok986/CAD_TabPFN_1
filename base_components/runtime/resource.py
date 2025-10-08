"""System resource monitoring helpers."""
from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Optional

try:  # pragma: no cover - optional dependency
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore


@dataclass(frozen=True)
class ResourceSnapshot:
    timestamp: float
    cpu_utilisation: Optional[float] = None
    gpu_utilisation: Optional[float] = None
    memory_utilisation: Optional[float] = None


class ResourceMonitor:
    """Interface for resource monitors."""

    def snapshot(self) -> Optional[ResourceSnapshot]:  # pragma: no cover - interface
        raise NotImplementedError


class NullResourceMonitor(ResourceMonitor):
    def snapshot(self) -> Optional[ResourceSnapshot]:
        return ResourceSnapshot(timestamp=time.monotonic())


class PsUtilResourceMonitor(ResourceMonitor):
    """Collect CPU/MEM stats via psutil; GPU via nvidia-smi when available."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._supports_gpu = shutil.which("nvidia-smi") is not None
        if psutil is not None:
            # prime cpu_percent so first call is meaningful
            psutil.cpu_percent(interval=None)

    def snapshot(self) -> Optional[ResourceSnapshot]:
        timestamp = time.monotonic()
        cpu = None
        mem = None
        gpu = None
        if psutil is not None:
            with self._lock:
                cpu = psutil.cpu_percent(interval=None) / 100.0
                mem = psutil.virtual_memory().percent / 100.0
        if self._supports_gpu:
            gpu = self._query_gpu_utilisation()
        return ResourceSnapshot(timestamp=timestamp, cpu_utilisation=_clamp(cpu), gpu_utilisation=_clamp(gpu), memory_utilisation=_clamp(mem))

    @staticmethod
    def _query_gpu_utilisation() -> Optional[float]:
        try:
            proc = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
                timeout=0.5,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return None
        if proc.returncode != 0:
            return None
        values = []
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                values.append(float(line))
            except ValueError:
                continue
        if not values:
            return None
        return max(values) / 100.0


def _clamp(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return max(0.0, min(1.0, float(value)))


def get_default_resource_monitor() -> ResourceMonitor:
    if psutil is None:
        return NullResourceMonitor()
    return PsUtilResourceMonitor()

"""GPU resource discovery and coordination utilities."""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Sequence

from .logger import StructuredLogger


logger = StructuredLogger.get_logger("cad.gpu.resources")


@dataclass(frozen=True)
class GPUDevice:
    index: int
    uuid: Optional[str] = None
    name: Optional[str] = None

    def visible_id(self) -> str:
        return str(self.index)


class GPUResourceManager:
    """Lightweight GPU device discovery and assignment helper."""

    def __init__(self, devices: Sequence[GPUDevice]) -> None:
        self._devices: List[GPUDevice] = list(devices)
        logger.info("gpu.manager.initialised", count=len(self._devices))

    @classmethod
    def discover(cls, preferred: Optional[Sequence[int]] = None) -> "GPUResourceManager":
        devices: List[GPUDevice] = []
        env_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if env_visible:
            indices = [idx.strip() for idx in env_visible.split(",") if idx.strip()]
            for pos, token in enumerate(indices):
                try:
                    index = int(token)
                except ValueError:
                    # token might already be UUID or masked value; fall back to positional index
                    devices.append(GPUDevice(index=pos, uuid=token, name=None))
                else:
                    devices.append(GPUDevice(index=index))
        else:
            try:
                import torch  # type: ignore

                count = int(torch.cuda.device_count())
                for idx in range(count):
                    name = torch.cuda.get_device_name(idx)
                    devices.append(GPUDevice(index=idx, name=name))
            except Exception:
                devices.extend(cls._discover_with_nvidia_smi())
        if preferred:
            devices = cls._filter_preferred(devices, preferred)
        return cls(devices)

    @staticmethod
    def _discover_with_nvidia_smi() -> List[GPUDevice]:
        devices: List[GPUDevice] = []
        try:
            proc = subprocess.run(
                ["nvidia-smi", "-L"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            output = proc.stdout.strip().splitlines()
            for line in output:
                if not line:
                    continue
                if not line.lower().startswith("gpu"):
                    continue
                try:
                    prefix, rest = line.split(":", 1)
                    idx_part = prefix.split()[1]
                    index = int(idx_part)
                except Exception:
                    continue
                name = rest.strip().split("(", 1)[0].strip()
                devices.append(GPUDevice(index=index, name=name))
        except FileNotFoundError:
            pass
        return devices

    def assign(self, worker_index: int) -> Optional[GPUDevice]:
        if not self._devices:
            return None
        return self._devices[worker_index % len(self._devices)]

    def available(self) -> int:
        return len(self._devices)

    @staticmethod
    def _filter_preferred(devices: Sequence[GPUDevice], preferred: Sequence[int]) -> List[GPUDevice]:
        if not devices:
            return list(devices)
        preferred_set = []
        for value in preferred:
            try:
                idx = int(value)
            except (TypeError, ValueError):
                logger.warning("gpu.manager.invalid_preferred", value=value)
                continue
            match = next((dev for dev in devices if dev.index == idx), None)
            if match is not None:
                preferred_set.append(match)
            else:
                logger.warning("gpu.manager.preferred_missing", index=idx)
        if preferred_set:
            logger.info("gpu.manager.preferred_applied", devices=[dev.index for dev in preferred_set])
            return preferred_set
        logger.warning("gpu.manager.preferred_unavailable", requested=list(preferred))
        return list(devices)


__all__ = ["GPUDevice", "GPUResourceManager"]

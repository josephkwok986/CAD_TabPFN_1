"""任务池封装（基于 FileQueue 实现持久化至少一次队列）。"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .file_queue import FileQueueConfig, LeaseRecord, MultiQueueGroup
from .logger import StructuredLogger
from .task_partitioner import TaskRecord
from .task_system_config import get_pool_value


logger = StructuredLogger.get_logger("cad.task_pool")


@dataclass
class LeasedTask:
    task: TaskRecord
    lease_id: str
    lease_deadline: float
    attempt: int
    metadata: Mapping[str, Union[str, int, float, None]] = field(default_factory=dict)


class TaskPool:
    """薄封装：所有操作委派给本地文件队列。

    新增退出机制：
    - 生产者调用 `seal()` 声明“无新项”。
    - 消费者对每个任务执行 `ack()`。
    - `is_drained()` 为真（所有入队项均已确认，且无可见/租约中任务）后可安全退出。
    - 可使用 `wait_until_drained()` 阻塞等待耗尽。
    """

    def __init__(
        self,
        queue_root: Optional[Union[str, Path]] = None,
        queue_name: Optional[str] = None,
        *,
        default_ttl: Optional[float] = None,
        max_attempts: Optional[int] = None,
        retry_delay_seconds: Optional[float] = None,
    ) -> None:
        backend_cfg = get_pool_value("backend", dict, default={})
        root_cfg = backend_cfg.get("path")
        name_cfg = backend_cfg.get("queue_name", "default")
        attempts_cfg = backend_cfg.get("max_attempts")
        retry_cfg = backend_cfg.get("retry_delay_seconds")

        root = Path(queue_root or root_cfg or "./task_queue").resolve()
        name = queue_name or name_cfg
        root.mkdir(parents=True, exist_ok=True)

        cfg = FileQueueConfig(
            max_segment_bytes=backend_cfg.get("max_segment_bytes", FileQueueConfig.max_segment_bytes),
            flush_each_put=bool(backend_cfg.get("flush_each_put", False)),
            lease_log_batch=backend_cfg.get("lease_log_batch", FileQueueConfig.lease_log_batch),
            lease_sample_size=backend_cfg.get("lease_sample_size", FileQueueConfig.lease_sample_size),
            max_attempts=max_attempts or attempts_cfg,
            retry_delay_seconds=retry_delay_seconds if retry_delay_seconds is not None else retry_cfg or 0.0,
        )

        ttl_raw = get_pool_value("default_ttl", Any, default=None)
        self._default_ttl = float(default_ttl if default_ttl is not None else ttl_raw) if (default_ttl is not None or ttl_raw is not None) else None

        self._group = MultiQueueGroup(root, cfg)
        self._queue_name = name
        self._queue = self._group.ensure_queue(self._queue_name)
        self._inflight: Dict[str, LeaseRecord] = {}
        # 新增：生产者封口标志（进程内）
        self._sealed: bool = False

        logger.info(
            "task_pool.initialised",
            backend="FileQueue",
            root=str(root),
            queue=name,
            default_ttl=self._default_ttl,
            max_attempts=cfg.max_attempts,
        )

    # ------------------------------------------------------------------ #
    # 生产者接口
    # ------------------------------------------------------------------ #

    def put(
        self,
        tasks: Union[TaskRecord, Sequence[TaskRecord]],
    ) -> List[int]:
        if self._sealed:
            raise RuntimeError("TaskPool is sealed. No new tasks are accepted.")
        if isinstance(tasks, TaskRecord):
            iterable = [tasks]
        else:
            iterable = list(tasks)
        payloads = [
            json.dumps(task.to_dict(), ensure_ascii=False).encode("utf-8")
            for task in iterable
        ]
        return self._queue.put_many(payloads)

    def seal(self) -> None:
        """生产者声明：后续不再入队新任务。"""
        if not self._sealed:
            self._sealed = True
            logger.info("task_pool.sealed", queue=self._queue_name)

    # ------------------------------------------------------------------ #
    # 消费者接口
    # ------------------------------------------------------------------ #

    def lease(
        self,
        max_n: int,
        lease_ttl: float,
        *,
        filters: Optional[Mapping[str, Union[str, int, float]]] = None,
    ) -> List[LeasedTask]:
        # filters 暂未实现；接口保留
        _ = filters
        ttl = lease_ttl if lease_ttl and lease_ttl > 0 else self._default_ttl
        ttl = max(ttl or 0.1, 0.1)
        leases = self._queue.lease(max_n, ttl)
        leased_tasks: List[LeasedTask] = []
        for lease in leases:
            try:
                task = self._decode_task(lease.payload)
            except Exception as exc:
                logger.error(
                    "task_pool.decode_failed",
                    rec_id=lease.rec_id,
                    error=str(exc),
                )
                self._queue.ack([lease.rec_id])
                continue
            lease_id = f"{self._queue_name}:{lease.rec_id}"
            leased_tasks.append(
                LeasedTask(
                    task=task,
                    lease_id=lease_id,
                    lease_deadline=lease.expire_at,
                    attempt=lease.attempt,
                )
            )
            self._inflight[task.task_id] = lease
        return leased_tasks

    def ack(self, task_id: str) -> bool:
        lease = self._inflight.pop(task_id, None)
        if lease is None:
            return False
        self._queue.ack([lease.rec_id])
        return True

    def nack(self, task_id: str, *, requeue: bool = True, delay: Optional[float] = None) -> bool:
        lease = self._inflight.pop(task_id, None)
        if lease is None:
            return False
        if requeue:
            self._queue.nack([lease.rec_id], delay=delay)
        else:
            # 明确放弃：按需求记为已确认完成，避免“卡住耗尽”
            self._queue.ack([lease.rec_id])
        return True

    def extend(self, task_id: str, ttl_seconds: float) -> bool:
        lease = self._inflight.get(task_id)
        if lease is None:
            return False
        self._queue.extend([lease.rec_id], ttl_seconds)
        return True

    def heartbeat(self, task_id: str) -> bool:
        return task_id in self._inflight

    def mark_dead(self, task_id: str, reason: str) -> bool:
        lease = self._inflight.pop(task_id, None)
        if lease is None:
            return False
        # 这里选择 ack 掉记录，避免无限重试与“耗尽等待”失活
        self._queue.ack([lease.rec_id])
        logger.error("task.dead", task_id=task_id, reason=reason)
        return True

    # ------------------------------------------------------------------ #
    # 监控与退出机制
    # ------------------------------------------------------------------ #

    def stats(self) -> Mapping[str, int]:
        return self._queue.stats()

    def is_sealed(self) -> bool:
        return self._sealed

    def is_drained(self) -> bool:
        """所有入队任务均已ACK，且无可见/租约中任务。"""
        s = self.stats()  # visible/leased/acked/total
        return (s.get("total", 0) == s.get("acked", -1)) and s.get("visible", 0) == 0 and s.get("leased", 0) == 0

    def wait_until_drained(self, *, poll_interval: float = 0.5, timeout: Optional[float] = None) -> bool:
        """阻塞等待队列耗尽。返回 True 表示耗尽，False 表示超时。"""
        start = time.time()
        while True:
            if self.is_drained():
                return True
            if timeout is not None and time.time() - start >= timeout:
                return False
            time.sleep(max(0.05, poll_interval))

    def drain(self, predicate) -> List[TaskRecord]:
        # 文件队列不支持选择性清空，返回空列表保持兼容
        _ = predicate
        return []

    # ------------------------------------------------------------------ #
    # 内部辅助
    # ------------------------------------------------------------------ #

    @staticmethod
    def _decode_task(payload: bytes) -> TaskRecord:
        data = json.loads(payload.decode("utf-8"))
        data["payload_ref"] = tuple(data.get("payload_ref", []))
        data["group_keys"] = tuple(data.get("group_keys", []))
        data.setdefault("extras", {})
        return TaskRecord(**data)

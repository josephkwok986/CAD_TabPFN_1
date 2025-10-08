"""Task pool implementations.

Provides TaskPool with configurable backends and concurrency safety.
"""
from __future__ import annotations

import heapq
import json
import os
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from logger import StructuredLogger
from task_partitioner import TaskRecord
from task_system_config import get_pool_value


logger = StructuredLogger.get_logger("cad.task_pool")


@dataclass
class LeasedTask:
    task: TaskRecord
    lease_id: str
    lease_deadline: float
    attempt: int
    metadata: Mapping[str, Union[str, int, float, None]] = field(default_factory=dict)


@dataclass
class _TaskEntry:
    task: TaskRecord
    priority: int
    visible_at: float
    expires_at: Optional[float]
    lease_deadline: Optional[float]
    lease_id: Optional[str]
    attempt: int
    idempotency_key: Optional[str]
    metadata: Dict[str, Union[str, int, float, None]]
    last_exception: Optional[str] = None
    last_reason: Optional[str] = None
    retry_count: int = 0

    def is_visible(self, now: float) -> bool:
        if self.expires_at is not None and now >= self.expires_at:
            return False
        return self.visible_at <= now and self.lease_deadline is None

    def lease_expired(self, now: float) -> bool:
        return self.lease_deadline is not None and now >= self.lease_deadline


class TaskPoolBackend:
    def put(self, entries: Sequence[_TaskEntry]) -> None:
        raise NotImplementedError

    def lease(self, max_n: int, lease_ttl: float, filters: Optional[Mapping[str, Union[str, int, float]]] = None) -> List[_TaskEntry]:
        raise NotImplementedError

    def ack(self, task_id: str) -> bool:
        raise NotImplementedError

    def nack(self, task_id: str, *, requeue: bool, delay: Optional[float]) -> bool:
        raise NotImplementedError

    def heartbeat(self, task_id: str) -> bool:
        raise NotImplementedError

    def mark_dead(self, task_id: str, reason: str) -> bool:
        raise NotImplementedError

    def stats(self) -> Mapping[str, Union[int, float]]:
        raise NotImplementedError

    def drain(self, predicate: Callable[[TaskRecord], bool]) -> List[TaskRecord]:
        raise NotImplementedError


class InMemoryBackend(TaskPoolBackend):
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._ready: List[Tuple[int, int, _TaskEntry]] = []
        self._delayed: List[Tuple[float, int, _TaskEntry]] = []
        self._leased: Dict[str, _TaskEntry] = {}
        self._dead: Dict[str, _TaskEntry] = {}
        self._sequence = 0

    def _log_queue_state(self, event: str) -> None:
        logger.debug(
            event,
            ready=len(self._ready),
            delayed=len(self._delayed),
            leased=len(self._leased),
            dead=len(self._dead),
        )

    def _match_filters(self, entry: _TaskEntry, filters: Optional[Mapping[str, Union[str, int, float]]]) -> bool:
        if not filters:
            return True
        for key, expected in filters.items():
            actual = entry.metadata.get(key) or entry.task.extras.get(key)
            if actual != expected:
                return False
        return True

    def _requeue_expired_locked(self, now: float) -> None:
        expired = [task_id for task_id, entry in self._leased.items() if entry.lease_expired(now)]
        if expired:
            for idx, task_id in enumerate(expired):
                entry = self._leased.pop(task_id)
                entry.lease_deadline = None
                entry.lease_id = None
                entry.retry_count += 1
                entry.visible_at = now
                self._push_ready(entry)
                if idx in (0, len(expired) // 2, len(expired) - 1):
                    logger.warning(
                        "lease.expired.requeued",
                        task_id=task_id,
                        attempt=entry.attempt,
                        retries=entry.retry_count,
                    )

    def _push_ready(self, entry: _TaskEntry) -> None:
        self._sequence += 1
        heapq.heappush(self._ready, (entry.priority, self._sequence, entry))

    def _push_delayed(self, entry: _TaskEntry) -> None:
        self._sequence += 1
        heapq.heappush(self._delayed, (entry.visible_at, self._sequence, entry))

    def _promote_ready(self, now: float) -> None:
        while self._delayed and self._delayed[0][0] <= now:
            _, _, entry = heapq.heappop(self._delayed)
            self._push_ready(entry)

    def put(self, entries: Sequence[_TaskEntry]) -> None:
        with self._lock:
            now = time.time()
            for idx, entry in enumerate(entries):
                if entry.expires_at is not None and entry.expires_at <= now:
                    continue
                if entry.visible_at <= now:
                    self._push_ready(entry)
                else:
                    self._push_delayed(entry)
                if idx in (0, len(entries) // 2, len(entries) - 1):
                    logger.info(
                        "task.put",
                        task_id=entry.task.task_id,
                        priority=entry.priority,
                        visible_at=entry.visible_at,
                    )
            self._log_queue_state("queue.updated")

    def lease(self, max_n: int, lease_ttl: float, filters: Optional[Mapping[str, Union[str, int, float]]] = None) -> List[_TaskEntry]:
        leased: List[_TaskEntry] = []
        with self._lock:
            now = time.time()
            self._requeue_expired_locked(now)
            self._promote_ready(now)
            if not self._ready:
                return leased
            count = min(max_n, len(self._ready))
            skipped: List[Tuple[int, int, _TaskEntry]] = []
            for _ in range(count):
                priority, seq, entry = heapq.heappop(self._ready)
                if entry.expires_at is not None and entry.expires_at <= now:
                    continue
                if not self._match_filters(entry, filters):
                    skipped.append((priority, seq, entry))
                    continue
                entry.lease_id = uuid.uuid4().hex
                entry.lease_deadline = now + lease_ttl
                entry.attempt += 1
                entry.metadata.setdefault("last_lease_ttl", lease_ttl)
                self._leased[entry.task.task_id] = entry
                leased.append(entry)
            for item in skipped:
                heapq.heappush(self._ready, item)
            for idx, entry in enumerate(leased):
                if idx in (0, len(leased) // 2, len(leased) - 1):
                    logger.debug(
                        "task.leased",
                        task_id=entry.task.task_id,
                        lease_deadline=entry.lease_deadline,
                        attempt=entry.attempt,
                    )
        return leased

    def ack(self, task_id: str) -> bool:
        with self._lock:
            entry = self._leased.pop(task_id, None)
            if entry is None:
                return False
            if random.random() <= 0.01:
                logger.info("task.ack", task_id=task_id, attempt=entry.attempt, stage="Execution")
            return True

    def nack(self, task_id: str, *, requeue: bool, delay: Optional[float]) -> bool:
        with self._lock:
            entry = self._leased.pop(task_id, None)
            if entry is None:
                return False
            entry.last_exception = entry.last_exception
            entry.last_reason = "nack"
            now = time.time()
            if requeue:
                entry.visible_at = now + (delay or 0.0)
                entry.lease_deadline = None
                entry.lease_id = None
                entry.retry_count += 1
                if entry.visible_at <= now:
                    self._push_ready(entry)
                else:
                    self._push_delayed(entry)
                logger.warning(
                    "task.nack.requeued",
                    task_id=task_id,
                    delay=delay,
                    retries=entry.retry_count,
                )
            else:
                self._dead[task_id] = entry
                logger.error("task.nack.dead", task_id=task_id)
            return True

    def heartbeat(self, task_id: str) -> bool:
        with self._lock:
            entry = self._leased.get(task_id)
            if entry is None or entry.lease_deadline is None:
                return False
            ttl = float(entry.metadata.get("last_lease_ttl", 0.0))
            if ttl <= 0:
                return False
            entry.lease_deadline = time.time() + ttl
            logger.debug("task.heartbeat", task_id=task_id, ttl=ttl)
            return True

    def mark_dead(self, task_id: str, reason: str) -> bool:
        with self._lock:
            entry = self._leased.pop(task_id, None)
            if entry is None:
                entry = self._pop_from_heaps(task_id)
            if entry is None:
                return False
            entry.last_reason = reason
            self._dead[task_id] = entry
            logger.error("task.dead", task_id=task_id, reason=reason)
            return True

    def stats(self) -> Mapping[str, Union[int, float]]:
        with self._lock:
            now = time.time()
            self._promote_ready(now)
            visible = len(self._ready)
            leased = len(self._leased)
            dead = len(self._dead)
            return {
                "visible": visible,
                "leased": leased,
                "dead": dead,
                "total": visible + leased + dead,
            }

    def drain(self, predicate: Callable[[TaskRecord], bool]) -> List[TaskRecord]:
        drained: List[TaskRecord] = []
        with self._lock:
            now = time.time()
            all_entries = self._collect_entries()
            self._ready.clear()
            self._delayed.clear()
            for entry in all_entries:
                if predicate(entry.task):
                    drained.append(entry.task)
                else:
                    if entry.visible_at <= now:
                        self._push_ready(entry)
                    else:
                        self._push_delayed(entry)
            logger.info("task.drain", drained=len(drained))
        return drained

    def _collect_entries(self) -> List[_TaskEntry]:
        ready_entries = [entry for _, _, entry in self._ready]
        delayed_entries = [entry for _, _, entry in self._delayed]
        return ready_entries + delayed_entries

    def snapshot_entries(self) -> List[_TaskEntry]:
        return self._collect_entries() + list(self._leased.values())

    def _pop_from_heaps(self, task_id: str) -> Optional[_TaskEntry]:
        found: Optional[_TaskEntry] = None
        rebuilt_ready: List[Tuple[int, int, _TaskEntry]] = []
        while self._ready:
            priority, seq, entry = heapq.heappop(self._ready)
            if entry.task.task_id == task_id and found is None:
                found = entry
                continue
            rebuilt_ready.append((priority, seq, entry))
        for item in rebuilt_ready:
            heapq.heappush(self._ready, item)
        rebuilt_delayed: List[Tuple[float, int, _TaskEntry]] = []
        while self._delayed:
            visible_at, seq, entry = heapq.heappop(self._delayed)
            if entry.task.task_id == task_id and found is None:
                found = entry
                continue
            rebuilt_delayed.append((visible_at, seq, entry))
        for item in rebuilt_delayed:
            heapq.heappush(self._delayed, item)
        return found


class FileBackend(TaskPoolBackend):
    def __init__(self, path: Union[str, os.PathLike[str]]) -> None:
        self._path = Path(path)
        self._lock = threading.RLock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._queue: InMemoryBackend = InMemoryBackend()
        if self._path.exists():
            self._load()

    def _load(self) -> None:
        with self._path.open("r", encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        entries: List[_TaskEntry] = []
        for row in rows:
            task_data = dict(row["task"])
            task_data["payload_ref"] = tuple(task_data.get("payload_ref", []))
            task_data["group_keys"] = tuple(task_data.get("group_keys", []))
            record = TaskRecord(**task_data)
            entry = _TaskEntry(
                task=record,
                priority=row["priority"],
                visible_at=row["visible_at"],
                expires_at=row.get("expires_at"),
                lease_deadline=None,
                lease_id=None,
                attempt=row.get("attempt", 0),
                idempotency_key=row.get("idempotency_key"),
                metadata=row.get("metadata", {}),
                last_exception=row.get("last_exception"),
                last_reason=row.get("last_reason"),
                retry_count=row.get("retry_count", 0),
            )
            entries.append(entry)
        self._queue.put(entries)

    def _persist(self) -> None:
        entries: List[Dict[str, Union[str, int, float, None, Dict[str, Union[str, int, float, None]]]]] = []
        for entry in self._queue.snapshot_entries():
            entries.append(
                {
                    "task": entry.task.to_dict(),
                    "priority": entry.priority,
                    "visible_at": entry.visible_at,
                    "expires_at": entry.expires_at,
                    "attempt": entry.attempt,
                    "idempotency_key": entry.idempotency_key,
                    "metadata": dict(entry.metadata),
                    "last_exception": entry.last_exception,
                    "last_reason": entry.last_reason,
                    "retry_count": entry.retry_count,
                }
            )
        tmp_path = self._path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as fh:
            for row in entries:
                fh.write(json.dumps(row) + "\n")
        os.replace(tmp_path, self._path)

    def put(self, entries: Sequence[_TaskEntry]) -> None:
        with self._lock:
            self._queue.put(entries)
            self._persist()

    def lease(self, max_n: int, lease_ttl: float, filters: Optional[Mapping[str, Union[str, int, float]]] = None) -> List[_TaskEntry]:
        with self._lock:
            leased = self._queue.lease(max_n, lease_ttl, filters)
            self._persist()
            return leased

    def ack(self, task_id: str) -> bool:
        with self._lock:
            ok = self._queue.ack(task_id)
            self._persist()
            return ok

    def nack(self, task_id: str, *, requeue: bool, delay: Optional[float]) -> bool:
        with self._lock:
            ok = self._queue.nack(task_id, requeue=requeue, delay=delay)
            self._persist()
            return ok

    def heartbeat(self, task_id: str) -> bool:
        with self._lock:
            ok = self._queue.heartbeat(task_id)
            self._persist()
            return ok

    def mark_dead(self, task_id: str, reason: str) -> bool:
        with self._lock:
            ok = self._queue.mark_dead(task_id, reason)
            self._persist()
            return ok

    def stats(self) -> Mapping[str, Union[int, float]]:
        return self._queue.stats()

    def drain(self, predicate: Callable[[TaskRecord], bool]) -> List[TaskRecord]:
        with self._lock:
            drained = self._queue.drain(predicate)
            self._persist()
            return drained


def _backend_from_config() -> TaskPoolBackend:
    settings = get_pool_value("backend", dict, default={})
    backend_type = str(settings.get("type", "memory")).lower()
    if backend_type == "memory":
        return InMemoryBackend()
    if backend_type == "file":
        path = settings.get("path")
        if not path:
            raise RuntimeError("File backend requires 'path' in configuration")
        return FileBackend(str(path))
    raise ValueError(f"Unsupported task pool backend: {backend_type}")


class TaskPool:
    def __init__(
        self,
        backend: Optional[TaskPoolBackend] = None,
        *,
        default_ttl: Optional[float] = None,
    ) -> None:
        self._backend = backend or _backend_from_config()
        ttl_raw = get_pool_value("default_ttl", Any, default=None)
        config_ttl = float(ttl_raw) if ttl_raw is not None else None
        self._default_ttl = default_ttl if default_ttl is not None else config_ttl
        self._idempotency_index: Dict[str, str] = {}
        self._lock = threading.RLock()
        logger.info(
            "task_pool.initialised",
            backend=self._backend.__class__.__name__,
            default_ttl=self._default_ttl,
        )

    def put(
        self,
        tasks: Union[TaskRecord, Sequence[TaskRecord]],
        *,
        ttl: Optional[float] = None,
        priority: Optional[int] = None,
        idempotency_key: Optional[str] = None,
        metadata: Optional[Mapping[str, Union[str, int, float, None]]] = None,
    ) -> None:
        if isinstance(tasks, TaskRecord):
            iterable = [tasks]
        else:
            iterable = list(tasks)
        ttl = ttl if ttl is not None else self._default_ttl
        now = time.time()
        entries: List[_TaskEntry] = []
        with self._lock:
            for idx, task in enumerate(iterable):
                key = idempotency_key or str(task.extras.get("idempotency_key", "")) or task.task_id
                if key in self._idempotency_index:
                    logger.debug("task.put.skip_duplicate", task_id=task.task_id, key=key)
                    continue
                entry = _TaskEntry(
                    task=task,
                    priority=priority if priority is not None else task.priority or 0,
                    visible_at=now,
                    expires_at=(now + ttl) if ttl is not None else None,
                    lease_deadline=None,
                    lease_id=None,
                    attempt=0,
                    idempotency_key=key,
                    metadata=dict(metadata or {}),
                )
                self._idempotency_index[key] = task.task_id
                entries.append(entry)
                if idx in (0, len(iterable) // 2, len(iterable) - 1):
                    logger.debug("task.put.entry", task_id=task.task_id, key=key)
        if entries:
            self._backend.put(entries)

    def lease(
        self,
        max_n: int,
        lease_ttl: float,
        *,
        filters: Optional[Mapping[str, Union[str, int, float]]] = None,
    ) -> List[LeasedTask]:
        raw = self._backend.lease(max_n, lease_ttl, filters)
        leased = [
            LeasedTask(
                task=entry.task,
                lease_id=entry.lease_id or "",
                lease_deadline=entry.lease_deadline or 0.0,
                attempt=entry.attempt,
                metadata=entry.metadata,
            )
            for entry in raw
        ]
        return leased

    def ack(self, task_id: str) -> bool:
        if self._backend.ack(task_id):
            with self._lock:
                for key, value in list(self._idempotency_index.items()):
                    if value == task_id:
                        self._idempotency_index.pop(key, None)
            return True
        return False

    def nack(self, task_id: str, *, requeue: bool = True, delay: Optional[float] = None) -> bool:
        return self._backend.nack(task_id, requeue=requeue, delay=delay)

    def heartbeat(self, task_id: str) -> bool:
        return self._backend.heartbeat(task_id)

    def mark_dead(self, task_id: str, reason: str) -> bool:
        return self._backend.mark_dead(task_id, reason)

    def stats(self) -> Mapping[str, Union[int, float]]:
        return self._backend.stats()

    def drain(self, predicate: Callable[[TaskRecord], bool]) -> List[TaskRecord]:
        drained = self._backend.drain(predicate)
        with self._lock:
            for task in drained:
                for key, value in list(self._idempotency_index.items()):
                    if value == task.task_id:
                        self._idempotency_index.pop(key, None)
        return drained


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import unittest
    from task_partitioner import TaskPartitioner

    class TaskPoolTests(unittest.TestCase):
        def setUp(self) -> None:
            plan = TaskPartitioner.plan(
                {"job_id": "job-1"},
                [
                    {"payload_ref": "item-1", "weight": 1.0, "group_key": "A"},
                    {"payload_ref": "item-2", "weight": 1.0, "group_key": "A"},
                    {"payload_ref": "item-3", "weight": 1.0, "group_key": "B"},
                ],
                "fixed",
                {"max_items_per_task": 2},
            )
            self.tasks = plan.tasks
            self.task_count = len(self.tasks)
            self.pool = TaskPool()

        def test_put_and_lease(self) -> None:
            self.pool.put(self.tasks)
            leased = self.pool.lease(2, 5.0)
            self.assertEqual(len(leased), 2)
            stats = self.pool.stats()
            self.assertEqual(stats["leased"], 2)

        def test_ack_removes_task(self) -> None:
            self.pool.put(self.tasks)
            leased = self.pool.lease(1, 5.0)
            task_id = leased[0].task.task_id
            self.assertTrue(self.pool.ack(task_id))
            stats = self.pool.stats()
            self.assertEqual(stats["leased"], 0)

        def test_nack_requeues(self) -> None:
            self.pool.put(self.tasks)
            leased = self.pool.lease(1, 1.0)
            task_id = leased[0].task.task_id
            self.assertTrue(self.pool.nack(task_id, requeue=True, delay=0.0))
            leased_again = self.pool.lease(2, 1.0)
            self.assertIn(task_id, {entry.task.task_id for entry in leased_again})

        def test_lease_expiry(self) -> None:
            self.pool.put(self.tasks)
            leased = self.pool.lease(1, 0.1)
            task_id = leased[0].task.task_id
            time.sleep(0.2)
            leased_again = self.pool.lease(2, 0.1)
            task_ids = {lease.task.task_id for lease in leased_again}
            self.assertIn(task_id, task_ids)

        def test_mark_dead_moves_task(self) -> None:
            self.pool.put(self.tasks)
            leased = self.pool.lease(1, 1.0)
            task_id = leased[0].task.task_id
            self.assertTrue(self.pool.mark_dead(task_id, "test"))
            stats = self.pool.stats()
            self.assertEqual(stats["dead"], 1)

        def test_drain_removes_tasks(self) -> None:
            self.pool.put(self.tasks)
            drained = self.pool.drain(lambda task: task.payload_ref[0] == "item-1")
            self.assertTrue(drained)
            stats = self.pool.stats()
            self.assertEqual(stats["visible"], max(0, self.task_count - len(drained)))

    unittest.main()

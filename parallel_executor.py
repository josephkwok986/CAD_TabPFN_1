"""Parallel executor coordinating work from TaskPool."""
from __future__ import annotations

import concurrent.futures
import queue
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Mapping, Optional

from logger import StructuredLogger

from task_pool import LeasedTask, TaskPool
from task_system_config import get_executor_value


logger = StructuredLogger.get_logger("cad.parallel_executor")


def _policy_value(key: str, typ: type, *, default: Optional[object] = None, required: bool = False):
    if not required and default is None:
        raw = get_executor_value(key, Any, default=None)
        if raw is None:
            return None
        if isinstance(raw, typ):
            return raw
        if typ in (int, float, bool):
            return typ(raw)
        if typ is dict and isinstance(raw, Mapping):
            return raw
        raise TypeError(f"Configuration key task_system.executor.{key} has incompatible type: {type(raw)!r}")
    return get_executor_value(key, typ, default, required=required)


@dataclass
class ExecutorPolicy:
    max_concurrency: int = field(default_factory=lambda: _policy_value("max_concurrency", int, required=True))
    lease_batch_size: int = field(default_factory=lambda: _policy_value("lease_batch_size", int, required=True))
    lease_ttl: float = field(default_factory=lambda: _policy_value("lease_ttl", float, required=True))
    prefetch: int = field(default_factory=lambda: _policy_value("prefetch", int, required=True))
    idle_sleep: float = field(default_factory=lambda: _policy_value("idle_sleep", float, required=True))
    max_retries: int = field(default_factory=lambda: _policy_value("max_retries", int, required=True))
    backoff_base: float = field(default_factory=lambda: _policy_value("backoff_base", float, required=True))
    backoff_jitter: float = field(default_factory=lambda: _policy_value("backoff_jitter", float, required=True))
    task_timeout: Optional[float] = field(default_factory=lambda: _policy_value("task_timeout", float, default=None))
    rate_limit_per_sec: Optional[float] = field(default_factory=lambda: _policy_value("rate_limit_per_sec", float, default=None))
    rate_limit_burst: Optional[int] = field(default_factory=lambda: _policy_value("rate_limit_burst", int, default=None))
    failure_delay: Optional[float] = field(default_factory=lambda: _policy_value("failure_delay", float, default=None))
    filters: Optional[Mapping[str, object]] = field(default_factory=lambda: _policy_value("filters", dict, default=None))
    requeue_on_failure: bool = field(default_factory=lambda: _policy_value("requeue_on_failure", bool, required=True))


@dataclass
class ExecutorEvents:
    on_start: Callable[["ParallelExecutor"], None] = lambda executor: None
    on_lease: Callable[["ParallelExecutor", LeasedTask], None] = lambda executor, leased: None
    on_success: Callable[["ParallelExecutor", LeasedTask, float], None] = lambda executor, leased, latency: None
    on_retry: Callable[["ParallelExecutor", LeasedTask, int, Exception], None] = lambda executor, leased, attempt, exc: None
    on_dead: Callable[["ParallelExecutor", LeasedTask, Exception], None] = lambda executor, leased, exc: None
    on_stop: Callable[["ParallelExecutor"], None] = lambda executor: None


class TokenBucket:
    def __init__(self, rate: float, burst: Optional[int]) -> None:
        self._rate = rate
        self._capacity = float(burst if burst is not None else max(1.0, rate))
        self._tokens = self._capacity
        self._updated = time.time()
        self._lock = threading.Lock()

    def acquire(self, amount: float = 1.0) -> None:
        while True:
            with self._lock:
                now = time.time()
                elapsed = now - self._updated
                self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
                self._updated = now
                if self._tokens >= amount:
                    self._tokens -= amount
                    return
                deficit = (amount - self._tokens) / self._rate
            if deficit > 0:
                time.sleep(deficit)


class ParallelExecutor:
    def __init__(
        self,
        policy: ExecutorPolicy,
        events: Optional[ExecutorEvents] = None,
    ) -> None:
        if policy.max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        if policy.lease_batch_size <= 0:
            raise ValueError("lease_batch_size must be positive")
        if policy.rate_limit_per_sec is not None and policy.rate_limit_per_sec <= 0:
            raise ValueError("rate_limit_per_sec must be positive when provided")
        self._policy = policy
        self._events = events or ExecutorEvents()
        self._stop_event = threading.Event()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=policy.max_concurrency)
        self._inflight: List[concurrent.futures.Future] = []
        self._inflight_lock = threading.Lock()
        self._prefetch_queue: "queue.Queue[LeasedTask]" = queue.Queue(maxsize=policy.prefetch)
        self._rate_limiter = (
            TokenBucket(policy.rate_limit_per_sec, policy.rate_limit_burst)
            if policy.rate_limit_per_sec
            else None
        )

    @classmethod
    def run(
        cls,
        handler: Callable[[LeasedTask], None],
        pool: TaskPool,
        policy: ExecutorPolicy,
        events: Optional[ExecutorEvents] = None,
    ) -> "ParallelExecutor":
        executor = cls(policy, events)
        executor.start()
        try:
            executor._loop(handler, pool)
        finally:
            executor.graceful_shutdown()
        return executor

    def start(self) -> None:
        logger.info("executor.start", concurrency=self._policy.max_concurrency)
        self._events.on_start(self)

    def stop(self) -> None:
        logger.info("executor.stop_requested")
        self._stop_event.set()

    def graceful_shutdown(self, timeout: Optional[float] = None) -> None:
        logger.info("executor.graceful_shutdown.begin")
        self._stop_event.set()
        with self._inflight_lock:
            futures = list(self._inflight)
        for idx, future in enumerate(futures):
            if idx in (0, len(futures) // 2, len(futures) - 1):
                state = "done" if future.done() else "running" if future.running() else "pending"
                logger.debug("executor.wait_future", future_state=state)
            if timeout is not None:
                future.result(timeout=timeout)
            else:
                future.result()
        self._executor.shutdown(wait=True)
        self._events.on_stop(self)
        logger.info("executor.graceful_shutdown.end")

    def submit(self, leased: LeasedTask, handler: Callable[[LeasedTask], None], pool: TaskPool) -> None:
        if self._rate_limiter is not None:
            self._rate_limiter.acquire(1.0)
        future = self._executor.submit(self._execute_task, leased, handler, pool)
        with self._inflight_lock:
            self._inflight.append(future)
        future.add_done_callback(self._on_future_done)

    def _on_future_done(self, future: concurrent.futures.Future) -> None:
        with self._inflight_lock:
            if future in self._inflight:
                self._inflight.remove(future)

    def _loop(self, handler: Callable[[LeasedTask], None], pool: TaskPool) -> None:
        while not self._stop_event.is_set():
            self._fill_prefetch(pool)
            dispatched = 0
            while not self._prefetch_queue.empty():
                try:
                    leased = self._prefetch_queue.get_nowait()
                except queue.Empty:  # pragma: no cover - defensive
                    break
                self.submit(leased, handler, pool)
                dispatched += 1
            if dispatched:
                logger.debug("executor.dispatched", count=dispatched)
            with self._inflight_lock:
                active = len(self._inflight)
            if self._prefetch_queue.empty() and active == 0:
                idle_stats = pool.stats()
                if idle_stats.get("visible", 0) == 0:
                    logger.info("executor.idle_exit")
                    break
            time.sleep(self._policy.idle_sleep)

    def _fill_prefetch(self, pool: TaskPool) -> None:
        if self._prefetch_queue.full():
            return
        request = min(
            self._policy.lease_batch_size,
            self._policy.prefetch - self._prefetch_queue.qsize(),
        )
        if request <= 0:
            return
        leased = pool.lease(request, self._policy.lease_ttl, filters=self._policy.filters)
        for idx, item in enumerate(leased):
            if idx in (0, len(leased) // 2, len(leased) - 1):
                logger.debug(
                    "executor.leased",
                    task_id=item.task.task_id,
                    deadline=item.lease_deadline,
                    attempt=item.attempt,
                )
            self._events.on_lease(self, item)
            try:
                self._prefetch_queue.put_nowait(item)
            except queue.Full:  # pragma: no cover - defensive
                pool.nack(item.task.task_id, requeue=True, delay=self._policy.failure_delay)
                break
        if not leased:
            logger.debug("executor.no_lease")

    def _execute_task(self, leased: LeasedTask, handler: Callable[[LeasedTask], None], pool: TaskPool) -> None:
        attempt = 0
        while attempt <= self._policy.max_retries:
            start = time.time()
            try:
                result = self._invoke_with_timeout(handler, leased)
                latency = time.time() - start
                pool.ack(leased.task.task_id)
                logger.info(
                    "executor.task_success",
                    task_id=leased.task.task_id,
                    attempt=attempt,
                    latency=latency,
                )
                self._events.on_success(self, leased, latency)
                return result
            except Exception as exc:  # pragma: no cover - high level catch
                latency = time.time() - start
                attempt += 1
                if attempt <= self._policy.max_retries:
                    backoff = self._policy.backoff_base * (2 ** (attempt - 1))
                    jitter = random.uniform(0, self._policy.backoff_jitter)
                    delay = backoff + jitter
                    logger.warning(
                        "executor.task_retry",
                        task_id=leased.task.task_id,
                        attempt=attempt,
                        delay=delay,
                        error=str(exc),
                    )
                    pool.heartbeat(leased.task.task_id)
                    self._events.on_retry(self, leased, attempt, exc)
                    time.sleep(delay)
                    continue
                logger.error(
                    "executor.task_failed",
                    task_id=leased.task.task_id,
                    error=str(exc),
                    latency=latency,
                )
                self._events.on_dead(self, leased, exc)
                pool.nack(
                    leased.task.task_id,
                    requeue=self._policy.requeue_on_failure,
                    delay=self._policy.failure_delay,
                )
                return

    def _invoke_with_timeout(self, handler: Callable[[LeasedTask], None], leased: LeasedTask):
        if self._policy.task_timeout is None:
            return handler(leased)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as tmp:
            future = tmp.submit(handler, leased)
            return future.result(timeout=self._policy.task_timeout)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import unittest

    from task_partitioner import TaskPartitioner

    class ParallelExecutorTests(unittest.TestCase):
        def setUp(self) -> None:
            plan = TaskPartitioner.plan(
                {"job_id": "job-ex"},
                [
                    {"payload_ref": "item-1", "weight": 1.0, "group_key": "A"},
                    {"payload_ref": "item-2", "weight": 1.0, "group_key": "A"},
                    {"payload_ref": "item-3", "weight": 1.0, "group_key": "B"},
                ],
                "fixed",
                {"max_items_per_task": 2},
            )
            self.pool = TaskPool()
            self.pool.put(plan.tasks)
            self.handled: List[str] = []
            self.expected_tasks = len(plan.tasks)

        def handler(self, leased: LeasedTask) -> None:
            time.sleep(0.05)
            self.handled.append(leased.task.task_id)

        def test_executor_processes_all_tasks(self) -> None:
            policy = ExecutorPolicy(max_concurrency=2, lease_batch_size=2, lease_ttl=5.0, prefetch=4)
            ParallelExecutor.run(self.handler, self.pool, policy)
            self.assertGreaterEqual(len(self.handled), self.expected_tasks)

        def test_retry_flow(self) -> None:
            self.pool.put(TaskPartitioner.plan(
                {"job_id": "job-retry"},
                [{"payload_ref": "item-4", "weight": 1.0, "group_key": "B"}],
                "fixed",
                {"max_items_per_task": 1},
            ).tasks)

            calls: List[int] = []

            def flaky(task: LeasedTask) -> None:
                calls.append(1)
                if len(calls) < 2:
                    raise RuntimeError("boom")

            policy = ExecutorPolicy(max_concurrency=1, lease_batch_size=1, lease_ttl=5.0, prefetch=1, max_retries=1)
            ParallelExecutor.run(flaky, self.pool, policy)
            self.assertGreaterEqual(len(calls), 2)

    unittest.main()

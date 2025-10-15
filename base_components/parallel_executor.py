"""Process-based parallel executor coordinating work from TaskPool."""
from __future__ import annotations

import concurrent.futures
import csv
import json
import os
import multiprocessing as mp
import queue
import random
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .gpu_resources import GPUDevice, GPUResourceManager
from .logger import StructuredLogger
from .task_pool import LeasedTask, TaskPool
from .task_system_config import get_executor_value


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


def _preferred_gpus_value() -> Optional[Tuple[int, ...]]:
    raw = _policy_value("preferred_gpus", list, default=None)
    if raw is None:
        return None
    try:
        return tuple(int(value) for value in raw)
    except Exception as exc:  # pragma: no cover - defensive
        raise TypeError("preferred_gpus must be a list of integers") from exc


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
    failure_delay: Optional[float] = field(default_factory=lambda: _policy_value("failure_delay", float, default=None))
    filters: Optional[Mapping[str, object]] = field(default_factory=lambda: _policy_value("filters", dict, default=None))
    requeue_on_failure: bool = field(default_factory=lambda: _policy_value("requeue_on_failure", bool, required=True))
    preferred_gpus: Optional[Tuple[int, ...]] = field(default_factory=_preferred_gpus_value)
    heartbeat_interval: Optional[float] = field(default_factory=lambda: _policy_value("heartbeat_interval", float, default=None))


@dataclass
class TaskResult:
    """Outcome of a task execution.

    Handlers must populate output_directory, output_filename, and is_final_output
    so workers can persist results immediately without keeping payloads in memory.
    """
    payload: Any = None
    processed: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    output_directory: Optional[str] = None
    output_filename: Optional[str] = None
    is_final_output: Optional[bool] = None
    written_path: Optional[str] = None


@dataclass
class ExecutorEvents:
    on_start: Callable[["ParallelExecutor"], None] = lambda executor: None
    on_lease: Callable[["ParallelExecutor", LeasedTask], None] = lambda executor, leased: None
    on_success: Callable[["ParallelExecutor", LeasedTask, float, TaskResult], None] = (
        lambda executor, leased, latency, result: None
    )
    on_retry: Callable[["ParallelExecutor", LeasedTask, int, Exception], None] = (
        lambda executor, leased, attempt, exc: None
    )
    on_dead: Callable[["ParallelExecutor", LeasedTask, Exception], None] = lambda executor, leased, exc: None
    on_stop: Callable[["ParallelExecutor"], None] = lambda executor: None


def _configure_worker_logging(min_level: str) -> None:
    try:
        from .config import Config  # type: ignore
    except Exception:
        Config = None  # type: ignore
    sinks = []
    level = "INFO"
    sampling = {}
    redaction = {}
    timezone = None
    namespace = "cad.logger"
    if Config is not None:
        try:
            cfg = Config.get_singleton()
        except Exception:
            cfg = None
        if cfg is not None:
            sinks = [dict(spec) for spec in cfg.get("logger.sinks", list, [])]
            level = cfg.get("logger.level", str, "INFO")
            sampling = cfg.get("logger.sampling", dict, {})
            redaction = cfg.get("logger.redact", dict, {})
            timezone = cfg.get("logger.timezone.render", str, None)
            namespace = cfg.get("logger.namespace", str, namespace)
    if not sinks:
        sinks = [{"type": "console", "stream": "stdout"}]
    console_found = False
    for sink in sinks:
        if sink.get("type") == "console":
            sink["min_level"] = min_level
            console_found = True
    if not console_found:
        sinks.append({"type": "console", "stream": "stdout", "min_level": min_level})
    StructuredLogger.configure(
        sinks=sinks,
        level=level,
        sampling=sampling,
        redaction=redaction,
        timezone=timezone,
        namespace=namespace,
    )


def _call_with_timeout(
    handler: Callable[[LeasedTask, Any], TaskResult],
    leased: LeasedTask,
    context: Any,
    timeout: Optional[float],
) -> TaskResult:
    if timeout is None:
        return handler(leased, context)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(handler, leased, context)
        return future.result(timeout=timeout)


def _heartbeat_pump(result_queue: "mp.Queue[Any]", task_id: str, interval: float, stop_event: threading.Event) -> None:
    while not stop_event.wait(interval):
        try:
            result_queue.put(("hb", task_id))
        except (EOFError, OSError):
            break


def _ensure_parent_dir(path: Path) -> None:
    """Create parent directories so we can flush results immediately."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _normalise_rows(payload: Any) -> List[Dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, Mapping):
        return [dict(payload)]
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        rows: List[Dict[str, Any]] = []
        for item in payload:
            if isinstance(item, Mapping):
                rows.append(dict(item))
            else:
                raise TypeError("Intermediate payload must be a sequence of mappings to form CSV rows.")
        return rows
    raise TypeError("Intermediate payload must provide tabular data (mapping or list of mappings).")


def _write_csv_output(destination: Path, payload: Any) -> None:
    if hasattr(payload, "to_csv") and callable(getattr(payload, "to_csv")):
        _ensure_parent_dir(destination)
        payload.to_csv(destination, index=False)
        return
    rows = _normalise_rows(payload)
    _ensure_parent_dir(destination)
    if not rows:
        destination.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = sorted({key for row in rows for key in row.keys()})
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_final_output(destination: Path, payload: Any) -> None:
    _ensure_parent_dir(destination)
    if isinstance(payload, (bytes, bytearray)):
        destination.write_bytes(payload)
    elif isinstance(payload, str):
        destination.write_text(payload, encoding="utf-8")
    elif hasattr(payload, "to_dict") and callable(getattr(payload, "to_dict")):
        data = payload.to_dict()
        destination.write_text(json.dumps(data, ensure_ascii=False, default=str), encoding="utf-8")
    else:
        destination.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")


def _persist_task_result(result: TaskResult, worker_logger: StructuredLogger, task_id: str) -> None:
    """Write task results to disk immediately to prevent worker memory growth."""
    if not result.output_directory or not result.output_filename or result.is_final_output is None:
        # Enforce contract so handlers always provide persistence hints.
        raise ValueError(
            "TaskResult must define output_directory, output_filename, and is_final_output for disk persistence."
        )
    root = Path(result.output_directory).expanduser()
    file_component = Path(result.output_filename)
    if file_component.is_absolute():
        raise ValueError("output_filename must be a relative path.")
    if result.is_final_output:
        destination = root / file_component
        _write_final_output(destination, result.payload)
    else:
        cache_dir = root / "cache"
        # Intermediate artefacts must be CSV files under the cache directory.
        destination = (cache_dir / file_component).with_suffix(".csv")
        _write_csv_output(destination, result.payload)
    resolved = destination.resolve()
    meta_source = result.metadata or {}
    meta = dict(meta_source)
    final_flag = bool(result.is_final_output)
    meta["written_path"] = str(resolved)
    meta["is_final_output"] = final_flag
    result.metadata = meta
    result.written_path = str(resolved)
    result.is_final_output = final_flag
    result.payload = str(resolved)
    worker_logger.debug(
        "executor.worker.persisted",
        task_id=task_id,
        path=str(resolved),
        final=result.is_final_output,
    )


def _worker_main(
    worker_id: int,
    task_queue: "mp.Queue[Optional[LeasedTask]]",
    result_queue: "mp.Queue[Any]",
    handler: Callable[[LeasedTask, Any], TaskResult],
    context: Any,
    policy: ExecutorPolicy,
    gpu_device: Optional[GPUDevice],
    namespace: str,
    console_min_level: str,
) -> None:
    _configure_worker_logging(console_min_level)
    worker_logger = StructuredLogger.get_logger(f"{namespace}.worker")
    worker_logger.bind(stage="Execution")
    if gpu_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device.visible_id()
        worker_logger.info(
            "executor.worker.gpu",
            worker=worker_id,
            gpu_index=gpu_device.index,
            gpu_name=gpu_device.name,
            gpu_uuid=gpu_device.uuid,
        )
    rng = random.Random(1337 + worker_id)
    while True:
        try:
            leased = task_queue.get()
        except (EOFError, OSError):
            break
        if leased is None:
            break
        attempt = 0
        heartbeat_interval = policy.heartbeat_interval
        if heartbeat_interval is None:
            heartbeat_interval = max(0.5, policy.lease_ttl * 0.3) if policy.lease_ttl > 0 else 0.0
        while attempt <= policy.max_retries:
            start = time.time()
            hb_event: Optional[threading.Event] = None
            hb_thread: Optional[threading.Thread] = None
            try:
                if heartbeat_interval and heartbeat_interval > 0:
                    hb_event = threading.Event()
                    hb_thread = threading.Thread(
                        target=_heartbeat_pump,
                        args=(result_queue, leased.task.task_id, heartbeat_interval, hb_event),
                        daemon=True,
                    )
                    hb_thread.start()
                result = _call_with_timeout(handler, leased, context, policy.task_timeout)
                _persist_task_result(result, worker_logger, leased.task.task_id)
                latency = time.time() - start
                result_queue.put(("ack", leased, latency, result))
                break
            except Exception as exc:  # pragma: no cover - worker level safety
                attempt += 1
                latency = time.time() - start
                tb = traceback.format_exc()
                if attempt <= policy.max_retries:
                    backoff = policy.backoff_base * (2 ** (attempt - 1))
                    jitter = rng.uniform(0, policy.backoff_jitter)
                    delay = backoff + jitter
                    result_queue.put(("retry", leased, attempt, str(exc)))
                    time.sleep(delay)
                    continue
                result_queue.put(("dead", leased, str(exc), tb, latency))
                break
            finally:
                if hb_event is not None:
                    hb_event.set()
                if hb_thread is not None:
                    hb_thread.join(timeout=0.2)


class ParallelExecutor:
    def __init__(
        self,
        handler: Callable[[LeasedTask, Any], TaskResult],
        pool: TaskPool,
        policy: ExecutorPolicy,
        *,
        events: Optional[ExecutorEvents] = None,
        handler_context: Any = None,
        result_handler: Optional[Callable[[LeasedTask, TaskResult], None]] = None,
        gpu_manager: Optional[GPUResourceManager] = None,
        console_min_level: str = "WARN",
    ) -> None:
        if policy.max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        if policy.lease_batch_size <= 0:
            raise ValueError("lease_batch_size must be positive")
        if policy.prefetch <= 0:
            raise ValueError("prefetch must be positive")
        self._handler = handler
        self._pool = pool
        self._policy = policy
        self._handler_context = handler_context
        self._events = events or ExecutorEvents()
        self._result_handler = result_handler or (lambda leased, result: None)
        self._gpu_manager = gpu_manager or GPUResourceManager.discover(policy.preferred_gpus)
        self._console_min_level = console_min_level.upper()
        self._ctx = mp.get_context("spawn")
        self._task_queue: "mp.Queue[Optional[LeasedTask]]" = self._ctx.Queue(maxsize=policy.prefetch)
        self._result_queue: "mp.Queue[Any]" = self._ctx.Queue()
        self._stop_event = threading.Event()
        self._dispatch_done = threading.Event()
        self._workers: List[mp.Process] = []
        self._active_lock = threading.Lock()
        self._active_tasks = 0
        self._last_idle_log = 0.0
        self._idle_sleep = max(0.01, min(self._policy.idle_sleep, 0.05))
        self._events.on_start(self)
        logger.info(
            "executor.start",
            concurrency=policy.max_concurrency,
            prefetch=policy.prefetch,
            gpu_devices=self._gpu_manager.available(),
            preferred_gpus=list(policy.preferred_gpus) if policy.preferred_gpus else None,
        )
        self._fatal_error: Optional[Exception] = None
        self._fatal_lock = threading.Lock()

    @classmethod
    def run(
        cls,
        handler: Callable[[LeasedTask, Any], TaskResult],
        pool: TaskPool,
        policy: ExecutorPolicy,
        *,
        events: Optional[ExecutorEvents] = None,
        handler_context: Any = None,
        result_handler: Optional[Callable[[LeasedTask, TaskResult], None]] = None,
        gpu_manager: Optional[GPUResourceManager] = None,
        console_min_level: str = "WARN",
    ) -> "ParallelExecutor":
        executor = cls(
            handler,
            pool,
            policy,
            events=events,
            handler_context=handler_context,
            result_handler=result_handler,
            gpu_manager=gpu_manager,
            console_min_level=console_min_level,
        )
        executor.start()
        executor.wait()
        return executor

    def start(self) -> None:
        self._start_workers()
        self._dispatcher_thread = threading.Thread(target=self._dispatch_loop, name="executor-dispatch", daemon=True)
        self._dispatcher_thread.start()
        self._result_thread = threading.Thread(target=self._result_loop, name="executor-results", daemon=True)
        self._result_thread.start()

    def wait(self) -> None:
        if hasattr(self, "_dispatcher_thread") and self._dispatcher_thread:
            self._dispatcher_thread.join()
        if hasattr(self, "_result_thread") and self._result_thread:
            self._result_thread.join()
        self._stop_workers()
        for proc in self._workers:
            proc.join()
        self._events.on_stop(self)
        logger.info("executor.stop")
        if self._fatal_error is not None:
            raise self._fatal_error

    def stop(self) -> None:
        self._stop_event.set()

    def _start_workers(self) -> None:
        namespace = "cad.parallel_executor"
        for worker_id in range(self._policy.max_concurrency):
            device = self._gpu_manager.assign(worker_id)
            proc = self._ctx.Process(
                target=_worker_main,
                args=(
                    worker_id,
                    self._task_queue,
                    self._result_queue,
                    self._handler,
                    self._handler_context,
                    self._policy,
                    device,
                    namespace,
                    self._console_min_level,
                ),
                name=f"executor-worker-{worker_id}",
            )
            proc.daemon = False
            proc.start()
            self._workers.append(proc)

    def _record_fatal(self, exc: Exception) -> None:
        with self._fatal_lock:
            if self._fatal_error is None:
                self._fatal_error = exc
        with self._active_lock:
            self._active_tasks = 0
        self._stop_event.set()

    def _log_dispatch_idle(self, reason: str, stats: Mapping[str, Union[int, float]]) -> None:
        now = time.time()
        if now - self._last_idle_log < 10.0:
            return
        self._last_idle_log = now
        visible = int(stats.get("visible", 0))
        leased = int(stats.get("leased", 0))
        dead = int(stats.get("dead", 0))
        logger.info(
            "executor.dispatch.idle",
            reason=reason,
            visible=visible,
            leased=leased,
            dead=dead,
            active=self._active_tasks,
        )

    def _should_stop_dispatch(self, stats: Mapping[str, Union[int, float]]) -> bool:
        if not self._pool.is_sealed():
            return False
        if self._active_tasks > 0:
            return False
        visible = int(stats.get("visible", 0))
        leased = int(stats.get("leased", 0))
        if visible > 0 or leased > 0:
            return False
        total = int(stats.get("total", 0))
        acked = int(stats.get("acked", -1))
        return acked >= total

    def _stop_workers(self) -> None:
        for _ in self._workers:
            try:
                self._task_queue.put_nowait(None)
            except queue.Full:  # pragma: no cover - defensive
                self._task_queue.put(None)

    def _increment_active(self, amount: int = 1) -> None:
        with self._active_lock:
            self._active_tasks += amount

    def _decrement_active(self, amount: int = 1) -> None:
        with self._active_lock:
            self._active_tasks = max(0, self._active_tasks - amount)

    def _dispatch_loop(self) -> None:
        logger.bind(stage="Execution")
        try:
            while not self._stop_event.is_set():
                for proc in list(self._workers):
                    if not proc.is_alive():
                        exit_code = proc.exitcode
                        self._record_fatal(RuntimeError(f"Worker {proc.name} exited unexpectedly with code {exit_code}"))
                        logger.error(
                            "executor.worker.exit",
                            worker=proc.name,
                            exitcode=exit_code,
                        )
                        self._stop_event.set()
                        break
                if self._stop_event.is_set():
                    break
                queue_size = self._task_queue.qsize()
                available_slots = max(0, self._policy.prefetch - queue_size)
                if available_slots <= 0:
                    stats = self._pool.stats()
                    self._log_dispatch_idle("prefetch_full", stats)
                    time.sleep(self._idle_sleep)
                    continue
                request = min(self._policy.lease_batch_size, available_slots)
                if request <= 0:
                    time.sleep(self._idle_sleep)
                    continue
                leased = self._pool.lease(request, self._policy.lease_ttl, filters=self._policy.filters)
                if not leased:
                    stats = self._pool.stats()
                    self._log_dispatch_idle("no_visible_tasks", stats)
                    if self._should_stop_dispatch(stats):
                        break
                    time.sleep(self._idle_sleep)
                    continue
                for idx, item in enumerate(leased):
                    if idx in (0, len(leased) // 2, len(leased) - 1):
                        logger.debug(
                            "executor.leased",
                            task_id=item.task.task_id,
                            attempt=item.attempt,
                            lease_deadline=item.lease_deadline,
                        )
                    self._events.on_lease(self, item)
                    self._task_queue.put(item)
                    self._increment_active()
            logger.info("executor.dispatch.complete")
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("executor.dispatch.error", exc, error=str(exc))
            self._record_fatal(RuntimeError(f"Dispatch loop failed: {exc}"))
        finally:
            self._dispatch_done.set()

    def _result_loop(self) -> None:
        logger.bind(stage="Execution")
        while True:
            if self._stop_event.is_set() and self._dispatch_done.is_set() and self._active_tasks == 0:
                break
            try:
                message = self._result_queue.get(timeout=0.5)
            except queue.Empty:
                # 当分发线程结束且无活跃任务时收尾
                if self._dispatch_done.is_set() and self._active_tasks == 0:
                    break
                continue
            msg_type = message[0]
            if msg_type == "ack":
                _, leased, latency, result = message
                success = self._pool.ack(leased.task.task_id)
                if not success:
                    logger.debug("executor.ack_failed", task_id=leased.task.task_id)
                self._events.on_success(self, leased, latency, result)
                try:
                    self._result_handler(leased, result)
                except Exception as exc:  # pragma: no cover - handler safety
                    logger.error("executor.result_handler_error", task_id=leased.task.task_id, error=str(exc))
                processed = int(getattr(result, "processed", 0) or 0)
                if processed <= 0:
                    processed = 1
                self._decrement_active()
            elif msg_type == "retry":
                _, leased, attempt, error_text = message
                exc_obj = RuntimeError(str(error_text))
                self._events.on_retry(self, leased, attempt, exc_obj)
            elif msg_type == "dead":
                _, leased, error_text, tb, latency = message
                exc_obj = RuntimeError(str(error_text))
                self._events.on_dead(self, leased, exc_obj)
                if not self._pool.mark_dead(leased.task.task_id, str(error_text)):
                    logger.error("executor.mark_dead_failed", task_id=leased.task.task_id, error=error_text)
                else:
                    logger.error(
                        "executor.task_dead",
                        task_id=leased.task.task_id,
                        error=error_text,
                        traceback=tb,
                        latency=latency,
                    )
                self._decrement_active()
            elif msg_type == "hb":
                _, task_id = message
                if not self._pool.heartbeat(task_id):
                    logger.debug("executor.heartbeat_lost", task_id=task_id)
            else:  # pragma: no cover - defensive
                logger.warning("executor.unknown_message", message=msg_type)
        logger.info("executor.results.complete")


__all__ = [
    "ExecutorEvents",
    "ExecutorPolicy",
    "ParallelExecutor",
    "TaskResult",
]

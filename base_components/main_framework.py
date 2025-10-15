# base_components/main_framework.py
from __future__ import annotations

import contextlib
import signal
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Iterable, Optional, Sequence, Union, List

from .config import Config
from .logger import StructuredLogger
from .task_partitioner import TaskRecord
from .task_pool import TaskPool, LeasedTask
from .parallel_executor import (
    ParallelExecutor,
    ExecutorPolicy,
    ExecutorEvents,
    TaskResult,
    _persist_task_result,
)
from .task_system_config import ensure_task_config


class TaskFramework(ABC):
    """大任务框架。继承并实现 `produce_tasks` 与 `handle` 即可。

    典型用法：
        class MyJob(TaskFramework):
            def produce_tasks(self) -> Iterable[TaskRecord]:
                ...  # 生成 TaskRecord 流
            def handle(self, leased: LeasedTask, context: Any) -> TaskResult:
                ...  # Worker 执行单个任务

        if __name__ == "__main__":
            MyJob().run()
    """

    def __init__(self) -> None:
        self._cfg: Optional[Config] = None
        self._pool: Optional[TaskPool] = None
        self._executor: Optional[ParallelExecutor] = None
        self._stop_flag = threading.Event()
        self._logger = StructuredLogger.get_logger("cad.main_framework")
        self._job_id: str = "job"

    # ------------------------- 需子类实现 -------------------------

    @abstractmethod
    def produce_tasks(self) -> Iterable[TaskRecord]:
        """扫描并生成任务。返回 TaskRecord 的迭代器。"""
        raise NotImplementedError

    @abstractmethod
    def handle(self, leased: LeasedTask, context: Any) -> TaskResult:
        """Worker 执行单个任务。必须返回 TaskResult（含持久化提示）。"""
        raise NotImplementedError

    # ------------------------- 可选覆盖点 -------------------------

    def result_handler(self, leased: LeasedTask, result: TaskResult) -> None:
        """主进程处理 worker 返回结果。默认无动作。"""
        _ = leased, result

    def build_handler_context(self, cfg: Config) -> Any:
        """构建传入 worker 的只读上下文。默认 None。"""
        return None

    @property
    def job_id(self) -> str:
        return self._job_id

    def set_job_id(self, job_id: str) -> None:
        self._job_id = str(job_id)

    def before_run(self, cfg: Config) -> None:
        """任务开始前的准备钩子。默认无动作。"""
        _ = cfg

    def after_run(self, cfg: Config) -> None:
        """全部结束后的收尾钩子。默认无动作。"""
        _ = cfg

    # ------------------------- 主流程 -------------------------

    def run(self) -> int:
        self._cfg = ensure_task_config()
        StructuredLogger.configure_from_config(self._cfg)

        self._logger.bind(stage="Main")
        self._logger.info("framework.start")

        exit_code = 0
        total_emitted = 0
        drained = False
        before_run_called = False
        parallel_enabled = True

        try:
            self.before_run(self._cfg)
            before_run_called = True

            self._pool = TaskPool()
            _ = self._pool.stats()

            policy = ExecutorPolicy()
            events = ExecutorEvents(
                on_start=lambda ex: self._logger.info(
                    "executor.started", concurrency=policy.max_concurrency
                ),
                on_lease=lambda ex, leased: None,
                on_success=lambda ex, leased, latency, result: self._logger.debug(
                    "task.ok", task_id=leased.task.task_id, latency_ms=int(latency * 1000)
                ),
                on_retry=lambda ex, leased, attempt, exc: self._logger.warn(
                    "task.retry", task_id=leased.task.task_id, attempt=attempt, error=str(exc)
                ),
                on_dead=lambda ex, leased, exc: self._logger.error(
                    "task.dead", task_id=leased.task.task_id, error=str(exc)
                ),
                on_stop=lambda ex: self._logger.info("executor.stopped"),
            )
            handler_ctx = self.build_handler_context(self._cfg)
            try:
                self._executor = ParallelExecutor(
                    handler=self.handle,
                    pool=self._pool,
                    policy=policy,
                    events=events,
                    handler_context=handler_ctx,
                    result_handler=self.result_handler,
                )
                self._executor.start()
            except (PermissionError, OSError) as exc:
                parallel_enabled = False
                self._executor = None
                self._logger.warn("executor.parallel_unavailable", error=str(exc))

            self._install_signal_handlers()

            batch_size = int(self._cfg.get("task_system.producer.batch_size", int, 128))
            total_emitted = self._emit_all(self.produce_tasks(), batch_size=batch_size)

            if self._pool and not self._pool.is_sealed():
                self._pool.seal()
                self._logger.info("producer.sealed", emitted=total_emitted)

            if parallel_enabled and self._executor is not None:
                if self._pool:
                    self._pool.wait_until_drained()
                    drained = True
                self._executor.wait()
            else:
                self._consume_sequential(policy, handler_ctx)
                if self._pool:
                    self._pool.wait_until_drained()
                    drained = True

        except Exception as exc:
            exit_code = 1
            self._logger.exception("framework.error", exc)
            if self._pool and not self._pool.is_sealed():
                with contextlib.suppress(Exception):
                    self._pool.seal()
                    self._logger.info("producer.sealed", emitted=total_emitted)
            if self._executor:
                with contextlib.suppress(Exception):
                    self._executor.stop()
                    self._executor.wait()
            if self._pool and not drained:
                with contextlib.suppress(Exception):
                    self._pool.wait_until_drained(timeout=5.0)
        finally:
            if before_run_called and self._cfg is not None:
                try:
                    self.after_run(self._cfg)
                except Exception as exc:
                    self._logger.error("framework.after_run_error", error=str(exc))
            self._logger.info("framework.done", emitted=total_emitted, exit_code=exit_code)
        return exit_code

    # ------------------------- 内部工具 -------------------------

    def _emit_all(self, tasks: Iterable[TaskRecord], *, batch_size: int = 128) -> int:
        assert self._pool is not None
        buf: List[TaskRecord] = []
        total = 0
        for t in tasks:
            if self._stop_flag.is_set():
                break
            buf.append(t)
            if len(buf) >= max(1, batch_size):
                self._pool.put(buf)
                total += len(buf)
                buf.clear()
        if buf:
            self._pool.put(buf)
            total += len(buf)
        return total

    def _consume_sequential(self, policy: ExecutorPolicy, handler_ctx: Any) -> None:
        if self._pool is None:
            return
        lease_ttl = policy.lease_ttl if policy.lease_ttl and policy.lease_ttl > 0 else 10.0
        filters = policy.filters
        worker_logger = StructuredLogger.get_logger("cad.main_framework.sequential")
        while not self._stop_flag.is_set():
            leased_batch = self._pool.lease(1, lease_ttl, filters=filters)
            if not leased_batch:
                break
            leased = leased_batch[0]
            try:
                result = self.handle(leased, handler_ctx)
                _persist_task_result(result, worker_logger, leased.task.task_id)
                self._pool.ack(leased.task.task_id)
                self.result_handler(leased, result)
            except Exception as exc:  # pragma: no cover - sequential fallback safety
                self._logger.error(
                    "framework.sequential_error",
                    task_id=leased.task.task_id,
                    error=str(exc),
                )
                with contextlib.suppress(Exception):
                    self._pool.nack(leased.task.task_id, requeue=False, delay=None)

    def _install_signal_handlers(self) -> None:
        def _graceful(signum, frame):
            _ = frame
            if self._stop_flag.is_set():
                return
            self._stop_flag.set()
            self._logger.warn("signal.received", signum=signum)
            try:
                if self._pool:
                    self._pool.seal()
            except Exception:
                pass
            try:
                if self._executor:
                    self._executor.stop()
            except Exception:
                pass

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _graceful)
            except Exception:
                # 非主线程或不支持平台忽略
                pass

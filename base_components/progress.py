# progress.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多进程感知的进度汇报工具，基于 tqdm。

新增：
- 支持“流式进度模式”：当任务总量未知时（total_units=None，或显式 stream=True），以“已处理条数 + 吞吐率”形式持续滚动，不再显示百分比。
- 原有“有界进度模式”保持不变（百分比 0–100%）。

用法示例：
# 1) 有界进度（已知总量）
with ProgressController(total_units=1000, description="Train", unit_name="samples") as pc:
    ...
    pc.advance(1)

# 2) 流式进度（未知总量）
with ProgressController(total_units=None, description="Ingest", unit_name="items") as pc:
    ...
    pc.advance(1)
"""

from __future__ import annotations

import collections
import math
import multiprocessing as mp
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

from tqdm import tqdm


@dataclass
class ProgressProxy:
    """轻量代理，传递到子进程用于汇报增量。"""

    _queue: Any

    def advance(self, units: int = 1) -> None:
        """向主进程推送完成量，单位通常为“样本数”或“任务条目数”。"""
        if units <= 0:
            return
        try:
            self._queue.put(units, block=False)
        except queue.Full:  # pragma: no cover - 退化兜底
            self._queue.put(units)

    def discovered(self, units: int = 1) -> None:
        """向主进程推送【已扫描/已产生】的数量。"""
        if units <= 0:
            return
        msg = ('D', int(units))
        try:
            self._queue.put(msg, block=False)
        except queue.Full:  # pragma: no cover - 退化兜底
            self._queue.put(msg)


class ProgressController:
    """聚合多进程进度并在主进程用 tqdm 输出。

    模式
    - 有界进度：已知 total_units，显示百分比与剩余时间（tqdm 行为）。
    - 流式进度：未知总量（total_units=None 或显式 stream=True），连续累计“已处理数”，显示速率与耗时。

    线程安全：仅主进程打印；子进程通过 ProgressProxy 汇报。
    """

    def __init__(
        self,
        total_units: Optional[int],
        description: str = "",
        *,
        stream: Optional[bool] = None,
        unit_name: str = "it",
    ) -> None:
        # 判定模式
        inferred_stream = total_units is None or (isinstance(total_units, int) and total_units <= 0)
        self._is_stream = stream if stream is not None else inferred_stream

        # 公共属性
        self.description = description or "Progress"
        self.unit_name = unit_name

        # 有界模式参数
        if not self._is_stream:
            self.total_units = max(int(total_units), 1)
            self._scale = 100.0  # 统一映射为百分比刻度
            self._precision = 0.01  # 百分比最小粒度
        else:
            self.total_units = None
            self._scale = None
            self._precision = None

        self._ctx = mp.get_context()
        self._queue, self._queue_has_timeout = self._create_queue()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._bar: Optional[tqdm] = None
        self._completed_units: int = 0  # 兼容旧字段
        self._processed_units: int = 0
        self._discovered_units: int = 0
        self._last_units: float = 0.0  # 对应百分比或已更新的 bar 进度（流式下仅用于清理）

    def __enter__(self) -> "ProgressController":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def start(self) -> None:
        """启动进度条刷新线程。"""
        if self._thread is not None:
            return

        if self._is_stream:
            # total=None 表示未知总量，tqdm 会显示吞吐率与累计
            self._bar = tqdm(
                total=None,
                desc=self.description,
                unit=self.unit_name,
                dynamic_ncols=True,
                mininterval=0.2,
                leave=True,
                smoothing=0.0,
            )
        else:
            self._bar = tqdm(
                total=self._scale,
                desc=self.description,
                unit="%",
                dynamic_ncols=True,
                mininterval=0.2,
                leave=True,
                smoothing=0.0,
            )

        self._thread = threading.Thread(target=self._pump, name="progress-pump", daemon=True)
        self._thread.start()
        self._refresh_label()
        self._bar.refresh()

    def make_proxy(self) -> ProgressProxy:
        """生成可跨进程传递的进度代理。"""
        return ProgressProxy(self._queue)

    def advance(self, units: int = 1) -> None:
        """主线程直接推进【已处理完成】（便于无子进程场景）。"""
        ProgressProxy(self._queue).advance(units)

    def discovered(self, units: int = 1) -> None:
        """主线程直接推进【已扫描/已产生】数量。"""
        ProgressProxy(self._queue).discovered(units)

    def close(self) -> None:
        """结束进度刷新并回收资源。"""
        if self._thread is None:
            return
        self._stop_event.set()
        try:
            self._queue.put(None)
        except Exception:  # pragma: no cover - 防御
            pass
        self._thread.join()
        self._thread = None
        if self._bar is not None:
            if not self._is_stream:
                # 有界模式：确保进度到 100%
                if self._last_units < float(self._scale):  # type: ignore[arg-type]
                    self._bar.update(float(self._scale) - self._last_units)  # type: ignore[arg-type]
            # 流式模式无需补齐
            self._bar.close()
            self._bar = None

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    
    def _refresh_label(self) -> None:
        """刷新进度条文字，将“已扫描/产生 | 已处理”并列显示。"""
        if self._bar is None:
            return
        try:
            postfix = f"已扫描/产生 {self._discovered_units} | 已处理 {self._processed_units}"
            self._bar.set_postfix_str(postfix)
        except Exception:
            base = self.description or "Progress"
            self._bar.set_description(f"{base} [{postfix}]")
    def _pump(self) -> None:
        """后台线程从队列读取增量并驱动 tqdm。"""
        while True:
            if self._stop_event.is_set() and self._queue.empty():
                break
            try:
                delta = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if delta is None:
                break

            # 支持 ('D', n) 报文表示已扫描/已产生；兼容旧版 int
            kind = 'P'
            if isinstance(delta, tuple) and len(delta) == 2:
                kind, value = delta
                inc = int(value)
            else:
                inc = int(delta)
            if inc <= 0:
                continue
            if kind == 'D':
                self._discovered_units += inc
                self._refresh_label()
                continue
            self._processed_units += inc
            self._completed_units += inc
            self._refresh_label()

            if self._is_stream:
                # 未知总量：直接把增量映射为 bar.update(inc)
                if self._bar is not None:
                    self._bar.update(inc)
                    self._last_units += inc
            else:
                # 已知总量：按百分比推进
                self._advance_to_percent(self._completed_units)
                if self._last_units >= float(self._scale):  # type: ignore[arg-type]
                    self._stop_event.set()
                    break

        # 清理剩余队列，避免遗漏尾部增量
        while True:
            try:
                delta = self._queue.get_nowait()
            except queue.Empty:
                break
            if delta is None:
                continue
            # 支持 ('D', n) 报文表示已扫描/已产生；兼容旧版 int
            kind = 'P'
            if isinstance(delta, tuple) and len(delta) == 2:
                kind, value = delta
                inc = int(value)
            else:
                inc = int(delta)
            if inc <= 0:
                continue
            if kind == 'D':
                self._discovered_units += inc
                self._refresh_label()
                continue
            self._processed_units += inc
            self._completed_units += inc
            self._refresh_label()
            if self._is_stream:
                if self._bar is not None:
                    self._bar.update(inc)
                    self._last_units += inc
            else:
                self._advance_to_percent(self._completed_units)

    def _advance_to_percent(self, completed_units: int) -> None:
        """将累计完成量映射为百分比并推进 tqdm。仅用于有界模式。"""
        if self._bar is None or self._is_stream:
            return
        percent = min(float(self._scale), (completed_units / self.total_units) * 100.0)  # type: ignore[operator]
        percent = self._precision * math.floor(percent / self._precision)  # type: ignore[operator]
        if percent <= self._last_units:
            return
        diff = percent - self._last_units
        self._bar.update(diff)
        self._last_units = percent

    # ------------------------------------------------------------------
    # queue helpers
    # ------------------------------------------------------------------

    def _create_queue(self) -> tuple[Any, bool]:
        try:
            q = self._ctx.Queue()
            return q, True
        except (PermissionError, OSError):
            return _LocalQueue(), True


class _LocalQueue:
    """线程安全队列，作为 mp.Queue 的降级实现，支持 timeout。"""

    def __init__(self) -> None:
        self._data = collections.deque()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)

    def put(self, item: Any, block: bool = True) -> None:  # noqa: D401 - 与 mp.Queue 接口保持一致
        with self._not_empty:
            self._data.append(item)
            self._not_empty.notify()

    def get(self, timeout: Optional[float] = None) -> Any:
        with self._not_empty:
            if timeout is None:
                while not self._data:
                    self._not_empty.wait()
            elif timeout == 0:
                if not self._data:
                    raise queue.Empty
            else:
                end = time.monotonic() + timeout
                while not self._data:
                    remaining = end - time.monotonic()
                    if remaining <= 0:
                        raise queue.Empty
                    self._not_empty.wait(remaining)
            return self._data.popleft()

    def get_nowait(self) -> Any:
        return self.get(timeout=0)

    def empty(self) -> bool:
        with self._lock:
            return not self._data

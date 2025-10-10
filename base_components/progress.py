#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multi-process aware progress reporting utilities built on top of tqdm."""

from __future__ import annotations

import multiprocessing as mp
import queue
import threading
from dataclasses import dataclass
from typing import Optional

from tqdm import tqdm


@dataclass
class ProgressProxy:
    """Lightweight proxy passed到子进程，用于汇报进度增量。"""

    _queue: mp.Queue

    def advance(self, units: int = 1) -> None:
        """向主进程推送完成量，单位通常为“样本数”或“任务条目数”。

        子进程调用该方法不会直接打印，仅提交增量。
        """
        if units <= 0:
            return
        try:
            self._queue.put(units, block=False)
        except queue.Full:  # pragma: no cover - fallback
            self._queue.put(units)


class ProgressController:
    """负责聚合多进程进度的控制器，主进程使用 tqdm 输出进度条。

    - 进度条以 1% 为最小粒度。
    - 自动展示剩余时间（tqdm 默认行为）。
    - 多进程场景仅主进程打印，子进程通过 :class:`ProgressProxy` 汇报。
    """

    def __init__(self, total_units: int, description: str = "") -> None:
        self.total_units = max(int(total_units), 1)
        self.description = description or "Progress"
        self._ctx = mp.get_context()
        self._queue: mp.Queue = self._ctx.Queue()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._bar: Optional[tqdm] = None
        self._completed_units: int = 0
        self._last_percent: int = 0

    def __enter__(self) -> "ProgressController":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def start(self) -> None:
        """启动进度条刷新线程。"""
        if self._thread is not None:
            return
        self._bar = tqdm(
            total=100,
            desc=self.description,
            unit="%",
            dynamic_ncols=True,
            mininterval=0.2,
            leave=True,
            smoothing=0.0,
        )
        self._thread = threading.Thread(target=self._pump, name="progress-pump", daemon=True)
        self._thread.start()

    def make_proxy(self) -> ProgressProxy:
        """生成可跨进程传递的进度代理。"""
        return ProgressProxy(self._queue)

    def close(self) -> None:
        """结束进度刷新并回收资源。"""
        if self._thread is None:
            return
        self._stop_event.set()
        try:
            self._queue.put(None)
        except Exception:  # pragma: no cover - defensive
            pass
        self._thread.join()
        self._thread = None
        if self._bar is not None:
            if self._last_percent < 100:
                self._bar.update(100 - self._last_percent)
            self._bar.close()
            self._bar = None

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

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
            self._completed_units += int(delta)
            self._advance_to_percent(self._completed_units)
            if self._last_percent >= 100:
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
            self._completed_units += int(delta)
            self._advance_to_percent(self._completed_units)

    def _advance_to_percent(self, completed_units: int) -> None:
        if self._bar is None:
            return
        percent = min(100, int((completed_units / self.total_units) * 100))
        if percent <= self._last_percent:
            return
        diff = percent - self._last_percent
        self._bar.update(diff)
        self._last_percent = percent

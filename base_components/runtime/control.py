"""Generic scheduling and rate-control primitives."""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Optional

from ..logger import StructuredLogger

from .resource import ResourceMonitor, ResourceSnapshot, get_default_resource_monitor


@dataclass(frozen=True)
class DispatchDecision:
    """Result of a dispatch budget calculation."""

    dispatch: int
    reason: Optional[str] = None


class TokenBucket:
    """Token bucket with dynamic rate and capacity."""

    def __init__(self, rate: float, capacity: float, *, min_rate: float = 0.0, max_rate: Optional[float] = None) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if rate < 0:
            raise ValueError("rate must be non-negative")
        self._lock = threading.Lock()
        self._rate = float(rate)
        self._capacity = float(capacity)
        self._tokens = float(capacity)
        self._max_rate = float(max_rate) if max_rate is not None else None
        self._min_rate = max(0.0, float(min_rate))
        self._updated = time.monotonic()

    @property
    def rate(self) -> float:
        with self._lock:
            return self._rate

    def set_rate(self, rate: float) -> None:
        with self._lock:
            clamped = max(self._min_rate, rate)
            if self._max_rate is not None:
                clamped = min(self._max_rate, clamped)
            self._refill_locked(time.monotonic())
            self._rate = clamped

    def adjust_rate(self, delta: float) -> None:
        self.set_rate(self.rate + delta)

    def _refill_locked(self, now: float) -> None:
        elapsed = max(0.0, now - self._updated)
        if elapsed <= 0.0 or self._rate <= 0.0:
            self._updated = now
            return
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._updated = now

    def capacity(self) -> float:
        return self._capacity

    def available(self) -> float:
        with self._lock:
            self._refill_locked(time.monotonic())
            return self._tokens

    def consume(self, amount: float = 1.0) -> bool:
        if amount <= 0:
            return True
        with self._lock:
            self._refill_locked(time.monotonic())
            if self._tokens < amount:
                return False
            self._tokens -= amount
            return True

    def refund(self, amount: float = 1.0) -> None:
        if amount <= 0:
            return
        with self._lock:
            self._refill_locked(time.monotonic())
            self._tokens = min(self._capacity, self._tokens + amount)


class ExponentialMovingAverage:
    """Simple EMA helper."""

    def __init__(self, *, alpha: float, initial: Optional[float] = None) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self._alpha = alpha
        self._value = initial

    def update(self, sample: float) -> float:
        if self._value is None:
            self._value = float(sample)
        else:
            self._value = (self._alpha * float(sample)) + (1.0 - self._alpha) * self._value
        return self._value

    @property
    def value(self) -> Optional[float]:
        return self._value


class AIMDController:
    """Classic AIMD controller for discrete limits."""

    def __init__(
        self,
        *,
        minimum: int,
        maximum: int,
        additive_step: int,
        multiplicative_decay: float,
        initial: Optional[int] = None,
    ) -> None:
        if minimum <= 0:
            raise ValueError("minimum must be positive")
        if maximum < minimum:
            raise ValueError("maximum must be >= minimum")
        if additive_step <= 0:
            raise ValueError("additive_step must be positive")
        if not 0.0 < multiplicative_decay < 1.0:
            raise ValueError("multiplicative_decay must be in (0,1)")
        self._min = int(minimum)
        self._max = int(maximum)
        self._add = int(additive_step)
        self._decay = float(multiplicative_decay)
        start = initial if initial is not None else self._min
        self._value = max(self._min, min(self._max, int(start)))
        self._lock = threading.Lock()

    def increase(self) -> int:
        with self._lock:
            self._value = min(self._max, self._value + self._add)
            return self._value

    def decrease(self) -> int:
        with self._lock:
            decreased = max(self._min, int(math.floor(self._value * self._decay)))
            self._value = decreased
            return self._value

    @property
    def value(self) -> int:
        with self._lock:
            return self._value

    @property
    def additive_step(self) -> int:
        return self._add

    @property
    def multiplicative_decay(self) -> float:
        return self._decay

    def set_value(self, value: int) -> None:
        with self._lock:
            self._value = max(self._min, min(self._max, int(value)))


class AdaptiveDispatchController:
    """Coordinates dispatch bursts, rate limiting, and adaptive concurrency."""

    def __init__(
        self,
        *,
        max_concurrency: int,
        window_multiplier: float,
        initial_rate: float,
        rate_capacity: int,
        rate_min: float,
        rate_max: Optional[float],
        target_utilisation_low: float,
        target_utilisation_high: float,
        evaluation_interval: float,
        ema_alpha: float,
        aimd_step: int,
        aimd_decay: float,
        resource_monitor: Optional[ResourceMonitor] = None,
    ) -> None:
        if window_multiplier <= 0:
            raise ValueError("window_multiplier must be > 0")
        if not 0.0 < target_utilisation_low < target_utilisation_high <= 1.0:
            raise ValueError("target utilisation bounds must satisfy 0 < low < high <= 1")
        if evaluation_interval <= 0:
            raise ValueError("evaluation_interval must be positive")
        self._max_concurrency = max(1, int(max_concurrency))
        self._window_multiplier = float(window_multiplier)
        self._bucket = TokenBucket(
            rate=max(initial_rate, rate_min),
            capacity=float(rate_capacity),
            min_rate=rate_min,
            max_rate=rate_max,
        )
        self._aimd = AIMDController(
            minimum=1,
            maximum=self._max_concurrency,
            additive_step=max(1, aimd_step),
            multiplicative_decay=aimd_decay,
            initial=min(self._max_concurrency, max(1, aimd_step)),
        )
        self._target_low = target_utilisation_low
        self._target_high = target_utilisation_high
        self._evaluation_interval = evaluation_interval
        self._util_ema = ExponentialMovingAverage(alpha=ema_alpha)
        self._throughput_ema = ExponentialMovingAverage(alpha=ema_alpha)
        self._resource_monitor = resource_monitor or get_default_resource_monitor()
        self._lock = threading.Lock()
        self._last_eval = time.monotonic()
        self._completed_since_eval = 0
        self._last_completion_ts = None  # type: Optional[float]

    def compute_budget(
        self,
        *,
        active_tasks: int,
        queued: int,
        available_slots: int,
        desired: int,
    ) -> DispatchDecision:
        now = time.monotonic()
        with self._lock:
            self._maybe_evaluate(now, active_tasks)
            limit = self._aimd.value
            window_cap = max(1, int(math.ceil(limit * self._window_multiplier)))
            in_flight = active_tasks + queued
            remaining_window = max(0, window_cap - in_flight)
            if remaining_window <= 0:
                return DispatchDecision(0, "window_full")
            slots = min(remaining_window, available_slots, max(0, limit - active_tasks))
            if slots <= 0:
                return DispatchDecision(0, "concurrency_saturated")
            requested = max(0, min(desired, slots))
            if requested <= 0:
                return DispatchDecision(0, "no_slots")
            tokens_available = int(math.floor(self._bucket.available()))
            if tokens_available <= 0:
                return DispatchDecision(0, "rate_limited")
            dispatch = min(requested, tokens_available)
            if dispatch <= 0:
                return DispatchDecision(0, "rate_limited")
            self._bucket.consume(dispatch)
            return DispatchDecision(dispatch)

    def on_dispatched(self, count: int) -> None:
        # Included for interface symmetry; currently nothing extra needed.
        if count <= 0:
            return

    def refund(self, count: int) -> None:
        if count <= 0:
            return
        self._bucket.refund(count)

    def on_completed(self, count: int = 1, *, latency: Optional[float] = None, snapshot: Optional[ResourceSnapshot] = None) -> None:
        if count <= 0:
            return
        now = time.monotonic()
        with self._lock:
            self._completed_since_eval += count
            self._last_completion_ts = now
            if latency and latency > 0:
                throughput = count / max(latency, 1e-6)
                self._throughput_ema.update(throughput)
            if snapshot is not None:
                utilisation = self._select_utilisation(snapshot)
                if utilisation is not None:
                    self._util_ema.update(utilisation)

    def _maybe_evaluate(self, now: float, active_tasks: int) -> None:
        if (now - self._last_eval) < self._evaluation_interval:
            return
        snapshot = self._resource_monitor.snapshot()
        utilisation = self._select_utilisation(snapshot)
        if utilisation is None:
            # fallback to active ratio
            limit = self._aimd.value or 1
            utilisation = min(1.0, active_tasks / max(1, limit))
        self._util_ema.update(utilisation)
        observed = self._util_ema.value or utilisation
        if observed < self._target_low:
            new_limit = self._aimd.increase()
            self._bucket.adjust_rate(self._aimd.additive_step)
            StructuredLogger.get_logger("cad.dispatch.control").debug(
                "dispatch.aimd.increase",
                limit=new_limit,
                utilisation=float(observed),
            )
        elif observed > self._target_high:
            new_limit = self._aimd.decrease()
            current_rate = self._bucket.rate
            self._bucket.set_rate(current_rate * self._aimd.multiplicative_decay)
            StructuredLogger.get_logger("cad.dispatch.control").debug(
                "dispatch.aimd.decrease",
                limit=new_limit,
                utilisation=float(observed),
            )
        self._bucket.available()  # trigger refill bookkeeping
        self._completed_since_eval = 0
        self._last_eval = now

    @staticmethod
    def _select_utilisation(snapshot: Optional[ResourceSnapshot]) -> Optional[float]:
        if snapshot is None:
            return None
        candidates = []
        if snapshot.gpu_utilisation is not None:
            candidates.append(snapshot.gpu_utilisation)
        if snapshot.cpu_utilisation is not None:
            candidates.append(snapshot.cpu_utilisation)
        if snapshot.memory_utilisation is not None:
            candidates.append(snapshot.memory_utilisation)
        if not candidates:
            return None
        return max(candidates)

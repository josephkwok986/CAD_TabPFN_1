"""Task partitioner implementation.

This module implements the "任务分割" (Task Partitioner) component described in
the design notes.  The implementation follows the unified design principles:

* **Configurability first** – the :class:`Partitioner` accepts strategy and
  constraint parameters, and exposes deterministic behaviour through explicit
  seeds.
* **Safe defaults** – absence of optional settings falls back to conservative
  limits and validation.
* **Idempotent and re-entrant** – task identifiers are derived from stable
  hashes of their payload references.
* **Observability** – partition plans capture metrics (task counts, imbalance
  ratios, Gini coefficient, group crossing rate) ready for structured logging
  and metrics emission.
* **Extensible** – strategy handlers are pluggable and rely on adapter
  abstractions.
* **Deterministic** – optional ``shuffle_seed`` controls ordering while ULID-like
  task identifiers are derived from deterministic hashes.
* **Resource aware** – constraints cover weights, quotas and grouping.
* **Cross-environment resilient** – the code is pure Python, requiring only
  standard library and the dependency set declared in ``environment.yml``.

The public interface mirrors the specification::

    Partitioner.plan(job_spec, items, strategy, constraints) -> Plan
    Partitioner.emit(plan, sink=TaskPool)
    Partitioner.reshard(plan, reason)

The module is self-contained and ships with unit tests at the bottom so it can
be executed directly::

    python task_partitioner.py

"""
from __future__ import annotations

import base64
import dataclasses
import enum
import functools
import hashlib
import math
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Protocol, Sequence, Tuple

from typing_extensions import TypeAlias

__all__ = [
    "Partitioner",
    "Plan",
    "TaskRecord",
    "PartitionConstraints",
    "PartitionMetrics",
    "TaskSink",
]


_PAYLOAD_TYPE: TypeAlias = Any
_GROUP_KEY_TYPE: TypeAlias = Optional[Tuple[Any, ...]]


class Strategy(enum.Enum):
    """Supported partition strategies."""

    FIXED = "fixed"
    WEIGHT_BALANCED = "weight"
    HASH = "hash"

    @classmethod
    def from_value(cls, value: str | Strategy) -> Strategy:
        if isinstance(value, Strategy):
            return value
        lowered = value.lower().replace("-", "_")
        for member in cls:
            if member.value == lowered or member.name.lower() == lowered:
                return member
        raise ValueError(f"Unsupported partition strategy: {value!r}")


@dataclass(frozen=True)
class PartitionConstraints:
    """Runtime constraints for planning.

    The dataclass performs basic validation to guarantee sane inputs.
    """

    group_by: Optional[Sequence[str]] = None
    max_tasks: Optional[int] = None
    max_items_per_task: Optional[int] = None
    max_weight_per_task: Optional[float] = None
    shuffle_seed: Optional[int] = None
    affinity: Optional[Mapping[str, Any]] = None
    anti_affinity: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        if self.group_by is not None and isinstance(self.group_by, (str, bytes)):
            object.__setattr__(self, "group_by", (str(self.group_by),))
        if self.max_tasks is not None and self.max_tasks <= 0:
            raise ValueError("max_tasks must be positive")
        if self.max_items_per_task is not None and self.max_items_per_task <= 0:
            raise ValueError("max_items_per_task must be positive")
        if self.max_weight_per_task is not None and self.max_weight_per_task <= 0:
            raise ValueError("max_weight_per_task must be positive")
        if self.shuffle_seed is not None and not isinstance(self.shuffle_seed, int):
            raise TypeError("shuffle_seed must be an integer")


@dataclass(frozen=True)
class TaskRecord:
    """A concrete task description suitable for scheduling."""

    task_id: str
    job_id: str
    attempt: int
    payload_ref: Sequence[_PAYLOAD_TYPE]
    weight: float
    group_key: _GROUP_KEY_TYPE
    checksum: Optional[str]
    priority: Optional[int]
    deadline: Optional[str]
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)


@dataclass(frozen=True)
class PartitionMetrics:
    """Observability payload for the partition decision."""

    task_count: int
    total_weight: float
    min_weight: float
    max_weight: float
    mean_weight: float
    stddev_weight: float
    weight_gini: float
    imbalance_ratio: float
    group_cross_rate: float


@dataclass
class Plan:
    """A deterministic task plan for a job."""

    plan_id: str
    job_spec: Mapping[str, Any]
    strategy: Strategy
    constraints: PartitionConstraints
    tasks: List[TaskRecord]
    metrics: PartitionMetrics
    created_at: float
    reason: str = "initial"
    lineage: Tuple[str, ...] = field(default_factory=tuple)
    _bundles: Tuple["_GroupBundle", ...] = field(repr=False, default_factory=tuple)

    def snapshot(self) -> Mapping[str, Any]:
        """Return a serialisable snapshot."""

        return {
            "plan_id": self.plan_id,
            "job_spec": dict(self.job_spec),
            "strategy": self.strategy.value,
            "constraints": dataclasses.asdict(self.constraints),
            "tasks": [dataclasses.asdict(task) for task in self.tasks],
            "metrics": dataclasses.asdict(self.metrics),
            "created_at": self.created_at,
            "reason": self.reason,
            "lineage": list(self.lineage),
        }


class TaskSink(Protocol):
    """Protocol consumed by :meth:`Partitioner.emit`."""

    def put(self, task: TaskRecord) -> None:  # pragma: no cover - protocol definition
        ...


@dataclass(frozen=True)
class _NormalizedItem:
    payload_ref: _PAYLOAD_TYPE
    weight: float
    attributes: Mapping[str, Any]
    group_key: _GROUP_KEY_TYPE
    checksum: Optional[str]
    priority: Optional[int]
    deadline: Optional[str]


@dataclass(frozen=True)
class _GroupBundle:
    key: _GROUP_KEY_TYPE
    items: Tuple[_NormalizedItem, ...]

    @functools.cached_property
    def weight(self) -> float:
        return sum(item.weight for item in self.items)

    @functools.cached_property
    def payload_refs(self) -> Tuple[_PAYLOAD_TYPE, ...]:
        return tuple(item.payload_ref for item in self.items)

    @functools.cached_property
    def checksum(self) -> str:
        digest = hashlib.blake2b(digest_size=16)
        for item in self.items:
            digest.update(repr(item.payload_ref).encode("utf-8", "replace"))
            if item.checksum:
                digest.update(item.checksum.encode("utf-8"))
        return base64.b32encode(digest.digest()).decode("ascii").rstrip("=")


class Partitioner:
    """Plan tasks for homogeneous job items."""

    def __init__(self, *, id_factory: Optional[Callable[[str, Tuple[_PAYLOAD_TYPE, ...], _GROUP_KEY_TYPE], str]] = None) -> None:
        self._id_factory = id_factory or self._default_id_factory

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def plan(
        self,
        job_spec: Mapping[str, Any],
        items: Iterable[Mapping[str, Any] | _NormalizedItem],
        strategy: Strategy | str,
        constraints: PartitionConstraints,
    ) -> Plan:
        """Create a deterministic plan for the provided job items."""

        strategy_enum = Strategy.from_value(strategy)
        job_id = self._resolve_job_id(job_spec)
        normalized_items = tuple(self._normalize_items(items, constraints))
        bundles = self._bundle_items(normalized_items, constraints)
        tasks = self._build_tasks(job_id, strategy_enum, constraints, bundles)
        metrics = self._calculate_metrics(tasks)
        plan_id = self._make_plan_id(job_id, strategy_enum, constraints, tasks)
        now = time.time()
        return Plan(
            plan_id=plan_id,
            job_spec=dict(job_spec),
            strategy=strategy_enum,
            constraints=constraints,
            tasks=tasks,
            metrics=metrics,
            created_at=now,
            _bundles=bundles,
        )

    def emit(self, plan: Plan, sink: TaskSink) -> None:
        """Emit the plan into the provided sink."""

        for task in plan.tasks:
            sink.put(task)

    def reshard(
        self,
        plan: Plan,
        reason: str,
        *,
        strategy: Optional[Strategy | str] = None,
        constraints: Optional[PartitionConstraints] = None,
    ) -> Plan:
        """Re-plan existing bundles with an updated reason."""

        strategy_enum = Strategy.from_value(strategy or plan.strategy)
        constraints = constraints or plan.constraints
        tasks = self._build_tasks(plan.job_spec["job_id"], strategy_enum, constraints, plan._bundles)
        metrics = self._calculate_metrics(tasks)
        plan_id = self._make_plan_id(plan.job_spec["job_id"], strategy_enum, constraints, tasks)
        return Plan(
            plan_id=plan_id,
            job_spec=plan.job_spec,
            strategy=strategy_enum,
            constraints=constraints,
            tasks=tasks,
            metrics=metrics,
            created_at=time.time(),
            reason=reason,
            lineage=plan.lineage + (plan.plan_id,),
            _bundles=plan._bundles,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_job_id(self, job_spec: Mapping[str, Any]) -> str:
        job_id = job_spec.get("job_id")
        if not job_id:
            raise ValueError("job_spec must contain a 'job_id'")
        return str(job_id)

    def _normalize_items(
        self, items: Iterable[Mapping[str, Any] | _NormalizedItem], constraints: PartitionConstraints
    ) -> Iterator[_NormalizedItem]:
        for raw in items:
            if isinstance(raw, _NormalizedItem):
                yield raw
                continue
            if not isinstance(raw, Mapping):
                raise TypeError("Each item must be a mapping or _NormalizedItem")
            payload_ref = raw.get("payload_ref")
            if payload_ref is None:
                raise ValueError("Item missing payload_ref")
            weight = float(raw.get("weight", 1.0))
            if weight <= 0:
                raise ValueError("Item weight must be positive")
            priority = raw.get("priority")
            deadline = raw.get("deadline")
            checksum = raw.get("checksum")
            attributes = dict(raw)
            group_key = self._extract_group_key(raw, constraints.group_by)
            yield _NormalizedItem(
                payload_ref=payload_ref,
                weight=weight,
                attributes=attributes,
                group_key=group_key,
                checksum=checksum,
                priority=priority,
                deadline=deadline,
            )

    def _extract_group_key(
        self, attributes: Mapping[str, Any], group_by: Optional[Sequence[str]]
    ) -> _GROUP_KEY_TYPE:
        if not group_by:
            return None
        values: List[Any] = []
        for path in group_by:
            cur: Any = attributes
            for part in path.split("."):
                if isinstance(cur, Mapping) and part in cur:
                    cur = cur[part]
                else:
                    cur = None
                    break
            values.append(cur)
        return tuple(values)

    def _bundle_items(
        self, items: Sequence[_NormalizedItem], constraints: PartitionConstraints
    ) -> Tuple[_GroupBundle, ...]:
        if not constraints.group_by:
            return tuple(_GroupBundle(key=None, items=(item,)) for item in items)

        grouped: Dict[_GROUP_KEY_TYPE, List[_NormalizedItem]] = defaultdict(list)
        for item in items:
            grouped[item.group_key].append(item)
        bundles = []
        for key, bucket in grouped.items():
            bundle = _GroupBundle(key=key, items=tuple(bucket))
            if constraints.max_items_per_task is not None and len(bundle.items) > constraints.max_items_per_task:
                raise ValueError(
                    f"Group {key} exceeds max_items_per_task={constraints.max_items_per_task}"
                )
            if constraints.max_weight_per_task is not None and bundle.weight > constraints.max_weight_per_task:
                raise ValueError(
                    f"Group {key} exceeds max_weight_per_task={constraints.max_weight_per_task}"
                )
            bundles.append(bundle)
        return tuple(bundles)

    def _build_tasks(
        self,
        job_id: str,
        strategy: Strategy,
        constraints: PartitionConstraints,
        bundles: Sequence[_GroupBundle],
    ) -> List[TaskRecord]:
        if not bundles:
            return []
        if strategy is Strategy.FIXED:
            task_bundles = self._plan_fixed(bundles, constraints)
        elif strategy is Strategy.WEIGHT_BALANCED:
            task_bundles = self._plan_weight_balanced(bundles, constraints)
        elif strategy is Strategy.HASH:
            task_bundles = self._plan_hash(bundles, constraints)
        else:  # pragma: no cover - defensive
            raise AssertionError(f"Unsupported strategy: {strategy}")
        tasks: List[TaskRecord] = []
        for bucket in task_bundles:
            payloads = tuple(item.payload_ref for b in bucket for item in b.items)
            weight = sum(b.weight for b in bucket)
            group_keys = {b.key for b in bucket}
            group_key = next(iter(group_keys)) if len(group_keys) == 1 else None
            checksum = self._checksum_for_bucket(bucket)
            task_id = self._id_factory(job_id, payloads, group_key)
            priority = self._resolve_priority(bucket)
            deadline = self._resolve_deadline(bucket)
            metadata = {
                "groups": [b.key for b in bucket],
                "item_count": sum(len(b.items) for b in bucket),
            }
            tasks.append(
                TaskRecord(
                    task_id=task_id,
                    job_id=job_id,
                    attempt=0,
                    payload_ref=payloads,
                    weight=weight,
                    group_key=group_key,
                    checksum=checksum,
                    priority=priority,
                    deadline=deadline,
                    metadata=metadata,
                )
            )
        return tasks

    def _plan_fixed(
        self, bundles: Sequence[_GroupBundle], constraints: PartitionConstraints
    ) -> List[List[_GroupBundle]]:
        if constraints.max_items_per_task is None and constraints.max_tasks is None:
            raise ValueError("fixed strategy requires max_items_per_task or max_tasks")
        ordered = list(bundles)
        if constraints.shuffle_seed is not None:
            ordered.sort(key=lambda b: b.checksum)
        total_items = sum(len(bundle.items) for bundle in ordered)
        chunk_size = constraints.max_items_per_task or max(
            1, math.ceil(total_items / constraints.max_tasks)
        )
        tasks: List[List[_GroupBundle]] = []
        current: List[_GroupBundle] = []
        current_count = 0
        current_weight = 0.0
        for bundle in ordered:
            bundle_size = len(bundle.items)
            bundle_weight = bundle.weight
            if (
                current
                and (
                    current_count + bundle_size > chunk_size
                    or (
                        constraints.max_weight_per_task is not None
                        and current_weight + bundle_weight > constraints.max_weight_per_task
                    )
                )
            ):
                tasks.append(current)
                current = []
                current_count = 0
                current_weight = 0.0
            current.append(bundle)
            current_count += bundle_size
            current_weight += bundle_weight
        if current:
            tasks.append(current)
        return tasks

    def _plan_weight_balanced(
        self, bundles: Sequence[_GroupBundle], constraints: PartitionConstraints
    ) -> List[List[_GroupBundle]]:
        if not constraints.max_tasks and not constraints.max_weight_per_task:
            raise ValueError("weight strategy requires max_tasks or max_weight_per_task")
        ordered = sorted(bundles, key=lambda b: b.weight, reverse=True)
        task_count = constraints.max_tasks
        if task_count is None:
            total_weight = sum(b.weight for b in ordered)
            per_task = constraints.max_weight_per_task or total_weight
            task_count = max(1, math.ceil(total_weight / per_task))
        tasks: List[List[_GroupBundle]] = [[] for _ in range(task_count)]
        task_weights = [0.0 for _ in range(task_count)]
        for bundle in ordered:
            idx = min(range(task_count), key=lambda i: task_weights[i])
            if (
                constraints.max_weight_per_task is not None
                and task_weights[idx] + bundle.weight > constraints.max_weight_per_task
            ):
                raise ValueError(
                    f"Cannot place group {bundle.key} within max_weight_per_task={constraints.max_weight_per_task}"
                )
            tasks[idx].append(bundle)
            task_weights[idx] += bundle.weight
        return [bucket for bucket in tasks if bucket]

    def _plan_hash(
        self, bundles: Sequence[_GroupBundle], constraints: PartitionConstraints
    ) -> List[List[_GroupBundle]]:
        if not constraints.max_tasks:
            raise ValueError("hash strategy requires max_tasks")
        buckets: List[List[_GroupBundle]] = [[] for _ in range(constraints.max_tasks)]
        for bundle in bundles:
            key = bundle.key or bundle.checksum
            digest = hashlib.blake2b(digest_size=8)
            digest.update(repr(key).encode("utf-8", "replace"))
            if constraints.shuffle_seed is not None:
                digest.update(str(constraints.shuffle_seed).encode("ascii"))
            slot = int.from_bytes(digest.digest(), "big") % constraints.max_tasks
            buckets[slot].append(bundle)
        return [bucket for bucket in buckets if bucket]

    def _checksum_for_bucket(self, bucket: Sequence[_GroupBundle]) -> str:
        digest = hashlib.blake2b(digest_size=16)
        for bundle in bucket:
            digest.update(bundle.checksum.encode("ascii"))
        return base64.b32encode(digest.digest()).decode("ascii").rstrip("=")

    def _resolve_priority(self, bucket: Sequence[_GroupBundle]) -> Optional[int]:
        priorities = [item.priority for bundle in bucket for item in bundle.items if item.priority is not None]
        return min(priorities) if priorities else None

    def _resolve_deadline(self, bucket: Sequence[_GroupBundle]) -> Optional[str]:
        deadlines = [item.deadline for bundle in bucket for item in bundle.items if item.deadline is not None]
        return min(deadlines) if deadlines else None

    def _calculate_metrics(self, tasks: Sequence[TaskRecord]) -> PartitionMetrics:
        if not tasks:
            zero = PartitionMetrics(
                task_count=0,
                total_weight=0.0,
                min_weight=0.0,
                max_weight=0.0,
                mean_weight=0.0,
                stddev_weight=0.0,
                weight_gini=0.0,
                imbalance_ratio=0.0,
                group_cross_rate=0.0,
            )
            return zero
        weights = [task.weight for task in tasks]
        total_weight = sum(weights)
        min_weight = min(weights)
        max_weight = max(weights)
        mean_weight = total_weight / len(weights)
        stddev = statistics.pstdev(weights) if len(weights) > 1 else 0.0
        gini = self._gini(weights)
        imbalance_ratio = max_weight / mean_weight if mean_weight else 0.0
        group_cross_rate = self._group_cross_rate(tasks)
        return PartitionMetrics(
            task_count=len(tasks),
            total_weight=total_weight,
            min_weight=min_weight,
            max_weight=max_weight,
            mean_weight=mean_weight,
            stddev_weight=stddev,
            weight_gini=gini,
            imbalance_ratio=imbalance_ratio,
            group_cross_rate=group_cross_rate,
        )

    def _group_cross_rate(self, tasks: Sequence[TaskRecord]) -> float:
        total_groups = 0
        cross = 0
        for task in tasks:
            groups = task.metadata.get("groups", [])
            total_groups += len(groups)
            if len({tuple(g) for g in groups if g is not None}) > 1:
                cross += 1
        return cross / len(tasks) if tasks else 0.0

    def _gini(self, values: Sequence[float]) -> float:
        sorted_values = sorted(values)
        n = len(sorted_values)
        cumulative = 0.0
        cumulative_sum = 0.0
        for idx, value in enumerate(sorted_values, start=1):
            cumulative += value
            cumulative_sum += cumulative
        if not cumulative:
            return 0.0
        return (n + 1 - 2 * (cumulative_sum / cumulative)) / n

    def _make_plan_id(
        self,
        job_id: str,
        strategy: Strategy,
        constraints: PartitionConstraints,
        tasks: Sequence[TaskRecord],
    ) -> str:
        digest = hashlib.blake2b(digest_size=16)
        digest.update(job_id.encode("utf-8"))
        digest.update(strategy.value.encode("ascii"))
        digest.update(repr(dataclasses.asdict(constraints)).encode("utf-8"))
        for task in tasks:
            digest.update(task.task_id.encode("ascii"))
        return base64.b32encode(digest.digest()).decode("ascii").rstrip("=")

    def _default_id_factory(
        self, job_id: str, payloads: Tuple[_PAYLOAD_TYPE, ...], group_key: _GROUP_KEY_TYPE
    ) -> str:
        digest = hashlib.blake2b(digest_size=16)
        digest.update(job_id.encode("utf-8"))
        digest.update(repr(group_key).encode("utf-8", "replace"))
        for ref in payloads:
            digest.update(repr(ref).encode("utf-8", "replace"))
        return base64.b32encode(digest.digest()).decode("ascii").rstrip("=")


# ---------------------------------------------------------------------------
# In-memory sink for testing
# ---------------------------------------------------------------------------


class InMemoryTaskPool:
    """Simple sink collecting emitted tasks for testing and offline planning."""

    def __init__(self) -> None:
        self.tasks: List[TaskRecord] = []

    def put(self, task: TaskRecord) -> None:
        self.tasks.append(task)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


import unittest


class PartitionerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.partitioner = Partitioner()
        self.job_spec = {"job_id": "job-001", "owner": "cad"}

    def _items(self, weights: Sequence[float], group: Optional[str] = None) -> List[Dict[str, Any]]:
        items = []
        for idx, weight in enumerate(weights):
            payload = f"item-{idx}"
            attrs: Dict[str, Any] = {
                "payload_ref": payload,
                "weight": weight,
                "family": group or f"g{idx % 2}",
            }
            items.append(attrs)
        return items

    def test_fixed_partitioning_respects_chunk_size(self) -> None:
        items = self._items([1] * 10)
        constraints = PartitionConstraints(max_items_per_task=3)
        plan = self.partitioner.plan(self.job_spec, items, strategy="fixed", constraints=constraints)
        self.assertEqual(plan.metrics.task_count, 4)
        self.assertTrue(all(task.metadata["item_count"] <= 3 for task in plan.tasks))
        self.assertAlmostEqual(plan.metrics.group_cross_rate, 0.0)

    def test_weight_balanced_minimises_imbalance(self) -> None:
        items = self._items([9, 7, 5, 3, 1], group="alpha")
        constraints = PartitionConstraints(max_tasks=3)
        plan = self.partitioner.plan(self.job_spec, items, strategy="weight", constraints=constraints)
        self.assertEqual(plan.metrics.task_count, 3)
        self.assertLess(plan.metrics.imbalance_ratio, 1.6)

    def test_hash_strategy_is_stable(self) -> None:
        items = self._items([1, 2, 3, 4])
        constraints = PartitionConstraints(max_tasks=2, group_by=("family",))
        plan_a = self.partitioner.plan(self.job_spec, items, strategy="hash", constraints=constraints)
        plan_b = self.partitioner.plan(self.job_spec, items, strategy="hash", constraints=constraints)
        ids_a = sorted(task.task_id for task in plan_a.tasks)
        ids_b = sorted(task.task_id for task in plan_b.tasks)
        self.assertEqual(ids_a, ids_b)

    def test_reshard_updates_lineage(self) -> None:
        items = self._items([1, 1, 1, 1])
        constraints = PartitionConstraints(max_items_per_task=2)
        plan = self.partitioner.plan(self.job_spec, items, strategy="fixed", constraints=constraints)
        new_constraints = PartitionConstraints(max_items_per_task=4)
        reshaped = self.partitioner.reshard(plan, "manual-adjustment", constraints=new_constraints)
        self.assertEqual(reshaped.reason, "manual-adjustment")
        self.assertEqual(reshaped.lineage, (plan.plan_id,))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

#!/usr/bin/env python3
"""Task partitioning service.

This module implements the "任务分割类" described in the specification. The
design emphasises deterministic planning, strong observability and
configurable-yet-safe defaults.  The public interface is provided by the
``TaskPartitioner`` class which exposes ``plan``, ``emit`` and ``reshard``
methods.  The output of ``plan`` is a :class:`Plan` object containing
``TaskRecord`` instances suitable for submitting to a task pool.

The implementation favours composability:

* Strategies are pluggable via :class:`PartitionStrategy`.
* Constraints are modelled as an immutable dataclass ensuring idempotent
  execution and reproducible partitioning when ``shuffle_seed`` is used.
* Statistics are generated as part of the plan to help with observability.

Basic unit tests are provided at the bottom of the module and can be executed
directly::

    python task_partitioner.py
"""
from __future__ import annotations

import collections
import contextlib
import dataclasses
import hashlib
import math
import random
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .progress import ProgressController


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


def _canonical_bytes(value: Any) -> bytes:
    """Return canonical JSON bytes for hashing.

    ``orjson`` is not imported here to avoid hard dependency.  Python's built-in
    ``repr`` is sufficient for deterministic hashes because inputs are already
    normalised.
    """

    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8")
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if isinstance(value, Mapping):
        parts = [f"{k}:{_canonical_bytes(v).decode('utf-8', 'ignore')}" for k, v in sorted(value.items(), key=lambda kv: kv[0])]
        return ("{" + ",".join(parts) + "}").encode("utf-8")
    if isinstance(value, (list, tuple, set, frozenset)):
        seq = list(value)
        if isinstance(value, (set, frozenset)):
            seq = sorted(seq)
        parts = [
            _canonical_bytes(v).decode("utf-8", "ignore")
            for v in seq
        ]
        return ("[" + ",".join(parts) + "]").encode("utf-8")
    return repr(value).encode("utf-8")


def _blake2_hexdigest(*parts: Any) -> str:
    h = hashlib.blake2b(digest_size=16)
    for part in parts:
        h.update(_canonical_bytes(part))
    return h.hexdigest()


class TaskPoolProtocol:
    """Protocol-like base class for task pools.

    The pool must implement ``submit_task(task: TaskRecord) -> None``.  A simple
    in-memory implementation is provided via :class:`InMemoryTaskPool`.
    """

    def submit_task(self, task: "TaskRecord") -> None:  # pragma: no cover - documentation hook
        raise NotImplementedError


@dataclass(frozen=True)
class PartitionConstraints:
    group_by: Optional[Union[str, Callable[[Mapping[str, Any]], Any]]] = None
    max_tasks: Optional[int] = None
    max_items_per_task: Optional[int] = None
    max_weight_per_task: Optional[float] = None
    shuffle_seed: Optional[int] = None
    affinity: Optional[Mapping[str, Any]] = None
    anti_affinity: Optional[Mapping[str, Any]] = None

    @staticmethod
    def from_mapping(data: Optional[Mapping[str, Any]]) -> "PartitionConstraints":
        if data is None:
            return PartitionConstraints()
        kwargs: Dict[str, Any] = dict(data)
        return PartitionConstraints(**kwargs)


class PartitionStrategy(str):
    FIXED = "fixed"
    WEIGHTED = "weighted"
    HASH = "hash"
    GROUP_AWARE = "group-aware"

    _ALIASES = {
        "fixed-size": FIXED,
        "fixed-size chunk": FIXED,
        "chunk": FIXED,
        "weight": WEIGHTED,
        "weight-balanced": WEIGHTED,
        "lpt": WEIGHTED,
        "greedy": WEIGHTED,
        "hash": HASH,
        "mod": HASH,
        "modulo": HASH,
        "stable": HASH,
        "domain": GROUP_AWARE,
        "family": GROUP_AWARE,
    }

    @classmethod
    def parse(cls, value: Union[str, "PartitionStrategy"]) -> "PartitionStrategy":
        if isinstance(value, PartitionStrategy):
            return value
        key = value.strip().lower()
        if key in (cls.FIXED, cls.WEIGHTED, cls.HASH, cls.GROUP_AWARE):
            return cls(key)
        if key in cls._ALIASES:
            return cls(cls._ALIASES[key])
        raise ValueError(f"Unknown strategy: {value}")


def _default_constraints_from_config() -> Mapping[str, Any]:
    try:
        from .config import Config  # type: ignore
        cfg = Config.get_singleton()
    except Exception:
        return {}
    with contextlib.suppress(Exception):
        defaults = cfg.get("task_partitioner.defaults", dict, default={})
        if isinstance(defaults, Mapping):
            return defaults
    return {}


def _default_meta_from_config() -> Mapping[str, Any]:
    try:
        from .config import Config  # type: ignore
        cfg = Config.get_singleton()
    except Exception:
        return {}
    with contextlib.suppress(Exception):
        meta = cfg.get("task_partitioner.meta", dict, default={})
        if isinstance(meta, Mapping):
            return meta
    return {}


@dataclass(frozen=True)
class ItemRecord:
    payload_ref: Any
    weight: float
    group_key: Optional[str]
    metadata: Mapping[str, Any]
    checksum: Optional[str]
    raw: Any
    index: int


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    job_id: str
    attempt: int
    payload_ref: Tuple[Any, ...]
    weight: float
    group_keys: Tuple[str, ...]
    checksum: Optional[str]
    priority: Optional[int] = None
    deadline: Optional[str] = None
    extras: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "job_id": self.job_id,
            "attempt": self.attempt,
            "payload_ref": list(self.payload_ref),
            "weight": self.weight,
            "group_keys": list(self.group_keys),
            "checksum": self.checksum,
            "priority": self.priority,
            "deadline": self.deadline,
            "extras": dict(self.extras),
        }


@dataclass(frozen=True)
class PlanStatistics:
    task_count: int
    total_weight: float
    max_weight: float
    min_weight: float
    gini_coefficient: float
    group_crossing: bool


@dataclass(frozen=True)
class Plan:
    job_spec: Mapping[str, Any]
    strategy: PartitionStrategy
    constraints: PartitionConstraints
    tasks: Tuple[TaskRecord, ...]
    statistics: PlanStatistics
    meta: Mapping[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_spec.get("job_id"),
            "strategy": str(self.strategy),
            "tasks": [task.to_dict() for task in self.tasks],
            "statistics": dataclasses.asdict(self.statistics),
            "constraints": dataclasses.asdict(self.constraints),
            "meta": dict(self.meta),
        }

    def ensure_valid(self) -> None:
        if not self.tasks:
            raise ValueError("Plan has no tasks")
        job_id = self.job_spec.get("job_id")
        if job_id is None:
            raise ValueError("job_spec must define job_id")
        seen = set()
        for task in self.tasks:
            if task.task_id in seen:
                raise ValueError("Duplicate task_id detected")
            seen.add(task.task_id)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_group_key(item: Mapping[str, Any], constraints: PartitionConstraints) -> Optional[str]:
    if constraints.group_by is None:
        return item.get("group_key") if isinstance(item, Mapping) else None
    if callable(constraints.group_by):
        return str(constraints.group_by(item))
    key = constraints.group_by
    if isinstance(item, Mapping) and key in item:
        return str(item[key])
    if isinstance(item, Mapping):
        meta = item.get("metadata")
        if isinstance(meta, Mapping) and key in meta:
            return str(meta[key])
    return item.get("group_key") if isinstance(item, Mapping) else None


def _normalise_items(items: Iterable[Any], constraints: PartitionConstraints) -> List[ItemRecord]:
    normalised: List[ItemRecord] = []
    for idx, raw in enumerate(items):
        if isinstance(raw, ItemRecord):
            normalised.append(raw)
            continue
        if isinstance(raw, Mapping):
            payload_ref = raw.get("payload_ref", raw.get("ref", raw.get("id", raw)))
            weight = float(raw.get("weight", 1.0))
            meta = raw.get("metadata", {})
            checksum = raw.get("checksum")
            extras = raw.get("extras", {})
            combined_meta: Dict[str, Any] = {}
            if isinstance(meta, Mapping):
                combined_meta.update(meta)
            if isinstance(extras, Mapping):
                combined_meta.update(extras)
            group_key = raw.get("group_key")
        else:
            payload_ref = raw
            weight = 1.0
            combined_meta = {}
            checksum = None
            group_key = None
        if weight < 0:
            raise ValueError("weight must be non-negative")
        data_for_group = raw if isinstance(raw, Mapping) else {}
        resolved_group = _resolve_group_key(data_for_group, constraints)
        if resolved_group is not None:
            group_key = resolved_group
        if group_key is None:
            group_key = f"__item_{idx}"
        checksum_val = checksum or _blake2_hexdigest(payload_ref, weight, group_key)
        normalised.append(
            ItemRecord(
                payload_ref=payload_ref,
                weight=weight,
                group_key=group_key,
                metadata=combined_meta,
                checksum=checksum_val,
                raw=raw,
                index=idx,
            )
        )
    return normalised


def _group_items(items: Sequence[ItemRecord]) -> List[Tuple[Optional[str], List[ItemRecord]]]:
    grouped: Dict[Optional[str], List[ItemRecord]] = collections.OrderedDict()
    for item in items:
        grouped.setdefault(item.group_key, []).append(item)
    return [(k, grouped[k]) for k in grouped]


def _compute_gini(values: Sequence[float]) -> float:
    filtered = [v for v in values if v >= 0]
    if not filtered:
        return 0.0
    sorted_vals = sorted(filtered)
    cum = 0.0
    for i, val in enumerate(sorted_vals, 1):
        cum += i * val
    total = sum(sorted_vals)
    n = len(sorted_vals)
    if total == 0:
        return 0.0
    return (2 * cum) / (n * total) - (n + 1) / n


def _ensure_task_limits(builder: "_TaskBuilder", constraints: PartitionConstraints) -> None:
    if constraints.max_items_per_task is not None and builder.item_count > constraints.max_items_per_task:
        raise ValueError("Task exceeds max_items_per_task")
    if constraints.max_weight_per_task is not None and builder.weight > constraints.max_weight_per_task + 1e-9:
        raise ValueError("Task exceeds max_weight_per_task")


@dataclass
class _TaskBuilder:
    job_id: str
    attempt: int
    constraints: PartitionConstraints
    seed: int
    index: int
    items: List[ItemRecord] = field(default_factory=list)
    weight: float = 0.0

    def add_group(self, group_items: Sequence[ItemRecord]) -> None:
        self.items.extend(group_items)
        self.weight += sum(i.weight for i in group_items)
        _ensure_task_limits(self, self.constraints)

    @property
    def item_count(self) -> int:
        return len(self.items)

    def build(self) -> TaskRecord:
        payload_refs = tuple(item.payload_ref for item in self.items)
        group_keys = tuple(sorted({i.group_key for i in self.items if i.group_key is not None}))
        checksum = _blake2_hexdigest(payload_refs, self.weight, group_keys)
        task_id = _blake2_hexdigest(self.job_id, self.index, self.seed)
        extras: Dict[str, Any] = {
            "item_indexes": [item.index for item in self.items],
        }
        if self.constraints.affinity:
            extras["affinity"] = dict(self.constraints.affinity)
        if self.constraints.anti_affinity:
            extras["anti_affinity"] = dict(self.constraints.anti_affinity)
        extras["checksum_inputs"] = [item.checksum for item in self.items]
        extras["items"] = [item.raw for item in self.items]
        return TaskRecord(
            task_id=task_id,
            job_id=self.job_id,
            attempt=0,
            payload_ref=payload_refs,
            weight=self.weight,
            group_keys=group_keys,
            checksum=checksum,
            extras=extras,
        )


def _build_plan(job_spec: Mapping[str, Any], builders: Sequence[_TaskBuilder], strategy: PartitionStrategy, constraints: PartitionConstraints, meta: Optional[Mapping[str, Any]] = None) -> Plan:
    tasks = tuple(builder.build() for builder in builders if builder.item_count > 0)
    weights = [task.weight for task in tasks]
    stats = PlanStatistics(
        task_count=len(tasks),
        total_weight=sum(weights),
        max_weight=max(weights) if weights else 0.0,
        min_weight=min(weights) if weights else 0.0,
        gini_coefficient=_compute_gini(weights),
        group_crossing=_detect_group_crossing(tasks),
    )
    plan = Plan(
        job_spec=job_spec,
        strategy=strategy,
        constraints=constraints,
        tasks=tasks,
        statistics=stats,
        meta=meta or {},
    )
    plan.ensure_valid()
    return plan


def _detect_group_crossing(tasks: Sequence[TaskRecord]) -> bool:
    group_to_task: Dict[str, str] = {}
    for task in tasks:
        for group in task.group_keys:
            if group in group_to_task and group_to_task[group] != task.task_id:
                return True
            group_to_task[group] = task.task_id
    return False


def _estimate_task_count(total_items: int, total_weight: float, constraints: PartitionConstraints) -> int:
    if constraints.max_tasks:
        return constraints.max_tasks
    counts: List[int] = []
    if constraints.max_items_per_task:
        counts.append(math.ceil(total_items / constraints.max_items_per_task))
    if constraints.max_weight_per_task and constraints.max_weight_per_task > 0:
        counts.append(math.ceil(total_weight / constraints.max_weight_per_task))
    if not counts:
        return max(1, min(32, total_items))
    return max(1, max(counts))


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------


def _plan_fixed(
    job_spec: Mapping[str, Any],
    grouped_items: Sequence[Tuple[Optional[str], List[ItemRecord]]],
    constraints: PartitionConstraints,
    seed: int,
    progress: Optional[ProgressController] = None,
) -> List[_TaskBuilder]:
    builders: List[_TaskBuilder] = []
    current = _TaskBuilder(job_id=str(job_spec["job_id"]), attempt=0, constraints=constraints, seed=seed, index=0)
    for group_key, items in grouped_items:
        group_weight = sum(item.weight for item in items)
        group_len = len(items)
        if constraints.max_weight_per_task and group_weight > constraints.max_weight_per_task + 1e-9:
            raise ValueError(f"Group {group_key} weight exceeds max_weight_per_task")
        if constraints.max_items_per_task and group_len > constraints.max_items_per_task:
            raise ValueError(f"Group {group_key} size exceeds max_items_per_task")
        projected_weight = current.weight + group_weight
        projected_items = current.item_count + group_len
        should_flush = False
        if current.item_count > 0:
            if constraints.max_weight_per_task and projected_weight > constraints.max_weight_per_task + 1e-9:
                should_flush = True
            if constraints.max_items_per_task and projected_items > constraints.max_items_per_task:
                should_flush = True
        if should_flush:
            builders.append(current)
            current = _TaskBuilder(job_id=str(job_spec["job_id"]), attempt=0, constraints=constraints, seed=seed, index=len(builders))
        current.add_group(items)
        if progress:
            progress.advance(1)
    if current.item_count > 0 or not builders:
        builders.append(current)
    if constraints.max_tasks and len(builders) > constraints.max_tasks:
        raise ValueError("Planned tasks exceed max_tasks constraint")
    return builders


def _plan_weighted(
    job_spec: Mapping[str, Any],
    grouped_items: Sequence[Tuple[Optional[str], List[ItemRecord]]],
    constraints: PartitionConstraints,
    seed: int,
    progress: Optional[ProgressController] = None,
) -> List[_TaskBuilder]:
    job_id = str(job_spec["job_id"])
    totals = [sum(item.weight for item in group_items) for _, group_items in grouped_items]
    total_weight = sum(totals)
    total_items = sum(len(group) for _, group in grouped_items)
    task_count = _estimate_task_count(total_items, total_weight, constraints)
    if constraints.max_tasks and task_count > constraints.max_tasks:
        task_count = constraints.max_tasks
    builders = [
        _TaskBuilder(job_id=job_id, attempt=0, constraints=constraints, seed=seed, index=i)
        for i in range(task_count)
    ]
    ordered_groups = sorted(
        zip(grouped_items, totals),
        key=lambda pair: (pair[1], pair[0][0] if pair[0][0] is not None else ""),
        reverse=True,
    )
    for (group_key, items), group_weight in ordered_groups:
        if constraints.max_weight_per_task and group_weight > constraints.max_weight_per_task + 1e-9:
            raise ValueError(f"Group {group_key} weight exceeds max_weight_per_task")
        placed = False
        builders_sorted = sorted(builders, key=lambda b: (b.weight, b.item_count))
        for builder in builders_sorted:
            projected_weight = builder.weight + group_weight
            projected_items = builder.item_count + len(items)
            if constraints.max_weight_per_task and projected_weight > constraints.max_weight_per_task + 1e-9:
                continue
            if constraints.max_items_per_task and projected_items > constraints.max_items_per_task:
                continue
            builder.add_group(items)
            placed = True
            break
        if not placed:
            if constraints.max_tasks and len(builders) >= constraints.max_tasks:
                raise ValueError("No available task slot for group; max_tasks reached")
            new_index = len(builders)
            builder = _TaskBuilder(job_id=job_id, attempt=0, constraints=constraints, seed=seed, index=new_index)
            builder.add_group(items)
            builders.append(builder)
        if progress:
            progress.advance(1)
    return builders


def _plan_hash(
    job_spec: Mapping[str, Any],
    grouped_items: Sequence[Tuple[Optional[str], List[ItemRecord]]],
    constraints: PartitionConstraints,
    seed: int,
    progress: Optional[ProgressController] = None,
) -> List[_TaskBuilder]:
    if not constraints.max_tasks or constraints.max_tasks <= 0:
        raise ValueError("Hash strategy requires max_tasks to be specified and > 0")
    job_id = str(job_spec["job_id"])
    builders = [
        _TaskBuilder(job_id=job_id, attempt=0, constraints=constraints, seed=seed, index=i)
        for i in range(constraints.max_tasks)
    ]
    for group_key, items in grouped_items:
        key_material = group_key if group_key is not None else tuple(item.payload_ref for item in items)
        digest = int(_blake2_hexdigest(seed, key_material, job_id), 16)
        index = digest % len(builders)
        builder = builders[index]
        projected_weight = builder.weight + sum(item.weight for item in items)
        projected_items = builder.item_count + len(items)
        if constraints.max_weight_per_task and projected_weight > constraints.max_weight_per_task + 1e-9:
            raise ValueError("Hash bucket would exceed max_weight_per_task")
        if constraints.max_items_per_task and projected_items > constraints.max_items_per_task:
            raise ValueError("Hash bucket would exceed max_items_per_task")
        builder.add_group(items)
        if progress:
            progress.advance(1)
    return builders


def _plan_group_aware(
    job_spec: Mapping[str, Any],
    grouped_items: Sequence[Tuple[Optional[str], List[ItemRecord]]],
    constraints: PartitionConstraints,
    seed: int,
    progress: Optional[ProgressController] = None,
) -> List[_TaskBuilder]:
    # Group-aware strategy first clusters by group key and then applies weighted placement.
    return _plan_weighted(job_spec, grouped_items, constraints, seed, progress)


_STRATEGY_IMPL = {
    PartitionStrategy.FIXED: _plan_fixed,
    PartitionStrategy.WEIGHTED: _plan_weighted,
    PartitionStrategy.HASH: _plan_hash,
    PartitionStrategy.GROUP_AWARE: _plan_group_aware,
}


# ---------------------------------------------------------------------------
# Task partitioner
# ---------------------------------------------------------------------------


class InMemoryTaskPool(TaskPoolProtocol):
    """Simple in-memory task pool used for tests and defaults."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.tasks: List[TaskRecord] = []

    def submit_task(self, task: TaskRecord) -> None:
        with self._lock:
            self.tasks.append(task)


class TaskPartitioner:
    """Partitioner service implementing the plan/emit/reshard contract."""

    @classmethod
    def plan(
        cls,
        job_spec: Mapping[str, Any],
        items: Iterable[Any],
        strategy: Union[str, PartitionStrategy],
        constraints: Optional[Union[PartitionConstraints, Mapping[str, Any]]] = None,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Plan:
        if "job_id" not in job_spec:
            raise ValueError("job_spec must include 'job_id'")
        parsed_constraints = constraints
        if isinstance(parsed_constraints, Mapping) and not isinstance(parsed_constraints, PartitionConstraints):
            parsed_constraints = PartitionConstraints.from_mapping(parsed_constraints)
        if parsed_constraints is None:
            defaults = _default_constraints_from_config()
            parsed_constraints = PartitionConstraints.from_mapping(defaults)
        strategy_obj = PartitionStrategy.parse(strategy)
        seed = parsed_constraints.shuffle_seed if parsed_constraints.shuffle_seed is not None else 0
        items_normalised = _normalise_items(items, parsed_constraints)
        if parsed_constraints.shuffle_seed is not None:
            rnd = random.Random(parsed_constraints.shuffle_seed)
            rnd.shuffle(items_normalised)
        grouped = _group_items(items_normalised)
        impl = _STRATEGY_IMPL[strategy_obj]
        cfg_meta = dict(_default_meta_from_config())
        if metadata:
            cfg_meta.update(metadata)
        source_items = [
            {
                "payload_ref": item.payload_ref,
                "weight": item.weight,
                "group_key": item.group_key,
                "metadata": dict(item.metadata),
                "checksum": item.checksum,
                "index": item.index,
            }
            for item in items_normalised
        ]
        cfg_meta.setdefault("source_items", source_items)
        history = list(cfg_meta.get("strategy_history", []))
        history.append(str(strategy_obj))
        cfg_meta["strategy_history"] = history
        progress: Optional[ProgressController] = None
        if grouped:
            progress = ProgressController(total_units=len(grouped), description="TaskPartitioner")
            progress.start()
        try:
            builders = impl(job_spec, grouped, parsed_constraints, seed, progress)
        finally:
            if progress is not None:
                progress.close()
        plan = _build_plan(job_spec, builders, strategy_obj, parsed_constraints, cfg_meta)
        return plan

    @classmethod
    def emit(cls, plan: Plan, sink: Optional[TaskPoolProtocol] = None) -> TaskPoolProtocol:
        if sink is None:
            sink = InMemoryTaskPool()
        for task in plan.tasks:
            sink.submit_task(task)
        return sink

    @classmethod
    def reshard(
        cls,
        plan: Plan,
        reason: str,
        *,
        strategy: Optional[Union[str, PartitionStrategy]] = None,
        constraints_override: Optional[Mapping[str, Any]] = None,
    ) -> Plan:
        new_constraints = dataclasses.asdict(plan.constraints)
        if constraints_override:
            new_constraints.update(constraints_override)
        new_strategy = strategy or plan.strategy
        metadata = dict(plan.meta)
        metadata.setdefault("reshard_events", []).append(reason)
        source_items = metadata.get("source_items") or plan.meta.get("source_items")
        if not source_items:
            raise ValueError("Plan metadata missing source_items; cannot reshard deterministically")
        simplified_items = [
            {
                "payload_ref": entry["payload_ref"],
                "weight": entry.get("weight", 1.0),
                "group_key": entry.get("group_key"),
                "metadata": entry.get("metadata", {}),
                "checksum": entry.get("checksum"),
            }
            for entry in source_items
        ]
        return cls.plan(
            plan.job_spec,
            simplified_items,
            new_strategy,
            PartitionConstraints.from_mapping(new_constraints),
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _build_sample_items() -> List[Dict[str, Any]]:
    return [
        {"payload_ref": f"item-{i}", "weight": float((i % 3) + 1), "group": "A" if i < 4 else "B"}
        for i in range(8)
    ]


def _annotate_group(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result = []
    for item in items:
        data = dict(item)
        data["group_key"] = data.get("group")
        result.append(data)
    return result


def _run_basic_tests() -> None:
    import unittest

    class TaskPartitionerTestCase(unittest.TestCase):
        def setUp(self) -> None:
            self.job_spec = {"job_id": "job-123"}
            self.items = _annotate_group(_build_sample_items())

        def test_fixed_strategy_respects_group(self) -> None:
            plan = TaskPartitioner.plan(
                self.job_spec,
                self.items,
                "fixed-size chunk",
                {"max_items_per_task": 5, "group_by": "group_key"},
            )
            self.assertGreaterEqual(len(plan.tasks), 2)
            seen_groups = [tuple(task.group_keys) for task in plan.tasks]
            for groups in seen_groups:
                self.assertLessEqual(len(groups), 1)

        def test_weighted_strategy_balances(self) -> None:
            plan = TaskPartitioner.plan(
                self.job_spec,
                self.items,
                "weighted",
                {"max_tasks": 3, "group_by": "group_key", "shuffle_seed": 7},
            )
            weights = [task.weight for task in plan.tasks]
            self.assertAlmostEqual(sum(weights), sum(item["weight"] for item in self.items))
            self.assertLess(plan.statistics.gini_coefficient, 0.3)

        def test_hash_strategy_is_deterministic(self) -> None:
            constraints = {"max_tasks": 4, "group_by": "group_key", "shuffle_seed": 11}
            plan1 = TaskPartitioner.plan(self.job_spec, self.items, "hash", constraints)
            plan2 = TaskPartitioner.plan(self.job_spec, self.items, "hash", constraints)
            self.assertEqual([t.payload_ref for t in plan1.tasks], [t.payload_ref for t in plan2.tasks])

        def test_emit_collects_tasks(self) -> None:
            plan = TaskPartitioner.plan(
                self.job_spec,
                self.items,
                "weighted",
                {"max_tasks": 2, "group_by": "group_key"},
            )
            pool = TaskPartitioner.emit(plan)
            self.assertEqual(len(pool.tasks), len(plan.tasks))

    unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(TaskPartitionerTestCase))


if __name__ == "__main__":
    _run_basic_tests()

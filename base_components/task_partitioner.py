#!/usr/bin/env python3
"""Task partitioning service.

This module implements the "任务分割类" described in the specification. The
design emphasises deterministic streaming、strong observability and
configurable-yet-safe defaults。``TaskPartitioner`` 通过 ``iter_tasks``
持续输出 ``TaskRecord``，调用方可在生成后立即提交到任务池，实现边计划边消费的
流水模式。

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

import contextlib
import dataclasses
import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


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
    """Protocol-like base class for legacy emit semantics."""

    def submit_task(self, task: "TaskRecord") -> None:  # pragma: no cover - documentation hook
        raise NotImplementedError


@dataclass(frozen=True)
class PartitionConstraints:
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
    }

    @classmethod
    def parse(cls, value: Union[str, "PartitionStrategy"]) -> "PartitionStrategy":
        if isinstance(value, PartitionStrategy):
            return value
        key = value.strip().lower()
        if key in (cls.FIXED, cls.WEIGHTED, cls.HASH):
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise_items(items: Iterable[Any]) -> Iterable[ItemRecord]:
    for idx, raw in enumerate(items):
        if isinstance(raw, ItemRecord):
            yield raw
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
        else:
            payload_ref = raw
            weight = 1.0
            combined_meta = {}
            checksum = None
        if weight < 0:
            raise ValueError("weight must be non-negative")
        checksum_val = checksum or _blake2_hexdigest(payload_ref, weight, idx)
        yield (
            ItemRecord(
                payload_ref=payload_ref,
                weight=weight,
                metadata=combined_meta,
                checksum=checksum_val,
                raw=raw,
                index=idx,
            )
        )


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

    def can_accept(self, item: ItemRecord) -> bool:
        projected_weight = self.weight + item.weight
        projected_items = len(self.items) + 1
        if self.constraints.max_weight_per_task is not None and projected_weight > self.constraints.max_weight_per_task + 1e-9:
            return False
        if self.constraints.max_items_per_task is not None and projected_items > self.constraints.max_items_per_task:
            return False
        return True

    def add_item(self, item: ItemRecord) -> None:
        self.items.append(item)
        self.weight += item.weight
        _ensure_task_limits(self, self.constraints)

    @property
    def item_count(self) -> int:
        return len(self.items)

    def build(self) -> TaskRecord:
        payload_refs = tuple(item.payload_ref for item in self.items)
        checksum = _blake2_hexdigest(payload_refs, self.weight, self.index)
        task_id = _blake2_hexdigest(self.job_id, self.index, self.seed)
        extras: Dict[str, Any] = {
            "item_indexes": [item.index for item in self.items],
            "item_weights": [item.weight for item in self.items],
            "item_metadata": [dict(item.metadata) for item in self.items],
        }
        if self.constraints.affinity:
            extras["affinity"] = dict(self.constraints.affinity)
        if self.constraints.anti_affinity:
            extras["anti_affinity"] = dict(self.constraints.anti_affinity)
        extras["checksum_inputs"] = [item.checksum for item in self.items]
        extras["item_checksums"] = [item.checksum for item in self.items]
        return TaskRecord(
            task_id=task_id,
            job_id=self.job_id,
            attempt=0,
            payload_ref=payload_refs,
            weight=self.weight,
            group_keys=tuple(),
            checksum=checksum,
            extras=extras,
        )


def _validate_item_against_constraints(item: ItemRecord, constraints: PartitionConstraints) -> None:
    if constraints.max_weight_per_task is not None and item.weight > constraints.max_weight_per_task + 1e-9:
        raise ValueError("Single item weight exceeds max_weight_per_task constraint")
    if constraints.max_items_per_task is not None and constraints.max_items_per_task <= 0:
        raise ValueError("max_items_per_task must be positive when specified")


def _build_stream_task(job_id: str, items: Sequence[ItemRecord], index: int, seed: int) -> TaskRecord:
    payload_refs = tuple(item.payload_ref for item in items)
    weight = sum(item.weight for item in items)
    checksum = _blake2_hexdigest(payload_refs, weight, index, seed)
    extras = {
        "item_indexes": [item.index for item in items],
        "item_weights": [item.weight for item in items],
        "item_metadata": [dict(item.metadata) for item in items],
        "item_checksums": [item.checksum for item in items],
    }
    task_id = _blake2_hexdigest(job_id, "stream", index, seed)
    return TaskRecord(
        task_id=task_id,
        job_id=job_id,
        attempt=0,
        payload_ref=payload_refs,
        weight=weight,
        group_keys=tuple(),
        checksum=checksum,
        extras=extras,
    )


def _prepare_builders(
    job_spec: Mapping[str, Any],
    items: Iterable[Any],
    strategy: Union[str, PartitionStrategy],
    constraints: Optional[Union[PartitionConstraints, Mapping[str, Any]]],
    metadata: Optional[Mapping[str, Any]],
) -> Tuple[List[_TaskBuilder], PartitionConstraints, PartitionStrategy, Mapping[str, Any]]:
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
    items_iter: Iterable[ItemRecord] = _normalise_items(items)
    if parsed_constraints.shuffle_seed is not None:
        buffered = list(items_iter)
        rnd = random.Random(parsed_constraints.shuffle_seed)
        rnd.shuffle(buffered)
        items_iter = buffered
    impl = _STRATEGY_IMPL[strategy_obj]
    cfg_meta = dict(_default_meta_from_config())
    if metadata:
        cfg_meta.update(metadata)
    history = list(cfg_meta.get("strategy_history", []))
    history.append(str(strategy_obj))
    cfg_meta["strategy_history"] = history
    builders = impl(job_spec, items_iter, parsed_constraints, seed)
    return builders, parsed_constraints, strategy_obj, cfg_meta


# ---------------------------------------------------------------------------
# Task partitioner
# ---------------------------------------------------------------------------


class TaskPartitioner:
    """Streaming task partitioner."""

    @classmethod
    def iter_tasks(
        cls,
        job_spec: Mapping[str, Any],
        items: Iterable[Any],
        strategy: Union[str, PartitionStrategy],
        constraints: Optional[Union[PartitionConstraints, Mapping[str, Any]]] = None,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Iterable[TaskRecord]:
        _ = strategy, metadata
        if "job_id" not in job_spec:
            raise ValueError("job_spec must include 'job_id'")
        parsed_constraints = constraints
        if isinstance(parsed_constraints, Mapping) and not isinstance(parsed_constraints, PartitionConstraints):
            parsed_constraints = PartitionConstraints.from_mapping(parsed_constraints)
        if parsed_constraints is None:
            defaults = _default_constraints_from_config()
            parsed_constraints = PartitionConstraints.from_mapping(defaults)
        job_id = str(job_spec["job_id"])
        seed = parsed_constraints.shuffle_seed if parsed_constraints.shuffle_seed is not None else 0
        rng = random.Random(seed) if parsed_constraints.shuffle_seed is not None else None
        max_items = parsed_constraints.max_items_per_task or 0
        max_weight = parsed_constraints.max_weight_per_task

        buffer: List[ItemRecord] = []
        weight = 0.0
        index = 0
        for item in _normalise_items(items):
            _validate_item_against_constraints(item, parsed_constraints)
            buffer.append(item)
            weight += item.weight
            flush = False
            if max_items and len(buffer) >= max_items:
                flush = True
            if max_weight is not None and weight >= max_weight - 1e-9:
                flush = True
            if flush:
                batch = list(buffer)
                if rng is not None:
                    rng.shuffle(batch)
                yield _build_stream_task(job_id, batch, index, seed)
                index += 1
                buffer.clear()
                weight = 0.0
        if buffer:
            batch = list(buffer)
            if rng is not None:
                rng.shuffle(batch)
            yield _build_stream_task(job_id, batch, index, seed)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _build_sample_items() -> List[Dict[str, Any]]:
    return [
        {"payload_ref": f"item-{i}", "weight": float((i % 4) + 1)}
        for i in range(12)
    ]


def _run_basic_tests() -> None:
    import unittest

    class TaskPartitionerTestCase(unittest.TestCase):
        def setUp(self) -> None:
            self.job_spec = {"job_id": "job-123"}
            self.items = _build_sample_items()

        def test_iter_chunks(self) -> None:
            tasks = list(TaskPartitioner.iter_tasks(self.job_spec, self.items, "fixed", {"max_items_per_task": 3}))
            self.assertTrue(all(len(task.payload_ref) <= 3 for task in tasks))

        def test_iter_respects_weight(self) -> None:
            tasks = list(TaskPartitioner.iter_tasks(self.job_spec, self.items, "fixed", {"max_weight_per_task": 5.0}))
            self.assertTrue(all(task.weight <= 5.0 + 1e-9 for task in tasks))

    unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(TaskPartitionerTestCase))


if __name__ == "__main__":
    _run_basic_tests()

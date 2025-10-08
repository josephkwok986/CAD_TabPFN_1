"""Topology-to-Table S2 pipeline with task system integration."""
from __future__ import annotations

import argparse
import csv
import os
import hashlib
import json
import math
import random
import re
import sys
import threading
import time
import dataclasses
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    import pandas as pd  # type: ignore
except ModuleNotFoundError:
    pd = None  # type: ignore

from config import Config
from logger import StructuredLogger
from parallel_executor import ExecutorEvents, ExecutorPolicy, ParallelExecutor, TaskResult
from task_partitioner import (
    ItemRecord,
    Plan,
    PlanStatistics,
    PartitionConstraints,
    PartitionStrategy,
    TaskPartitioner,
    TaskRecord,
)
from task_pool import LeasedTask, TaskPool
from task_system_config import ensure_task_config

from .settings import S2PipelineConfig, SourceSettings

from tqdm import tqdm

try:  # Optional dependency for clustering
    from sklearn.cluster import DBSCAN  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    DBSCAN = None  # type: ignore

try:  # Optional GPU acceleration
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore


@dataclass(frozen=True)
class PartCandidate:
    dataset: str
    dataset_tier: str
    root: Path
    path: Path
    rel_path: str
    part_id: str
    file_size: int
    modified_ns: int


@dataclass(frozen=True)
class PartFeatureRecord:
    part_id: str
    dataset: str
    dataset_tier: str
    rel_path: str
    abs_path: str
    file_size: int
    modified_utc: str
    content_hash: str
    geom_hash: str
    descriptor_vector: Tuple[float, ...]
    descriptor_norm: float
    bbox_ratios: Tuple[float, float]


@dataclass(frozen=True)
class S2Paths:
    root: Path
    signatures: Path
    part_index: Path
    part_index_for_split: Path
    family_hist: Path
    diagnostics: Path
    cache_root: Path
    discovery_cache: Path
    discovery_meta: Path
    plan_cache: Path
    plan_meta: Path
    execution_cache: Path
    execution_meta: Path
    dataframe_cache: Path
    dataframe_meta: Path
    enrichment_cache: Path
    enrichment_meta: Path

    @classmethod
    def from_config(cls, config: S2PipelineConfig) -> "S2Paths":
        outputs = config.outputs
        root = outputs.root
        cache_root = root / "_cache"
        return cls(
            root=root,
            signatures=root / outputs.signatures_file,
            part_index=root / outputs.part_index_file,
            part_index_for_split=root / outputs.part_index_for_split_file,
            family_hist=root / outputs.family_hist_file,
            diagnostics=root / outputs.diagnostics_file,
            cache_root=cache_root,
            discovery_cache=cache_root / "discovery.csv",
            discovery_meta=cache_root / "discovery.meta.json",
            plan_cache=cache_root / "plan.csv",
            plan_meta=cache_root / "plan.meta.json",
            execution_cache=cache_root / "execution_records.csv",
            execution_meta=cache_root / "execution_records.meta.json",
            dataframe_cache=cache_root / "dataframe.csv",
            dataframe_meta=cache_root / "dataframe.meta.json",
            enrichment_cache=cache_root / "enriched.csv",
            enrichment_meta=cache_root / "enriched.meta.json",
        )


def _candidate_to_dict(candidate: PartCandidate) -> Dict[str, object]:
    return {
        "dataset": candidate.dataset,
        "dataset_tier": candidate.dataset_tier,
        "root": str(candidate.root),
        "path": str(candidate.path),
        "rel_path": candidate.rel_path,
        "part_id": candidate.part_id,
        "file_size": int(candidate.file_size),
        "modified_ns": int(candidate.modified_ns),
    }


def _candidate_from_dict(data: Mapping[str, object]) -> PartCandidate:
    root = Path(str(data["root"]))
    path = Path(str(data["path"]))
    return PartCandidate(
        dataset=str(data["dataset"]),
        dataset_tier=str(data["dataset_tier"]),
        root=root,
        path=path,
        rel_path=str(data["rel_path"]),
        part_id=str(data["part_id"]),
        file_size=int(data["file_size"]),
        modified_ns=int(data["modified_ns"]),
    )


def _task_record_to_dict(task: TaskRecord) -> Dict[str, Any]:
    data = task.to_dict()
    extras = data.get("extras", {})
    items = extras.get("items")
    if isinstance(items, list):
        serialised_items = []
        for item in items:
            if isinstance(item, Mapping) and "candidate" in item:
                candidate = item["candidate"]
                if isinstance(candidate, PartCandidate):
                    serialised_items.append({"candidate": _candidate_to_dict(candidate)})
                    continue
            serialised_items.append(item)
        extras["items"] = serialised_items
    data["extras"] = extras
    return data


def _task_record_from_dict(data: Mapping[str, Any]) -> TaskRecord:
    extras = dict(data.get("extras", {}))
    items = extras.get("items")
    if isinstance(items, list):
        restored_items = []
        for item in items:
            if isinstance(item, Mapping) and "candidate" in item and isinstance(item["candidate"], Mapping):
                restored_items.append({"candidate": _candidate_from_dict(item["candidate"])})
            else:
                restored_items.append(item)
        extras["items"] = restored_items
    return TaskRecord(
        task_id=str(data["task_id"]),
        job_id=str(data["job_id"]),
        attempt=int(data.get("attempt", 0)),
        payload_ref=tuple(data.get("payload_ref", [])),
        weight=float(data.get("weight", 0.0)),
        group_keys=tuple(data.get("group_keys", [])),
        checksum=data.get("checksum"),
        priority=data.get("priority"),
        deadline=data.get("deadline"),
        extras=extras,
    )


def _record_to_dict(record: PartFeatureRecord) -> Dict[str, object]:
    return {
        "part_id": record.part_id,
        "dataset": record.dataset,
        "dataset_tier": record.dataset_tier,
        "rel_path": record.rel_path,
        "abs_path": record.abs_path,
        "file_size": record.file_size,
        "modified_utc": record.modified_utc,
        "content_hash": record.content_hash,
        "geom_hash": record.geom_hash,
        "descriptor_vector": list(record.descriptor_vector),
        "descriptor_norm": record.descriptor_norm,
        "bbox_ratios": list(record.bbox_ratios),
    }


def _record_from_dict(data: Mapping[str, Any]) -> PartFeatureRecord:
    return PartFeatureRecord(
        part_id=str(data["part_id"]),
        dataset=str(data["dataset"]),
        dataset_tier=str(data["dataset_tier"]),
        rel_path=str(data["rel_path"]),
        abs_path=str(data["abs_path"]),
        file_size=int(data["file_size"]),
        modified_utc=str(data["modified_utc"]),
        content_hash=str(data["content_hash"]),
        geom_hash=str(data["geom_hash"]),
        descriptor_vector=tuple(float(x) for x in data.get("descriptor_vector", [])),
        descriptor_norm=float(data["descriptor_norm"]),
        bbox_ratios=tuple(float(x) for x in data.get("bbox_ratios", [])),
    )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return str(value)


def _dataframe_to_records(df: "pd.DataFrame") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        rows.append({k: _json_safe(v) for k, v in row.to_dict().items()})
    return rows


def _records_to_dataframe(rows: Sequence[Mapping[str, Any]]) -> "pd.DataFrame":
    if pd is None:
        raise RuntimeError("pandas is required to load cached dataframe")
    return pd.DataFrame(list(rows))


class StageCacheManager:
    def __init__(self, paths: S2Paths) -> None:
        self._paths = paths

    def _ensure_root(self) -> None:
        self._paths.cache_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _load_meta(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def _save_meta(self, path: Path, meta: Mapping[str, Any]) -> None:
        self._ensure_root()
        with path.open("w", encoding="utf-8") as fh:
            json.dump(dict(meta), fh, ensure_ascii=False)

    def load_discovery(self) -> Optional[Tuple[List[PartCandidate], Optional[str]]]:
        path = self._paths.discovery_cache
        meta = self._load_meta(self._paths.discovery_meta)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            candidates = [
                PartCandidate(
                    dataset=row["dataset"],
                    dataset_tier=row["dataset_tier"],
                    root=Path(row["root"]),
                    path=Path(row["path"]),
                    rel_path=row["rel_path"],
                    part_id=row["part_id"],
                    file_size=int(row["file_size"]),
                    modified_ns=int(row["modified_ns"]),
                )
                for row in reader
            ]
        return candidates, meta.get("fingerprint")

    def save_discovery(self, candidates: Sequence[PartCandidate], fingerprint: str) -> None:
        self._ensure_root()
        fieldnames = [
            "dataset",
            "dataset_tier",
            "root",
            "path",
            "rel_path",
            "part_id",
            "file_size",
            "modified_ns",
        ]
        with self._paths.discovery_cache.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for candidate in candidates:
                writer.writerow(
                    {
                        "dataset": candidate.dataset,
                        "dataset_tier": candidate.dataset_tier,
                        "root": str(candidate.root),
                        "path": str(candidate.path),
                        "rel_path": candidate.rel_path,
                        "part_id": candidate.part_id,
                        "file_size": int(candidate.file_size),
                        "modified_ns": int(candidate.modified_ns),
                    }
                )
        meta = {
            "fingerprint": fingerprint,
            "count": len(candidates),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save_meta(self._paths.discovery_meta, meta)

    def load_plan(self, expected_input: str) -> Optional[Tuple[Plan, Optional[str]]]:
        path = self._paths.plan_cache
        meta = self._load_meta(self._paths.plan_meta)
        if not path.exists():
            return None
        if meta.get("input_fingerprint") != expected_input:
            return None
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            task_dicts: List[Dict[str, Any]] = []
            for row in reader:
                extras = json.loads(row["extras"]) if row.get("extras") else {}
                task_dicts.append(
                    {
                        "task_id": row["task_id"],
                        "job_id": row["job_id"],
                        "attempt": int(row["attempt"]) if row.get("attempt") else 0,
                        "payload_ref": json.loads(row["payload_ref"]) if row.get("payload_ref") else [],
                        "weight": float(row["weight"]) if row.get("weight") else 0.0,
                        "group_keys": json.loads(row["group_keys"]) if row.get("group_keys") else [],
                        "checksum": row.get("checksum"),
                        "priority": int(row["priority"]) if row.get("priority") else None,
                        "deadline": row.get("deadline") or None,
                        "extras": extras,
                    }
                )
        tasks = tuple(_task_record_from_dict(item) for item in task_dicts)
        constraints_data = meta.get("constraints", {})
        if isinstance(constraints_data.get("group_by"), dict):
            constraints_data["group_by"] = None
        constraints = PartitionConstraints.from_mapping(constraints_data)
        strategy = PartitionStrategy.parse(meta.get("strategy", "weighted"))
        stats_dict = meta.get("statistics", {})
        statistics = PlanStatistics(**stats_dict) if stats_dict else PlanStatistics(0, 0.0, 0.0, 0.0, 0.0, False)
        plan = Plan(
            job_spec=dict(meta.get("job_spec", {})),
            strategy=strategy,
            constraints=constraints,
            tasks=tasks,
            statistics=statistics,
            meta=dict(meta.get("plan_meta", {})),
        )
        return plan, meta.get("fingerprint")

    def save_plan(self, plan: Plan, fingerprint: str, input_fp: str) -> None:
        self._ensure_root()
        fieldnames = [
            "task_id",
            "job_id",
            "attempt",
            "payload_ref",
            "weight",
            "group_keys",
            "checksum",
            "priority",
            "deadline",
            "extras",
        ]
        with self._paths.plan_cache.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for task in plan.tasks:
                writer.writerow(
                    {
                        "task_id": task.task_id,
                        "job_id": task.job_id,
                        "attempt": task.attempt,
                        "payload_ref": json.dumps(list(task.payload_ref), ensure_ascii=False),
                        "weight": f"{task.weight:.12f}",
                        "group_keys": json.dumps(list(task.group_keys), ensure_ascii=False),
                        "checksum": task.checksum or "",
                        "priority": "" if task.priority is None else task.priority,
                        "deadline": task.deadline or "",
                        "extras": json.dumps(task.extras, ensure_ascii=False),
                    }
                )
        constraints_dict: Dict[str, Any] = {
            "group_by": plan.constraints.group_by if isinstance(plan.constraints.group_by, (str, type(None))) else None,
            "max_tasks": plan.constraints.max_tasks,
            "max_items_per_task": plan.constraints.max_items_per_task,
            "max_weight_per_task": plan.constraints.max_weight_per_task,
            "shuffle_seed": plan.constraints.shuffle_seed,
            "affinity": dict(plan.constraints.affinity) if plan.constraints.affinity else None,
            "anti_affinity": dict(plan.constraints.anti_affinity) if plan.constraints.anti_affinity else None,
        }
        meta = {
            "fingerprint": fingerprint,
            "input_fingerprint": input_fp,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "job_spec": dict(plan.job_spec),
            "strategy": str(plan.strategy),
            "constraints": constraints_dict,
            "statistics": dataclasses.asdict(plan.statistics),
            "plan_meta": dict(plan.meta),
        }
        self._save_meta(self._paths.plan_meta, meta)

    def load_execution(self, expected_input: str) -> Optional[Tuple[List[PartFeatureRecord], Optional[str]]]:
        path = self._paths.execution_cache
        meta = self._load_meta(self._paths.execution_meta)
        if not path.exists():
            return None
        if meta.get("input_fingerprint") != expected_input:
            return None
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            rows = []
            for row in reader:
                parsed = dict(row)
                for key in ("descriptor_vector", "bbox_ratios"):
                    if parsed.get(key):
                        parsed[key] = json.loads(parsed[key])
                rows.append(parsed)
        records = [_record_from_dict(row) for row in rows]
        return records, meta.get("fingerprint")

    def save_execution(self, records: Sequence[PartFeatureRecord], fingerprint: str, input_fp: str) -> None:
        self._ensure_root()
        fieldnames = [
            "part_id",
            "dataset",
            "dataset_tier",
            "rel_path",
            "abs_path",
            "file_size",
            "modified_utc",
            "content_hash",
            "geom_hash",
            "descriptor_vector",
            "descriptor_norm",
            "bbox_ratios",
        ]
        with self._paths.execution_cache.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                data = _record_to_dict(record)
                writer.writerow(
                    {
                        "part_id": data["part_id"],
                        "dataset": data["dataset"],
                        "dataset_tier": data["dataset_tier"],
                        "rel_path": data["rel_path"],
                        "abs_path": data["abs_path"],
                        "file_size": data["file_size"],
                        "modified_utc": data["modified_utc"],
                        "content_hash": data["content_hash"],
                        "geom_hash": data["geom_hash"],
                        "descriptor_vector": json.dumps(data["descriptor_vector"], ensure_ascii=False),
                        "descriptor_norm": f"{float(data['descriptor_norm']):.12f}",
                        "bbox_ratios": json.dumps(data["bbox_ratios"], ensure_ascii=False),
                    }
                )
        meta = {
            "fingerprint": fingerprint,
            "input_fingerprint": input_fp,
            "count": len(records),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save_meta(self._paths.execution_meta, meta)

    def load_dataframe(self, expected_input: str) -> Optional[Tuple["pd.DataFrame", Optional[str]]]:
        path = self._paths.dataframe_cache
        meta = self._load_meta(self._paths.dataframe_meta)
        if not path.exists():
            return None
        if meta.get("input_fingerprint") != expected_input:
            return None
        if pd is None:
            raise RuntimeError("pandas is required to load cached dataframe")
        df = pd.read_csv(path)
        return df, meta.get("fingerprint")

    def save_dataframe(self, df: "pd.DataFrame", fingerprint: str, input_fp: str) -> None:
        self._ensure_root()
        if pd is None:
            raise RuntimeError("pandas is required to save cached dataframe")
        df.to_csv(self._paths.dataframe_cache, index=False)
        meta = {
            "fingerprint": fingerprint,
            "input_fingerprint": input_fp,
            "count": len(df),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save_meta(self._paths.dataframe_meta, meta)

    def load_enrichment(self, expected_input: str) -> Optional[Tuple["pd.DataFrame", Optional[str]]]:
        path = self._paths.enrichment_cache
        meta = self._load_meta(self._paths.enrichment_meta)
        if not path.exists():
            return None
        if meta.get("input_fingerprint") != expected_input:
            return None
        if pd is None:
            raise RuntimeError("pandas is required to load cached enrichment dataframe")
        df = pd.read_csv(path)
        return df, meta.get("fingerprint")

    def save_enrichment(self, df: "pd.DataFrame", fingerprint: str, input_fp: str) -> None:
        self._ensure_root()
        if pd is None:
            raise RuntimeError("pandas is required to save cached enrichment dataframe")
        df.to_csv(self._paths.enrichment_cache, index=False)
        meta = {
            "fingerprint": fingerprint,
            "input_fingerprint": input_fp,
            "count": len(df),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save_meta(self._paths.enrichment_meta, meta)


@dataclass(frozen=True)
class S2WorkerContext:
    config: S2PipelineConfig
    logger_name: str = "cad.s2.worker"


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _compute_part_features(candidate: PartCandidate, config: S2PipelineConfig) -> PartFeatureRecord:
    content_hash = _sha256_file(candidate.path)
    seed_material = f"{content_hash}:{config.features.random_seed}".encode("utf-8")
    seed = int(hashlib.blake2b(seed_material, digest_size=16).hexdigest(), 16)
    rng = random.Random(seed)
    descriptor = [rng.uniform(-1.0, 1.0) for _ in range(config.features.descriptor_length)]
    norm = math.sqrt(sum(val * val for val in descriptor)) + 1e-9
    vec = [float(val / norm) for val in descriptor]
    bbox_dims = sorted(abs(rng.gauss(0.0, 1.0)) + 1e-3 for _ in range(3))
    bbox_ratios = (float(bbox_dims[0] / bbox_dims[2]), float(bbox_dims[1] / bbox_dims[2]))
    geom_payload = json.dumps(vec, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    geom_hash = hashlib.blake2b(geom_payload, digest_size=12).hexdigest()
    modified_dt = datetime.fromtimestamp(candidate.modified_ns / 1e9, tz=timezone.utc)
    record = PartFeatureRecord(
        part_id=candidate.part_id,
        dataset=candidate.dataset,
        dataset_tier=candidate.dataset_tier,
        rel_path=candidate.rel_path,
        abs_path=str(candidate.path),
        file_size=candidate.file_size,
        modified_utc=modified_dt.isoformat(),
        content_hash=content_hash,
        geom_hash=geom_hash,
        descriptor_vector=tuple(vec),
        descriptor_norm=norm,
        bbox_ratios=bbox_ratios,
    )
    return record


def s2_worker_handler(leased: LeasedTask, context: S2WorkerContext) -> TaskResult:
    logger = StructuredLogger.get_logger(context.logger_name)
    items = leased.task.extras.get("items", [])
    total = len(items)
    loop_indices = set(_loop_log_indices(total))
    records: List[PartFeatureRecord] = []
    for idx, item in enumerate(items):
        candidate = item.get("candidate") if isinstance(item, dict) else None
        if not isinstance(candidate, PartCandidate):
            logger.warning(
                "s2.worker.invalid_item",
                task_id=leased.task.task_id,
                index=idx,
                detail=str(type(item)),
            )
            continue
        record = _compute_part_features(candidate, context.config)
        records.append(record)
        if idx in loop_indices:
            logger.info(
                "s2.task.progress",
                task_id=leased.task.task_id,
                index=idx,
                total=total,
                part_id=record.part_id,
            )
    return TaskResult(
        payload={"records": records},
        processed=len(records),
        metadata={
            "task_id": leased.task.task_id,
            "total_items": total,
        },
    )

class PipelineProgressTracker:
    """Manage tqdm progress bars for the pipeline."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._bars: List[tqdm] = []
        self._position = 1  # position 0 is reserved for the overall bar
        self._overall: Optional[tqdm] = None
        self._stages: set[str] = set()

    def ensure_overall(self, *, desc: str, unit: str) -> Optional[tqdm]:
        if self._overall is not None:
            return self._overall
        bar = tqdm(
            total=0,
            desc=desc,
            unit=unit,
            position=0,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            disable=False,
        )
        with self._lock:
            if self._overall is None:
                self._overall = bar
                self._bars.append(bar)
                self._stages.add("overall")
                self._position = max(self._position, 1)
            else:  # pragma: no cover - defensive
                bar.close()
        return self._overall

    def create_bar(self, stage: str, **kwargs) -> Optional[tqdm]:
        with self._lock:
            position = self._position
            self._position += 1
        kwargs.setdefault("position", position)
        kwargs.setdefault("dynamic_ncols", True)
        kwargs.setdefault("leave", True)
        kwargs.setdefault("file", sys.stdout)
        kwargs.setdefault("disable", False)
        bar = tqdm(**kwargs)
        with self._lock:
            self._bars.append(bar)
            self._stages.add(stage)
        return bar

    def add_overall_total(self, amount: int) -> None:
        bar = self._overall
        if bar is None:
            return
        with self._lock:
            bar.total = (bar.total or 0) + amount
        bar.refresh()

    def update_overall(self, amount: int = 1) -> None:
        bar = self._overall
        if bar is None:
            return
        bar.update(amount)

    def stages(self) -> Tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._stages))

    def close(self) -> None:
        with self._lock:
            bars = list(self._bars)
            self._bars.clear()
            self._stages.clear()
            self._overall = None
            self._position = 1
        for bar in bars:
            bar.refresh()
            bar.close()


def _loop_log_indices(total: int) -> Tuple[int, ...]:
    if total <= 0:
        return tuple()
    if total <= 3:
        return tuple(range(total))
    mid = total // 2
    return (0, mid, total - 1)


class S2Pipeline:
    """Drive the S2 deduplication and family grouping pipeline."""

    def __init__(self, config: S2PipelineConfig, logger: Optional[StructuredLogger] = None) -> None:
        self.config = config
        self.logger = logger or StructuredLogger.get_logger("cad.s2.pipeline")
        self.paths = S2Paths.from_config(config)
        self._results: Dict[str, PartFeatureRecord] = {}
        self._results_lock = threading.RLock()
        self._progress: Optional[PipelineProgressTracker] = None
        self._cache = StageCacheManager(self.paths)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> None:
        if pd is None:
            raise RuntimeError("pandas is required to run the S2 pipeline. Install pandas>=1.3.")
        progress = PipelineProgressTracker()
        self._progress = progress
        progress.ensure_overall(desc="S2 pipeline (overall)", unit="work")
        with self.logger.stage("Discovery"):
            self.logger.info(
                "s2.pipeline.start",
                sources=len(self.config.sources),
                output_root=str(self.paths.root),
            )
            self.logger.info(
                "s2.sampling.config",
                probability=float(self.config.sampling.probability),
                subset_n=(int(self.config.sampling.subset_n) if self.config.sampling.subset_n is not None else None),
                subset_frac=(float(self.config.sampling.subset_frac) if self.config.sampling.subset_frac is not None else None),
                seed=int(self.config.sampling.seed),
            )
            self.logger.info("s2.progress.layout", bars=list(progress.stages()))
        try:
            ensure_task_config()
            self.paths.root.mkdir(parents=True, exist_ok=True)
            cache_cfg = self.config.stage_cache

            with self.logger.stage("Discovery"):
                # Discovery -------------------------------------------------
                candidates: List[PartCandidate]
                candidate_fp: str
                if cache_cfg.discovery:
                    cached = self._cache.load_discovery()
                    if cached is not None:
                        candidates, stored_fp = cached
                        candidate_fp = stored_fp or self._fingerprint_candidates(candidates)
                        if stored_fp is None:
                            self._cache.save_discovery(candidates, candidate_fp)
                        self.logger.info("s2.discovery.cache.hit", count=len(candidates))
                        if self._progress is not None and len(candidates):
                            self._progress.add_overall_total(len(candidates))
                            self._progress.update_overall(len(candidates))
                            bar = self._progress.create_bar(
                                "discover:cache",
                                total=len(candidates),
                                desc="Discovery (cache)",
                                unit="path",
                            )
                            if bar is not None:
                                bar.update(len(candidates))
                    else:
                        self.logger.info("s2.discovery.cache.miss")
                        candidates = self._discover_candidates()
                        candidate_fp = self._fingerprint_candidates(candidates)
                        self._cache.save_discovery(candidates, candidate_fp)
                else:
                    candidates = self._discover_candidates()
                    candidate_fp = self._fingerprint_candidates(candidates)
                if not candidates:
                    self.logger.warning("s2.pipeline.empty_sources")
                    self._write_empty_outputs()
                    return

            with self.logger.stage("Planning"):
                # Planning --------------------------------------------------
                plan: Optional[Plan] = None
                plan_fp: str
                if cache_cfg.plan:
                    cached_plan = self._cache.load_plan(candidate_fp)
                    if cached_plan is not None:
                        plan, stored_plan_fp = cached_plan
                        plan_fp = stored_plan_fp or self._fingerprint_plan(plan, candidate_fp)
                        if stored_plan_fp is None:
                            self._cache.save_plan(plan, plan_fp, candidate_fp)
                        self.logger.info("s2.plan.cache.hit", tasks=len(plan.tasks))
                    else:
                        self.logger.info("s2.plan.cache.miss")
                if plan is None:
                    plan = self._build_plan(candidates)
                    plan_fp = self._fingerprint_plan(plan, candidate_fp)
                    if cache_cfg.plan:
                        self._cache.save_plan(plan, plan_fp, candidate_fp)
                if not plan.tasks:
                    self.logger.warning("s2.pipeline.empty_plan")
                    self._write_empty_outputs()
                    return

            with self.logger.stage("Execution"):
                # Execution -------------------------------------------------
                records: List[PartFeatureRecord]
                records_fp: str
                execution_cached = False
                if cache_cfg.execution:
                    cached_exec = self._cache.load_execution(plan_fp)
                    if cached_exec is not None:
                        records, stored_records_fp = cached_exec
                        records_fp = stored_records_fp or self._fingerprint_records(records)
                        if stored_records_fp is None:
                            self._cache.save_execution(records, records_fp, plan_fp)
                        with self._results_lock:
                            self._results = {record.part_id: record for record in records}
                        execution_cached = True
                        self.logger.info("s2.execution.cache.hit", count=len(records))
                        if self._progress is not None and len(records):
                            self._progress.add_overall_total(len(records))
                            self._progress.update_overall(len(records))
                            bar = self._progress.create_bar(
                                "execution",
                                total=len(records),
                                desc="Execution (cache)",
                                unit="part",
                            )
                            if bar is not None:
                                bar.update(len(records))
                    else:
                        self.logger.info("s2.execution.cache.miss")
                if not execution_cached:
                    self._execute_plan(plan)
                    records = list(self._results.values())
                    records_fp = self._fingerprint_records(records)
                    if cache_cfg.execution:
                        self._cache.save_execution(records, records_fp, plan_fp)
                if not records:
                    self.logger.warning("s2.pipeline.no_records")
                    self._write_empty_outputs()
                    return

            with self.logger.stage("Dataframe-build"):
                # DataFrame -------------------------------------------------
                df: Optional[pd.DataFrame] = None
                df_fp = records_fp
                if cache_cfg.dataframe:
                    cached_df = self._cache.load_dataframe(records_fp)
                    if cached_df is not None:
                        df, stored_df_fp = cached_df
                        df_fp = stored_df_fp or records_fp
                        if stored_df_fp is None:
                            self._cache.save_dataframe(df, df_fp, records_fp)
                        self.logger.info("s2.dataframe.cache.hit", rows=len(df))
                        if self._progress is not None:
                            bar = self._progress.create_bar(
                                "dataframe",
                                total=1,
                                desc="Dataframe",
                                unit="stage",
                            )
                            if bar is not None:
                                bar.update(1)
                    else:
                        self.logger.info("s2.dataframe.cache.miss")
                if df is None:
                    df = self._build_dataframe(records)
                    df_fp = records_fp
                    if cache_cfg.dataframe:
                        self._cache.save_dataframe(df, df_fp, records_fp)

            with self.logger.stage("Enrichment"):
                # Enrichment -------------------------------------------------
                family_count = 0
                enrichment_fp: Optional[str] = None
                enrichment_cached = False
                enriched: Optional[pd.DataFrame] = None
                if cache_cfg.enrichment:
                    cached_enriched = self._cache.load_enrichment(df_fp)
                    if cached_enriched is not None:
                        enriched, stored_en_fp = cached_enriched
                        enrichment_fp = stored_en_fp or self._fingerprint_enriched(enriched)
                        if stored_en_fp is None and enrichment_fp is not None:
                            self._cache.save_enrichment(enriched, enrichment_fp, df_fp)
                        family_count = int(enriched["family_id"].nunique()) if len(enriched) else 0
                        enrichment_cached = True
                        self.logger.info("s2.enrichment.cache.hit", parts=len(enriched), families=family_count)
                        if self._progress is not None:
                            bar = self._progress.create_bar(
                                "enrichment",
                                total=1,
                                desc="Enrichment",
                                unit="stage",
                            )
                            if bar is not None:
                                bar.update(1)
                    else:
                        self.logger.info("s2.enrichment.cache.miss")
                if not enrichment_cached:
                    enrichment_bar: Optional[tqdm] = None
                    if self._progress is not None:
                        enrichment_bar = self._progress.create_bar(
                            "enrichment",
                            total=1,
                            desc="Enrichment",
                            unit="stage",
                        )
                    self.logger.info("s2.enrichment.start", parts=len(df))
                    enrich_start = time.time()
                    enriched = self._enrich_dataframe(df)
                    family_count = int(enriched["family_id"].nunique()) if len(enriched) else 0
                    enrich_duration = time.time() - enrich_start
                    if enrichment_bar is not None:
                        enrichment_bar.update(1)
                    self.logger.info(
                        "s2.enrichment.complete",
                        parts=len(enriched),
                        families=family_count,
                        seconds=float(enrich_duration),
                    )
                    enrichment_fp = self._fingerprint_enriched(enriched)
                    if cache_cfg.enrichment:
                        self._cache.save_enrichment(enriched, enrichment_fp, df_fp)
                if enriched is None:
                    raise RuntimeError("Enrichment stage did not produce results")

            with self.logger.stage("Outputs"):
                # Outputs ----------------------------------------------------
                output_bar: Optional[tqdm] = None
                if self._progress is not None:
                    output_bar = self._progress.create_bar(
                        "outputs",
                        total=1,
                        desc="Outputs",
                        unit="stage",
                    )
                self.logger.info(
                    "s2.outputs.start",
                    root=str(self.paths.root),
                    parts=len(enriched),
                )
                output_start = time.time()
                self._write_outputs(enriched, records)
                output_duration = time.time() - output_start
                if output_bar is not None:
                    output_bar.update(1)
                self.logger.info(
                    "s2.outputs.complete",
                    seconds=float(output_duration),
                    root=str(self.paths.root),
                )
                self.logger.info(
                    "s2.pipeline.completed",
                    parts=len(enriched),
                    families=family_count,
                    cross_tier_quarantine=int(enriched["cross_tier_quarantine"].sum()),
                )
        finally:
            stages = progress.stages()
            with self.logger.stage("Outputs"):
                self.logger.info("s2.progress.summary", bar_count=len(stages), stages=list(stages))
            progress.close()
            self._progress = None

    # ------------------------------------------------------------------
    # Discovery and planning
    # ------------------------------------------------------------------
    def _discover_candidates(self) -> List[PartCandidate]:
        discovered: List[PartCandidate] = []
        for source in self.config.sources:
            files = self._discover_source(source)
            discovered.extend(files)
        total = len(discovered)
        if total:
            self.logger.info("s2.discovery.total", count=total)
        sampled = self._apply_sampling(discovered)
        if len(sampled) != total:
            self.logger.info("s2.discovery.sampled", original=total, sampled=len(sampled))
        return sampled

    def _discover_source(self, source: SourceSettings) -> List[PartCandidate]:
        root = source.root
        results: List[PartCandidate] = []
        if not root.exists():
            self.logger.warning("s2.discovery.missing_root", dataset=source.dataset, root=str(root))
            return results
        allowed = {ext.lower() for ext in source.allowed_extensions}
        pattern = source.pattern
        paths = sorted(root.rglob(pattern)) if pattern else sorted(root.rglob("*"))
        total = len(paths)
        progress_bar: Optional[tqdm] = None
        skipped_probability = 0
        if self._progress is not None and total:
            self._progress.add_overall_total(total)
            progress_bar = self._progress.create_bar(
                f"discover:{source.dataset}",
                total=total,
                desc=f"Discovery ({source.dataset})",
                unit="path",
            )
        indices = set(_loop_log_indices(total))
        for idx, path in enumerate(paths):
            if progress_bar is not None:
                progress_bar.update(1)
            if self._progress is not None and total:
                self._progress.update_overall(1)
            if not path.is_file():
                continue
            if allowed and path.suffix.lower() not in allowed:
                continue
            rel_path = path.relative_to(root).as_posix()
            part_id = self._make_part_id(source.dataset, rel_path)
            stat = path.stat()
            candidate = PartCandidate(
                dataset=source.dataset,
                dataset_tier=source.tier,
                root=root,
                path=path,
                rel_path=rel_path,
                part_id=part_id,
                file_size=int(stat.st_size),
                modified_ns=int(stat.st_mtime_ns),
            )
            if not self._should_keep_candidate(candidate):
                skipped_probability += 1
                continue
            results.append(candidate)
            if idx in indices:
                self.logger.info(
                    "s2.discovery.progress",
                    dataset=source.dataset,
                    index=idx,
                    total=total,
                    part_id=part_id,
                )
        self.logger.info(
            "s2.discovery.source_complete",
            dataset=source.dataset,
            tier=source.tier,
            count=len(results),
            skipped_probability=skipped_probability,
        )
        return results

    def _should_keep_candidate(self, candidate: PartCandidate) -> bool:
        sampling = self.config.sampling
        probability = float(sampling.probability)
        if probability >= 1.0:
            return True
        if probability <= 0.0:
            return False
        material = f"{candidate.part_id}:{sampling.seed}".encode("utf-8")
        digest = hashlib.blake2b(material, digest_size=8).digest()
        value = int.from_bytes(digest, byteorder="big")
        threshold = int(probability * (1 << 64))
        return value < threshold

    def _apply_sampling(self, candidates: List[PartCandidate]) -> List[PartCandidate]:
        cfg = self.config.sampling
        total = len(candidates)
        if total == 0:
            return candidates
        rng = random.Random(cfg.seed)
        if cfg.subset_n is not None and cfg.subset_n < total:
            indices = list(range(total))
            rng.shuffle(indices)
            keep = sorted(indices[: cfg.subset_n])
            return [candidates[i] for i in keep]
        if cfg.subset_frac is not None and 0.0 < cfg.subset_frac < 1.0:
            target = max(1, int(math.ceil(total * cfg.subset_frac)))
            indices = list(range(total))
            rng.shuffle(indices)
            keep = sorted(indices[:target])
            return [candidates[i] for i in keep]
        return candidates

    def _build_plan(self, candidates: Sequence[PartCandidate]) -> Plan:
        job_spec = {"job_id": f"s2-{int(time.time())}"}
        items: List[ItemRecord] = []
        dataset_counts: Dict[str, int] = {}
        max_items_per_task = 0
        raw_max_items = self.config.partition.constraints.get("max_items_per_task")
        if raw_max_items is not None:
            try:
                max_items_per_task = int(raw_max_items)
            except (TypeError, ValueError):
                max_items_per_task = 0
            if max_items_per_task < 0:
                max_items_per_task = 0
        for idx, candidate in enumerate(candidates):
            weight = max(1.0, candidate.file_size / 1024.0)
            dataset_counts.setdefault(candidate.dataset, 0)
            count = dataset_counts[candidate.dataset]
            dataset_counts[candidate.dataset] = count + 1
            if max_items_per_task:
                chunk_index = count // max_items_per_task
                group_key = f"{candidate.dataset}#{chunk_index:06d}"
            else:
                group_key = candidate.dataset
            metadata = {
                "dataset": candidate.dataset,
                "tier": candidate.dataset_tier,
            }
            if max_items_per_task:
                metadata["dataset_chunk"] = group_key
            items.append(
                ItemRecord(
                    payload_ref=f"{candidate.dataset}:{candidate.part_id}",
                    weight=weight,
                    group_key=group_key,
                    metadata=metadata,
                    checksum=None,
                    raw={"candidate": candidate},
                    index=idx,
                )
            )
        plan = TaskPartitioner.plan(
            job_spec,
            items,
            self.config.partition.strategy,
            self.config.partition.constraints,
            metadata={"created_at": datetime.now(timezone.utc).isoformat(), "items": len(items)},
        )
        self.logger.info(
            "s2.plan.created",
            tasks=len(plan.tasks),
            strategy=str(plan.strategy),
            gini=float(plan.statistics.gini_coefficient),
        )
        return plan

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def _execute_plan(self, plan: Plan) -> None:
        pool = TaskPool()
        pool.put(plan.tasks)
        policy = ExecutorPolicy()
        total_items = sum(len(task.extras.get("items", [])) for task in plan.tasks)
        execution_bar: Optional[tqdm] = None
        execution_managed = False
        if total_items > 0:
            miniters = max(1, math.ceil(total_items / 100))
            if self._progress is not None:
                self._progress.add_overall_total(total_items)
                execution_bar = self._progress.create_bar(
                    "execution",
                    total=total_items,
                    desc="Execution (parts)",
                    unit="part",
                    miniters=miniters,
                )
                execution_managed = execution_bar is not None
            if execution_bar is None:
                execution_bar = tqdm(
                    total=total_items,
                    desc="S2 pipeline",
                    unit="part",
                    miniters=miniters,
                    leave=True,
                    dynamic_ncols=True,
                    file=sys.stdout,
                    disable=False,
                )
            execution_bar.refresh()

        def handle_result(leased: LeasedTask, result: TaskResult) -> None:
            records = []
            if result.payload:
                records = list(result.payload.get("records", []))
            processed = int(result.processed or len(records))
            if execution_bar is not None and processed:
                execution_bar.update(processed)
            if self._progress is not None and processed:
                self._progress.update_overall(processed)
            if records:
                with self._results_lock:
                    for record in records:
                        self._results[record.part_id] = record

        def on_lease(executor: ParallelExecutor, leased: LeasedTask) -> None:
            items = leased.task.extras.get("items", [])
            if random.random() <= 0.01:
                self.logger.info(
                    "s2.task.start",
                    task_id=leased.task.task_id,
                    items=len(items),
                    stage="Execution",
                )

        def on_retry(executor: ParallelExecutor, leased: LeasedTask, attempt: int, exc: Exception) -> None:
            self.logger.warning(
                "s2.task.retry",
                task_id=leased.task.task_id,
                attempt=attempt,
                error=str(exc),
            )

        def on_dead(executor: ParallelExecutor, leased: LeasedTask, exc: Exception) -> None:
            self.logger.error(
                "s2.task.failed",
                task_id=leased.task.task_id,
                error=str(exc),
            )

        events = ExecutorEvents(
            on_lease=on_lease,
            on_retry=on_retry,
            on_dead=on_dead,
        )
        worker_context = S2WorkerContext(config=self.config, logger_name="cad.s2.worker")
        try:
            ParallelExecutor.run(
                s2_worker_handler,
                pool,
                policy,
                events=events,
                handler_context=worker_context,
                result_handler=handle_result,
                console_min_level="WARN",
            )
        finally:
            if execution_bar is not None and not execution_managed:
                execution_bar.refresh()
                execution_bar.close()

    # ------------------------------------------------------------------
    # Dataframe enrichment
    # ------------------------------------------------------------------
    def _build_dataframe(self, records: Sequence[PartFeatureRecord]) -> pd.DataFrame:
        rows: List[Dict[str, object]] = []
        for record in records:
            row: Dict[str, object] = {
                "part_id": record.part_id,
                "source_dataset": record.dataset,
                "dataset_tier": record.dataset_tier,
                "rel_path": record.rel_path,
                "abs_path": record.abs_path,
                "file_size": record.file_size,
                "modified_utc": record.modified_utc,
                "content_hash": record.content_hash,
                "geom_hash": record.geom_hash,
                "descriptor_vector": tuple(record.descriptor_vector),
                "descriptor_norm": record.descriptor_norm,
                "bbox_ratios_vector": tuple(record.bbox_ratios),
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def _enrich_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df.assign(
                duplicate_canonical=df["part_id"],
                duplicate_rank=0,
                family_id=df["part_id"],
                family_size=1,
                cross_tier_quarantine=0,
                quarantine_reason="",
            )
        if torch is None:
            raise RuntimeError("未检测到 PyTorch，无法执行 GPU 加速的重复检测与聚类。")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用，请检查 GPU 驱动或 CUDA 环境。")
        df = df.copy().reset_index(drop=True)
        df["duplicate_canonical"] = df.groupby("content_hash")["part_id"].transform("min")
        df["duplicate_rank"] = (
            df.groupby("content_hash")["part_id"].rank(method="first").astype(int) - 1
        )
        descriptors = [list(v) for v in df["descriptor_vector"].tolist()] if len(df) else []
        device = torch.device("cuda")
        with torch.no_grad():
            descriptor_tensor = torch.tensor(descriptors, dtype=torch.float32, device=device)
            similarities_tensor = torch.zeros(len(df), dtype=torch.float32, device=device)
            geom_flags_tensor = torch.zeros(len(df), dtype=torch.int32, device=device)
            threshold = float(self.config.dedup.similarity_threshold)
            for canonical, group in df.groupby("duplicate_canonical"):
                indices = torch.tensor(group.index.to_list(), dtype=torch.long, device=device)
                canonical_vec = descriptor_tensor[indices[0]]
                base_vec = canonical_vec / (canonical_vec.norm() + 1e-9)
                sims = descriptor_tensor[indices] @ base_vec
                similarities_tensor[indices] = sims
                geom_flags_tensor[indices] = (sims >= threshold).to(torch.int32)
            df["duplicate_similarity"] = similarities_tensor.cpu().numpy().tolist()
            df["is_geom_duplicate"] = geom_flags_tensor.cpu().numpy().tolist()
            labels = self._cluster_descriptors(descriptors, descriptor_tensor)
        family_ids = [
            self._format_family_id(label, pid) for label, pid in zip(labels, df["part_id"])
        ]
        df["family_id"] = family_ids
        df["family_size"] = df.groupby("family_id")["part_id"].transform("count")
        quarantine_flags = [0] * len(df)
        quarantine_reason = [""] * len(df)
        preferred = self.config.quarantine.preferred_tier.lower()
        reason = self.config.quarantine.reason
        for content_hash, group in df.groupby("content_hash"):
            tiers = {str(t).lower() for t in group["dataset_tier"].unique()}
            if len(tiers) <= 1:
                continue
            for idx in group.index:
                tier = str(df.at[idx, "dataset_tier"]).lower()
                if tier != preferred:
                    quarantine_flags[idx] = 1
                    quarantine_reason[idx] = reason
        df["cross_tier_quarantine"] = quarantine_flags
        df["quarantine_reason"] = quarantine_reason
        return df

    def _cluster_descriptors(
        self,
        descriptors: List[List[float]],
        descriptor_tensor: Optional["torch.Tensor"] = None,
    ) -> List[int]:
        if not descriptors:
            return []
        eps = self.config.clustering.eps
        if self.config.clustering.auto_eps.enabled and len(descriptors) > 1:
            eps = self._estimate_eps(descriptors, self.config.clustering.auto_eps)
        min_samples = max(1, self.config.clustering.min_samples)
        if descriptor_tensor is None:
            raise RuntimeError("缺少 GPU 描述符张量，无法执行聚类。")
        return self._cluster_descriptors_gpu(descriptor_tensor, float(eps), min_samples)

    def _estimate_eps(self, descriptors: List[List[float]], cfg) -> float:
        sample = list(descriptors)
        if len(sample) > cfg.sample_size:
            rng = random.Random(cfg.sample_size)
            indices = list(range(len(sample)))
            rng.shuffle(indices)
            sample = [sample[i] for i in indices[: cfg.sample_size]]
        if len(sample) < 2:
            return float(cfg.scale)
        distances: List[float] = []
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                distances.append(self._euclidean_distance(sample[i], sample[j]))
        if not distances:
            return float(cfg.scale)
        quant = self._quantile(distances, cfg.quantile)
        return max(quant * float(cfg.scale), 1e-6)

    def _simple_density_cluster(self, descriptors: List[List[float]], eps: float, min_samples: int) -> List[int]:
        n = len(descriptors)
        labels = [-1] * n
        cluster_id = 0
        for idx in range(n):
            if labels[idx] != -1:
                continue
            distances = [self._euclidean_distance(descriptors[idx], other) for other in descriptors]
            neighbours = [i for i, value in enumerate(distances) if value <= eps]
            if len(neighbours) < min_samples:
                continue
            for neighbour in neighbours:
                labels[neighbour] = cluster_id
            cluster_id += 1
        return labels

    def _cluster_descriptors_gpu(
        self,
        descriptor_tensor: "torch.Tensor",
        eps: float,
        min_samples: int,
    ) -> List[int]:
        if torch is None or not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用，无法执行 GPU 聚类。")
        if descriptor_tensor.is_cuda is False:
            descriptor_tensor = descriptor_tensor.cuda()
        n = descriptor_tensor.size(0)
        if n == 0:
            return []
        if n == 1:
            return [0]
        with torch.no_grad():
            labels = torch.full((n,), -1, dtype=torch.int32, device=descriptor_tensor.device)
            cluster_id = 0
            # Process descriptors in batches to reduce peak memory usage and surface progress.
            batch_size = max(1, min(512, n))
            processed = 0
            last_logged_percent = -1
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_indices = torch.arange(start, end, device=descriptor_tensor.device, dtype=torch.long)
                batch_vecs = descriptor_tensor[batch_indices]
                distances = torch.cdist(batch_vecs, descriptor_tensor, p=2)
                for local_idx, global_idx in enumerate(batch_indices.tolist()):
                    if labels[global_idx] != -1:
                        continue
                    neighbours = torch.nonzero(distances[local_idx] <= eps, as_tuple=False).flatten()
                    if neighbours.numel() < min_samples:
                        continue
                    labels[neighbours] = cluster_id
                    cluster_id += 1
                processed = end
                current_percent = int((processed / n) * 100)
                if current_percent > last_logged_percent:
                    for percent in range(last_logged_percent + 1, current_percent + 1):
                        if percent > 100:
                            break
                        reported_processed = min(processed, math.ceil((percent / 100) * n))
                        self.logger.info(
                            "s2.enrichment.cluster.progress",
                            processed=reported_processed,
                            total=n,
                            percent=percent,
                        )
                    last_logged_percent = current_percent
        return labels.cpu().tolist()

    @staticmethod
    def _format_family_id(label: int, part_id: str) -> str:
        if label < 0:
            suffix = hashlib.blake2b(part_id.encode("utf-8"), digest_size=6).hexdigest()
            return f"fam_iso_{suffix}"
        return f"fam_{label:05d}"

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------
    def _write_outputs(self, df: pd.DataFrame, records: Sequence[PartFeatureRecord]) -> None:
        df = df.copy()
        df["descriptor_json"] = df["descriptor_vector"].apply(lambda arr: json.dumps(list(arr)))
        df["bbox_ratios_json"] = df["bbox_ratios_vector"].apply(lambda arr: json.dumps(list(arr)))
        df_output = df.drop(columns=["descriptor_vector", "bbox_ratios_vector"])
        df_output.to_csv(self.paths.part_index, index=False)
        for_split_cols = [
            "part_id",
            "source_dataset",
            "dataset_tier",
            "rel_path",
            "content_hash",
            "duplicate_canonical",
            "family_id",
            "cross_tier_quarantine",
            "quarantine_reason",
        ]
        df_output[for_split_cols].to_csv(self.paths.part_index_for_split, index=False)
        family_hist = (
            df_output.groupby("family_id")
            .agg(family_size=("part_id", "count"), datasets=("source_dataset", lambda s: "|".join(sorted(set(s)))))
            .reset_index()
        )
        family_hist.to_csv(self.paths.family_hist, index=False)
        if self.config.outputs.write_signatures:
            self._write_signatures(records)
        diagnostics = {
            "total_parts": int(len(df_output)),
            "unique_canonical": int(df_output["duplicate_canonical"].nunique()),
            "families": int(family_hist.shape[0]),
            "cross_tier_quarantine": int(df_output["cross_tier_quarantine"].sum()),
            "geom_duplicates": int(df_output["is_geom_duplicate"].sum()),
            "config": {
                "similarity_threshold": self.config.dedup.similarity_threshold,
                "clustering_eps": self.config.clustering.eps,
                "auto_eps": self.config.clustering.auto_eps.enabled,
            },
        }
        with self.paths.diagnostics.open("w", encoding="utf-8") as fh:
            json.dump(diagnostics, fh, indent=2, ensure_ascii=False)

    def _write_signatures(self, records: Sequence[PartFeatureRecord]) -> None:
        rows = []
        for rec in records:
            rows.append(
                {
                    "part_id": rec.part_id,
                    "source_dataset": rec.dataset,
                    "dataset_tier": rec.dataset_tier,
                    "rel_path": rec.rel_path,
                    "abs_path": rec.abs_path,
                    "content_hash": rec.content_hash,
                    "geom_hash": rec.geom_hash,
                    "descriptor": list(rec.descriptor_vector),
                    "descriptor_norm": rec.descriptor_norm,
                    "bbox_ratios": list(rec.bbox_ratios),
                }
            )
        df = pd.DataFrame(rows)
        try:
            df.to_parquet(self.paths.signatures, index=False)
            self.logger.info("s2.signatures.parquet", path=str(self.paths.signatures))
        except Exception as exc:  # pragma: no cover - fallback path
            fallback = self.paths.signatures.with_suffix(".json")
            df.to_json(fallback, orient="records", indent=2)
            self.logger.warning(
                "s2.signatures.fallback_json",
                path=str(fallback),
                error=str(exc),
            )

    def _write_empty_outputs(self) -> None:
        empty = pd.DataFrame(
            columns=[
                "part_id",
                "source_dataset",
                "dataset_tier",
                "rel_path",
                "abs_path",
                "file_size",
                "modified_utc",
                "content_hash",
                "geom_hash",
                "duplicate_canonical",
                "duplicate_rank",
                "duplicate_similarity",
                "is_geom_duplicate",
                "family_id",
                "family_size",
                "cross_tier_quarantine",
                "quarantine_reason",
            ]
        )
        empty.to_csv(self.paths.part_index, index=False)
        empty[
            [
                "part_id",
                "source_dataset",
                "dataset_tier",
                "rel_path",
                "content_hash",
                "duplicate_canonical",
                "family_id",
                "cross_tier_quarantine",
                "quarantine_reason",
            ]
        ].to_csv(self.paths.part_index_for_split, index=False)
        pd.DataFrame(columns=["family_id", "family_size", "datasets"]).to_csv(self.paths.family_hist, index=False)
        if self.config.outputs.write_signatures:
            with self.paths.signatures.open("w", encoding="utf-8") as fh:
                json.dump([], fh)
        with self.paths.diagnostics.open("w", encoding="utf-8") as fh:
            json.dump({"total_parts": 0, "unique_canonical": 0, "families": 0, "cross_tier_quarantine": 0, "geom_duplicates": 0}, fh, indent=2)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _vector_norm(vec: Sequence[float]) -> float:
        return math.sqrt(sum(float(x) * float(x) for x in vec))

    @staticmethod
    def _dot(vec1: Sequence[float], vec2: Sequence[float]) -> float:
        return sum(float(x) * float(y) for x, y in zip(vec1, vec2))

    @staticmethod
    def _euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
        return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))

    @staticmethod
    def _quantile(values: Sequence[float], q: float) -> float:
        if not values:
            return 0.0
        data = sorted(float(v) for v in values)
        if q <= 0.0:
            return data[0]
        if q >= 1.0:
            return data[-1]
        pos = (len(data) - 1) * q
        lower = int(math.floor(pos))
        upper = int(math.ceil(pos))
        if lower == upper:
            return data[lower]
        lower_val = data[lower]
        upper_val = data[upper]
        return lower_val + (upper_val - lower_val) * (pos - lower)

    @staticmethod
    def _make_part_id(dataset: str, rel_path: str) -> str:
        slug = re.sub(r"[^A-Za-z0-9]+", "_", rel_path).strip("_")
        digest = hashlib.blake2b(f"{dataset}:{rel_path}".encode("utf-8"), digest_size=10).hexdigest()
        slug = slug[-48:] if len(slug) > 48 else slug
        return f"{dataset}__{slug}__{digest}"

    def _fingerprint_candidates(self, candidates: Sequence[PartCandidate]) -> str:
        h = hashlib.sha256()
        for candidate in sorted(candidates, key=lambda c: c.part_id):
            h.update(candidate.part_id.encode("utf-8"))
            h.update(str(candidate.file_size).encode("utf-8"))
            h.update(str(candidate.modified_ns).encode("utf-8"))
        return h.hexdigest()

    def _fingerprint_plan(self, plan: Plan, candidate_fp: str) -> str:
        h = hashlib.sha256()
        h.update(candidate_fp.encode("utf-8"))
        h.update(str(plan.strategy).encode("utf-8"))
        for task in sorted(plan.tasks, key=lambda t: t.task_id):
            h.update(task.task_id.encode("utf-8"))
            h.update(str(task.weight).encode("utf-8"))
            for ref in task.payload_ref:
                h.update(str(ref).encode("utf-8"))
        return h.hexdigest()

    def _fingerprint_records(self, records: Sequence[PartFeatureRecord]) -> str:
        h = hashlib.sha256()
        for record in sorted(records, key=lambda r: r.part_id):
            h.update(record.part_id.encode("utf-8"))
            h.update(record.content_hash.encode("utf-8"))
            h.update(record.geom_hash.encode("utf-8"))
        return h.hexdigest()

    def _fingerprint_enriched(self, df: "pd.DataFrame") -> str:
        h = hashlib.sha256()
        if df is None or df.empty:
            h.update(b"empty")
            return h.hexdigest()
        for _, row in df.sort_values("part_id").iterrows():
            h.update(str(row.get("part_id", "")).encode("utf-8"))
            h.update(str(row.get("duplicate_canonical", "")).encode("utf-8"))
            h.update(str(row.get("family_id", "")).encode("utf-8"))
            h.update(str(row.get("cross_tier_quarantine", 0)).encode("utf-8"))
        return h.hexdigest()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the S2 deduplication pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to main YAML configuration")
    args = parser.parse_args(argv)
    config_path = Path(args.config).expanduser() if args.config else Path(__file__).resolve().parents[1] / "main.yaml"
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}", file=sys.stderr)
        return 1
    cfg = Config.load_singleton(config_path)
    try:
        StructuredLogger.configure_from_config(cfg)
    except Exception as exc:
        print(f"Logger initialization failed: {exc}", file=sys.stderr)
        return 1
    sinks_config = cfg.get("logger.sinks", list, [])
    for sink in sinks_config:
        if not isinstance(sink, dict):
            continue
        if sink.get("type") not in {"file", "rotating_file"}:
            continue
        sink_path = sink.get("path")
        if not sink_path:
            continue
        resolved_path = Path(os.path.realpath(str(Path(sink_path).expanduser())))
        if not resolved_path.exists():
            print(f"Log file was not created: {resolved_path}", file=sys.stderr)
            return 1
    logger = StructuredLogger.get_logger("cad.s2.cli")
    marker = "=" * 12 + " S2 PIPELINE EXECUTION " + "=" * 12
    with logger.stage("Discovery"):
        logger.info("s2.cli.banner", pipeline="S2", marker=marker)
    try:
        exported_cfg = cfg.export(fmt="yaml", redact_secrets=False)
    except Exception:
        exported_cfg = cfg.export(fmt="yaml", redact_secrets=True)
    config_text = exported_cfg if isinstance(exported_cfg, str) else str(exported_cfg)
    with logger.stage("Discovery"):
        logger.info("s2.cli.config_dump", pipeline="S2", config_yaml=config_text)
        logger.info("s2.cli.start", config=str(config_path))
    pipeline = S2Pipeline(S2PipelineConfig.from_config(cfg))
    pipeline.run()
    with logger.stage("Outputs"):
        logger.info("s2.cli.complete")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

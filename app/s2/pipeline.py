"""Topology-to-Table S2 pipeline with task system integration."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import pandas as pd  # type: ignore
except ModuleNotFoundError:
    pd = None  # type: ignore

from config import Config
from logger import StructuredLogger
from parallel_executor import ExecutorPolicy, ParallelExecutor
from task_partitioner import ItemRecord, TaskPartitioner, Plan
from task_pool import TaskPool
from task_system_config import ensure_task_config

from .settings import S2PipelineConfig, SourceSettings

try:  # Optional dependency for clustering
    from sklearn.cluster import DBSCAN  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    DBSCAN = None  # type: ignore


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

    @classmethod
    def from_config(cls, config: S2PipelineConfig) -> "S2Paths":
        outputs = config.outputs
        root = outputs.root
        return cls(
            root=root,
            signatures=root / outputs.signatures_file,
            part_index=root / outputs.part_index_file,
            part_index_for_split=root / outputs.part_index_for_split_file,
            family_hist=root / outputs.family_hist_file,
            diagnostics=root / outputs.diagnostics_file,
        )


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
        self._candidate_by_index: Dict[int, PartCandidate] = {}
        self._results: Dict[str, PartFeatureRecord] = {}
        self._results_lock = threading.RLock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> None:
        if pd is None:
            raise RuntimeError("pandas is required to run the S2 pipeline. Install pandas>=1.3.")
        self.logger.info(
            "s2.pipeline.start",
            sources=len(self.config.sources),
            output_root=str(self.paths.root),
        )
        ensure_task_config()
        self.paths.root.mkdir(parents=True, exist_ok=True)
        candidates = self._discover_candidates()
        if not candidates:
            self.logger.warning("s2.pipeline.empty_sources")
            self._write_empty_outputs()
            return
        plan = self._build_plan(candidates)
        self._execute_plan(plan)
        records = list(self._results.values())
        if not records:
            self.logger.warning("s2.pipeline.no_records")
            self._write_empty_outputs()
            return
        df = self._build_dataframe(records)
        enriched = self._enrich_dataframe(df)
        self._write_outputs(enriched, records)
        self.logger.info(
            "s2.pipeline.completed",
            parts=len(enriched),
            families=int(enriched["family_id"].nunique()),
            cross_tier_quarantine=int(enriched["cross_tier_quarantine"].sum()),
        )

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
        indices = set(_loop_log_indices(total))
        for idx, path in enumerate(paths):
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
        )
        return results

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
        for idx, candidate in enumerate(candidates):
            self._candidate_by_index[idx] = candidate
            weight = max(1.0, candidate.file_size / 1024.0)
            items.append(
                ItemRecord(
                    payload_ref=f"{candidate.dataset}:{candidate.part_id}",
                    weight=weight,
                    group_key=candidate.dataset,
                    metadata={"dataset": candidate.dataset, "tier": candidate.dataset_tier},
                    checksum=None,
                    raw={},
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
        indices_cache: Dict[str, Tuple[int, ...]] = {
            task.task_id: tuple(int(i) for i in task.extras.get("item_indexes", []))
            for task in plan.tasks
        }

        def handler(leased) -> None:
            indexes = indices_cache.get(leased.task.task_id, tuple())
            total = len(indexes)
            if total == 0:
                self.logger.warning("s2.task.empty", task_id=leased.task.task_id)
                return
            self.logger.info("s2.task.start", task_id=leased.task.task_id, items=total)
            loop_indices = set(_loop_log_indices(total))
            for pos, item_index in enumerate(indexes):
                candidate = self._candidate_by_index[item_index]
                record = self._extract_features(candidate)
                with self._results_lock:
                    self._results[record.part_id] = record
                if pos in loop_indices:
                    self.logger.info(
                        "s2.task.progress",
                        task_id=leased.task.task_id,
                        index=pos,
                        total=total,
                        part_id=record.part_id,
                    )

        ParallelExecutor.run(handler, pool, policy)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    def _extract_features(self, candidate: PartCandidate) -> PartFeatureRecord:
        content_hash = self._sha256_file(candidate.path)
        seed_material = f"{content_hash}:{self.config.features.random_seed}".encode("utf-8")
        seed = int(hashlib.blake2b(seed_material, digest_size=16).hexdigest(), 16)
        rng = random.Random(seed)
        descriptor = [rng.uniform(-1.0, 1.0) for _ in range(self.config.features.descriptor_length)]
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

    @staticmethod
    def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

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
        df = df.copy()
        df["duplicate_canonical"] = df.groupby("content_hash")["part_id"].transform("min")
        df["duplicate_rank"] = (
            df.groupby("content_hash")["part_id"].rank(method="first").astype(int) - 1
        )
        similarities = [0.0] * len(df)
        geom_flags = [0] * len(df)
        for canonical, group in df.groupby("duplicate_canonical"):
            canonical_idx = group.index[0]
            canonical_vec = list(df.at[canonical_idx, "descriptor_vector"])
            norm = self._vector_norm(canonical_vec) or 1e-9
            base_vec = [value / norm for value in canonical_vec]
            for idx in group.index:
                vec = list(df.at[idx, "descriptor_vector"])
                sim = self._dot(vec, base_vec)
                similarities[idx] = sim
                if sim >= self.config.dedup.similarity_threshold:
                    geom_flags[idx] = 1
        df["duplicate_similarity"] = similarities
        df["is_geom_duplicate"] = geom_flags
        descriptors = [list(v) for v in df["descriptor_vector"].tolist()] if len(df) else []
        labels = self._cluster_descriptors(descriptors)
        family_ids = [self._format_family_id(label, pid) for label, pid in zip(labels, df["part_id"]) ]
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

    def _cluster_descriptors(self, descriptors: List[List[float]]) -> List[int]:
        if not descriptors:
            return []
        eps = self.config.clustering.eps
        if self.config.clustering.auto_eps.enabled and len(descriptors) > 1:
            eps = self._estimate_eps(descriptors, self.config.clustering.auto_eps)
        min_samples = max(1, self.config.clustering.min_samples)
        if DBSCAN is not None:
            try:
                import numpy as _np  # type: ignore

                arr = _np.asarray(descriptors, dtype=float)
                labels = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit_predict(arr)
                return labels.tolist()
            except Exception as exc:  # pragma: no cover - optional path
                self.logger.warning("s2.cluster.dbscan_error", error=str(exc))
        return self._simple_density_cluster(descriptors, eps, min_samples)

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


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the S2 deduplication pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to main YAML configuration")
    args = parser.parse_args(argv)
    config_path = Path(args.config).expanduser() if args.config else Path(__file__).resolve().parents[1] / "main.yaml"
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}", file=sys.stderr)
        return 1
    cfg = Config.load_singleton(config_path)
    StructuredLogger.configure_from_config(cfg)
    logger = StructuredLogger.get_logger("cad.s2.cli")
    logger.info("s2.cli.start", config=str(config_path))
    pipeline = S2Pipeline(S2PipelineConfig.from_config(cfg))
    pipeline.run()
    logger.info("s2.cli.complete")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

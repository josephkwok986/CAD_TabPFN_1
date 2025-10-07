"""Configuration dataclasses for the S2 pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

from config import Config


_DEFAULT_EXTENSIONS = (".step", ".stp", ".stpz", ".brep", ".brp")


def _to_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    return path


def _tuple_from(value: Optional[Sequence[str]], fallback: Sequence[str]) -> Tuple[str, ...]:
    if value is None:
        return tuple(fallback)
    return tuple(str(v) for v in value)


@dataclass(frozen=True)
class SourceSettings:
    dataset: str
    tier: str
    root: Path
    pattern: str
    allowed_extensions: Tuple[str, ...]

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SourceSettings":
        dataset = str(data.get("dataset"))
        tier = str(data.get("tier"))
        root = _to_path(str(data.get("root")))
        pattern = str(data.get("pattern", "**/*"))
        allowed = _tuple_from(data.get("allowed_extensions"), _DEFAULT_EXTENSIONS)
        return cls(dataset=dataset, tier=tier, root=root, pattern=pattern, allowed_extensions=allowed)


@dataclass(frozen=True)
class SamplingSettings:
    subset_n: Optional[int]
    subset_frac: Optional[float]
    seed: int

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> "SamplingSettings":
        if data is None:
            return cls(subset_n=None, subset_frac=None, seed=42)
        subset_n = data.get("subset_n")
        subset_frac = data.get("subset_frac")
        seed = int(data.get("seed", 42))
        return cls(
            subset_n=int(subset_n) if subset_n is not None else None,
            subset_frac=float(subset_frac) if subset_frac is not None else None,
            seed=seed,
        )


@dataclass(frozen=True)
class FeatureSettings:
    descriptor_length: int
    sample_points: int
    random_seed: int

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> "FeatureSettings":
        if data is None:
            return cls(descriptor_length=64, sample_points=4096, random_seed=1337)
        return cls(
            descriptor_length=int(data.get("descriptor_length", 64)),
            sample_points=int(data.get("sample_points", 4096)),
            random_seed=int(data.get("random_seed", 1337)),
        )


@dataclass(frozen=True)
class DedupSettings:
    similarity_threshold: float
    bbox_tolerance: float

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> "DedupSettings":
        if data is None:
            return cls(similarity_threshold=0.995, bbox_tolerance=0.02)
        return cls(
            similarity_threshold=float(data.get("similarity_threshold", 0.995)),
            bbox_tolerance=float(data.get("bbox_tolerance", 0.02)),
        )


@dataclass(frozen=True)
class AutoEpsSettings:
    enabled: bool
    quantile: float
    sample_size: int
    scale: float

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> "AutoEpsSettings":
        if data is None:
            return cls(enabled=False, quantile=0.25, sample_size=10000, scale=1.0)
        return cls(
            enabled=bool(data.get("enabled", False)),
            quantile=float(data.get("quantile", 0.25)),
            sample_size=int(data.get("sample_size", data.get("sample_n", 10000))),
            scale=float(data.get("scale", 1.0)),
        )


@dataclass(frozen=True)
class ClusteringSettings:
    method: str
    eps: float
    min_samples: int
    auto_eps: AutoEpsSettings

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> "ClusteringSettings":
        if data is None:
            return cls(method="dbscan", eps=0.012, min_samples=6, auto_eps=AutoEpsSettings.from_mapping(None))
        auto = AutoEpsSettings.from_mapping(data.get("auto_eps"))
        return cls(
            method=str(data.get("method", "dbscan")),
            eps=float(data.get("eps", data.get("distance_threshold", 0.012))),
            min_samples=int(data.get("min_samples", 6)),
            auto_eps=auto,
        )


@dataclass(frozen=True)
class PartitionSettings:
    strategy: str
    constraints: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> "PartitionSettings":
        if data is None:
            return cls(strategy="weighted", constraints={"max_items_per_task": 64, "shuffle_seed": 42})
        strategy = str(data.get("strategy", "weighted"))
        constraints = dict(data.get("constraints", {}))
        return cls(strategy=strategy, constraints=constraints)


@dataclass(frozen=True)
class OutputSettings:
    root: Path
    signatures_file: str
    part_index_file: str
    part_index_for_split_file: str
    family_hist_file: str
    diagnostics_file: str
    write_signatures: bool

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "OutputSettings":
        root = _to_path(str(data.get("root")))
        return cls(
            root=root,
            signatures_file=str(data.get("signatures_file", "signatures.parquet")),
            part_index_file=str(data.get("part_index_file", "part_index.csv")),
            part_index_for_split_file=str(data.get("part_index_for_split_file", "part_index.for_split.csv")),
            family_hist_file=str(data.get("family_hist_file", "family_hist.csv")),
            diagnostics_file=str(data.get("diagnostics_file", "diagnostics.json")),
            write_signatures=bool(data.get("write_signatures", True)),
        )


@dataclass(frozen=True)
class QuarantineSettings:
    preferred_tier: str
    reason: str

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> "QuarantineSettings":
        if data is None:
            return cls(preferred_tier="gold", reason="cross_tier_duplicate")
        return cls(
            preferred_tier=str(data.get("preferred_tier", "gold")),
            reason=str(data.get("reason", "cross_tier_duplicate")),
        )


@dataclass(frozen=True)
class S2PipelineConfig:
    sources: Tuple[SourceSettings, ...]
    sampling: SamplingSettings
    features: FeatureSettings
    dedup: DedupSettings
    clustering: ClusteringSettings
    partition: PartitionSettings
    outputs: OutputSettings
    quarantine: QuarantineSettings

    @classmethod
    def from_config(cls, config: Optional[Config] = None) -> "S2PipelineConfig":
        cfg = config or Config.get_singleton()
        data = cfg.get("pipelines.s2", dict, default=None)
        if not data:
            raise RuntimeError("Missing pipelines.s2 configuration")
        sources_raw: Iterable[Mapping[str, Any]] = data.get("sources", [])
        sources = tuple(SourceSettings.from_mapping(item) for item in sources_raw)
        if not sources:
            raise RuntimeError("pipelines.s2.sources must contain at least one source")
        sampling = SamplingSettings.from_mapping(data.get("sampling"))
        features = FeatureSettings.from_mapping(data.get("features"))
        dedup = DedupSettings.from_mapping(data.get("dedup"))
        clustering = ClusteringSettings.from_mapping(data.get("clustering"))
        partition = PartitionSettings.from_mapping(data.get("partition"))
        outputs = OutputSettings.from_mapping(data.get("outputs", {}))
        quarantine = QuarantineSettings.from_mapping(data.get("quarantine"))
        return cls(
            sources=sources,
            sampling=sampling,
            features=features,
            dedup=dedup,
            clustering=clustering,
            partition=partition,
            outputs=outputs,
            quarantine=quarantine,
        )


__all__ = [
    "SourceSettings",
    "SamplingSettings",
    "FeatureSettings",
    "DedupSettings",
    "ClusteringSettings",
    "PartitionSettings",
    "OutputSettings",
    "QuarantineSettings",
    "S2PipelineConfig",
]

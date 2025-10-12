#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S2.1：基于 base_components 的几何签名提取流水线。"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import shutil
import multiprocessing as mp
import os
import re
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from base_components import (
    Config,
    ExecutorPolicy,
    ParallelExecutor,
    ProgressController,
    StructuredLogger,
    TaskPartitioner,
    TaskPool,
)
from base_components.parallel_executor import TaskResult, _persist_task_result
from base_components.task_pool import LeasedTask
from base_components.task_partitioner import PartitionConstraints

from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Reader, STEPControl_Writer
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.BRepTools import breptools_Read
from OCC.Core.BRep import BRep_Builder, BRep_Tool
from OCC.Core.TopoDS import TopoDS_Edge, TopoDS_Face, TopoDS_Shape, topods
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.GeomAbs import (
    GeomAbs_Cone,
    GeomAbs_Cylinder,
    GeomAbs_Plane,
    GeomAbs_Sphere,
    GeomAbs_Torus,
)
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface

# lazy import guards
cp = None  # type: ignore
_GPU_CUPY = False

SUPPORTED_STEP = {".step", ".stp", ".stpz"}
SUPPORTED_BREP = {".brep", ".brp"}


# ---------------------------------------------------------------------------
# 配置结构
# ---------------------------------------------------------------------------


@dataclass
class PartitionConfig:
    strategy: str = "weighted"
    group_by: Optional[str] = "source_dataset"
    max_items_per_task: int = 16
    max_weight_per_task: Optional[float] = None
    shuffle_seed: Optional[int] = 17
    max_tasks: Optional[int] = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PartitionConfig":
        return cls(
            strategy=str(data.get("strategy", "weighted")),
            group_by=data.get("group_by", "source_dataset"),
            max_items_per_task=int(data.get("max_items_per_task", 16)) if data.get("max_items_per_task") else 16,
            max_weight_per_task=(
                float(data["max_weight_per_task"])
                if data.get("max_weight_per_task") is not None
                else None
            ),
            shuffle_seed=(
                int(data["shuffle_seed"])
                if data.get("shuffle_seed") is not None
                else None
            ),
            max_tasks=int(data["max_tasks"]) if data.get("max_tasks") is not None else None,
        )


@dataclass
class SignatureConfig:
    raw_root: Path
    out_root: Path
    sample_points: int = 4096
    subset_n: Optional[int] = None
    subset_frac: Optional[float] = None
    device: str = "auto"
    demo_occ: bool = False
    partition: PartitionConfig = field(default_factory=PartitionConfig)

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "SignatureConfig":
        raw_root = Path(cfg["raw_root"]).expanduser().resolve()
        out_root = Path(cfg["out_root"]).expanduser().resolve()
        partition_cfg = PartitionConfig.from_mapping(cfg.get("partition", {}))
        subset_n = cfg.get("subset_n")
        subset_frac = cfg.get("subset_frac")
        return cls(
            raw_root=raw_root,
            out_root=out_root,
            sample_points=int(cfg.get("sample_points", 4096)),
            subset_n=int(subset_n) if subset_n is not None else None,
            subset_frac=float(subset_frac) if subset_frac is not None else None,
            device=str(cfg.get("device", "auto")),
            demo_occ=bool(cfg.get("demo_occ", False)),
            partition=partition_cfg,
        )


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------


def _ensure_cupy() -> None:
    global cp, _GPU_CUPY
    if cp is not None or _GPU_CUPY:
        return
    try:
        import cupy as _cp  # type: ignore

        cp = _cp
        _GPU_CUPY = True
    except Exception:
        cp = None
        _GPU_CUPY = False


def read_text_prefix(path: str, max_bytes: int = 200000) -> Optional[str]:
    try:
        with open(path, "rb") as handle:
            blob = handle.read(max_bytes)
    except Exception:
        return None
    for encoding in ("utf-8", "latin1"):
        with contextlib.suppress(Exception):
            return blob.decode(encoding, errors="ignore")
    return None


_STEP_FILE_NAME_RE = re.compile(
    r"FILE_NAME\s*\(\s*'([^']*)'\s*,\s*'([^']*)'\s*,\s*\((.*?)\)\s*,\s*\((.*?)\)\s*,\s*'([^']*)'\s*,\s*'([^']*)'\s*,\s*'([^']*)'\s*\)",
    re.IGNORECASE | re.DOTALL,
)
_STEP_FILE_SCHEMA_RE = re.compile(r"FILE_SCHEMA\s*\(\s*\(\s*'([^']+)'\s*\)\s*\)", re.IGNORECASE)


def parse_step_header_text(text: str) -> Dict[str, Optional[str]]:
    meta = {
        "step_schema": None,
        "originating_system": None,
        "author": None,
        "created_at_header": None,
        "unit": None,
        "unit_source": None,
    }
    if not text:
        return meta
    with contextlib.suppress(Exception):
        m_schema = _STEP_FILE_SCHEMA_RE.search(text)
        if m_schema:
            meta["step_schema"] = m_schema.group(1).strip()
        m_name = _STEP_FILE_NAME_RE.search(text)
        if m_name:
            timestamp = m_name.group(2).strip()
            meta["created_at_header"] = _normalize_datetime(timestamp)
            authors_raw = m_name.group(3).strip()
            authors = [a.strip().strip("'") for a in authors_raw.split(",") if a.strip()]
            meta["author"] = "|".join(authors) if authors else None
            meta["originating_system"] = m_name.group(6).strip().strip("'")
    up = text.upper()
    if "INCH" in up:
        meta["unit"], meta["unit_source"] = "inch", "inferred_step_data"
    elif "MILLI" in up and "METRE" in up:
        meta["unit"], meta["unit_source"] = "mm", "inferred_step_data"
    elif "METRE" in up:
        meta["unit"], meta["unit_source"] = "m", "inferred_step_data"
    return meta


def get_file_meta(path: str) -> Dict[str, Optional[str]]:
    try:
        stat = os.stat(path)
        size = int(stat.st_size)
        mtime_iso = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    except Exception:
        size = None
        mtime_iso = None
    ext = os.path.splitext(path)[1].lower()
    fmt = "STEP" if ext in SUPPORTED_STEP else ("BREP" if ext in SUPPORTED_BREP else "UNKNOWN")
    return {"file_ext": ext, "format": fmt, "file_size_bytes": size, "file_mtime_utc": mtime_iso}


def guess_kernel(fmt: str, originating_system: Optional[str]) -> Optional[str]:
    if fmt == "BREP":
        return "OpenCascade"
    if fmt == "STEP":
        system = (originating_system or "").lower()
        if any(key in system for key in ("parasolid", "nx", "solidworks")):
            return "Parasolid"
        if any(key in system for key in ("acis", "autocad", "inventor")):
            return "ACIS"
        if any(key in system for key in ("catia", "cgm", "3dexperience", "solidedge")):
            return "CGM"
        return "STEP"
    return None


def occ_load_shape(path: str, logger: StructuredLogger) -> Optional[TopoDS_Shape]:
    ext = os.path.splitext(path)[1].lower()
    if ext in SUPPORTED_STEP:
        reader = STEPControl_Reader()
        status = reader.ReadFile(path)
        if status != IFSelect_RetDone:
            logger.debug("occ.step.read_failed", path=path)
            return None
        reader.TransferRoots()
        return reader.OneShape()
    if ext in SUPPORTED_BREP:
        shape = TopoDS_Shape()
        try:
            ok = breptools_Read(shape, path, BRep_Builder())
        except TypeError:
            ok = breptools_Read(shape, path)
        if not ok:
            logger.debug("occ.brep.read_failed", path=path)
            return None
        return shape
    return None


def triangulate(shape: TopoDS_Shape, lin_defl: float = 0.5) -> None:
    with contextlib.suppress(TypeError):
        BRepMesh_IncrementalMesh(shape, lin_defl, True, 0.5, True)
        return
    BRepMesh_IncrementalMesh(shape, deflection=lin_defl, isRelative=True, angle=0.5, parallel=True)


def _append_nodes_from_triangulation(face: TopoDS_Face, loc: TopLoc_Location, pts_out: List[List[float]]) -> int:
    tri = BRep_Tool.Triangulation(face, loc)
    if not tri:
        return 0
    tri_obj = tri.GetObject() if hasattr(tri, "GetObject") else tri
    added = 0
    try:
        if hasattr(tri_obj, "Nodes") and hasattr(tri_obj, "NbNodes"):
            nodes = tri_obj.Nodes()
            n = tri_obj.NbNodes()
            for i in range(1, n + 1):
                pnt = nodes.Value(i).Transformed(loc.Transformation())
                pts_out.append([pnt.X(), pnt.Y(), pnt.Z()])
                added += 1
            return added
    except Exception:
        pass
    try:
        if hasattr(tri_obj, "NbNodes") and hasattr(tri_obj, "Node"):
            n = tri_obj.NbNodes()
            for i in range(1, n + 1):
                pnt = tri_obj.Node(i).Transformed(loc.Transformation())
                pts_out.append([pnt.X(), pnt.Y(), pnt.Z()])
                added += 1
            return added
    except Exception:
        pass
    return added


def occ_shape_points(shape: TopoDS_Shape, n: int = 4096, lin_defl: float = 0.5) -> np.ndarray:
    triangulate(shape, lin_defl=lin_defl)
    pts: List[List[float]] = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = topods.Face(exp.Current())
        loc = TopLoc_Location()
        _append_nodes_from_triangulation(face, loc, pts)
        exp.Next()
    if not pts:
        bbox = Bnd_Box()
        brepbndlib_Add(shape, bbox, False)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        pts = [
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmin, ymax, zmin],
            [xmax, ymax, zmin],
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmin, ymax, zmax],
            [xmax, ymax, zmax],
        ]
    arr = np.asarray(pts, dtype=np.float64)
    if arr.shape[0] > n:
        idx = np.random.default_rng(0).choice(arr.shape[0], size=n, replace=False)
        arr = arr[idx]
    return arr


def normalize_points(points: np.ndarray) -> Tuple[np.ndarray, float]:
    pts = points.astype(np.float64)
    centroid = pts.mean(axis=0, keepdims=True)
    pts -= centroid
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    diag = float(np.linalg.norm(maxs - mins) + 1e-9)
    pts /= diag
    return pts, diag


def d2_descriptor(points: np.ndarray, num_pairs: int = 20000, num_bins: int = 64, seed: int = 42, device: str = "auto") -> np.ndarray:
    n = points.shape[0]
    if n < 4:
        raise ValueError("Not enough points for D2 descriptor")
    use_gpu = device == "gpu"
    if use_gpu:
        _ensure_cupy()
    if use_gpu and _GPU_CUPY and cp is not None:
        rng = cp.random.default_rng(seed)
        i = rng.integers(0, n, size=num_pairs)
        j = rng.integers(0, n, size=num_pairs)
        p = cp.asarray(points)
        d = cp.linalg.norm(p[i] - p[j], axis=1)
        d = cp.clip(d / (math.sqrt(3) + 1e-9), 0.0, 1.0)
        hist, _ = cp.histogram(d, bins=num_bins, range=(0, 1))
        hist = hist.astype(cp.float64)
        hist /= hist.sum() + 1e-12
        return cp.asnumpy(hist)
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, size=num_pairs)
    j = rng.integers(0, n, size=num_pairs)
    d = np.linalg.norm(points[i] - points[j], axis=1)
    d = np.clip(d / (math.sqrt(3) + 1e-9), 0.0, 1.0)
    hist, _ = np.histogram(d, bins=num_bins, range=(0, 1))
    hist = hist.astype(np.float64)
    hist /= hist.sum() + 1e-12
    return hist


def bbox_ratios(points: np.ndarray) -> np.ndarray:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    lengths = np.sort((maxs - mins).astype(np.float64))
    l1, l2, l3 = lengths
    return np.array([l1 / (l3 + 1e-12), l2 / (l3 + 1e-12)], dtype=np.float64)


def occ_surface_type_hist(shape: TopoDS_Shape) -> np.ndarray:
    counters = dict(plane=0, cylinder=0, cone=0, sphere=0, torus=0, free=0)
    total = 0
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = topods.Face(exp.Current())
        adaptor = BRepAdaptor_Surface(face, True)
        surface_type = adaptor.GetType()
        if surface_type == GeomAbs_Plane:
            counters["plane"] += 1
        elif surface_type == GeomAbs_Cylinder:
            counters["cylinder"] += 1
        elif surface_type == GeomAbs_Cone:
            counters["cone"] += 1
        elif surface_type == GeomAbs_Sphere:
            counters["sphere"] += 1
        elif surface_type == GeomAbs_Torus:
            counters["torus"] += 1
        else:
            counters["free"] += 1
        total += 1
        exp.Next()
    total = max(total, 1)
    return np.array([counters[key] / total for key in ["plane", "cylinder", "cone", "sphere", "torus", "free"]], dtype=np.float64)


def occ_dihedral_hist(shape: TopoDS_Shape, bins: int = 36) -> np.ndarray:
    def edge_key(edge: TopoDS_Edge) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        curve = BRepAdaptor_Curve(edge)
        u0, u1 = curve.FirstParameter(), curve.LastParameter()
        p0, p1 = curve.Value(u0), curve.Value(u1)
        a = np.round([p0.X(), p0.Y(), p0.Z()], 6)
        b = np.round([p1.X(), p1.Y(), p1.Z()], 6)
        t0, t1 = tuple(a), tuple(b)
        return (t0, t1) if t0 <= t1 else (t1, t0)

    edge_map: Dict[Tuple[Tuple[float, float, float], Tuple[float, float, float]], Dict[str, Any]] = {}
    fexp = TopExp_Explorer(shape, TopAbs_FACE)
    while fexp.More():
        face = topods.Face(fexp.Current())
        eexp = TopExp_Explorer(face, TopAbs_EDGE)
        while eexp.More():
            edge = topods.Edge(eexp.Current())
            try:
                key = edge_key(edge)
            except Exception:
                eexp.Next()
                continue
            rec = edge_map.get(key)
            if rec is None:
                rec = {"edge": edge, "faces": []}
                edge_map[key] = rec
            rec["faces"].append(face)
            eexp.Next()
        fexp.Next()

    angles: List[float] = []

    def face_normal(face: TopoDS_Face, point) -> Optional[np.ndarray]:
        surf = BRep_Tool.Surface(face)
        uv = ShapeAnalysis_Surface(surf).ValueOfUV(point, 1e-6)
        props = GeomLProp_SLProps(surf, uv.X(), uv.Y(), 1, 1e-6)
        if not props.IsNormalDefined():
            return None
        normal = props.Normal()
        vec = np.array([normal.X(), normal.Y(), normal.Z()], dtype=np.float64)
        if face.Orientation() == TopAbs_REVERSED:
            vec = -vec
        return vec / (np.linalg.norm(vec) + 1e-12)

    for rec in edge_map.values():
        faces = rec["faces"]
        if len(faces) < 2:
            continue
        edge = rec["edge"]
        curve = BRepAdaptor_Curve(edge)
        mid = 0.5 * (curve.FirstParameter() + curve.LastParameter())
        point = curve.Value(mid)
        n1 = face_normal(faces[0], point)
        n2 = face_normal(faces[1], point)
        if n1 is None or n2 is None:
            continue
        cosine = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
        angles.append(math.acos(cosine))

    if not angles:
        return np.zeros(bins, dtype=np.float64)
    hist, _ = np.histogram(angles, bins=bins, range=(0, math.pi), density=False)
    hist = hist.astype(np.float64)
    hist /= hist.sum() + 1e-12
    return hist


@dataclass
class PartSig:
    part_id: str
    rel_path: str
    source_dataset: str
    content_hash: str
    has_points: bool
    d2_hist: Optional[np.ndarray] = None
    bbox_ratio: Optional[np.ndarray] = None
    surf_hist: Optional[np.ndarray] = None
    dih_hist: Optional[np.ndarray] = None
    geom_hash: Optional[str] = None


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def safe_rel(root: Path, path: Path) -> str:
    try:
        return os.path.relpath(path, root)
    except Exception:
        return str(path)


def get_file_meta_all(raw_root: Path, fpath: Path) -> Dict[str, Optional[str]]:
    rel = safe_rel(raw_root, fpath)
    src = rel.split(os.sep)[0] if os.sep in rel else "unknown"
    ch = sha256_file(str(fpath))
    fm = get_file_meta(str(fpath))
    text = None
    if fm["format"] == "STEP" and fm["file_ext"] != ".stpz":
        text = read_text_prefix(str(fpath), 300000)
    step_hdr = parse_step_header_text(text or "")
    kernel = guess_kernel(fm["format"], step_hdr.get("originating_system"))
    parts = rel.split(os.sep)
    repo = parts[1] if len(parts) >= 2 else None
    created_hdr = step_hdr.get("created_at_header")
    ts_src = "header" if created_hdr else ("mtime" if fm.get("file_mtime_utc") else "none")
    return {
        "rel": rel,
        "src": src,
        "ch": ch,
        **fm,
        **step_hdr,
        "kernel": kernel,
        "repo": repo,
        "domain_hint": src,
        "timestamp_source": ts_src,
    }


def _count_parts(raw_root: Path) -> Tuple[int, int, int]:
    total = 0
    step_cnt = 0
    brep_cnt = 0
    for dirpath, _, filenames in os.walk(raw_root, followlinks=True):
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext in SUPPORTED_STEP | SUPPORTED_BREP:
                total += 1
                if ext in SUPPORTED_STEP:
                    step_cnt += 1
                else:
                    brep_cnt += 1
    return total, step_cnt, brep_cnt


def discover_parts(
    raw_root: Path,
    subset_n: Optional[int],
    subset_frac: Optional[float],
    logger: StructuredLogger,
) -> Tuple[int, Iterable[Path]]:
    total, step_cnt, brep_cnt = _count_parts(raw_root)
    if total == 0:
        logger.debug("s2.1_signature.discover", step_count=0, brep_count=0, total=0, selected=0)
        return 0, []

    limit = total
    if subset_frac and 0 < subset_frac < 1:
        limit = min(limit, max(1, int(total * subset_frac)))
    if subset_n:
        limit = min(limit, subset_n)

    logger.debug(
        "s2.1_signature.discover.start",
        total=total,
        step_count=step_cnt,
        brep_count=brep_cnt,
        limit=limit,
    )

    def iterator() -> Iterable[Path]:
        emitted = 0
        selected_step = 0
        selected_brep = 0
        for dirpath, _, filenames in os.walk(raw_root, followlinks=True):
            for name in filenames:
                ext = os.path.splitext(name)[1].lower()
                if ext not in SUPPORTED_STEP | SUPPORTED_BREP:
                    continue
                if limit and emitted >= limit:
                    break
                full = Path(dirpath) / name
                if ext in SUPPORTED_STEP:
                    selected_step += 1
                else:
                    selected_brep += 1
                emitted += 1
                if emitted % 100 == 0:
                    logger.debug("s2.1_signature.discover.progress", emitted=emitted, limit=limit)
                yield full
            if limit and emitted >= limit:
                break
        logger.debug(
            "s2.1_signature.discover",
            step_count=selected_step,
            brep_count=selected_brep,
            total=total,
            selected=emitted,
        )

    return limit, iterator()


def compute_signature_for_file(
    raw_root: Path,
    fpath: Path,
    sample_points: int,
    logger: StructuredLogger,
    *,
    device: str = "auto",
) -> Tuple[Optional[PartSig], Optional[str], Dict[str, Optional[str]]]:
    meta = get_file_meta_all(raw_root, fpath)
    rel = meta["rel"] or fpath.name
    src = meta["src"] or "unknown"
    content_hash = meta["ch"] or ""
    shape = occ_load_shape(str(fpath), logger)
    if shape is None:
        sig = PartSig(rel, rel, src, content_hash, False, None, None, None, None, None)
        return sig, f"load_shape_failed:{rel}", meta
    try:
        points = occ_shape_points(shape, n=sample_points)
        if points.shape[0] >= 4:
            normed, _ = normalize_points(points)
            d2 = d2_descriptor(normed, device=device)
            br = bbox_ratios(normed)
            surf = occ_surface_type_hist(shape)
            dih = occ_dihedral_hist(shape, bins=36)
            geom_hash = hashlib.sha256(
                np.round(np.concatenate([d2, br, surf, dih]), 3).tobytes()
            ).hexdigest()
            sig = PartSig(rel, rel, src, content_hash, True, d2, br, surf, dih, geom_hash)
        else:
            sig = PartSig(rel, rel, src, content_hash, False, None, None, None, None, None)
        return sig, None, meta
    except Exception as exc:
        logger.debug("s2.1_signature.compute_failed", path=str(fpath), error=str(exc), traceback=traceback.format_exc())
        return None, f"signature_failed:{rel}:{type(exc).__name__}", meta


def _encode_array(arr: Optional[np.ndarray]) -> Optional[str]:
    if arr is None:
        return None
    return json.dumps([float(x) for x in np.asarray(arr).ravel().tolist()])


def build_row(sig: Optional[PartSig], meta: Mapping[str, Any], error: Optional[str]) -> Dict[str, Any]:
    rel_path = meta.get("rel") or (sig.rel_path if sig else None) or "unknown"
    source = meta.get("src") or (sig.source_dataset if sig else "unknown")
    content_hash = meta.get("ch") or (sig.content_hash if sig else "")
    part_id = sig.part_id if sig else rel_path
    row = {
        "part_id": part_id,
        "rel_path": rel_path,
        "source_dataset": source,
        "content_hash": content_hash,
        "has_points": int(sig.has_points) if sig else 0,
        "d2_hist": _encode_array(sig.d2_hist if sig and sig.has_points else None),
        "bbox_ratio": _encode_array(sig.bbox_ratio if sig and sig.has_points else None),
        "surf_hist": _encode_array(sig.surf_hist if sig and sig.has_points else None),
        "dih_hist": _encode_array(sig.dih_hist if sig and sig.has_points else None),
        "geom_hash": sig.geom_hash if sig and sig.geom_hash else None,
        "meta_json": json.dumps(meta, ensure_ascii=False),
        "error": error or "",
    }
    return row


def prefer_fork_context() -> None:
    if getattr(prefer_fork_context, "_patched", False):
        return
    original_get_context = mp.get_context

    def _patched(method: Optional[str] = None):
        target = method
        if method == "spawn":
            try:
                return original_get_context("fork")
            except ValueError:
                target = method
        return original_get_context(target)

    mp.get_context = _patched  # type: ignore[assignment]
    prefer_fork_context._patched = True  # type: ignore[attr-defined]


def run_sequential_executor(
    pool: TaskPool,
    handler,
    context: Dict[str, Any],
    result_handler,
    policy: ExecutorPolicy,
    driver_logger: StructuredLogger,
) -> None:
    worker_logger = StructuredLogger.get_logger("app.s2.1_signature.sequential")
    lease_ttl = getattr(policy, "lease_ttl", None)
    if lease_ttl is None or lease_ttl <= 0:
        lease_ttl = 10.0
    filters = getattr(policy, "filters", None)

    while True:
        leased_batch = pool.lease(1, lease_ttl, filters=filters)
        if not leased_batch:
            break
        leased = leased_batch[0]
        try:
            result = handler(leased, context)
            _persist_task_result(result, worker_logger, leased.task.task_id)
            pool.ack(leased.task.task_id)
            result_handler(leased, result)
        except Exception as exc:
            pool.nack(leased.task.task_id, requeue=False, delay=None)
            driver_logger.error("s2.1_signature.sequential_error", task_id=leased.task.task_id, error=str(exc))


def aggregate_outputs(output_root: Path, logger: StructuredLogger) -> Dict[str, Any]:
    cache_dir = output_root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames: List[pd.DataFrame] = []
    for csv_path in sorted(cache_dir.glob("*.csv")):
        try:
            if csv_path.stat().st_size == 0:
                continue
        except FileNotFoundError:
            continue
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                frames.append(df)
        except Exception as exc:
            logger.warning("s2.1_signature.load_cache_failed", file=str(csv_path), error=str(exc))
    if frames:
        df_all = pd.concat(frames, ignore_index=True)
    else:
        df_all = pd.DataFrame(
            columns=[
                "part_id",
                "rel_path",
                "source_dataset",
                "content_hash",
                "has_points",
                "d2_hist",
                "bbox_ratio",
                "surf_hist",
                "dih_hist",
                "geom_hash",
                "meta_json",
                "error",
            ]
        )

    sig_rows: List[Dict[str, Any]] = []
    metas: Dict[str, Any] = {}
    failures: List[str] = []

    for _, row in df_all.iterrows():
        err_val = row.get("error")
        if pd.isna(err_val):
            error = ""
        else:
            error = str(err_val or "").strip()
        meta_raw = row.get("meta_json")
        try:
            meta = json.loads(meta_raw) if isinstance(meta_raw, str) and meta_raw else {}
        except Exception:
            meta = {}
        part_id = str(row.get("part_id"))
        if error:
            failures.append(error)
            continue
        entry = {
            "part_id": part_id,
            "rel_path": row.get("rel_path"),
            "source_dataset": row.get("source_dataset"),
            "content_hash": row.get("content_hash"),
            "has_points": int(row.get("has_points", 0)),
            "d2_hist": row.get("d2_hist"),
            "bbox_ratio": row.get("bbox_ratio"),
            "surf_hist": row.get("surf_hist"),
            "dih_hist": row.get("dih_hist"),
            "geom_hash": row.get("geom_hash"),
        }
        sig_rows.append(entry)
        metas[part_id] = meta

    columns_order = [
        "part_id",
        "rel_path",
        "source_dataset",
        "content_hash",
        "has_points",
        "d2_hist",
        "bbox_ratio",
        "surf_hist",
        "dih_hist",
        "geom_hash",
    ]
    signatures_path = output_root / "signatures.csv"
    metas_path = output_root / "metas.json"
    failures_path = output_root / "failures.txt"
    summary_path = output_root / "summary_s2_1.json"

    if sig_rows:
        pd.DataFrame(sig_rows)[columns_order].to_csv(signatures_path, index=False)
    else:
        pd.DataFrame(columns=columns_order).to_csv(signatures_path, index=False)
    metas_path.write_text(json.dumps(metas, indent=2, ensure_ascii=False), encoding="utf-8")
    if failures:
        failures_path.write_text("\n".join(sorted(set(failures))), encoding="utf-8")
    else:
        if failures_path.exists():
            failures_path.unlink()
    summary = {
        "num_parts": len(sig_rows),
        "num_failed": len(failures),
        "gpu_available": {"cupy": _GPU_CUPY},
        "out_files": {
            "signatures": str(signatures_path),
            "metas": str(metas_path),
            "failures": str(failures_path) if failures else None,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.debug(
        "s2.1_signature.summary",
        parts=len(sig_rows),
        failed=len(failures),
        signatures=str(signatures_path),
        metas=str(metas_path),
    )
    return summary


def iter_items(paths: Iterable[Path], raw_root: Path, items_map: Dict[str, Dict[str, Any]], logger: StructuredLogger) -> Iterable[Dict[str, Any]]:
    for idx, path in enumerate(paths, 1):
        rel = safe_rel(raw_root, path)
        dataset = rel.split(os.sep)[0] if os.sep in rel else "unknown"
        try:
            size = float(os.path.getsize(path))
        except Exception:
            size = 0.0
        weight = max(1.0, size / (1024.0 * 1024.0))
        metadata = {
            "abs_path": str(path),
            "source_dataset": dataset,
            "file_ext": os.path.splitext(path)[1].lower(),
            "rel_path": rel,
        }
        items_map[rel] = metadata
        if idx % 100 == 0:
            logger.debug("s2.1_signature.scan_progress", processed=idx)
        yield {
            "payload_ref": rel,
            "weight": weight,
            "metadata": metadata,
        }


def signature_task_handler(leased: LeasedTask, context: Dict[str, Any]) -> TaskResult:
    logger = StructuredLogger.get_logger("app.s2.1_signature.worker")
    raw_root: Path = context["raw_root"]
    sample_points: int = context["sample_points"]
    device: str = context["device"]
    items_map: Dict[str, Dict[str, Any]] = context["items_map"]
    rows: List[Dict[str, Any]] = []
    failures: List[str] = []
    for ref in leased.task.payload_ref:
        meta_hint = items_map.get(ref, {})
        abs_path = meta_hint.get("abs_path", ref)
        sig, err, meta = compute_signature_for_file(raw_root, Path(abs_path), sample_points, logger, device=device)
        if meta_hint:
            meta = {**meta_hint, **meta}
        row = build_row(sig, meta, err)
        rows.append(row)
        if err:
            failures.append(err)
    metadata = {
        "task_id": leased.task.task_id,
        "processed": len(rows),
        "failures": failures,
        "group_keys": list(leased.task.group_keys),
    }
    progress = context.get("progress")
    if progress is not None:
        progress.advance(len(rows))
    return TaskResult(
        payload=rows,
        processed=len(rows),
        metadata=metadata,
        output_directory=context["output_root"],
        output_filename=f"{leased.task.task_id}",
        is_final_output=False,
    )


class SignatureExtractionRunner:
    """S2.1 签名提取主控程序。"""

    def __init__(self, config_path: Optional[Path] = None) -> None:
        self._config_path = Path(config_path).expanduser().resolve() if config_path else None

    def _ensure_config(self) -> Config:
        log_dir = Path(__file__).resolve().parents[2] / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        Config.set_singleton(None)
        if self._config_path is not None:
            os.environ["CAD_TASK_CONFIG"] = str(self._config_path)
        else:
            default_cfg = Path(__file__).resolve().parents[2] / "main.yaml"
            os.environ.setdefault("CAD_TASK_CONFIG", str(default_cfg))
        cfg = Config.load_singleton(os.environ["CAD_TASK_CONFIG"])
        StructuredLogger.configure_from_config(cfg)
        return cfg

    def run(self) -> Dict[str, Any]:
        cfg = self._ensure_config()
        s2_block = cfg.get("s2", dict, default={})
        section = s2_block.get("s2.1_signature")
        if section is None:
            raise RuntimeError("缺少 s2.1_signature 配置段")
        sig_cfg = SignatureConfig.from_config(section)

        logger = StructuredLogger.get_logger("app.s2.1_signature")
        logger.debug(
            "s2.1_signature.start",
            raw_root=str(sig_cfg.raw_root),
            out_root=str(sig_cfg.out_root),
            device=sig_cfg.device,
            sample_points=sig_cfg.sample_points,
        )
        sig_cfg.out_root.mkdir(parents=True, exist_ok=True)

        raw_root = sig_cfg.raw_root
        if sig_cfg.demo_occ:
            from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder

            demo_root = sig_cfg.out_root / "demo_occ"
            demo_root.mkdir(parents=True, exist_ok=True)

            def _write_step(shape, path: Path) -> bool:
                writer = STEPControl_Writer()
                writer.Transfer(shape, STEPControl_AsIs)
                return writer.Write(str(path)) == IFSelect_RetDone

            ok = []
            ok.append(_write_step(BRepPrimAPI_MakeBox(10, 10, 10).Solid(), demo_root / "DATASETA_box.step"))
            ok.append(_write_step(BRepPrimAPI_MakeBox(10, 10, 10).Solid(), demo_root / "DATASETA_box_dup.step"))
            ok.append(_write_step(BRepPrimAPI_MakeBox(13, 13, 13).Solid(), demo_root / "DATASETB_box_scaled.step"))
            ok.append(_write_step(BRepPrimAPI_MakeCylinder(4.0, 12.0).Solid(), demo_root / "DATASETB_cylinder.step"))
            logger.debug("s2.1_signature.demo_generated", ok=sum(ok), total=4, path=str(demo_root))
            raw_root = demo_root

        limit, path_iter = discover_parts(raw_root, sig_cfg.subset_n, sig_cfg.subset_frac, logger)
        if limit == 0:
            logger.warning("s2.1_signature.no_files", raw_root=str(raw_root))
            return aggregate_outputs(sig_cfg.out_root, logger)

        queue_path = cfg.get("task_system.pool.backend.path", str, default=None)
        if queue_path:
            qdir = Path(queue_path).expanduser().resolve()
            if qdir.exists():
                shutil.rmtree(qdir)
        items_map: Dict[str, Dict[str, Any]] = {}
        item_iter = iter_items(path_iter, raw_root, items_map, logger)
        constraints = PartitionConstraints(
            max_items_per_task=sig_cfg.partition.max_items_per_task,
            max_weight_per_task=sig_cfg.partition.max_weight_per_task,
            shuffle_seed=sig_cfg.partition.shuffle_seed,
        )
        pool = TaskPool()
        weights: List[float] = []
        task_count = 0
        for task in TaskPartitioner.iter_tasks({"job_id": "s2.1-signature"}, item_iter, sig_cfg.partition.strategy, constraints=constraints):
            pool.put(task)
            weights.append(task.weight)
            task_count += 1
        total_files = len(items_map)
        logger.debug(
            "s2.1_signature.plan_ready",
            task_count=task_count,
            total_weight=sum(weights),
        )
        logger.debug("s2.1_signature.pool_stats", stats=dict(pool.stats()))

        policy = ExecutorPolicy()
        collected_metadata: List[Dict[str, Any]] = []

        def on_result(_leased: LeasedTask, result: TaskResult) -> None:
            collected_metadata.append(result.metadata)
            logger.debug(
                "s2.1_signature.task_complete",
                task_id=result.metadata.get("task_id"),
                processed=result.metadata.get("processed"),
                failures=len(result.metadata.get("failures") or []),
                cache_path=result.written_path,
            )

        prefer_fork_context()
        with ProgressController(total_units=max(1, total_files), description="S2.1 签名提取") as progress:
            context = {
                "raw_root": raw_root,
                "sample_points": sig_cfg.sample_points,
                "device": sig_cfg.device,
                "items_map": items_map,
                "output_root": str(sig_cfg.out_root),
                "progress": progress.make_proxy(),
            }
            try:
                ParallelExecutor.run(
                    handler=signature_task_handler,
                    pool=pool,
                    policy=policy,
                    handler_context=context,
                    result_handler=on_result,
                    console_min_level="INFO",
                )
            except PermissionError as exc:
                logger.warning("s2.1_signature.parallel_unavailable", error=str(exc))
                run_sequential_executor(pool, signature_task_handler, context, on_result, policy, logger)
            except OSError as exc:
                logger.warning("s2.1_signature.parallel_oserror", error=str(exc))
                run_sequential_executor(pool, signature_task_handler, context, on_result, policy, logger)

        summary = aggregate_outputs(sig_cfg.out_root, logger)
        logger.debug("s2.1_signature.done", summary=summary, tasks=len(collected_metadata))
        return summary


def _normalize_datetime(value: str) -> Optional[str]:
    text = (value or "").strip()
    if not text:
        return None
    for candidate in (text, text.replace(" ", "T")):
        with contextlib.suppress(ValueError):
            dt = datetime.fromisoformat(candidate)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc).isoformat()
            return dt.astimezone(timezone.utc).isoformat()
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y%m%dT%H%M%S", "%Y-%m-%d %H:%M:%S"):
        with contextlib.suppress(ValueError):
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc).isoformat()
    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 S2.1 签名提取流水线")
    parser.add_argument(
        "-config",
        "--config",
        dest="config",
        default=None,
        help="配置文件路径（默认读取 main.yaml）",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    runner = SignatureExtractionRunner(config_path=args.config)
    runner.run()


if __name__ == "__main__":
    main()

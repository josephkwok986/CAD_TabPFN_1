#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S2.1：基于 TaskFramework 的几何签名提取流水线（增强字段版）。
变更要点：
- 在 worker 流程中补充计算并输出以下字段：
  dataset_tier, unit_recorded, unit_source, timestamp_raw, timestamp_source, domain,
  d2_000…d2_063, dih_000…dih_035, surf_000…surf_005, d2_bins_id, dih_bins_id, surf_order_id,
  sig_version, code_git_sha, sig_sha256, geom_hash_scale_inv,
  bbox_long_mm, bbox_mid_mm, bbox_short_mm, pca_ratio_1, pca_ratio_2, bbox_anisotropy_log,
  vol_mm3, area_mm2, n_faces, n_edges, n_verts, euler_char,
  n_plane, n_cyl, n_cone, n_sphere, n_torus, n_other,
  geom_hash_scale_aware, avg_face_degree, face_degree_mean, face_degree_std,
  quality_flags, hist_norm_l1, run_datetime_utc
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import shutil
import multiprocessing as mp
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from base_components import (
    Config,
    ProgressController,
    StructuredLogger,
    TaskPartitioner,
)
from base_components.main_framework import TaskFramework  # 继承框架
from base_components.task_partitioner import PartitionConstraints, TaskRecord
from base_components.task_pool import LeasedTask
from base_components.parallel_executor import TaskResult, _persist_task_result

from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Reader, STEPControl_Writer
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.BRepTools import breptools_Read
from OCC.Core.BRep import BRep_Builder, BRep_Tool
from OCC.Core.TopoDS import TopoDS_Edge, TopoDS_Face, TopoDS_Shape, topods
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED, TopAbs_VERTEX
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
try:  # 新旧 API 兼容
    from OCC.Core import BRepBndLib  # type: ignore
except ImportError:  # pragma: no cover - 旧版本仅提供函数
    BRepBndLib = None  # type: ignore
from OCC.Core.BRepBndLib import brepbndlib_Add  # type: ignore

try:
    from OCC.Core import BRepGProp  # type: ignore
except ImportError:  # pragma: no cover
    BRepGProp = None  # type: ignore
from OCC.Core.BRepGProp import brepgprop_VolumeProperties, brepgprop_SurfaceProperties  # type: ignore
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface
from OCC.Core.GProp import GProp_GProps

# lazy import guards
cp = None  # type: ignore
_GPU_CUPY = False

SUPPORTED_STEP = {".step", ".stp", ".stpz"}
SUPPORTED_BREP = {".brep", ".brp"}

# ---------------------------
# 配置
# ---------------------------

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
            max_weight_per_task=(float(data["max_weight_per_task"]) if data.get("max_weight_per_task") is not None else None),
            shuffle_seed=(int(data["shuffle_seed"]) if data.get("shuffle_seed") is not None else None),
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

# ---------------------------
# 工具
# ---------------------------

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

# 粗糙 STEP 单位检测（header 与 data 前缀文本内搜索）
def _parse_step_unit(text: str) -> Tuple[Optional[str], Optional[str]]:
    if not text:
        return None, None
    t = text.upper()
    # INCH 优先
    if re.search(r"\bINCH\b|\bINCH_UNIT\b", t):
        return "inch", "header"
    # MILLI METRE => mm
    if re.search(r"SI_UNIT\s*\(\s*\.\s*MILLI\s*\.\s*\)\s*;\s*LENGTH_UNIT\s*;.*\b(METRE|METERS?)\b", t, re.DOTALL):
        return "mm", "header"
    # METRE => m
    if re.search(r"LENGTH_UNIT\s*;.*\b(METRE|METERS?)\b", t, re.DOTALL) or "SI_UNIT(.NONE.)" in t:
        return "m", "header"
    # CENTI/DECI 微米等
    if "SI_UNIT(.CENTI.)" in t:
        return "cm", "header"
    if "SI_UNIT(.MICRO.)" in t:
        return "um", "header"
    return None, None

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
            authors = [a.strip().strip("'") for a in authors_raw.split(",")]
            meta["author"] = authors[0] if authors else None
            meta["originating_system"] = m_name.group(5).strip()
        u, src = _parse_step_unit(text)
        if u:
            meta["unit"] = u
            meta["unit_source"] = src
    return meta

def _normalize_datetime(text: str) -> Optional[str]:
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d"):
        with contextlib.suppress(Exception):
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc).isoformat()
    return None

def safe_rel(root: Path, path: Path) -> str:
    try:
        return os.path.relpath(path, root)
    except Exception:
        return str(path)

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
    return {
        "file_ext": ext,
        "format": fmt,
        "file_size_bytes": size,
        "file_mtime_utc": mtime_iso,
    }


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


def get_file_meta_all(raw_root: Path, fpath: Path) -> Dict[str, Optional[str]]:
    rel = safe_rel(raw_root, fpath)
    src = rel.split(os.sep)[0] if os.sep in rel else "unknown"
    ch = sha256_file(str(fpath)) if fpath.exists() else ""
    fm = get_file_meta(str(fpath))
    text = None
    if fm["format"] == "STEP" and fm["file_ext"] != ".stpz":
        text = read_text_prefix(str(fpath), 300000)
    step_hdr = parse_step_header_text(text or "")
    kernel = guess_kernel(fm["format"], step_hdr.get("originating_system"))
    parts = rel.split(os.sep)
    repo = parts[1] if len(parts) >= 2 else None
    created_hdr = step_hdr.get("created_at_header")
    # 统一 timestamp_source 值域
    if created_hdr:
        ts_src = "header"
    elif fm.get("file_mtime_utc"):
        ts_src = "file_mtime"
    else:
        ts_src = "unknown"
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
        "step_header_text_present": bool(text),
    }

import os, math
from pathlib import Path
from typing import Iterable, Tuple, Optional

def _norm_exts(exts):
    out = []
    for e in exts:
        e = e.lower()
        if not e.startswith("."):
            e = "." + e
        out.append(e)
    return tuple(out)

def _walk_lex(root: Path, exts: tuple) -> Iterable[str]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        filenames.sort()
        for name in filenames:
            if name.lower().endswith(exts):
                yield os.path.join(dirpath, name)

def discover_parts(raw_root: Path, subset_n: Optional[int], subset_frac: Optional[float], logger) -> Tuple[int, Iterable[Path]]:
    logger.info("s2.1_signature.scan_start", root=str(raw_root))
    exts = _norm_exts(SUPPORTED_STEP | SUPPORTED_BREP)

    if subset_n is not None:
        k = max(0, int(subset_n))
        picked = []
        for i, p in enumerate(_walk_lex(raw_root, exts)):
            if i >= k:
                break
            picked.append(Path(p))
        total = i if k == 0 else (i if len(picked) < k else i + 1)
        logger.info("s2.1_signature.scan_done", total=total, picked=len(picked), root=str(raw_root))
        return len(picked), picked

    if subset_frac is not None:
        total = sum(1 for _ in _walk_lex(raw_root, exts))
        k = max(0, min(total, int(math.ceil(float(subset_frac) * total))))
        picked = []
        for i, p in enumerate(_walk_lex(raw_root, exts)):
            if i >= k:
                break
            picked.append(Path(p))
        logger.info("s2.1_signature.scan_done", total=total, picked=len(picked), root=str(raw_root))
        return len(picked), picked

    picked = [Path(p) for p in _walk_lex(raw_root, exts)]
    total = len(picked)
    logger.info("s2.1_signature.scan_done", total=total, picked=total, root=str(raw_root))
    return total, picked

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
        yield {"payload_ref": rel, "weight": weight, "metadata": metadata}

# ---------------------------
# 几何签名与属性
# ---------------------------

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
    surf_counts: Optional[Tuple[int,int,int,int,int,int]] = None  # plane,cyl,cone,sphere,torus,free
    dih_hist: Optional[np.ndarray] = None
    geom_hash_scale_inv: Optional[str] = None
    # 新增：形状与拓扑统计
    bbox_lengths_model: Optional[Tuple[float,float,float]] = None  # long, mid, short（模型原单位）
    diag_model: Optional[float] = None
    pca_ratio_1: Optional[float] = None
    pca_ratio_2: Optional[float] = None
    bbox_anisotropy_log: Optional[float] = None
    vol_model: Optional[float] = None
    area_model: Optional[float] = None
    n_faces: Optional[int] = None
    n_edges: Optional[int] = None
    n_verts: Optional[int] = None
    euler_char: Optional[int] = None
    avg_face_degree: Optional[float] = None
    face_degree_mean: Optional[float] = None
    face_degree_std: Optional[float] = None
    geom_hash_scale_aware: Optional[str] = None

def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()

def _encode_array(arr: Optional[np.ndarray]) -> str:
    if arr is None:
        return ""
    return json.dumps(arr.tolist(), ensure_ascii=False)

def _infer_dataset_tier(rel_or_src: str) -> str:
    s = (rel_or_src or "").lower()
    if "gold" in s:
        return "gold"
    if "silver" in s:
        return "silver"
    # 默认银层，便于隔离（可在上游配置覆盖）
    return "silver"

def _infer_unit_from_filename(path: str) -> Optional[str]:
    name = os.path.basename(path).lower()
    if re.search(r"[_\-\.]mm\b", name) or re.search(r"\bmm[_\-\.]", name):
        return "mm"
    if re.search(r"[_\-\.]in(ch)?\b", name) or re.search(r'\b\d+(\.\d+)?"', name):
        return "inch"
    if re.search(r"[_\-\.]cm\b", name):
        return "cm"
    return None

def _unit_to_mm_factor(unit_recorded: str) -> float:
    u = (unit_recorded or "").lower()
    if u == "mm":
        return 1.0
    if u == "cm":
        return 10.0
    if u in ("m", "metre", "meter"):
        return 1000.0
    if u in ("inch", "in"):
        return 25.4
    if u in ("um", "micron", "micro"):
        return 0.001
    return 1.0  # 保守

def _domain_normalize(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    return re.sub(r"[^a-z0-9\-]+", "-", x.strip().lower())[:64] or None

def triangulate(shape: TopoDS_Shape, lin_defl: float = 0.5) -> None:
    with contextlib.suppress(TypeError):
        BRepMesh_IncrementalMesh(shape, lin_defl, True, 0.5, True)
        return
    BRepMesh_IncrementalMesh(shape, deflection=lin_defl, isRelative=True, angle=0.5, parallel=True)


def _brepbndlib_add(shape: TopoDS_Shape, bbox: Bnd_Box, use_triangulation: bool = False) -> None:
    if BRepBndLib is not None and hasattr(BRepBndLib, "brepbndlib"):
        try:
            BRepBndLib.brepbndlib.Add(shape, bbox, use_triangulation)  # type: ignore[attr-defined]
            return
        except (AttributeError, TypeError):  # pragma: no cover - fallback
            pass
    brepbndlib_Add(shape, bbox, use_triangulation)


def _brepgprop_volume(shape: TopoDS_Shape, props, skip_closed: bool = True, use_tri: bool = True) -> None:
    if BRepGProp is not None and hasattr(BRepGProp, "brepgprop"):
        try:
            BRepGProp.brepgprop.VolumeProperties(shape, props, skip_closed, use_tri)  # type: ignore[attr-defined]
            return
        except (AttributeError, TypeError):  # pragma: no cover
            pass
    brepgprop_VolumeProperties(shape, props, skip_closed, use_tri)


def _brepgprop_surface(shape: TopoDS_Shape, props) -> None:
    if BRepGProp is not None and hasattr(BRepGProp, "brepgprop"):
        try:
            BRepGProp.brepgprop.SurfaceProperties(shape, props)  # type: ignore[attr-defined]
            return
        except (AttributeError, TypeError):  # pragma: no cover
            pass
    brepgprop_SurfaceProperties(shape, props)

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
        _brepbndlib_add(shape, bbox, False)
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

def d2_descriptor(
    points: np.ndarray,
    num_pairs: int = 20000,
    num_bins: int = 64,
    seed: int = 42,
    device: str = "auto",
) -> np.ndarray:
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

def occ_surface_type_counts_and_hist(shape: TopoDS_Shape) -> Tuple[Tuple[int,int,int,int,int,int], np.ndarray]:
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
    order = ["plane", "cylinder", "cone", "sphere", "torus", "free"]
    counts = tuple(int(counters[k]) for k in order)
    hist = np.array([counters[key] / total for key in order], dtype=np.float64)
    return counts, hist

def occ_dihedral_hist(shape: TopoDS_Shape, bins: int = 36) -> Tuple[np.ndarray, int]:
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

    undefined_normals = 0
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
            undefined_normals += 1
            continue
        cosine = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
        angles.append(math.acos(cosine))

    if not angles:
        return np.zeros(bins, dtype=np.float64), undefined_normals
    hist, _ = np.histogram(angles, bins=bins, range=(0, math.pi), density=False)
    hist = hist.astype(np.float64)
    hist /= hist.sum() + 1e-12
    return hist, undefined_normals

def _shape_bbox_lengths(shape: TopoDS_Shape) -> Tuple[float,float,float,float]:
    bbox = Bnd_Box()
    _brepbndlib_add(shape, bbox, False)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    lengths = np.sort(np.array([xmax-xmin, ymax-ymin, zmax-zmin], dtype=np.float64))
    l1, l2, l3 = lengths  # short, mid, long
    diag = float(np.linalg.norm([l3-l1, l3-l2, l2-l1]))
    return float(l3), float(l2), float(l1), diag

def _topo_counts(shape: TopoDS_Shape) -> Tuple[int,int,int]:
    f = e = v = 0
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        f += 1
        exp.Next()
    exp = TopExp_Explorer(shape, TopAbs_EDGE)
    while exp.More():
        e += 1
        exp.Next()
    exp = TopExp_Explorer(shape, TopAbs_VERTEX)
    while exp.More():
        v += 1
        exp.Next()
    return f, e, v

def _face_degrees(shape: TopoDS_Shape) -> Tuple[float,float,float]:
    degs = []
    fexp = TopExp_Explorer(shape, TopAbs_FACE)
    while fexp.More():
        face = topods.Face(fexp.Current())
        edges = set()
        eexp = TopExp_Explorer(face, TopAbs_EDGE)
        while eexp.More():
            edges.add(eexp.Current().__hash__())  # 使用句柄哈希近似去重
            eexp.Next()
        degs.append(len(edges))
        fexp.Next()
    if not degs:
        return 0.0, 0.0, 0.0
    arr = np.asarray(degs, dtype=np.float64)
    return float(arr.mean()), float(arr.mean()), float(arr.std(ddof=0))

def compute_signature_for_file(
    raw_root: Path,
    fpath: Path,
    sample_points: int,
    logger: StructuredLogger,
    *,
    device: str = "auto",
) -> Tuple[Optional[PartSig], Optional[str], Dict[str, Optional[str]]]:
    if not fpath.is_absolute():
        fpath = (raw_root / fpath).resolve()
    meta = get_file_meta_all(raw_root, fpath)
    meta["abs_path"] = str(fpath)
    rel = meta.get("rel") or fpath.name
    src = meta.get("src") or "unknown"
    content_hash = meta.get("ch") or ""
    shape = occ_load_shape(str(fpath), logger)
    if shape is None:
        sig = PartSig(rel, rel, src, content_hash, False, None, None, None, None, None)
        return sig, f"load_shape_failed:{rel}", meta
    try:
        # 基础点云与归一化
        points = occ_shape_points(shape, n=sample_points)
        if points.shape[0] < 4:
            sig = PartSig(rel, rel, src, content_hash, False, None, None, None, None, None)
            return sig, None, meta

        normed, diag_model = normalize_points(points)
        d2 = d2_descriptor(normed, device=device)
        br = bbox_ratios(normed)
        surf_counts, surf = occ_surface_type_counts_and_hist(shape)
        dih, undef_normals = occ_dihedral_hist(shape, bins=36)

        # bbox（模型原单位）
        L, M, S, _ = _shape_bbox_lengths(shape)  # long, mid, short
        bbox_anisotropy_log = float(np.log((L + 1e-12) / (S + 1e-12)))

        # PCA 比例（在归一化坐标系）
        cov = np.cov(normed.T)
        w, _ = np.linalg.eig(cov)
        w = np.sort(np.real(w))[::-1]  # λ1 >= λ2 >= λ3
        if w[0] <= 0:
            pca_r1 = 0.0
            pca_r2 = 0.0
        else:
            pca_r1 = float(w[1] / (w[0] + 1e-12))
            pca_r2 = float(w[2] / (w[0] + 1e-12))

        # 体积与面积（模型原单位）
        props = GProp_GProps()
        with contextlib.suppress(Exception):
            _brepgprop_volume(shape, props, True, True)
        vol_model = float(props.Mass()) if props.Mass() is not None else 0.0
        props2 = GProp_GProps()
        with contextlib.suppress(Exception):
            _brepgprop_surface(shape, props2)
        area_model = float(props2.Mass()) if props2.Mass() is not None else 0.0

        # 拓扑规模
        n_faces, n_edges, n_verts = _topo_counts(shape)
        euler_char = int(n_verts - n_edges + n_faces)

        # 面度统计
        avg_deg, deg_mean, deg_std = _face_degrees(shape)

        # 单位推断
        unit_hdr = meta.get("unit")
        unit_src = meta.get("unit_source")
        if not unit_hdr:
            unit_fn = _infer_unit_from_filename(str(fpath))
            if unit_fn:
                unit_hdr = unit_fn
                unit_src = "file_name"
        if not unit_hdr:
            unit_hdr = "mm"
            unit_src = "heuristic"
        factor = _unit_to_mm_factor(unit_hdr)

        # 归一/尺度相关几何哈希
        geom_hash_scale_inv = hashlib.sha256(
            np.round(np.concatenate([d2, br, surf, dih]), 6).tobytes()
        ).hexdigest()
        geom_hash_scale_aware = hashlib.sha256(
            np.round(np.concatenate([d2, br, surf, dih, np.array([L*factor, M*factor, S*factor], dtype=np.float64)]), 6).tobytes()
        ).hexdigest()

        sig = PartSig(
            rel, rel, src, content_hash, True,
            d2, br, surf, surf_counts, dih,
            geom_hash_scale_inv=geom_hash_scale_inv,
            bbox_lengths_model=(L, M, S),
            diag_model=diag_model,
            pca_ratio_1=pca_r1, pca_ratio_2=pca_r2,
            bbox_anisotropy_log=bbox_anisotropy_log,
            vol_model=vol_model, area_model=area_model,
            n_faces=n_faces, n_edges=n_edges, n_verts=n_verts,
            euler_char=euler_char,
            avg_face_degree=avg_deg, face_degree_mean=deg_mean, face_degree_std=deg_std,
            geom_hash_scale_aware=geom_hash_scale_aware
        )

        # 附加质量标记
        qflags = []
        if unit_src != "header":
            qflags.append("unit_heuristic")
        if bbox_anisotropy_log > math.log(50.0):
            qflags.append("high_anisotropy")
        if undef_normals > 0:
            qflags.append("normal_undefined")
        meta["quality_flags_worker"] = "|".join(qflags) if qflags else ""

        # 将单位推断结果塞入 meta，供 build_row 使用
        meta["unit_recorded"] = unit_hdr
        meta["unit_source"] = unit_src

        return sig, None, meta
    except Exception as exc:
        logger.debug(
            "s2.1_signature.compute_failed",
            path=str(fpath),
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        return None, f"signature_failed:{rel}:{type(exc).__name__}", meta

# ---------------------------
# 执行器兼容：首选 fork
# ---------------------------

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

# ---------------------------
# 任务处理函数
# ---------------------------

def _expand_hist(prefix: str, arr: Optional[np.ndarray], expected_len: int) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if arr is None:
        for i in range(expected_len):
            out[f"{prefix}_{i:03d}"] = float("nan")
        return out
    if arr.shape[0] != expected_len:
        # 尺寸不符也填充
        for i in range(expected_len):
            out[f"{prefix}_{i:03d}"] = float(arr[i]) if i < arr.shape[0] else 0.0
        return out
    for i, v in enumerate(arr):
        out[f"{prefix}_{i:03d}"] = float(v)
    return out

def _code_git_sha() -> str:
    # 优先环境变量，其次脚本文件哈希
    for k in ("CODE_GIT_SHA", "GIT_COMMIT", "GIT_SHA"):
        v = os.getenv(k)
        if v:
            return v.strip()
    try:
        here = Path(__file__).resolve()
        return sha256_file(str(here))[:16]
    except Exception:
        return "unknown"

def _sig_sha256_from_row(row: Dict[str, Any]) -> str:
    # 固定顺序：D2(64)+DIH(36)+SURF(6)+bbox_ratio(2)+bbox_mm(3)+PCA(2)+vol+area+n_faces+n_edges+n_verts+euler
    keys = (
        [f"d2_{i:03d}" for i in range(64)] +
        [f"dih_{i:03d}" for i in range(36)] +
        [f"surf_{i:03d}" for i in range(6)] +
        ["bbox_ratio_0", "bbox_ratio_1",
         "bbox_long_mm", "bbox_mid_mm", "bbox_short_mm",
         "pca_ratio_1", "pca_ratio_2",
         "vol_mm3", "area_mm2",
         "n_faces", "n_edges", "n_verts", "euler_char"]
    )
    buf = bytearray()
    for k in keys:
        v = row.get(k)
        try:
            x = float(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else 0.0
        except Exception:
            x = 0.0
        buf += np.float64(x).tobytes()
    return hashlib.sha256(buf).hexdigest()

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
        sig, err, meta = compute_signature_for_file(raw_root, Path(abs_path), sample_points, logger, device=device)  # noqa: F821
        if meta_hint:
            meta = {**meta_hint, **meta}

        # 基础字段
        rel_path = meta.get("rel") or (sig.rel_path if sig else None) or "unknown"
        source = meta.get("src") or (sig.source_dataset if sig else "unknown")
        content_hash = meta.get("ch") or (sig.content_hash if sig else "")
        part_id = sig.part_id if sig else rel_path

        # 时间戳
        timestamp_raw = meta.get("created_at_header") or meta.get("file_mtime_utc")
        timestamp_source = meta.get("timestamp_source") or ("header" if meta.get("created_at_header") else ("file_mtime" if meta.get("file_mtime_utc") else "unknown"))

        # 单位
        unit_recorded = meta.get("unit_recorded") or meta.get("unit") or "mm"
        unit_source = meta.get("unit_source") or "heuristic"
        factor_mm = _unit_to_mm_factor(unit_recorded)

        # 领域与分层
        domain = _domain_normalize(meta.get("domain_hint") or source)
        dataset_tier = _infer_dataset_tier(source)

        # 直方图展开
        d2_cols = _expand_hist("d2", sig.d2_hist if sig and sig.has_points else None, 64)
        dih_cols = _expand_hist("dih", sig.dih_hist if sig and sig.has_points else None, 36)
        surf_cols = _expand_hist("surf", sig.surf_hist if sig and sig.has_points else None, 6)

        # bbox 转 mm
        if sig and sig.bbox_lengths_model:
            L_mm = float(sig.bbox_lengths_model[0]) * factor_mm
            M_mm = float(sig.bbox_lengths_model[1]) * factor_mm
            S_mm = float(sig.bbox_lengths_model[2]) * factor_mm
        else:
            L_mm = M_mm = S_mm = float("nan")

        # 拓扑计数与面型计数
        n_plane, n_cyl, n_cone, n_sphere, n_torus, n_other = sig.surf_counts if sig and sig.surf_counts else (0,0,0,0,0,0)

        # 汇总行
        row: Dict[str, Any] = {
            "part_id": part_id,
            "rel_path": rel_path,
            "source_dataset": source,
            "content_hash": content_hash,
            "dataset_tier": dataset_tier,
            "unit_recorded": unit_recorded,
            "unit_source": unit_source,
            "timestamp_raw": timestamp_raw,
            "timestamp_source": timestamp_source,
            "domain": domain,
            "d2_bins_id": "d2_64_v1",
            "dih_bins_id": "dih_36_v1",
            "surf_order_id": "surf_v1_plane_cyl_cone_sphere_torus_free",
            "sig_version": "v2.1-d2-64-dih-36",
            "code_git_sha": _code_git_sha(),
            "geom_hash_scale_inv": sig.geom_hash_scale_inv if sig else None,
            "geom_hash_scale_aware": sig.geom_hash_scale_aware if sig else None,
            "bbox_long_mm": L_mm,
            "bbox_mid_mm": M_mm,
            "bbox_short_mm": S_mm,
            "pca_ratio_1": sig.pca_ratio_1 if sig else None,
            "pca_ratio_2": sig.pca_ratio_2 if sig else None,
            "bbox_anisotropy_log": sig.bbox_anisotropy_log if sig else None,
            "vol_mm3": (sig.vol_model * (factor_mm ** 3)) if sig and sig.vol_model is not None else None,
            "area_mm2": (sig.area_model * (factor_mm ** 2)) if sig and sig.area_model is not None else None,
            "n_faces": sig.n_faces if sig else None,
            "n_edges": sig.n_edges if sig else None,
            "n_verts": sig.n_verts if sig else None,
            "euler_char": sig.euler_char if sig else None,
            "n_plane": n_plane, "n_cyl": n_cyl, "n_cone": n_cone, "n_sphere": n_sphere, "n_torus": n_torus, "n_other": n_other,
            "avg_face_degree": sig.avg_face_degree if sig else None,
            "face_degree_mean": sig.face_degree_mean if sig else None,
            "face_degree_std": sig.face_degree_std if sig else None,
            "hist_norm_l1": True,
            "run_datetime_utc": datetime.now(timezone.utc).isoformat(),
            "has_points": int(sig.has_points) if sig else 0,
            "bbox_ratio_0": float(sig.bbox_ratio[0]) if sig and sig.bbox_ratio is not None else None,
            "bbox_ratio_1": float(sig.bbox_ratio[1]) if sig and sig.bbox_ratio is not None else None,
            "meta_json": json.dumps(meta, ensure_ascii=False),
            "error": err or "",
            "quality_flags": meta.get("quality_flags_worker", ""),
        }
        # 合并直方图列
        row.update(d2_cols)
        row.update(dih_cols)
        row.update(surf_cols)

        # 最终签名整体哈希
        row["sig_sha256"] = _sig_sha256_from_row(row)

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

# ---------------------------
# 聚合
# ---------------------------

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
    if not frames:
        out = {"rows": 0, "fingerprint": None, "output": None}
        logger.info("s2.1_signature.aggregate.empty")
        return out
    full = pd.concat(frames, ignore_index=True)
    out_dir = output_root
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"s2.1_signature_{ts}.csv"
    full.to_csv(out_path, index=False)
    fp = hashlib.sha256(";".join(sorted(full["part_id"].astype(str))).encode("utf-8")).hexdigest()
    logger.info("s2.1_signature.aggregate.done", rows=int(full.shape[0]), output=str(out_path), fingerprint=fp)
    return {"rows": int(full.shape[0]), "fingerprint": fp, "output": str(out_path)}

# ---------------------------
# 基于 TaskFramework 的作业
# ---------------------------

class S21SignatureJob(TaskFramework):
    def __init__(self) -> None:
        super().__init__()
        self._logger = StructuredLogger.get_logger("app.s2.1_signature")
        self._sig_cfg: Optional[SignatureConfig] = None
        self._raw_root: Optional[Path] = None
        self._items_map: Dict[str, Dict[str, Any]] = {}
        self._item_iter = None
        self._progress: Optional[ProgressController] = None
        self._collected: List[Dict[str, Any]] = []
        self.set_job_id("s2.1-signature")

    # 1) 构建 handler 上下文（在启动执行器之前调用）
    def build_handler_context(self, cfg: Config) -> Dict[str, Any]:
        prefer_fork_context()  # 兼容容器/WSL
        s2_block = cfg.get("s2", dict, default={})
        section = s2_block.get("s2.1_signature")
        if section is None:
            raise RuntimeError("缺少 s2.1_signature 配置段")
        self._sig_cfg = SignatureConfig.from_config(section)
        self._sig_cfg.out_root.mkdir(parents=True, exist_ok=True)
        raw_root = self._sig_cfg.raw_root

        # demo OCC
        if self._sig_cfg.demo_occ:
            from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
            demo_root = self._sig_cfg.out_root / "demo_occ"
            demo_root.mkdir(parents=True, exist_ok=True)
            def _write_step(shape, path: Path) -> bool:
                writer = STEPControl_Writer()
                writer.Transfer(shape, STEPControl_AsIs)
                return writer.Write(str(path)) == IFSelect_RetDone
            _ = [
                _write_step(BRepPrimAPI_MakeBox(10, 10, 10).Solid(), demo_root / "DATASETA_box.step"),
                _write_step(BRepPrimAPI_MakeBox(10, 10, 10).Solid(), demo_root / "DATASETA_box_dup.step"),
                _write_step(BRepPrimAPI_MakeBox(13, 13, 13).Solid(), demo_root / "DATASETB_box_scaled.step"),
                _write_step(BRepPrimAPI_MakeCylinder(4.0, 12.0).Solid(), demo_root / "DATASETB_cylinder.step"),
            ]
            raw_root = demo_root

        # 预扫描，供 handler 与分片共用
        limit, path_iter = discover_parts(raw_root, self._sig_cfg.subset_n, self._sig_cfg.subset_frac, self._logger)
        if limit == 0:
            self._logger.warning("s2.1_signature.no_files", raw_root=str(raw_root))
        self._items_map = {}
        self._item_iter = iter_items(path_iter, raw_root, self._items_map, self._logger)
        self._raw_root = raw_root

        # 进度（流式）
        self._progress = ProgressController(total_units=None, description="S2.1 签名提取", unit_name="file")
        self._progress.start()

        # handler 上下文
        return {
            "raw_root": raw_root,
            "sample_points": self._sig_cfg.sample_points,
            "device": self._sig_cfg.device,
            "items_map": self._items_map,
            "output_root": str(self._sig_cfg.out_root),
            "progress": self._progress.make_proxy(),
        }

    # 2) 产出任务
    def produce_tasks(self) -> Iterable[TaskRecord]:
        assert self._sig_cfg is not None and self._item_iter is not None
        constraints = PartitionConstraints(
            max_items_per_task=self._sig_cfg.partition.max_items_per_task,
            max_weight_per_task=self._sig_cfg.partition.max_weight_per_task,
            shuffle_seed=self._sig_cfg.partition.shuffle_seed,
        )
        count = 0
        total_w = 0.0
        for task in TaskPartitioner.iter_tasks(
            {"job_id": self.job_id},
            self._item_iter,
            self._sig_cfg.partition.strategy,
            constraints=constraints,
        ):
            count += 1
            total_w += task.weight
            if self._progress is not None:
                self._progress.discovered(len(task.payload_ref))
            yield task
        self._logger.debug("s2.1_signature.plan_ready", task_count=count, total_weight=total_w)

    # 3) 处理单个任务
    def handle(self, leased: LeasedTask, context: Dict[str, Any]) -> TaskResult:
        return signature_task_handler(leased, context)

    # 4) 处理结果回调（主进程）
    def result_handler(self, _leased: LeasedTask, result: TaskResult) -> None:
        self._collected.append(result.metadata)
        self._logger.debug(
            "s2.1_signature.task_complete",
            task_id=result.metadata.get("task_id"),
            processed=result.metadata.get("processed"),
            failures=len(result.metadata.get("failures") or []),
        )

    # 5) 收尾
    def after_run(self, cfg: Config) -> None:
        if self._progress is not None:
            with contextlib.suppress(Exception):
                self._progress.close()
        if self._sig_cfg is not None:
            aggregate_outputs(self._sig_cfg.out_root, self._logger)

# ---------------------------
# CLI
# ---------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="main.yaml 路径")
    args = parser.parse_args()
    if args.config:
        os.environ["CAD_TASK_CONFIG"] = str(Path(args.config).expanduser().resolve())
    job = S21SignatureJob()
    return job.run()

if __name__ == "__main__":
    raise SystemExit(main())

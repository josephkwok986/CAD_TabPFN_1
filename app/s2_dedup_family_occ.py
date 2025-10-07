#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S2 (OCC + GPU): 零件级去重与“家族”聚类（STEP/B-Rep）
方案B：父进程看门狗 + 每worker单任务 + GPU优先，稳定可靠。

新增：
- family_min_samples（默认5） → DBSCAN.min_samples，抑制“链式连通”。
- family_auto_eps（bool或dict） → 自动确定 DBSCAN.eps：
  基于 L2 归一化特征的 kNN 第 k 邻居（k=knn_k 或 min_samples）欧氏距离分位数（quantile），
  再乘以 scale，得到 eps。支持抽样（sample_n）与 FAISS 加速。
"""

import os, json, argparse, hashlib, math, time, logging, traceback, re, sys
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# 并行
import multiprocessing as mp
from queue import Empty

# --------- OCC (pythonocc-core) ----------
from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.BRepTools import breptools_Read
from OCC.Core.BRep import BRep_Tool, BRep_Builder
from OCC.Core.TopoDS import TopoDS_Shape, topods, TopoDS_Face, TopoDS_Edge
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_REVERSED
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface

# --------- GPU 依赖（延迟导入） ----------
cp = None
faiss = None
cuDBSCAN = None
_GPU_CUPY = False
_GPU_FAISS = False
_GPU_CUML = False

def _ensure_cupy():
    global cp, _GPU_CUPY
    if cp is not None or _GPU_CUPY: return
    try:
        import cupy as _cp
        cp = _cp; _GPU_CUPY = True
    except Exception:
        cp = None; _GPU_CUPY = False

def _ensure_faiss():
    global faiss, _GPU_FAISS
    if faiss is not None or _GPU_FAISS: return
    try:
        import faiss as _faiss
        faiss = _faiss
        try:
            _ = _faiss.StandardGpuResources()
            _GPU_FAISS = True
        except Exception:
            _GPU_FAISS = False
    except Exception:
        faiss = None; _GPU_FAISS = False

def _ensure_cuml():
    global cuDBSCAN, _GPU_CUML
    if cuDBSCAN is not None or _GPU_CUML: return
    try:
        from cuml.cluster import DBSCAN as _cuDBSCAN
        cuDBSCAN = _cuDBSCAN; _GPU_CUML = True
    except Exception:
        cuDBSCAN = None; _GPU_CUML = False

# CPU 回退
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN as skDBSCAN

SUPPORTED_STEP = {".step", ".stp", ".stpz"}
SUPPORTED_BREP = {".brep", ".brp"}

# ---------------- Logging ----------------
import logging
def _to_level(level: str) -> int:
    return {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING}.get(
        (level or "info").lower(), logging.INFO
    )

def setup_logger(level: str = "info", log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger("S2_OCC_GPU")
    logger.setLevel(_to_level(level))
    logger.propagate = False

    def _has_stream_handler():
        return any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers)
    def _has_file_handler(path: str):
        for h in logger.handlers:
            if isinstance(h, logging.FileHandler):
                try:
                    if os.path.abspath(getattr(h, 'baseFilename', '')) == os.path.abspath(path):
                        return True
                except Exception:
                    pass
        return False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    if not _has_stream_handler():
        sh = logging.StreamHandler(); sh.setFormatter(fmt); logger.addHandler(sh)
    if log_file:
        try: os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        except Exception: pass
        if not _has_file_handler(log_file):
            fh = logging.FileHandler(log_file, encoding="utf-8"); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger

# ---------------- 1% 进度 ----------------
class ProgressPrinter:
    def __init__(self, total: int, prefix: str = "", stream = sys.stdout):
        self.total = max(1, int(total)); self.prefix = prefix; self.stream = stream
        self._last_percent = -1; self._finished = False
    def _emit(self, percent: int):
        print(f"\r{self.prefix} {percent:3d}%", end="", file=self.stream, flush=True)
    def update(self, current: int):
        if self._finished: return
        current = min(max(0, int(current)), self.total)
        percent = int(current * 100 / self.total)
        if percent >= self._last_percent + 1:
            self._emit(percent); self._last_percent = percent
        if current >= self.total: self.finish()
    def print_start(self):
        self._emit(0); self._last_percent = 0
    def finish(self):
        if not self._finished:
            self._emit(100); print("", file=self.stream); self._finished = True

# ---------------- Utils ----------------
def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def safe_rel(root: str, path: str) -> str:
    try: return os.path.relpath(path, root)
    except Exception: return path

def normalize_points(points: np.ndarray) -> Tuple[np.ndarray, float]:
    pts = points.astype(np.float64)
    c = pts.mean(axis=0, keepdims=True); pts = pts - c
    mins = pts.min(axis=0); maxs = pts.max(axis=0)
    diag = float(np.linalg.norm(maxs - mins) + 1e-9)
    pts = pts / diag
    return pts, diag

def d2_descriptor(points: np.ndarray, num_pairs: int = 20000, num_bins: int = 64, seed: int = 42, device: str = "auto") -> np.ndarray:
    n = points.shape[0]
    if n < 4: raise ValueError("Not enough points for D2")
    use_gpu = (device == "gpu"); 
    if use_gpu: _ensure_cupy()
    if use_gpu and _GPU_CUPY:
        rng = cp.random.default_rng(seed)
        i = rng.integers(0, n, size=num_pairs)
        j = rng.integers(0, n, size=num_pairs)
        p = cp.asarray(points)
        d = cp.linalg.norm(p[i] - p[j], axis=1)
        d = cp.clip(d / (math.sqrt(3) + 1e-9), 0.0, 1.0)
        hist, _ = cp.histogram(d, bins=num_bins, range=(0,1))
        hist = hist.astype(cp.float64); hist /= (hist.sum() + 1e-12)
        return cp.asnumpy(hist)
    else:
        rng = np.random.default_rng(seed)
        i = rng.integers(0, n, size=num_pairs)
        j = rng.integers(0, n, size=num_pairs)
        d = np.linalg.norm(points[i] - points[j], axis=1)
        d = np.clip(d / (math.sqrt(3) + 1e-9), 0.0, 1.0)
        hist, _ = np.histogram(d, bins=num_bins, range=(0,1))
        hist = hist.astype(np.float64); hist /= (hist.sum() + 1e-12)
        return hist

def bbox_ratios(points: np.ndarray) -> np.ndarray:
    mins = points.min(axis=0); maxs = points.max(axis=0)
    lengths = np.sort((maxs - mins).astype(np.float64))
    l1,l2,l3 = lengths
    return np.array([l1/(l3+1e-12), l2/(l3+1e-12)], np.float64)

_STEP_FILE_NAME_RE = re.compile(
    r"FILE_NAME\s*\(\s*'([^']*)'\s*,\s*'([^']*)'\s*,\s*\((.*?)\)\s*,\s*\((.*?)\)\s*,\s*'([^']*)'\s*,\s*'([^']*)'\s*,\s*'([^']*)'\s*\)",
    re.IGNORECASE | re.DOTALL,
)
_STEP_FILE_SCHEMA_RE = re.compile(r"FILE_SCHEMA\s*\(\s*\(\s*'([^']+)'\s*\)\s*\)", re.IGNORECASE)

def read_text_prefix(path: str, max_bytes: int = 200000) -> Optional[str]:
    try:
        with open(path, "rb") as f: chunk = f.read(max_bytes)
        try: return chunk.decode("utf-8", errors="ignore")
        except Exception: return chunk.decode("latin1", errors="ignore")
    except Exception:
        return None

def _normalize_datetime(s: str) -> Optional[str]:
    s = s.strip()
    for cand in [s, s.replace(" ", "T")]:
        try:
            dt = datetime.fromisoformat(cand)
            if dt.tzinfo is None: return dt.isoformat()
            return dt.astimezone(timezone.utc).isoformat()
        except Exception: pass
    for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y%m%dT%H%M%S", "%Y-%m-%d %H:%M:%S"]:
        try: return datetime.strptime(s, fmt).isoformat()
        except Exception: pass
    return None

def parse_step_header_text(text: str) -> Dict[str, Optional[str]]:
    meta = {"step_schema": None, "originating_system": None, "author": None, "created_at_header": None, "unit": None, "unit_source": None}
    try:
        m_schema = _STEP_FILE_SCHEMA_RE.search(text or "")
        if m_schema: meta["step_schema"] = m_schema.group(1).strip()
        m_name = _STEP_FILE_NAME_RE.search(text or "")
        if m_name:
            timestamp = m_name.group(2).strip()
            meta["created_at_header"] = _normalize_datetime(timestamp)
            authors_raw = m_name.group(3).strip()
            authors = [a.strip().strip("'") for a in authors_raw.split(",") if a.strip()]
            meta["author"] = "|".join(authors) if authors else None
            meta["originating_system"] = m_name.group(6).strip().strip("'")
        up = (text or "").upper()
        if "INCH" in up:
            meta["unit"], meta["unit_source"] = "inch", "inferred_step_data"
        elif "MILLI" in up and "METRE" in up:
            meta["unit"], meta["unit_source"] = "mm", "inferred_step_data"
        elif "METRE" in up:
            meta["unit"], meta["unit_source"] = "m", "inferred_step_data"
    except Exception:
        pass
    return meta

def get_file_meta(fpath: str) -> Dict[str, Optional[str]]:
    try: size = os.path.getsize(fpath)
    except Exception: size = None
    try:
        mtime = os.path.getmtime(fpath)
        mtime_iso = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
    except Exception:
        mtime_iso = None
    ext = os.path.splitext(fpath)[1].lower()
    fmt = "STEP" if ext in SUPPORTED_STEP else ("BREP" if ext in SUPPORTED_BREP else "UNKNOWN")
    return {"file_ext": ext, "format": fmt, "file_size_bytes": int(size) if size is not None else None, "file_mtime_utc": mtime_iso}

def guess_kernel(format_str: str, originating_system: Optional[str]) -> Optional[str]:
    if format_str == "BREP": return "OpenCascade"
    if format_str == "STEP":
        s = (originating_system or "").lower()
        if "parasolid" in s or "nx" in s or "solidworks" in s: return "Parasolid"
        if "acis" in s or "autocad" in s or "inventor" in s:   return "ACIS"
        if "catia" in s or "cgm" in s or "3dexperience" in s or "solidedge" in s: return "CGM"
        return "STEP"
    return None

# ------------- OCC 加载与特征提取 -------------
def occ_load_shape(path: str, logger: logging.Logger) -> Optional["TopoDS_Shape"]:
    ext = os.path.splitext(path)[1].lower()
    if ext in SUPPORTED_STEP:
        reader = STEPControl_Reader()
        status = reader.ReadFile(path)
        if status != IFSelect_RetDone:
            logger.debug(f"[OCC] STEP read failed: {path}")
            return None
        reader.TransferRoots()
        return reader.OneShape()
    elif ext in SUPPORTED_BREP:
        shape = TopoDS_Shape()
        try:
            ok = breptools_Read(shape, path, BRep_Builder())
        except TypeError:
            ok = breptools_Read(shape, path)
        if not ok:
            logger.debug(f"[OCC] BREP read failed: {path}")
            return None
        return shape
    else:
        return None

def triangulate(shape, lin_defl=0.5):
    try:
        BRepMesh_IncrementalMesh(shape, lin_defl, True, 0.5, True)
    except TypeError:
        BRepMesh_IncrementalMesh(shape, deflection=lin_defl, isRelative=True, angle=0.5, parallel=True)

def _append_nodes_from_triangulation(face: TopoDS_Face, loc: TopLoc_Location, pts_out: list):
    tri = BRep_Tool.Triangulation(face, loc)
    if not tri: return 0
    tri_obj = tri.GetObject() if hasattr(tri, "GetObject") else tri
    n_added = 0
    try:
        if hasattr(tri_obj, "Nodes") and hasattr(tri_obj, "NbNodes"):
            nodes = tri_obj.Nodes(); n = tri_obj.NbNodes()
            for i in range(1, n+1):
                p = nodes.Value(i).Transformed(loc.Transformation())
                pts_out.append([p.X(), p.Y(), p.Z()]); n_added += 1
            return n_added
    except Exception: pass
    try:
        if hasattr(tri_obj, "NbNodes") and hasattr(tri_obj, "Node"):
            n = tri_obj.NbNodes()
            for i in range(1, n+1):
                p = tri_obj.Node(i).Transformed(loc.Transformation())
                pts_out.append([p.X(), p.Y(), p.Z()]); n_added += 1
            return n_added
    except Exception: pass
    return n_added

def occ_shape_points(shape: "TopoDS_Shape", n: int = 4096, lin_defl: float = 0.5) -> np.ndarray:
    triangulate(shape, lin_defl=lin_defl)
    pts = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = topods.Face(exp.Current())
        loc = TopLoc_Location()
        _append_nodes_from_triangulation(face, loc, pts)
        exp.Next()
    if len(pts) == 0:
        bbox = Bnd_Box(); brepbndlib_Add(shape, bbox, False)
        xmin,ymin,zmin,xmax,ymax,zmax = bbox.Get()
        pts = [
            [xmin,ymin,zmin],[xmax,ymin,zmin],[xmin,ymax,zmin],[xmax,ymax,zmin],
            [xmin,ymin,zmax],[xmax,ymin,zmax],[xmin,ymax,zmax],[xmax,ymax,zmax],
        ]
    pts = np.array(pts, dtype=np.float64)
    if pts.shape[0] > n:
        idx = np.random.default_rng(0).choice(pts.shape[0], size=n, replace=False)
        pts = pts[idx]
    return pts

def occ_surface_type_hist(shape: "TopoDS_Shape") -> np.ndarray:
    counters = dict(plane=0,cylinder=0,cone=0,sphere=0,torus=0,free=0); total=0
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = topods.Face(exp.Current())
        ad = BRepAdaptor_Surface(face, True)
        t = ad.GetType()
        if   t == GeomAbs_Plane: counters["plane"] += 1
        elif t == GeomAbs_Cylinder: counters["cylinder"] += 1
        elif t == GeomAbs_Cone: counters["cone"] += 1
        elif t == GeomAbs_Sphere: counters["sphere"] += 1
        elif t == GeomAbs_Torus: counters["torus"] += 1
        else: counters["free"] += 1
        total += 1; exp.Next()
    total = max(total, 1)
    return np.array([counters[k]/total for k in ["plane","cylinder","cone","sphere","torus","free"]], np.float64)

def occ_dihedral_hist(shape: "TopoDS_Shape", bins: int = 36) -> np.ndarray:
    def edge_key(edge: TopoDS_Edge):
        c = BRepAdaptor_Curve(edge)
        u0, u1 = c.FirstParameter(), c.LastParameter()
        p0, p1 = c.Value(u0), c.Value(u1)
        a = np.round([p0.X(), p0.Y(), p0.Z()], 6)
        b = np.round([p1.X(), p1.Y(), p1.Z()], 6)
        t0, t1 = tuple(a), tuple(b)
        return (t0, t1) if t0 <= t1 else (t1, t0)

    edge_map: Dict[Tuple[Tuple[float,float,float],Tuple[float,float,float]], Dict[str, list]] = {}
    fexp = TopExp_Explorer(shape, TopAbs_FACE)
    while fexp.More():
        face = topods.Face(fexp.Current())
        eexp = TopExp_Explorer(face, TopAbs_EDGE)
        while eexp.More():
            edge = topods.Edge(eexp.Current())
            try: k = edge_key(edge)
            except Exception: eexp.Next(); continue
            rec = edge_map.get(k)
            if rec is None: rec = {"edge": edge, "faces": []}; edge_map[k] = rec
            rec["faces"].append(face)
            eexp.Next()
        fexp.Next()

    angles = []
    for rec in edge_map.values():
        faces = rec["faces"]
        if len(faces) < 2: continue
        edge = rec["edge"]
        c = BRepAdaptor_Curve(edge)
        tmid = 0.5 * (c.FirstParameter() + c.LastParameter())
        P = c.Value(tmid)

        def face_normal_at(face, P):
            srf = BRep_Tool.Surface(face)
            uv = ShapeAnalysis_Surface(srf).ValueOfUV(P, 1e-6)
            props = GeomLProp_SLProps(srf, uv.X(), uv.Y(), 1, 1e-6)
            if not props.IsNormalDefined(): return None
            n = props.Normal()
            v = np.array([n.X(), n.Y(), n.Z()], np.float64)
            if face.Orientation() == TopAbs_REVERSED: v = -v
            nv = np.linalg.norm(v) + 1e-12
            return v / nv

        n1 = face_normal_at(faces[0], P); n2 = face_normal_at(faces[1], P)
        if n1 is None or n2 is None: continue
        c12 = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
        ang = math.acos(c12); angles.append(ang)

    if not angles: return np.zeros(bins, dtype=np.float64)
    hist, _ = np.histogram(angles, bins=bins, range=(0, math.pi), density=False)
    hist = hist.astype(np.float64); hist /= (hist.sum() + 1e-12)
    return hist

# ---------------- 签名与主流程 ----------------
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
    geom_hash: Optional[np.ndarray] = None

def discover_parts(raw_root: str, subset_n: Optional[int], subset_frac: Optional[float], logger: logging.Logger) -> List[str]:
    files = []; step_cnt = 0; brep_cnt = 0
    for r,_,fns in os.walk(raw_root, followlinks=True):
        for fn in fns:
            ext = os.path.splitext(fn)[1].lower()
            if ext in SUPPORTED_STEP | SUPPORTED_BREP:
                p = os.path.join(r,fn); files.append(p)
                if ext in SUPPORTED_STEP: step_cnt += 1
                else: brep_cnt += 1
    files.sort(); total = len(files); selected = files
    if subset_frac and 0 < subset_frac < 1: selected = selected[:max(1,int(total*subset_frac))]
    if subset_n: selected = selected[:subset_n]
    logger.info(f"[Discover] STEP={step_cnt}, BREP={brep_cnt}, total={total}, selected={len(selected)}")
    return selected

def get_file_meta_all(raw_root: str, fpath: str) -> Dict[str, Optional[str]]:
    rel = safe_rel(raw_root, fpath); src = rel.split(os.sep)[0] if os.sep in rel else "unknown"
    ch = sha256_file(fpath); fm = get_file_meta(fpath)
    text = read_text_prefix(fpath, 300000) if (fm["format"]=="STEP" and fm["file_ext"]!=".stpz") else None
    step_hdr = parse_step_header_text(text or "") if text is not None else \
        {"step_schema": None, "originating_system": None, "author": None, "created_at_header": None, "unit": None, "unit_source": None}
    kernel = guess_kernel(fm["format"], step_hdr.get("originating_system"))
    parts = rel.split(os.sep); repo = parts[1] if len(parts)>=2 else None
    created_hdr = step_hdr.get("created_at_header")
    timestamp_source = "header" if created_hdr else ("mtime" if fm.get("file_mtime_utc") else "none")
    return {"rel": rel, "src": src, "ch": ch, **fm, **step_hdr, "kernel": kernel, "repo": repo,
            "domain_hint": src, "timestamp_source": timestamp_source}

def compute_signature_for_file(raw_root: str, fpath: str, sample_points: int, logger: logging.Logger, device: str="auto") -> Tuple[Optional[PartSig], Optional[str], Dict[str, Optional[str]]]:
    meta = get_file_meta_all(raw_root, fpath)
    rel, src, ch = meta["rel"], meta["src"], meta["ch"]
    shape = occ_load_shape(fpath, logger)
    if shape is None:
        sig = PartSig(rel, rel, src, ch, False, None, None, None, None, None)
        return sig, "load_shape_failed:"+rel, meta
    try:
        pts = occ_shape_points(shape, n=sample_points)
        if pts.shape[0] >= 4:
            npts,_ = normalize_points(pts)
            d2 = d2_descriptor(npts, device=device)
            br = bbox_ratios(npts)
            surf = occ_surface_type_hist(shape)
            dih = occ_dihedral_hist(shape, bins=36)
            geom_hash = hashlib.sha256(np.round(np.concatenate([d2,br,surf,dih]),3).tobytes()).hexdigest()
            sig = PartSig(rel, rel, src, ch, True, d2, br, surf, dih, geom_hash)
        else:
            sig = PartSig(rel, rel, src, ch, False, None, None, None, None, None)
        return sig, None, meta
    except Exception as e:
        logger.debug(traceback.format_exc())
        return None, f"signature_failed:{rel}:{type(e).__name__}", meta

# ============ 方案 B：自管 worker（父进程看门狗） ============
def _worker_loop(wid: int, in_q: "mp.Queue", out_q: "mp.Queue",
                 raw_root: str, sample_points: int, device: str,
                 log_level: str, log_file: Optional[str]):
    logger = setup_logger(log_level, log_file=log_file)
    logger.debug(f"[Worker-{wid}] started (pid={os.getpid()})")
    while True:
        task = in_q.get()
        if task is None:
            logger.debug(f"[Worker-{wid}] got sentinel, exiting."); break
        idx, fpath = task
        try:
            sig, err, meta = compute_signature_for_file(raw_root, fpath, sample_points, logger, device=device)
        except Exception as e:
            sig, err, meta = None, f"worker_exception:{type(e).__name__}:{e}", {}
        out_q.put((wid, idx, sig, err, meta))
    logger.debug(f"[Worker-{wid}] exit.")

def compute_signatures(raw_root: str, files: List[str], sample_points: int, logger: logging.Logger, device: str="auto",
                       workers: int = 1, mp_start: str = "spawn", log_file: Optional[str] = None,
                       timeout_sec: Optional[int] = None,
                       max_inflight: Optional[int] = None,
                       max_inflight_multiplier: Optional[int] = None) -> Tuple[List[PartSig], List[str], Dict[str, Dict[str, Optional[str]]]]:
    n_files = len(files)
    logger.info(f"[Signatures|B] start: files={n_files}, sample_points={sample_points}, device={device}, workers={workers}, mp_start={mp_start}, timeout={timeout_sec or 'none'}s")

    t0 = time.time()
    prog = ProgressPrinter(total=n_files, prefix="[Signatures]"); prog.print_start()

    # 限流窗口（方案B：同时在飞=min(workers, cap)）
    if isinstance(max_inflight, int) and max_inflight > 0:
        cap = max_inflight
    elif isinstance(max_inflight_multiplier, int) and max_inflight_multiplier > 0:
        cap = max(workers, max_inflight_multiplier * int(workers))
    else:
        cap = max(32, 4 * int(workers))
    cap = min(workers, cap)

    try: ctx = mp.get_context(mp_start)
    except Exception: ctx = mp.get_context("spawn")
    out_q: mp.Queue = ctx.Queue()

    def spawn_worker(wid: int):
        in_q = ctx.Queue()
        wlog_level = "debug" if logger.level <= logging.DEBUG else "warning"
        p = ctx.Process(target=_worker_loop, args=(wid, in_q, out_q, raw_root, sample_points, device, wlog_level, log_file), daemon=False)
        p.start()
        return {"id": wid, "proc": p, "in_q": in_q, "busy": False, "start_t": None, "task": None}

    workers_state = [spawn_worker(i) for i in range(cap)]
    next_task_idx = 1; done_cnt = 0
    results_map: Dict[int, Tuple[Optional[PartSig], Optional[str], Dict[str, Optional[str]]]] = {}
    fails: List[str] = []; metas: Dict[str, Dict[str, Optional[str]]] = {}

    while done_cnt < n_files:
        for st in workers_state:
            if next_task_idx > n_files: break
            if not st["busy"]:
                fpath = files[next_task_idx - 1]
                try:
                    st["in_q"].put((next_task_idx, fpath))
                    st["busy"] = True; st["start_t"] = time.monotonic(); st["task"] = (next_task_idx, fpath)
                except Exception as e:
                    logger.warning(f"[Signatures|B] in_q.put failed for worker-{st['id']}: {e}; respawn.")
                    try:
                        if st["proc"].is_alive(): st["proc"].terminate(); st["proc"].join(timeout=5)
                    except Exception: pass
                    new_st = spawn_worker(st["id"]); st.update(new_st)
                    st["in_q"].put((next_task_idx, fpath))
                    st["busy"] = True; st["start_t"] = time.monotonic(); st["task"] = (next_task_idx, fpath)
                next_task_idx += 1

        pulled_any = False
        while True:
            try:
                wid, idx, sig, err, meta = out_q.get_nowait(); pulled_any = True
            except Empty:
                break
            for st in workers_state:
                if st["id"] == wid:
                    st["busy"] = False; st["start_t"] = None; st["task"] = None; break
            results_map[idx] = (sig, err, meta)
            if sig is not None: metas[sig.part_id] = meta
            if err: fails.append(err)
            done_cnt += 1; prog.update(done_cnt)

        if timeout_sec and timeout_sec > 0:
            now = time.monotonic()
            for st in workers_state:
                if st["busy"] and st["start_t"] is not None and (now - st["start_t"]) > timeout_sec:
                    idx, fpath = st["task"]; base = os.path.basename(fpath)
                    try:
                        if st["proc"].is_alive(): st["proc"].terminate(); st["proc"].join(timeout=5)
                    except Exception: pass
                    results_map[idx] = (None, f"timeout:{base}", {}); fails.append(f"timeout:{base}")
                    done_cnt += 1; prog.update(done_cnt)
                    new_st = spawn_worker(st["id"]); st.update(new_st)

        if not pulled_any: time.sleep(0.05)

    prog.finish()
    for st in workers_state:
        try: st["in_q"].put(None)
        except Exception: pass
    for st in workers_state:
        try:
            st["proc"].join(timeout=3)
            if st["proc"].is_alive(): st["proc"].terminate(); st["proc"].join(timeout=2)
        except Exception: pass

    sigs: List[PartSig] = []
    for i in range(1, n_files+1):
        sig, err, meta = results_map.get(i, (None, f"missing_result_for_index:{i}", {}))
        if sig is not None:
            sigs.append(sig)
            if sig.part_id not in metas: metas[sig.part_id] = meta
        if err and err not in fails: fails.append(err)

    logger.info(f"[Signatures|B] done: ok={len(sigs)}, fail={len(fails)}, elapsed={time.time()-t0:.2f}s")
    if logger.level<=logging.DEBUG and fails:
        logger.debug("[Signatures|B] failures (first 200):\n  " + "\n  ".join(fails[:200]))
        if len(fails)>200: logger.debug(f"... and {len(fails)-200} more")
    return sigs, fails, metas

# ---------------- Dedup & Families ----------------
class DSU:
    def __init__(self, items: List[str]):
        self.parent = {x:x for x in items}; self.rank={x:0 for x in items}
    def find(self,x:str)->str:
        while self.parent[x]!=x:
            self.parent[x]=self.parent[self.parent[x]]; x=self.parent[x]
        return x
    def union(self,a:str,b:str):
        ra,rb=self.find(a),self.find(b)
        if ra==rb: return
        if self.rank[ra]<self.rank[rb]: self.parent[ra]=rb
        elif self.rank[ra]>self.rank[rb]: self.parent[rb]=ra
        else: self.parent[rb]=ra; self.rank[ra]+=1

def pack_feat(s: PartSig) -> np.ndarray:
    return np.concatenate([s.d2_hist, s.bbox_ratio, s.surf_hist, s.dih_hist]).astype(np.float32)

def _l2_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / norms

def deduplicate(signatures: List[PartSig], d2_sim_threshold: float, bbox_tol: float, logger: logging.Logger, device: str="auto") -> Tuple[Dict[str,str], Dict[str,List[str]]]:
    logger.info(f"[Dedup] start: n={len(signatures)}, d2_sim_threshold={d2_sim_threshold}, bbox_tol={bbox_tol}, device={device}")
    ids = [s.part_id for s in signatures]; dsu = DSU(ids)

    by_ch: Dict[str,List[PartSig]] = {}
    for s in signatures: by_ch.setdefault(s.content_hash, []).append(s)
    for lst in by_ch.values():
        if len(lst)>1:
            canon = lst[0].part_id
            for x in lst[1:]: dsu.union(canon, x.part_id)

    with_geom = [s for s in signatures if s.has_points and s.d2_hist is not None]
    merged_pairs = 0
    if with_geom:
        F = np.stack([pack_feat(s) for s in with_geom]); F = _l2_normalize(F).astype(np.float32, copy=False)
        prog = ProgressPrinter(total=len(with_geom), prefix="[Dedup geom]")

        use_gpu = (device=="gpu"); 
        if use_gpu: _ensure_faiss()
        if use_gpu and _GPU_FAISS and faiss is not None:
            prog.print_start()
            res = faiss.StandardGpuResources()
            index = faiss.GpuIndexFlatIP(res, F.shape[1]); index.add(F)
            k = min(10, F.shape[0]); sims, idxs = index.search(F, k)
            for ii, si in enumerate(with_geom):
                for sim, jpos in zip(sims[ii], idxs[ii]):
                    if jpos == ii or jpos < 0: continue
                    sj = with_geom[jpos]
                    br_ok = float(np.max(np.abs(si.bbox_ratio - sj.bbox_ratio))) <= bbox_tol
                    if float(sim) >= d2_sim_threshold and br_ok:
                        if dsu.find(si.part_id)!=dsu.find(sj.part_id):
                            dsu.union(si.part_id, sj.part_id); merged_pairs += 1
                prog.update(ii+1)
            prog.finish()
        else:
            prog.print_start()
            nbrs = NearestNeighbors(metric="cosine", n_neighbors=min(10,len(with_geom))).fit(F)
            distances, indices = nbrs.kneighbors(F, return_distance=True)
            for ii, si in enumerate(with_geom):
                for d, jpos in zip(distances[ii], indices[ii]):
                    if jpos==ii: continue
                    sj = with_geom[jpos]
                    sim = 1.0 - float(d)
                    br_ok = float(np.max(np.abs(si.bbox_ratio - sj.bbox_ratio))) <= bbox_tol
                    if sim >= d2_sim_threshold and br_ok:
                        if dsu.find(si.part_id)!=dsu.find(sj.part_id):
                            dsu.union(si.part_id, sj.part_id); merged_pairs += 1
                prog.update(ii+1)
            prog.finish()

    dup2canon: Dict[str,str] = {}; groups: Dict[str,List[str]] = {}
    for s in signatures:
        root = dsu.find(s.part_id); dup2canon[s.part_id]=root; groups.setdefault(root,[]).append(s.part_id)
    logger.info(f"[Dedup] done: geom-merged-pairs={merged_pairs}, duplicate_groups={sum(1 for g in groups.values() if len(g)>1)}")
    return dup2canon, groups

def _auto_eps_from_knn(F: np.ndarray,
                       k: int,
                       quantile: float,
                       scale: float,
                       sample_n: Optional[int],
                       seed: int,
                       device: str,
                       logger: logging.Logger) -> float:
    """基于 L2 归一化特征 F（Nxd）的第 k 邻居欧氏距离分位数，计算 eps。"""
    N = F.shape[0]
    if N <= 1:
        return 0.0
    k_search = min(max(2, k+1), N)  # +1含self
    rng = np.random.default_rng(seed)
    if sample_n and sample_n < N:
        idx = rng.choice(N, size=sample_n, replace=False)
        Q = F[idx]
        n_query = sample_n
    else:
        Q = F
        idx = None
        n_query = N

    use_gpu = (device=="gpu")
    # 优先 FAISS L2（返回平方距离）
    if use_gpu:
        _ensure_faiss()
    d_k = None
    if use_gpu and _GPU_FAISS and faiss is not None:
        res = faiss.StandardGpuResources()
        # 索引库：全量 F
        index = faiss.GpuIndexFlatL2(res, F.shape[1])
        index.add(F.astype(np.float32, copy=False))
        # 查询：Q
        D2, I = index.search(Q.astype(np.float32, copy=False), k_search)  # D2: squared L2
        # 对每行，取第 k 个邻居（0是self）
        pos = min(k, k_search-1)
        d = np.sqrt(D2[:, pos])  # 转欧氏距离
        d_k = d
    else:
        nbrs = NearestNeighbors(metric="euclidean", n_neighbors=k_search, n_jobs=-1).fit(F)
        dists, inds = nbrs.kneighbors(Q, return_distance=True)
        pos = min(k, k_search-1)
        d_k = dists[:, pos]

    qv = float(np.quantile(d_k, quantile))
    eps = qv * float(scale)
    # 记录抽样/超参
    logger.info(f"[Family|auto_eps] k={k}, quantile={quantile}, scale={scale}, sample={n_query}/{N}, eps={eps:.6f}, Tc≈{(eps*eps)/2:.6f}")
    return eps

def group_families(signatures: List[PartSig],
                   distance_threshold: float,
                   logger: logging.Logger,
                   device: str="auto",
                   min_samples: int = 5,
                   auto_eps: bool = False,
                   auto_eps_quantile: float = 0.25,
                   auto_eps_scale: float = 1.05,
                   auto_eps_sample_n: Optional[int] = 200000,
                   auto_eps_knn_k: Optional[int] = None,
                   auto_eps_seed: int = 42) -> Dict[str,str]:
    """
    家族聚类（DBSCAN）
    - 特征：拼接(D2直方图+bbox比+表面类型直方图+二面角直方图)，L2归一化后用欧氏距离。
    - 手动阈值：distance_threshold 是“余弦距离阈值”Tc，转欧氏半径 eps = sqrt(2*Tc)。
    - 自动半径：auto_eps=True 时，基于 kNN 第 k 邻居的欧氏距离分位数确定 eps。
    """
    logger.info(f"[Family] start: n={len(signatures)}, Tc={distance_threshold}, min_samples={min_samples}, device={device}, auto_eps={auto_eps}")
    with_geom = [s for s in signatures if s.has_points and s.d2_hist is not None]
    fam_map: Dict[str,str] = {}
    if with_geom:
        F = np.stack([pack_feat(s) for s in with_geom]).astype(np.float32)
        F = _l2_normalize(F)
        # 计算 eps（欧氏半径）
        if auto_eps:
            k_for_eps = int(auto_eps_knn_k) if (auto_eps_knn_k is not None and int(auto_eps_knn_k)>0) else int(max(1, min_samples))
            eps_euclid = _auto_eps_from_knn(
                F=F, k=k_for_eps, quantile=float(auto_eps_quantile), scale=float(auto_eps_scale),
                sample_n=(int(auto_eps_sample_n) if auto_eps_sample_n else None),
                seed=int(auto_eps_seed), device=device, logger=logger
            )
            if not np.isfinite(eps_euclid) or eps_euclid <= 0.0:
                # 回退：手动阈值
                eps_euclid = float(math.sqrt(max(1e-12, 2.0 * float(distance_threshold))))
                logger.warning(f"[Family|auto_eps] invalid eps from kNN; fallback eps={eps_euclid:.6f} (Tc≈{(eps_euclid*eps_euclid)/2:.6f})")
        else:
            eps_euclid = float(math.sqrt(max(1e-12, 2.0 * float(distance_threshold))))
            logger.info(f"[Family] manual eps={eps_euclid:.6f} (Tc≈{(eps_euclid*eps_euclid)/2:.6f})")

        use_gpu = (device=="gpu"); 
        if use_gpu: _ensure_cuml()
        if use_gpu and _GPU_CUML and cuDBSCAN is not None:
            db = cuDBSCAN(eps=eps_euclid, min_samples=int(min_samples), metric="euclidean")
            labels = db.fit_predict(F)
        else:
            db = skDBSCAN(eps=eps_euclid, min_samples=int(min_samples), metric="euclidean", n_jobs=-1)
            labels = db.fit_predict(F)
        for pid, lab in zip([s.part_id for s in with_geom], labels):
            fam_map[pid] = f"fam_{int(lab)}"
        logger.info(f"[Family] with_geom={len(with_geom)}, clusters={len(set(labels))}, eps={eps_euclid:.6f}, device={'gpu' if (use_gpu and _GPU_CUML and cuDBSCAN is not None) else 'cpu'}")
    # 无几何者：按内容哈希自成簇
    for s in signatures:
        if s.part_id not in fam_map:
            fam_map[s.part_id] = f"fam_ch_{s.content_hash[:8]}"
    return fam_map

# === JSON 序列化 ===
def _jsonable(x):
    if x is None: return None
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, (np.generic,)): return x.item()
    return x

def run_pipeline(raw_root: str, out_root: str,
                 subset_n: Optional[int], subset_frac: Optional[float],
                 sample_points: int,
                 d2_sim_threshold: float, bbox_tol: float,
                 family_distance_threshold: float,
                 logger: logging.Logger,
                 device: str="auto",
                 workers: int = 1,
                 mp_start: str = "spawn",
                 log_file: Optional[str] = None,
                 per_file_timeout_sec: Optional[int] = None,
                 max_inflight: Optional[int] = None,
                 max_inflight_multiplier: Optional[int] = None,
                 family_min_samples: int = 5,
                 # auto_eps 配置
                 family_auto_eps_enabled: bool = False,
                 family_auto_eps_quantile: float = 0.25,
                 family_auto_eps_scale: float = 1.05,
                 family_auto_eps_sample_n: Optional[int] = 200000,
                 family_auto_eps_knn_k: Optional[int] = None,
                 family_auto_eps_seed: int = 42) -> Dict[str,str]:
    os.makedirs(out_root, exist_ok=True)

    logger.info(f"[Config] raw_root={raw_root}")
    logger.info(f"[Config] out_root={out_root}")
    logger.info(f"[Config] sample_points={sample_points}, subset_n={subset_n}, subset_frac={subset_frac}")
    logger.info(f"[Config] d2_sim_threshold={d2_sim_threshold}, bbox_tol={bbox_tol}")
    logger.info(f"[Config] family_distance_threshold(Tc)={family_distance_threshold}, family_min_samples={family_min_samples}")
    logger.info(f"[Config] auto_eps={{enabled:{family_auto_eps_enabled}, q:{family_auto_eps_quantile}, scale:{family_auto_eps_scale}, sample_n:{family_auto_eps_sample_n}, knn_k:{family_auto_eps_knn_k}, seed:{family_auto_eps_seed}}}")
    logger.info(f"[Device] requested={device} (GPU libs lazy import)")
    logger.info(f"[Config] per_file_timeout_sec={per_file_timeout_sec or 'none'}, max_inflight={max_inflight or (f'{max_inflight_multiplier}x{workers}' if max_inflight_multiplier else 'default')}, mp_start={mp_start}")

    logger.info("[Stage] 1/4 Discover files")
    files = discover_parts(raw_root, subset_n, subset_frac, logger)
    if not files: raise SystemExit("No STEP/BREP files found under raw_root.")

    logger.info("[Stage] 2/4 Compute signatures")
    sigs, fails, metas = compute_signatures(raw_root, files, sample_points, logger, device=device,
                                            workers=workers, mp_start=mp_start, log_file=log_file,
                                            timeout_sec=per_file_timeout_sec,
                                            max_inflight=max_inflight,
                                            max_inflight_multiplier=max_inflight_multiplier)
    if len(sigs) == 0:
        if fails:
            with open(os.path.join(out_root,"failures.txt"),"w",encoding="utf-8") as f:
                f.write("\n".join(fails))
        raise SystemExit(2)

    recs=[]
    for s in sigs:
        rec = asdict(s)
        for k in ["d2_hist","bbox_ratio","surf_hist","dih_hist"]:
            val = getattr(s, k)
            rec[k] = json.dumps(_jsonable(val)) if val is not None else None
        recs.append(rec)
    sig_csv = os.path.join(out_root,"signatures.csv")
    pd.DataFrame(recs).to_csv(sig_csv, index=False)
    logger.info(f"[Output] signatures.csv -> {sig_csv}")

    logger.info("[Stage] 3/4 Deduplicate (content hash + geometry)")
    dup2canon, groups = deduplicate(sigs, d2_sim_threshold, bbox_tol, logger, device=device)

    logger.info("[Stage] 4/4 Family clustering (DBSCAN)")
    fam = group_families(
        signatures=sigs,
        distance_threshold=family_distance_threshold,
        logger=logger, device=device,
        min_samples=family_min_samples,
        auto_eps=family_auto_eps_enabled,
        auto_eps_quantile=family_auto_eps_quantile,
        auto_eps_scale=family_auto_eps_scale,
        auto_eps_sample_n=family_auto_eps_sample_n,
        auto_eps_knn_k=family_auto_eps_knn_k,
        auto_eps_seed=family_auto_eps_seed
    )

    rows=[]
    for s in sigs:
        meta = metas.get(s.part_id, {})
        created_hdr = meta.get("created_at_header")
        file_mtime = meta.get("file_mtime_utc")
        timestamp_source = meta.get("timestamp_source") or ("header" if created_hdr else ("mtime" if file_mtime else "none"))
        unit = meta.get("unit")
        unit_source = meta.get("unit_source") or ("unknown" if unit is None else "inferred_step_data")
        rows.append({
            "part_id": s.part_id,
            "rel_path": s.rel_path,
            "source_dataset": s.source_dataset,
            "repo": meta.get("repo"),
            "format": meta.get("format"),
            "file_ext": meta.get("file_ext"),
            "file_size_bytes": meta.get("file_size_bytes"),
            "file_mtime_utc": file_mtime,
            "timestamp_source": timestamp_source,
            "step_schema": meta.get("step_schema"),
            "originating_system": meta.get("originating_system"),
            "author": meta.get("author"),
            "created_at_header": created_hdr,
            "kernel": meta.get("kernel"),
            "unit": unit,
            "unit_source": unit_source,
            "domain_hint": meta.get("domain_hint"),
            "content_hash": s.content_hash,
            "geom_hash": s.geom_hash or "",
            "duplicate_canonical": dup2canon[s.part_id],
            "family_id": fam[s.part_id],
            "has_points": int(s.has_points),
        })
    part_index_csv = os.path.join(out_root,"part_index.csv")
    pd.DataFrame(rows).to_csv(part_index_csv, index=False)
    logger.info(f"[Output] part_index.csv -> {part_index_csv}")

    dup_groups_json = os.path.join(out_root,"duplicate_groups.json")
    with open(dup_groups_json,"w",encoding="utf-8") as f:
        json.dump(groups, f, indent=2)
    logger.info(f"[Output] duplicate_groups.json -> {dup_groups_json}")

    summary = {
        "num_parts": len(sigs),
        "num_failed": len(fails),
        "num_duplicate_groups": sum(1 for g in groups.values() if len(g)>1),
        "num_families": len(set(fam.values())),
        "device": device,
        "gpu_available": {"cupy": _GPU_CUPY, "faiss": _GPU_FAISS, "cuml": _GPU_CUML},
        "out_files": {
            "signatures": sig_csv,
            "part_index": part_index_csv,
            "duplicate_groups": dup_groups_json,
        }
    }
    summary_json = os.path.join(out_root,"summary.json")
    with open(summary_json,"w",encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"[Output] summary.json -> {summary_json}")
    logger.info(f"[Summary] {json.dumps(summary, ensure_ascii=False)}")

    if fails:
        failures_txt = os.path.join(out_root,"failures.txt")
        with open(failures_txt,"w",encoding="utf-8") as f:
            f.write("\n".join([x for x in fails if x]))
        logger.info(f"[Output] failures.txt -> {failures_txt}")

    logger.info("[Done] S2 pipeline finished successfully.")
    return summary

def demo_build_occ_step(root: str, logger: logging.Logger):
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    os.makedirs(root, exist_ok=True)
    box = BRepPrimAPI_MakeBox(10.0,10.0,10.0).Solid()
    box_dup = BRepPrimAPI_MakeBox(10.0,10.0,10.0).Solid()
    box_scaled = BRepPrimAPI_MakeBox(13.0,13.0,13.0).Solid()
    cyl = BRepPrimAPI_MakeCylinder(4.0, 12.0).Solid()
    def write_step(shape, path):
        w = STEPControl_Writer()
        w.Transfer(shape, STEPControl_AsIs)
        status = w.Write(path)
        return status == IFSelect_RetDone
    ok1 = write_step(box,       os.path.join(root,"DATASETA_box.step"))
    ok2 = write_step(box_dup,   os.path.join(root,"DATASETA_box_dup.step"))
    ok3 = write_step(box_scaled,os.path.join(root,"DATASETB_box_scaled.step"))
    ok4 = write_step(cyl,       os.path.join(root,"DATASETB_cylinder.step"))
    logger.info(f"[Demo-OCC] STEP written ok={sum([ok1,ok2,ok3,ok4])}/4 at {root}")

# ---------------- 配置读取 ----------------
def _cfg_get(d: dict, key: str, default):
    v = d.get(key, default)
    return v if v is not None else default

def load_config(config_path: str) -> Tuple[dict, dict]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_all = json.load(f)
    s2 = cfg_all.get("s2_dedup_family_occ", {})
    log_cfg = cfg_all.get("log", {})
    return s2, log_cfg

def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("用法: python s2_dedup_family_occ.py <config.json>\n说明: 参数从 config.json 的 s2_dedup_family_occ 段读取。")
        sys.exit(0)

    config_path = sys.argv[1]
    if not os.path.isfile(config_path):
        print(f"[ERROR] 配置文件不存在: {config_path}"); sys.exit(1)

    s2cfg, log_cfg = load_config(config_path)

    raw_root  = _cfg_get(s2cfg, "raw_root", None)
    out_root  = _cfg_get(s2cfg, "out_root", None)
    subset_n  = _cfg_get(s2cfg, "subset_n", None)
    subset_frac = _cfg_get(s2cfg, "subset_frac", None)
    sample_points = int(_cfg_get(s2cfg, "sample_points", 4096))
    d2_sim_threshold = float(_cfg_get(s2cfg, "d2_sim_threshold", 0.995))
    bbox_tol = float(_cfg_get(s2cfg, "bbox_tol", 0.02))
    family_distance_threshold = float(_cfg_get(s2cfg, "family_distance_threshold", 0.02))
    device_req = _cfg_get(s2cfg, "device", "auto")
    demo_occ  = bool(_cfg_get(s2cfg, "demo_occ", False))
    log_level = _cfg_get(s2cfg, "log_level", "info")
    workers   = int(_cfg_get(s2cfg, "workers", os.cpu_count() or 1))
    mp_start  = _cfg_get(s2cfg, "mp_start", "spawn")
    per_file_timeout_sec = _cfg_get(s2cfg, "per_file_timeout_sec", None)
    max_inflight = _cfg_get(s2cfg, "max_inflight", None)
    max_inflight_multiplier = _cfg_get(s2cfg, "max_inflight_multiplier", 4)

    # 新增：family_min_samples
    family_min_samples = int(_cfg_get(s2cfg, "family_min_samples", 5))

    # 新增：family_auto_eps（可为 bool 或 dict）
    auto_raw = _cfg_get(s2cfg, "family_auto_eps", False)
    if isinstance(auto_raw, bool):
        family_auto_eps_enabled = auto_raw
        family_auto_eps_quantile = 0.25
        family_auto_eps_scale = 1.05
        family_auto_eps_sample_n = 200000
        family_auto_eps_knn_k = None
        family_auto_eps_seed = 42
    else:
        family_auto_eps_enabled = bool(_cfg_get(auto_raw, "enabled", True))
        family_auto_eps_quantile = float(_cfg_get(auto_raw, "quantile", 0.25))
        family_auto_eps_scale = float(_cfg_get(auto_raw, "scale", 1.05))
        family_auto_eps_sample_n = _cfg_get(auto_raw, "sample_n", 200000)
        family_auto_eps_knn_k = _cfg_get(auto_raw, "knn_k", None)
        family_auto_eps_seed = int(_cfg_get(auto_raw, "seed", 42))

    log_file = s2cfg.get("log_file", None) or log_cfg.get("file", None)
    logger = setup_logger(log_level, log_file=log_file)
    logger.info(f"[Boot] using config: {os.path.abspath(config_path)}")
    logger.info(f"[Boot] log_level={log_level}, log_file={log_file or 'STDOUT only'}")

    if demo_occ:
        if not out_root:
            logger.error("demo_occ 模式需要 out_root"); sys.exit(1)
        demo_root = os.path.join(out_root, "raw_occ_demo")
        demo_build_occ_step(demo_root, logger)
        raw_root = demo_root

    if not raw_root or not out_root:
        logger.error("raw_root 与 out_root 必填"); sys.exit(1)

    device_eff = device_req
    os.makedirs(out_root, exist_ok=True)
    run_pipeline(
        raw_root=raw_root, out_root=out_root,
        subset_n=subset_n, subset_frac=subset_frac,
        sample_points=sample_points,
        d2_sim_threshold=d2_sim_threshold, bbox_tol=bbox_tol,
        family_distance_threshold=family_distance_threshold,
        logger=logger, device=device_eff,
        workers=max(1, workers or 1),
        mp_start=mp_start,
        log_file=log_file,
        per_file_timeout_sec=per_file_timeout_sec,
        max_inflight=max_inflight,
        max_inflight_multiplier=max_inflight_multiplier,
        family_min_samples=family_min_samples,
        family_auto_eps_enabled=family_auto_eps_enabled,
        family_auto_eps_quantile=family_auto_eps_quantile,
        family_auto_eps_scale=family_auto_eps_scale,
        family_auto_eps_sample_n=family_auto_eps_sample_n,
        family_auto_eps_knn_k=family_auto_eps_knn_k,
        family_auto_eps_seed=family_auto_eps_seed
    )

if __name__ == "__main__":
    main()

# 备注：
# - auto_eps 开启后优先使用 kNN 距离自动确定 eps；若异常则回退到手动 Tc（family_distance_threshold）。
# - eps 为欧氏半径；等效的余弦距离阈值 Tc≈eps^2/2（因 L2 归一化下 ||x-y||^2 = 2*Dcos）。
# - 推荐 min_samples>=5，quantile≈0.2~0.3，scale≈1.02~1.10。

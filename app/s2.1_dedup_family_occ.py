#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S2.1: 发现STEP/BREP文件 + 并行提取几何签名（OCC），落盘 signatures.csv 与 metas.json

保持原逻辑：候选文件发现 -> OCC 载入 -> 采样点云 -> D2直方图/包围盒比例/曲面类型直方图/二面角直方图
多进程 worker（父进程看门狗），GPU优先（CuPy做D2直方图采样），控制台1%进度；关键日志3-5行。
"""

import os, re, sys, json, math, time, hashlib, argparse, logging, traceback
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import multiprocessing as mp
from queue import Empty

# ---------- Logging ----------
def _to_level(level: str) -> int:
    import logging as _lg
    return {"debug": _lg.DEBUG, "info": _lg.INFO, "warning": _lg.WARNING}.get((level or "info").lower(), _lg.INFO)

def setup_logger(level: str = "info", log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger("S2_1")
    logger.setLevel(_to_level(level))
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        sh = logging.StreamHandler(); sh.setFormatter(fmt); logger.addHandler(sh)
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            fh = logging.FileHandler(log_file, encoding="utf-8"); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger

class ProgressPrinter:
    def __init__(self, total: int, prefix: str=""):
        self.total = max(1, int(total)); self.prefix = prefix; self.last = -1; self.done=False
    def _emit(self, p): print(f"\r{self.prefix} {p:3d}%", end="", flush=True)
    def print_start(self): self._emit(0); self.last=0
    def update(self, cur:int):
        if self.done: return
        p = int(min(max(0, cur), self.total) * 100 / self.total)
        if p >= self.last + 1: self._emit(p); self.last = p
        if cur >= self.total: self.finish()
    def finish(self):
        if not self.done: self._emit(100); print(); self.done=True

# ---------- GPU lazy flags ----------
cp = None
_GPU_CUPY = False
def _ensure_cupy():
    global cp, _GPU_CUPY
    if cp is not None or _GPU_CUPY: return
    try:
        import cupy as _cp
        cp=_cp; _GPU_CUPY=True
    except Exception:
        cp=None; _GPU_CUPY=False

# ---------- OCC imports ----------
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

SUPPORTED_STEP = {".step", ".stp", ".stpz"}
SUPPORTED_BREP = {".brep", ".brp"}

# ---------- STEP header parsing ----------
_STEP_FILE_NAME_RE = re.compile(
    r"FILE_NAME\s*\(\s*'([^']*)'\s*,\s*'([^']*)'\s*,\s*\((.*?)\)\s*,\s*\((.*?)\)\s*,\s*'([^']*)'\s*,\s*'([^']*)'\s*,\s*'([^']*)'\s*\)",
    re.IGNORECASE | re.DOTALL)
_STEP_FILE_SCHEMA_RE = re.compile(r"FILE_SCHEMA\s*\(\s*\(\s*'([^']+)'\s*\)\s*\)", re.IGNORECASE)

def read_text_prefix(path: str, max_bytes: int = 200000) -> Optional[str]:
    try:
        with open(path, "rb") as f: b = f.read(max_bytes)
        return b.decode("utf-8", errors="ignore")
    except Exception:
        try:
            return b.decode("latin1", errors="ignore")
        except Exception:
            return None

def _normalize_datetime(s: str) -> Optional[str]:
    s = (s or "").strip()
    for cand in [s, s.replace(" ", "T")]:
        try:
            dt = datetime.fromisoformat(cand)
            return (dt if dt.tzinfo is None else dt.astimezone(timezone.utc)).isoformat()
        except Exception: pass
    for fmt in ["%Y-%m-%dT%H:%M:%S","%Y%m%dT%H%M%S","%Y-%m-%d %H:%M:%S"]:
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
    try: mtime = os.path.getmtime(fpath); mtime_iso = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
    except Exception: mtime_iso = None
    ext = os.path.splitext(fpath)[1].lower()
    fmt = "STEP" if ext in SUPPORTED_STEP else ("BREP" if ext in SUPPORTED_BREP else "UNKNOWN")
    return {"file_ext": ext, "format": fmt, "file_size_bytes": int(size) if size is not None else None, "file_mtime_utc": mtime_iso}

def guess_kernel(fmt: str, originating_system: Optional[str]) -> Optional[str]:
    if fmt == "BREP": return "OpenCascade"
    if fmt == "STEP":
        s = (originating_system or "").lower()
        if "parasolid" in s or "nx" in s or "solidworks" in s: return "Parasolid"
        if "acis" in s or "autocad" in s or "inventor" in s:   return "ACIS"
        if "catia" in s or "cgm" in s or "3dexperience" in s or "solidedge" in s: return "CGM"
        return "STEP"
    return None

# ---------- OCC I/O & features ----------
def sha256_file(path: str, chunk: int = 1<<20) -> str:
    h=hashlib.sha256()
    with open(path,"rb") as f:
        while True:
            b=f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def occ_load_shape(path: str, logger: logging.Logger) -> Optional["TopoDS_Shape"]:
    ext = os.path.splitext(path)[1].lower()
    if ext in SUPPORTED_STEP:
        reader = STEPControl_Reader()
        status = reader.ReadFile(path)
        if status != IFSelect_RetDone:
            logger.debug(f"[OCC] STEP read failed: {path}"); return None
        reader.TransferRoots(); return reader.OneShape()
    elif ext in SUPPORTED_BREP:
        shape = TopoDS_Shape()
        try: ok = breptools_Read(shape, path, BRep_Builder())
        except TypeError: ok = breptools_Read(shape, path)
        if not ok: logger.debug(f"[OCC] BREP read failed: {path}"); return None
        return shape
    return None

def triangulate(shape, lin_defl=0.5):
    try: BRepMesh_IncrementalMesh(shape, lin_defl, True, 0.5, True)
    except TypeError: BRepMesh_IncrementalMesh(shape, deflection=lin_defl, isRelative=True, angle=0.5, parallel=True)

def _append_nodes_from_triangulation(face: TopoDS_Face, loc: TopLoc_Location, pts_out: list):
    tri = BRep_Tool.Triangulation(face, loc); 
    if not tri: return 0
    tri_obj = tri.GetObject() if hasattr(tri, "GetObject") else tri
    n_added = 0
    try:
        if hasattr(tri_obj,"Nodes") and hasattr(tri_obj,"NbNodes"):
            nodes = tri_obj.Nodes(); n = tri_obj.NbNodes()
            for i in range(1,n+1):
                p = nodes.Value(i).Transformed(loc.Transformation())
                pts_out.append([p.X(),p.Y(),p.Z()]); n_added+=1
            return n_added
    except Exception: pass
    try:
        if hasattr(tri_obj,"NbNodes") and hasattr(tri_obj,"Node"):
            n=tri_obj.NbNodes()
            for i in range(1,n+1):
                p=tri_obj.Node(i).Transformed(loc.Transformation())
                pts_out.append([p.X(),p.Y(),p.Z()]); n_added+=1
            return n_added
    except Exception: pass
    return n_added

def occ_shape_points(shape: "TopoDS_Shape", n: int = 4096, lin_defl: float = 0.5) -> np.ndarray:
    triangulate(shape, lin_defl=lin_defl)
    pts=[]
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = topods.Face(exp.Current())
        loc = TopLoc_Location()
        _append_nodes_from_triangulation(face, loc, pts)
        exp.Next()
    if len(pts)==0:
        bbox = Bnd_Box(); brepbndlib_Add(shape, bbox, False)
        xmin,ymin,zmin,xmax,ymax,zmax = bbox.Get()
        pts = [[xmin,ymin,zmin],[xmax,ymin,zmin],[xmin,ymax,zmin],[xmax,ymax,zmin],[xmin,ymin,zmax],[xmax,ymin,zmax],[xmin,ymax,zmax],[xmax,ymax,zmax]]
    pts = np.array(pts, dtype=np.float64)
    if pts.shape[0] > n:
        idx = np.random.default_rng(0).choice(pts.shape[0], size=n, replace=False)
        pts = pts[idx]
    return pts

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
        i = rng.integers(0, n, size=num_pairs); j = rng.integers(0, n, size=num_pairs)
        p = cp.asarray(points)
        d = cp.linalg.norm(p[i]-p[j], axis=1)
        d = cp.clip(d / (math.sqrt(3)+1e-9), 0.0, 1.0)
        hist,_ = cp.histogram(d, bins=num_bins, range=(0,1))
        hist = hist.astype(cp.float64); hist /= (hist.sum() + 1e-12)
        return cp.asnumpy(hist)
    else:
        rng = np.random.default_rng(seed)
        i = rng.integers(0, n, size=num_pairs); j = rng.integers(0, n, size=num_pairs)
        d = np.linalg.norm(points[i]-points[j], axis=1)
        d = np.clip(d/(math.sqrt(3)+1e-9), 0.0, 1.0)
        hist,_ = np.histogram(d, bins=64, range=(0,1))
        hist = hist.astype(np.float64); hist /= (hist.sum()+1e-12)
        return hist

def bbox_ratios(points: np.ndarray) -> np.ndarray:
    mins = points.min(axis=0); maxs = points.max(axis=0)
    l1,l2,l3 = np.sort((maxs - mins).astype(np.float64))
    return np.array([l1/(l3+1e-12), l2/(l3+1e-12)], np.float64)

def occ_surface_type_hist(shape: "TopoDS_Shape") -> np.ndarray:
    counters=dict(plane=0,cylinder=0,cone=0,sphere=0,torus=0,free=0); total=0
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
        c = BRepAdaptor_Curve(edge); u0,u1 = c.FirstParameter(), c.LastParameter()
        p0,p1 = c.Value(u0), c.Value(u1)
        a = np.round([p0.X(),p0.Y(),p0.Z()], 6); b = np.round([p1.X(),p1.Y(),p1.Z()], 6)
        t0,t1 = tuple(a), tuple(b); return (t0,t1) if t0<=t1 else (t1,t0)
    edge_map={}
    fexp = TopExp_Explorer(shape, TopAbs_FACE)
    while fexp.More():
        face = topods.Face(fexp.Current())
        eexp = TopExp_Explorer(face, TopAbs_EDGE)
        while eexp.More():
            edge = topods.Edge(eexp.Current())
            try: k = edge_key(edge)
            except Exception: eexp.Next(); continue
            rec = edge_map.get(k); 
            if rec is None: rec = {"edge": edge, "faces": []}; edge_map[k]=rec
            rec["faces"].append(face); eexp.Next()
        fexp.Next()
    angles=[]
    for rec in edge_map.values():
        faces = rec["faces"]; 
        if len(faces)<2: continue
        edge = rec["edge"]; c = BRepAdaptor_Curve(edge); tmid = 0.5*(c.FirstParameter()+c.LastParameter()); P = c.Value(tmid)
        def face_normal_at(face,P):
            srf = BRep_Tool.Surface(face); uv = ShapeAnalysis_Surface(srf).ValueOfUV(P, 1e-6)
            props = GeomLProp_SLProps(srf, uv.X(), uv.Y(), 1, 1e-6)
            if not props.IsNormalDefined(): return None
            n = props.Normal(); v = np.array([n.X(),n.Y(),n.Z()], np.float64)
            if face.Orientation()==TopAbs_REVERSED: v=-v
            return v/(np.linalg.norm(v)+1e-12)
        n1 = face_normal_at(faces[0],P); n2 = face_normal_at(faces[1],P)
        if n1 is None or n2 is None: continue
        c12 = float(np.clip(np.dot(n1,n2), -1.0, 1.0)); ang = math.acos(c12); angles.append(ang)
    if not angles: return np.zeros(bins, dtype=np.float64)
    hist,_ = np.histogram(angles, bins=bins, range=(0, math.pi), density=False)
    hist = hist.astype(np.float64); hist /= (hist.sum()+1e-12)
    return hist

# ---------- dataclass ----------
@dataclass
class PartSig:
    part_id: str
    rel_path: str
    source_dataset: str
    content_hash: str
    has_points: bool
    d2_hist: Optional[np.ndarray]=None
    bbox_ratio: Optional[np.ndarray]=None
    surf_hist: Optional[np.ndarray]=None
    dih_hist: Optional[np.ndarray]=None
    geom_hash: Optional[str]=None

# ---------- Discover & meta ----------
def safe_rel(root: str, path: str) -> str:
    try: return os.path.relpath(path, root)
    except Exception: return path

def discover_parts(raw_root: str, subset_n: Optional[int], subset_frac: Optional[float], logger: logging.Logger) -> List[str]:
    files=[]; step_cnt=0; brep_cnt=0
    for r,_,fns in os.walk(raw_root, followlinks=True):
        for fn in fns:
            ext = os.path.splitext(fn)[1].lower()
            if ext in (SUPPORTED_STEP|SUPPORTED_BREP):
                p = os.path.join(r,fn); files.append(p)
                if ext in SUPPORTED_STEP: step_cnt+=1
                else: brep_cnt+=1
    files.sort(); total=len(files); selected=files
    if subset_frac and 0<subset_frac<1: selected=selected[:max(1,int(total*subset_frac))]
    if subset_n: selected=selected[:subset_n]
    logger.info(f"[Discover] STEP={step_cnt}, BREP={brep_cnt}, total={total}, selected={len(selected)}")
    return selected

def get_file_meta_all(raw_root: str, fpath: str) -> Dict[str, Optional[str]]:
    rel = safe_rel(raw_root, fpath); src = rel.split(os.sep)[0] if os.sep in rel else "unknown"
    ch = sha256_file(fpath); fm = get_file_meta(fpath)
    text = read_text_prefix(fpath, 300000) if (fm["format"]=="STEP" and fm["file_ext"]!=".stpz") else None
    step_hdr = parse_step_header_text(text or "") if text is not None else {"step_schema": None, "originating_system": None, "author": None, "created_at_header": None, "unit": None, "unit_source": None}
    kernel = guess_kernel(fm["format"], step_hdr.get("originating_system"))
    parts = rel.split(os.sep); repo = parts[1] if len(parts)>=2 else None
    created_hdr = step_hdr.get("created_at_header")
    ts_src = "header" if created_hdr else ("mtime" if fm.get("file_mtime_utc") else "none")
    return {"rel": rel, "src": src, "ch": ch, **fm, **step_hdr, "kernel": kernel, "repo": repo, "domain_hint": src, "timestamp_source": ts_src}

# ---------- Worker & signatures ----------
def compute_signature_for_file(raw_root: str, fpath: str, sample_points: int, logger: logging.Logger, device: str="auto") -> Tuple[Optional[PartSig], Optional[str], Dict[str, Optional[str]]]:
    meta = get_file_meta_all(raw_root, fpath)
    rel, src, ch = meta["rel"], meta["src"], meta["ch"]
    shape = occ_load_shape(fpath, logger)
    if shape is None:
        sig = PartSig(rel, rel, src, ch, False, None, None, None, None, None)
        return sig, f"load_shape_failed:{rel}", meta
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

def _worker_loop(wid: int, in_q: "mp.Queue", out_q: "mp.Queue", raw_root: str, sample_points: int, device: str, log_level: str, log_file: Optional[str]):
    logger = setup_logger(log_level, log_file=log_file)
    while True:
        task = in_q.get()
        if task is None: break
        idx, fpath = task
        try:
            sig, err, meta = compute_signature_for_file(raw_root, fpath, sample_points, logger, device=device)
        except Exception as e:
            sig, err, meta = None, f"worker_exception:{type(e).__name__}:{e}", {}
        out_q.put((wid, idx, sig, err, meta))

def compute_signatures(raw_root: str, files: List[str], sample_points: int, logger: logging.Logger, device: str="auto",
                       workers: int = 1, mp_start: str = "spawn", log_file: Optional[str] = None,
                       timeout_sec: Optional[int] = None, inflight_cap: Optional[int] = None) -> Tuple[List[PartSig], List[str], Dict[str, Dict[str, Optional[str]]]]:
    n = len(files); logger.info(f"[Signatures] start: files={n}, sample_points={sample_points}, device={device}, workers={workers}, timeout={timeout_sec or 'none'}s")
    t0 = time.time(); prog = ProgressPrinter(total=n, prefix="[S2.1]"); prog.print_start()
    try: ctx = mp.get_context(mp_start)
    except Exception: ctx = mp.get_context("spawn")
    out_q: mp.Queue = ctx.Queue()
    def spawn_worker(wid: int):
        in_q = ctx.Queue()
        wlog = "debug" if logger.level <= logging.DEBUG else "warning"
        p = ctx.Process(target=_worker_loop, args=(wid, in_q, out_q, raw_root, sample_points, device, wlog, log_file), daemon=False)
        p.start(); return {"id":wid,"proc":p,"in_q":in_q,"busy":False,"start":None,"task":None}
    cap = max(1, min(workers, inflight_cap if (inflight_cap and inflight_cap>0) else workers))
    ws = [spawn_worker(i) for i in range(cap)]
    next_i = 1; done = 0; results={}; fails=[]; metas={}
    while done < n:
        for st in ws:
            if next_i > n: break
            if not st["busy"]:
                f = files[next_i-1]
                try:
                    st["in_q"].put((next_i, f)); st["busy"]=True; st["start"]=time.monotonic(); st["task"]=(next_i,f)
                except Exception:
                    try:
                        if st["proc"].is_alive(): st["proc"].terminate(); st["proc"].join(timeout=3)
                    except Exception: pass
                    new=spawn_worker(st["id"]); st.update(new); st["in_q"].put((next_i,f)); st["busy"]=True; st["start"]=time.monotonic(); st["task"]=(next_i,f)
                next_i+=1
        pulled=False
        while True:
            try: wid, idx, sig, err, meta = out_q.get_nowait(); pulled=True
            except Empty: break
            for st in ws:
                if st["id"]==wid: st["busy"]=False; st["start"]=None; st["task"]=None; break
            results[idx]=(sig,err,meta)
            if sig is not None: metas[sig.part_id]=meta
            if err: fails.append(err)
            done+=1; prog.update(done)
        if timeout_sec and timeout_sec>0:
            now=time.monotonic()
            for st in ws:
                if st["busy"] and st["start"] and (now-st["start"]>timeout_sec):
                    idx,f = st["task"]; base=os.path.basename(f)
                    try:
                        if st["proc"].is_alive(): st["proc"].terminate(); st["proc"].join(timeout=3)
                    except Exception: pass
                    results[idx]=(None, f"timeout:{base}", {}); fails.append(f"timeout:{base}")
                    done+=1; prog.update(done)
                    new=spawn_worker(st["id"]); st.update(new)
        if not pulled: time.sleep(0.05)
    prog.finish()
    for st in ws:
        try: st["in_q"].put(None)
        except Exception: pass
    for st in ws:
        try: st["proc"].join(timeout=2)
        except Exception: pass
    sigs=[]
    for i in range(1,n+1):
        sig,err,meta = results.get(i,(None, f"missing_result_for_index:{i}", {}))
        if sig is not None:
            sigs.append(sig)
            if sig.part_id not in metas: metas[sig.part_id]=meta
        if err and err not in fails: fails.append(err)
    logger.info(f"[Signatures] done: ok={len(sigs)}, fail={len(fails)}, elapsed={time.time()-t0:.2f}s")
    return sigs, fails, metas

# ---------- Config / main ----------
def _cfg_get(d: dict, key: str, default):
    v = d.get(key, default); return v if v is not None else default

def load_config(config_path: str) -> Tuple[dict, dict]:
    with open(config_path,"r",encoding="utf-8") as f: cfg_all=json.load(f)
    return cfg_all.get("s2_dedup_family_occ", {}), cfg_all.get("log", {})

def main():
    if len(sys.argv)<2 or sys.argv[1] in ("-h","--help"):
        print("用法: python s2_1_signatures.py <config.json>"); sys.exit(0)
    config_path=sys.argv[1]
    s2cfg, log_cfg = load_config(config_path)
    raw_root  = _cfg_get(s2cfg,"raw_root",None)
    out_root  = _cfg_get(s2cfg,"out_root",None)
    subset_n  = _cfg_get(s2cfg,"subset_n",None)
    subset_frac = _cfg_get(s2cfg,"subset_frac",None)
    sample_points = int(_cfg_get(s2cfg,"sample_points",4096))
    device_req = _cfg_get(s2cfg,"device","auto")
    demo_occ  = bool(_cfg_get(s2cfg,"demo_occ",False))
    log_level = _cfg_get(s2cfg,"log_level","info")
    log_file  = s2cfg.get("log_file", None) or log_cfg.get("file", None)
    workers   = max(1, int(_cfg_get(s2cfg,"workers", os.cpu_count() or 1)))
    mp_start  = _cfg_get(s2cfg,"mp_start","spawn")
    per_file_timeout_sec = _cfg_get(s2cfg,"per_file_timeout_sec",None)
    max_inflight_multiplier = _cfg_get(s2cfg,"max_inflight_multiplier",4)
    max_inflight = _cfg_get(s2cfg,"max_inflight", None)
    if max_inflight is None:
        inflight = max(workers, int(max_inflight_multiplier)*int(workers))
    else:
        inflight = int(max_inflight)

    logger = setup_logger(log_level, log_file=log_file)
    logger.info(f"[S2.1] boot config={os.path.abspath(config_path)}")
    logger.info(f"[S2.1] raw_root={raw_root} out_root={out_root} device={device_req} workers={workers}")
    if not raw_root or not out_root:
        logger.error("raw_root 与 out_root 必填"); sys.exit(1)
    os.makedirs(out_root, exist_ok=True)

    # (可选) demo STEP 创建，保持原逻辑
    if demo_occ:
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
        demo_root = os.path.join(out_root,"raw_occ_demo"); os.makedirs(demo_root, exist_ok=True)
        def write_step(shape, path):
            w=STEPControl_Writer(); w.Transfer(shape, STEPControl_AsIs); return w.Write(path)==IFSelect_RetDone
        ok=[]
        ok.append(write_step(BRepPrimAPI_MakeBox(10,10,10).Solid(), os.path.join(demo_root,"DATASETA_box.step")))
        ok.append(write_step(BRepPrimAPI_MakeBox(10,10,10).Solid(), os.path.join(demo_root,"DATASETA_box_dup.step")))
        ok.append(write_step(BRepPrimAPI_MakeBox(13,13,13).Solid(), os.path.join(demo_root,"DATASETB_box_scaled.step")))
        ok.append(write_step(BRepPrimAPI_MakeCylinder(4.0,12.0).Solid(), os.path.join(demo_root,"DATASETB_cylinder.step")))
        logger.info(f"[S2.1] demo STEP written ok={sum(ok)}/4 at {demo_root}")
        raw_root = demo_root

    # Stage 1: discover
    files = discover_parts(raw_root, subset_n, subset_frac, logger)
    # 关键日志（少量）
    logger.info(f"[S2.1] Stage1 done: discovered={len(files)}")

    # Stage 2: signatures
    sigs, fails, metas = compute_signatures(
        raw_root, files, sample_points, logger, device=device_req,
        workers=workers, mp_start=mp_start, log_file=log_file,
        timeout_sec=per_file_timeout_sec, inflight_cap=inflight
    )

    # 落盘 signatures.csv
    recs=[]
    for s in sigs:
        rec = asdict(s)
        for k in ["d2_hist","bbox_ratio","surf_hist","dih_hist"]:
            v = getattr(s, k); rec[k] = json.dumps(v.tolist()) if v is not None else None
        recs.append(rec)
    sig_csv = os.path.join(out_root,"signatures.csv")
    pd.DataFrame(recs).to_csv(sig_csv, index=False)
    logger.info(f"[S2.1] Output signatures.csv -> {sig_csv}")

    # 落盘 metas.json
    metas_path = os.path.join(out_root,"metas.json")
    with open(metas_path,"w",encoding="utf-8") as f:
        json.dump(metas, f, indent=2, ensure_ascii=False)
    logger.info(f"[S2.1] Output metas.json -> {metas_path}")

    # 失败列表
    if fails:
        failures_txt = os.path.join(out_root,"failures.txt")
        with open(failures_txt,"w",encoding="utf-8") as f:
            f.write("\n".join([x for x in fails if x]))
        logger.info(f"[S2.1] Output failures.txt -> {failures_txt}")

    # 小结
    summary = {
        "num_parts": len(sigs),
        "num_failed": len(fails),
        "gpu_available": {"cupy": _GPU_CUPY},
        "out_files": {"signatures": sig_csv, "metas": metas_path, "failures": (os.path.join(out_root,"failures.txt") if fails else None)}
    }
    summary_json = os.path.join(out_root,"summary_s2_1.json")
    with open(summary_json,"w",encoding="utf-8") as f: json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"[S2.1] Output summary_s2_1.json -> {summary_json}")
    logger.info("[S2.1] DONE")

if __name__ == "__main__":
    main()

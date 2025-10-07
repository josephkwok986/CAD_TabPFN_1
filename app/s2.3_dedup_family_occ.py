#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S2.3 (改进版): 家族聚类（DBSCAN，自动eps可选） + 大簇细分（可选） + 生成 part_index.csv / summary.json
变更要点：
- 噪声点 label == -1 不再汇总到 "fam_-1"；而是每件形成 "fam_iso_<hash>"
- 自动 eps 默认更保守（quantile≈0.10, scale≈1.02, knn_k=min_samples），失败回退 Tc -> eps = sqrt(2*Tc)
- 可选：对超大簇做二次细分（family_post_split），避免巨型 family 垄断切分
- 额外输出 family_hist.csv、family_diagnostics.json，便于 S3/S4 前 sanity-check
- 新增：控制台 1% 粒度进度打印（特征打包/簇细分/family 赋值/补齐/写出）
"""

import os, sys, json, math, logging, hashlib
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# ---------- logging + progress ----------
def _to_level(level: str):
    import logging as _lg
    return {"debug": _lg.DEBUG, "info": _lg.INFO, "warning": _lg.WARNING}.get((level or "info").lower(), _lg.INFO)

def setup_logger(level: str="info", log_file: Optional[str]=None)->logging.Logger:
    logger = logging.getLogger("S2_3"); logger.setLevel(_to_level(level)); logger.propagate=False
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s","%Y-%m-%d %H:%M:%S")
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        sh=logging.StreamHandler(); sh.setFormatter(fmt); logger.addHandler(sh)
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            fh=logging.FileHandler(log_file, encoding="utf-8"); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger

class ProgressPrinter:
    def __init__(self,total:int,prefix:str=""):
        self.total=max(1,int(total)); self.prefix=prefix; self.last=-1; self.done=False
    def _emit(self,p): print(f"\r{self.prefix} {p:3d}%", end="", flush=True)
    def print_start(self): self._emit(0); self.last=0
    def update(self,cur:int):
        if self.done: return
        p = int(min(max(0,cur), self.total) * 100 / self.total)
        if p>=self.last+1:
            self._emit(p); self.last=p
        if cur>=self.total: self.finish()
    def finish(self):
        if not self.done:
            self._emit(100); print(); self.done=True

# ---------- GPU backends (FAISS / cuML) ----------
faiss=None; _GPU_FAISS=False
def _ensure_faiss():
    global faiss,_GPU_FAISS
    if faiss is not None: return
    try:
        import faiss as _faiss
        faiss=_faiss
        try: _ = _faiss.StandardGpuResources(); _GPU_FAISS=True
        except Exception: _GPU_FAISS=False
    except Exception:
        faiss=None; _GPU_FAISS=False

cuDBSCAN=None; _GPU_CUML=False
def _ensure_cuml():
    global cuDBSCAN,_GPU_CUML
    if cuDBSCAN is not None: return
    try:
        from cuml.cluster import DBSCAN as _cuDBSCAN
        cuDBSCAN=_cuDBSCAN; _GPU_CUML=True
    except Exception:
        cuDBSCAN=None; _GPU_CUML=False

from sklearn.cluster import DBSCAN as skDBSCAN
from sklearn.neighbors import NearestNeighbors

# ---------- utils ----------
def _l2_normalize(X: np.ndarray)->np.ndarray:
    return X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-12)

def _pack_feat(row)->np.ndarray:
    d2 = np.array(json.loads(row["d2_hist"]), dtype=np.float32) if pd.notna(row["d2_hist"]) else None
    br = np.array(json.loads(row["bbox_ratio"]), dtype=np.float32) if pd.notna(row["bbox_ratio"]) else None
    surf = np.array(json.loads(row["surf_hist"]), dtype=np.float32) if pd.notna(row["surf_hist"]) else None
    dih = np.array(json.loads(row["dih_hist"]), dtype=np.float32) if pd.notna(row["dih_hist"]) else None
    return np.concatenate([d2, br, surf, dih]).astype(np.float32)

def _auto_eps_from_knn(F: np.ndarray, k: int, quantile: float, scale: float,
                       sample_n: Optional[int], seed: int, device: str, logger: logging.Logger)->float:
    N = F.shape[0]
    if N<=1: return 0.0
    k_search = min(max(2, k+1), N)
    rng = np.random.default_rng(seed)
    if sample_n and sample_n < N:
        idx = rng.choice(N, size=int(sample_n), replace=False)
        Q = F[idx]
    else:
        Q = F

    d_k=None
    if device=="gpu":
        _ensure_faiss()
    if device=="gpu" and _GPU_FAISS and faiss is not None:
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, F.shape[1]); index.add(F.astype(np.float32, copy=False))
        D2,_ = index.search(Q.astype(np.float32, copy=False), k_search)  # squared L2
        pos=min(k,k_search-1); d_k = np.sqrt(D2[:,pos])
    else:
        nbrs = NearestNeighbors(metric="euclidean", n_neighbors=k_search, n_jobs=-1).fit(F)
        dists,_ = nbrs.kneighbors(Q, return_distance=True)
        pos=min(k,k_search-1); d_k=dists[:,pos]

    qv = float(np.quantile(d_k, quantile))
    eps = qv * float(scale)
    logger.info(f"[S2.3|auto_eps] k={k}, q={quantile}, scale={scale}, sample={len(Q)}/{N}, eps={eps:.6f}, Tc≈{(eps*eps)/2:.6f}")
    return eps

def _dbscan(F: np.ndarray, eps: float, min_samples: int, device: str):
    if device=="gpu":
        _ensure_cuml()
    if device=="gpu" and _GPU_CUML and cuDBSCAN is not None:
        lab = cuDBSCAN(eps=float(eps), min_samples=int(min_samples), metric="euclidean").fit_predict(F)
    else:
        lab = skDBSCAN(eps=float(eps), min_samples=int(min_samples), metric="euclidean", n_jobs=-1).fit_predict(F)
    return lab

def _apply_noise_policy(pid: str, dup2canon: dict, content_hash_map: dict, policy: str)->str:
    """
    policy:
      - "isolate": 每个噪声单独成族，命名 fam_iso_<稳定hash>
      - "group": 老行为，全部噪声归到 fam_-1（不推荐）
    """
    if policy == "group":
        return "fam_-1"
    can = dup2canon.get(pid, pid)
    ch = content_hash_map.get(pid)
    if isinstance(ch, str) and len(ch) >= 10:
        key = ch[:10]
    else:
        key = hashlib.sha1(str(can).encode("utf-8")).hexdigest()[:10]
    return f"fam_iso_{key}"

from math import ceil
try:
    from cuml.cluster import KMeans as cuKMeans
    _GPU_KMEANS = True
except Exception:
    _GPU_KMEANS = False
from sklearn.cluster import KMeans as skKMeans

def _kmeans_cut(F_sub: np.ndarray, seg_idx: List[int], threshold: int, seed: int, device: str):
    """把一个仍然过大的段用 KMeans 均匀切到 <= threshold."""
    n = len(seg_idx)
    k = max(2, int(ceil(n / max(1, threshold))))
    if device=="gpu" and _GPU_KMEANS:
        km = cuKMeans(n_clusters=k, random_state=seed, max_iter=100)
        lab = km.fit_predict(F_sub)
    else:
        km = skKMeans(n_clusters=k, random_state=seed, n_init=10, max_iter=300)
        lab = km.fit_predict(F_sub)
    out = {}
    for loc, l in enumerate(lab):
        out.setdefault(int(l), []).append(seg_idx[loc])
    return list(out.values())

def _post_split_large_clusters(F: np.ndarray, pids: List[str], labels: np.ndarray,
                               eps: float, min_samples: int, device: str,
                               cfg: Dict, logger: logging.Logger)->Dict[str,str]:
    out_map: Dict[str,str] = {}
    enabled = bool(cfg.get("enabled", False))
    if not enabled: return out_map

    max_fraction = float(cfg.get("max_fraction", 0.10))
    max_size     = int(cfg.get("max_size", 5000))
    split_factor = float(cfg.get("split_factor", 0.85))
    max_rounds   = int(cfg.get("max_rounds", 2))
    min_eps      = float(cfg.get("min_eps", 1e-6))
    fallback     = cfg.get("fallback", {"enabled": True, "method": "kmeans", "seed": 42})
    fb_enabled   = bool(fallback.get("enabled", True))
    fb_seed      = int(fallback.get("seed", 42))

    N = len(pids)
    threshold = max(1, min(int(max_fraction * N), int(max_size)))
    logger.info(f"[S2.3|post_split] hard threshold={threshold} (max_fraction={max_fraction}, max_size={max_size}), "
                f"split_factor={split_factor}, max_rounds={max_rounds}, fallback={'on' if fb_enabled else 'off'}")

    # 只细分初始的非噪声簇
    clusters = {}
    for idx, lab in enumerate(labels):
        if lab < 0:
            continue
        clusters.setdefault(int(lab), []).append(idx)

    for base_lab, idx_list in clusters.items():
        segments = [idx_list]
        cur_eps = float(eps)
        for _ in range(max_rounds):
            progressed = False
            new_segments = []
            for seg in segments:
                if len(seg) <= threshold:
                    new_segments.append(seg)
                    continue
                subF = F[seg]
                sub_labels = _dbscan(subF, eps=max(min_eps, cur_eps*split_factor), 
                                     min_samples=min_samples, device=device)
                uniq = np.unique(sub_labels)
                # 如果 DBSCAN 只有一个标签（含全 -1），直接走 KMeans 兜底
                if len(uniq) <= 1 and fb_enabled:
                    new_segments.extend(_kmeans_cut(subF, seg, threshold, fb_seed, device))
                else:
                    # 正常收集子簇与噪声
                    sub_map = {}
                    for loc, l in enumerate(sub_labels):
                        sub_map.setdefault(int(l), []).append(seg[loc])
                    # 对仍超阈的子段，如果下一轮也可能无效，允许立刻兜底
                    for sub_idx in sub_map.values():
                        if len(sub_idx) > threshold and fb_enabled:
                            new_segments.extend(_kmeans_cut(F[sub_idx], sub_idx, threshold, fb_seed, device))
                        else:
                            new_segments.append(sub_idx)
                progressed = True
            segments = new_segments
            cur_eps = max(min_eps, cur_eps*split_factor)
            if not progressed:
                break

        # 最后一关：凡仍 > 阈值，一律兜底 KMeans，确保硬上限
        really_final = []
        for seg in segments:
            if len(seg) > threshold and fb_enabled:
                really_final.extend(_kmeans_cut(F[seg], seg, threshold, fb_seed, device))
            else:
                really_final.append(seg)

        # 命名
        if len(really_final) == 1 and len(really_final[0]) == len(idx_list):
            fam_name = f"fam_{base_lab}"
            for gidx in really_final[0]:
                out_map[pids[gidx]] = fam_name
        else:
            for si, seg in enumerate(really_final):
                fam_name = f"fam_{base_lab}__s{si}"
                for gidx in seg:
                    out_map[pids[gidx]] = fam_name
    return out_map


def load_cfg(path:str)->Tuple[dict,dict]:
    with open(path,"r",encoding="utf-8") as f: allcfg=json.load(f)
    return allcfg.get("s2_dedup_family_occ",{}), allcfg.get("log",{})

# ---------- main ----------
def main():
    if len(sys.argv)<2 or sys.argv[1] in ("-h","--help"):
        print("用法: python s2_3_family_and_index.py <config.json>"); sys.exit(0)
    cfg_path=sys.argv[1]
    s2cfg, log_cfg = load_cfg(cfg_path)

    out_root = s2cfg.get("out_root")
    device   = s2cfg.get("device","auto")
    Tc       = float(s2cfg.get("family_distance_threshold", 0.012))
    min_samples = int(s2cfg.get("family_min_samples", 8))

    auto_raw = s2cfg.get("family_auto_eps", {"enabled": True, "quantile": 0.10, "scale": 1.02, "knn_k": None, "sample_n": 200000, "seed": 42})
    if isinstance(auto_raw,bool):
        auto_enabled = auto_raw; q=0.10; scale=1.02; sample_n=200000; knn_k=None; seed=42
    else:
        auto_enabled = bool(auto_raw.get("enabled", True))
        q=float(auto_raw.get("quantile",0.10)); scale=float(auto_raw.get("scale",1.02))
        sample_n=auto_raw.get("sample_n",200000); knn_k=auto_raw.get("knn_k",None); seed=int(auto_raw.get("seed",42))

    noise_policy = str(s2cfg.get("family_noise_policy", "isolate")).lower()  # "isolate" / "group"
    post_split_cfg = s2cfg.get("family_post_split", {"enabled": True, "max_fraction": 0.10, "max_size": 5000, "split_factor": 0.85, "max_rounds": 2, "min_eps": 1e-6})

    log_level = s2cfg.get("log_level","info")
    log_file  = s2cfg.get("log_file") or log_cfg.get("file")
    diag_topk = int(s2cfg.get("diagnostics_topk", 10))
    write_hist = bool(s2cfg.get("diagnostics_write_hist", True))
    logger = setup_logger(log_level, log_file)
    logger.info(f"[S2.3] boot config={os.path.abspath(cfg_path)}")

    # inputs
    sig_csv = os.path.join(out_root,"signatures.csv")
    metas_json = os.path.join(out_root,"metas.json")
    dup2canon_json = os.path.join(out_root,"dup2canon.json")
    if not (os.path.isfile(sig_csv) and os.path.isfile(metas_json) and os.path.isfile(dup2canon_json)):
        logger.error("缺少中间产物：请先运行 S2.1 与 S2.2"); sys.exit(1)

    df = pd.read_csv(sig_csv)
    with open(metas_json,"r",encoding="utf-8") as f: metas=json.load(f)
    with open(dup2canon_json,"r",encoding="utf-8") as f: dup2canon=json.load(f)
    content_hash_map = dict(zip(df["part_id"], df["content_hash"]))

    # family clustering
    df_geom = df[(df["has_points"]==1) & df["d2_hist"].notna()]
    fam_map: Dict[str, str] = {}
    eps = math.sqrt(max(1e-12, 2.0*Tc))
    used_device = "cpu"

    if len(df_geom)>0:
        # --- 几何特征打包进度 ---
        print("[build features] start")
        feats = []
        prog_feat = ProgressPrinter(len(df_geom), prefix="[build features] progress")
        prog_feat.print_start()
        for i, (_, r) in enumerate(df_geom.iterrows(), 1):
            feats.append(_pack_feat(r))
            prog_feat.update(i)
        prog_feat.finish()
        F = np.stack(feats).astype(np.float32)
        F = _l2_normalize(F)

        if auto_enabled:
            k_for = int(knn_k) if (knn_k and int(knn_k)>0) else max(1, min_samples)
            eps_try = _auto_eps_from_knn(F, k=k_for, quantile=q, scale=scale,
                                         sample_n=(int(sample_n) if sample_n else None),
                                         seed=int(seed), device=device, logger=logger)
            if np.isfinite(eps_try) and eps_try>0.0:
                eps = eps_try
            else:
                logger.warning(f"[S2.3|auto_eps] invalid eps; fallback eps={eps:.6f} (Tc≈{(eps*eps)/2:.6f})")

        labels = _dbscan(F, eps=eps, min_samples=min_samples, device=device)
        used_device = "gpu" if (device=="gpu" and _GPU_CUML and cuDBSCAN is not None) else "cpu"

        # 可选大簇细分（带进度）
        refined = _post_split_large_clusters(F, df_geom["part_id"].tolist(), labels, eps, min_samples, device, post_split_cfg, logger)

        # --- family 赋值进度（有几何部分） ---
        print("[assign families] start (geom)")
        pids_geom = df_geom["part_id"].tolist()
        prog_assign = ProgressPrinter(len(pids_geom), prefix="[assign families] progress")
        prog_assign.print_start()
        for i, (pid, lab) in enumerate(zip(pids_geom, labels), 1):
            if pid in refined:
                fam_map[pid] = refined[pid]
            elif lab >= 0:
                fam_map[pid] = f"fam_{int(lab)}"
            else:
                fam_map[pid] = _apply_noise_policy(pid, dup2canon, content_hash_map, noise_policy)
            prog_assign.update(i)
        prog_assign.finish()
        logger.info(f"[S2.3] with_geom={len(df_geom)}, eps={eps:.6f}, device={used_device}")
    else:
        logger.warning("[S2.3] 没有可用于聚类的几何样本（has_points==1 且 d2_hist 非空）")

    # --- 无几何样本补齐（进度） ---
    print("[fill no-geom families] start")
    remaining = [pid for pid in df["part_id"].tolist() if pid not in fam_map]
    prog_fill = ProgressPrinter(len(remaining), prefix="[fill no-geom] progress")
    prog_fill.print_start()
    for i, pid in enumerate(remaining, 1):
        fam_map[pid] = f"fam_ch_{str(df.loc[df['part_id']==pid, 'content_hash'].values[0])[:8]}"
        prog_fill.update(i)
    prog_fill.finish()

    # --- 写 part_index.csv（进度） ---
    print("[write part_index.csv] start")
    rows=[]; total_rows=len(df)
    prog_rows = ProgressPrinter(total_rows, prefix="[write part_index] progress")
    prog_rows.print_start()
    for i, (_,r) in enumerate(df.iterrows(), 1):
        pid = r["part_id"]; meta = metas.get(pid, {})
        created_hdr = meta.get("created_at_header"); file_mtime = meta.get("file_mtime_utc")
        ts_src = meta.get("timestamp_source") or ("header" if created_hdr else ("mtime" if file_mtime else "none"))
        unit = meta.get("unit"); unit_src = meta.get("unit_source") or ("unknown" if unit is None else "inferred_step_data")
        rows.append({
            "part_id": pid,
            "rel_path": r["rel_path"],
            "source_dataset": r["source_dataset"],
            "repo": meta.get("repo"),
            "format": meta.get("format"),
            "file_ext": meta.get("file_ext"),
            "file_size_bytes": meta.get("file_size_bytes"),
            "file_mtime_utc": file_mtime,
            "timestamp_source": ts_src,
            "step_schema": meta.get("step_schema"),
            "originating_system": meta.get("originating_system"),
            "author": meta.get("author"),
            "created_at_header": created_hdr,
            "kernel": meta.get("kernel"),
            "unit": unit,
            "unit_source": unit_src,
            "domain_hint": meta.get("domain_hint"),
            "content_hash": r["content_hash"],
            "geom_hash": r["geom_hash"] if isinstance(r["geom_hash"], str) else "",
            "duplicate_canonical": dup2canon.get(pid, pid),
            "family_id": fam_map[pid],
            "has_points": int(r["has_points"]),
        })
        prog_rows.update(i)
    prog_rows.finish()

    out_root_abs = os.path.abspath(out_root)
    os.makedirs(out_root_abs, exist_ok=True)
    part_index_csv = os.path.join(out_root_abs,"part_index.csv")
    pd.DataFrame(rows).to_csv(part_index_csv, index=False)
    print(f"[write part_index.csv] done -> {part_index_csv}")

    # family 直方图与诊断
    fam_series = pd.Series([fam_map[pid] for pid in df["part_id"].tolist()])
    fam_counts = fam_series.value_counts()
    N_all = int(len(df))
    topk = fam_counts.head(diag_topk)
    noise_like = fam_counts[[k for k in fam_counts.index if k.startswith("fam_iso_")]].sum() if any(k.startswith("fam_iso_") for k in fam_counts.index) else 0
    frac_noise = float(noise_like)/max(1,N_all)

    if write_hist:
        hist_df = pd.DataFrame({"family_id": fam_counts.index, "count": fam_counts.values})
        hist_df["pct"] = hist_df["count"] / max(1,N_all)
        hist_csv = os.path.join(out_root_abs, "family_hist.csv")
        hist_df.to_csv(hist_csv, index=False)
        print(f"[family_hist] done -> {hist_csv}")

    logger.info("[S2.3] Top-%d families:\n%s", diag_topk, "\n".join([f"  {idx:>3}. {k:20s}  {v:8d}  ({v/max(1,N_all):.4f})"
                                                                   for idx,(k,v) in enumerate(topk.items(),1)]))
    # summary.json（与原总汇一致字段 + 诊断）
    failures_txt = os.path.join(out_root_abs,"failures.txt")
    num_failed = 0
    if os.path.isfile(failures_txt):
        with open(failures_txt,"r",encoding="utf-8") as f: num_failed = sum(1 for _ in f if _)
    with open(os.path.join(out_root_abs,"duplicate_groups.json"),"r",encoding="utf-8") as f:
        groups=json.load(f)
    summary = {
        "num_parts": int(len(df)),
        "num_failed": int(num_failed),
        "num_duplicate_groups": int(sum(1 for g in groups.values() if len(g)>1)),
        "num_families": int(len(set(fam_map.values()))),
        "device": device,
        "gpu_available": {"faiss": _GPU_FAISS, "cuml": _GPU_CUML},
        "used_device": "gpu" if (device=="gpu" and _GPU_CUML and cuDBSCAN is not None) else "cpu",
        "clustering": {
            "eps": float(eps),
            "min_samples": int(min_samples),
            "auto_eps": {"enabled": bool(auto_enabled), "quantile": float(q), "scale": float(scale),
                         "knn_k": (int(knn_k) if knn_k else None), "sample_n": int(sample_n) if sample_n else None,
                         "seed": int(seed)},
            "Tc_fallback": float(Tc),
            "noise_policy": noise_policy,
            "post_split": post_split_cfg,
        },
        "diagnostics": {
            "noise_like_count": int(noise_like),
            "noise_like_fraction": float(frac_noise),
            "largest_family": {"id": str(fam_counts.index[0]), "count": int(fam_counts.iloc[0]),
                               "fraction": float(fam_counts.iloc[0]/max(1,N_all))} if len(fam_counts)>0 else {},
            "topk": [{"family_id": str(k), "count": int(v), "pct": float(v/max(1,N_all))} for k,v in topk.items()]
        },
        "out_files": {
            "signatures": os.path.join(out_root_abs,"signatures.csv"),
            "part_index": part_index_csv,
            "duplicate_groups": os.path.join(out_root_abs,"duplicate_groups.json"),
            "family_hist": os.path.join(out_root_abs,"family_hist.csv") if write_hist else None
        }
    }
    summary_json = os.path.join(out_root_abs,"summary.json")
    with open(summary_json,"w",encoding="utf-8") as f: json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[summary] done -> {summary_json}")

    # 附带单独诊断文件
    diag_json = os.path.join(out_root_abs,"family_diagnostics.json")
    with open(diag_json,"w",encoding="utf-8") as f: json.dump(summary.get("diagnostics", {}), f, indent=2, ensure_ascii=False)
    print(f"[diagnostics] done -> {diag_json}")

    logger.info("[S2.3] DONE")
    print("[S2.3] DONE")

if __name__=="__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S2.2: 去重（内容哈希 + 几何近邻），输出 duplicate_groups.json 与 dup2canon.json

变更点（支持“块级归一 + 加权 + 全局归一”以避免 D2 高维主导）：
- 新增 feature_norm 配置（位于 s2_dedup_family_occ 下）：
  "feature_norm": {
    "block_l2": true,          # 先对每个 block（D2 / bbox / SurfHist / DihedralHist）各自 L2
    "global_l2": true,         # 拼接后整体 L2（建议 true，以便内积=余弦）
    "block_weight": "equal"    # "equal" 或 "dim_sqrt"（以 1/sqrt(dim_block) 加权）
  }
- kNN 相似度依旧基于余弦（FAISS 内积 + 全局 L2 后等价于 cosine）

保持原逻辑：
- 先按 content_hash 并查集；
- 对有几何者做近邻合并（余弦相似>=d2_sim_threshold 且 bbox_ratio 在容差内）；
- FAISS-GPU 优先，sklearn 回退。
"""

import os, sys, json, time, logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# ---------- logging + progress ----------
def _to_level(level: str):
    import logging as _lg
    return {"debug": _lg.DEBUG, "info": _lg.INFO, "warning": _lg.WARNING}.get((level or "info").lower(), _lg.INFO)

def setup_logger(level: str="info", log_file: Optional[str]=None)->logging.Logger:
    logger = logging.getLogger("S2_2"); logger.setLevel(_to_level(level)); logger.propagate=False
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s","%Y-%m-%d %H:%M:%S")
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        sh=logging.StreamHandler(); sh.setFormatter(fmt); logger.addHandler(sh)
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            fh=logging.FileHandler(log_file, encoding="utf-8"); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger

class ProgressPrinter:
    def __init__(self,total:int,prefix:str=""): self.total=max(1,int(total)); self.prefix=prefix; self.last=-1; self.done=False
    def _emit(self,p): print(f"\r{self.prefix} {p:3d}%", end="", flush=True)
    def print_start(self): self._emit(0); self.last=0
    def update(self,cur:int):
        if self.done: return
        p = int(min(max(0,cur), self.total)*100/self.total)
        if p>=self.last+1: self._emit(p); self.last=p
        if cur>=self.total: self.finish()
    def finish(self):
        if not self.done: self._emit(100); print(); self.done=True

# ---------- GPU FAISS ----------
faiss=None; _GPU_FAISS=False
def _ensure_faiss():
    global faiss, _GPU_FAISS
    if faiss is not None: return
    try:
        import faiss as _faiss
        faiss=_faiss
        try: _ = _faiss.StandardGpuResources(); _GPU_FAISS=True
        except Exception: _GPU_FAISS=False
    except Exception:
        faiss=None; _GPU_FAISS=False

# ---------- sklearn fallback ----------
from sklearn.neighbors import NearestNeighbors

# ---------- DSU ----------
class DSU:
    def __init__(self, items: List[str]): self.p={x:x for x in items}; self.r={x:0 for x in items}
    def find(self,x:str)->str:
        while self.p[x]!=x:
            self.p[x]=self.p[self.p[x]]; x=self.p[x]
        return x
    def union(self,a:str,b:str):
        ra,rb=self.find(a),self.find(b)
        if ra==rb: return
        if self.r[ra]<self.r[rb]: self.p[ra]=rb
        elif self.r[ra]>self.r[rb]: self.p[rb]=ra
        else: self.p[rb]=ra; self.r[ra]+=1

# ---------- feature blocks & normalization ----------
_BLOCK_ORDER = [
    ("d2_hist", "d2"),
    ("bbox_ratio", "bbox"),
    ("surf_hist", "surf"),
    ("dih_hist", "dih"),
]

def _infer_block_dims(df_geom: pd.DataFrame, logger: logging.Logger)->Dict[str,int]:
    dims = {name:0 for _, name in _BLOCK_ORDER}
    for col, name in _BLOCK_ORDER:
        # 找到第一条非空记录以推断维度
        s = df_geom[col].dropna()
        if len(s)==0:
            dims[name]=0
            continue
        try:
            arr = np.array(json.loads(s.iloc[0]), dtype=np.float32)
            dims[name]=int(arr.size)
        except Exception:
            logger.warning(f"[S2.2] 解析首条 {col} 失败，将忽略该 block")
            dims[name]=0
    keep = {k:v for k,v in dims.items() if v>0}
    if len(keep)==0:
        logger.error("[S2.2] 无任何有效特征块（d2/bbox/surf/dih 全为缺失或解析失败）")
        raise ValueError("no valid feature blocks")
    return dims

def _pack_feat_blocks(row: pd.Series, dims: Dict[str,int])->List[np.ndarray]:
    blocks=[]
    for col, name in _BLOCK_ORDER:
        d = dims.get(name, 0)
        if d<=0: 
            continue
        if pd.notna(row[col]):
            try:
                b = np.array(json.loads(row[col]), dtype=np.float32)
                # 容错：若长度与预期不符则截断/补零
                if b.size != d:
                    if b.size > d:
                        b = b[:d]
                    else:
                        pad = np.zeros(d, dtype=np.float32); pad[:b.size] = b; b = pad
            except Exception:
                b = np.zeros(d, dtype=np.float32)
        else:
            b = np.zeros(d, dtype=np.float32)
        blocks.append(b)
    return blocks

def _normalize_blocks(blocks: List[np.ndarray], cfg_norm: Dict)->np.ndarray:
    """block-level L2 -> optional block-weight -> concat -> optional global L2"""
    if len(blocks)==0:
        return np.zeros((0,), dtype=np.float32)

    out=[]
    # 1) block L2
    if cfg_norm.get("block_l2", True):
        for b in blocks:
            nb = b / (np.linalg.norm(b) + 1e-12)
            out.append(nb.astype(np.float32, copy=False))
    else:
        out = [b.astype(np.float32, copy=False) for b in blocks]

    # 2) block weighting
    mode = (cfg_norm.get("block_weight") or "equal").lower()
    if mode == "dim_sqrt":
        weighted=[]
        for b in out:
            d = float(b.size)
            w = 1.0/np.sqrt(max(1.0, d))
            weighted.append(b * w)
        out = weighted
    else:
        # "equal" 权重：不作额外缩放
        pass

    # 3) concat
    x = np.concatenate(out).astype(np.float32, copy=False)

    # 4) global L2（确保 FAISS InnerProduct 等价于 cosine）
    if cfg_norm.get("global_l2", True):
        x = x / (np.linalg.norm(x) + 1e-12)

    return x

# ---------- config ----------
def load_config(path:str)->Tuple[dict,dict]:
    with open(path,"r",encoding="utf-8") as f: allcfg=json.load(f)
    return allcfg.get("s2_dedup_family_occ",{}), allcfg.get("log",{})

# ---------- main ----------
def main():
    if len(sys.argv)<2 or sys.argv[1] in ("-h","--help"):
        print("用法: python s2_2_dedup.py <config.json>"); sys.exit(0)
    cfg_path=sys.argv[1]
    s2cfg, log_cfg = load_config(cfg_path)
    out_root  = s2cfg.get("out_root")
    device    = s2cfg.get("device","auto")
    d2_sim_threshold = float(s2cfg.get("d2_sim_threshold",0.995))
    bbox_tol  = float(s2cfg.get("bbox_tol",0.02))
    feat_norm_cfg = s2cfg.get("feature_norm", {"block_l2": True, "global_l2": True, "block_weight": "equal"})
    log_level = s2cfg.get("log_level","info")
    log_file  = s2cfg.get("log_file") or (log_cfg.get("file"))
    sig_csv   = os.path.join(out_root,"signatures.csv")
    logger = setup_logger(log_level, log_file)

    logger.info(f"[S2.2] boot config={os.path.abspath(cfg_path)}")
    logger.info(f"[S2.2] inputs: signatures.csv={sig_csv}")
    logger.info(f"[S2.2] feature_norm={feat_norm_cfg}")
    if not os.path.isfile(sig_csv):
        logger.error("缺少 signatures.csv，请先运行 S2.1"); sys.exit(1)

    df = pd.read_csv(sig_csv)
    ids = df["part_id"].tolist()
    dsu = DSU(ids)

    # a) 内容哈希去重（完全一致）
    by_ch={}
    for _,r in df.iterrows():
        by_ch.setdefault(r["content_hash"], []).append(r["part_id"])
    for ch, lst in by_ch.items():
        if len(lst)>1:
            canon = lst[0]
            for x in lst[1:]: dsu.union(canon, x)
    logger.info(f"[S2.2] content-hash groups merged={sum(1 for v in by_ch.values() if len(v)>1)}")

    # b) 几何近邻去重（有几何者）
    df_geom = df[(df["has_points"]==1) & df["d2_hist"].notna()]
    merged_pairs=0
    if len(df_geom)>0:
        # 推断各 block 维度（用于补齐缺失块为 0 向量，以保证拼接维度一致）
        dims = _infer_block_dims(df_geom, logger)
        logger.info(f"[S2.2] block dims: {dims}")

        # 构建归一化后的特征矩阵
        feats=[]
        for _, r in df_geom.iterrows():
            blocks = _pack_feat_blocks(r, dims)
            x = _normalize_blocks(blocks, feat_norm_cfg)
            feats.append(x)
        F = np.stack(feats).astype(np.float32, copy=False)

        # 若未做 global L2，则此处强制做一次，以确保内积 = 余弦
        if not feat_norm_cfg.get("global_l2", True):
            norms = np.linalg.norm(F, axis=1, keepdims=True) + 1e-12
            F = F / norms

        pids = df_geom["part_id"].tolist()

        use_gpu = (device=="gpu")
        prog = ProgressPrinter(len(pids), prefix="[S2.2]")
        prog.print_start()
        if use_gpu:
            _ensure_faiss()
        if use_gpu and _GPU_FAISS and faiss is not None:
            res = faiss.StandardGpuResources()
            index = faiss.GpuIndexFlatIP(res, F.shape[1]); index.add(F)
            k = min(10, F.shape[0]); sims, idxs = index.search(F, k)
            for i, pid_i in enumerate(pids):
                # 自身也会出现在近邻里，跳过
                for sim, jpos in zip(sims[i], idxs[i]):
                    if jpos==i or jpos<0: continue
                    pid_j = pids[jpos]
                    # bbox 容差（直接从csv读取）
                    br_i = np.array(json.loads(df_geom.iloc[i]["bbox_ratio"]), dtype=np.float32)
                    br_j = np.array(json.loads(df_geom.iloc[jpos]["bbox_ratio"]), dtype=np.float32)
                    br_ok = float(np.max(np.abs(br_i - br_j))) <= bbox_tol
                    if float(sim) >= d2_sim_threshold and br_ok:
                        if dsu.find(pid_i)!=dsu.find(pid_j):
                            dsu.union(pid_i, pid_j); merged_pairs += 1
                prog.update(i+1)
        else:
            nbrs = NearestNeighbors(metric="cosine", n_neighbors=min(10, len(F))).fit(F)
            distances, indices = nbrs.kneighbors(F, return_distance=True)
            for i, pid_i in enumerate(pids):
                for d, jpos in zip(distances[i], indices[i]):
                    if jpos==i: continue
                    pid_j = pids[jpos]
                    sim = 1.0 - float(d)
                    br_i = np.array(json.loads(df_geom.iloc[i]["bbox_ratio"]), dtype=np.float32)
                    br_j = np.array(json.loads(df_geom.iloc[jpos]["bbox_ratio"]), dtype=np.float32)
                    br_ok = float(np.max(np.abs(br_i - br_j))) <= bbox_tol
                    if sim >= d2_sim_threshold and br_ok:
                        if dsu.find(pid_i)!=dsu.find(pid_j):
                            dsu.union(pid_i, pid_j); merged_pairs += 1
                prog.update(i+1)
        prog.finish()
    else:
        logger.warning("[S2.2] 没有可用于几何近邻去重的样本（has_points==1 且 d2_hist 非空）")
    logger.info(f"[S2.2] geom-merged-pairs={merged_pairs}")

    # 汇总输出
    dup2canon={}; groups={}
    for pid in ids:
        root = dsu.find(pid); dup2canon[pid]=root; groups.setdefault(root,[]).append(pid)

    dup_groups_json = os.path.join(out_root,"duplicate_groups.json")
    with open(dup_groups_json,"w",encoding="utf-8") as f: json.dump(groups, f, indent=2, ensure_ascii=False)
    logger.info(f"[S2.2] Output duplicate_groups.json -> {dup_groups_json}")

    dup2canon_json = os.path.join(out_root,"dup2canon.json")
    with open(dup2canon_json,"w",encoding="utf-8") as f: json.dump(dup2canon, f, indent=2, ensure_ascii=False)
    logger.info(f"[S2.2] Output dup2canon.json -> {dup2canon_json}")

    summary = {
        "num_parts": len(ids),
        "duplicate_groups": sum(1 for v in groups.values() if len(v)>1),
        "geom_merged_pairs": merged_pairs,
        "gpu_available": {"faiss": _GPU_FAISS},
        "feature_norm": feat_norm_cfg,
        "out_files": {"duplicate_groups": dup_groups_json, "dup2canon": dup2canon_json}
    }
    out_sum = os.path.join(out_root,"dedup_summary.json")
    with open(out_sum,"w",encoding="utf-8") as f: json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"[S2.2] Output dedup_summary.json -> {out_sum}")
    logger.info("[S2.2] DONE")

if __name__=="__main__":
    main()

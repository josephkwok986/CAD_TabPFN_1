#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S2.3.1: 修正 family_id（按 duplicate_canonical 主族），并输出新的 part_index 与 family_hist

改动点：
1) 配置改为从 <config.json> 读取，路径：s2_dedup_family_occ.s2_fix_family
2) 支持以下配置字段：
   {
     "s2_dedup_family_occ": {
       "s2_fix_family": {
         "in_part_index": "/workspace/Gjj Local/data/CAD/step_out/s2_out/part_index.csv",
         "out_part_index": "/workspace/Gjj Local/data/CAD/step_out/s2_out/part_index.fix1.csv",
         "out_family_hist": "/workspace/Gjj Local/data/CAD/step_out/s2_out/family_hist.fix1.csv",
         "log_level": "info",
         "log_file": null
       }
     }
   }
用法：
  python s2_3_1_fix_family.py <config.json>
"""

import os
import sys
import json
import logging
import pandas as pd

# ---------- logging ----------
def _to_level(level: str):
    import logging as _lg
    return {"debug": _lg.DEBUG, "info": _lg.INFO, "warning": _lg.WARNING}.get((level or "info").lower(), _lg.INFO)

def setup_logger(level: str = "info", log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger("S2_3_1_FixFamily")
    logger.setLevel(_to_level(level))
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger

# ---------- config ----------
def load_fix_cfg(cfg_path: str) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        allcfg = json.load(f)
    s2_cfg = allcfg.get("s2_dedup_family_occ", {})
    fix = s2_cfg.get("s2_fix_family", {})
    # 合法性校验 & 默认值
    required = ["in_part_index", "out_part_index", "out_family_hist"]
    missing = [k for k in required if not fix.get(k)]
    if missing:
        raise ValueError(f"配置缺少必填项（s2_dedup_family_occ.s2_fix_family）：{missing}")
    fix.setdefault("log_level", "info")
    fix.setdefault("log_file", None)
    return fix

# ---------- core logic ----------
def canonical_major_family_map(df: pd.DataFrame) -> dict:
    """对每个 duplicate_canonical 取出现次数最多的 family_id（并列取字典序最小）"""
    m = {}
    grp = df.groupby("duplicate_canonical")
    for k, g in grp:
        fam_counts = g["family_id"].value_counts()
        if len(fam_counts) == 0:
            continue
        top_cnt = fam_counts.iloc[0]
        pick = sorted(fam_counts[fam_counts == top_cnt].index.tolist())[0]
        m[k] = pick
    return m

def apply_fix(df: pd.DataFrame, cmap: dict) -> pd.DataFrame:
    """仅修正那些同一 duplicate_canonical 下 family_id 不唯一的行"""
    df2 = df.copy()
    multi = df.groupby("duplicate_canonical")["family_id"].nunique()
    cands = set(multi[multi > 1].index.tolist())
    mask = df2["duplicate_canonical"].isin(cands)
    df2.loc[mask, "family_id"] = df2.loc[mask, "duplicate_canonical"].map(cmap)
    return df2

def report(df: pd.DataFrame, title: str, logger: logging.Logger):
    logger.info(f"=== {title} ===")
    n = len(df)
    iso_ratio = df["family_id"].astype(str).str.startswith("fam_iso_").mean()
    logger.info(f"rows={n}, iso_ratio={iso_ratio:.4f}")
    if "duplicate_canonical" in df.columns:
        g = df.groupby("duplicate_canonical")["family_id"].nunique()
        bad = int((g > 1).sum())
        logger.info(f"canonical with >1 families: {bad} / {g.size} ({bad / max(1, g.size):.4f})")
    top = df["family_id"].value_counts().head(10)
    logger.info("Top-10 families:\n%s", top.to_string())

def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("用法: python s2_3_1_fix_family.py <config.json>")
        sys.exit(0)

    cfg_path = sys.argv[1]
    fix_cfg = load_fix_cfg(cfg_path)

    logger = setup_logger(fix_cfg.get("log_level", "info"), fix_cfg.get("log_file"))
    logger.info("[S2.3.1] boot config=%s", os.path.abspath(cfg_path))

    in_pi = fix_cfg["in_part_index"]
    out_pi = fix_cfg["out_part_index"]
    out_hist = fix_cfg["out_family_hist"]

    assert os.path.isfile(in_pi), f"not found: {in_pi}"
    os.makedirs(os.path.dirname(os.path.abspath(out_pi)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_hist)), exist_ok=True)

    df = pd.read_csv(in_pi)
    # 基础列检查
    need_cols = {"duplicate_canonical", "family_id"}
    missing_cols = need_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"输入文件缺少必要列: {missing_cols}")

    report(df, "before", logger)

    cmap = canonical_major_family_map(df)
    df_fix = apply_fix(df, cmap)

    report(df_fix, "after", logger)

    df_fix.to_csv(out_pi, index=False)
    hist = df_fix["family_id"].value_counts().rename_axis("family_id").reset_index(name="count")
    hist["pct"] = hist["count"] / len(df_fix)
    hist.to_csv(out_hist, index=False)

    logger.info("[write] %s", out_pi)
    logger.info("[write] %s", out_hist)
    logger.info("[S2.3.1] DONE")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S2.3.2: 生成 for_split 用的 part_index（新增列 family_major）

配置读取路径：
  s2_dedup_family_occ.s2_fix_family

可用字段（均在 s2_dedup_family_occ.s2_fix_family 下）：
  - in_part_index_raw         可选；默认回退到 in_part_index
  - in_part_index_fixed       可选；默认回退到 out_part_index
  - out_part_index_for_split  可选；默认放到 out_part_index 同目录/part_index.for_split.csv
  - log_level                 可选；默认 "info"
  - log_file                  可选；默认 None

用法：
  python s2_3_2_for_split.py <config.json>
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
    logger = logging.getLogger("S2_3_2_ForSplit")
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
def load_cfg(cfg_path: str) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        allcfg = json.load(f)
    s2 = allcfg.get("s2_dedup_family_occ", {})
    fix = s2.get("s2_fix_family", {})

    # 兼容：允许复用 S2.3.1 的 in_part_index / out_part_index
    in_raw = fix.get("in_part_index_raw") or fix.get("in_part_index")
    in_fix = fix.get("in_part_index_fixed") or fix.get("out_part_index")
    out_split = fix.get("out_part_index_for_split")
    if not out_split and in_fix:
        out_split = os.path.join(os.path.dirname(os.path.abspath(in_fix)), "part_index.for_split.csv")

    missing = [k for k, v in {
        "in_part_index_raw": in_raw,
        "in_part_index_fixed": in_fix,
        "out_part_index_for_split": out_split
    }.items() if not v]
    if missing:
        raise ValueError(f"配置缺少必填项（s2_dedup_family_occ.s2_fix_family）：{missing}")

    return {
        "in_part_index_raw": in_raw,
        "in_part_index_fixed": in_fix,
        "out_part_index_for_split": out_split,
        "log_level": fix.get("log_level", "info"),
        "log_file": fix.get("log_file"),
    }

# ---------- core ----------
def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("用法: python s2_3_2_for_split.py <config.json>")
        sys.exit(0)

    cfg_path = sys.argv[1]
    cfg = load_cfg(cfg_path)
    logger = setup_logger(cfg["log_level"], cfg["log_file"])
    logger.info("[S2.3.2] boot config=%s", os.path.abspath(cfg_path))

    RAW = cfg["in_part_index_raw"]
    FIX = cfg["in_part_index_fixed"]
    OUT = cfg["out_part_index_for_split"]

    assert os.path.isfile(RAW), f"not found: {RAW}"
    assert os.path.isfile(FIX), f"not found: {FIX}"
    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)

    df_raw = pd.read_csv(RAW)
    df_fix = pd.read_csv(FIX)

    # 基础列检查
    need_cols = {"duplicate_canonical", "family_id"}
    miss_raw = need_cols - set(df_raw.columns)
    miss_fix = need_cols - set(df_fix.columns)
    if miss_raw:
        raise ValueError(f"原始输入缺少必要列: {miss_raw}")
    if miss_fix:
        raise ValueError(f"覆盖输入缺少必要列: {miss_fix}")

    # 从覆盖版提取每个 canonical 的主族（覆盖版里 family_id 已被主族替换）
    can2major = (
        df_fix.groupby("duplicate_canonical")["family_id"]
        .agg(lambda s: s.value_counts().index[0])  # 与你的原逻辑保持一致
        .to_dict()
    )

    df_out = df_raw.copy()
    df_out["family_major"] = df_out["duplicate_canonical"].map(can2major)
    df_out["family_major"] = df_out["family_major"].fillna(df_out["family_id"])

    # 简要对比
    iso_raw = df_out["family_id"].astype(str).str.startswith("fam_iso_", na=False).mean()
    iso_major = df_out["family_major"].astype(str).str.startswith("fam_iso_", na=False).mean()
    logger.info("iso_ratio(raw family)   = %.4f", iso_raw)
    logger.info("iso_ratio(family_major) = %.4f", iso_major)

    df_out.to_csv(OUT, index=False)
    logger.info("[write] %s", OUT)
    logger.info("[S2.3.2] DONE")

if __name__ == "__main__":
    main()

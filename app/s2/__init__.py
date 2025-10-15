#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S2 阶段的统一入口。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_MODULE_DIR = Path(__file__).resolve().parent


def _load_module(alias: str, filename: str):
    spec = importlib.util.spec_from_file_location(alias, _MODULE_DIR / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块 {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


_sig_module = _load_module("s2_1_signature_pipeline", "s2.1_signature_pipeline.py")
_dedup_module = _load_module("s2_2_dedup_pipeline", "s2.2_dedup_pipeline.py")

SignatureExtractionRunner = _sig_module.SignatureExtractionRunner
DeduplicationRunner = _dedup_module.DeduplicationRunner

__all__ = ["SignatureExtractionRunner", "DeduplicationRunner"]

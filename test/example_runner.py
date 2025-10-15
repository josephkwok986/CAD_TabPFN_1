#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""示例任务：改为继承 base_components.main_framework.TaskFramework。"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# 统一从包入口导入
from base_components import (
    TaskPartitioner,
    PartitionConstraints,
    PartitionStrategy,
    TaskResult,
)
from base_components.main_framework import TaskFramework
from base_components.task_partitioner import TaskRecord
from base_components.task_pool import LeasedTask
from base_components.progress import ProgressController
from base_components.logger import StructuredLogger


# ------------------------- 小工具 -------------------------

def _load_items(raw_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """解析示例数据，返回 (planner_items, items_by_ref)。"""
    with raw_path.open("r", encoding="utf-8") as handle:
        raw_items: Sequence[Dict[str, Any]] = json.load(handle)

    planner_items: List[Dict[str, Any]] = []
    items_by_ref: Dict[str, Dict[str, Any]] = {}

    for entry in raw_items:
        payload_ref = entry["part_id"]
        planner_items.append(
            {
                "payload_ref": payload_ref,
                "weight": float(entry["weight"]),
                "metadata": {
                    "group": entry["group"],
                    "category": entry["category"],
                    "description": entry["description"],
                },
            }
        )
        items_by_ref[payload_ref] = dict(entry)

    return planner_items, items_by_ref


def _expand_items(
    planner_items: List[Dict[str, Any]],
    items_by_ref: Dict[str, Dict[str, Any]],
    target_items: int,
) -> None:
    """复制样例条目，扩充到至少 target_items 条。"""
    if len(planner_items) >= target_items:
        return
    base_items = list(planner_items)
    if not base_items:
        raise ValueError("示例数据为空，无法扩充任务")
    clone_id = 0
    while len(planner_items) < target_items:
        src = base_items[clone_id % len(base_items)]
        src_ref = src["payload_ref"]
        clone_id += 1
        new_ref = f"{src_ref}_dup{clone_id}"
        while new_ref in items_by_ref:
            clone_id += 1
            new_ref = f"{src_ref}_dup{clone_id}"
        meta = dict(src["metadata"])
        meta["duplicate_of"] = src_ref
        planner_items.append({"payload_ref": new_ref, "weight": src["weight"], "metadata": meta})
        full = dict(items_by_ref[src_ref])
        full["part_id"] = new_ref
        full["duplicate_of"] = src_ref
        items_by_ref[new_ref] = full


def _ensure_output_root(base_dir: Path) -> Path:
    root = base_dir / "demo_runtime"
    (root / "cache").mkdir(parents=True, exist_ok=True)
    return root


def _aggregate_results(output_root: Path) -> Path:
    cache_dir = output_root / "cache"
    summary: Dict[str, Any] = {"tasks": [], "total_items": 0, "total_weight": 0.0}

    for csv_path in sorted(cache_dir.glob("*.csv")):
        with csv_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        if not rows:
            continue
        task_id = rows[0]["task_id"]
        task_weight = sum(float(row["weight"]) for row in rows)
        summary["tasks"].append(
            {"task_id": task_id, "row_count": len(rows), "weight": task_weight, "file": str(csv_path)}
        )
        summary["total_items"] += len(rows)
        summary["total_weight"] += task_weight

    out = output_root / "summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


# ------------------------- 任务框架实现 -------------------------

@dataclass
class _HandlerContext:
    items_map: Dict[str, Dict[str, Any]]
    output_root: Path
    progress: Any  # ProgressProxy


class ExampleFrameworkJob(TaskFramework):
    """把示例程序改写为 TaskFramework 子类。"""

    def __init__(self) -> None:
        super().__init__()
        self._base_dir = Path(__file__).resolve().parent
        self._output_root = _ensure_output_root(self._base_dir)
        self._planner_items: List[Dict[str, Any]] = []
        self._items_map: Dict[str, Dict[str, Any]] = {}
        self._job_id = "demo-job"
        self._pc: Optional[ProgressController] = None
        self._log = StructuredLogger.get_logger("demo.framework_job")

    # ------- 生命周期钩子 -------

    def before_run(self, cfg) -> None:
        # 读取样例并按需扩充
        planner, by_ref = _load_items(self._base_dir / "sample_items.json")
        desired_tasks = int(cfg.get("example.desired_tasks", int, 12))
        chunk_size = int(cfg.get("example.chunk_size", int, 1))
        target_items = max(desired_tasks * max(1, chunk_size), len(planner))
        _expand_items(planner, by_ref, target_items)

        self._planner_items = planner
        self._items_map = by_ref
        self._log.info(
            "example.loaded",
            items=len(self._planner_items),
            desired_tasks=desired_tasks,
            chunk_size=chunk_size,
        )

    def build_handler_context(self, cfg) -> _HandlerContext:
        total = len(self._planner_items) if self._planner_items else None
        self._pc = ProgressController(total_units=total, description="处理示例零件", unit_name="row")
        self._pc.start()
        return _HandlerContext(
            items_map=self._items_map,
            output_root=self._output_root,
            progress=self._pc.make_proxy(),
        )

    def after_run(self, cfg) -> None:
        if self._pc is not None:
            self._pc.close()
            self._pc = None
        out = _aggregate_results(self._output_root)
        self._log.info("example.summary_written", path=str(out))

    # ------- 任务生产与执行 -------

    def produce_tasks(self) -> Iterable[TaskRecord]:
        # 约束与策略从配置读取，缺省与原示例一致
        cfg = self._cfg  # 由框架设置
        chunk_size = int(cfg.get("example.chunk_size", int, 1)) if cfg else 1
        strategy = cfg.get("example.strategy", str, PartitionStrategy.FIXED) if cfg else PartitionStrategy.FIXED

        constraints = PartitionConstraints(max_items_per_task=chunk_size)
        job_spec = {"job_id": self._job_id}

        for task in TaskPartitioner.iter_tasks(job_spec, self._planner_items, strategy, constraints):
            if self._pc is not None:
                self._pc.discovered(len(task.payload_ref))
            yield task

    def handle(self, leased: LeasedTask, context: _HandlerContext) -> TaskResult:
        # 与原示例相同：为每个任务写出一份中间 CSV（由执行器持久化）
        items_map = context.items_map
        output_root = context.output_root
        task_items = [items_map[ref] for ref in leased.task.payload_ref]

        rows = [
            {
                "task_id": leased.task.task_id,
                "part_id": item["part_id"],
                "group": item["group"],
                "category": item["category"],
                "weight": item["weight"],
            }
            for item in task_items
        ]
        total_weight = sum(float(it["weight"]) for it in task_items)

        if context.progress is not None:
            context.progress.advance(len(task_items))

        return TaskResult(
            payload=rows,
            processed=len(task_items),
            metadata={"task_id": leased.task.task_id, "total_weight": total_weight},
            output_directory=str(output_root),
            output_filename=f"{leased.task.task_id}",
            is_final_output=False,
        )


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    default_cfg = base_dir / "example_config.yaml"
    os.environ.setdefault("CAD_TASK_CONFIG", str(default_cfg))
    ExampleFrameworkJob().run()

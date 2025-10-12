#!/usr/bin/env python3
"""演示 base_components 核心模块协作流程的完整示例程序。"""

from __future__ import annotations

import csv
import json
import os
import shutil
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from base_components import (
    Config,
    ExecutorPolicy,
    ParallelExecutor,
    ProgressController,
    StructuredLogger,
    TaskPartitioner,
    TaskResult,
    ensure_task_config,
)
# 使示例覆盖文件队列与任务池核心接口
from base_components import (
    PartitionConstraints,
    PartitionStrategy,
    TaskPool,
)
from base_components.gpu_resources import GPUResourceManager
from base_components.parallel_executor import _persist_task_result
from base_components.task_pool import LeasedTask


def load_items(raw_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """读取示例数据并整理出任务规划器所需的结构。"""
    with raw_path.open("r", encoding="utf-8") as handle:
        raw_items: Sequence[Dict[str, Any]] = json.load(handle)  # 原始 JSON 列表，仅用于解析。

    planner_items: List[Dict[str, Any]] = []  # 提供给 TaskPartitioner 的精简视图。
    items_by_ref: Dict[str, Dict[str, Any]] = {}  # 用于执行阶段根据 payload_ref 找回原始记录。

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
        items_by_ref[payload_ref] = dict(entry)  # 缓存完整行，供 worker 解析字段。

    return planner_items, items_by_ref


def _compute_gini(weights: List[float]) -> float:
    filtered = [w for w in weights if w >= 0]
    if not filtered:
        return 0.0
    sorted_vals = sorted(filtered)
    total = sum(sorted_vals)
    n = len(sorted_vals)
    if total == 0 or n == 0:
        return 0.0
    cum = 0.0
    for idx, value in enumerate(sorted_vals, 1):
        cum += idx * value
    return (2 * cum) / (n * total) - (n + 1) / n


def ensure_output_root(base_dir: Path) -> Path:
    """创建输出目录及缓存子目录，保持示例的文件布局。"""
    output_root = base_dir / "output"  # 所有输出结果的根路径。
    cache_dir = output_root / "cache"  # worker 写入中间 CSV 的约定位置。
    queue_dir = output_root / "queue"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if queue_dir.exists():
        shutil.rmtree(queue_dir)
    queue_dir.mkdir(parents=True, exist_ok=True)
    return output_root


def bootstrap_config(base_dir: Path) -> Config:
    """加载示例配置并确保 Config 单例已就绪。"""
    config_path = (base_dir / "example_config.yaml").resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"示例配置缺失: {config_path}")
    os.environ["CAD_TASK_CONFIG"] = str(config_path)
    Config.set_singleton(None)
    return Config.load_singleton(config_path)


def task_handler(leased: LeasedTask, context: Dict[str, Any]) -> TaskResult:
    """worker 进程根据租赁到的任务生成摘要并立即落盘。"""
    items_map: Dict[str, Dict[str, Any]] = context["items_map"]  # payload_ref -> 原始数据。
    output_root: Path = context["output_root"]
    task_items = [items_map[ref] for ref in leased.task.payload_ref]  # 本次任务涉及的零件列表。

    total_weight = sum(float(item["weight"]) for item in task_items)  # 任务总权重，用于统计。
    csv_rows = [
        {
            "task_id": leased.task.task_id,
            "part_id": item["part_id"],
            "group": item["group"],
            "category": item["category"],
            "weight": item["weight"],
        }
        for item in task_items
    ]

    progress_proxy = context.get("progress")
    if progress_proxy is not None:
        progress_proxy.advance(len(task_items))

    return TaskResult(
        payload=csv_rows,
        processed=len(task_items),
        metadata={
            "task_id": leased.task.task_id,
            "total_weight": total_weight,
        },
        output_directory=str(output_root),
        output_filename=f"{leased.task.task_id}",
        is_final_output=False,
    )


def aggregate_results(output_root: Path) -> Path:
    """将缓存中的 CSV 合并为 JSON 汇总，便于快速浏览结果。"""
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
            {
                "task_id": task_id,
                "row_count": len(rows),
                "weight": task_weight,
                "file": str(csv_path),
            }
        )
        summary["total_items"] += len(rows)
        summary["total_weight"] += task_weight

    output_path = output_root / "summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output_path


def run_sequential_executor(
    pool: TaskPool,
    handler,
    context: Dict[str, Any],
    result_handler,
    policy: ExecutorPolicy,
    driver_logger: StructuredLogger,
) -> None:
    """在多进程不可用时，使用串行方式模拟执行器行为。"""
    worker_logger = StructuredLogger.get_logger("demo.runner.serial")
    lease_ttl = getattr(policy, "lease_ttl", None)
    if lease_ttl is None or lease_ttl <= 0:
        lease_ttl = 10.0  # 兜底租约时长，确保任务能重新进入可见状态。
    filters = getattr(policy, "filters", None)  # 保持与并行执行器一致的过滤策略。

    while True:
        leased_batch = pool.lease(1, lease_ttl, filters=filters)  # 串行模式下每次只租借一个任务。
        if not leased_batch:
            break
        leased = leased_batch[0]
        try:
            result = handler(leased, context)  # 本质上与 worker 逻辑一致。
            _persist_task_result(result, worker_logger, leased.task.task_id)  # 仍然复用统一的落盘工具。
            pool.ack(leased.task.task_id)  # 租约完成即从池中清除。
            result_handler(leased, result)  # 回调 driver 以便记录指标。
        except Exception as exc:  # pragma: no cover - example fallback safety
            pool.nack(leased.task.task_id, requeue=False, delay=None)
            driver_logger.error("example.sequential_error", task_id=leased.task.task_id, error=str(exc))


def main() -> None:
    """脚本入口：串起读数、规划、执行、汇总的全流程。"""
    base_dir = Path(__file__).resolve().parent  # 示例资源所在目录。
    output_root = ensure_output_root(base_dir)  # 初始化输出目录结构。

    config = bootstrap_config(base_dir)
    ensure_task_config()

    logger = StructuredLogger.get_logger("demo.runner")
    logger.info("example.start", base_dir=str(base_dir))

    planner_items, items_by_ref = load_items(base_dir / "sample_items.json")  # 解析样例零件。
    queue_root = output_root / "queue"
    job_spec = {"job_id": "demo-job"}
    constraints = PartitionConstraints(max_items_per_task=2)
    strategy = PartitionStrategy.FIXED

    manifest_path = queue_root / "manifest.jsonl"
    pool = TaskPool(queue_root=queue_root, queue_name="default")
    weights: List[float] = []
    task_count = 0
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for task in TaskPartitioner.iter_tasks(job_spec, planner_items, strategy, constraints=constraints):
            pool.put(task)
            manifest.write(json.dumps({"task_id": task.task_id, "payload_ref": list(task.payload_ref)}) + "\n")
            weights.append(task.weight)
            task_count += 1

    logger.info(
        "example.plan",
        job_id=job_spec["job_id"],
        tasks=task_count,
        total_weight=sum(weights),
        gini=_compute_gini(weights),
    )

    logger.info(
        "example.queue_ready",
        directory=str(queue_root),
        manifest=str(manifest_path),
        records=task_count,
    )

    logger.info("example.pool_ready", stats=dict(pool.stats()))

    gpu_manager = GPUResourceManager.discover(preferred=None)  # 演示 GPU 探针能力。
    logger.info("example.gpu", device_count=gpu_manager.available())

    collected_metadata: List[Dict[str, Any]] = []  # 汇总每个任务的 metadata 以便最终展示。

    def on_result(_leased: LeasedTask, result: TaskResult) -> None:
        collected_metadata.append(result.metadata)
        logger.info(
            "example.task_complete",
            task_id=result.metadata.get("task_id"),
            total_weight=result.metadata.get("total_weight"),
            output=result.written_path,
        )

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    policy = ExecutorPolicy()  # 从配置加载执行策略。

    with ProgressController(total_units=len(planner_items), description="处理示例零件") as progress:
        context = {
            "items_map": items_by_ref,
            "output_root": output_root,
            "progress": progress.make_proxy(),
        }  # handler 运行所需的共享上下文。

        try:
            # 优先尝试使用多进程执行器从任务池中并行消费任务。
            ParallelExecutor.run(
                handler=task_handler,
                pool=pool,
                policy=policy,
                handler_context=context,
                result_handler=on_result,
                console_min_level="INFO",
            )
        except PermissionError as exc:
            logger.warning("example.parallel_unavailable", error=str(exc))
            run_sequential_executor(pool, task_handler, context, on_result, policy, logger)
        except OSError as exc:  # pragma: no cover - defensive fallback
            logger.warning("example.parallel_oserror", error=str(exc))
            run_sequential_executor(pool, task_handler, context, on_result, policy, logger)

    logger.info("example.executor_done", active=len(collected_metadata))
    logger.info("example.pool_after", stats=dict(pool.stats()))

    summary_path = aggregate_results(output_root)
    logger.info("example.summary_ready", path=str(summary_path))

    logger.info(
        "example.config",
        logger_namespace=config.get("logger.namespace", str, default="demo"),
    )
    logger.info("example.metadata", tasks=collected_metadata)
    logger.info("example.complete", output_root=str(output_root))


if __name__ == "__main__":
    main()

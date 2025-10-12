"""Base components shared across CAD pipelines."""

from .config import Config
from .logger import StructuredLogger
from .parallel_executor import ExecutorEvents, ExecutorPolicy, ParallelExecutor, TaskResult
from .task_partitioner import (
    ItemRecord,
    PartitionConstraints,
    PartitionStrategy,
    TaskPartitioner,
    TaskRecord,
)
from .task_pool import TaskPool, LeasedTask
from .task_system_config import ensure_task_config, get_executor_value, get_pool_value, get_task_value
from .gpu_resources import GPUDevice, GPUResourceManager
from .progress import ProgressController, ProgressProxy
from .file_queue import FileQueue, FileQueueConfig, LeaseRecord, MultiQueueGroup

__all__ = [
    "Config",
    "StructuredLogger",
    "ExecutorEvents",
    "ExecutorPolicy",
    "ParallelExecutor",
    "TaskResult",
    "ItemRecord",
    "PartitionConstraints",
    "PartitionStrategy",
    "TaskPartitioner",
    "TaskRecord",
    "TaskPool",
    "LeasedTask",
    "ensure_task_config",
    "get_executor_value",
    "get_pool_value",
    "get_task_value",
    "GPUDevice",
    "GPUResourceManager",
    "ProgressController",
    "ProgressProxy",
    "FileQueue",
    "FileQueueConfig",
    "LeaseRecord",
    "MultiQueueGroup",
]

"""Shared helpers for task system configuration."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .config import Config

_TASK_CONFIG_ENV = "CAD_TASK_CONFIG"
_DEFAULT_CONFIG_NAME = "main.yaml"


def _resolve_config_path() -> Path:
    env_path = os.environ.get(_TASK_CONFIG_ENV)
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Configured task system file not found: {path}")
        return path
    default_path = Path(__file__).with_name(_DEFAULT_CONFIG_NAME)
    if not default_path.exists():
        raise FileNotFoundError(
            "Task system configuration file not found. Set CAD_TASK_CONFIG or "
            f"place {_DEFAULT_CONFIG_NAME} alongside the modules."
        )
    return default_path


def ensure_task_config() -> Config:
    """Return the Config singleton initialised with task system settings."""
    try:
        return Config.get_singleton()
    except RuntimeError:
        path = _resolve_config_path()
        return Config.load_singleton(path)


def _task_config_key(key: str) -> str:
    return f"task_system.{key}" if key else "task_system"


def get_task_value(key: str, typ: type, default: Any = None, *, required: bool = False) -> Any:
    cfg = ensure_task_config()
    value = cfg.get(_task_config_key(key), typ, default)
    if required and value is None:
        raise RuntimeError(f"Missing required configuration key: {_task_config_key(key)}")
    return value


def get_pool_value(key: str, typ: type, default: Any = None, *, required: bool = False) -> Any:
    return get_task_value(f"pool.{key}", typ, default, required=required)


def get_executor_value(key: str, typ: type, default: Any = None, *, required: bool = False) -> Any:
    return get_task_value(f"executor.{key}", typ, default, required=required)


def reset_task_config() -> None:
    """Clear the Config singleton (primarily for tests)."""
    Config.set_singleton(None)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Structured Logging Service
-----------------------------

Implements a configurable structured logger with contextual bindings,
sampling, and pluggable sinks.

Key capabilities:
* JSON line output with deterministic fields.
* Context propagation with bind/unbind semantics using ``contextvars``.
* Probabilistic sampling with optional burst protection via token buckets.
* Multiple sinks: console, rotating file, timed rotating file, syslog,
  generic HTTP endpoints, and an in-memory sink for testing.
* Configuration helper that reuses the local ``Config`` service
  (``config_service.Config``) to load YAML definitions.
* Security-aware sanitisation: secret fields redacted, large payloads summarised.

The module also contains unit tests (``unittest``) at the bottom.
"""
from __future__ import annotations

import contextlib
import contextvars
import datetime as _dt
import hashlib
import json
import logging
import logging.handlers
import os
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

try:  # Python 3.9+
    from zoneinfo import ZoneInfo
except Exception as exc:  # pragma: no cover
    raise RuntimeError("zoneinfo is required (Python 3.9+) for timezone handling") from exc

try:
    import requests
except Exception:  # pragma: no cover - optional dependency in tests
    requests = None  # type: ignore[assignment]

try:
    from config_service import Config
except RuntimeError as exc:  # pragma: no cover - optional dependency in tests
    Config = None  # type: ignore[assignment]
    _CONFIG_IMPORT_ERROR = exc
else:
    _CONFIG_IMPORT_ERROR = None


__all__ = [
    "Logger",
    "Sampler",
    "SamplingRule",
    "StructuredJSONFormatter",
    "HTTPSinkHandler",
    "MemorySinkHandler",
    "SinkFactory",
]


_SECRET_MARKERS = ("secret", "password", "token", "key", "credential", "auth")
_STANDARD_FIELDS = (
    "trace_id",
    "span_id",
    "corr_id",
    "job_id",
    "task_id",
    "attempt",
    "latency_ms",
)


def _coerce_level(level: Union[str, int, None]) -> int:
    if level is None:
        return logging.INFO
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        lvl = logging.getLevelName(level.upper())
        if isinstance(lvl, int):
            return lvl
    raise ValueError(f"invalid logging level: {level!r}")


def _utc_iso(dt: _dt.datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_dt.timezone.utc)
    return dt.astimezone(_dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _hash_text(value: Union[str, bytes]) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8", errors="ignore")
    return hashlib.sha256(value).hexdigest()


def _is_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple, set, frozenset))


def _json_default(value: Any) -> str:
    return repr(value)


def _redact_value(key: str, value: Any) -> Any:
    key_lower = key.lower()
    if any(marker in key_lower for marker in _SECRET_MARKERS):
        return "[REDACTED]"
    return value


def _sanitize_value(key: str, value: Any, *, depth: int = 0, max_depth: int = 4) -> Any:
    value = _redact_value(key, value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, _dt.datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=_dt.timezone.utc)
        return value.astimezone(_dt.timezone.utc).isoformat()
    if isinstance(value, (_dt.date, _dt.time)):
        return value.isoformat()
    if isinstance(value, bytes):
        if len(value) > 512:
            return {
                "__redacted__": True,
                "hint": f"bytes[{len(value)}]",
                "sha256": _hash_text(value),
            }
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        if len(value) > 1024:
            return {
                "__redacted__": True,
                "hint": f"str[{len(value)}]",
                "sha256": _hash_text(value),
            }
        return value
    if depth >= max_depth:
        return {"__summary__": f"depth>{max_depth}", "repr": repr(value)}
    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for sub_key, sub_value in value.items():
            result[str(sub_key)] = _sanitize_value(f"{key}.{sub_key}", sub_value, depth=depth + 1, max_depth=max_depth)
        return result
    if _is_sequence(value):
        seq = list(value)
        if len(seq) > 64:
            digest = _hash_text(json.dumps(seq[:32], default=_json_default).encode("utf-8"))
            return {
                "__summary__": f"{type(value).__name__}[len={len(seq)}]",
                "sha256": digest,
            }
        return [_sanitize_value(f"{key}[{idx}]", item, depth=depth + 1, max_depth=max_depth) for idx, item in enumerate(seq)]
    if hasattr(value, "__dict__"):
        return _sanitize_value(f"{key}.dict", vars(value), depth=depth + 1, max_depth=max_depth)
    return repr(value)


def _sanitize_fields(fields: Mapping[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    for key, value in fields.items():
        sanitized[key] = _sanitize_value(key, value)
    return sanitized


@dataclass
class SamplingRule:
    event: Optional[str] = None
    level: Optional[str] = None
    rate: float = 1.0
    burst: int = 0
    interval: float = 60.0

    def key(self) -> Tuple[Optional[str], Optional[str]]:
        return (self.event, self.level)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SamplingRule":
        event = data.get("event")
        level = data.get("level")
        rate = float(data.get("rate", 1.0))
        burst = int(data.get("burst", 0))
        interval = float(data.get("interval", 60.0))
        return cls(event=event, level=level.upper() if isinstance(level, str) else None, rate=rate, burst=burst, interval=interval)


class _TokenBucket:
    def __init__(self, capacity: int, interval: float) -> None:
        if capacity <= 0:
            capacity = 1
        if interval <= 0:
            interval = 1.0
        self.capacity = float(capacity)
        self.tokens = float(capacity)
        self.interval = float(interval)
        self.refill_rate = self.capacity / self.interval
        self.updated = time.monotonic()

    def allow(self) -> bool:
        now = time.monotonic()
        elapsed = now - self.updated
        if elapsed > 0:
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.updated = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


class Sampler:
    def __init__(self, *, default_rate: float = 1.0, rules: Optional[Sequence[SamplingRule]] = None, seed: Optional[int] = None) -> None:
        self._default_rate = float(default_rate)
        self._rng = random.Random(seed)
        self._rules = list(rules or [])
        self._buckets: Dict[Tuple[Optional[str], Optional[str]], _TokenBucket] = {}

    @classmethod
    def from_config(cls, config: Optional[Mapping[str, Any]]) -> "Sampler":
        if not config:
            return cls()
        default_rate = float(config.get("default_rate", 1.0))
        seed = config.get("seed")
        rules_cfg = config.get("rules", []) or []
        rules = [SamplingRule.from_dict(rule) for rule in rules_cfg]
        return cls(default_rate=default_rate, rules=rules, seed=seed)

    def _match_rule(self, event: str, level: str) -> Optional[SamplingRule]:
        level = level.upper()
        for rule in self._rules:
            if rule.event and rule.event != event:
                continue
            if rule.level and rule.level != level:
                continue
            return rule
        return None

    def _allow_rate(self, rate: float) -> bool:
        if rate <= 0.0:
            return False
        if rate >= 1.0:
            return True
        return self._rng.random() <= rate

    def _check_burst(self, rule: SamplingRule, key: Tuple[Optional[str], Optional[str]]) -> bool:
        if rule.burst <= 0:
            return True
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _TokenBucket(capacity=rule.burst, interval=rule.interval)
            self._buckets[key] = bucket
        return bucket.allow()

    def allow(self, event: str, level: str) -> bool:
        rule = self._match_rule(event, level)
        if rule is None:
            return self._allow_rate(self._default_rate)
        key = rule.key()
        if not self._check_burst(rule, key):
            return False
        return self._allow_rate(rule.rate)


class StructuredJSONFormatter(logging.Formatter):
    def __init__(
        self,
        *,
        tz_render: Optional[str] = None,
        app_info: Optional[Mapping[str, Any]] = None,
        ensure_ascii: bool = False,
    ) -> None:
        super().__init__(datefmt="%Y-%m-%dT%H:%M:%S")
        self._render_zone = ZoneInfo(tz_render) if tz_render else None
        self._app_info = dict(app_info or {})
        self._ensure_ascii = ensure_ascii

    def format(self, record: logging.LogRecord) -> str:
        created = _dt.datetime.fromtimestamp(record.created, tz=_dt.timezone.utc)
        entry: Dict[str, Any] = {
            "ts": _utc_iso(created),
            "level": record.levelname,
            "event": getattr(record, "_structured_event", record.getMessage()),
            "msg": getattr(record, "_structured_msg", None),
            "logger": record.name,
            "file:line": f"{record.pathname}:{record.lineno}",
        }
        if self._render_zone is not None:
            entry["ts_render"] = created.astimezone(self._render_zone).isoformat()
        standard = getattr(record, "_structured_standard", {})
        for field in _STANDARD_FIELDS:
            value = standard.get(field)
            if value is not None:
                entry[field] = value
        extras = getattr(record, "_structured_extras", {})
        entry["extras"] = extras
        if self._app_info:
            entry["app"] = self._app_info
        if record.exc_info:
            with contextlib.suppress(Exception):
                entry["exception"] = {
                    "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                    "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                    "stack": self.formatException(record.exc_info),
                }
        return json.dumps(entry, ensure_ascii=self._ensure_ascii, default=_json_default)


class HTTPSinkHandler(logging.Handler):
    """Generic HTTP sink for ELK / OpenSearch / Datadog style ingestion."""

    def __init__(self, *, url: str, method: str = "POST", headers: Optional[Mapping[str, str]] = None, timeout: float = 5.0, verify: bool = True) -> None:
        if requests is None:  # pragma: no cover - handled at runtime
            raise RuntimeError("requests library is required for HTTP sink")
        super().__init__()
        self.url = url
        self.method = method.upper()
        self.headers = dict(headers or {"Content-Type": "application/json"})
        self.timeout = timeout
        self.verify = verify

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - network may be unavailable
        try:
            payload = self.format(record)
            requests.request(
                self.method,
                self.url,
                data=payload.encode("utf-8"),
                headers=self.headers,
                timeout=self.timeout,
                verify=self.verify,
            )
        except Exception:
            self.handleError(record)


class MemorySinkHandler(logging.Handler):
    """In-memory sink primarily for unit tests and diagnostics."""

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.records: List[str] = []
        self.is_memory_sink = True

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(self.format(record))


class SinkFactory:
    @staticmethod
    def create(sink_cfg: Union[str, Mapping[str, Any]]) -> logging.Handler:
        if isinstance(sink_cfg, str):
            sink_cfg = {"type": sink_cfg}
        if not isinstance(sink_cfg, Mapping):
            raise TypeError("sink configuration must be mapping or string")
        sink_type = str(sink_cfg.get("type", "console")).lower()
        if sink_type == "console":
            stream = sink_cfg.get("stream", "stdout")
            handler = logging.StreamHandler(stream=os.sys.stdout if stream == "stdout" else os.sys.stderr)
        elif sink_type == "rotating_file":
            path = sink_cfg.get("path")
            if not path:
                raise ValueError("rotating_file sink requires 'path'")
            max_bytes = int(sink_cfg.get("max_bytes", 10 * 1024 * 1024))
            backups = int(sink_cfg.get("backups", 5))
            Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
            handler = logging.handlers.RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backups, encoding="utf-8")
        elif sink_type == "timed_rotating_file":
            path = sink_cfg.get("path")
            if not path:
                raise ValueError("timed_rotating_file sink requires 'path'")
            when = sink_cfg.get("when", "midnight")
            interval = int(sink_cfg.get("interval", 1))
            backup = int(sink_cfg.get("backups", 7))
            Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
            handler = logging.handlers.TimedRotatingFileHandler(path, when=when, interval=interval, backupCount=backup, encoding="utf-8")
        elif sink_type == "syslog":
            address = sink_cfg.get("address", "/dev/log")
            facility = sink_cfg.get("facility", "user")
            handler = logging.handlers.SysLogHandler(address=address, facility=facility)
        elif sink_type == "http":
            url = sink_cfg.get("url")
            if not url:
                raise ValueError("http sink requires 'url'")
            method = sink_cfg.get("method", "POST")
            headers = sink_cfg.get("headers")
            timeout = float(sink_cfg.get("timeout", 5.0))
            verify = bool(sink_cfg.get("verify", True))
            handler = HTTPSinkHandler(url=url, method=method, headers=headers, timeout=timeout, verify=verify)
        elif sink_type == "memory":
            name = sink_cfg.get("name", "memory")
            handler = MemorySinkHandler(name=str(name))
        else:
            raise ValueError(f"unsupported sink type: {sink_type}")
        level = sink_cfg.get("level")
        if level is not None:
            handler.setLevel(_coerce_level(level))
        return handler


class Logger:
    _context: contextvars.ContextVar[Mapping[str, Any]] = contextvars.ContextVar("structured_logger_context", default={})
    _lock = threading.RLock()
    _configured = False
    _sampler = Sampler()
    _root_logger = logging.getLogger("cad")
    _root_name = "cad"
    _memory_sinks: Dict[str, MemorySinkHandler] = {}
    _app_info: Dict[str, Any] = {}

    def __init__(self, name: str = "app") -> None:
        self._ensure_configured()
        self._logger = self._root_logger.getChild(name)

    # -------------------- configuration --------------------
    @classmethod
    def _ensure_configured(cls) -> None:
        if not cls._configured:
            cls.configure(sinks=[{"type": "console", "stream": "stdout"}])

    @classmethod
    def configure(
        cls,
        *,
        sinks: Optional[Sequence[Union[str, Mapping[str, Any]]]] = None,
        level: Union[str, int, None] = None,
        sampling: Optional[Mapping[str, Any]] = None,
        app: Optional[Mapping[str, Any]] = None,
        tz_render: Optional[str] = None,
        namespace: str = "cad",
    ) -> None:
        with cls._lock:
            previous_handlers = list(cls._root_logger.handlers)
            for handler in previous_handlers:
                cls._root_logger.removeHandler(handler)
                with contextlib.suppress(Exception):
                    handler.close()
            cls._root_name = namespace
            cls._root_logger = logging.getLogger(namespace)
            cls._root_logger.setLevel(_coerce_level(level) if level is not None else logging.INFO)
            cls._root_logger.propagate = False
            cls._memory_sinks.clear()

            formatter = StructuredJSONFormatter(tz_render=tz_render, app_info=app)
            for sink in sinks or [{"type": "console", "stream": "stdout"}]:
                handler = SinkFactory.create(sink)
                handler.setFormatter(formatter)
                cls._root_logger.addHandler(handler)
                if getattr(handler, "is_memory_sink", False):
                    cls._memory_sinks[getattr(handler, "name", "memory")] = handler  # type: ignore[arg-type]
            cls._sampler = Sampler.from_config(sampling)
            cls._app_info = dict(app or {})
            cls._configured = True

    @classmethod
    def configure_from_file(
        cls,
        path: Union[str, Path] = "log.yaml",
        *,
        defaults: Optional[Mapping[str, Any]] = None,
        env_prefix: Optional[str] = None,
        cli_overrides: Optional[Dict[str, Any]] = None,
        migrate_to: Optional[int] = None,
    ) -> None:
        if Config is None:
            raise RuntimeError("config_service dependencies are unavailable") from _CONFIG_IMPORT_ERROR
        cfg = Config.load(
            path,
            defaults=dict(defaults or {}),
            env_prefix=env_prefix,
            cli_overrides=cli_overrides,
            migrate_to=migrate_to,
        )
        data = cfg.export("dict", redact_secrets=True)
        app_cfg = data.get("app", {}) or {}
        logger_cfg = data.get("logger", {}) or {}
        cls.configure(
            sinks=logger_cfg.get("sinks"),
            level=logger_cfg.get("level"),
            sampling=logger_cfg.get("sampling"),
            app={
                "name": app_cfg.get("name"),
                "env": app_cfg.get("env"),
            },
            tz_render=app_cfg.get("tz_render"),
            namespace=logger_cfg.get("namespace", "cad"),
        )

    # -------------------- context management --------------------
    @classmethod
    def reset_context(cls) -> None:
        cls._context.set({})

    def bind(self, **context: Any) -> "Logger":
        current = dict(self._context.get() or {})
        current.update(context)
        self._context.set(current)
        return self

    def unbind(self, *keys: str) -> "Logger":
        current = dict(self._context.get() or {})
        for key in keys:
            current.pop(key, None)
        self._context.set(current)
        return self

    # -------------------- logging primitives --------------------
    def _should_log(self, level: int, event: str) -> bool:
        if not self._logger.isEnabledFor(level):
            return False
        return self._sampler.allow(event, logging.getLevelName(level))

    def _log(self, level: int, event: str, *, msg: Optional[str] = None, exc_info: Optional[Tuple[type, BaseException, Any]] = None, fields: Optional[Mapping[str, Any]] = None) -> None:
        if not event:
            raise ValueError("event must be non-empty")
        if not self._should_log(level, event):
            return
        combined: Dict[str, Any] = {}
        context = self._context.get() or {}
        combined.update(context)
        combined.update(fields or {})
        sanitized = _sanitize_fields(combined)
        standard: Dict[str, Any] = {}
        for field in _STANDARD_FIELDS:
            if field in sanitized:
                standard[field] = sanitized.pop(field)
        structured_msg = msg or sanitized.pop("msg", None)
        self._logger.log(
            level,
            structured_msg or event,
            extra={
                "_structured_event": event,
                "_structured_msg": structured_msg,
                "_structured_extras": sanitized,
                "_structured_standard": standard,
            },
            exc_info=exc_info,
            stacklevel=3,
        )

    def info(self, event: str, **fields: Any) -> None:
        self._log(logging.INFO, event, fields=fields)

    def debug(self, event: str, **fields: Any) -> None:
        self._log(logging.DEBUG, event, fields=fields)

    def warn(self, event: str, **fields: Any) -> None:
        self._log(logging.WARNING, event, fields=fields)

    def warning(self, event: str, **fields: Any) -> None:
        self._log(logging.WARNING, event, fields=fields)

    def error(self, event: str, **fields: Any) -> None:
        exc = fields.pop("exc", None)
        exc_info = None
        if isinstance(exc, BaseException):
            exc_info = (type(exc), exc, exc.__traceback__)
        self._log(logging.ERROR, event, fields=fields, exc_info=exc_info)

    def exception(self, event: str, exc: BaseException, **fields: Any) -> None:
        fields = dict(fields)
        fields.setdefault("error_type", type(exc).__name__)
        fields.setdefault("error_message", str(exc))
        exc_info = (type(exc), exc, exc.__traceback__)
        self._log(logging.ERROR, event, fields=fields, exc_info=exc_info)

    # -------------------- diagnostics --------------------
    @classmethod
    def get_memory_sink(cls, name: str = "memory") -> Optional[MemorySinkHandler]:
        return cls._memory_sinks.get(name)


# -------------------- Unit Tests --------------------
if __name__ == "__main__":
    import tempfile
    import unittest

    class StructuredLoggerTests(unittest.TestCase):
        def tearDown(self) -> None:
            Logger.reset_context()

        def configure_for_test(self, sink_name: str, **kwargs: Any) -> Logger:
            Logger.configure(
                sinks=[{"type": "memory", "name": sink_name}],
                level="DEBUG",
                sampling={"default_rate": 1.0, "seed": 42},
                app={"name": "test-service", "env": "test"},
                tz_render="UTC",
                namespace=f"test.{sink_name}",
                **kwargs,
            )
            return Logger("unit")

        def test_basic_logging_and_context(self) -> None:
            sink_name = "basic"
            logger = self.configure_for_test(sink_name)
            logger.bind(trace_id="trace-123", user="alice")
            logger.info("user_login", msg="user logged in", latency_ms=12.5, role="admin")
            handler = Logger.get_memory_sink(sink_name)
            self.assertIsNotNone(handler)
            assert handler is not None
            self.assertEqual(len(handler.records), 1)
            payload = json.loads(handler.records[0])
            self.assertEqual(payload["event"], "user_login")
            self.assertEqual(payload["trace_id"], "trace-123")
            self.assertEqual(payload["extras"]["user"], "alice")
            self.assertEqual(payload["extras"]["role"], "admin")
            self.assertEqual(payload["latency_ms"], 12.5)
            self.assertEqual(payload["app"]["name"], "test-service")

        def test_unbind_removes_context(self) -> None:
            sink_name = "unbind"
            logger = self.configure_for_test(sink_name)
            logger.bind(request_id="req-1")
            logger.unbind("request_id")
            logger.info("operation", msg="done")
            handler = Logger.get_memory_sink(sink_name)
            assert handler is not None
            payload = json.loads(handler.records[0])
            self.assertNotIn("request_id", payload["extras"])

        def test_sampling_controls_output(self) -> None:
            sink_name = "sampling"
            Logger.configure(
                sinks=[{"type": "memory", "name": sink_name}],
                level="INFO",
                sampling={
                    "default_rate": 0.0,
                    "seed": 1,
                    "rules": [
                        {"event": "keep_me", "rate": 1.0},
                    ],
                },
                app={"name": "sample", "env": "test"},
                tz_render="UTC",
                namespace=f"test.{sink_name}",
            )
            logger = Logger("unit")
            logger.info("drop_me")
            logger.info("keep_me")
            handler = Logger.get_memory_sink(sink_name)
            assert handler is not None
            self.assertEqual(len(handler.records), 1)
            payload = json.loads(handler.records[0])
            self.assertEqual(payload["event"], "keep_me")

        @unittest.skipIf(Config is None, "Config service unavailable")
        def test_configure_from_yaml(self) -> None:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "log.yaml"
                path.write_text(
                    """
version: 1
app:
  name: yaml-service
  env: unit
  tz_store: UTC
  tz_render: UTC
logger:
  level: DEBUG
  namespace: yaml
  sinks:
    - type: memory
      name: yaml
  sampling:
    default_rate: 1.0
""".strip()
                )
                Logger.configure_from_file(path)
                logger = Logger("yaml")
                logger.info("yaml_event", msg="configured from yaml")
                handler = Logger.get_memory_sink("yaml")
                assert handler is not None
                self.assertEqual(len(handler.records), 1)
                payload = json.loads(handler.records[0])
                self.assertEqual(payload["event"], "yaml_event")
                self.assertEqual(payload["app"]["name"], "yaml-service")

    unittest.main()

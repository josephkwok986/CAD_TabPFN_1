#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Structured logging service.

Design goals (per specification):
    * Configurable first, sensible secure defaults.
    * Structured JSON line logs with contextual binding.
    * Deterministic sampling with burst protection.
    * Multiple sinks: console, rotating file, syslog, extensible adapters.
    * Secrets redaction and large object summarisation.
    * UTC storage timestamps with optional rendered timezone.
    * Support for trace/span/correlation/job/task identifiers.

The module exposes :class:`StructuredLogger` which mirrors the requested
interface:

    >>> logger = StructuredLogger.get_logger("cad.example")
    >>> logger.info("service.start", msg="Service has started")

Configuration can be bootstrapped from the :class:`config.Config`
singleton or provided programmatically.

Basic tests are provided at the bottom of the file and can be executed with::

    python logger.py
"""
from __future__ import annotations

import contextlib
import contextvars
import datetime as _dt
import inspect
import io
import json
import logging
import logging.handlers
import os
import random
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, MutableMapping, Optional, Tuple

try:  # Prefer ultra fast JSON serialiser when available.
    import orjson as _orjson  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _orjson = None

try:
    from dateutil import tz  # type: ignore
except Exception:
    from datetime import timezone as _timezone
    try:
        from zoneinfo import ZoneInfo  # type: ignore
    except Exception:  # pragma: no cover - Python <3.9
        ZoneInfo = None  # type: ignore

    class _FallbackTZ:
        @staticmethod
        def tzutc():
            return _timezone.utc

        @staticmethod
        def gettz(name: Optional[str]):
            if not name or ZoneInfo is None:
                return _timezone.utc
            with contextlib.suppress(Exception):
                return ZoneInfo(name)
            return _timezone.utc

    tz = _FallbackTZ()  # type: ignore

try:
    from .config import Config  # type: ignore
    _CONFIG_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover
    Config = None  # type: ignore
    _CONFIG_IMPORT_ERROR = exc


_LEVEL_MAP = {
    "DEBUG": 10,
    "INFO": 20,
    "WARN": 30,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}

_STANDARD_FIELDS = {
    "trace_id",
    "span_id",
    "corr_id",
    "job_id",
    "attempt",
    "latency_ms",
    "msg",
    "event",
    "ts",
    "logger",
    "file",
    "level",
    "extras",
}

_SECRET_PATTERNS = (
    "password",
    "secret",
    "token",
    "key",
    "credential",
    "pwd",
)

def _utcnow() -> _dt.datetime:
    return _dt.datetime.now(tz.tzutc())
def _safe_summary(value: Any, *, max_length: int = 16384) -> Any:
    """Return a representation safe for JSON serialisation."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        if len(value) > 16:
            return {
                "_type": type(value).__name__,
                "size": len(value),
                "sample": [_safe_summary(v) for v in list(value)[:5]],
            }
        return [_safe_summary(v) for v in value]
    if isinstance(value, dict):
        if len(value) > 16:
            return {
                "_type": "dict",
                "size": len(value),
                "keys": list(value.keys())[:8],
            }
        return {k: _safe_summary(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return {
            "_type": type(value).__name__,
            "size": len(value),
            "values": [_safe_summary(v) for v in list(value)[:8]],
        }
    if isinstance(value, (bytes, bytearray, memoryview)):
        size = len(value)
        head = bytes(value[:32]).hex()
        return {
            "_type": type(value).__name__,
            "size": size,
            "head": head,
        }
    text = str(value)
    if len(text) > max_length:
        return text[:max_length] + f"...<truncated {len(text) - max_length} chars>"
    return text


@dataclass
class SamplingRule:
    level: Optional[str] = None
    event: Optional[str] = None
    rate: float = 1.0
    burst: Optional[int] = None
    interval: float = 60.0

    def matches(self, level: str, event: str) -> bool:
        if self.level and self.level.upper() != level.upper():
            return False
        if self.event and self.event != event:
            return False
        return True


class Sampler:
    """Deterministic sampler with burst protection."""

    def __init__(
        self,
        default_rate: float = 1.0,
        rules: Optional[Iterable[SamplingRule]] = None,
        seed: int = 42,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self.default_rate = max(0.0, float(default_rate))
        self.rules: List[SamplingRule] = list(rules or [])
        self._rng = random.Random(seed)
        self._time_fn = time_fn
        self._windows: Dict[Tuple[str, str], deque] = {}
        self._lock = threading.Lock()

    def should_log(self, level: str, event: str) -> bool:
        level = level.upper()
        key = (level, event)
        rule = next((r for r in self.rules if r.matches(level, event)), None)
        rate = rule.rate if rule else self.default_rate
        burst = rule.burst if rule else None
        interval = rule.interval if rule else 60.0
        if burst and burst > 0:
            now = self._time_fn()
            with self._lock:
                dq = self._windows.setdefault(key, deque())
                while dq and now - dq[0] > interval:
                    dq.popleft()
                if len(dq) >= burst:
                    return False
                dq.append(now)
        if rate >= 1.0:
            return True
        if rate <= 0.0:
            return False
        with self._lock:
            return self._rng.random() < rate


@dataclass
class RedactionPolicy:
    allow: Tuple[str, ...] = field(default_factory=tuple)
    deny_patterns: Tuple[str, ...] = field(default_factory=lambda: _SECRET_PATTERNS)
    mask: str = "***"

    def should_mask(self, key: str) -> bool:
        k = key.lower()
        if key in self.allow:
            return False
        return any(pattern in k for pattern in self.deny_patterns)


class FieldRedactor:
    def __init__(self, policy: Optional[RedactionPolicy] = None) -> None:
        self.policy = policy or RedactionPolicy()

    def redact_mapping(self, data: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        for key in list(data.keys()):
            value = data[key]
            if self.policy.should_mask(key):
                data[key] = self.policy.mask
            else:
                data[key] = _safe_summary(value)
        return data


class Sink:
    def emit(self, record: Dict[str, Any], serialized: str) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - optional
        pass


class ConsoleSink(Sink):
    def __init__(self, stream: str = "stdout", min_level: Optional[str] = None) -> None:
        if stream not in {"stdout", "stderr"}:
            raise ValueError("stream must be 'stdout' or 'stderr'")
        self._stream = sys.stdout if stream == "stdout" else sys.stderr
        self._lock = threading.Lock()
        if min_level is not None:
            level_value = _LEVEL_MAP.get(min_level.upper())
            if level_value is None:
                raise ValueError(f"Unknown min_level for ConsoleSink: {min_level}")
            self._min_level = level_value
        else:
            self._min_level = None

    def emit(self, record: Dict[str, Any], serialized: str) -> None:
        if self._min_level is not None:
            level_name = str(record.get("level", "INFO")).upper()
            level_value = _LEVEL_MAP.get(level_name, 0)
            if level_value < self._min_level:
                return
        with self._lock:
            self._stream.write(serialized + "\n")
            self._stream.flush()


class RotatingFileSink(Sink):
    def __init__(self, path: str, max_bytes: int = 10 * 1024 * 1024, backups: int = 5) -> None:
        realpath = os.path.realpath(path)
        self._path = Path(realpath)
        parent = self._path.parent
        parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            retries = 0
            while retries < 8:
                try:
                    fd = os.open(realpath, os.O_APPEND | os.O_CREAT, 0o644)
                    os.close(fd)
                    break
                except FileNotFoundError:
                    time.sleep(0.2)
                    retries += 1
            else:
                parent_listing = list(parent.parent.glob("*"))
                raise FileNotFoundError(
                    f"无法创建日志文件: {self._path}；上层目录内容: {parent_listing}"
                )
        self._handler = logging.handlers.RotatingFileHandler(
            filename=str(self._path),
            maxBytes=int(max_bytes),
            backupCount=int(backups),
            encoding="utf-8",
        )
        self._lock = threading.Lock()

    def emit(self, record: Dict[str, Any], serialized: str) -> None:
        with self._lock:
            self._handler.emit(logging.makeLogRecord({"msg": serialized}))

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._handler.close()


class SyslogSink(Sink):
    def __init__(self, address: Optional[str] = None, port: Optional[int] = None, facility: str = "user") -> None:
        if address is None:
            if sys.platform.startswith("linux"):
                address = "/dev/log"
            elif sys.platform == "darwin":
                address = "/var/run/syslog"
            else:
                address = "localhost"
                port = port or 514
        if port is None:
            self._address = address
        else:
            self._address = (address, port)
        self._handler = logging.handlers.SysLogHandler(address=self._address, facility=facility)
        self._lock = threading.Lock()

    def emit(self, record: Dict[str, Any], serialized: str) -> None:
        with self._lock:
            self._handler.emit(logging.makeLogRecord({"msg": serialized}))

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._handler.close()


class AdapterSink(Sink):
    """Adapter sink allowing arbitrary callables."""

    def __init__(self, fn: Callable[[Dict[str, Any], str], None]) -> None:
        if not callable(fn):
            raise TypeError("fn must be callable")
        self._fn = fn

    def emit(self, record: Dict[str, Any], serialized: str) -> None:
        self._fn(record, serialized)


class _LoggerManager:
    def __init__(self) -> None:
        self._sinks: List[Sink] = [ConsoleSink()]
        self._lock = threading.RLock()
        self._level = _LEVEL_MAP["INFO"]
        self._sampler = Sampler()
        self._redactor = FieldRedactor()
        self._render_tz = tz.tzutc()
        self._namespace = "cad"

    # Configuration -------------------------------------------------
    def configure(
        self,
        *,
        sinks: Optional[Iterable[Sink]] = None,
        level: str = "INFO",
        sampler: Optional[Sampler] = None,
        redactor: Optional[FieldRedactor] = None,
        render_timezone: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> None:
        lvl = _LEVEL_MAP.get(level.upper())
        if lvl is None:
            raise ValueError(f"Unknown log level: {level}")
        with self._lock:
            if sinks:
                for sink in self._sinks:
                    with contextlib.suppress(Exception):
                        sink.close()
                self._sinks = list(sinks)
            self._level = lvl
            if sampler is not None:
                self._sampler = sampler
            if redactor is not None:
                self._redactor = redactor
            if render_timezone:
                try:
                    self._render_tz = tz.gettz(render_timezone) or tz.tzutc()
                except Exception:
                    self._render_tz = tz.tzutc()
            if namespace:
                self._namespace = namespace

    # Accessors -----------------------------------------------------
    @property
    def level(self) -> int:
        return self._level

    @property
    def namespace(self) -> str:
        return self._namespace

    @property
    def sampler(self) -> Sampler:
        return self._sampler

    @property
    def redactor(self) -> FieldRedactor:
        return self._redactor

    @property
    def render_tz(self):
        return self._render_tz

    # Emit ----------------------------------------------------------
    def emit(self, record: Dict[str, Any], serialized: str) -> None:
        with self._lock:
            for sink in self._sinks:
                try:
                    sink.emit(record, serialized)
                except Exception:
                    # Best-effort; continue to other sinks.
                    continue


_MANAGER = _LoggerManager()


class StructuredLogger:
    """Primary structured logger."""

    _context: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
        "structured_logger_context", default={}
    )

    def __init__(self, name: str) -> None:
        self.name = name

    # Factory -------------------------------------------------------
    @classmethod
    def get_logger(cls, name: Optional[str] = None) -> "StructuredLogger":
        full_name = name or _MANAGER.namespace
        return cls(full_name)

    # Configuration -------------------------------------------------
    @classmethod
    def configure(
        cls,
        *,
        sinks: Optional[List[Dict[str, Any]]] = None,
        level: Optional[str] = None,
        sampling: Optional[Dict[str, Any]] = None,
        redaction: Optional[Dict[str, Any]] = None,
        timezone: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> None:
        resolved_sinks = None
        if sinks is not None:
            resolved_sinks = [cls._build_sink(spec) for spec in sinks]
        sampler = None
        if sampling is not None:
            rules_cfg = sampling.get("rules", [])
            rules = [
                SamplingRule(
                    level=rule.get("level"),
                    event=rule.get("event"),
                    rate=float(rule.get("rate", 1.0)),
                    burst=rule.get("burst"),
                    interval=float(rule.get("interval", 60.0)),
                )
                for rule in rules_cfg
            ]
            sampler = Sampler(
                default_rate=float(sampling.get("default_rate", 1.0)),
                rules=rules,
                seed=int(sampling.get("seed", 42)),
            )
        redactor = None
        if redaction is not None:
            policy = RedactionPolicy(
                allow=tuple(redaction.get("allow", ())),
                deny_patterns=tuple(redaction.get("deny_patterns", _SECRET_PATTERNS)),
                mask=redaction.get("mask", "***"),
            )
            redactor = FieldRedactor(policy)
        _MANAGER.configure(
            sinks=resolved_sinks,
            level=level or "INFO",
            sampler=sampler,
            redactor=redactor,
            render_timezone=timezone,
            namespace=namespace,
        )

    @classmethod
    def configure_from_config(cls, config: Optional[Any] = None) -> None:
        cfg = config
        if cfg is None:
            if Config is None:
                raise RuntimeError("Config service is unavailable") from _CONFIG_IMPORT_ERROR
            try:
                cfg = Config.get_singleton()
            except RuntimeError as exc:
                raise RuntimeError("Config singleton is unavailable") from exc
        sinks = cfg.get("logger.sinks", list, [])
        level = cfg.get("logger.level", str, "INFO")
        sampling = cfg.get("logger.sampling", dict, {})
        redaction = cfg.get("logger.redact", dict, {})
        tz_render = cfg.get("logger.timezone.render", str, None)
        namespace = cfg.get("logger.namespace", str, _MANAGER.namespace)
        cls.configure(
            sinks=sinks,
            level=level,
            sampling=sampling,
            redaction=redaction,
            timezone=tz_render,
            namespace=namespace,
        )

    # Context -------------------------------------------------------
    def bind(self, **context: Any) -> None:
        current = dict(self._context.get())
        current.update(context)
        self._context.set(current)

    def unbind(self, *keys: str) -> None:
        current = dict(self._context.get())
        for key in keys:
            current.pop(key, None)
        self._context.set(current)

    # Logging -------------------------------------------------------
    def debug(self, event: str, **fields: Any) -> bool:
        return self._log("DEBUG", event, fields)

    def info(self, event: str, **fields: Any) -> bool:
        return self._log("INFO", event, fields)

    def warn(self, event: str, **fields: Any) -> bool:
        return self._log("WARN", event, fields)

    def warning(self, event: str, **fields: Any) -> bool:
        return self.warn(event, **fields)

    def error(self, event: str, **fields: Any) -> bool:
        return self._log("ERROR", event, fields)

    def exception(self, event: str, exc: BaseException, **fields: Any) -> bool:
        fields = dict(fields)
        fields.setdefault("error_type", type(exc).__name__)
        fields.setdefault("error_message", str(exc))
        fields["stack"] = _safe_summary("".join(logging.Formatter().formatException((type(exc), exc, exc.__traceback__))))
        return self._log("ERROR", event, fields)

    # Internals -----------------------------------------------------
    def _log(self, level: str, event: str, fields: Dict[str, Any]) -> bool:
        if _LEVEL_MAP[level] < _MANAGER.level:
            return False
        if not _MANAGER.sampler.should_log(level, event):
            return False
        record = self._make_record(level, event, fields)
        serialized = self._serialize(record)
        _MANAGER.emit(record, serialized)
        return True

    def _make_record(self, level: str, event: str, fields: Dict[str, Any]) -> Dict[str, Any]:
        now = _utcnow()
        offset = tz.gettz("Asia/Shanghai")
        if offset is None:
            offset = _dt.timezone(_dt.timedelta(hours=8))
        ts_value = now.astimezone(offset).isoformat()
        frame_info = self._caller_frame()
        context = self._context.get()
        combined: Dict[str, Any] = {**context, **fields}
        redacted = _MANAGER.redactor.redact_mapping(dict(combined))
        record: Dict[str, Any] = {
            "ts": ts_value,
            "level": level,
            "event": event,
            "file": f"{frame_info.filename}:{frame_info.lineno}",
        }
        for key in [
            "trace_id",
            "span_id",
            "corr_id",
            "job_id",
            "attempt",
            "latency_ms",
            "msg",
        ]:
            if key in redacted:
                record[key] = redacted.pop(key)
        redacted.pop("task_id", None)
        redacted.pop("stage", None)
        extras = {k: v for k, v in redacted.items() if k not in _STANDARD_FIELDS}
        record["extras"] = extras or None
        return record

    @staticmethod
    def _caller_frame() -> inspect.FrameInfo:
        stack = inspect.stack()
        # skip frames until outside logger module
        for frame in stack[2:]:
            mod = inspect.getmodule(frame.frame)
            if mod and mod.__name__ == __name__:
                continue
            return frame
        return stack[-1]

    @staticmethod
    def _serialize(record: Dict[str, Any]) -> str:
        def _stringify(value: Any) -> str:
            if isinstance(value, (dict, list, tuple)):
                return json.dumps(value, ensure_ascii=False)
            return str(value)

        segments: List[str] = []

        def append(field: Optional[str], val: Any) -> None:
            if val is None or val == "":
                return
            rendered = _stringify(val)
            if field is None:
                segments.append(rendered)
            else:
                segments.append(f"{field}={rendered}")

        append(None, record.get("ts"))
        append(None, record.get("file"))
        append(None, record.get("level"))
        append("event", record.get("event"))
        for key in ["trace_id", "span_id", "corr_id", "job_id", "attempt", "latency_ms", "msg"]:
            if key in record:
                append(key, record.get(key))

        extras = record.get("extras") or {}
        if isinstance(extras, dict):
            for key, value in extras.items():
                append(key, value)

        return "|".join(segments)

    # Utilities -----------------------------------------------------
    @staticmethod
    def _build_sink(spec: Dict[str, Any]) -> Sink:
        if isinstance(spec, Sink):
            return spec
        typ = spec.get("type")
        if typ == "console":
            return ConsoleSink(
                stream=spec.get("stream", "stdout"),
                min_level=spec.get("min_level"),
            )
        if typ in {"file", "rotating_file"}:
            try:
                return RotatingFileSink(
                    path=spec["path"],
                    max_bytes=int(spec.get("max_bytes", 10 * 1024 * 1024)),
                    backups=int(spec.get("backups", 5)),
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"无法创建日志文件 {spec.get('path')}: {exc}"
                ) from exc
        if typ == "syslog":
            return SyslogSink(
                address=spec.get("address"),
                port=spec.get("port"),
                facility=spec.get("facility", "user"),
            )
        if typ == "callable":
            target = spec.get("callable")
            if isinstance(target, str):
                fn = _import_string(target)
            else:
                fn = target
            return AdapterSink(fn)
        raise ValueError(f"Unsupported sink type: {typ}")


def _import_string(path: str) -> Callable[..., Any]:
    module_path, _, attr = path.rpartition(".")
    if not module_path:
        raise ValueError(f"Invalid import path: {path}")
    module = __import__(module_path, fromlist=[attr])
    return getattr(module, attr)


# Auto-configure from global configuration on import if available.
if Config is not None:
    try:
        StructuredLogger.configure_from_config(Config.get_singleton())
    except RuntimeError:
        StructuredLogger.configure(sinks=[{"type": "console", "stream": "stdout"}], level="INFO", sampling={"default_rate": 1.0, "seed": 42}, redaction={}, timezone="UTC", namespace="cad.logger")
else:
    StructuredLogger.configure(sinks=[{"type": "console", "stream": "stdout"}], level="INFO", sampling={"default_rate": 1.0, "seed": 42}, redaction={}, timezone="UTC", namespace="cad.logger")


# ---------------------------------------------------------------------------
# Tests (basic behaviour to guard core invariants)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile
    import unittest

    class TestStructuredLogger(unittest.TestCase):
        def setUp(self) -> None:
            # Reset configuration to avoid global state leaks between tests.
            StructuredLogger.configure(
                sinks=[{"type": "console", "stream": "stdout"}],
                level="DEBUG",
                sampling={"default_rate": 1.0, "seed": 123},
                redaction={},
                timezone="UTC",
                namespace="test.logger",
            )
            self.logger = StructuredLogger.get_logger("test.logger")

        def test_bind_and_unbind(self):
            self.logger.bind(trace_id="abc", custom="value")
            self.logger.info("event.test", msg="hello")
            self.logger.unbind("custom")
            self.logger.info("event.test", msg="hello2")
            ctx = StructuredLogger._context.get()
            self.assertNotIn("custom", ctx)
            self.assertEqual(ctx.get("trace_id"), "abc")

        def test_redaction_masks_sensitive_keys(self):
            stream = io.StringIO()
            sink = AdapterSink(lambda record, serialized: stream.write(serialized + "\n"))
            StructuredLogger.configure(
                sinks=[sink],
                level="INFO",
                sampling={"default_rate": 1.0},
                redaction={},
                timezone="UTC",
                namespace="test.logger",
            )
            logger = StructuredLogger.get_logger("test.logger")
            logger.bind(secret_token="should-hide")
            logger.info("event.redaction", msg="check")
            output = stream.getvalue()
            self.assertIn("\"secret_token\":\"***\"", output)

        def test_sampling_respected(self):
            StructuredLogger.configure(
                sinks=[{"type": "console"}],
                level="DEBUG",
                sampling={
                    "default_rate": 0.0,
                    "seed": 1,
                    "rules": [{"event": "keep", "rate": 1.0}],
                },
                redaction={"deny_patterns": []},
                timezone="UTC",
                namespace="test.logger",
            )
            logger = StructuredLogger.get_logger("test.logger")
            kept = logger.info("keep", foo="bar")
            dropped = logger.info("drop", foo="bar")
            self.assertTrue(kept)
            self.assertFalse(dropped)

        def test_rotating_file_sink_writes(self):
            with tempfile.TemporaryDirectory() as td:
                path = Path(td) / "app.log"
                StructuredLogger.configure(
                    sinks=[
                        {
                            "type": "rotating_file",
                            "path": str(path),
                            "max_bytes": 1024,
                            "backups": 1,
                        }
                    ],
                    level="INFO",
                    sampling={"default_rate": 1.0},
                    redaction={"deny_patterns": []},
                    timezone="UTC",
                    namespace="test.logger",
                )
                logger = StructuredLogger.get_logger("test.logger")
                logger.info("file.write", msg="hello file")
                logger.error("file.error", detail={"foo": "bar"})
                content = path.read_text(encoding="utf-8")
                self.assertIn("file.write", content)
                self.assertIn("file.error", content)

        def test_exception_logging_includes_stack(self):
            stream = io.StringIO()
            sink = AdapterSink(lambda record, serialized: stream.write(serialized + "\n"))
            StructuredLogger.configure(
                sinks=[sink],
                level="DEBUG",
                sampling={"default_rate": 1.0},
                redaction={"deny_patterns": []},
                timezone="UTC",
                namespace="test.logger",
            )
            logger = StructuredLogger.get_logger("test.logger")
            try:
                raise RuntimeError("boom")
            except RuntimeError as exc:
                logger.exception("error.runtime", exc, attempt=1)
            output = stream.getvalue()
            self.assertIn("error.runtime", output)
            self.assertIn("RuntimeError", output)

    unittest.main()

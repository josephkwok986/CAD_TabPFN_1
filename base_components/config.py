#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration Service (Config)
--------------------------------
Implements a unified configuration loader and validator with:
- YAML loading with !include and environment substitution
- layered overrides (defaults, file, env prefix, CLI)
- schema validation (Pydantic or JSON Schema)
- versioned migration pipeline
- snapshot freeze with canonical SHA-256 fingerprint
- reload()
- export() to yaml/json/dict

Design goals:
- Config.load(...) returns an instance. Methods match the requested interface.
- No logging. No print.
- Interfaces are general and pluggable.
- Tests included at the bottom with unittest.
"""
from __future__ import annotations

import os
import io
import re
import json
import copy
import hashlib
import signal
import threading
import contextlib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Dict, Iterable, Optional, Union, Tuple
from collections.abc import Mapping

try:
    import orjson as _orjson  # fast canonical JSON
except Exception:
    _orjson = None

try:
    import yaml  # PyYAML
except Exception as e:  # pragma: no cover
    raise RuntimeError("PyYAML is required") from e

# Optional validators
try:
    import jsonschema
except Exception:
    jsonschema = None

try:
    from pydantic import BaseModel
    import pydantic as _pydantic
    _PYDANTIC_V2 = hasattr(BaseModel, "model_validate")
except Exception:
    BaseModel = None  # type: ignore
    _pydantic = None
    _PYDANTIC_V2 = False


# ------------------------
# Helpers
# ------------------------

def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge overlay into base and return base."""
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def _deep_get(data: Mapping, path: str, default: Any = None) -> Any:
    cur: Any = data
    for part in path.split(".") if path else []:
        if isinstance(cur, Mapping) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def _ensure_path(data: dict, path: str) -> dict:
    cur = data
    for part in path.split("."):
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    return cur


def _deep_set(data: dict, path: str, value: Any) -> None:
    if not path:
        raise ValueError("empty key path")
    parts = path.split(".")
    cur = data
    for p in parts[:-1]:
        cur = _ensure_path(cur, p)
    cur[parts[-1]] = value


_ENV_RE = re.compile(r"\$\{\s*([A-Z0-9_]+)\s*(?::\s*([^}]*?)\s*)?\}", re.DOTALL)


def _sub_env_in_str(s: str) -> str:
    def repl(m: re.Match) -> str:
        key = m.group(1)
        default = m.group(2)
        default = default.strip() if default is not None else None
        return os.environ.get(key, default if default is not None else "")
    return _ENV_RE.sub(repl, s)


def _walk_env_substitution(obj: Any) -> Any:
    """Walk object and substitute ${VAR[:default]} inside strings."""
    if isinstance(obj, str):
        return _sub_env_in_str(obj)
    if isinstance(obj, list):
        return [ _walk_env_substitution(i) for i in obj ]
    if isinstance(obj, dict):
        return { k: _walk_env_substitution(v) for k, v in obj.items() }
    return obj


def _canonical_json(data: Any) -> bytes:
    """Canonical, stable JSON bytes for hashing and freeze."""
    def default(o):  # pragma: no cover
        raise TypeError(f"Unserializable type: {type(o)}")
    if _orjson is not None:
        return _orjson.dumps(data, option=_orjson.OPT_SORT_KEYS | _orjson.OPT_SERIALIZE_DATACLASS | _orjson.OPT_NON_STR_KEYS)
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=default).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _maybe_parse_scalar(s: str) -> Any:
    """Parse CLI/env override scalars to bool/int/float/None when obvious. Keep JSON literals as-is."""
    sl = s.lower()
    if sl == "null" or sl == "none":
        return None
    if sl == "true":
        return True
    if sl == "false":
        return False
    # int
    try:
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            return int(s)
    except Exception:
        pass
    # float
    try:
        if "." in s or "e" in sl:
            return float(s)
    except Exception:
        pass
    # JSON object/array
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        try:
            return json.loads(s)
        except Exception:
            return s
    return s


def parse_cli_overrides(items: Iterable[str]) -> Dict[str, Any]:
    """
    Convert ["a.b=1", "x.y.z=true", "list=[1,2]"] to {"a.b":1, ...}
    """
    out: Dict[str, Any] = {}
    for it in items:
        if "=" not in it:
            raise ValueError(f"Invalid override: {it!r}. Expect key=value")
        k, v = it.split("=", 1)
        k = k.strip()
        v = _maybe_parse_scalar(v.strip())
        out[k] = v
    return out


# ------------------------
# YAML loader with !include
# ------------------------

class _YamlLoader(yaml.SafeLoader):
    def __init__(self, stream, base_dir: Optional[Path] = None):
        if hasattr(stream, "name"):
            try:
                base = Path(stream.name).resolve().parent
            except Exception:
                base = None
        else:
            base = None
        super().__init__(stream)
        self._base_dir = Path(base_dir) if base_dir else base

def _yaml_include_constructor(loader: _YamlLoader, node):
    rel = loader.construct_scalar(node)
    if loader._base_dir is None:
        raise FileNotFoundError(f"!include used but base directory is unknown for {rel!r}")
    path = (loader._base_dir / rel).resolve()
    if not path.exists():
        raise FileNotFoundError(f"!include file not found: {path}")
    if path.suffix.lower() in {".yml", ".yaml", ".json"}:
        with open(path, "r", encoding="utf-8") as f:
            if path.suffix.lower() == ".json":
                return json.load(f)
            return yaml.load(f, Loader=lambda s: _YamlLoader(s, base_dir=path.parent))  # recursive includes
    else:
        return path.read_text(encoding="utf-8")

_YamlLoader.add_constructor("!include", _yaml_include_constructor)


# ------------------------
# Migrator registry
# ------------------------

class MigratorRegistry:
    """
    Register functions that migrate from version N to N+1.
    Each migrator: Callable[[dict], dict]
    """
    def __init__(self):
        self._migrators: Dict[int, Callable[[dict], dict]] = {}

    def register(self, from_version: int) -> Callable[[Callable[[dict], dict]], Callable[[dict], dict]]:
        def deco(fn: Callable[[dict], dict]):
            self._migrators[from_version] = fn
            return fn
        return deco

    def migrate(self, data: dict, target_version: Optional[int] = None) -> dict:
        cur = _deep_get(data, "config.version", None)
        if cur is None:
            return data
        max_v = max(self._migrators.keys()) if self._migrators else cur
        dst = target_version if target_version is not None else max_v
        if cur > dst:
            return data  # do not downgrade
        while cur < dst and cur in self._migrators:
            data = self._migrators[cur](data)
            cur = _deep_get(data, "config.version", cur + 1)
        return data


# ------------------------
# Core Config
# ------------------------

@dataclass(frozen=True)
class LoadInfo:
    source: str
    version: Optional[int]
    validated: bool
    fingerprint: str


class Config:
    """
    Main configuration service.
    """

    _singleton: Optional["Config"] = None
    _singleton_lock = threading.Lock()

    def __init__(self):
        self._data: dict = {}
        self._schema: Any = None
        self._schema_kind: str = "none"  # "pydantic" | "jsonschema" | "none"
        self._env_prefix: Optional[str] = None
        self._defaults: dict = {}
        self._path: Optional[Path] = None
        self._strict_mode: bool = True
        self._migrators = MigratorRegistry()
        self._last_info: Optional[LoadInfo] = None
        self._watcher_thread: Optional[threading.Thread] = None
        self._watch_stop = threading.Event()

    # ---------- Interface ----------

    @classmethod
    def _load_instance(
        cls,
        path_or_stream: Union[str, Path, io.TextIOBase],
        schema: Optional[Union[dict, "BaseModel"]] = None,
        defaults: Optional[dict] = None,
        env_prefix: Optional[str] = None,
        cli_overrides: Optional[Dict[str, Any]] = None,
        strict: bool = True,
        enable_sighup: bool = False,
        migrate_to: Optional[int] = None,
    ) -> "Config":
        """
        Load configuration with layering and validation.
        Priority:
          1) defaults
          2) YAML file (supports !include and ${ENV})
          3) environment variables with double underscores after env_prefix
          4) cli_overrides mapping of dot-path -> value
        Validation:
          - If schema is a Pydantic model class, use it.
          - Else if schema is a JSON Schema dict, use jsonschema.
          - Else skip validation.
        """
        self = cls()
        self._schema = schema
        self._schema_kind = "none"
        self._env_prefix = env_prefix
        self._defaults = copy.deepcopy(defaults or {})
        self._strict_mode = bool(str(strict).lower() not in {"0", "false"})

        # Read and parse YAML
        if isinstance(path_or_stream, (str, Path)):
            self._path = Path(path_or_stream).resolve()
            with open(self._path, "r", encoding="utf-8") as f:
                file_cfg = yaml.load(f, Loader=lambda s: _YamlLoader(s, base_dir=self._path.parent)) or {}
        else:
            self._path = None
            file_cfg = yaml.load(path_or_stream, Loader=_YamlLoader) or {}

        file_cfg = _walk_env_substitution(file_cfg)

        # Build merged config
        merged = copy.deepcopy(self._defaults)
        _deep_merge(merged, file_cfg)

        # Apply environment overrides
        if self._env_prefix:
            env_map = _extract_env_overrides(self._env_prefix)
            for k, v in env_map.items():
                _deep_set(merged, k, v)

        # Apply CLI overrides
        if cli_overrides:
            for k, v in cli_overrides.items():
                _deep_set(merged, k, v)

        # Version migrations
        merged = self._migrators.migrate(merged, target_version=migrate_to)

        # Validation
        validated = False
        if schema is not None:
            if BaseModel is not None and isinstance(schema, type) and issubclass(schema, BaseModel):
                self._schema_kind = "pydantic"
                merged = _validate_with_pydantic(schema, merged, strict=self._strict_mode)
                validated = True
            elif isinstance(schema, dict):
                if jsonschema is None:
                    raise RuntimeError("jsonschema not installed but JSON Schema provided")
                self._schema_kind = "jsonschema"
                jsonschema.validate(instance=merged, schema=schema)  # type: ignore
                validated = True
            else:
                raise TypeError("schema must be a Pydantic BaseModel subclass or a JSON Schema dict")

        # Store
        self._data = merged

        # Observability info
        fingerprint = self._fingerprint()
        version = _deep_get(merged, "config.version", None)
        source = str(self._path) if self._path else "<stream>"
        self._last_info = LoadInfo(source=source, version=version, validated=validated, fingerprint=fingerprint)

        # Optional SIGHUP reload
        if enable_sighup:
            _install_sighup_handler(self)

        return self

    @classmethod
    def load(
        cls,
        path_or_stream: Union[str, Path, io.TextIOBase],
        schema: Optional[Union[dict, "BaseModel"]] = None,
        defaults: Optional[dict] = None,
        env_prefix: Optional[str] = None,
        cli_overrides: Optional[Dict[str, Any]] = None,
        strict: bool = True,
        enable_sighup: bool = False,
        migrate_to: Optional[int] = None,
    ) -> "Config":
        return cls._load_instance(
            path_or_stream,
            schema=schema,
            defaults=defaults,
            env_prefix=env_prefix,
            cli_overrides=cli_overrides,
            strict=strict,
            enable_sighup=enable_sighup,
            migrate_to=migrate_to,
        )

    @classmethod
    def load_singleton(
        cls,
        path_or_stream: Union[str, Path, io.TextIOBase],
        schema: Optional[Union[dict, "BaseModel"]] = None,
        defaults: Optional[dict] = None,
        env_prefix: Optional[str] = None,
        cli_overrides: Optional[Dict[str, Any]] = None,
        strict: bool = True,
        enable_sighup: bool = False,
        migrate_to: Optional[int] = None,
    ) -> "Config":
        instance = cls._load_instance(
            path_or_stream,
            schema=schema,
            defaults=defaults,
            env_prefix=env_prefix,
            cli_overrides=cli_overrides,
            strict=strict,
            enable_sighup=enable_sighup,
            migrate_to=migrate_to,
        )
        with cls._singleton_lock:
            cls._singleton = instance
        return instance

    @classmethod
    def set_singleton(cls, instance: Optional["Config"]) -> None:
        with cls._singleton_lock:
            cls._singleton = instance

    @classmethod
    def get_singleton(cls) -> "Config":
        with cls._singleton_lock:
            if cls._singleton is None:
                raise RuntimeError("Config singleton has not been initialised")
            return cls._singleton

    def get(self, key: str, type: type, default: Any = None) -> Any:
        """Fetch dot-path key. Cast to given type."""
        val = _deep_get(self._data, key, default)
        if val is None and default is not None:
            return default
        if type is Any or type is None:
            return val
        try:
            if isinstance(val, type):
                return val
            # For complex types, try JSON round-trip
            if type in (dict, list) and isinstance(val, str):
                return type(json.loads(val))
            return type(val)
        except Exception:
            if default is not None:
                return default
            raise

    def reload(self) -> None:
        """Re-read the YAML source and re-apply all layers."""
        if self._path is None:
            raise RuntimeError("Cannot reload: original source was a stream")
        current = self.__class__.load(
            self._path,
            schema=self._schema,
            defaults=self._defaults,
            env_prefix=self._env_prefix,
            strict=self._strict_mode,
        )
        # Copy state
        self._data = current._data
        self._last_info = current._last_info

    def freeze(self) -> "ConfigSnapshot":
        """Return an immutable snapshot with version and fingerprint."""
        data = copy.deepcopy(self._data)
        return ConfigSnapshot(data)

    def export(self, fmt: str = "yaml", redact_secrets: bool = True) -> Union[str, dict]:
        """Export configuration in yaml/json/dict. `redact_secrets` is kept for compatibility."""
        data = self._exportable_dict(redact_secrets=redact_secrets)
        if fmt == "dict":
            return data
        if fmt == "json":
            return json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        if fmt == "yaml":
            return yaml.safe_dump(data, sort_keys=True, allow_unicode=True)
        raise ValueError("fmt must be one of: yaml, json, dict")

    @property
    def last_load_info(self) -> Optional[LoadInfo]:
        return self._last_info

    # ---------- Extension points ----------

    @property
    def migrators(self) -> MigratorRegistry:
        return self._migrators

    # ---------- Internals ----------

    def _exportable_dict(self, redact_secrets: bool = True) -> dict:  # pragma: no cover - param kept for compat
        return copy.deepcopy(self._data)

    def _fingerprint(self) -> str:
        exportable = self._exportable_dict(redact_secrets=True)
        return _sha256(_canonical_json(exportable))


class ConfigSnapshot:
    """Immutable view of configuration with fingerprint and helpers."""
    __slots__ = ("_data", "version", "fingerprint")

    def __init__(self, data: dict):
        d = copy.deepcopy(data)
        self._data = MappingProxyType(d)  # read-only mapping
        self.version = _deep_get(d, "config.version", None)
        self.fingerprint = _sha256(_canonical_json(d))

    def get(self, key: str, type: type, default: Any = None) -> Any:
        val = _deep_get(self._data, key, default)
        if val is None and default is not None:
            return default
        if type is Any or type is None:
            return val
        try:
            if isinstance(val, type):
                return val
            if type in (dict, list) and isinstance(val, str):
                return type(json.loads(val))
            return type(val)
        except Exception:
            if default is not None:
                return default
            raise

    def export(self, fmt: str = "yaml", redact_secrets: bool = True) -> Union[str, dict]:
        """Export snapshot data. `redact_secrets` is kept for compatibility."""
        data = copy.deepcopy(dict(self._data))
        if fmt == "dict":
            return data
        if fmt == "json":
            return json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        if fmt == "yaml":
            return yaml.safe_dump(data, sort_keys=True, allow_unicode=True)
        raise ValueError("fmt must be one of: yaml, json, dict")


# ------------------------
# Validators
# ------------------------

def _validate_with_pydantic(schema_cls: "BaseModel", data: dict, strict: bool) -> dict:
    """Return validated dict, regardless of pydantic v1/v2."""
    if _PYDANTIC_V2:
        # v2
        model = schema_cls.model_validate(data, strict=strict)  # type: ignore
        return model.model_dump()  # type: ignore
    # v1
    model = schema_cls.parse_obj(data)  # type: ignore
    return model.dict()  # type: ignore


# ------------------------
# Env override extraction
# ------------------------

def _extract_env_overrides(prefix: str) -> Dict[str, Any]:
    """
    Map PREFIX__A__B__C=val -> {"a.b.c": parsed(val)}
    Case-insensitive keys. Single underscores are preserved as part of names.
    """
    out: Dict[str, Any] = {}
    pref = f"{prefix}__"
    for k, v in os.environ.items():
        if k.startswith(pref):
            tail = k[len(pref):]
            parts = tail.split("__")
            # Keep original case for keys
            path = ".".join(parts)
            out[path] = _maybe_parse_scalar(v)
    return out


# ------------------------
# Optional: SIGHUP reload
# ------------------------

def _install_sighup_handler(cfg: Config) -> None:
    def _handler(signum, frame):  # pragma: no cover
        try:
            cfg.reload()
        except Exception:
            pass
    with contextlib.suppress(Exception):
        signal.signal(signal.SIGHUP, _handler)


# ------------------------
# -------------- Tests --------------
# ------------------------

def test():
    import tempfile
    import unittest
    from textwrap import dedent

    class TestConfig(unittest.TestCase):
        def setUp(self):
            # env for tests
            os.environ["APP__db__host"] = "ignored_lowercase"
            os.environ["APP__db__port"] = "6543"
            os.environ["APP__io__rate_limit"] = "42"
            os.environ["APP__list__values"] = "[1,2,3]"

        def tearDown(self) -> None:
            Config.set_singleton(None)

        def _write(self, text: str, dir: Path, name: str) -> Path:
            p = dir / name
            p.write_text(dedent(text).lstrip(), encoding="utf-8")
            return p

        def test_load_and_get_and_export(self):
            with tempfile.TemporaryDirectory() as td:
                d = Path(td)
                inc = self._write("""
                inner: 1
                """, d, "inc.yml")
                main = self._write(f"""
                config:
                  version: 1
                runtime:
                  tz: ${{
                      TZ:UTC
                    }}
                include_demo: !include {inc.name}
                io:
                  root: "/tmp"
                  retry: 3
                """, d, "conf.yml")

                cfg = Config.load(main, defaults={"io": {"retry": 1}}, env_prefix="APP")
                # defaults merged with file
                self.assertEqual(cfg.get("io.retry", int), 3)
                # env substitution present with default
                self.assertEqual(cfg.get("runtime.tz", str), "UTC")
                # include worked
                self.assertEqual(cfg.get("include_demo.inner", int), 1)
                # env overrides applied (prefix APP__)
                self.assertEqual(cfg.get("db.port", int, 0), 6543)
                # lists from env JSON
                self.assertEqual(cfg.get("list.values", list), [1,2,3])

                # exports
                as_dict = cfg.export("dict")
                self.assertEqual(as_dict["io"]["retry"], 3)
                yml = cfg.export("yaml")
                jsn = cfg.export("json")
                self.assertTrue(isinstance(yml, str) and isinstance(jsn, str))

                snap = cfg.freeze()
                self.assertEqual(snap.get("io.retry", int), 3)
                self.assertTrue(isinstance(snap.fingerprint, str))

        def test_cli_overrides_and_reload(self):
            with tempfile.TemporaryDirectory() as td:
                d = Path(td)
                main = self._write("""
                config: {version: 1}
                db: {host: "127.0.0.1", port: 5432}
                """, d, "conf.yml")
                overrides = parse_cli_overrides(["db.host=10.0.0.2", "limits.rate=100"])
                cfg = Config.load(main, cli_overrides=overrides)
                self.assertEqual(cfg.get("db.host", str), "10.0.0.2")
                # change file then reload
                main.write_text('config: {version: 1}\ndb: {host: "10.0.0.3", port: 5432}\n', encoding="utf-8")
                cfg.reload()
                self.assertEqual(cfg.get("db.host", str), "10.0.0.3")

        def test_pydantic_validation(self):
            if BaseModel is None:
                self.skipTest("pydantic not available")

            # v2 无警告配置；v1 兼容回退
            try:
                from pydantic import ConfigDict  # Pydantic v2
                class Model(BaseModel):  # type: ignore
                    model_config = ConfigDict(extra='forbid')
                    config: dict
                    db: dict
            except Exception:
                class Model(BaseModel):  # type: ignore
                    class Config:
                        extra = 'forbid'
                    config: dict
                    db: dict

            with tempfile.TemporaryDirectory() as td:
                d = Path(td)
                main = self._write("""
                config: {version: 1}
                db: {host: "x", port: 1}
                """, d, "conf.yml")
                cfg = Config.load(main, schema=Model)
                self.assertEqual(cfg.get("db.port", int), 1)
                self.assertTrue(cfg.last_load_info and cfg.last_load_info.validated)

        def test_jsonschema_validation(self):
            if jsonschema is None:
                self.skipTest("jsonschema not available")
            schema = {
                "type": "object",
                "properties": {
                    "config": {"type": "object"},
                    "db": {
                        "type": "object",
                        "properties": {
                            "host": {"type": "string"},
                            "port": {"type": "integer"},
                        },
                        "required": ["host", "port"]
                    }
                },
                "required": ["db"]
            }
            with tempfile.TemporaryDirectory() as td:
                d = Path(td)
                main = self._write("""
                config: {version: 1}
                db: {host: "x", port: 2}
                """, d, "conf.yml")
                cfg = Config.load(main, schema=schema)
                self.assertEqual(cfg.get("db.port", int), 2)
                self.assertTrue(cfg.last_load_info and cfg.last_load_info.validated)

        def test_fingerprint_changes_on_change(self):
            with tempfile.TemporaryDirectory() as td:
                d = Path(td)
                main = self._write("""
                config: {version: 1}
                x: 1
                """, d, "conf.yml")
                cfg = Config.load(main)
                fp1 = cfg.last_load_info.fingerprint if cfg.last_load_info else ""
                main.write_text('config: {version: 1}\nx: 2\n', encoding="utf-8")
                cfg.reload()
                fp2 = cfg.last_load_info.fingerprint if cfg.last_load_info else ""
                self.assertNotEqual(fp1, fp2)

        def test_singleton_helpers(self):
            with tempfile.TemporaryDirectory() as td:
                d = Path(td)
                main = self._write("""
                config: {version: 1}
                value: 123
                """, d, "conf.yml")

                Config.set_singleton(None)
                cfg = Config.load_singleton(main)
                self.assertIs(cfg, Config.get_singleton())
                self.assertEqual(Config.get_singleton().get("value", int), 123)

                Config.set_singleton(None)
                with self.assertRaises(RuntimeError):
                    Config.get_singleton()

    unittest.main()

    
if __name__ == "__main__":
    #test()
    cfg = Config.load_singleton('main.yaml')
    exported = cfg.export("json")
    print("## EXPORT(JSON)")
    print(json.dumps(json.loads(exported), ensure_ascii=False, indent=2))

    # 逐项读取示例
    def show(key, typ, default=None):
        try:
            val = cfg.get(key, typ, default)
            print(f"{key} = {val}")
        except Exception as e:
            print(f"{key} = <error: {e}>")

    print("\n## FIELDS")
    show("config.version", int)
    show("runtime.tz", str)
    show("io.root", str)
    show("io.retry", int)
    show("limits.rate", int, 0)
    show("limits.concurrency", int, 0)
    show("db.host", str)
    show("db.port", int)

    # 指纹与版本
    info = cfg.last_load_info
    if info:
        print("\n## META")
        print(f"source={info.source}")
        print(f"version={info.version}")
        print(f"validated={info.validated}")
        print(f"fingerprint={info.fingerprint}")

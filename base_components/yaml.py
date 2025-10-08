"""Minimal YAML loader for configurations when PyYAML is unavailable."""

import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple


class SafeLoader:
    """Compatibility shim exposing the subset of the PyYAML SafeLoader API."""

    _constructors: Dict[str, Callable[["SafeLoader", Any], Any]] = {}

    def __init__(self, stream: Any) -> None:
        self._text = _read_stream(stream)
        self._constructors = self.__class__._constructors.copy()

    @classmethod
    def add_constructor(cls, tag: str, constructor: Callable[["SafeLoader", Any], Any]) -> None:
        cls._constructors[tag] = constructor

    def read(self) -> str:
        return self._text

    def construct_scalar(self, node: Any) -> Any:  # pragma: no cover - shim only
        return node


_NUMERIC_RE = re.compile(r"^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$")


def _read_stream(stream: Any) -> str:
    if isinstance(stream, SafeLoader):
        return stream.read()
    if hasattr(stream, "read"):
        return stream.read()
    return str(stream)


def _parse_scalar(token: str) -> Any:
    token = token.strip()
    if not token:
        return None
    if token[0] == token[-1] and token[0] in {"'", '"'}:
        return token[1:-1]
    lowered = token.lower()
    if lowered in {"null", "~"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if _NUMERIC_RE.match(token):
        if any(c in token for c in ".eE"):
            try:
                return float(token)
            except Exception:
                return token
        try:
            return int(token)
        except Exception:
            return token
    if token.startswith("[") and token.endswith("]"):
        try:
            return json.loads(token)
        except Exception:
            return token
    if token.startswith("{") and token.endswith("}"):
        try:
            return json.loads(token)
        except Exception:
            return token
    return token


def _tokenize(text: str) -> List[str]:
    return [line.rstrip("\n") for line in text.splitlines()]


def _next_non_empty(lines: List[str], start: int) -> Tuple[Optional[int], Optional[int]]:
    i = start
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped and not stripped.startswith("#"):
            return i, len(lines[i]) - len(lines[i].lstrip(" "))
        i += 1
    return None, None


def _parse_block(lines: List[str], index: int, indent: int) -> Tuple[Any, int]:
    seq: Optional[List[Any]] = None
    mapping: Dict[str, Any] = {}
    i = index
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        cur_indent = len(raw) - len(raw.lstrip(" "))
        if cur_indent < indent:
            break
        if cur_indent > indent:
            raise ValueError(f"Unexpected indentation at line {i + 1}: {raw!r}")
        if stripped.startswith("- "):
            if seq is None:
                if mapping:
                    raise ValueError("Cannot mix list and mapping entries at the same level")
                seq = []
            value_part = stripped[2:].strip()
            child_indent = cur_indent + 2
            if not value_part:
                child, i = _parse_block(lines, i + 1, child_indent)
                seq.append(child)
                continue
            if ":" in value_part and not value_part.startswith("{"):
                key, rest = value_part.split(":", 1)
                item: Dict[str, Any] = {key.strip(): _parse_scalar(rest.strip())}
                j = i + 1
                while True:
                    next_index, next_indent = _next_non_empty(lines, j)
                    if next_index is None or next_indent is None or next_indent <= cur_indent:
                        break
                    child_map, j = _parse_block(lines, next_index, child_indent)
                    if not isinstance(child_map, dict):
                        raise ValueError("List item continuation must be a mapping")
                    item.update(child_map)
                seq.append(item)
                i = j
                continue
            seq.append(_parse_scalar(value_part))
            i += 1
            continue
        if seq is not None:
            raise ValueError("Cannot mix list and mapping entries at the same level")
        if ":" not in stripped:
            raise ValueError(f"Expected key-value pair at line {i + 1}: {raw!r}")
        key, rest = stripped.split(":", 1)
        key = key.strip()
        rest = rest.strip()
        if rest:
            mapping[key] = _parse_scalar(rest)
            i += 1
            continue
        child, i = _parse_block(lines, i + 1, indent + 2)
        mapping[key] = child
    return (seq if seq is not None else mapping), i


def load(stream: Any, Loader: Optional[Callable[[Any], SafeLoader]] = None) -> Any:
    if Loader is None:
        text = _read_stream(stream)
    else:
        loader = Loader(stream)
        text = loader.read() if hasattr(loader, "read") else _read_stream(loader)
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    text = text.strip()
    if not text:
        return None
    if text.lstrip().startswith("{") or text.lstrip().startswith("["):
        return json.loads(text)
    lines = _tokenize(text)
    data, _ = _parse_block(lines, 0, 0)
    return data


def safe_dump(data: Any, sort_keys: bool = True, allow_unicode: bool = True) -> str:
    return json.dumps(data, sort_keys=sort_keys, ensure_ascii=not allow_unicode)


__all__ = ["SafeLoader", "load", "safe_dump"]

"""Streaming helpers for CSV-based pipelines."""
from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, MutableMapping, Optional, Sequence, TypeVar

T = TypeVar("T")
Row = MutableMapping[str, object]


@dataclass
class StreamFingerprint:
    """Order-independent fingerprint builder based on record digests."""

    digest_size: int = 32
    _tokens: list[bytes] = None  # type: ignore[assignment]
    _hexdigest: Optional[str] = None

    def __post_init__(self) -> None:
        if self._tokens is None:
            self._tokens = []

    def add_values(self, *values: object) -> None:
        token_hasher = hashlib.sha256()
        for value in values:
            token_hasher.update(str(value).encode("utf-8"))
        self._tokens.append(token_hasher.digest()[: self.digest_size])
        self._hexdigest = None

    def hexdigest(self) -> str:
        if self._hexdigest is None:
            aggregate = hashlib.sha256()
            for token in sorted(self._tokens):
                aggregate.update(token)
            self._hexdigest = aggregate.hexdigest()
        return self._hexdigest

    def count(self) -> int:
        return len(self._tokens)


class CSVRecordWriter:
    """Stream records to CSV with optional fingerprint tracking."""

    def __init__(
        self,
        path: Path,
        fieldnames: Sequence[str],
        *,
        transform: Optional[Callable[[T], Mapping[str, object]]] = None,
        fingerprint: Optional[StreamFingerprint] = None,
        fingerprint_fields: Optional[Sequence[str]] = None,
        encoding: str = "utf-8",
    ) -> None:
        self._path = path
        self._fh = path.open("w", encoding=encoding, newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=list(fieldnames))
        self._writer.writeheader()
        self._transform = transform
        self._fingerprint = fingerprint
        self._fingerprint_fields = tuple(fingerprint_fields) if fingerprint_fields else None
        self._count = 0

    def write(self, record: T) -> None:
        row = self._make_row(record)
        self._writer.writerow(row)
        self._after_row(row)
        self._count += 1

    def write_batch(self, records: Iterable[T]) -> None:
        for record in records:
            self.write(record)

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.flush()
        finally:
            self._fh.close()

    def count(self) -> int:
        return self._count

    def fingerprint(self) -> Optional[str]:
        if self._fingerprint is None:
            return None
        return self._fingerprint.hexdigest()

    def _make_row(self, record: T) -> Row:
        if self._transform is None:
            if not isinstance(record, Mapping):
                raise TypeError("record must be mapping when no transform is provided")
            raw = record
        else:
            raw = self._transform(record)
        row: Row = {key: self._normalise_value(value) for key, value in raw.items()}
        return row

    def _after_row(self, row: Mapping[str, object]) -> None:
        if self._fingerprint is None or self._fingerprint_fields is None:
            return
        values = tuple(row.get(field) for field in self._fingerprint_fields)
        self._fingerprint.add_values(*values)

    @staticmethod
    def _normalise_value(value: object) -> object:
        if value is None:
            return ""
        return value


class CSVRecordReader(Iterator[Mapping[str, str]]):
    """Yield rows from a CSV file as dictionaries."""

    def __init__(
        self,
        path: Path,
        *,
        encoding: str = "utf-8",
    ) -> None:
        self._fh = path.open("r", encoding=encoding, newline="")
        self._reader = csv.DictReader(self._fh)

    def __iter__(self) -> "CSVRecordReader":
        return self

    def __next__(self) -> Mapping[str, str]:
        row = next(self._reader)
        if row is None:
            raise StopIteration
        return row

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "CSVRecordReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


"""基于本地文件的任务队列实现。

该实现遵循要求的磁盘布局与操作语义，提供 PUT/LEASE/ACK/NACK/EXTEND
等最少一次投递接口。此版本聚焦功能正确性，未对极端规模做额外优化，
但代码结构已按段文件 + ACK 位图 + 租约日志划分，后续可逐步扩展压实与
多进程互斥策略。
"""
from __future__ import annotations

import json
import heapq
import random
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Optional, Tuple

__all__ = [
    "FileQueueConfig",
    "LeaseRecord",
    "FileQueue",
    "MultiQueueGroup",
]


_HEADER_STRUCT = struct.Struct("<IBQ")  # u32 len, u8 flags, u64 rec_id
_HEADER_SIZE = _HEADER_STRUCT.size
_ALIGNMENT = 8


@dataclass
class FileQueueConfig:
    """队列运行参数。"""

    max_segment_bytes: int = 128 * 1024 * 1024
    flush_each_put: bool = False
    lease_log_batch: int = 32
    lease_sample_size: int = 2
    max_attempts: Optional[int] = None
    retry_delay_seconds: float = 0.0


@dataclass
class RecordInfo:
    """段内单条记录的索引信息（仅存最小必要字段）。"""

    rec_id: int
    segment: int
    segment_index: int
    offset: int
    length: int
    acked: bool = False
    visible_at: float = 0.0
    attempts: int = 0


@dataclass
class LeaseState:
    """租约状态，用于内存快速判断过期。"""

    rec_id: int
    expire_at: float


@dataclass
class LeaseRecord:
    """租约返回给调用者的信息。"""

    queue: str
    rec_id: int
    payload: bytes
    expire_at: float
    attempt: int


class _SegmentWriter:
    """负责管理单个段的写入与 ack 位图扩展。"""

    def __init__(self, base_dir: Path, segment: int, max_bytes: int, existing_records: int = 0) -> None:
        self.segment = segment
        self.base_dir = base_dir
        self.max_bytes = max_bytes
        self.data_path = base_dir / f"seg{segment:04d}.dat"
        self.ack_path = base_dir / f"seg{segment:04d}.ackmap"
        self._data = self.data_path.open("ab+")
        self._ack = self.ack_path.open("ab+")
        self._data.seek(0, 2)
        self._ack.seek(0, 2)
        self._size = self._data.tell()
        self.record_count = existing_records

    def close(self) -> None:
        self._data.close()
        self._ack.close()

    def has_capacity(self, payload_len: int) -> bool:
        padded = _padded_size(payload_len)
        if self._size == 0:
            return True  # 新段无论如何允许写入首条
        return self._size + padded <= self.max_bytes

    def append(self, rec_id: int, payload: bytes, flags: int = 0) -> Tuple[int, int]:
        payload_len = len(payload)
        padded = _padded_size(payload_len)
        offset = self._size
        header = _HEADER_STRUCT.pack(payload_len, flags, rec_id)
        self._data.write(header)
        self._data.write(payload)
        pad = padded - (_HEADER_SIZE + payload_len)
        if pad:
            self._data.write(b"\x00" * pad)
        self._size += padded
        seg_index = self.record_count
        self.record_count += 1
        self._ensure_ack_capacity(seg_index + 1)
        return offset, seg_index

    def flush(self) -> None:
        self._data.flush()
        self._ack.flush()

    def _ensure_ack_capacity(self, slots: int) -> None:
        """确保 ack 位图拥有 slots 位（向上取整到字节）。"""
        needed_bytes = (max(0, slots - 1) // 8) + 1 if slots else 0
        if needed_bytes <= 0:
            return
        self._ack.seek(0, 2)
        current = self._ack.tell()
        if current >= needed_bytes:
            return
        self._ack.write(b"\x00" * (needed_bytes - current))


def _padded_size(payload_len: int) -> int:
    raw = _HEADER_SIZE + payload_len
    padding = (-raw) % _ALIGNMENT
    return raw + padding


class FileQueue:
    """单个目录上的本地任务队列。"""

    def __init__(self, root: Path, config: Optional[FileQueueConfig] = None) -> None:
        self.root = Path(root)
        self.config = config or FileQueueConfig()
        self.segments_dir = self.root / "segments"
        self.meta_path = self.root / "meta.json"
        self.lease_log_path = self.root / "leases.log"
        self._lock = threading.RLock()

        self._records: List[RecordInfo] = []
        self._records_by_id: Dict[int, RecordInfo] = {}
        self._lease_map: Dict[int, LeaseState] = {}
        self._lease_heap: List[Tuple[float, int]] = []
        self._tail_writer: Optional[_SegmentWriter] = None
        self._segment_counts: Dict[int, int] = {}
        self._next_rec_id: int = 1
        self._scan_index: int = 0
        self._meta: MutableMapping[str, int] = {}
        self._max_attempts = self.config.max_attempts if self.config.max_attempts and self.config.max_attempts > 0 else None

        self._initialised = False

    # ---------- 公共 API ----------

    def initialise(self) -> None:
        with self._lock:
            if self._initialised:
                return
            self._init_storage()
            self._load_state()
            self._initialised = True

    def put_many(self, payloads: Iterable[bytes]) -> List[int]:
        ids: List[int] = []
        with self._lock:
            self._ensure_initialised()
            now = time.time()
            for payload in payloads:
                rec_id = self._next_rec_id
                self._next_rec_id += 1
                writer = self._ensure_tail_writer(len(payload))
                offset, seg_index = writer.append(rec_id, payload)
                record = RecordInfo(
                    rec_id=rec_id,
                    segment=writer.segment,
                    segment_index=seg_index,
                    offset=offset,
                    length=len(payload),
                    acked=False,
                    visible_at=now,
                    attempts=0,
                )
                self._records.append(record)
                self._records_by_id[rec_id] = record
                self._segment_counts[writer.segment] = writer.record_count
                ids.append(rec_id)
                self._meta["tail_segment"] = writer.segment
                self._meta["tail_offset"] = offset + _padded_size(len(payload))
                self._meta["record_count"] = len(self._records)
                self._meta["next_rec_id"] = self._next_rec_id
            if self.config.flush_each_put and self._tail_writer is not None:
                self._tail_writer.flush()
            self._persist_meta()
        return ids

    def put(self, payload: bytes) -> int:
        return self.put_many([payload])[0]

    def lease(self, max_n: int, ttl_seconds: float) -> List[LeaseRecord]:
        now = time.time()
        leases: List[LeaseRecord] = []
        with self._lock:
            self._ensure_initialised()
            self._reap_expired(now)
            if max_n <= 0:
                return leases
            idx = 0
            total = len(self._records)
            while idx < total and len(leases) < max_n:
                record = self._records[idx]
                idx += 1
                if record.acked:
                    continue
                if record.visible_at > now:
                    continue
                if record.rec_id in self._lease_map:
                    continue
                if self._max_attempts is not None and record.attempts >= self._max_attempts:
                    record.acked = True
                    self._set_ack_bit(record.segment, record.segment_index)
                    self._update_head_segment()
                    continue
                payload = self._read_payload(record)
                expire_at = now + max(ttl_seconds, 0.1)
                record.attempts += 1
                lease = LeaseState(rec_id=record.rec_id, expire_at=expire_at)
                self._lease_map[record.rec_id] = lease
                heapq.heappush(self._lease_heap, (expire_at, record.rec_id))
                leases.append(
                    LeaseRecord(
                        queue=self.root.name,
                        rec_id=record.rec_id,
                        payload=payload,
                        expire_at=expire_at,
                        attempt=record.attempts,
                    )
                )
                self._append_lease_log(record.rec_id, expire_at)
            self._scan_index = 0
        return leases

    def ack(self, rec_ids: Iterable[int]) -> None:
        with self._lock:
            self._ensure_initialised()
            for rec_id in rec_ids:
                record = self._records_by_id.get(rec_id)
                if record is None or record.acked:
                    continue
                record.acked = True
                self._clear_lease(rec_id)
                self._set_ack_bit(record.segment, record.segment_index)
            self._update_head_segment()
            self._persist_meta()

    def nack(self, rec_ids: Iterable[int], delay: Optional[float] = None) -> None:
        now = time.time()
        delay_value = delay if delay is not None else max(0.0, self.config.retry_delay_seconds)
        with self._lock:
            self._ensure_initialised()
            for rec_id in rec_ids:
                record = self._records_by_id.get(rec_id)
                if record is None:
                    continue
                self._clear_lease(rec_id)
                record.visible_at = now + delay_value
                self._append_lease_log(rec_id, now)
            self._scan_index = 0

    def extend(self, rec_ids: Iterable[int], ttl_seconds: float) -> None:
        now = time.time()
        expire_at = now + max(ttl_seconds, 0.1)
        with self._lock:
            self._ensure_initialised()
            for rec_id in rec_ids:
                lease = self._lease_map.get(rec_id)
                if lease is None:
                    continue
                lease.expire_at = expire_at
                heapq.heappush(self._lease_heap, (expire_at, rec_id))
                self._append_lease_log(rec_id, expire_at)

    def stats(self) -> Dict[str, int]:
        with self._lock:
            self._ensure_initialised()
            now = time.time()
            visible = 0
            for record in self._records:
                if record.acked:
                    continue
                if record.rec_id in self._lease_map:
                    continue
                if record.visible_at <= now:
                    visible += 1
            leased = len(self._lease_map)
            acked = sum(1 for record in self._records if record.acked)
            total = len(self._records)
            return {
                "visible": visible,
                "leased": leased,
                "acked": acked,
                "dead": 0,
                "total": total,
            }

    # ---------- 内部辅助 ----------

    def _ensure_initialised(self) -> None:
        if not self._initialised:
            raise RuntimeError("FileQueue 尚未 initialise()")

    def _init_storage(self) -> None:
        self.segments_dir.mkdir(parents=True, exist_ok=True)
        if not self.meta_path.exists():
            self._meta = {
                "head_segment": 0,
                "tail_segment": 0,
                "tail_offset": 0,
                "next_rec_id": 1,
                "record_count": 0,
            }
            self._persist_meta()

    def _load_state(self) -> None:
        self._meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self._next_rec_id = int(self._meta.get("next_rec_id", 1))
        self._records.clear()
        self._records_by_id.clear()
        self._segment_counts.clear()
        data_files = sorted(self.segments_dir.glob("seg*.dat"))
        for data_file in data_files:
            segment = int(data_file.stem.replace("seg", ""))
            ack_path = data_file.with_suffix(".ackmap")
            ack_bytes = ack_path.read_bytes() if ack_path.exists() else b""
            with data_file.open("rb") as handle:
                offset = 0
                idx = 0
                while True:
                    header = handle.read(_HEADER_SIZE)
                    if not header or len(header) < _HEADER_SIZE:
                        break
                    payload_len, _flags, rec_id = _HEADER_STRUCT.unpack(header)
                    padded = _padded_size(payload_len)
                    handle.seek(payload_len, 1)
                    pad = padded - (_HEADER_SIZE + payload_len)
                    if pad:
                        handle.seek(pad, 1)
                    acked = _bit_test(ack_bytes, idx)
                    record = RecordInfo(
                        rec_id=rec_id,
                        segment=segment,
                        segment_index=idx,
                        offset=offset,
                        length=payload_len,
                        acked=acked,
                        visible_at=0.0,
                        attempts=0,
                    )
                    self._records.append(record)
                    self._records_by_id[rec_id] = record
                    offset += padded
                    idx += 1
                self._segment_counts[segment] = idx
        self._records.sort(key=lambda r: (r.segment, r.segment_index))
        self._meta["record_count"] = len(self._records)
        if self._records:
            last = self._records[-1]
            self._meta["tail_segment"] = last.segment
        self._update_head_segment()
        # 加载租约日志
        if self.lease_log_path.exists():
            now = time.time()
            for line in self.lease_log_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec_id_s, expire_s = line.split(" ")
                rec_id = int(rec_id_s)
                expire_at = float(expire_s)
                record = self._records_by_id.get(rec_id)
                if record is None or record.acked:
                    continue
                if expire_at > now:
                    lease = LeaseState(rec_id=rec_id, expire_at=expire_at)
                    self._lease_map[rec_id] = lease
                    heapq.heappush(self._lease_heap, (expire_at, rec_id))
        tail_segment = self._meta.get("tail_segment", 0)
        tail_records = self._segment_counts.get(tail_segment, 0)
        self._tail_writer = _SegmentWriter(self.segments_dir, tail_segment, self.config.max_segment_bytes, tail_records)
        self._scan_index = 0

    def _ensure_tail_writer(self, payload_len: int) -> _SegmentWriter:
        writer = self._tail_writer
        if writer is None:
            writer = _SegmentWriter(self.segments_dir, 0, self.config.max_segment_bytes)
            self._tail_writer = writer
        if not writer.has_capacity(payload_len):
            writer.flush()
            writer.close()
            new_segment = writer.segment + 1
            writer = _SegmentWriter(self.segments_dir, new_segment, self.config.max_segment_bytes)
            self._tail_writer = writer
            self._meta["tail_segment"] = new_segment
            self._meta["tail_offset"] = 0
            self._segment_counts[new_segment] = 0
        return writer

    def _persist_meta(self) -> None:
        tmp_path = self.meta_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(self._meta, sort_keys=True), encoding="utf-8")
        tmp_path.replace(self.meta_path)

    def _read_payload(self, record: RecordInfo) -> bytes:
        data_path = self.segments_dir / f"seg{record.segment:04d}.dat"
        with data_path.open("rb") as handle:
            handle.seek(record.offset + _HEADER_SIZE)
            return handle.read(record.length)

    def _set_ack_bit(self, segment: int, index: int) -> None:
        ack_path = self.segments_dir / f"seg{segment:04d}.ackmap"
        if not ack_path.exists():
            ack_path.parent.mkdir(parents=True, exist_ok=True)
            ack_path.write_bytes(b"\x00")
        with ack_path.open("r+b") as handle:
            byte_index = index // 8
            bit_mask = 1 << (index % 8)
            handle.seek(0, 2)
            current_size = handle.tell()
            if current_size <= byte_index:
                handle.write(b"\x00" * (byte_index + 1 - current_size))
            handle.seek(byte_index)
            current = handle.read(1)
            current_byte = current[0] if current else 0
            if current_byte & bit_mask:
                return
            handle.seek(byte_index)
            handle.write(bytes([current_byte | bit_mask]))

    def _append_lease_log(self, rec_id: int, expire_at: float) -> None:
        line = f"{rec_id} {expire_at:.9f}\n"
        with self.lease_log_path.open("a", encoding="utf-8") as handle:
            handle.write(line)

    def _update_head_segment(self) -> None:
        for record in self._records:
            if not record.acked:
                self._meta["head_segment"] = record.segment
                return
        self._meta["head_segment"] = self._meta.get("tail_segment", 0)

    def _clear_lease(self, rec_id: int) -> None:
        lease = self._lease_map.pop(rec_id, None)
        if lease is None:
            return
        heapq.heappush(self._lease_heap, (0.0, rec_id))  # 触发堆清理
        self._scan_index = 0

    def _reap_expired(self, now: float) -> None:
        while self._lease_heap:
            expire_at, rec_id = self._lease_heap[0]
            if expire_at == 0.0:
                heapq.heappop(self._lease_heap)
                continue
            if expire_at > now:
                break
            heapq.heappop(self._lease_heap)
            lease = self._lease_map.get(rec_id)
            if lease is None:
                continue
            if lease.expire_at > now:
                heapq.heappush(self._lease_heap, (lease.expire_at, rec_id))
                continue
            self._lease_map.pop(rec_id, None)
            record = self._records_by_id.get(rec_id)
            if record is not None and not record.acked:
                record.visible_at = now + max(0.0, self.config.retry_delay_seconds)
            self._scan_index = 0


def _bit_test(data: bytes, index: int) -> bool:
    byte_index = index // 8
    if byte_index >= len(data):
        return False
    bit = index % 8
    return bool(data[byte_index] & (1 << bit))


class MultiQueueGroup:
    """管理多个 FileQueue 的轮询调度器。"""

    def __init__(self, root: Path, config: Optional[FileQueueConfig] = None) -> None:
        self.root = Path(root)
        self.config = config or FileQueueConfig()
        self._queues: Dict[str, FileQueue] = {}
        self._lock = threading.RLock()

    def ensure_queue(self, name: str) -> FileQueue:
        with self._lock:
            queue = self._queues.get(name)
            if queue is not None:
                return queue
            queue_root = self.root / name
            queue = FileQueue(queue_root, self.config)
            queue.initialise()
            self._queues[name] = queue
            return queue

    def put(self, queue_name: str, payload: bytes) -> int:
        queue = self.ensure_queue(queue_name)
        return queue.put(payload)

    def put_many(self, queue_name: str, payloads: Iterable[bytes]) -> List[int]:
        queue = self.ensure_queue(queue_name)
        return queue.put_many(payloads)

    def lease(self, max_n: int, ttl_seconds: float) -> List[LeaseRecord]:
        with self._lock:
            if not self._queues:
                return []
            queues = list(self._queues.values())
        sample_size = min(self.config.lease_sample_size, len(queues))
        random.shuffle(queues)
        leases: List[LeaseRecord] = []
        for queue in queues[:sample_size]:
            leases.extend(queue.lease(max_n - len(leases), ttl_seconds))
            if len(leases) >= max_n:
                break
        return leases

    def ack(self, items: Iterable[LeaseRecord]) -> None:
        grouped: Dict[str, List[int]] = {}
        for item in items:
            grouped.setdefault(item.queue, []).append(item.rec_id)
        for name, rec_ids in grouped.items():
            queue = self.ensure_queue(name)
            queue.ack(rec_ids)

    def nack(self, items: Iterable[LeaseRecord]) -> None:
        grouped: Dict[str, List[int]] = {}
        for item in items:
            grouped.setdefault(item.queue, []).append(item.rec_id)
        for name, rec_ids in grouped.items():
            queue = self.ensure_queue(name)
            queue.nack(rec_ids)

    def extend(self, items: Iterable[LeaseRecord], ttl_seconds: float) -> None:
        grouped: Dict[str, List[int]] = {}
        for item in items:
            grouped.setdefault(item.queue, []).append(item.rec_id)
        for name, rec_ids in grouped.items():
            queue = self.ensure_queue(name)
            queue.extend(rec_ids, ttl_seconds)

    def stats(self, queue_name: str) -> Dict[str, int]:
        queue = self.ensure_queue(queue_name)
        return queue.stats()

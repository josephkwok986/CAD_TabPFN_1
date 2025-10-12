import json
import time
from pathlib import Path

import pytest

from base_components.file_queue import FileQueue, FileQueueConfig, MultiQueueGroup


@pytest.fixture
def queue_root(tmp_path: Path) -> Path:
    root = tmp_path / "queues"
    root.mkdir()
    return root


def test_single_queue_basic_flow(queue_root: Path) -> None:
    queue_dir = queue_root / "default"
    cfg = FileQueueConfig(flush_each_put=True)
    queue = FileQueue(queue_dir, cfg)
    queue.initialise()

    ids = queue.put_many([b"task-a", b"task-b"])
    assert len(ids) == 2

    leases = queue.lease(1, ttl_seconds=1.0)
    assert len(leases) == 1
    first = leases[0]
    assert first.payload == b"task-a"

    queue.ack([first.rec_id])

    leases = queue.lease(1, ttl_seconds=0.5)
    assert len(leases) == 1
    second = leases[0]
    assert second.payload == b"task-b"

    queue.nack([second.rec_id])

    leases = queue.lease(1, ttl_seconds=0.1)
    assert len(leases) == 1
    second = leases[0]
    queue.extend([second.rec_id], ttl_seconds=0.5)
    time.sleep(0.2)

    # 仍在租约内，不应重复出队
    assert not queue.lease(1, ttl_seconds=0.5)

    time.sleep(0.4)
    # 过期回收后再次租约
    leases = queue.lease(1, ttl_seconds=0.5)
    assert len(leases) == 1
    queue.ack([leases[0].rec_id])


def test_multi_queue_group_round_robin(queue_root: Path) -> None:
    cfg = FileQueueConfig(flush_each_put=True)
    group = MultiQueueGroup(queue_root, cfg)
    group.put("q1", b"one")
    group.put("q2", b"two")

    leases = group.lease(2, ttl_seconds=0.5)
    queues = {lease.queue for lease in leases}
    assert queues == {"q1", "q2"}

    group.ack(leases)

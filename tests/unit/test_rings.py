import multiprocessing as mp
import struct
import time

import pytest

from ai_phone.telephony import rings
from ai_phone.telephony.rings import FRAME_BYTES, SILENCE, FrameRing


def test_push_pop_fifo():
    r = FrameRing(None, frames=4)
    try:
        r.push(b"\x01" * FRAME_BYTES)
        r.push(b"\x02" * FRAME_BYTES)
        assert r.available() == 2
        assert r.pop() == b"\x01" * FRAME_BYTES
        assert r.pop() == b"\x02" * FRAME_BYTES
        assert r.pop() is None
    finally:
        r.close()
        r.unlink()


def test_overflow_drops_newest_and_counts():
    r = FrameRing(None, frames=2)
    try:
        for i in range(1, 4):
            r.push(bytes([i]) * FRAME_BYTES)
        assert r.overruns == 1
        assert r.pop() == b"\x01" * FRAME_BYTES  # the queued frames are untouched; the third was dropped
        assert r.pop() == b"\x02" * FRAME_BYTES
        assert r.pop() is None
    finally:
        r.close()
        r.unlink()


def test_underflow_returns_silence_and_counts():
    r = FrameRing(None, frames=2)
    try:
        assert r.pop_or_silence() == SILENCE
        assert r.underruns == 1
    finally:
        r.close()
        r.unlink()


def test_flush_discards_everything():
    r = FrameRing(None, frames=8)
    try:
        for _ in range(5):
            r.push(b"\x07" * FRAME_BYTES)
        r.flush()
        assert r.available() == 0 and r.pop() is None
    finally:
        r.close()
        r.unlink()


def test_wrong_frame_size_rejected():
    r = FrameRing(None, frames=2)
    try:
        with pytest.raises(ValueError):
            r.push(b"\x00" * 10)
    finally:
        r.close()
        r.unlink()


def _child(name, n):
    r = FrameRing.attach(name)
    for i in range(n):
        r.push(bytes([i % 256]) * FRAME_BYTES)
    r.close()


def test_cross_process_push():
    r = FrameRing(None, frames=16)
    try:
        p = mp.get_context("spawn").Process(target=_child, args=(r.name, 5))
        p.start()
        p.join(10)
        assert p.exitcode == 0
        assert r.available() == 5
        assert r.pop() == b"\x00" * FRAME_BYTES
    finally:
        r.close()
        r.unlink()


def _producer(name, n):
    r = FrameRing.attach(name)
    for i in range(n):
        while True:  # spin until there is room: this producer never drops
            before = r.overruns
            r.push(i.to_bytes(4, "little") * (FRAME_BYTES // 4))
            if r.overruns == before:
                break
    r.close()


def test_concurrent_producer_and_consumer_never_lose_or_duplicate_frames():
    r = FrameRing(None, frames=8)
    try:
        n = 3000
        p = mp.get_context("spawn").Process(target=_producer, args=(r.name, n))
        p.start()
        seen = []
        import time
        deadline = time.time() + 30
        while len(seen) < n and time.time() < deadline:
            f = r.pop()
            if f is None:
                continue
            seen.append(int.from_bytes(f[:4], "little"))
        p.join(10)
        assert seen == list(range(n)), f"got {len(seen)} frames, first mismatch at {next((i for i, v in enumerate(seen) if v != i), None)}"
    finally:
        r.close()
        r.unlink()


def test_capacity_check_cannot_be_invalidated_by_a_transient_zero_tail(monkeypatch):
    """Expose the zero-fill stage of CPython's pack_into before publishing tail.

    A producer has already observed spare capacity when the consumer publishes
    its next tail. A live counter store must preserve that capacity: serializing
    it with pack_into first clears the destination and falsely makes it full.
    The observer runs in that known intermediate stage if serialization is used,
    or immediately after the completed scalar publication otherwise.
    """
    producer = FrameRing(None, frames=2)
    consumer = FrameRing.attach(producer.name)
    frames = [bytes([i]) * FRAME_BYTES for i in range(1, 5)]
    observed = False
    u64 = struct.Struct("<Q")

    class ObservePackZeroFill:
        def unpack_from(self, buffer, offset):
            return u64.unpack_from(buffer, offset)

        def pack_into(self, buffer, offset, value):
            nonlocal observed
            if offset == rings._TAIL and value == 3 and not observed:
                observed = True
                # CPython s_pack_internal does this memset before packing.
                u64.pack_into(buffer, offset, 0)
                producer.push(frames[3])
            u64.pack_into(buffer, offset, value)

    try:
        producer.push(frames[0])
        producer.push(frames[1])
        assert consumer.pop() == frames[0]
        producer.push(frames[2])
        assert consumer.pop() == frames[1]
        assert producer.available() == 1  # The producer may enqueue another frame.
        monkeypatch.setattr(rings, "_U64", ObservePackZeroFill(), raising=False)
        assert consumer.pop() == frames[2]
        if not observed:
            producer.push(frames[3])
        assert producer.overruns == 0
        assert producer.available() == 1
        assert consumer.pop() == frames[3]
        assert consumer.pop() is None
    finally:
        consumer.close()
        producer.close()
        producer.unlink()


@pytest.mark.parametrize("start", [(1 << 32) - 1, (1 << 63) + 17])
def test_attached_ring_preserves_wide_counters_through_capacity_and_flush(start):
    producer = FrameRing(None, frames=2)
    consumer = FrameRing.attach(producer.name)
    try:
        producer._wr(rings._HEAD, start)
        consumer._wr(rings._TAIL, start)
        producer._wr(rings._OVER, start)
        consumer._wr(rings._UNDER, start)
        first, second = b"a" * FRAME_BYTES, b"b" * FRAME_BYTES
        producer.push(first)
        producer.push(second)
        assert consumer.available() == 2
        producer.push(b"c" * FRAME_BYTES)
        assert consumer.overruns == start + 1
        assert consumer.pop() == first
        consumer.flush()
        assert producer.available() == 0
        assert producer._rd(rings._HEAD) == consumer._rd(rings._TAIL) == start + 2
        assert consumer.pop_or_silence() == SILENCE
        assert producer.underruns == start + 1
    finally:
        consumer.close()
        producer.close()
        producer.unlink()


def test_owner_and_attached_view_close_without_exported_buffer_leaks():
    owner = FrameRing(None, frames=2)
    attached = FrameRing.attach(owner.name)
    name = owner.name
    try:
        owner.push(b"a" * FRAME_BYTES)
        owner.close()
        # Closing one mapping must not invalidate the separately attached one.
        assert attached.pop() == b"a" * FRAME_BYTES
        attached.close()
        owner.unlink()
        with pytest.raises(FileNotFoundError):
            FrameRing.attach(name)
    finally:
        attached.close()
        owner.close()
        owner.unlink()


def _producer_without_retries(name, count, stop):
    ring = FrameRing.attach(name)
    deadline = time.monotonic() + 20
    try:
        for number in range(1, count + 1):
            while ring.available() >= ring.frames:
                if stop.is_set() or time.monotonic() >= deadline:
                    return
            # Exactly one push after capacity is observed. Retrying after an
            # overrun would hide the transient false-full counter defect.
            ring.push(number.to_bytes(4, "little") * (ring.frame_bytes // 4))
    finally:
        ring.close()


@pytest.mark.parametrize("frame_bytes", [644, 1924])
def test_spawned_producer_never_drops_after_waiting_for_capacity(frame_bytes):
    ring = FrameRing(None, frames=2, frame_bytes=frame_bytes)
    ctx = mp.get_context("spawn")
    stop = ctx.Event()
    count = 50000
    child = ctx.Process(target=_producer_without_retries, args=(ring.name, count, stop))
    seen = 0
    try:
        child.start()
        deadline = time.monotonic() + 20
        while seen < count and time.monotonic() < deadline:
            frame = ring.pop()
            if frame is None:
                if child.exitcode is not None:
                    break
                continue
            expected = (seen + 1).to_bytes(4, "little") * (frame_bytes // 4)
            assert frame == expected, f"frame {seen + 1}: received marker {int.from_bytes(frame[:4], 'little')}"
            seen += 1
        child.join(2)
        assert child.exitcode == 0
        assert seen == count, f"received {seen}/{count} frames; overruns={ring.overruns}"
        assert ring.available() == 0 and ring.overruns == 0
    finally:
        stop.set()
        if child.pid is not None:
            child.join(2)
            if child.is_alive():
                child.kill()
                child.join(2)
        ring.close()
        ring.unlink()

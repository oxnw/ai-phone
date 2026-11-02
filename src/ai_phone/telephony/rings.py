"""Single-producer / single-consumer ring of fixed-size audio frames on shared memory.

Layout: [head u64][tail u64][overruns u64][underruns u64][frames u64][frame_bytes u64][frames × frame_bytes].
The geometry lives in the header because the OS rounds the segment up to a page size, so an
attaching process cannot infer the frame count from the segment length.
Every counter has exactly one writer: the producer owns `head` and `overruns`, the consumer owns
`tail` and `underruns`. Live counters use aligned ctypes uint64 accesses: supported CPython
3.11–3.13 macOS arm64 builds use a single load/store, without struct.pack_into's intermediate
zero-fill. This is a platform-specific implementation, not a portable Python atomic API.
A full ring drops the NEW frame
(counted as an overrun) rather than overwriting the slot the consumer may be reading.

Memory ordering: a private lock acquire/release supplies an ordering fence after reading a peer
counter and before publishing our own counter, for both slot publication and slot reuse. Only the
consumer may ever write `tail` (flush included): a producer-side or third-thread flush must be
requested from the consumer (see RingMediaPort.request_flush).
"""
from __future__ import annotations

import ctypes
import struct
import sys
import threading
from multiprocessing import shared_memory

FRAME_BYTES = 640  # 320 samples × int16 at 16 kHz = 20 ms
SILENCE = bytes(FRAME_BYTES)
_U64 = struct.Struct("<Q")
_HEAD, _TAIL, _OVER, _UNDER, _FRAMES, _FBYTES = 0, 8, 16, 24, 32, 40
_HDR_SIZE = 48
_Header = ctypes.c_uint64 * 6


class FrameRing:
    def __init__(self, name: str | None, frames: int, frame_bytes: int = FRAME_BYTES, create: bool = True):
        if frames < 1 or frame_bytes < 1:
            raise ValueError("frames and frame_bytes must be >= 1")
        if sys.byteorder != "little":
            raise RuntimeError("shared-memory audio rings require a little-endian host")
        self.frames, self.frame_bytes = frames, frame_bytes
        size = _HDR_SIZE + frames * frame_bytes
        self._shm = shared_memory.SharedMemory(name=name, create=create, size=size)
        self._buf = self._shm.buf
        self._header = _Header.from_buffer(self._buf)
        self._fence_lock = threading.Lock()
        if create:
            for off in (_HEAD, _TAIL, _OVER, _UNDER):
                self._wr(off, 0)
            self._wr(_FRAMES, frames)
            self._wr(_FBYTES, frame_bytes)

    @classmethod
    def attach(cls, name: str, frames: int | None = None, frame_bytes: int | None = None) -> "FrameRing":
        """Attach to an existing ring; geometry comes from its header (arguments are only cross-checked)."""
        shm = shared_memory.SharedMemory(name=name, create=False)
        n = _U64.unpack_from(shm.buf, _FRAMES)[0]
        fb = _U64.unpack_from(shm.buf, _FBYTES)[0]
        shm.close()
        if frames is not None and frames != n:
            raise ValueError(f"ring {name} has {n} frames, caller expected {frames}")
        if frame_bytes is not None and frame_bytes != fb:
            raise ValueError(f"ring {name} has {fb}-byte frames, caller expected {frame_bytes}")
        return cls(name, n, fb, create=False)

    @property
    def name(self) -> str:
        return self._shm.name

    def _rd(self, off: int) -> int:
        return self._header[off // 8]

    def _wr(self, off: int, value: int) -> None:
        self._header[off // 8] = value

    def _fence(self) -> None:
        with self._fence_lock:
            pass

    @property
    def overruns(self) -> int:
        return self._rd(_OVER)

    @property
    def underruns(self) -> int:
        return self._rd(_UNDER)

    def available(self) -> int:
        return self._rd(_HEAD) - self._rd(_TAIL)

    # ---- producer side --------------------------------------------------------------------
    def push(self, frame: bytes) -> None:
        if len(frame) != self.frame_bytes:
            raise ValueError(f"frame must be {self.frame_bytes} bytes, got {len(frame)}")
        head = self._rd(_HEAD)
        if head - self._rd(_TAIL) >= self.frames:  # full: drop this frame, never touch the consumer's slot
            self._wr(_OVER, self._rd(_OVER) + 1)
            return
        self._fence()  # acquire the peer's tail before reusing a consumed slot
        off = _HDR_SIZE + (head % self.frames) * self.frame_bytes
        self._buf[off:off + self.frame_bytes] = frame  # data first,
        self._fence()
        self._wr(_HEAD, head + 1)  # then publish

    # ---- consumer side --------------------------------------------------------------------
    def pop(self) -> bytes | None:
        tail = self._rd(_TAIL)
        if self._rd(_HEAD) == tail:
            return None
        self._fence()
        off = _HDR_SIZE + (tail % self.frames) * self.frame_bytes
        frame = bytes(self._buf[off:off + self.frame_bytes])
        self._fence()  # finish the copy before allowing the producer to reuse this slot
        self._wr(_TAIL, tail + 1)
        return frame

    def pop_or_silence(self) -> bytes:
        f = self.pop()
        if f is None:
            self._wr(_UNDER, self._rd(_UNDER) + 1)
            return SILENCE if self.frame_bytes == FRAME_BYTES else bytes(self.frame_bytes)
        return f

    def flush(self) -> None:
        """Consumer-side only: discard everything currently queued. Never call from a producer thread."""
        head = self._rd(_HEAD)
        self._fence()
        self._wr(_TAIL, head)

    def close(self) -> None:
        self._header = None  # release the exported ctypes view before closing shared memory
        self._buf = None
        self._shm.close()

    def unlink(self) -> None:
        try:
            self._shm.unlink()
        except FileNotFoundError:
            pass

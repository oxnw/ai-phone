"""float32 audio → int16 20 ms playback frames, carrying partial frames across feeds."""
from __future__ import annotations

import numpy as np
import soxr

FRAME_BYTES = 640  # 320 int16 samples at 16 kHz = 20 ms; must equal telephony.rings.FRAME_BYTES


class Framer:
    def __init__(self, src_rate: int, dst_rate: int = 16000):
        if type(dst_rate) is not int or dst_rate not in (16000, 48000):
            raise ValueError("playback rate must be 16000 or 48000 Hz")
        self.src_rate, self.dst_rate = src_rate, dst_rate
        self.frame_bytes = dst_rate // 50 * 2
        self._rs = soxr.ResampleStream(src_rate, dst_rate, 1, dtype="float32", quality="VHQ") if src_rate != dst_rate else None
        self._carry = b""

    @staticmethod
    def _to_bytes(f32: np.ndarray) -> bytes:
        return (np.clip(np.asarray(f32, np.float32).reshape(-1), -1.0, 1.0) * 32767).astype("<i2").tobytes()

    def _frames_from_carry(self) -> list[bytes]:
        n = len(self._carry) // self.frame_bytes
        out = [self._carry[i * self.frame_bytes:(i + 1) * self.frame_bytes] for i in range(n)]
        self._carry = self._carry[n * self.frame_bytes:]
        return out

    def feed(self, f32: np.ndarray) -> list[bytes]:
        x = np.asarray(f32, np.float32).reshape(-1)
        if self._rs is not None:
            x = self._rs.resample_chunk(x)
        self._carry += self._to_bytes(x)
        return self._frames_from_carry()

    def flush(self) -> list[bytes]:
        """Drain the resampler tail (without re-feeding it) and pad the final partial frame with silence."""
        if self._rs is not None:
            self._carry += self._to_bytes(self._rs.resample_chunk(np.zeros(0, np.float32), last=True))
        out = self._frames_from_carry()
        if self._carry:
            out.append(self._carry + bytes(self.frame_bytes - len(self._carry)))
            self._carry = b""
        return out

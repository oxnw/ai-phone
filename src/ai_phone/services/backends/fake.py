"""Deterministic fake backends for tests: no models, no network."""
from __future__ import annotations

import math
import time

import numpy as np


class FakeSTT:
    def __init__(self):
        self._n = 0
        self._buf: dict[int, list] = {}
        self._cancelled: set[int] = set()

    def warmup(self):
        return {"warm_ms": 1.0}

    def push(self, gen_id, pcm):
        self._buf.setdefault(gen_id, []).append(pcm)
        return "fake"

    def finalize(self, gen_id):
        self._n += 1
        self._buf.pop(gen_id, None)
        return f"fake utterance {self._n}", {"stt_ms": 1.0}

    def cancel(self, gen_id):
        self._cancelled.add(gen_id)
        self._buf.pop(gen_id, None)


class FakeLLM:
    REPLY = "Hello, this is a fake reply."
    TOKENS = ["Hello", ",", " this", " is", " a", " fake", " reply", "."]

    def __init__(self, reply: str | None = None):
        self._cancelled: set[int] = set()
        self.tokens = [reply] if reply is not None else self.TOKENS

    def warmup(self, system_prompt: str = ""):
        return {"warm_ms": 1.0}

    def stream(self, gen_id, messages, max_tokens):
        for tok in self.tokens:
            if gen_id in self._cancelled:
                return
            time.sleep(0.01)
            yield tok

    def cancel(self, gen_id):
        self._cancelled.add(gen_id)


class FlakyLLM(FakeLLM):
    """Fails the listed generations (1-based call numbers) with a recoverable BrainError (tests the brain_error path)."""

    def __init__(self, fail_calls: str = "2"):
        super().__init__()
        self._calls = 0
        self._fail = {int(x) for x in fail_calls.split(",") if x.strip()}

    def stream(self, gen_id, messages, max_tokens):
        from ai_phone.services.errors import BrainError
        self._calls += 1
        if self._calls in self._fail:
            raise BrainError("http_503")
        yield from super().stream(gen_id, messages, max_tokens)


class StallLLM(FakeLLM):
    """Says nothing for `stall_s` seconds (a paged-out or hung model), then replies normally unless cancelled."""

    def __init__(self, stall_s: str = "3"):
        super().__init__()
        self._stall = float(stall_s)

    def stream(self, gen_id, messages, max_tokens):
        end = time.monotonic() + self._stall
        while time.monotonic() < end:
            if gen_id in self._cancelled:
                return
            time.sleep(0.05)
        yield from super().stream(gen_id, messages, max_tokens)


class EndingLLM(FakeLLM):
    """Says goodbye and asks to end the call, with the marker split over tokens as a real model may do."""
    TOKENS = ["Goodbye", ",", " thanks", ".", " [", "END", "]"]


class PartialFailLLM(FakeLLM):
    """Streams the start of a reply (with a stray end marker), then fails with a recoverable BrainError."""
    TOKENS = ["Sure", ",", " one", " moment", ".", " [end]"]

    def stream(self, gen_id, messages, max_tokens):
        from ai_phone.services.errors import BrainError
        yield from super().stream(gen_id, messages, max_tokens)
        raise BrainError("http_503")


class SlowSTT(FakeSTT):
    """Takes `delay_s` to finalise (a slow or swapping host), so speech can arrive while a transcript is pending."""

    def __init__(self, delay_s: str = "2"):
        super().__init__()
        self._delay = float(delay_s)

    def finalize(self, gen_id):
        time.sleep(self._delay)
        return super().finalize(gen_id)


class KeepWarmProbeSTT(FakeSTT):
    """Counts keep_warm ticks into a file the test can read (the worker runs in another process)."""

    def __init__(self, path: str = ""):
        super().__init__()
        self._path = path

    def keep_warm(self):
        with open(self._path, "a") as f:
            f.write("tick\n")


class FakeTTS:
    def __init__(self, output_rate: int = 16000):
        if type(output_rate) is not int or output_rate not in (16000, 48000):
            raise ValueError("playback rate must be 16000 or 48000 Hz")
        self.output_rate = output_rate
        self._cancelled: set[int] = set()

    def warmup(self):
        return {"warm_ms": 1.0}

    def stream(self, gen_id, seq, text):
        samples_per_frame = self.output_rate // 50
        for i in range(25):
            if gen_id in self._cancelled:
                return
            t = (np.arange(samples_per_frame) + i * samples_per_frame) / self.output_rate
            yield (np.sin(2 * math.pi * 440 * t) * 8000).astype("<i2").tobytes()

    def cancel(self, gen_id):
        self._cancelled.add(gen_id)


class CountingSTT(FakeSTT):
    """Reports how many samples it received, as text: "<n> samples"."""

    def __init__(self):
        super().__init__()
        self._count: dict[int, int] = {}

    def push(self, gen_id, pcm):
        self._count[gen_id] = self._count.get(gen_id, 0) + len(pcm)
        return None

    def finalize(self, gen_id):
        n = self._count.pop(gen_id, 0)
        return f"{n} samples", {"stt_ms": 1.0}


class EnvProbeLLM(FakeLLM):
    """Replies with the names of any line-password variables visible in the worker's environment (tests)."""

    def stream(self, gen_id, messages, max_tokens):
        import os
        from ai_phone.services.client import is_secret_env
        leaked = sorted(n for n in os.environ if is_secret_env(n))
        yield ",".join(leaked) or "clean"

"""Kokoro-82M via mlx-audio, emitting 20 ms playback frames from native 24 kHz speech."""
from __future__ import annotations

import time

import numpy as np

from ai_phone import voices
from ai_phone.pipeline.audio_out import Framer


class KokoroTTS:
    def __init__(self, model_id: str = "mlx-community/Kokoro-82M-bf16", voice: str = "af_heart", speed: float = 1.0,
                 output_rate: int = 16000):
        if type(output_rate) is not int or output_rate not in (16000, 48000):
            raise ValueError("playback rate must be 16000 or 48000 Hz")
        self.model_id, self.voice, self.speed = model_id, voice, speed
        self.output_rate = output_rate
        self._voice_ref = voice
        self._model = None
        self._mx = None
        self._cancelled: set[int] = set()
        self.last_timings: dict = {}

    def warmup(self, wire_memory: bool = True) -> dict:
        import mlx.core as mx
        from mlx_audio.tts.utils import load_model

        from ai_phone.services.backends.mlx_memory import begin_wiring, wire_resident

        self._mx = mx
        t = time.monotonic_ns()
        path = voices.voice_path(self.voice)
        # a file path loads directly; a bare name makes mlx-audio look in another repo (and download there)
        self._voice_ref = str(path) if path is not None else self.voice
        begin_wiring(wire_memory)
        self._model = load_model(self.model_id)
        wired = wire_resident(self._model, wire_memory)
        list(self.stream(0, 0, "Warm up."))
        return {"warm_ms": (time.monotonic_ns() - t) / 1e6, **wired}

    def keep_warm(self) -> None:
        saved = self.last_timings
        for _ in self.stream(-1, 0, "Hi."):  # frames are discarded: the worker only pushes frames for real requests
            pass
        self.last_timings = saved

    def _to_np(self, x) -> np.ndarray:
        return np.array(self._mx.array(x).astype(self._mx.float32)).reshape(-1)  # bf16-safe conversion

    LEAD_IN_THRESHOLD = 200  # int16 units; Kokoro starts every utterance with ~200 ms of digital silence

    def stream(self, gen_id: int, seq: int, text: str):
        framer = Framer(24000, self.output_rate)
        t0 = time.monotonic_ns()
        first = None
        speech_started = False
        for res in self._model.generate(text=text, voice=self._voice_ref, speed=self.speed, lang_code=voices.lang_code(self.voice)):
            if gen_id in self._cancelled:
                break
            # Kokoro returns a whole sentence per result, so cancellation is checked per frame, not per result
            for frame in framer.feed(self._to_np(res.audio)):
                if gen_id in self._cancelled:
                    break
                if not speech_started:
                    # drop the leading silence: it is pure latency in the caller's ear
                    if np.abs(np.frombuffer(frame, dtype="<i2")).max() < self.LEAD_IN_THRESHOLD:
                        continue
                    speech_started = True
                if first is None:
                    first = time.monotonic_ns()
                yield frame
        if gen_id not in self._cancelled:
            for frame in framer.flush():
                yield frame
        self.last_timings = {"first_chunk_ms": ((first or time.monotonic_ns()) - t0) / 1e6}
        self._cancelled.discard(gen_id)

    def cancel(self, gen_id: int) -> None:
        self._cancelled.add(gen_id)

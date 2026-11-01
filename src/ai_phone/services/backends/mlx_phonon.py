"""Phonon-31M via direct MLX, streamed as int16 20 ms playback frames."""
from __future__ import annotations

import time

import numpy as np

from ai_phone import models, voices
from ai_phone.pipeline.audio_out import Framer


class PhononTTS:
    native_rate = 24000
    LEAD_IN_THRESHOLD = 200

    def __init__(self, model_id: str = models.PHONON, voice: str = voices.PHONON_DEFAULT, speed: float = 1.0,
                 output_rate: int = 16000):
        if model_id != models.PHONON:
            raise ValueError("Phonon requires the pinned gradium/phonon-31m-en model")
        if voice != voices.PHONON_DEFAULT or speed != 1.0:
            raise ValueError("Phonon requires voice='default' and speed=1.0")
        if type(output_rate) is not int or output_rate not in (16000, 48000):
            raise ValueError("playback rate must be 16000 or 48000 Hz")
        self.model_id, self.voice, self.speed = model_id, voice, speed
        self.output_rate = output_rate
        self._model = None
        self._mx = None
        self._voice_ref = None
        self._cancelled: set[int] = set()
        self.last_timings: dict = {}

    def warmup(self, wire_memory: bool = True) -> dict:
        # Local paths prevent both Hub and Pocket TTS's separate voice/tokenizer download paths.
        path = models.local_path(self.model_id)
        if path is None:
            raise models.ModelsError("Phonon weights, tokenizer or default voice are missing; run ai-phone models pull")
        voice = voices.voice_path(self.voice, path, engine="phonon")
        if voice is None:
            raise models.ModelsError("Phonon's default voice is missing; run ai-phone models pull")

        import mlx.core as mx
        from mlx_audio.tts.utils import load_model

        from ai_phone.services.backends.mlx_memory import begin_wiring, wire_resident

        self._mx = mx
        # Do not resolve this symlink: the .safetensors suffix tells upstream this is an embedding, not audio.
        self._voice_ref = str(voice.absolute())
        t0 = time.monotonic_ns()
        begin_wiring(wire_memory)
        self._model = load_model(path)
        wired = wire_resident(self._model, wire_memory)
        list(self.stream(0, 0, "Warm up."))
        return {"warm_ms": (time.monotonic_ns() - t0) / 1e6, **wired}

    def keep_warm(self) -> None:
        saved = self.last_timings
        try:
            for _ in self.stream(-1, 0, "Hi."):
                pass
        finally:
            self.last_timings = saved

    def stream(self, gen_id: int, seq: int, text: str):
        if gen_id in self._cancelled:
            return
        framer = Framer(self.native_rate, self.output_rate)
        t0 = time.monotonic_ns()
        first = None
        speech_started = False
        source = self._model.generate(text=text, voice=self._voice_ref, stream=True,
                                      streaming_interval=0.08, temperature=0.3)

        def audible(frame: bytes) -> bool:
            nonlocal speech_started, first
            if not speech_started:
                if np.abs(np.frombuffer(frame, dtype="<i2").astype(np.int32)).max() < self.LEAD_IN_THRESHOLD:
                    return False
                speech_started = True
            if first is None:
                first = time.monotonic_ns()
            return True

        try:
            while gen_id not in self._cancelled:
                try:
                    result = next(source)
                except StopIteration:
                    break
                if gen_id in self._cancelled:
                    return
                audio = np.array(self._mx.array(result.audio).astype(self._mx.float32)).reshape(-1)
                if not audio.size:  # the upstream final result can carry timings with no remaining samples
                    continue
                for frame in framer.feed(audio):
                    if gen_id in self._cancelled:
                        return
                    if audible(frame):
                        yield frame
            if gen_id not in self._cancelled:
                for frame in framer.flush():
                    if gen_id in self._cancelled:
                        return
                    if audible(frame):
                        yield frame
        finally:
            close = getattr(source, "close", None)
            if close is not None:
                close()
            self.last_timings = {"first_chunk_ms": ((first or time.monotonic_ns()) - t0) / 1e6}

    def cancel(self, gen_id: int) -> None:
        self._cancelled.add(gen_id)

import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from ai_phone import models
from ai_phone.pipeline.audio_out import Framer
from ai_phone.services.backends.mlx_phonon import PhononTTS


def tone(samples=4800):
    return (0.4 * np.sin(2 * np.pi * 440 * np.arange(samples) / 24000)).astype(np.float32)


class Speech:
    def __init__(self, parts):
        self.parts = parts
        self.calls = []
        self.read = 0
        self.closed = 0

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        try:
            for audio in self.parts:
                self.read += 1
                yield types.SimpleNamespace(audio=audio)
        finally:
            self.closed += 1


def backend(parts, output_rate=16000):
    tts = PhononTTS(output_rate=output_rate)
    tts._mx = types.SimpleNamespace(array=np.array, float32=np.float32)
    tts._model = Speech(parts)
    tts._voice_ref = "/local/default-voice.safetensors"
    return tts


@pytest.mark.parametrize("output_rate", [16000, 48000])
def test_stream_resamples_and_frames_chunks_including_empty_final_result(output_rate):
    parts = [tone(1920), tone(1920), np.array([], np.float32)]
    tts = backend(parts, output_rate)
    frames = list(tts.stream(1, 0, "Hello."))
    f = Framer(24000, output_rate)
    expected = f.feed(parts[0]) + f.feed(parts[1]) + f.flush()
    assert frames == expected
    assert frames and all(len(frame) == output_rate // 50 * 2 for frame in frames)
    assert tts._model.calls == [{"text": "Hello.", "voice": "/local/default-voice.safetensors",
                                 "stream": True, "streaming_interval": 0.08, "temperature": 0.3}]
    assert tts._model.closed == 1
    assert tts.last_timings["first_chunk_ms"] >= 0


def test_leading_silence_and_an_all_silent_resampler_tail_are_not_emitted():
    assert list(backend([np.zeros(100, np.float32)]).stream(1, 0, "")) == []
    frames = list(backend([np.zeros(4800, np.float32), tone()]).stream(2, 0, "Hello."))
    assert frames
    assert np.abs(np.frombuffer(frames[0], dtype="<i2").astype(np.int32)).max() >= 200


@pytest.mark.parametrize("output_rate", [16000, 48000])
def test_cancelled_generation_does_not_start_model_or_a_later_clause(output_rate):
    tts = backend([tone()], output_rate)
    tts.cancel(7)
    assert list(tts.stream(7, 0, "Cancelled.")) == []
    assert list(tts.stream(7, 1, "Late queued clause.")) == []
    assert tts._model.calls == []
    assert list(tts.stream(8, 0, "New reply."))


@pytest.mark.parametrize("output_rate", [16000, 48000])
def test_cancel_between_frames_drops_buffered_frames_and_does_not_pull_next_chunk(output_rate):
    tts = backend([tone(), tone()], output_rate)
    stream = tts.stream(7, 0, "A long reply.")
    assert len(next(stream)) == output_rate // 50 * 2
    tts.cancel(7)
    assert list(stream) == []
    assert tts._model.read == 1
    assert tts._model.closed == 1
    assert list(tts.stream(7, 1, "Stale reply.")) == []
    assert list(tts.stream(8, 0, "Next reply."))


def test_closing_consumer_releases_upstream_generator_and_sets_timings():
    tts = backend([tone(), tone()])
    stream = tts.stream(7, 0, "A long reply.")
    next(stream)
    stream.close()
    assert tts._model.read == 1
    assert tts._model.closed == 1
    assert tts.last_timings["first_chunk_ms"] >= 0


def test_cancel_during_resampler_tail_stops_remaining_tail(monkeypatch):
    from ai_phone.services.backends import mlx_phonon

    class Tail:
        def __init__(self, rate, output_rate):
            pass

        def feed(self, audio):
            return []

        def flush(self):
            return [b"\x00\x10" * 320] * 3

    monkeypatch.setattr(mlx_phonon, "Framer", Tail)
    tts = backend([tone(100)])
    stream = tts.stream(7, 0, "Short.")
    next(stream)
    tts.cancel(7)
    assert list(stream) == []


def test_warmup_refuses_missing_local_assets_before_importing_the_model(monkeypatch):
    monkeypatch.setattr(models, "local_path", lambda repo: None)
    with pytest.raises(models.ModelsError, match="weights, tokenizer or default voice are missing"):
        PhononTTS().warmup()


@pytest.mark.parametrize("offline", [None, "0", "1"])
def test_warmup_loads_local_paths_preserves_voice_suffix_and_setup_environment(tmp_path, monkeypatch, offline):
    from ai_phone.services.backends import mlx_memory

    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if offline is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, offline)

    voice_blob = tmp_path / "extensionless-blob"
    voice_blob.write_bytes(b"voice")
    voice = tmp_path / "default-voice.safetensors"
    voice.symlink_to(voice_blob)
    monkeypatch.setattr(models, "local_path", lambda repo: tmp_path if repo == models.PHONON else None)
    model = Speech([tone()])
    loaded = []

    def load(path):
        assert isinstance(path, Path) and path == tmp_path
        loaded.append(path)
        return model

    mx = types.ModuleType("mlx.core")
    mx.array, mx.float32 = np.array, np.float32
    mlx = types.ModuleType("mlx")
    mlx.core = mx
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    monkeypatch.setitem(sys.modules, "mlx_audio.tts.utils", types.SimpleNamespace(load_model=load))
    monkeypatch.setattr(mlx_memory, "begin_wiring", lambda enabled: None)
    monkeypatch.setattr(mlx_memory, "wire_resident", lambda model, enabled: {})
    tts = PhononTTS()
    assert tts.warmup()["warm_ms"] >= 0
    assert loaded == [tmp_path]
    assert model.calls[0]["voice"] == str(voice)
    assert model.calls[0]["voice"] != str(voice.resolve())
    assert os.environ.get("HF_HUB_OFFLINE") == offline
    assert os.environ.get("TRANSFORMERS_OFFLINE") == offline


@pytest.mark.parametrize("kwargs", [{"voice": "af_heart"}, {"speed": 1.1}, {"model_id": "org/other"},
                                    {"output_rate": 44100}, {"output_rate": 16000.0}])
def test_unsupported_model_voice_or_speed_is_refused(kwargs):
    with pytest.raises(ValueError):
        PhononTTS(**kwargs)

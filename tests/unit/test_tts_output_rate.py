"""TTS output geometry without loading model weights or importing MLX."""
from types import SimpleNamespace

import numpy as np
import pytest

from ai_phone.pipeline.audio_out import Framer
from ai_phone.services.backends.fake import FakeTTS
from ai_phone.services.backends.mlx_kokoro import KokoroTTS


@pytest.mark.parametrize("rate", [16000, 48000])
def test_kokoro_emits_the_configured_rate_without_changing_speech_duration(rate):
    time = np.arange(12000) / 24000
    source = (0.3 * np.sin(2 * np.pi * 1000 * time) + 0.3 * np.sin(2 * np.pi * 10000 * time)).astype(np.float32)
    tts = KokoroTTS(output_rate=rate)
    tts._mx = SimpleNamespace(array=np.array, float32=np.float32)
    tts._model = SimpleNamespace(generate=lambda **kwargs: iter([SimpleNamespace(audio=source)]))
    frames = list(tts.stream(1, 0, "Example."))
    control = Framer(24000, rate)
    expected = control.feed(source) + control.flush()
    assert frames == expected and all(len(frame) == rate // 50 * 2 for frame in frames)
    assert len(frames) == 25
    if rate == 48000:
        pcm = np.frombuffer(b"".join(frames), dtype="<i2").astype(np.float64) / 32768
        assert np.sqrt(np.mean(pcm[4800:-4800] ** 2)) > 0.29


@pytest.mark.parametrize("rate", [16000, 48000])
def test_fake_tts_keeps_its_frequency_and_half_second_duration_at_both_rates(rate):
    frames = list(FakeTTS(output_rate=rate).stream(1, 0, "Example."))
    assert len(frames) == 25 and all(len(frame) == rate // 50 * 2 for frame in frames)
    pcm = np.frombuffer(b"".join(frames), dtype="<i2").astype(np.float64)
    bins = np.fft.rfftfreq(len(pcm), 1 / rate)
    assert bins[np.argmax(np.abs(np.fft.rfft(pcm)))] == 440


@pytest.mark.parametrize("backend", [KokoroTTS, FakeTTS])
@pytest.mark.parametrize("rate", [24000, 44100, 48000.0, True])
def test_tts_refuses_unsupported_frame_geometry(backend, rate):
    with pytest.raises(ValueError, match="playback rate"):
        backend(output_rate=rate)

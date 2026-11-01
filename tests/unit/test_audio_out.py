import numpy as np
import pytest

from ai_phone.pipeline.audio_out import Framer


def test_framer_resamples_24k_to_16k_and_emits_640_byte_frames():
    f = Framer(24000)
    t = np.arange(24000) / 24000.0  # 1 s at 24 kHz
    frames = f.feed((0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)) + f.flush()
    assert all(len(x) == 640 for x in frames)
    assert 49 <= len(frames) <= 51  # 1 s at 16 kHz = 50 frames
    s = np.frombuffer(b"".join(frames), dtype="<i2").astype(np.float32) / 32768
    assert 0.3 < np.sqrt(np.mean(s * s)) < 0.4  # sine RMS of 0.5 amplitude ≈ 0.354


def test_framer_carries_remainder_between_feeds():
    f = Framer(24000)
    a = f.feed(np.zeros(100, np.float32))
    b = f.feed(np.zeros(500, np.float32))
    c = f.flush()
    # 600 samples at 24 kHz -> 400 at 16 kHz -> one full frame plus one padded frame; a streaming
    # resampler may hold samples back (filter delay), so only the total is invariant
    assert a == [] and len(b) + len(c) == 2 and all(len(x) == 640 for x in b + c)


def test_fullband_framer_preserves_audio_above_the_recognition_bandwidth():
    source = (0.4 * np.sin(2 * np.pi * 10000 * np.arange(44100) / 44100)).astype(np.float32)
    outputs = {}
    for rate in (16000, 48000):
        framer = Framer(44100, rate)
        frames = framer.feed(source) + framer.flush()
        assert len(frames) == 50 and all(len(frame) == rate // 50 * 2 for frame in frames)
        samples = np.frombuffer(b"".join(frames), dtype="<i2").astype(np.float64) / 32768
        # Exclude the resampler's onset/tail when checking the 10 kHz signal.
        samples = samples[rate // 10:-rate // 10]
        outputs[rate] = np.sqrt(np.mean(samples ** 2))
    assert 0.27 < outputs[48000] < 0.29
    assert outputs[16000] < 0.001


@pytest.mark.parametrize("rate", [16000, 48000])
def test_partial_feeds_and_final_padding_preserve_every_sample(rate):
    source = np.linspace(-0.9, 0.9, rate // 50 * 2 + 73, dtype=np.float32)
    framer = Framer(rate, rate)
    frames = []
    for chunk in np.array_split(source, 13):
        frames.extend(framer.feed(chunk))
    frames.extend(framer.flush())
    assert len(frames) == 3 and all(len(frame) == rate // 50 * 2 for frame in frames)
    result = np.frombuffer(b"".join(frames), dtype="<i2")
    assert np.array_equal(result[:len(source)], (source * 32767).astype("<i2"))
    assert not result[len(source):].any()


def test_fullband_resampling_is_independent_of_chunk_boundaries():
    source = np.random.default_rng(42).uniform(-0.5, 0.5, 14701).astype(np.float32)
    whole = Framer(44100, 48000)
    expected = whole.feed(source) + whole.flush()
    chunked = Framer(44100, 48000)
    actual = []
    for part in np.array_split(source, 37):
        actual.extend(chunked.feed(part))
    actual.extend(chunked.flush())
    assert all(len(frame) == 1920 for frame in actual)
    assert actual == expected


@pytest.mark.parametrize("rate", [8000, 24000, 44100, 16000.0, True])
def test_framer_refuses_unsupported_playback_geometry(rate):
    with pytest.raises(ValueError, match="playback rate"):
        Framer(24000, rate)

"""Codec policy, PCM boundaries and the clock barrier without native sockets."""
import asyncio
from collections import deque
from types import SimpleNamespace
import weakref

import pytest

from ai_phone.config import Config, ConfigError, load_config, validate_config
from ai_phone.telephony import call as call_module
from ai_phone.telephony import endpoint as module
from ai_phone.telephony import events as ev
from ai_phone.telephony.endpoint import Telephony
from ai_phone.telephony.media_port import ReceiveMediaPort, RingMediaPort


@pytest.mark.parametrize('codecs,rate', [
    (['PCMU', 'PCMA'], 16000), (['G722', 'PCMU'], 16000),
    (['opus', 'G722', 'PCMU', 'PCMA'], 48000), (['opus'], 48000),
])
def test_codec_config_roundtrip_and_playback_rate(tmp_path, codecs, rate):
    path = tmp_path / 'config.yaml'
    path.write_text('codecs: [' + ', '.join(codecs) + ']\n')
    cfg = load_config(path)
    assert cfg.codecs == codecs and cfg.playback_rate == rate


@pytest.mark.parametrize('codecs', [[], ['opus', 'opus'], ['OPUS'], ['G722/16000'], ['G729']])
def test_unsupported_empty_or_duplicate_codec_lists_fail(codecs):
    with pytest.raises(ConfigError, match='config.codecs'):
        validate_config(Config(codecs=codecs), {})


def test_codec_priorities_follow_configuration_and_disable_unselected_codecs():
    cfg = Config(codecs=['opus', 'G722', 'PCMU', 'PCMA'])
    tel = Telephony(cfg, [], None, asyncio.Queue(), None, None)
    priorities = {}
    available = ['PCMA/8000/1', 'opus/48000/2', 'GSM/8000/1', 'G722/16000/1', 'PCMU/8000/1']
    ep = SimpleNamespace(codecEnum2=lambda: [SimpleNamespace(codecId=c) for c in available],
                         codecSetPriority=lambda name, value: priorities.update({name: value}))
    tel._configure_codecs(ep)
    assert priorities == {'opus/48000/2': 254, 'G722/16000/1': 253,
                          'PCMU/8000/1': 252, 'PCMA/8000/1': 251, 'GSM/8000/1': 0}


def test_missing_selected_codec_is_an_error_instead_of_silent_downgrade():
    tel = Telephony(Config(codecs=['opus', 'PCMU']), [], None, asyncio.Queue(), None, None)
    ep = SimpleNamespace(codecEnum2=lambda: [SimpleNamespace(codecId='PCMU/8000/1')])
    with pytest.raises(RuntimeError, match='opus.*unavailable'):
        tel._configure_codecs(ep)


def test_old_native_binding_rejects_opus_before_starting_endpoint(monkeypatch):
    monkeypatch.setattr(module.pj, 'ai_phone_opus_version', lambda: 0, raising=False)
    tel = Telephony(Config(codecs=['opus', 'PCMU']), [], None, asyncio.Queue(), None, None)
    with pytest.raises(RuntimeError, match='rebuilt native binding'):
        tel.start()
    assert tel.ep is None


def test_fullband_echo_diagnostic_is_rejected_before_endpoint_start():
    tel = Telephony(Config(codecs=['opus']), [], None, asyncio.Queue(), None, None, echo_mode=True)
    with pytest.raises(ValueError, match='echo mode'):
        tel.start()
    assert tel.ep is None


class QueueRing:
    def __init__(self, frames=()):
        self.frames = deque(frames)

    def pop(self):
        return self.frames.popleft() if self.frames else None

    def flush(self):
        self.frames.clear()

    def push(self, frame):
        self.frames.append(frame)


def test_fullband_silence_does_not_mark_first_audible_and_flush_drops_old_audio():
    old, fresh = b'\x01\x00' * 960, b'\x02\x00' * 960
    ring = QueueRing([old])
    port = SimpleNamespace(tx=ring, _silence=bytes(1920), playing=False, _armed=True,
                           first_nonsilent_ns=None, _flush_requested=True)
    frame = SimpleNamespace()
    RingMediaPort.onFrameRequested(port, frame)
    assert bytes(frame.buf) == bytes(1920) and frame.size == 1920
    assert port.first_nonsilent_ns is None and not port._flush_requested
    ring.push(fresh)
    RingMediaPort.onFrameRequested(port, frame)
    assert bytes(frame.buf) == fresh and frame.size == 1920
    assert port.first_nonsilent_ns is not None and not port._armed


def test_receive_callback_keeps_fixed_recognition_frame_size():
    ring = QueueRing()
    port = SimpleNamespace(rx=ring, last_rx_ns=0)
    pcm = b'\x03\x00' * 320
    ReceiveMediaPort.onFrameReceived(port, SimpleNamespace(buf=pcm))
    assert ring.pop() == pcm and port.last_rx_ns > 0


@pytest.mark.parametrize('separate_receive_port', [False, True])
def test_media_connection_defers_logging_and_preserves_media_active_event(monkeypatch, separate_receive_port):
    queued, logs, links = [], [], []
    loop = SimpleNamespace(call_soon_threadsafe=lambda callback, *args: queued.append((callback, args)))
    events = asyncio.Queue()
    media = SimpleNamespace(startTransmit=lambda target: links.append(('receive', target)))
    playback = SimpleNamespace(startTransmit=lambda target: links.append(('playback', target)))
    receive = object() if separate_receive_port else playback
    info = SimpleNamespace(codecName='opus' if separate_receive_port else 'PCMU',
                           codecClockRate=48000 if separate_receive_port else 8000)
    expected_codec, expected_rate = info.codecName, info.codecClockRate
    call = SimpleNamespace(
        acc=SimpleNamespace(telephony=SimpleNamespace(port=playback, receive_port=receive),
                            loop=loop, events=events, line_id='test'),
        getAudioMedia=lambda index: media, getStreamInfo=lambda index: info,
        getId=lambda: 7, media_connected=False)
    monkeypatch.setattr(call_module.log, 'info', lambda *args: logs.append(args))

    call_module.AiCall._connect_media(call, 0)

    assert links == [('receive', receive), ('playback', media)]
    assert call.media_connected and events.empty() and logs == []
    # The queued callback must own primitive snapshots, not borrowed native stream information.
    info.codecName, info.codecClockRate = 'changed', 0
    for callback, args in queued:
        callback(*args)
    assert logs == [('media line=%s codec=%s rate=%d', 'test', expected_codec, expected_rate)]
    event = events.get_nowait()
    assert isinstance(event, ev.MediaActive) and event.call_id == 7 and event.line_id == 'test'
    assert events.empty()


@pytest.mark.parametrize('clock_fails', [False, True])
def test_both_media_ports_survive_until_clock_stops(clock_fails):
    tel = Telephony(Config(), [], None, asyncio.Queue(), None, None)
    class Port:
        pass
    tel.port, tel.receive_port = Port(), Port()
    tx_ref, rx_ref = weakref.ref(tel.port), weakref.ref(tel.receive_port)
    order = []
    class Endpoint:
        def audDevManager(self):
            return self
        def setNoDev(self):
            assert tx_ref() is not None and rx_ref() is not None
            if clock_fails:
                raise RuntimeError('clock failure')
            order.append('clock')
        def libDestroy(self):
            assert tx_ref() is None and rx_ref() is None
            order.append('destroy')
    tel.ep = Endpoint()
    if clock_fails:
        with pytest.raises(RuntimeError, match='clock failure'):
            tel.stop()
        assert tx_ref() is tel.port and rx_ref() is tel.receive_port and tel.ep is not None
        clock_fails = False
    tel.stop()
    assert order == ['clock', 'destroy'] and tel.ep is None

"""Real loopback negotiation/audio with independent Python native endpoints.

No Homebrew pjsua executable, models, user settings or carrier credentials are
used. Missing Opus is a failure: the installer/CI must build the required codec.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import wave

import numpy as np
import pytest

from tests.integration.ports import SIP_PORT, SECOND_SIP_PORT, rtp_range

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
HELPER = Path(__file__).with_name('fullband_peer.py')
ALL_CODECS = ('opus', 'G722', 'PCMU', 'PCMA')


def _environment(home):
    home.mkdir(mode=0o700)
    return {'PATH': os.environ.get('PATH', '/usr/bin:/bin'), 'HOME': str(home),
            'TMPDIR': str(home), 'AI_PHONE_HOME': str(home / 'settings'),
            'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
            'HF_HUB_DISABLE_TELEMETRY': '1', 'PYTHONUNBUFFERED': '1'}


def _port_free(port, kind):
    with socket.socket(socket.AF_INET, kind) as probe:
        if kind == socket.SOCK_STREAM:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(('127.0.0.1', port))


def _kill_group(process):
    # Also clean up a group whose leader crashed. Each child owns its own group.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def _audio(path):
    with wave.open(str(path), 'rb') as audio:
        assert audio.getnchannels() == 1 and audio.getsampwidth() == 2
        rate = audio.getframerate()
        pcm = np.frombuffer(audio.readframes(audio.getnframes()), dtype='<i2').astype(np.float64)
    assert len(pcm) > rate, f'{path.name} contains too little received audio'
    return pcm, rate


def _tone_amplitude(pcm, rate, hz):
    """Maximum 200 ms coherent tone amplitude, independent of transport delay."""
    count = rate // 5
    oscillator = np.exp(-2j * np.pi * hz * np.arange(count) / rate)
    return max((2 * abs(np.dot(pcm[i:i + count], oscillator)) / count
                for i in range(0, len(pcm) - count + 1, rate // 50)), default=0)


def _defaults(report):
    defaults = report['opus_defaults']
    assert defaults['clock_rate'] == 48000
    assert defaults['channels'] == 1
    assert defaults['avg_bps'] == 64000
    assert defaults['frame_len_ms'] / defaults['frame_len_denum'] * defaults['frames_per_packet'] == 20
    assert defaults['cbr'] and not defaults['vad']


def _run(tmp_path, *, direction, subject=ALL_CODECS, peer=ALL_CODECS, cycles=1, flush=False):
    assert SIP_PORT != 5060 and SECOND_SIP_PORT == SIP_PORT + 4
    for port in (SIP_PORT, SECOND_SIP_PORT):
        _port_free(port, socket.SOCK_DGRAM)
        _port_free(port, socket.SOCK_STREAM)
        _port_free(rtp_range(port)[0], socket.SOCK_DGRAM)
    logs, processes = {}, {}
    try:
        for role in ('peer', 'subject'):
            output = (tmp_path / f'{role}.log').open('w')
            logs[role] = output
            argv = [sys.executable, '-I', '-B', str(HELPER), '--root', str(ROOT),
                    '--directory', str(tmp_path), '--role', role, '--direction', direction,
                    '--subject-codecs', ','.join(subject), '--peer-codecs', ','.join(peer),
                    '--subject-port', str(SIP_PORT), '--peer-port', str(SECOND_SIP_PORT),
                    '--cycles', str(cycles)]
            if flush:
                argv.append('--flush')
            processes[role] = subprocess.Popen(argv, cwd=tmp_path, env=_environment(tmp_path / f'{role}-home'),
                                               stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 30 * cycles
        while any(p.poll() is None for p in processes.values()):
            if any(p.poll() not in (None, 0) for p in processes.values()):
                break
            if time.monotonic() > deadline:
                pytest.fail('native fullband children exceeded their bounded lifetime')
            time.sleep(.03)
    finally:
        for process in processes.values():
            _kill_group(process)
        for log in logs.values():
            log.close()
    details = '\n'.join(f'{role}: exit={p.returncode}\n{(tmp_path / (role + ".log")).read_text()[-12000:]}'
                        for role, p in processes.items())
    assert len(processes) == 2 and all(p.returncode == 0 for p in processes.values()), details
    for port in (SIP_PORT, SECOND_SIP_PORT):
        _port_free(port, socket.SOCK_DGRAM)
        _port_free(port, socket.SOCK_STREAM)
        _port_free(rtp_range(port)[0], socket.SOCK_DGRAM)
    reports = []
    for cycle in range(cycles):
        subject_report = json.loads((tmp_path / f'subject-{cycle}.json').read_text())
        peer_report = json.loads((tmp_path / f'peer-{cycle}.json').read_text())
        assert subject_report['stopped'] and peer_report['stopped'], details
        assert subject_report['ended'] and peer_report['ended'], details
        reports.append((subject_report, peer_report))
    return reports


def _check_audio(tmp_path, cycle, subject, peer, expected, fullband):
    # Metadata from both active native streams, rather than offered SDP alone.
    for report in (subject, peer):
        assert len(report['negotiated']) == 1
        assert report['negotiated'][0]['name'].lower() == expected.lower()
    assert subject['rx_frame_bytes'] == 640 and subject['rx_lengths'] == [640]
    assert subject['rx_port'] == {'rate': 16000, 'channels': 1, 'bits': 16, 'frame_usec': 20000}
    assert subject['rx_frames'] > 50 and subject['rx_overruns'] == 0
    assert peer['unexpected_capture_frames'] == [], peer['unexpected_capture_frames']
    assert peer['capture_lengths'] == [1920], 'fullband recorder received incorrectly sized native audio'
    assert peer['capture_port'] == {'rate': 48000, 'channels': 1, 'bits': 16, 'frame_usec': 20000}
    incoming, rx_rate = _audio(tmp_path / f'subject-rx-{cycle}.wav')
    captured, capture_rate = _audio(tmp_path / f'peer-rx-{cycle}.wav')
    assert rx_rate == 16000 and capture_rate == 48000
    assert _tone_amplitude(incoming, rx_rate, 2000) > 500, 'speech-band return audio did not reach fixed-rate STT ring'
    assert _tone_amplitude(captured, capture_rate, 1000) > 500, 'remote recording is silent or disconnected'
    high = _tone_amplitude(captured, capture_rate, 12000)
    assert high > 300 if fullband else high < 100, f'12 kHz marker amplitude={high}'
    assert subject['tx_overruns'] == 0
    return captured


@pytest.mark.parametrize('direction', ['out', 'in'])
def test_opus_transports_fullband_both_directions_and_keeps_stt_16k(tmp_path, direction):
    cycles = 2 if direction == 'out' else 1
    reports = _run(tmp_path, direction=direction, cycles=cycles, flush=True)
    for cycle, (subject, peer) in enumerate(reports):
        _defaults(subject)
        _defaults(peer)
        assert subject['playback_rate'] == 48000 and subject['tx_frame_bytes'] == 1920
        assert subject['separate_receive_port']
        assert subject['tx_port']['rate'] == 48000
        captured = _check_audio(tmp_path, cycle, subject, peer, 'opus', True)
        assert subject['negotiated'][0]['clock_rate'] == 48000
        assert subject['negotiated'][0]['channels'] == 1
        assert subject['queued_before_flush'] >= 50 and subject['queued_after_flush'] == 0
        assert (subject['flush_ack_ns'] - subject['flush_requested_ns']) / 1e9 < .5
        # The peer's callback timestamps and subject's monotonic timestamps share
        # this host's clock. Wait 250 ms for already transmitted RTP/jitter data.
        frames = captured.reshape(-1, 960)
        stamps = np.asarray(peer['capture_times_ns'], dtype=np.int64)
        assert len(frames) == len(stamps)
        quiet = frames[(stamps >= subject['flush_ack_ns'] + 250_000_000) &
                       (stamps <= subject['fresh_requested_ns'] - 100_000_000)].reshape(-1)
        assert len(quiet) >= 48000 // 3
        assert np.sqrt(np.mean(quiet ** 2)) < 100, 'cancelled queued audio leaked into the quiet interval'
        fresh = frames[(stamps >= subject['fresh_requested_ns'] + 250_000_000) &
                       (stamps <= subject['fresh_requested_ns'] + 650_000_000)].reshape(-1)
        assert _tone_amplitude(fresh, 48000, 12000) > 300, 'fresh fullband audio was discarded or zeroed after flush'


@pytest.mark.parametrize('codec', ['G722', 'PCMU', 'PCMA'])
def test_fullband_offer_falls_back_without_changing_receive_geometry(tmp_path, codec):
    [(subject, peer)] = _run(tmp_path, direction='out', peer=(codec,))
    _defaults(subject)
    assert subject['playback_rate'] == 48000 and subject['tx_frame_bytes'] == 1920
    assert subject['separate_receive_port']
    _check_audio(tmp_path, 0, subject, peer, codec, False)


def test_default_g711_keeps_existing_16k_duplex_port(tmp_path):
    [(subject, peer)] = _run(tmp_path, direction='in', subject=('PCMU', 'PCMA'), peer=('PCMU',))
    assert subject['playback_rate'] == 16000 and subject['tx_frame_bytes'] == 640
    assert not subject['separate_receive_port']
    assert subject['tx_port']['rate'] == 16000
    _check_audio(tmp_path, 0, subject, peer, 'PCMU', False)

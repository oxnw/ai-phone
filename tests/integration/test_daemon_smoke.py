"""End-to-end with fake model backends: the daemon answers a LAN call and writes a turn record.

The helpers here (_tone, _start_daemon, _call_and_wait) are shared with the other daemon integration tests."""
import json
import math
import os
import queue
import signal
import struct
import subprocess
import threading
import time
import wave
from pathlib import Path

import pytest
from tests.integration.ports import SIP_PORT, sip_config  # noqa: E402

pytestmark = pytest.mark.integration


def _tone(path: Path, seconds=1.2, rate=8000, tail_silence=3.0, turns=1, lead_silence=0.0):
    """`turns` utterances, each `lead_silence` of silence, a tone, then `tail_silence` of silence: the endpointer needs
    silence after speech to close a turn, and a lead lets the other side finish its greeting first."""
    tone = b"".join(struct.pack("<h", int(12000 * math.sin(2 * math.pi * 300 * i / rate))) for i in range(int(seconds * rate)))
    one = bytes(2 * int(lead_silence * rate)) + tone + bytes(2 * int(tail_silence * rate))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(one * turns)


def _start_daemon(home: Path, cfg_text: str, extra_args=(), port=SIP_PORT):
    cfg = home / "config.yaml"
    cfg.write_text(sip_config(port) + cfg_text)  # the suite's ports, so it can run beside a live daemon (AI_PHONE_TEST_SIP_PORT)
    # start_new_session: `uv run` is a parent of the real daemon; killing only uv would leave the daemon (and port 5060) behind
    d = subprocess.Popen(["uv", "run", "ai-phone", "run", "--config", str(cfg), "--lan", "--fake", *extra_args],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env={**os.environ, "AI_PHONE_HOME": str(home)},
                         start_new_session=True)
    lines: queue.Queue = queue.Queue()
    threading.Thread(target=lambda: [lines.put(l) for l in d.stdout], daemon=True).start()  # readline would block forever
    seen = []  # everything read while waiting, so a failure message can show it
    deadline = time.time() + 90
    while time.time() < deadline:
        try:
            line = lines.get(timeout=1)
        except queue.Empty:
            if d.poll() is not None:
                break
            continue
        seen.append(line)
        if line.strip() == "ready":
            return d, lines
    time.sleep(0.5)
    while not lines.empty():
        seen.append(lines.get_nowait())
    _kill_group(d)
    raise AssertionError("daemon never became ready; output tail:\n" + "".join(seen)[-3000:])


def _kill_group(d):
    try:
        os.killpg(os.getpgid(d.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        d.wait(timeout=5)  # reap; a killed group still needs its parent to collect it
    except subprocess.TimeoutExpired:
        pass


def _stop_daemon(d):
    try:
        os.killpg(os.getpgid(d.pid), signal.SIGTERM)  # the daemon and uv alike
    except ProcessLookupError:
        return
    try:
        d.wait(15)
    except subprocess.TimeoutExpired:
        _kill_group(d)


def _call_and_wait(d, lines: queue.Queue, home: Path, turns: int = 1):
    """One LAN call from pjsua playing `turns` utterances; returns the call's turn records once `turns` are written."""
    play, rec = home / "tone.wav", home / "rec.wav"
    _tone(play, turns=turns)
    duration = 4 + 5 * turns
    peer = subprocess.Popen(["bash", "scripts/pjsua_peer.sh", str(play), str(rec), str(duration)],
                            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    turns_path = home / "turns.jsonl"
    deadline = time.time() + duration + 15
    try:
        while time.time() < deadline:
            if turns_path.exists() and len(turns_path.read_text().splitlines()) >= turns:
                break
            time.sleep(0.5)
    finally:
        try:
            peer.stdin.write(b"q\n")
            peer.stdin.flush()
            peer.wait(timeout=10)
        except Exception:
            peer.terminate()
            try:
                peer.wait(timeout=5)
            except Exception:
                peer.kill()
    time.sleep(1.0)
    recs = [json.loads(l) for l in turns_path.read_text().splitlines() if l.strip()] if turns_path.exists() else []
    complete = [r for r in recs if not r.get("incomplete")]
    # exactly `turns` complete turns, plus at most one incomplete tail record written when the peer hung up mid-turn
    if len(complete) != turns or len(recs) > turns + 1 or any(r.get("incomplete") for r in recs[:turns]):
        tail = []
        while not lines.empty():
            tail.append(lines.get_nowait())
        raise AssertionError(f"expected exactly {turns} complete turn records, got {recs}; daemon output tail:\n" + "".join(tail)[-3000:])
    return recs


CFG = ('inbound:\n  greeting: "Hi."\n  instructions: "s"\n'
       'conversation:\n  endpoint_silence_ms: 300\nlog:\n  dir: "%s"\n')


@pytest.mark.parametrize("codecs", ["[PCMU, PCMA]", "[opus, G722, PCMU, PCMA]"])
def test_daemon_answers_and_logs_a_turn_with_fake_backends(tmp_path: Path, codecs):
    # The independent pjsua peer uses PCMU: the Opus offer exercises 48 kHz
    # worker output, cached greeting and playback through negotiated fallback.
    d, lines = _start_daemon(tmp_path, f"codecs: {codecs}\n" + CFG % tmp_path)
    try:
        recs = _call_and_wait(d, lines, tmp_path, turns=1)
        assert recs[0]["reply_text"] == "Hello, this is a fake reply." and recs[0]["first_audible_ms"] is not None
        assert recs[0]["line_id"] == "local" and recs[0]["direction"] == "in"
        state = json.loads((tmp_path / "state.json").read_text())
        assert state["lines"]["local"]["kind"] == "sip" and state["brain"]["kind"] == "local"
    finally:
        _stop_daemon(d)

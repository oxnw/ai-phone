"""A stray end marker must not end a real SIP call before the caller is finished.

The daemon, workers and RTP path are real. Only the model factories in the child
daemon are replaced; the scripted backends never load models or use an account.
"""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import wave

import numpy as np
import pytest

from ai_phone import paths
from ai_phone.control.client import request
from tests.integration.ports import PEER_PORT, SIP_PORT, rtp_range
from tests.integration.test_agent_modes import CODECS, LINE, home
from tests.integration.test_daemon_smoke import _tone
from tests.integration.test_incoming_cancel import _cleanup, _wait

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]

# Spawned workers import this temporary module using its factory names, so no
# production backend or global pytest process needs patching.
BACKENDS = '''\
import json
from pathlib import Path

import numpy as np

from ai_phone.services.backends.fake import FakeLLM, FakeSTT, FakeTTS


class ScriptedSTT(FakeSTT):
    def finalize(self, gen_id):
        _, timings = super().finalize(gen_id)
        text = {1: "Hello, I need some help.",
                2: "No, that is all. Goodbye."}[self._n]
        return text, timings


class ScriptedLLM(FakeLLM):
    def __init__(self):
        super().__init__()
        self.turn = 0

    def stream(self, gen_id, messages, max_tokens):
        self.turn += 1
        self.tokens = (["How can", " I help", " you today?", " [", "EN", "D]"]
                       if self.turn == 1 else ["Thanks", ", goodbye.", " [E", "ND", "]"])
        yield from super().stream(gen_id, messages, max_tokens)


class ObservedTTS(FakeTTS):
    def stream(self, gen_id, seq, text):
        with Path(__file__).with_name("tts-input.jsonl").open("a") as log:
            log.write(json.dumps({"gen": gen_id, "seq": seq, "text": text}) + "\\n")
        # The farewell has a distinct pitch, so the received WAV proves it made
        # it through the real audio path before the agent sent BYE.
        hz = 880 if "goodbye" in text.lower() else 440
        for i in range(25):
            if gen_id in self._cancelled:
                return
            t = (np.arange(320) + i * 320) / 16000.0
            yield (np.sin(2 * np.pi * hz * t) * 8000).astype("<i2").tobytes()
'''

CHILD = '''\
import os
import resource
import sys


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if sys.argv[1] == "peer":
        os.execvp(sys.argv[2], sys.argv[2:])
    from ai_phone.daemon import FACTORIES
    FACTORIES["fake"]["stt"] = "call_ending_backends:ScriptedSTT"
    FACTORIES["fake"]["llm"]["local"] = "call_ending_backends:ScriptedLLM"
    FACTORIES["fake"]["tts"] = "call_ending_backends:ObservedTTS"
    from ai_phone.cli import main as run
    return run(["run", "--config", sys.argv[1], "--lan", "--fake"])


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _json_lines(path):
    if not path.exists():
        return []
    # A line may be in the middle of a write while the daemon is running.
    return [json.loads(line) for line in path.read_text().splitlines(keepends=True)
            if line.endswith("\n") and line.strip()]


def _farewell_audio_seconds(path):
    with wave.open(str(path), "rb") as rec:
        assert rec.getsampwidth() == 2
        rate, channels = rec.getframerate(), rec.getnchannels()
        pcm = np.frombuffer(rec.readframes(rec.getnframes()), dtype="<i2")
    samples = pcm.reshape(-1, channels).astype(np.float64).mean(axis=1)
    frame_size = rate // 50
    frames = samples[:len(samples) // frame_size * frame_size].reshape(-1, frame_size)
    reference = np.exp(-2j * np.pi * 880 * np.arange(frame_size) / rate)
    amplitude = np.abs(frames @ reference) * 2 / frame_size
    return float(np.count_nonzero(amplitude > 1500) * frame_size / rate)


def test_stray_marker_keeps_call_open_until_a_real_farewell(tmp_path, home, monkeypatch):
    assert 5060 not in range(SIP_PORT, PEER_PORT + 1), "Set AI_PHONE_TEST_SIP_PORT=5080"
    pjsua = shutil.which("pjsua")
    if pjsua is None:
        pytest.skip("pjsua is required for the real SIP/RTP regression")
    for port in (SIP_PORT, PEER_PORT):
        for kind in (socket.SOCK_DGRAM, socket.SOCK_STREAM):
            with socket.socket(socket.AF_INET, kind) as probe:
                if kind == socket.SOCK_STREAM:
                    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(("127.0.0.1", port))

    monkeypatch.setenv("AI_PHONE_HOME", str(home))
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "PYTHONPATH": os.pathsep.join([str(tmp_path), str(ROOT / "src"), os.environ.get("PYTHONPATH", "")])}
    low, high = rtp_range(SIP_PORT)
    config = tmp_path / "config.yaml"
    config.write_text(
        f"sip:\n  bind: {{address: '127.0.0.1', udp_port: {SIP_PORT}, tcp_port: {SIP_PORT}, tls_port: {SIP_PORT + 1}}}\n"
        f"rtp: {{port_min: {low}, port_max: {high}}}\n" + LINE.format(id="agents") +
        "inbound:\n  greeting: 'Hi.'\n  instructions: 'Take a message.'\n"
        "conversation:\n  max_call_sec: 60\n  silence_prompt_sec: 20\n  silence_hangup_sec: 20\n"
        f"log:\n  dir: '{tmp_path}'\n  pjsip_level: 5\n")
    config.chmod(0o600)
    (tmp_path / "call_ending_backends.py").write_text(BACKENDS)
    runner = tmp_path / "run_call_ending.py"
    runner.write_text(CHILD)
    play, received = tmp_path / "caller.wav", tmp_path / "received.wav"
    _tone(play, seconds=0.7, lead_silence=1.5, tail_silence=4.0, turns=2)
    processes = []
    daemon_log = tmp_path / "daemon.log"
    with ExitStack() as stack:
        def start(args, log):
            proc = subprocess.Popen([sys.executable, str(runner), *args], cwd=ROOT, env=env,
                                    stdin=subprocess.PIPE, stdout=stack.enter_context(log.open("w")),
                                    stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(proc)
            return proc

        daemon = start([str(config)], daemon_log)
        peer = None

        def alive():
            assert daemon.poll() is None, f"daemon exited {daemon.returncode}"

        def control(message, timeout=5):
            alive()
            return request(paths.socket_path(), message, timeout=timeout)

        def turns():
            alive()
            return _json_lines(tmp_path / "turns.jsonl")

        try:
            _wait(lambda: (alive() is None) and "\nready\n" in daemon_log.read_text(), timeout=60)
            # No EOF hangup or duration limit: only the daemon can end this call.
            peer = start(["peer", pjsua, "--local-port", str(PEER_PORT), "--bound-addr", "127.0.0.1",
                          "--ip-addr", "127.0.0.1", "--rtp-port", str(rtp_range(PEER_PORT)[0]),
                          "--null-audio", "--play-file", str(play), "--auto-play",
                          "--rec-file", str(received), "--auto-rec", *CODECS,
                          f"sip:agents@127.0.0.1:{SIP_PORT}"], tmp_path / "peer.log")
            active = _wait(lambda: control({"cmd": "status"})["call"], timeout=10)
            call_id = active["call_id"]
            first = _wait(lambda: next((row for row in turns() if not row.get("incomplete")), None), timeout=15)
            assert first["stt_text"] == "Hello, I need some help.", first
            assert first["reply_text"] == "How can I help you today?", first
            assert first["first_audible_ms"] is not None and first["call"] == call_id, first
            # The turn record follows audio drainage. Allow the daemon's hangup
            # loop to run too, so a premature end is caught before the next tone.
            time.sleep(0.4)
            still_active = control({"cmd": "status"})
            assert still_active["call"] is not None, still_active
            assert still_active["call"]["call_id"] == call_id, still_active

            _wait(lambda: control({"cmd": "status"})["active_call"] is None, timeout=20)
            ended = control({"cmd": "listen", "call_id": call_id, "timeout_sec": 1})
            assert ended.get("ok") and ended["kind"] == "ended", ended
            result = ended["result"]
            assert result["call_id"] == call_id and result["ended_by"] == "agent", result
            assert result["outcome"] == "completed" and result["direction"] == "in", result
            rows = turns()
            assert len(rows) == 2 and all(not row.get("incomplete") for row in rows), rows
            assert all(row["call"] == call_id and row["first_audible_ms"] is not None for row in rows), rows
            assert rows[1]["stt_text"] == "No, that is all. Goodbye.", rows
            assert rows[1]["reply_text"] == "Thanks, goodbye.", rows
            assert [line["text"] for line in result["transcript"]] == [
                "Hi.", "Hello, I need some help.", "How can I help you today?",
                "No, that is all. Goodbye.", "Thanks, goodbye."], result
            spoken = [row["text"] for row in _json_lines(tmp_path / "tts-input.jsonl")]
            assert "How can I help you today?" in spoken and "Thanks, goodbye." in spoken, spoken
            assert all("[" not in text and "]" not in text for text in spoken), spoken
            assert peer.poll() is None, "peer exited before the daemon ended the call"
            time.sleep(0.2)  # receive BYE before quitting the active WAV player
            peer.stdin.write(b"q\n")
            peer.stdin.flush()
            assert peer.wait(timeout=10) == 0
            assert _farewell_audio_seconds(received) >= 0.3, "peer did not receive the farewell audio"
            daemon.send_signal(signal.SIGTERM)
            assert daemon.wait(timeout=15) == 0
            output = daemon_log.read_text()
            assert "Assertion failed" not in output and "Traceback (most recent call last)" not in output
        except Exception as error:
            error.add_note(f"SIP regression logs: {tmp_path}\n" + daemon_log.read_text()[-5000:])
            raise
        finally:
            failure = sys.exception()
            cleanup_errors = []
            # Even a failed assertion must release the call before the peer quits;
            # destroying pjsua while its WAV player is attached can abort natively.
            if peer is not None and peer.poll() is None and daemon.poll() is None:
                try:
                    state = control({"cmd": "status"})
                    if state["active_call"] is not None:
                        control({"cmd": "hangup"}, timeout=15)
                        _wait(lambda: control({"cmd": "status"})["active_call"] is None)
                    time.sleep(0.2)
                    peer.stdin.write(b"q\n")
                    peer.stdin.flush()
                    peer.wait(timeout=10)
                except Exception as error:
                    cleanup_errors.append(str(error))
            for proc in reversed(processes):
                try:
                    _cleanup(proc)
                except Exception as error:
                    cleanup_errors.append(str(error))
            if cleanup_errors:
                detail = "Test process cleanup failed: " + "; ".join(cleanup_errors)
                if failure is not None:
                    failure.add_note(detail)
                else:
                    raise AssertionError(detail)

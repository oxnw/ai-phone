import asyncio
import json
import os
import time
from pathlib import Path

import pytest

from ai_phone.services.backends.fake import FakeLLM, FakeTTS
from ai_phone.services.client import WorkerClient
from ai_phone.services.errors import BrainError
from ai_phone.services.ipc import STT_IN_FRAME, TTS_OUT_FRAME, tag_frame, untag_frame
from ai_phone.telephony.rings import FRAME_BYTES, FrameRing


async def _collect(client, until_ev, gen, timeout=10.0):
    out = []
    while True:
        e = await asyncio.wait_for(client.events.get(), timeout)
        out.append(e)
        if e.get("ev") == until_ev and e.get("gen") == gen:
            return out
        if e.get("ev") == "error":
            raise AssertionError(e)


@pytest.mark.asyncio
async def test_llm_worker_streams_tokens_and_done():
    loop = asyncio.get_running_loop()
    c = WorkerClient("llm", "ai_phone.services.backends.fake:FakeLLM", loop, None, None, {})
    t = await c.start()
    assert "warm_ms" in t
    gen = c.next_gen()
    c.send({"op": "llm.generate", "gen": gen, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 50})
    evs = await _collect(c, "done", gen)
    text = "".join(e["text"] for e in evs if e["ev"] == "token")
    assert text == "Hello, this is a fake reply."
    await c.stop()


@pytest.mark.asyncio
async def test_cancel_drops_stale_generation():
    loop = asyncio.get_running_loop()
    c = WorkerClient("llm", "ai_phone.services.backends.fake:FakeLLM", loop, None, None, {})
    await c.start()
    g1 = c.next_gen()
    c.send({"op": "llm.generate", "gen": g1, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 50})
    await asyncio.sleep(0.05)
    g2 = c.next_gen()
    c.cancel(g1)
    # tokens that arrived before next_gen() are legitimately g1 and already queued; from now on the
    # client drops anything tagged g1 on receipt, so only events after this drain matter
    while not c.events.empty():
        c.events.get_nowait()
    c.send({"op": "llm.generate", "gen": g2, "messages": [{"role": "user", "content": "again"}], "max_tokens": 50})
    evs = await _collect(c, "done", g2)
    assert all(e["gen"] == g2 for e in evs if e["ev"] in ("token", "done")), "stale g1 events leaked"
    assert not c.gen_is_current(g1) and c.gen_is_current(g2)
    await c.stop()


@pytest.mark.asyncio
async def test_tts_worker_writes_frames_to_ring():
    loop = asyncio.get_running_loop()
    out = FrameRing(None, 200, frame_bytes=TTS_OUT_FRAME)
    try:
        c = WorkerClient("tts", "ai_phone.services.backends.fake:FakeTTS", loop, None, out, {})
        await c.start()
        gen = c.next_gen()
        c.send({"op": "tts.synth", "gen": gen, "seq": 0, "text": "Hello there."})
        evs = await _collect(c, "done", gen)
        n = sum(e["nframes"] for e in evs if e["ev"] == "chunk")
        assert n == 25 and out.available() == 25
        g, payload = untag_frame(out.pop())
        assert g == gen and len(payload) == FRAME_BYTES and payload != bytes(FRAME_BYTES)
        await c.stop()
    finally:
        out.close()
        out.unlink()


class OrderedTTS(FakeTTS):
    """Immediate numbered frames make loss, reordering and generator cleanup observable across the pipe."""
    def __init__(self, closed_path=None):
        super().__init__()
        self.closed_path = closed_path

    def stream(self, gen_id, seq, text):
        try:
            for i in range(int(text)):
                if gen_id in self._cancelled:
                    return
                yield (i + 1).to_bytes(2, "little") * 320
        finally:
            if self.closed_path:
                with Path(self.closed_path).open("a") as f:
                    f.write(f"{gen_id}\n")


ORDERED_TTS = "tests.unit.test_worker_ipc:OrderedTTS"


async def _read_speech(client, ring, gen, *, delay=0.01):
    frames, announced = [], 0
    done = False
    while not done or ring.available():
        if not done:
            event = await asyncio.wait_for(client.events.get(), 5)
            assert event["ev"] != "error", event
            assert event.get("gen") == gen
            announced += event.get("nframes", 0)
            done = event["ev"] == "done"
        while (frame := ring.pop()) is not None:
            frame_gen, payload = untag_frame(frame)
            assert frame_gen == gen
            frames.append(payload)
            await asyncio.sleep(delay)
    return frames, announced


@pytest.mark.asyncio
async def test_fast_tts_waits_for_small_ring_without_losing_or_reordering_frames():
    ring = FrameRing(None, 2, frame_bytes=TTS_OUT_FRAME)  # smaller than a normal five-frame chunk notification
    c = WorkerClient("tts", ORDERED_TTS, asyncio.get_running_loop(), None, ring, {})
    try:
        await c.start()
        gen = c.next_gen()
        c.send({"op": "tts.synth", "gen": gen, "seq": 0, "text": "40"})
        frames, announced = await asyncio.wait_for(_read_speech(c, ring, gen), 8)
        assert frames == [(i + 1).to_bytes(2, "little") * 320 for i in range(40)]
        assert announced == 40 and ring.overruns == 0
    finally:
        await c.stop()
        ring.close()
        ring.unlink()


@pytest.mark.asyncio
async def test_cancel_while_tts_ring_is_full_closes_generator_and_allows_next_generation(tmp_path):
    ring = FrameRing(None, 2, frame_bytes=TTS_OUT_FRAME)
    closed = tmp_path / "closed"
    c = WorkerClient("tts", ORDERED_TTS, asyncio.get_running_loop(), None, ring, {},
                     factory_args={"closed_path": str(closed)})
    try:
        await c.start()
        gen = c.next_gen()
        c.send({"op": "tts.synth", "gen": gen, "seq": 0, "text": "1000"})
        await _collect(c, "chunk", gen)
        assert ring.available() == 2
        c.cancel(gen)
        await _collect(c, "done", gen)
        assert closed.read_text().splitlines() == [str(gen)]
        assert ring.overruns == 0
        ring.flush()
        # A stale clause queued after cancellation is ignored, and the next generation still streams.
        c.send({"op": "tts.synth", "gen": gen, "seq": 1, "text": "1000"})
        new_gen = c.next_gen()
        c.send({"op": "tts.synth", "gen": new_gen, "seq": 0, "text": "6"})
        frames, _ = await _read_speech(c, ring, new_gen)
        assert frames == [(i + 1).to_bytes(2, "little") * 320 for i in range(6)]
        assert closed.read_text().splitlines() == [str(gen), str(new_gen)]
    finally:
        await c.stop()
        ring.close()
        ring.unlink()


@pytest.mark.asyncio
async def test_shutdown_while_tts_ring_is_full_exits_without_forced_kill(tmp_path):
    ring = FrameRing(None, 2, frame_bytes=TTS_OUT_FRAME)
    closed = tmp_path / "closed"
    c = WorkerClient("tts", ORDERED_TTS, asyncio.get_running_loop(), None, ring, {},
                     factory_args={"closed_path": str(closed)})
    try:
        await c.start()
        gen = c.next_gen()
        c.send({"op": "tts.synth", "gen": gen, "seq": 0, "text": "1000"})
        await _collect(c, "chunk", gen)
        before = time.monotonic()
        await c.stop()
        assert time.monotonic() - before < 2
        assert c._proc.exitcode == 0
        assert closed.read_text().splitlines() == [str(gen)]
    finally:
        if c._proc is not None and c._proc.is_alive():
            await c.stop()
        ring.close()
        ring.unlink()


@pytest.mark.asyncio
async def test_stt_worker_push_finalize():
    loop = asyncio.get_running_loop()
    inp = FrameRing(None, 200, frame_bytes=STT_IN_FRAME)
    try:
        c = WorkerClient("stt", "ai_phone.services.backends.fake:FakeSTT", loop, inp, None, {})
        await c.start()
        gen = c.next_gen()
        for _ in range(10):
            inp.push(tag_frame(gen, b"\x01\x00" * 512))
        c.send({"op": "stt.push", "gen": gen, "nframes": 10})
        c.send({"op": "stt.finalize", "gen": gen})
        evs = await _collect(c, "final", gen)
        assert evs[-1]["text"].startswith("fake utterance")
        await c.stop()
    finally:
        inp.close()
        inp.unlink()


@pytest.mark.asyncio
async def test_worker_death_is_reported_once_as_an_error():
    loop = asyncio.get_running_loop()
    c = WorkerClient("llm", "ai_phone.services.backends.fake:FakeLLM", loop, None, None, {})
    await c.start()
    c._proc.kill()
    e = await asyncio.wait_for(c.events.get(), 10)
    assert e["ev"] == "error" and "disconnected" in e["message"]
    # deterministic: the error path must already have unregistered the reader, so nothing can be enqueued again
    assert not loop.remove_reader(c._parent.fileno()), "reader still registered on a dead pipe"
    assert c.events.empty()
    await c.stop()


@pytest.mark.asyncio
async def test_llm_worker_reports_brain_error_and_survives():
    loop = asyncio.get_running_loop()
    c = WorkerClient("llm", "ai_phone.services.backends.fake:FlakyLLM", loop, None, None, {})
    await c.start()
    g1 = c.next_gen()
    c.send({"op": "llm.generate", "gen": g1, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10})
    evs = await _collect(c, "done", g1)
    assert "".join(e["text"] for e in evs if e["ev"] == "token")
    g2 = c.next_gen()
    c.send({"op": "llm.generate", "gen": g2, "messages": [{"role": "user", "content": "again"}], "max_tokens": 10})
    evs = await _collect(c, "brain_error", g2)
    assert evs[-1]["kind"] == "http_503" and not any(e["ev"] == "done" and e["gen"] == g2 for e in evs)
    g3 = c.next_gen()
    c.send({"op": "llm.generate", "gen": g3, "messages": [{"role": "user", "content": "third"}], "max_tokens": 10})
    evs = await _collect(c, "done", g3)   # the worker is still alive
    assert any(e["ev"] == "token" for e in evs)
    await c.stop()


@pytest.mark.asyncio
async def test_factory_args_reach_the_backend():
    loop = asyncio.get_running_loop()
    c = WorkerClient("llm", "ai_phone.services.backends.fake:FakeLLM", loop, None, None, {}, factory_args={"reply": "Custom."})
    await c.start()
    g = c.next_gen()
    c.send({"op": "llm.generate", "gen": g, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10})
    evs = await _collect(c, "done", g)
    assert "".join(e["text"] for e in evs if e["ev"] == "token") == "Custom."
    await c.stop()


@pytest.mark.parametrize("name", ["SIGNALWIRE_PROJECT_ID", "SIGNALWIRE_SPACE", "SIGNALWIRE_API_TOKEN",
                                 "CUSTOM_CARRIER_API_TOKEN"])
def test_worker_spawn_hides_carrier_credentials_and_account_context(monkeypatch, name):
    from ai_phone.services.client import scrubbed_environment

    monkeypatch.setenv(name, "synthetic-carrier-context")
    monkeypatch.setenv("LANG", "en_GB.UTF-8")
    with scrubbed_environment():
        assert name not in os.environ
        assert os.environ["LANG"] == "en_GB.UTF-8"
    assert os.environ[name] == "synthetic-carrier-context"


@pytest.mark.asyncio
async def test_workers_never_see_line_passwords(monkeypatch):
    monkeypatch.setenv("AI_PHONE_LINE_MAIN_PASSWORD", "hunter2")
    monkeypatch.setenv("TELNYX_API_KEY", "KEYabc")           # the README tells the user to export exactly this
    monkeypatch.setenv("SOME_SERVICE_TOKEN", "tok")
    loop = asyncio.get_running_loop()
    c = WorkerClient("llm", "ai_phone.services.backends.fake:EnvProbeLLM", loop, None, None, {})
    await c.start()
    g = c.next_gen()
    c.send({"op": "llm.generate", "gen": g, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10})
    evs = await _collect(c, "done", g)
    assert "".join(e["text"] for e in evs if e["ev"] == "token") == "clean"
    assert os.environ.get("TELNYX_API_KEY") == "KEYabc"  # the parent keeps its own environment
    await c.stop()


@pytest.mark.asyncio
async def test_idle_worker_keeps_warm_and_still_serves(tmp_path):
    loop = asyncio.get_running_loop()
    probe = tmp_path / "ticks"
    ring = FrameRing(None, 400, frame_bytes=STT_IN_FRAME)
    c = WorkerClient("stt", "ai_phone.services.backends.fake:KeepWarmProbeSTT", loop, ring, None, {},
                     factory_args={"path": str(probe)}, keep_warm_sec=0.2)
    await c.start()
    try:
        await asyncio.sleep(0.9)
        assert probe.exists() and probe.read_text().count("tick") >= 2, "no keep-warm ticks while idle"
        gen = c.next_gen()
        c.send({"op": "stt.finalize", "gen": gen})  # a request after ticks is served normally
        evs = await _collect(c, "final", gen)
        assert evs[-1]["ev"] == "final"
    finally:
        await c.stop()
        ring.close()
        ring.unlink()


class PrepareProbeLLM(FakeLLM):
    """A brain that holds a session: it says which instructions it was prepared with, or refuses to prepare at all."""

    def __init__(self, fail: str | None = None):
        super().__init__()
        self.fail, self.prepared, self.closed = fail, None, 0

    def prepare(self, system_prompt):
        if self.fail:
            raise BrainError(self.fail)
        self.prepared = system_prompt

    def close(self):
        self.closed += 1
        self.prepared = None

    def stream(self, gen_id, messages, max_tokens):
        yield f"prepared with {self.prepared!r}"


PROBE = "tests.unit.test_worker_ipc:PrepareProbeLLM"


async def _generate(c, text="hi"):
    gen = c.next_gen()
    c.send({"op": "llm.generate", "gen": gen, "messages": [{"role": "user", "content": text}], "max_tokens": 10})
    evs = await _collect(c, "done", gen)
    return "".join(e["text"] for e in evs if e["ev"] == "token")


@pytest.mark.asyncio
async def test_llm_prepare_reaches_a_brain_that_holds_a_session():
    loop = asyncio.get_running_loop()
    c = WorkerClient("llm", PROBE, loop, None, None, {"system_prompt": "warm"})
    await c.start()
    c.send({"op": "llm.prepare", "gen": c.current_gen, "system_prompt": "This call: take a message."})
    assert await _generate(c) == "prepared with 'This call: take a message.'"
    await c.stop()


@pytest.mark.asyncio
async def test_llm_prepare_is_ignored_by_a_brain_that_needs_no_preparing():
    loop = asyncio.get_running_loop()
    c = WorkerClient("llm", "ai_phone.services.backends.fake:FakeLLM", loop, None, None, {})
    await c.start()
    c.send({"op": "llm.prepare", "gen": c.current_gen, "system_prompt": "This call: take a message."})
    assert await _generate(c) == "Hello, this is a fake reply."  # no event of its own, and nothing is disturbed
    await c.stop()


@pytest.mark.asyncio
async def test_llm_prepare_runs_even_though_the_last_call_cancelled_that_generation():
    """The daemon prepares with the generation the call before it ended on, and that one was cancelled as it ended."""
    loop = asyncio.get_running_loop()
    c = WorkerClient("llm", PROBE, loop, None, None, {"system_prompt": "warm"})
    await c.start()
    c.cancel(c.current_gen)
    c.send({"op": "llm.prepare", "gen": c.current_gen, "system_prompt": "Next call."})
    assert await _generate(c) == "prepared with 'Next call.'"
    await c.stop()


@pytest.mark.asyncio
async def test_a_brain_that_cannot_be_prepared_reports_it_and_keeps_serving():
    loop = asyncio.get_running_loop()
    c = WorkerClient("llm", PROBE, loop, None, None, {"system_prompt": "warm"}, factory_args={"fail": "not_logged_in"})
    await c.start()
    c.send({"op": "llm.prepare", "gen": c.current_gen, "system_prompt": "This call: take a message."})
    e = await asyncio.wait_for(c.events.get(), 10)
    assert e["ev"] == "brain_error" and e["kind"] == "not_logged_in"
    assert await _generate(c) == "prepared with None"  # the worker is alive; the turn itself will start a session
    await c.stop()


@pytest.mark.asyncio
async def test_a_subscription_brain_through_the_real_worker(tmp_path, monkeypatch):
    """The whole wiring the daemon uses: the factory args it passes, warm-up, llm.prepare, then a turn — against the
    fake CLI, so no plan is spent."""
    from tests.unit.test_cli_brain import FAKE_CLI

    exe = tmp_path / "fake-cli"
    exe.write_text(FAKE_CLI)
    exe.chmod(0o755)
    spec = tmp_path / "script.json"
    spec.write_text(json.dumps({"protocol": "claude", "log": str(tmp_path / "cli.jsonl"),
                                "turns": [{"deltas": ["OK"]}, {"deltas": ["Hello", " there."]}]}))
    monkeypatch.setenv("AI_PHONE_FAKE_CLI", str(spec))
    monkeypatch.setenv("AI_PHONE_HOME", str(tmp_path / "home"))
    loop = asyncio.get_running_loop()
    c = WorkerClient("llm", "ai_phone.services.backends.cli_brain:make_claude", loop, None, None,
                     {"system_prompt": "You are a phone assistant."},
                     factory_args={"model": "haiku", "cli": str(exe), "first_token_timeout_ms": 12000,
                                   "stream_idle_timeout_ms": 4000, "max_response_chars": 2000})
    timings = await c.start()
    assert timings["model"] == "haiku" and timings["cli_version"] == "9.9.9 (fake)"  # the login was proved
    c.send({"op": "llm.prepare", "gen": c.current_gen, "system_prompt": "This call: take a message."})
    gen = c.next_gen()
    c.send({"op": "llm.generate", "gen": gen, "messages": [{"role": "system", "content": "This call: take a message."},
                                                           {"role": "user", "content": "Hi"}], "max_tokens": 60})
    evs = await _collect(c, "done", gen)
    assert "".join(e["text"] for e in evs if e["ev"] == "token") == "Hello there."
    await c.stop()
    log = [json.loads(l) for l in (tmp_path / "cli.jsonl").read_text().splitlines()]
    # the login is read first and costs no turn, then the probe's session, then the call's
    assert [r["event"] for r in log] == ["login", "start", "turn", "start", "turn"]
    assert [r["text"] for r in log if r["event"] == "turn"] == ["Say OK.", "Hi"]
    assert log[-1]["system"] == "This call: take a message."  # prepare's session is the one the turn used


@pytest.mark.asyncio
async def test_llm_release_ends_the_session_the_call_was_using():
    """The call is over: a brain that holds a CLI session lets it go, so no process of ours idles between calls."""
    loop = asyncio.get_running_loop()
    c = WorkerClient("llm", PROBE, loop, None, None, {"system_prompt": "warm"})
    await c.start()
    c.send({"op": "llm.prepare", "gen": c.current_gen, "system_prompt": "This call: take a message."})
    assert await _generate(c) == "prepared with 'This call: take a message.'"
    c.cancel(c.current_gen)  # exactly what the daemon does as a call ends, before it releases
    c.send({"op": "llm.release", "gen": c.current_gen})
    assert await _generate(c) == "prepared with None"  # released, and the worker still serves the next call
    c.send({"op": "llm.prepare", "gen": c.current_gen, "system_prompt": "Next call."})
    assert await _generate(c) == "prepared with 'Next call.'"
    await c.stop()


@pytest.mark.asyncio
async def test_llm_release_is_nothing_to_a_brain_with_no_session():
    loop = asyncio.get_running_loop()
    c = WorkerClient("llm", "ai_phone.services.backends.fake:FakeLLM", loop, None, None, {})
    await c.start()
    c.send({"op": "llm.release", "gen": c.current_gen})
    assert await _generate(c) == "Hello, this is a fake reply."
    await c.stop()


def _fake_cli(tmp_path, monkeypatch, protocol, turns, **extra):
    """The fake CLI from the backend tests, ready for the real worker to drive."""
    from tests.unit.test_cli_brain import FAKE_CLI, fake_codex_home

    fake_codex_home(tmp_path, monkeypatch)
    exe = tmp_path / "fake-cli"
    exe.write_text(FAKE_CLI)
    exe.chmod(0o755)
    from ai_phone.services.backends.cli_brain import CODEX_TOOLS_OFF

    spec = tmp_path / "script.json"
    extra.setdefault("features", [f"{f} stable true" for f in CODEX_TOOLS_OFF])
    spec.write_text(json.dumps({"protocol": protocol, "log": str(tmp_path / "cli.jsonl"), "turns": turns, **extra}))
    monkeypatch.setenv("AI_PHONE_FAKE_CLI", str(spec))
    monkeypatch.setenv("AI_PHONE_HOME", str(tmp_path / "home"))
    return str(exe)


def _cli_log(tmp_path):
    path = tmp_path / "cli.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []


@pytest.mark.asyncio
async def test_a_hangup_during_session_startup_spends_no_turn(tmp_path, monkeypatch):
    """A caller who hangs up while the session is still coming up. `prepare()` is a second of synchronous work, so the
    generate, the cancel and the release all arrive while it runs; the worker used to read the generate first and send
    the turn, and only check for the cancel once a token had come back — by which time the plan had been billed."""
    exe = _fake_cli(tmp_path, monkeypatch, "codex", [{"deltas": ["OK"]}, {"deltas": ["Hello."]}], thread_ms=900)
    loop = asyncio.get_running_loop()
    c = WorkerClient("llm", "ai_phone.services.backends.cli_brain:make_codex", loop, None, None,
                     {"system_prompt": "You are a phone assistant."},
                     factory_args={"model": "gpt-5.6-luna", "cli": exe, "first_token_timeout_ms": 12000,
                                   "stream_idle_timeout_ms": 4000, "max_response_chars": 2000})
    await c.start()
    assert (tmp_path / "home" / "brain" / "codex" / "auth.json").readlink() == (
        Path(os.environ["CODEX_HOME"]) / "auth.json"
    )
    turns_after_warmup = len([r for r in _cli_log(tmp_path) if r["event"] == "turn"])
    c.send({"op": "llm.prepare", "gen": c.current_gen, "system_prompt": "This call: take a message."})
    gen = c.next_gen()
    c.send({"op": "llm.generate", "gen": gen, "messages": [{"role": "system", "content": "This call: take a message."},
                                                           {"role": "user", "content": "Hi"}], "max_tokens": 60})
    c.cancel(gen)                                  # the caller hangs up while the session is still starting
    c.send({"op": "llm.release", "gen": gen})
    await asyncio.sleep(2.5)                       # long enough for the session to come up and the turn to be sent
    assert len([r for r in _cli_log(tmp_path) if r["event"] == "turn"]) == turns_after_warmup, "a turn was still sent"
    assert not any(e.get("ev") == "token" for e in _drain(c)), "the caller is gone; nothing to say"
    # and the worker is still the one the next call uses
    assert await _generate_on(c, "Hi again") == "Hello."
    await c.stop()


def _drain(c):
    out = []
    while not c.events.empty():
        out.append(c.events.get_nowait())
    return out


async def _generate_on(c, text):
    gen = c.next_gen()
    c.send({"op": "llm.generate", "gen": gen, "messages": [{"role": "system", "content": "This call: take a message."},
                                                           {"role": "user", "content": text}], "max_tokens": 60})
    evs = await _collect(c, "done", gen, timeout=20.0)
    return "".join(e["text"] for e in evs if e["ev"] == "token")


@pytest.mark.asyncio
async def test_a_brain_that_fails_to_warm_up_reports_the_kind_and_nothing_else(tmp_path, monkeypatch):
    """A warm-up failure is raised outside the per-turn handler: the worker sent the exception and its traceback, and
    the daemon logged them. Whatever a provider said in that failure is not the daemon's log's business."""
    sentinel = "zq7-sentinel-the-provider-said-this"
    exe = _fake_cli(tmp_path, monkeypatch, "claude", [{"deltas": [], "fail": f"Overloaded: {sentinel}"}])
    loop = asyncio.get_running_loop()
    c = WorkerClient("llm", "ai_phone.services.backends.cli_brain:make_claude", loop, None, None,
                     {"system_prompt": "You are a phone assistant."},
                     factory_args={"model": "haiku", "cli": exe})
    with pytest.raises(RuntimeError) as e:
        await c.start()
    assert "the brain failed: failed" in str(e.value) and sentinel not in str(e.value)
    assert "Traceback" not in str(e.value)
    await c.stop()

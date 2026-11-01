import asyncio
import time

import numpy as np
import pytest

from ai_phone.config import ConversationConfig
from ai_phone.pipeline.timing import TurnTimer
from ai_phone.pipeline.turn import TurnController
from ai_phone.pipeline.vad import Endpointer, EnergyVad
from ai_phone.services.client import WorkerClient
from ai_phone.services.ipc import STT_IN_FRAME, TTS_OUT_FRAME, tag_frame, tts_out_frame_bytes, untag_frame
from ai_phone.telephony.rings import FRAME_BYTES, SILENCE, FrameRing


class StubPort:
    def __init__(self, tx=None):
        self.first_nonsilent_ns = None
        self.armed = 0
        self.tx = tx

    def request_flush(self):
        if self.tx is not None:
            self.tx.flush()

    def arm_t5(self):
        self.armed += 1
        self.first_nonsilent_ns = None


class IdleWorker:
    def __init__(self):
        self.current_gen = 0
        self.events = asyncio.Queue()

    def set_gen(self, gen):
        self.current_gen = gen

    def cancel(self, gen):
        pass

    def send(self, msg):
        pass


@pytest.fixture(params=[16000, 48000])
def small_buffers(request):
    rate = request.param
    rx, tx = FrameRing(None, 2), FrameRing(None, 2, frame_bytes=rate // 50 * 2)
    stt_in = FrameRing(None, 2, frame_bytes=STT_IN_FRAME)
    tts_out = FrameRing(None, 10, frame_bytes=tts_out_frame_bytes(rate))
    workers = [IdleWorker() for _ in range(3)]
    turns = []
    tc = TurnController(ConversationConfig(), StubPort(tx), *workers, rx, tx, stt_in, tts_out,
                        Endpointer(EnergyVad()), [], "system", turns.append)
    yield tc, turns
    for ring in (rx, tx, stt_in, tts_out):
        ring.close()
        ring.unlink()


def test_playback_full_keeps_tts_frames_in_order_without_overruns(small_buffers):
    tc, _ = small_buffers
    tc.state = "SPEAKING"
    frames = [(i + 1).to_bytes(2, "little") * (tc.tx.frame_bytes // 2) for i in range(8)]
    for frame in frames:
        tc.tts_out.push(tag_frame(0, frame))
    tc._pump_tts_out()
    assert tc.tx.available() == 2 and tc._tts_pending is not None
    heard = []
    for _ in frames:
        heard.append(tc.tx.pop())
        tc._pump_tts_out()
    assert heard == frames
    assert tc.tx.overruns == 0 and tc.tts_out.overruns == 0
    assert tc.tts_out.available() == 0 and tc._tts_pending is None


def test_stale_pending_audio_is_discarded_even_while_playback_is_full(small_buffers):
    tc, _ = small_buffers
    tc.state = "SPEAKING"
    old, new = b"\x01\x00" * (tc.tx.frame_bytes // 2), b"\x02\x00" * (tc.tx.frame_bytes // 2)
    tc.tx.push(old)
    tc.tx.push(old)
    tc.tts_out.push(tag_frame(0, old))
    tc._pump_tts_out()
    tc.tts.set_gen(1)  # pending frame is still tagged 0; the pump must not block on it
    tc.tts_out.push(tag_frame(0, old))
    tc.tts_out.push(tag_frame(1, new))
    tc._pump_tts_out()
    assert tc.tts_out.available() == 0
    assert untag_frame(tc._tts_pending) == (1, new)
    tc.tx.flush()
    tc._pump_tts_out()
    assert tc.tx.pop() == new and tc.tx.pop() is None


@pytest.mark.asyncio
async def test_done_waits_until_all_buffered_tts_has_played(small_buffers):
    tc, turns = small_buffers
    stop = asyncio.Event()
    runner = asyncio.create_task(tc.run(stop))
    try:
        await asyncio.sleep(0.02)
        tc.state = "SPEAKING"
        tc.timer = TurnTimer(gen=tc.tts.current_gen)
        tc._seq, tc._tts_done_seq, tc._llm_done = 1, 0, True
        tc._reply_parts = ["Complete reply."]
        frames = [(i + 1).to_bytes(2, "little") * (tc.tx.frame_bytes // 2) for i in range(8)]
        for frame in frames:
            tc.tts_out.push(tag_frame(tc.tts.current_gen, frame))
        heard = []
        for _ in frames:
            await asyncio.sleep(0.02)
            assert turns == []
            heard.append(tc.tx.pop())
        await asyncio.sleep(0.02)
        assert heard == frames and len(turns) == 1
        assert tc.state == "LISTENING" and tc.tx.overruns == 0
    finally:
        stop.set()
        await runner


@pytest.mark.asyncio
async def test_long_cached_greeting_waits_for_playback_capacity(small_buffers):
    tc, _ = small_buffers
    tc.greeting = [(i + 1).to_bytes(2, "little") * (tc.tx.frame_bytes // 2) for i in range(8)]
    stop = asyncio.Event()
    runner = asyncio.create_task(tc.run(stop))
    try:
        heard = []
        for _ in tc.greeting:
            await asyncio.sleep(0.02)
            assert tc.state == "GREETING"
            heard.append(tc.tx.pop())
        await asyncio.sleep(0.02)
        assert heard == tc.greeting and tc.tx.overruns == 0
        assert tc.state == "LISTENING" and tc._after_flush == []
    finally:
        stop.set()
        await runner


def test_interrupt_discards_cached_audio_waiting_for_capacity(small_buffers):
    tc, _ = small_buffers
    tc._queue_cached([b"\x01\x00" * (tc.tx.frame_bytes // 2)] * 8)
    assert tc._after_flush
    tc._flush_tx()
    tc._push_after_flush()
    assert tc.tx.available() == 0 and tc._after_flush == []


def test_held_model_opening_replays_in_order_through_small_playback_ring(small_buffers):
    tc, _ = small_buffers
    frames = [(i + 1).to_bytes(2, "little") * (tc.tx.frame_bytes // 2) for i in range(8)]
    tc.state, tc._opening_pending = "OUTBOUND_OPENING", True
    tc._held = list(frames)
    tc.timer = TurnTimer(gen=0)
    tc._seq = 1
    tc._greeting_over(time.monotonic_ns())
    assert tc.state == "SPEAKING"
    heard = []
    for _ in frames:
        heard.append(tc.tx.pop())
        tc._push_after_flush()
    assert heard == frames and tc.tx.overruns == 0
    assert tc._held == [] and tc._after_flush == []


def speech_frames(seconds, amp=0.5):
    n = int(seconds * 16000)
    t = np.arange(n) / 16000.0
    pcm = ((amp * np.sin(2 * np.pi * 300 * t)) * 32767).astype("<i2").tobytes()
    return [pcm[i:i + FRAME_BYTES] for i in range(0, len(pcm) - FRAME_BYTES + 1, FRAME_BYTES)]


async def _drain_tx(tx, port, seconds):
    """Play the role of PJSIP: pop tx every 20 ms, stamping t5 on the first non-silent frame."""
    end = time.monotonic() + seconds
    n = 0
    while time.monotonic() < end:
        f = tx.pop()
        if f is not None:
            n += 1
            if port.first_nonsilent_ns is None and f != SILENCE and port.armed:
                port.first_nonsilent_ns = time.monotonic_ns()
        await asyncio.sleep(0.02)
    return n


async def make_stack(loop, cfg=None, greeting=None):
    rx, tx = FrameRing(None, 100), FrameRing(None, 400)
    stt_in, tts_out = FrameRing(None, 400, frame_bytes=STT_IN_FRAME), FrameRing(None, 400, frame_bytes=TTS_OUT_FRAME)
    stt = WorkerClient("stt", "ai_phone.services.backends.fake:FakeSTT", loop, stt_in, None, {})
    llm = WorkerClient("llm", "ai_phone.services.backends.fake:FakeLLM", loop, None, None, {"system_prompt": "sys"})
    tts = WorkerClient("tts", "ai_phone.services.backends.fake:FakeTTS", loop, None, tts_out, {})
    await asyncio.gather(stt.start(), llm.start(), tts.start())
    port, turns = StubPort(tx), []
    tc = TurnController(cfg or ConversationConfig(endpoint_silence_ms=300), port, stt, llm, tts, rx, tx, stt_in, tts_out,
                        Endpointer(EnergyVad()), greeting if greeting is not None else [], "sys", turns.append)
    return tc, port, turns, (rx, tx, stt_in, tts_out), (stt, llm, tts)


async def teardown(stop, runner, rings, clients):
    stop.set()
    await runner
    await asyncio.gather(*(c.stop() for c in clients))
    for r in rings:
        r.close()
        r.unlink()


@pytest.mark.asyncio
async def test_full_turn_with_fake_backends():
    loop = asyncio.get_running_loop()
    tc, port, turns, rings, clients = await make_stack(loop, greeting=[b"\x10\x00" * 320] * 10)
    rx, tx = rings[0], rings[1]
    stop = asyncio.Event()
    runner = asyncio.create_task(tc.run(stop))
    try:
        n = await _drain_tx(tx, port, 0.5)  # greeting plays first
        assert n >= 10 and tc.state == "LISTENING"
        for f in speech_frames(0.6):  # caller speaks 0.6 s then is silent
            rx.push(f)
            await asyncio.sleep(0.02)
        for _ in range(25):
            rx.push(SILENCE)
            await asyncio.sleep(0.02)
        n = await _drain_tx(tx, port, 2.0)
        assert n > 0 and turns, "no turn completed"
        t = turns[0]
        assert t.stt_text.startswith("fake utterance")
        assert t.reply_text == "Hello, this is a fake reply."
        assert t.t0 and t.t1 and t.t2 and t.t3 and t.t4 and t.t5 and t.t0 < t.t1 < t.t2 <= t.t3 < t.t4 < t.t5
        assert tc.history[-2]["role"] == "user" and tc.history[-1]["role"] == "assistant"
        assert tc.state == "LISTENING"
    finally:
        await teardown(stop, runner, rings, clients)


@pytest.mark.asyncio
async def test_bargein_cancels_and_flushes():
    loop = asyncio.get_running_loop()
    tc, port, turns, rings, clients = await make_stack(loop)
    rx, tx = rings[0], rings[1]
    stop = asyncio.Event()
    runner = asyncio.create_task(tc.run(stop))
    try:
        for f in speech_frames(0.6):
            rx.push(f)
            await asyncio.sleep(0.02)
        for _ in range(20):
            rx.push(SILENCE)
            await asyncio.sleep(0.02)
        for _ in range(100):  # wait until the agent is speaking, without draining tx, so frames pile up
            if tc.state == "SPEAKING":
                break
            await asyncio.sleep(0.02)
        assert tc.state == "SPEAKING" and tx.available() > 0
        for f in speech_frames(0.4, amp=0.8):  # caller talks over it, loudly, for 0.4 s
            rx.push(f)
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.05)
        assert turns and turns[0].bargein and turns[0].bargein_stop_ns is not None
        assert tx.available() == 0, "tx not flushed"
        assert tc.history[-1]["content"].endswith("[interrupted]")
        assert tc.state in ("LISTENING", "THINKING")
    finally:
        await teardown(stop, runner, rings, clients)


@pytest.mark.asyncio
async def test_speech_during_greeting_stops_it_and_is_heard():
    loop = asyncio.get_running_loop()
    tc, port, turns, rings, clients = await make_stack(loop, greeting=[b"\x10\x00" * 320] * 150)  # 3 s greeting
    rx, tx = rings[0], rings[1]
    stop = asyncio.Event()
    runner = asyncio.create_task(tc.run(stop))
    try:
        await asyncio.sleep(0.1)
        assert tc.state == "GREETING" and tx.available() > 100
        for f in speech_frames(0.6):  # caller talks over the greeting
            rx.push(f)
            await asyncio.sleep(0.02)
        assert tx.available() < 100, "greeting was not cut"
        for _ in range(25):
            rx.push(SILENCE)
            await asyncio.sleep(0.02)
        await _drain_tx(tx, port, 2.0)
        assert turns and turns[0].stt_text.startswith("fake utterance")
        assert turns[0].t0_end and turns[0].t0_end > turns[0].t0
    finally:
        await teardown(stop, runner, rings, clients)


async def _speak_then_silence(rx, seconds=0.6):
    for f in speech_frames(seconds):
        rx.push(f)
        await asyncio.sleep(0.02)
    for _ in range(25):
        rx.push(SILENCE)
        await asyncio.sleep(0.02)


async def make_flaky_stack(loop, fail_calls="2", **kw):
    rx, tx = FrameRing(None, 100), FrameRing(None, 400)
    stt_in, tts_out = FrameRing(None, 400, frame_bytes=STT_IN_FRAME), FrameRing(None, 400, frame_bytes=TTS_OUT_FRAME)
    stt = WorkerClient("stt", "ai_phone.services.backends.fake:FakeSTT", loop, stt_in, None, {})
    llm = WorkerClient("llm", "ai_phone.services.backends.fake:FlakyLLM", loop, None, None, {"system_prompt": "sys"}, factory_args={"fail_calls": fail_calls})
    tts = WorkerClient("tts", "ai_phone.services.backends.fake:FakeTTS", loop, None, tts_out, {})
    await asyncio.gather(stt.start(), llm.start(), tts.start())
    port, turns = StubPort(tx), []
    apology, goodbye = [b"\x0a\x00" * 320] * 5, [b"\x0b\x00" * 320] * 5
    tc = TurnController(ConversationConfig(endpoint_silence_ms=300), port, stt, llm, tts, rx, tx, stt_in, tts_out,
                        Endpointer(EnergyVad()), [], "sys", turns.append, apology_frames=apology, goodbye_frames=goodbye, **kw)
    return tc, port, turns, (rx, tx, stt_in, tts_out), (stt, llm, tts), apology, goodbye


@pytest.mark.asyncio
async def test_brain_error_plays_apology_and_call_continues():
    loop = asyncio.get_running_loop()
    tc, port, turns, rings, clients, apology, goodbye = await make_flaky_stack(loop)
    rx, tx = rings[0], rings[1]
    stop = asyncio.Event()
    runner = asyncio.create_task(tc.run(stop))
    try:
        await _drain_tx(tx, port, 0.2)
        await _speak_then_silence(rx)            # turn 1: FlakyLLM answers normally
        await _drain_tx(tx, port, 2.0)
        assert len(turns) == 1 and turns[0].brain_error is None
        gen_before = tc.llm.current_gen
        await _speak_then_silence(rx)            # turn 2: FlakyLLM raises http_503
        popped = []
        end = time.monotonic() + 2.0
        while time.monotonic() < end:
            f = tx.pop()
            if f is not None:
                popped.append(f)
            await asyncio.sleep(0.02)
        assert any(f == apology[0] for f in popped), "apology frames were not played"
        assert tc.state == "LISTENING" and tc.end_reason is None
        assert tc.llm.current_gen > gen_before
        assert turns[-1].brain_error == "http_503"
        await _speak_then_silence(rx)            # turn 3: FlakyLLM answers again; the counter reset
        await _drain_tx(tx, port, 2.0)
        assert turns[-1].brain_error is None and tc.end_reason is None
    finally:
        await teardown(stop, runner, rings, clients)


@pytest.mark.asyncio
async def test_two_consecutive_brain_errors_end_the_call():
    loop = asyncio.get_running_loop()
    tc, port, turns, rings, clients, apology, goodbye = await make_flaky_stack(loop, fail_calls="2,3")
    rx, tx = rings[0], rings[1]
    stop = asyncio.Event()
    runner = asyncio.create_task(tc.run(stop))
    try:
        await _drain_tx(tx, port, 0.2)
        await _speak_then_silence(rx)            # FlakyLLM: call 1 ok
        await _drain_tx(tx, port, 2.0)
        await _speak_then_silence(rx)            # FlakyLLM: call 2 raises -> apology, back to LISTENING
        await _drain_tx(tx, port, 1.0)
        await _speak_then_silence(rx)            # FlakyLLM: call 3 raises -> second error in a row -> goodbye, drain, then hangup
        for _ in range(200):
            tx.pop()                              # PJSIP keeps draining the goodbye
            if tc.hangup_requested.is_set():
                break
            await asyncio.sleep(0.02)
        assert tc.end_reason == "brain_error" and tc.hangup_requested.is_set()
    finally:
        await teardown(stop, runner, rings, clients)


@pytest.mark.asyncio
async def test_outbound_opening_plays_then_listens():
    loop = asyncio.get_running_loop()
    opening = [b"\x0c\x00" * 320] * 8
    tc, port, turns, rings, clients = await make_stack(loop, greeting=[b"\x10\x00" * 320] * 10)
    tc.opening_frames, tc.direction = opening, "out"
    rx, tx = rings[0], rings[1]
    stop = asyncio.Event()
    runner = asyncio.create_task(tc.run(stop))
    try:
        await asyncio.sleep(0.05)               # let run() start and pick the opening state
        assert tc.state == "OUTBOUND_OPENING"
        popped = []
        end = time.monotonic() + 0.5
        while time.monotonic() < end:
            f = tx.pop()
            if f is not None:
                popped.append(f)
            await asyncio.sleep(0.02)
        assert popped[:8] == opening and tc.state == "LISTENING"
        await _speak_then_silence(rx)
        await _drain_tx(tx, port, 2.0)
        assert turns and turns[0].direction == "out" and turns[0].to_record("c", 0)["direction"] == "out"
    finally:
        await teardown(stop, runner, rings, clients)


class DeferredFlushPort(StubPort):
    """Behaves like the real RingMediaPort: request_flush() only arms a flag; the (simulated) PJSIP thread flushes later."""

    def __init__(self, tx):
        super().__init__(tx)
        self.flush_pending = False

    def request_flush(self):
        self.flush_pending = True

    def pjsip_flush(self):
        if self.flush_pending:
            self.tx.flush()
            self.flush_pending = False


@pytest.mark.asyncio
async def test_apology_is_not_lost_to_a_deferred_flush_and_partial_reply_is_kept():
    loop = asyncio.get_running_loop()
    tc, port, turns, rings, clients, apology, goodbye = await make_flaky_stack(loop)
    rx, tx = rings[0], rings[1]
    dport = DeferredFlushPort(tx)
    tc.port = dport
    stop = asyncio.Event()
    runner = asyncio.create_task(tc.run(stop))
    try:
        await _drain_tx(tx, dport, 0.2)
        await _speak_then_silence(rx)            # turn 1 ok
        await _drain_tx(tx, dport, 2.0)
        await _speak_then_silence(rx)            # turn 2: brain error while the flush is still pending
        for _ in range(100):
            if dport.flush_pending:
                break
            await asyncio.sleep(0.02)
        assert dport.flush_pending and tc._after_flush, "the apology must wait for the PJSIP-side flush"
        assert tx.available() == 0 or all(f != apology[0] for f in [tx.pop() for _ in range(tx.available())])
        dport.pjsip_flush()                       # the PJSIP thread performs the flush
        popped = []
        end = time.monotonic() + 1.0
        while time.monotonic() < end:
            f = tx.pop()
            if f is not None:
                popped.append(f)
            await asyncio.sleep(0.02)
        assert any(f == apology[0] for f in popped), "the apology was pushed after the flush"
        assert any(h["content"].endswith("[interrupted]") for h in tc.history if h["role"] == "assistant") or not turns[-1].reply_text
    finally:
        await teardown(stop, runner, rings, clients)


@pytest.mark.asyncio
async def test_goodbye_drains_before_hangup_is_requested():
    loop = asyncio.get_running_loop()
    tc, port, turns, rings, clients, apology, goodbye = await make_flaky_stack(loop, fail_calls="2,3")
    rx, tx = rings[0], rings[1]
    stop = asyncio.Event()
    runner = asyncio.create_task(tc.run(stop))
    try:
        await _drain_tx(tx, port, 0.2)
        await _speak_then_silence(rx)
        await _drain_tx(tx, port, 2.0)
        await _speak_then_silence(rx)            # error 1 -> apology
        await _drain_tx(tx, port, 1.0)
        await _speak_then_silence(rx)            # error 2 -> goodbye, then hangup once drained
        for _ in range(100):
            if tc.state == "ENDING_DRAIN":
                break
            await asyncio.sleep(0.02)
        assert tc.state == "ENDING_DRAIN" and not tc.hangup_requested.is_set() and tx.available() > 0
        await _drain_tx(tx, port, 1.0)           # PJSIP drains the goodbye
        for _ in range(100):
            if tc.hangup_requested.is_set():
                break
            await asyncio.sleep(0.02)
        assert tc.hangup_requested.is_set() and tc.end_reason == "brain_error" and tx.available() == 0
    finally:
        await teardown(stop, runner, rings, clients)


def test_real_media_port_clears_the_flag_only_after_flushing():
    pytest.importorskip("ai_phone.vendor.pjsua2")
    import inspect
    from ai_phone.telephony import media_port
    src = inspect.getsource(media_port.RingMediaPort.onFrameRequested)
    assert src.index("self.tx.flush()") < src.index("self._flush_requested = False")


@pytest.mark.asyncio
async def test_a_turn_cut_short_by_the_end_of_the_call_keeps_what_was_said():
    """The daemon writes the turn that was still running when the call ended; it must carry the agent's words and the
    ring counters, like every other end-of-turn path."""
    loop = asyncio.get_running_loop()
    tc, port, turns, rings, clients = await make_stack(loop, greeting=[b"\x10\x00" * 320] * 10)
    rx, tx = rings[0], rings[1]
    stop = asyncio.Event()
    runner = asyncio.create_task(tc.run(stop))
    try:
        await _drain_tx(tx, port, 0.5)                 # the greeting plays
        for f in speech_frames(0.6):                   # the caller speaks, then stops
            rx.push(f)
            await asyncio.sleep(0.02)
        for _ in range(25):
            rx.push(SILENCE)
            await asyncio.sleep(0.02)
        for _ in range(100):                           # stop as soon as the agent is talking
            if tc.state == "SPEAKING" and tc._reply_parts:
                break
            await asyncio.sleep(0.02)
        assert tc.state == "SPEAKING" and tc.timer is not None, "the agent never started speaking"
        record = tc.incomplete_record("7", 0)
        assert record["reply_text"] and record["reply_text"] in "Hello, this is a fake reply."
        assert record["rx_overruns"] == rx.overruns and record["tx_underruns"] == tx.underruns
        assert record["call_id"] == "7" and record["turn"] == 0
    finally:
        await teardown(stop, runner, rings, clients)


@pytest.mark.asyncio
async def test_t4_comes_from_the_frames_not_from_the_event_that_announces_them():
    """The TTS worker pushes its frames before it sends `chunk`, so the port can stamp t5 before that event is
    drained. t4 is taken from the frames, which keeps t5 >= t4 (and egress_ms positive) whoever wins the race."""
    loop = asyncio.get_running_loop()
    tc, port, turns, rings, clients = await make_stack(loop)
    tts_out = rings[3]
    try:
        gen = tc.tts.current_gen
        tc.timer = TurnTimer(gen=gen, t0=time.monotonic_ns(), t3=time.monotonic_ns())
        tc.state = "SPEAKING"
        tts_out.push(tag_frame(gen, b"\x10\x00" * 320))
        tc._pump_tts_out()
        assert tc.timer.t4 is not None and tc.timer.t4 >= tc.timer.t3
        frames = tc.timer.t4
        tc._on_event({"ev": "chunk", "kind": "tts", "gen": gen, "seq": 0, "nframes": 1})
        assert tc.timer.t4 == frames, "the announcement moved t4 later than the frames it announced"
        t5 = time.monotonic_ns()
        port.armed, port.first_nonsilent_ns = 1, t5
        tc._pump_tts_out()
        assert tc.timer.t5 == t5 and tc.timer.t5 >= tc.timer.t4
        assert tc.timer.to_record("7", 0)["egress_ms"] >= 0
    finally:
        tc.timer = None  # the controller was never run here, so there is no runner to stop
        await asyncio.gather(*(c.stop() for c in clients))
        for r in rings:
            r.close()
            r.unlink()

"""A model's end marker is a request; ordinary replies must keep the call open."""
import asyncio

import pytest

from ai_phone.config import ConversationConfig
from ai_phone.pipeline.turn import TurnController
from ai_phone.pipeline.vad import Endpointer, EnergyVad
from ai_phone.services.ipc import STT_IN_FRAME, TTS_OUT_FRAME
from ai_phone.telephony.rings import FrameRing
from tests.unit.test_turn_controller import StubPort
from tests.unit.test_turn_modes import NoWorker


class RecordingWorker(NoWorker):
    def __init__(self):
        self.current_gen = 0
        self.sent = []
        self.cancelled = []

    def send(self, message):
        self.sent.append(message)

    def cancel(self, gen):
        self.cancelled.append(gen)


@pytest.fixture
def controller():
    rings = (FrameRing(None, 2), FrameRing(None, 2),
             FrameRing(None, 2, frame_bytes=STT_IN_FRAME),
             FrameRing(None, 2, frame_bytes=TTS_OUT_FRAME))
    workers = [RecordingWorker() for _ in range(3)]
    turns, lines = [], []
    tc = TurnController(ConversationConfig(), StubPort(rings[1]), *workers, *rings,
                        Endpointer(EnergyVad()), [], "system", turns.append,
                        on_line=lambda who, text, interrupted: lines.append((who, text, interrupted)))
    tc.state = "THINKING"
    tc.timer = tc._timer()
    tc._on_event({"ev": "final", "kind": "stt", "gen": tc.timer.gen,
                  "text": "Please tell me the opening hours."})
    try:
        yield tc, workers, turns, lines
    finally:
        for ring in rings:
            ring.close()
            ring.unlink()


def complete_reply(tc, reply):
    """Deliver actual token/done events, then finish the already-drained synthetic speech."""
    gen = tc.timer.gen
    # Character-sized tokens exercise markers split across arbitrary model token boundaries.
    for token in reply:
        tc._on_event({"ev": "token", "kind": "llm", "gen": gen, "text": token})
    tc._on_event({"ev": "done", "kind": "llm", "gen": gen})
    if tc.timer is not None:
        tc._on_event({"ev": "done", "kind": "tts", "gen": gen, "seq": tc._seq - 1})
        tc._finish_turn()


@pytest.mark.parametrize("reply", [
    "What name should I put this under? [END]",
    "Yes, I can hear you perfectly. What is your name and what is this call about? [END]",
    "Would you like anything else? Goodbye. [END]",
    "Goodbye. [END]?",
    "We open at nine. [END]",
    "I have your message. [END]",
    "Thanks. [END]",
    "[END]",
    "  [end].  ",
    "[END] Goodbye.",
    "Goodbye. [END] Please hold.",
    "Goodbye. [END] Goodbye. [END]",
    "Goodbye. [END][END]",
    "Don't say goodbye. [END]",
    "The password is goodbye. [END]",
    "Say goodbye to John. [END]",
    'He said "goodbye". [END]',
    '"Goodbye." [END]',
    "Goodbye. I still need your name. [END]",
    "Goodbye, please tell me your name. [END]",
    "Goodbye.",
])
def test_invalid_end_request_keeps_listening(controller, reply):
    tc, workers, turns, lines = controller
    complete_reply(tc, reply)
    assert not tc.hangup_requested.is_set(), reply
    assert tc.end_reason is None and tc.state == "LISTENING"
    assert len(turns) == 1
    assert all("[end]" not in line[1].lower() for line in lines)
    assert all("[end]" not in message["text"].lower() for message in workers[2].sent)
    assert all("[end]" not in message["content"].lower() for message in tc.history)


@pytest.mark.parametrize("reply, spoken", [
    ("Goodbye, thanks. [END]", "Goodbye, thanks."),
    ("Thanks, bye. [END]", "Thanks, bye."),
    ("Bye now. [END]", "Bye now."),
    ("Thank you. Goodbye. [END]", "Thank you. Goodbye."),
    ("We open at nine. Goodbye. [END]", "We open at nine. Goodbye."),
    ("Goodbye. [eNd]", "Goodbye."),
    ("Thanks, bye [END].", "Thanks, bye."),
    ("Goodbye. [END].", "Goodbye."),
])
def test_explicit_terminal_farewell_can_end_without_a_caller_goodbye(controller, reply, spoken):
    tc, workers, turns, lines = controller
    complete_reply(tc, reply)
    assert tc.hangup_requested.is_set() and tc.end_reason == "agent"
    assert tc.state == "ENDING"
    assert lines[-1] == ("agent", spoken, False)
    assert turns[-1].reply_text == spoken
    assert tc.history[-1] == {"role": "assistant", "content": spoken}
    assert all("[end]" not in message["text"].lower() for message in workers[2].sent)


@pytest.mark.asyncio
async def test_question_with_end_marker_still_accepts_the_next_caller_turn():
    from tests.unit.test_turn_modes import drain, speak, stack, teardown, until

    reply = "What name should I put this under? [END]"
    tc, rings, clients, _, lines = await stack(asyncio.get_running_loop(), llm_args={"reply": reply})
    stop = asyncio.Event()
    runner = asyncio.create_task(tc.run(stop))
    try:
        for turn in (1, 2):
            await speak(rings[0])
            assert len(await drain(rings[1], 1.5)) >= 25
            assert await until(lambda: len([line for line in lines if line[0] == "agent"]) == turn)
            assert tc.state == "LISTENING" and not tc.hangup_requested.is_set()
        assert [(who, text) for who, text, _ in lines] == [
            ("them", "fake utterance 1"),
            ("agent", "What name should I put this under?"),
            ("them", "fake utterance 2"),
            ("agent", "What name should I put this under?"),
        ]
        assert all("[end]" not in message["content"].lower() for message in tc.history)
    finally:
        await teardown(stop, runner, rings, clients)


@pytest.mark.asyncio
async def test_interrupted_farewell_drops_the_old_generations_end_request():
    from ai_phone.services.ipc import tag_frame
    from tests.unit.test_turn_modes import drain, speak, stack, teardown, until

    tc, rings, clients, _, lines = await stack(asyncio.get_running_loop(), llm="EndingLLM")
    stop = asyncio.Event()
    runner = asyncio.create_task(tc.run(stop))
    try:
        await speak(rings[0])
        assert await until(lambda: tc.state == "SPEAKING" and tc._llm_done)
        old_gen = tc.timer.gen
        assert rings[1].available() > 0 and not tc.hangup_requested.is_set()
        # The queued farewell is still playing when the caller starts talking again.
        await speak(rings[0], seconds=0.4, silence_frames=0)
        assert await until(lambda: ("agent", "Goodbye, thanks.", True) in lines)
        assert tc.timer.gen > old_gen and tc.state == "LISTENING"
        assert tc.history[-1] == {"role": "assistant", "content": "Goodbye, thanks. [interrupted]"}

        # Simulate delivery already queued before the cancellation reached the worker.
        tc._on_event({"ev": "token", "kind": "llm", "gen": old_gen, "text": "Goodbye. [END]"})
        tc._on_event({"ev": "done", "kind": "llm", "gen": old_gen})
        tc._on_event({"ev": "done", "kind": "tts", "gen": old_gen, "seq": 99})
        rings[3].push(tag_frame(old_gen, b"\x01\x00" * 320))
        assert await drain(rings[1], 0.2) == []
        assert tc._reply_parts == [] and not tc._llm_done
        assert not tc.hangup_requested.is_set() and tc.end_reason is None
    finally:
        await teardown(stop, runner, rings, clients)


@pytest.mark.asyncio
async def test_valid_farewell_waits_for_queued_audio_before_hanging_up():
    from tests.unit.test_turn_modes import drain, speak, stack, teardown, until

    tc, rings, clients, _, lines = await stack(asyncio.get_running_loop(), llm="EndingLLM")
    stop = asyncio.Event()
    runner = asyncio.create_task(tc.run(stop))
    try:
        await speak(rings[0])
        assert await until(lambda: tc._llm_done and tc._seq > 0
                           and tc._tts_done_seq == tc._seq - 1
                           and rings[3].available() == 0 and tc._tts_pending is None)
        queued = rings[1].available()
        assert queued >= 25 and tc.state == "SPEAKING"
        # Even fully generated text/audio cannot end the call before playback drains.
        await asyncio.sleep(0.05)
        assert rings[1].available() == queued
        assert not tc.hangup_requested.is_set() and tc.end_reason is None
        assert not any(who == "agent" for who, _, _ in lines)
        assert len(await drain(rings[1], 1.5)) == queued
        assert await until(tc.hangup_requested.is_set)
        assert tc.end_reason == "agent" and rings[1].available() == 0
        assert lines[-1] == ("agent", "Goodbye, thanks.", False)
    finally:
        await teardown(stop, runner, rings, clients)

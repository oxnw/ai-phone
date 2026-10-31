"""Opening context must not replace the task or constrain later caller turns."""
from __future__ import annotations

import asyncio
from copy import deepcopy
import time

import pytest

from ai_phone.config import ConversationConfig
from ai_phone.pipeline.prompts import DISCLOSURE, OPENING_CUE, fixed_instructions, system_prompt
from ai_phone.pipeline.turn import TurnController
from ai_phone.pipeline.vad import Endpointer, EnergyVad
from ai_phone.services.ipc import STT_IN_FRAME, TTS_OUT_FRAME
from ai_phone.telephony.rings import FrameRing, SILENCE
from tests.unit.test_turn_controller import StubPort, speech_frames

TASK = "This is a test, say a paragraph of lorem ipsum"
CALLER = "Please read the paragraph now."


class RecordingWorker:
    def __init__(self, kind):
        self.kind, self.current_gen = kind, 0
        self.events = asyncio.Queue()
        self.sent, self.cancelled = [], []

    def set_gen(self, gen):
        self.current_gen = gen

    def send(self, message):
        self.sent.append(deepcopy(message))

    def cancel(self, gen):
        self.cancelled.append(gen)

    def generations(self):
        return [m for m in self.sent if m["op"] == "llm.generate"]


@pytest.fixture
def opening():
    rx, tx = FrameRing(None, 100), FrameRing(None, 100)
    stt_in = FrameRing(None, 100, frame_bytes=STT_IN_FRAME)
    tts_out = FrameRing(None, 100, frame_bytes=TTS_OUT_FRAME)
    workers = [RecordingWorker(kind) for kind in ("stt", "llm", "tts")]
    turns, lines = [], []
    prompt = system_prompt(fixed_instructions(None), TASK)
    tc = TurnController(ConversationConfig(), StubPort(tx), *workers, rx, tx, stt_in, tts_out,
                        Endpointer(EnergyVad()), [], prompt, turns.append, opening_frames=[], direction="out",
                        generate_opening=True, on_line=lambda *line: lines.append(line))
    tc.state = "OUTBOUND_OPENING"
    tc._new_gen()
    tc._start_opening()
    yield tc, turns, lines
    for ring in (rx, tx, stt_in, tts_out):
        ring.close()
        ring.unlink()


def caller_final(tc):
    tc.state, tc.timer = "THINKING", tc._timer()
    tc._on_event({"ev": "final", "gen": tc.timer.gen, "text": CALLER})
    return tc.llm.generations()[-1]["messages"]


def assert_call_context(messages):
    assert messages[0]["role"] == "system"
    assert messages[0]["content"].endswith(TASK)
    assert "longer response" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": OPENING_CUE}
    assert "why you are calling" not in messages[1]["content"]
    assert "one sentence" not in messages[1]["content"]
    assert "Now say" not in messages[1]["content"]


def test_opening_payload_preserves_requested_task_with_a_factual_connection_event(opening):
    tc, _, _ = opening
    messages = tc.llm.generations()[0]["messages"]
    assert_call_context(messages)
    assert messages[1]["content"] == (
        f'(The call is connected. The other person has answered and has already heard: "{DISCLOSURE}")'
    )
    assert len(messages) == 2
    assert tc.history == messages[1:]


def test_completed_opening_preserves_spoken_content_and_answers_the_actual_caller(opening):
    tc, _, lines = opening
    spoken = "Lorem ipsum dolor sit amet. Consectetur adipiscing elit."
    tc._reply_parts = [spoken]
    tc._finish_turn()
    messages = caller_final(tc)
    assert_call_context(messages)
    assert messages[2:] == [{"role": "assistant", "content": spoken}, {"role": "user", "content": CALLER}]
    assert lines == [("agent", spoken, False), ("them", CALLER, False)]
    assert not tc.hangup_requested.is_set()


@pytest.mark.parametrize("failure", ["cancelled", "brain_error"])
def test_unheard_opening_does_not_leave_a_task_restricting_instruction(opening, failure):
    tc, _, lines = opening
    tc._reply_parts = ["Unplayed generated text."]
    if failure == "cancelled":
        tc._cancel_opening()
    else:
        tc._brain_error(time.monotonic_ns(), "http_503")
    messages = caller_final(tc)
    assert_call_context(messages)
    assert messages[2:] == [{"role": "assistant", "content": "[interrupted]"},
                           {"role": "user", "content": CALLER}]
    assert lines == [("them", CALLER, False)]
    assert not tc.hangup_requested.is_set()


def test_caller_speech_after_disclosure_interrupts_a_delayed_model_opening(opening):
    tc, turns, lines = opening
    opening_gen = tc.timer.gen
    initial_request = deepcopy(tc.llm.generations()[0])
    now = time.monotonic_ns()
    # The disclosure has drained, but the model has not emitted any token or audio.
    tc._greeting_over(now)
    assert tc.state == "THINKING" and tc._opening_pending
    for i, frame in enumerate(speech_frames(0.6) + [SILENCE] * 25):
        tc._on_frame(frame, now + (i + 1) * 20_000_000)
    assert len(turns) == 1 and turns[0].bargein
    assert opening_gen in tc.llm.cancelled and opening_gen in tc.tts.cancelled
    assert not tc._opening_pending and tc.stt.current_gen != opening_gen
    assert any(m["op"] == "stt.push" for m in tc.stt.sent)
    assert tc.stt.sent[-1] == {"op": "stt.finalize", "gen": tc.stt.current_gen}
    # A late token for the cancelled generation must not speak or change history.
    tc._on_event({"ev": "token", "gen": opening_gen, "text": "I can only explain the purpose."})
    messages = caller_final(tc)
    assert_call_context(messages)
    assert messages[2:] == [{"role": "assistant", "content": "[interrupted]"},
                           {"role": "user", "content": CALLER}]
    assert tc.llm.generations()[0] == initial_request
    assert lines == [("them", CALLER, False)]
    assert not tc.hangup_requested.is_set()

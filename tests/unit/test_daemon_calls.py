"""The daemon's call records and command handlers, driven without PJSIP: a stub telephony object stands in."""
import asyncio

import pytest

from ai_phone.config import CallsConfig, Config, LineConfig
from ai_phone.daemon import Daemon
from ai_phone.telephony import events as ev
from ai_phone.telephony.endpoint import LineSpec


@pytest.mark.asyncio
@pytest.mark.parametrize("output_rate", [16000, 48000])
async def test_caching_a_phrase_larger_than_the_worker_ring_drains_during_generation(tmp_path, monkeypatch, output_rate):
    from ai_phone.services.client import WorkerClient
    from ai_phone.services.ipc import tts_out_frame_bytes
    from ai_phone.telephony.rings import FrameRing
    from tests.unit.test_worker_ipc import ORDERED_TTS

    monkeypatch.setenv("AI_PHONE_HOME", str(tmp_path))
    ring = FrameRing(None, 2, frame_bytes=tts_out_frame_bytes(output_rate))
    tts = WorkerClient("tts", ORDERED_TTS, asyncio.get_running_loop(), None, ring, {},
                       factory_args={"output_rate": output_rate})
    try:
        await tts.start()
        daemon = Daemon(Config(), "t4", tmp_path, lan=True, fake=True)
        frames = await asyncio.wait_for(daemon._cache_phrase(tts, ring, "40"), 5)
        assert frames == [(i + 1).to_bytes(2, "little") * (output_rate // 50) for i in range(40)]
        assert ring.overruns == 0 and ring.available() == 0
    finally:
        await tts.stop()
        ring.close()
        ring.unlink()


class Tel:
    def __init__(self, pj_id=7):
        self.pj_id, self.calls, self.hung, self.answered = pj_id, [], [], []
        self.current_call_id = None
        self.answer_result = True

    def make_call(self, line_id, to, timeout_sec):
        self.calls.append((line_id, to))
        return self.pj_id

    def hangup(self, call_id, code=200):
        self.hung.append(call_id)

    def answer(self, call_id):
        self.answered.append(call_id)
        return self.answer_result

    def release_call(self, call_id):
        return True


def ctx(tel):
    spec = LineSpec.from_config("agents", LineConfig(kind="sip", user="agents", aor="sip:agents@127.0.0.1", display_name="agents",
                                                     accept_invites_from="lan", caller_id="sip:agents@127.0.0.1"), None)
    return {"tel": tel, "specs": {"agents": spec}, "events": asyncio.Queue()}


def daemon(tmp_path, monkeypatch, limit=0, **calls):
    monkeypatch.setenv("AI_PHONE_HOME", str(tmp_path))
    return Daemon(Config(calls=CallsConfig(daily_limit=limit, **calls)), "t4", tmp_path, lan=True, fake=True)


class Sink:
    def __init__(self):
        self.msgs = []

    async def __call__(self, m):
        self.msgs.append(m)


async def progress(d, state, code=200, reason="OK"):
    for q in d._call_waiters:
        q.put_nowait(ev.OutboundProgress(7, "agents", state, code, reason))
    await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_emergency_numbers_are_refused_before_dialling(tmp_path, monkeypatch):
    d, tel, out = daemon(tmp_path, monkeypatch), Tel(), Sink()
    await d._control({"cmd": "call", "to": "+44999", "line": "agents"}, out, ctx(tel))
    assert out.msgs == [{"ok": False, "error": "+44999 matches an emergency number; ai-phone never dials those", "code": 13}]
    assert tel.calls == []


@pytest.mark.asyncio
async def test_daily_limit(tmp_path, monkeypatch):
    d, tel, out = daemon(tmp_path, monkeypatch, limit=1), Tel(), Sink()
    d.counter.record()
    await d._control({"cmd": "call", "to": "sip:x@127.0.0.1", "line": "agents"}, out, ctx(tel))
    assert out.msgs[-1]["code"] == 14 and tel.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("relay", [False, True])
async def test_disabled_outbound_calls_are_refused_before_dialling_or_counting(tmp_path, monkeypatch, relay):
    d, tel, out = daemon(tmp_path, monkeypatch, allow_outbound=False), Tel(), Sink()

    def unexpected():
        pytest.fail("a disabled outbound call must not reach the daily counter")

    monkeypatch.setattr(d.counter, "allowed", unexpected)
    monkeypatch.setattr(d.counter, "record", unexpected)
    await d._control({"cmd": "call", "to": "sip:x@127.0.0.1", "line": "agents", "relay": relay}, out, ctx(tel))
    assert out.msgs == [{"ok": False, "error": "outbound calls are disabled (calls.allow_outbound is false)", "code": 7}]
    assert tel.calls == [] and d.calls.current is None and d._call_waiters == [] and d._outbound is None


@pytest.mark.asyncio
@pytest.mark.parametrize("allow_inbound", [True, False])
async def test_relay_call_returns_once_connected_and_the_call_goes_on(tmp_path, monkeypatch, allow_inbound):
    d, tel, out = daemon(tmp_path, monkeypatch, allow_inbound=allow_inbound), Tel(), Sink()
    task = asyncio.create_task(d._control({"cmd": "call", "to": "sip:x@127.0.0.1", "line": "agents", "relay": True}, out, ctx(tel)))
    await asyncio.sleep(0.05)
    rec = d.calls.current
    assert rec.mode == "relay" and rec.pj_call_id == 7 and d.counter._read() == 1
    assert {"event": "created", "call_id": rec.id} in out.msgs
    await progress(d, "answered")
    assert not task.done()                       # answered, but no controller yet
    rec.mark_connected()
    await asyncio.wait_for(task, 2)
    assert out.msgs[-1] == {"ok": True, "event": "answered", "call_id": rec.id, "mode": "relay"}
    assert d.calls.current is rec and not rec.is_ended and d.active_call["call"] == rec.id


@pytest.mark.asyncio
async def test_busy_finishes_the_record_and_reports_it(tmp_path, monkeypatch):
    d, tel, out = daemon(tmp_path, monkeypatch), Tel(), Sink()
    task = asyncio.create_task(d._control({"cmd": "call", "to": "sip:x@127.0.0.1", "line": "agents", "prompt": "hi"}, out, ctx(tel)))
    await asyncio.sleep(0.05)
    rec = d.calls.current
    await progress(d, "busy", 486, "Busy Here")
    await asyncio.wait_for(task, 2)
    last = out.msgs[-1]
    assert last["event"] == "ended" and last["exit_code"] == 5 and last["result"]["outcome"] == "busy"
    assert rec.is_ended and d.calls.current is None and rec.prompt == "hi" and d.active_call is None


@pytest.mark.asyncio
async def test_agent_call_reports_the_finished_conversation(tmp_path, monkeypatch):
    d, tel, out = daemon(tmp_path, monkeypatch), Tel(), Sink()
    task = asyncio.create_task(d._control({"cmd": "call", "to": "sip:x@127.0.0.1", "line": "agents", "prompt": "hi"}, out, ctx(tel)))
    await asyncio.sleep(0.05)
    rec = d.calls.current
    await progress(d, "answered")
    rec.add_line("agent", "Hi, this is an AI assistant calling.")
    rec.finish("completed", ended_by="agent")           # what _run_call does when the conversation is over
    d.calls.end_current(rec)
    for q in d._call_waiters:
        q.put_nowait(("call_done", 7))
    await asyncio.wait_for(task, 2)
    last = out.msgs[-1]
    assert last["exit_code"] == 0 and last["call_id"] == rec.id and last["result"]["ended_by"] == "agent"
    assert last["result"]["transcript"][0]["text"] == "Hi, this is an AI assistant calling."


@pytest.mark.asyncio
async def test_hangup_while_ringing_is_cancelled(tmp_path, monkeypatch):
    d, tel, out, out2 = daemon(tmp_path, monkeypatch), Tel(), Sink(), Sink()
    task = asyncio.create_task(d._control({"cmd": "call", "to": "sip:x@127.0.0.1", "line": "agents"}, out, ctx(tel)))
    await asyncio.sleep(0.05)
    hang = asyncio.create_task(d._control({"cmd": "hangup"}, out2, ctx(tel)))
    await asyncio.sleep(0.05)
    assert tel.hung == [7]
    # what the endpoint posts for an unanswered outbound call: the outcome first (487, not a ring timeout), then `ended`
    await progress(d, "rejected", 487, "Request Terminated")
    d._on_call_ended(ev.CallStateChanged(7, "agents", "ended", 487, "Request Terminated"), tel)
    await asyncio.wait_for(asyncio.gather(task, hang), 2)
    assert out.msgs[-1]["reason"] == "cancelled" and out.msgs[-1]["exit_code"] == 7
    assert out.msgs[-1]["result"]["outcome"] == "cancelled" and out.msgs[-1]["detail"] == "487 Request Terminated"
    events = [m.get("event") for m in out.msgs]
    assert "cancelled" in events and "rejected" not in events
    assert out2.msgs[-1]["ok"] and out2.msgs[-1]["result"]["ended_by"] == "hangup_command"


@pytest.mark.asyncio
async def test_listen_waits_for_an_incoming_call_and_only_one_listener(tmp_path, monkeypatch):
    d, out1, out2 = daemon(tmp_path, monkeypatch), Sink(), Sink()
    first = asyncio.create_task(d._control({"cmd": "listen", "timeout_sec": 2}, out1, ctx(Tel())))
    await asyncio.sleep(0.05)
    await d._control({"cmd": "listen", "timeout_sec": 2}, out2, ctx(Tel()))
    assert out2.msgs[-1]["code"] == 12
    tel = Tel()
    rec = d._on_incoming(ev.IncomingCall(3, "agents", "sip:caller@10.0.0.2", "10.0.0.2"), tel)
    await asyncio.wait_for(first, 1)
    assert rec.mode == "relay" and tel.answered == [3]
    assert out1.msgs[-1] == {"ok": True, "kind": "incoming", "call_id": rec.id, "line": "agents", "remote": "sip:caller@10.0.0.2"}


@pytest.mark.asyncio
async def test_an_incoming_call_cancelled_before_answer_creates_no_record_or_active_state(tmp_path, monkeypatch):
    d, tel = daemon(tmp_path, monkeypatch), Tel()
    tel.answer_result = False
    writes = []
    monkeypatch.setattr(d, "_write_state", lambda: writes.append(d.active_call))

    rec = d._on_incoming(ev.IncomingCall(3, "agents", "sip:a@10.0.0.2", "10.0.0.2"), tel)

    assert rec is None and tel.answered == [3]
    assert d.calls.current is None and d.calls._records == {}
    assert d.active_call is None and d._controllers == {} and d._outbound is None
    assert writes == []


@pytest.mark.asyncio
async def test_an_incoming_call_is_answered_before_its_record_is_published(tmp_path, monkeypatch):
    d, tel = daemon(tmp_path, monkeypatch), Tel()
    answer = tel.answer

    def checked_answer(call_id):
        assert d.calls.current is None and d.calls._records == {} and d.active_call is None
        return answer(call_id)

    monkeypatch.setattr(tel, "answer", checked_answer)
    rec = d._on_incoming(ev.IncomingCall(3, "agents", "sip:a@10.0.0.2", "10.0.0.2"), tel)

    assert rec is d.calls.current and rec.pj_call_id == 3 and rec.mode == "agent"
    assert tel.answered == [3] and d.active_call["call"] == rec.id


@pytest.mark.asyncio
@pytest.mark.parametrize("ended_before_next_call", [True, False])
async def test_cancelled_incoming_call_keeps_relay_listener_for_the_next_call(
        tmp_path, monkeypatch, ended_before_next_call):
    d, tel, out = daemon(tmp_path, monkeypatch), CallTel(), Sink()
    listener = asyncio.create_task(d._control({"cmd": "listen", "timeout_sec": 2}, out, ctx(tel)))
    await asyncio.sleep(0)
    assert d.calls.listening
    cancelled = ev.CallStateChanged(3, "agents", "ended", 487, "Request Terminated")

    try:
        tel.answer_result = False
        assert d._on_incoming(ev.IncomingCall(3, "agents", "sip:a@10.0.0.2", "10.0.0.2"), tel) is None
        await asyncio.sleep(0)
        assert not listener.done() and d.calls.listening and out.msgs == []
        assert d.calls._records == {} and d.calls.current is None and d.active_call is None

        if ended_before_next_call:
            d._on_call_ended(cancelled, tel)
            assert tel.released == [3] and d.calls.unread_ended() is None
            assert d.calls.listening and not listener.done() and out.msgs == []

        tel.answer_result = True
        rec = d._on_incoming(ev.IncomingCall(4, "agents", "sip:b@10.0.0.3", "10.0.0.3"), tel)
        if not ended_before_next_call:
            d._on_call_ended(cancelled, tel)

        await asyncio.wait_for(listener, 1)
        assert tel.answered == [3, 4] and tel.released == [3]
        assert d.calls.current is rec and not rec.is_ended and rec.mode == "relay" and rec.pj_call_id == 4
        assert d.calls._records == {rec.id: rec} and d.calls.unread_ended() is None
        assert d.active_call["call"] == rec.id and not d.calls.listening
        assert out.msgs == [{"ok": True, "kind": "incoming", "call_id": rec.id,
                             "line": "agents", "remote": "sip:b@10.0.0.3"}]
    finally:
        listener.cancel()
        await asyncio.gather(listener, return_exceptions=True)


@pytest.mark.asyncio
async def test_disabled_inbound_listen_never_reserves_the_listener(tmp_path, monkeypatch):
    d, out = daemon(tmp_path, monkeypatch, allow_inbound=False), Sink()

    def unexpected():
        pytest.fail("inbound-disabled listen must not reserve a listener")

    monkeypatch.setattr(d.calls, "start_listening", unexpected)
    await d._control({"cmd": "listen", "timeout_sec": 3600}, out, ctx(Tel()))
    assert out.msgs == [{"ok": False, "error": "inbound calls are disabled (calls.allow_inbound is false)", "code": 7}]
    assert not d.calls.listening


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_call", [False, True])
async def test_outbound_only_listen_can_read_speech_and_the_ended_result(tmp_path, monkeypatch, explicit_call):
    d, out = daemon(tmp_path, monkeypatch, allow_inbound=False), Sink()
    rec = d.calls.new("relay", "out", "agents", "x")
    msg = {"cmd": "listen", "timeout_sec": 1}
    if explicit_call:
        msg["call_id"] = rec.id
    rec.heard("Hello?")
    await d._control(msg, out, ctx(Tel()))
    assert out.msgs[-1] == {"ok": True, "kind": "heard", "text": "Hello?"}
    rec.finish("completed", ended_by="them")
    d.calls.end_current(rec)
    await d._control(msg, out, ctx(Tel()))
    assert out.msgs[-1]["kind"] == "ended" and out.msgs[-1]["result"]["call_id"] == rec.id
    assert not d.calls.listening


@pytest.mark.asyncio
@pytest.mark.parametrize("allow_outbound", [True, False])
async def test_incoming_calls_without_a_listener_are_phone_agent_calls(tmp_path, monkeypatch, allow_outbound):
    d, tel = daemon(tmp_path, monkeypatch, allow_outbound=allow_outbound), Tel()
    rec = d._on_incoming(ev.IncomingCall(3, "agents", "sip:a@10.0.0.2", "10.0.0.2"), tel)
    assert rec.mode == "agent" and rec.direction == "in" and rec.pj_call_id == 3 and tel.answered == [3]
    assert d.active_call["call"] == rec.id and d.active_call["mode"] == "agent"


@pytest.mark.asyncio
async def test_listen_timeout_and_heard(tmp_path, monkeypatch):
    d, out = daemon(tmp_path, monkeypatch), Sink()
    await d._control({"cmd": "listen", "timeout_sec": 0.1}, out, ctx(Tel()))
    assert out.msgs[-1] == {"ok": True, "kind": "timeout"} and not d.calls.listening
    rec = d.calls.new("relay", "out", "agents", "x")
    rec.heard("Hello?")
    await d._control({"cmd": "listen", "timeout_sec": 1}, out, ctx(Tel()))
    assert out.msgs[-1] == {"ok": True, "kind": "heard", "text": "Hello?"}


@pytest.mark.asyncio
async def test_listen_reports_a_relay_call_that_ended_unseen(tmp_path, monkeypatch):
    d, out = daemon(tmp_path, monkeypatch), Sink()
    rec = d.calls.new("relay", "out", "agents", "x")
    rec.finish("completed", ended_by="them")
    d.calls.end_current(rec)
    await d._control({"cmd": "say", "text": "hi"}, out, ctx(Tel()))
    assert out.msgs[-1]["code"] == 11 and rec.id in out.msgs[-1]["error"]
    await d._control({"cmd": "listen", "timeout_sec": 0.1}, out, ctx(Tel()))
    assert out.msgs[-1]["kind"] == "ended" and out.msgs[-1]["result"]["call_id"] == rec.id
    await d._control({"cmd": "listen", "timeout_sec": 0.1}, out, ctx(Tel()))
    assert out.msgs[-1] == {"ok": True, "kind": "timeout"}


@pytest.mark.asyncio
async def test_listen_and_say_refuse_agent_calls_and_missing_calls(tmp_path, monkeypatch):
    d, out = daemon(tmp_path, monkeypatch), Sink()
    await d._control({"cmd": "say", "text": "hi"}, out, ctx(Tel()))
    assert out.msgs[-1] == {"ok": False, "error": "no call in progress", "code": 11}
    await d._control({"cmd": "say", "text": "hi", "call_id": "abcdef"}, out, ctx(Tel()))
    assert out.msgs[-1]["code"] == 11
    d.calls.new("agent", "out", "agents", "x")
    await d._control({"cmd": "say", "text": "hi"}, out, ctx(Tel()))
    assert out.msgs[-1]["code"] == 11 and "phone agent mode" in out.msgs[-1]["error"]
    await d._control({"cmd": "listen", "timeout_sec": 1}, out, ctx(Tel()))
    assert out.msgs[-1]["code"] == 11


@pytest.mark.asyncio
@pytest.mark.parametrize("allow_inbound", [True, False])
async def test_say_goes_to_the_calls_controller(tmp_path, monkeypatch, allow_inbound):
    d, out = daemon(tmp_path, monkeypatch, allow_inbound=allow_inbound), Sink()
    rec = d.calls.new("relay", "out", "agents", "x")
    rec.mark_connected()

    class TC:
        def say(self, text, end_reason=None):
            f = asyncio.get_running_loop().create_future()
            f.set_result("said")
            self.said = (text, end_reason)
            return f

    tc = TC()
    d._controllers[rec.id] = tc
    await d._control({"cmd": "say", "text": "Hello."}, out, ctx(Tel()))
    assert out.msgs[-1] == {"ok": True, "result": "said", "call_id": rec.id} and tc.said == ("Hello.", None)


@pytest.mark.asyncio
@pytest.mark.parametrize("allow_inbound", [True, False])
@pytest.mark.parametrize("mode", ["agent", "relay"])
async def test_hangup_marks_the_record_and_returns_its_result(tmp_path, monkeypatch, allow_inbound, mode):
    d, out, tel = daemon(tmp_path, monkeypatch, allow_inbound=allow_inbound), Sink(), Tel()
    await d._control({"cmd": "hangup"}, out, ctx(tel))
    assert out.msgs[-1]["code"] == 11
    rec = d.calls.new(mode, "out", "agents", "x")
    rec.pj_call_id = 7
    task = asyncio.create_task(d._control({"cmd": "hangup"}, out, ctx(tel)))
    await asyncio.sleep(0.05)
    assert tel.hung == [7] and rec.ended_by == "hangup_command"
    rec.finish("completed", ended_by="them")
    await asyncio.wait_for(task, 2)
    assert out.msgs[-1]["ok"] and out.msgs[-1]["result"]["ended_by"] == "hangup_command"


@pytest.mark.asyncio
async def test_unrouted_commands_never_reach_the_dial_path(tmp_path, monkeypatch):
    d, tel, out = daemon(tmp_path, monkeypatch), Tel(), Sink()
    await d._control({"cmd": "dial", "to": "sip:x@127.0.0.1", "line": "agents"}, out, ctx(tel))
    assert out.msgs[-1]["code"] == 2 and tel.calls == []


def test_ended_by_mapping():
    f = Daemon._ended_by
    assert f("agent", False) == "agent" and f("relay_timeout", False) == "timeout"
    assert f("silence", False) == "silence" and f("max_duration", False) == "max_duration"
    assert f("hangup_command", False) == "hangup_command" and f("rtp_timeout", False) == "error"
    assert f("worker error: llm", False) == "error" and f(None, True) == "them"


@pytest.mark.asyncio
async def test_status_reports_the_call_and_the_listener(tmp_path, monkeypatch):
    d, out = daemon(tmp_path, monkeypatch), Sink()
    rec = d.calls.new("relay", "in", "agents", "x")
    await d._control({"cmd": "status"}, out, ctx(Tel()))
    s = out.msgs[-1]
    assert s["ok"] and s["call"]["call_id"] == rec.id and s["relay_listener"] is False and "transcript" not in s["call"]


@pytest.mark.asyncio
async def test_a_listen_cancelled_while_replying_keeps_the_speech(tmp_path, monkeypatch):
    d = daemon(tmp_path, monkeypatch)
    rec = d.calls.new("relay", "out", "agents", "x")
    rec.heard("Hello?")
    stuck = asyncio.Event()

    async def slow_send(m):
        stuck.set()
        await asyncio.sleep(30)

    task = asyncio.create_task(d._control({"cmd": "listen", "timeout_sec": 1}, slow_send, ctx(Tel())))
    await asyncio.wait_for(stuck.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    out = Sink()
    await d._control({"cmd": "listen", "timeout_sec": 1}, out, ctx(Tel()))
    assert out.msgs[-1] == {"ok": True, "kind": "heard", "text": "Hello?"}


@pytest.mark.asyncio
async def test_a_listen_cancelled_while_reporting_an_end_reports_it_again(tmp_path, monkeypatch):
    d = daemon(tmp_path, monkeypatch)
    rec = d.calls.new("relay", "out", "agents", "x")
    rec.finish("completed", ended_by="them")
    d.calls.end_current(rec)
    stuck = asyncio.Event()

    async def slow_send(m):
        stuck.set()
        await asyncio.sleep(30)

    task = asyncio.create_task(d._control({"cmd": "listen", "timeout_sec": 1}, slow_send, ctx(Tel())))
    await asyncio.wait_for(stuck.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert d.calls.unread_ended() is rec and not d.calls.listening
    out = Sink()
    await d._control({"cmd": "listen", "timeout_sec": 0.1}, out, ctx(Tel()))
    assert out.msgs[-1]["kind"] == "ended" and out.msgs[-1]["result"]["call_id"] == rec.id


@pytest.mark.asyncio
async def test_a_call_cancelled_at_shutdown_cleans_up_and_finishes_its_record(tmp_path, monkeypatch):
    d, tel, out = daemon(tmp_path, monkeypatch), Tel(), Sink()
    task = asyncio.create_task(d._control({"cmd": "call", "to": "sip:x@127.0.0.1", "line": "agents"}, out, ctx(tel)))
    await asyncio.sleep(0.05)
    rec = d.calls.current
    assert d._outbound is not None and d.active_call is not None
    task.cancel()                                # what ControlServer.stop() does to running handlers
    with pytest.raises(asyncio.CancelledError):
        await task
    assert d._call_waiters == [] and d._outbound is None and d.active_call is None
    assert rec.is_ended and d.calls.current is None
    assert rec.outcome == "cancelled" and rec.ended_by == "error" and rec.detail == "ai-phone stopped"


@pytest.mark.asyncio
async def test_a_relay_call_handed_over_is_not_finished_by_its_call_command(tmp_path, monkeypatch):
    d, tel, out = daemon(tmp_path, monkeypatch), Tel(), Sink()
    task = asyncio.create_task(d._control({"cmd": "call", "to": "sip:x@127.0.0.1", "line": "agents", "relay": True}, out, ctx(tel)))
    await asyncio.sleep(0.05)
    rec = d.calls.current
    await progress(d, "answered")
    rec.mark_connected()
    await asyncio.wait_for(task, 2)
    assert not rec.is_ended and d.calls.current is rec and d._call_waiters == []


@pytest.mark.asyncio
async def test_busy_leaves_the_call_in_progress_alone(tmp_path, monkeypatch):
    from ai_phone.telephony.endpoint import BusyError

    class BusyTel(Tel):
        def make_call(self, line_id, to, timeout_sec):
            raise BusyError("a call is already active")

    d, out = daemon(tmp_path, monkeypatch), Sink()
    cur = d.calls.new("relay", "out", "agents", "sip:first@127.0.0.1")
    cur.pj_call_id = 3
    first_send = Sink()
    d._outbound = {"call_id": 3, "record": cur, "send": first_send}
    d.active_call = {"call_id": 3, "call": cur.id, "mode": "relay", "line_id": "agents", "remote": "sip:first@127.0.0.1",
                     "started_at": 1.0, "direction": "out"}
    await d._control({"cmd": "call", "to": "sip:x@127.0.0.1", "line": "agents", "relay": True}, out, ctx(BusyTel()))
    assert out.msgs == [{"ok": False, "error": "busy", "code": 5}]
    assert d.calls.current is cur and not cur.is_ended and d._outbound["record"] is cur
    assert d.active_call["call"] == cur.id and d._call_waiters == [] and d.counter._read() == 0


@pytest.mark.asyncio
async def test_an_inbound_call_that_ends_before_its_audio_is_finished(tmp_path, monkeypatch):
    d, tel = daemon(tmp_path, monkeypatch), Tel()
    rec = d._on_incoming(ev.IncomingCall(3, "agents", "sip:a@10.0.0.2", "10.0.0.2"), tel)
    d._on_call_ended(ev.CallStateChanged(3, "agents", "ended", 487, "Request Terminated"), tel)
    assert rec.is_ended and rec.ended_by == "them" and d.calls.current is None and d.active_call is None


class _Racing:
    """The asyncio module as daemon.py sees it, with one function replaced."""

    def __init__(self, wait_for):
        self.wait_for = wait_for

    def __getattr__(self, name):
        return getattr(asyncio, name)


# ---- _run_call with a stand-in controller ------------------------------------------------------------------
class FakeTimer:
    t0 = None
    bargein = False
    brain_error = None

    def to_record(self, call_id, n):
        return {"call_id": call_id, "turn": n, "first_audible_ms": 500.0, "bargein": False}

    def first_audible_ms(self):
        return 500.0


class FakeTC:
    """Takes one turn, then does what `script` says: end with a reason, or wait for the call to be stopped."""
    script = "agent"

    def __init__(self, *args, **kw):
        self.args, self.kw = args, kw
        self.hangup_requested = asyncio.Event()
        self.end_reason, self.timer, self.state = None, None, "LISTENING"
        FakeTC.last = self

    async def run(self, stop):
        self.args[12](FakeTimer())            # on_turn
        if FakeTC.script != "wait":
            self.end_reason = FakeTC.script
            self.hangup_requested.set()
        await stop.wait()


class SpeakingTC(FakeTC):
    """A controller that is still speaking when the call ends: the turn has started and has no record yet."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.recorded: list[tuple] = []

    async def run(self, stop):
        self.timer, self.state = FakeTimer(), "SPEAKING"
        self.timer.t0 = 1
        self.end_reason = "agent"
        self.hangup_requested.set()
        await stop.wait()

    def incomplete_record(self, call_id, n):
        self.recorded.append((call_id, n))
        return {"call_id": call_id, "turn": n, "reply_text": "Half a sen", "rx_overruns": 0, "tx_underruns": 2}


class Worker:
    current_gen = 1

    def __init__(self):
        self.sent = []

    def cancel(self, gen):
        pass

    def send(self, msg):
        self.sent.append(msg)


class Ring:
    def flush(self):
        pass


class TurnLog:
    def __init__(self):
        self.rows = []

    def write(self, r):
        self.rows.append(r)


class CallTel(Tel):
    port = None

    def __init__(self, pj_id=7):
        super().__init__(pj_id)
        self.released = []

    def release_call(self, call_id):
        self.released.append(call_id)
        return True


def run_ctx(tel):
    c = ctx(tel)
    c.update(stt=Worker(), llm=Worker(), tts=Worker(), rx=Ring(), tx=Ring(), stt_in=Ring(), tts_out=Ring(),
             cached={"prompt": [], "warning": [], "apology": [], "goodbye": []}, disclosure=[b"d"], greeting=[b"g"],
             relay_phrases=None, fixed="FIXED", turn_log=TurnLog(), stop=asyncio.Event())
    return c


@pytest.mark.asyncio
async def test_run_call_logs_turns_with_the_call_and_mode_and_finishes_the_record(tmp_path, monkeypatch):
    from ai_phone import daemon as daemon_mod
    from ai_phone.pipeline import prompts
    from ai_phone.pipeline.replies import ModelReplies

    monkeypatch.setattr(daemon_mod, "TurnController", FakeTC)
    monkeypatch.setattr(FakeTC, "script", "agent")
    d, tel = daemon(tmp_path, monkeypatch), CallTel()
    rec = d.calls.new("agent", "out", "agents", "sip:x@127.0.0.1", "Book a table for two.")
    rec.pj_call_id = 7
    d.active_call = {"call_id": 7, "call": rec.id}
    c, events = run_ctx(tel), []

    async def on_turn_event(r):
        events.append(r)

    await asyncio.wait_for(d._run_call(rec, c, on_turn_event=on_turn_event), 2)
    await asyncio.sleep(0)
    row = c["turn_log"].rows[0]
    assert row["call"] == rec.id and row["mode"] == "agent" and row["line_id"] == "agents" and "persona" not in row
    assert row["call_id"] == "7" and events and events[0]["call"] == rec.id
    assert rec.is_ended and rec.outcome == "completed" and rec.ended_by == "agent" and rec.detail == ""
    assert tel.hung == [7] and tel.released == [7] and d.calls.current is None and d._controllers == {}
    assert d.active_call is None
    tc = FakeTC.last
    assert tc.args[11] == prompts.system_prompt("FIXED", "Book a table for two.")
    # the brain is told this call's instructions before the greeting plays, so a session can be ready for the first
    # reply, and it lets that session go when the call ends rather than leaving a CLI idle between calls
    assert c["llm"].sent == [{"op": "llm.prepare", "gen": 1, "system_prompt": tc.args[11]},
                             {"op": "llm.release", "gen": 1}]
    assert tc.kw["generate_opening"] is True and tc.kw["opening_text"] == prompts.DISCLOSURE
    assert tc.kw["opening_frames"] == [b"d"] and tc.kw["direction"] == "out" and isinstance(tc.kw["replies"], ModelReplies)


@pytest.mark.asyncio
async def test_rtp_watchdog_reads_the_receive_port_when_playback_is_separate(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from ai_phone import daemon as daemon_mod

    monkeypatch.setattr(daemon_mod, "TurnController", FakeTC)
    monkeypatch.setattr(FakeTC, "script", "agent")
    d, tel = daemon(tmp_path, monkeypatch), CallTel()
    tel.port = SimpleNamespace(last_rx_ns=11)
    tel.receive_port = SimpleNamespace(last_rx_ns=22)
    rec = d.calls.new("agent", "out", "agents", "sip:x@127.0.0.1", "Say hello.")
    rec.pj_call_id = 7
    await asyncio.wait_for(d._run_call(rec, run_ctx(tel)), 2)
    received_at = FakeTC.last.kw["rtp_last_rx_ns"]
    assert received_at() == 22
    tel.receive_port.last_rx_ns = 33
    assert received_at() == 33


@pytest.mark.asyncio
async def test_run_call_on_an_inbound_relay_call_feeds_listen_and_ends_when_they_hang_up(tmp_path, monkeypatch):
    from ai_phone import daemon as daemon_mod
    from ai_phone.pipeline import prompts
    from ai_phone.pipeline.replies import RelayReplies

    monkeypatch.setattr(daemon_mod, "TurnController", FakeTC)
    monkeypatch.setattr(FakeTC, "script", "wait")
    d, tel = daemon(tmp_path, monkeypatch), CallTel(3)
    rec = d.calls.new("relay", "in", "agents", "sip:a@10.0.0.2")
    rec.pj_call_id = 3
    c = run_ctx(tel)
    run = asyncio.create_task(d._run_call(rec, c))
    assert await rec.wait_connected(1)
    tc = FakeTC.last
    assert d._controllers[rec.id] is tc and isinstance(tc.kw["replies"], RelayReplies)
    assert tc.kw["opening_frames"] is None and tc.kw["generate_opening"] is False
    assert tc.kw["opening_text"] == d.cfg.inbound.greeting and tc.args[10] == [b"g"]
    assert tc.args[11] == prompts.system_prompt("FIXED", d.cfg.inbound.instructions)
    tc.kw["replies"].on_heard(tc, "Hello?")   # what the controller does with a finished utterance
    assert await rec.next_event(0) == {"kind": "heard", "text": "Hello?"}
    c["events"].put_nowait(ev.CallStateChanged(9, "agents", "ended"))   # someone else's call: ignored
    await asyncio.sleep(0.2)
    assert not run.done()
    c["events"].put_nowait(ev.CallStateChanged(3, "agents", "ended"))
    await asyncio.wait_for(run, 2)
    assert rec.is_ended and rec.ended_by == "them" and tel.hung == [] and tel.released == [3]
    assert c["turn_log"].rows[0]["mode"] == "relay" and d._controllers == {} and d.calls.current is None


@pytest.mark.asyncio
async def test_run_call_stopped_by_the_daemon_ends_with_an_error(tmp_path, monkeypatch):
    from ai_phone import daemon as daemon_mod

    monkeypatch.setattr(daemon_mod, "TurnController", FakeTC)
    monkeypatch.setattr(FakeTC, "script", "wait")
    d, tel = daemon(tmp_path, monkeypatch), CallTel()
    rec = d.calls.new("relay", "out", "agents", "sip:x@127.0.0.1")
    rec.pj_call_id = 7
    c = run_ctx(tel)
    run = asyncio.create_task(d._run_call(rec, c))
    assert await rec.wait_connected(1)
    c["stop"].set()
    await asyncio.wait_for(run, 2)
    assert rec.ended_by == "error" and rec.detail == "ai-phone stopped" and d.calls.current is None


@pytest.mark.asyncio
async def test_a_call_that_arrives_as_listen_times_out_is_reported_not_dropped(tmp_path, monkeypatch):
    from ai_phone import daemon as daemon_mod

    d, out, tel = daemon(tmp_path, monkeypatch), Sink(), Tel()
    arrived = []

    async def racing_wait_for(aw, timeout):
        """The timeout fires, and before the listen resumes the main loop hands it an incoming call."""
        aw.cancel()
        arrived.append(d._on_incoming(ev.IncomingCall(3, "agents", "sip:a@10.0.0.2", "10.0.0.2"), tel))
        raise asyncio.TimeoutError

    with monkeypatch.context() as m:
        m.setattr(daemon_mod, "asyncio", _Racing(racing_wait_for))   # only daemon.py sees the patched wait_for
        await d._control({"cmd": "listen", "timeout_sec": 1}, out, ctx(tel))
    rec = arrived[0]
    assert rec.mode == "relay" and out.msgs[-1]["kind"] == "incoming" and out.msgs[-1]["call_id"] == rec.id
    assert not d.calls.listening


# ---- fix round 1 -------------------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_a_relay_call_answered_then_gone_before_its_audio_ends_at_once(tmp_path, monkeypatch):
    d, tel, out, lis = daemon(tmp_path, monkeypatch), Tel(), Sink(), Sink()
    task = asyncio.create_task(d._control({"cmd": "call", "to": "sip:x@127.0.0.1", "line": "agents", "relay": True}, out, ctx(tel)))
    await asyncio.sleep(0.05)
    rec = d.calls.current
    await progress(d, "answered")
    assert not task.done()
    d._on_call_ended(ev.CallStateChanged(7, "agents", "ended", 200, "Normal call clearing"), tel)  # no audio ever came
    await asyncio.wait_for(task, 0.5)
    last = out.msgs[-1]
    assert last["event"] == "ended" and last["exit_code"] == 0 and last["result"]["ended_by"] == "them"
    assert last["detail"] == "ended before the conversation started"
    assert rec.is_ended and d.calls.current is None and d._outbound is None and d.active_call is None
    assert d._call_waiters == []
    # the next incoming call goes to a waiting listener (the result went to `call`, so listen does not report it again)
    listener = asyncio.create_task(d._control({"cmd": "listen", "timeout_sec": 2}, lis, ctx(tel)))
    await asyncio.sleep(0.05)
    inc = d._on_incoming(ev.IncomingCall(8, "agents", "sip:a@10.0.0.2", "10.0.0.2"), tel)
    await asyncio.wait_for(listener, 1)
    assert inc.mode == "relay" and lis.msgs[-1] == {"ok": True, "kind": "incoming", "call_id": inc.id, "line": "agents",
                                                    "remote": "sip:a@10.0.0.2"}


@pytest.mark.asyncio
async def test_a_relay_call_waiting_for_its_audio_passes_other_events_by(tmp_path, monkeypatch):
    d, tel, out = daemon(tmp_path, monkeypatch), Tel(), Sink()
    task = asyncio.create_task(d._control({"cmd": "call", "to": "sip:x@127.0.0.1", "line": "agents", "relay": True}, out, ctx(tel)))
    await asyncio.sleep(0.05)
    rec = d.calls.current
    await progress(d, "answered")
    d._on_call_ended(ev.CallStateChanged(9, "agents", "ended", 486, "Busy Here"), tel)   # someone else's call
    await asyncio.sleep(0.05)
    assert not task.done() and d.calls.current is rec
    rec.mark_connected()
    await asyncio.wait_for(task, 1)
    assert out.msgs[-1] == {"ok": True, "event": "answered", "call_id": rec.id, "mode": "relay"}
    assert not rec.is_ended and d._call_waiters == []


@pytest.mark.asyncio
async def test_a_relay_record_finished_elsewhere_while_waiting_for_audio_ends_the_command(tmp_path, monkeypatch):
    d, tel, out = daemon(tmp_path, monkeypatch), Tel(), Sink()
    task = asyncio.create_task(d._control({"cmd": "call", "to": "sip:x@127.0.0.1", "line": "agents", "relay": True}, out, ctx(tel)))
    await asyncio.sleep(0.05)
    rec = d.calls.current
    await progress(d, "answered")
    rec.finish("completed", ended_by="them")
    d.calls.end_current(rec)
    await asyncio.wait_for(task, 0.5)
    assert out.msgs[-1]["event"] == "ended" and out.msgs[-1]["exit_code"] == 0


class SayTC:
    """A controller whose says never finish on their own."""

    def __init__(self):
        self.said = []

    def say(self, text, end_reason=None):
        self.said.append((text, end_reason))
        return asyncio.get_running_loop().create_future()


@pytest.mark.asyncio
async def test_hangup_say_queues_the_goodbye_with_the_hangup_reason(tmp_path, monkeypatch):
    d, out, tel = daemon(tmp_path, monkeypatch), Sink(), Tel()
    rec = d.calls.new("relay", "out", "agents", "x")
    rec.pj_call_id = 7
    tc = d._controllers[rec.id] = SayTC()
    task = asyncio.create_task(d._control({"cmd": "hangup", "say": "Goodbye."}, out, ctx(tel)))
    await asyncio.sleep(0.05)
    assert tc.said == [("Goodbye.", "hangup_command")] and tel.hung == [] and not task.done()
    rec.finish("completed", ended_by="agent")   # the controller played it and hung up
    await asyncio.wait_for(task, 1)
    assert out.msgs[-1]["ok"] and out.msgs[-1]["result"]["ended_by"] == "hangup_command" and rec.end_reported


@pytest.mark.asyncio
async def test_hangup_say_that_never_plays_hangs_up_anyway(tmp_path, monkeypatch):
    from ai_phone import daemon as daemon_mod

    monkeypatch.setattr(daemon_mod, "HANGUP_FIRST_WAIT_SEC", 0.1)
    d, out, tel = daemon(tmp_path, monkeypatch), Sink(), Tel()
    rec = d.calls.new("relay", "out", "agents", "x")
    rec.pj_call_id = 7
    d._controllers[rec.id] = SayTC()
    task = asyncio.create_task(d._control({"cmd": "hangup", "say": "Goodbye."}, out, ctx(tel)))
    await asyncio.sleep(0.05)
    assert tel.hung == []
    await asyncio.sleep(0.15)
    assert tel.hung == [7] and not task.done()
    rec.finish("completed", ended_by="them")
    await asyncio.wait_for(task, 1)
    assert out.msgs[-1]["ok"] and out.msgs[-1]["result"]["ended_by"] == "hangup_command"


@pytest.mark.asyncio
async def test_hangup_of_a_call_that_never_ends_is_an_error(tmp_path, monkeypatch):
    from ai_phone import daemon as daemon_mod

    monkeypatch.setattr(daemon_mod, "HANGUP_FIRST_WAIT_SEC", 0.05)
    monkeypatch.setattr(daemon_mod, "HANGUP_SECOND_WAIT_SEC", 0.05)
    d, out, tel = daemon(tmp_path, monkeypatch), Sink(), Tel()
    rec = d.calls.new("agent", "in", "agents", "x")
    rec.pj_call_id = 7
    await asyncio.wait_for(d._control({"cmd": "hangup"}, out, ctx(tel)), 1)
    assert tel.hung == [7, 7]
    assert out.msgs == [{"ok": False, "error": f"call {rec.id} did not end after the hangup", "code": 9}]
    assert not rec.is_ended and not rec.end_reported


def media_ctx(tel):
    return {"tel": tel, "specs": {}, "events": asyncio.Queue()}


@pytest.mark.asyncio
async def test_media_without_a_live_record_is_hung_up_and_never_run(tmp_path, monkeypatch):
    d, tel = daemon(tmp_path, monkeypatch), Tel()
    runs = []

    async def fake_run_call(rec, c, on_turn_event=None):
        runs.append(rec)

    monkeypatch.setattr(d, "_run_call", fake_run_call)
    await d._on_media_active(ev.MediaActive(5, "agents"), media_ctx(tel))      # no record at all
    dead = d.calls.new("relay", "out", "agents", "x")
    dead.pj_call_id = 7
    dead.finish("busy")
    d._outbound = {"call_id": 7, "record": dead, "send": Sink()}              # the call command is still reporting it
    q = asyncio.Queue()
    d._call_waiters.append(q)
    await d._on_media_active(ev.MediaActive(7, "agents"), media_ctx(tel))
    assert tel.hung == [5, 7] and runs == [] and q.empty()


@pytest.mark.asyncio
async def test_media_runs_the_call_and_streams_turns_only_for_outbound_agent_calls(tmp_path, monkeypatch):
    d, tel, send = daemon(tmp_path, monkeypatch), Tel(), Sink()
    runs = []

    async def fake_run_call(rec, c, on_turn_event=None):
        runs.append((rec, on_turn_event))
        if on_turn_event is not None:
            await on_turn_event({"turn": 0, "first_audible_ms": 480.0, "bargein": False, "call": rec.id})

    monkeypatch.setattr(d, "_run_call", fake_run_call)
    out_rec = d.calls.new("agent", "out", "agents", "x")
    out_rec.pj_call_id = 7
    d._outbound = {"call_id": 7, "record": out_rec, "send": send}
    q = asyncio.Queue()
    d._call_waiters.append(q)
    await d._on_media_active(ev.MediaActive(7, "agents"), media_ctx(tel))
    assert runs[0][0] is out_rec and send.msgs == [{"event": "turn", "turn": 0, "first_audible_ms": 480.0, "bargein": False}]
    assert q.get_nowait() == ("call_done", 7) and d._outbound is None and tel.hung == []
    out_rec.finish("completed")
    d.calls.end_current(out_rec)
    in_rec = d._on_incoming(ev.IncomingCall(3, "agents", "sip:a@10.0.0.2", "10.0.0.2"), tel)
    await d._on_media_active(ev.MediaActive(3, "agents"), media_ctx(tel))
    assert runs[1] == (in_rec, None) and q.get_nowait() == ("call_done", 3)


@pytest.mark.asyncio
async def test_a_call_command_that_fails_is_not_labelled_as_a_shutdown(tmp_path, monkeypatch):
    d, tel, out = daemon(tmp_path, monkeypatch), Tel(), Sink()
    seen = []
    real_write = d._write_state

    def write_fails_once():
        seen.append(d.calls.current)
        if len(seen) == 1:
            raise OSError("disk full")
        real_write()

    monkeypatch.setattr(d, "_write_state", write_fails_once)
    with pytest.raises(OSError):
        await d._control({"cmd": "call", "to": "sip:x@127.0.0.1", "line": "agents"}, out, ctx(tel))
    rec = seen[0]
    assert rec.is_ended and rec.outcome == "rejected" and rec.ended_by == "error"
    assert rec.detail == "the call command failed (OSError)" and d.calls.current is None
    assert d._outbound is None and d._call_waiters == [] and d.active_call is None
    assert tel.hung == [7]                       # the phone must not go on ringing with nobody behind the call


@pytest.mark.asyncio
async def test_a_call_whose_daily_count_cannot_be_written_is_hung_up_and_refused(tmp_path, monkeypatch):
    d, tel, out = daemon(tmp_path, monkeypatch, limit=5), Tel(), Sink()

    def record_fails():
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(d.counter, "record", record_fails)
    await asyncio.wait_for(d._control({"cmd": "call", "to": "sip:x@127.0.0.1", "line": "agents", "prompt": "hi"},
                                      out, ctx(tel)), 2)
    error = "could not update the daily call counter: [Errno 28] No space left on device"
    assert out.msgs == [{"ok": False, "error": error, "code": 7}]
    assert tel.calls == [("agents", "sip:x@127.0.0.1")] and tel.hung == [7]
    [rec] = d.calls._records.values()
    assert rec.is_ended and rec.outcome == "rejected" and rec.ended_by == "error" and rec.detail == error
    assert d.calls.current is None and d._outbound is None and d.active_call is None and d._call_waiters == []


@pytest.mark.asyncio
async def test_a_state_file_that_cannot_be_written_never_stops_an_incoming_call(tmp_path, monkeypatch):
    from ai_phone.telephony.state import StateFile

    def write_fails(self, lines, active_call, brain):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(StateFile, "write", write_fails)
    d, tel, lis = daemon(tmp_path, monkeypatch), Tel(), Sink()
    listener = asyncio.create_task(d._control({"cmd": "listen", "timeout_sec": 2}, lis, ctx(tel)))
    await asyncio.sleep(0.05)
    rec = d._on_incoming(ev.IncomingCall(3, "agents", "sip:a@10.0.0.2", "10.0.0.2"), tel)
    await asyncio.wait_for(listener, 1)
    assert tel.answered == [3] and rec.mode == "relay" and d.active_call["call"] == rec.id
    assert lis.msgs[-1] == {"ok": True, "kind": "incoming", "call_id": rec.id, "line": "agents", "remote": "sip:a@10.0.0.2"}


@pytest.mark.asyncio
async def test_run_call_releases_the_line_and_ends_the_record_before_writing_state(tmp_path, monkeypatch):
    from ai_phone import daemon as daemon_mod

    monkeypatch.setattr(daemon_mod, "TurnController", FakeTC)
    monkeypatch.setattr(FakeTC, "script", "agent")
    d, tel = daemon(tmp_path, monkeypatch), CallTel()
    rec = d.calls.new("agent", "in", "agents", "sip:a@10.0.0.2")
    rec.pj_call_id = 7

    def write_breaks():
        raise RuntimeError("state writer broke")

    monkeypatch.setattr(d, "_write_state", write_breaks)
    with pytest.raises(RuntimeError):
        await asyncio.wait_for(d._run_call(rec, run_ctx(tel)), 2)
    assert tel.released == [7] and rec.is_ended and d.calls.current is None and d.active_call is None


@pytest.mark.asyncio
async def test_an_incoming_call_with_blank_instructions_gets_the_inbound_default(tmp_path, monkeypatch):
    from ai_phone import daemon as daemon_mod
    from ai_phone.config import InboundConfig
    from ai_phone.pipeline import prompts

    monkeypatch.setattr(daemon_mod, "TurnController", FakeTC)
    monkeypatch.setattr(FakeTC, "script", "agent")
    d, tel = daemon(tmp_path, monkeypatch), CallTel(3)
    d.cfg.inbound.instructions = ""
    rec = d.calls.new("agent", "in", "agents", "sip:a@10.0.0.2")
    rec.pj_call_id = 3
    await asyncio.wait_for(d._run_call(rec, run_ctx(tel)), 2)
    composed = FakeTC.last.args[11]
    assert "calling on behalf" not in composed
    assert composed == prompts.system_prompt("FIXED", InboundConfig().instructions)


def test_the_llm_worker_runs_the_model_the_tier_names(tmp_path, monkeypatch):
    """One factory for every local brain: the catalog key (or a repo named by hand) becomes its model_id."""
    import ai_phone.daemon

    made = []

    class Recorder:
        def __init__(self, name, factory, *a, **kw):
            made.append((name, factory, kw.get("factory_args")))

        async def start(self):
            raise RuntimeError("far enough")

    monkeypatch.setenv("AI_PHONE_HOME", str(tmp_path))
    monkeypatch.setattr(ai_phone.daemon, "WorkerClient", Recorder)
    for tier, repo in (("gemma4b", "mlx-community/gemma-3-4b-it-4bit"), ("org/my-own-4bit", "org/my-own-4bit")):
        made.clear()
        with pytest.raises(SystemExit):
            asyncio.run(Daemon(Config(), tier, tmp_path, lan=True, fake=False).run())
        assert ("llm", "ai_phone.services.backends.mlx_lm_backend:make_llm", {"model_id": repo}) in made

    made.clear()
    with pytest.raises(SystemExit):
        asyncio.run(Daemon(Config(), "gemma12b", tmp_path, lan=True, fake=True).run())
    assert ("llm", "ai_phone.services.backends.fake:FakeLLM", None) in made  # the fake brain ignores the tier


@pytest.mark.parametrize("engine, voice, factory", [
    ("kokoro", "bf_emma", "ai_phone.services.backends.mlx_kokoro:KokoroTTS"),
    ("phonon", "default", "ai_phone.services.backends.mlx_phonon:PhononTTS"),
])
@pytest.mark.parametrize("fake", [False, True])
@pytest.mark.parametrize("codecs, output_rate", [(["PCMU", "PCMA"], 16000), (["opus", "G722", "PCMU", "PCMA"], 48000)])
def test_daemon_selects_the_configured_speech_backend(tmp_path, monkeypatch, engine, voice, factory, fake, codecs, output_rate):
    import ai_phone.daemon
    from ai_phone.config import TtsConfig

    made = []
    rings = {}

    class Recorder:
        def __init__(self, name, factory, *args, **kwargs):
            made.append((name, factory, kwargs.get("factory_args")))
            # args: loop, audio_in, audio_out, warm_args. Capture geometry before cleanup.
            rings[name] = tuple(r.frame_bytes if r is not None else None for r in args[1:3])

        async def start(self):
            raise RuntimeError("workers selected")

    monkeypatch.setenv("AI_PHONE_HOME", str(tmp_path))
    monkeypatch.setattr(ai_phone.daemon, "WorkerClient", Recorder)
    cfg = Config(tts=TtsConfig(engine=engine, voice=voice), codecs=codecs)
    with pytest.raises(SystemExit):
        asyncio.run(Daemon(cfg, "t4", tmp_path, lan=True, fake=fake).run())
    assert ("tts", "ai_phone.services.backends.fake:FakeTTS" if fake else factory,
            {"output_rate": output_rate} if fake else {"voice": voice, "speed": 1.0, "output_rate": output_rate}) in made
    assert rings["stt"] == (1028, None)
    assert rings["tts"] == (None, 4 + output_rate // 50 * 2)


@pytest.mark.asyncio
async def test_a_turn_still_speaking_when_the_call_ends_is_written_with_what_was_said(tmp_path, monkeypatch):
    """The teardown record goes through the controller, so it carries the agent's words and the ring counters; it
    used to be taken straight off the timer, and a call that ended mid-sentence lost the reply."""
    from ai_phone import daemon as daemon_mod

    monkeypatch.setattr(daemon_mod, "TurnController", SpeakingTC)
    d, tel = daemon(tmp_path, monkeypatch), CallTel()
    rec = d.calls.new("agent", "out", "agents", "sip:x@127.0.0.1", "Book a table for two.")
    rec.pj_call_id = 7
    c = run_ctx(tel)
    await asyncio.wait_for(d._run_call(rec, c), 2)
    tc = SpeakingTC.last
    assert tc.recorded == [("7", 0)], "the record did not go through the controller"
    row = c["turn_log"].rows[-1]
    assert row["reply_text"] == "Half a sen" and row["tx_underruns"] == 2
    assert row["incomplete"] is True and row["state_at_end"] == "SPEAKING"
    assert row["call"] == rec.id and row["line_id"] == "agents" and row["mode"] == "agent"

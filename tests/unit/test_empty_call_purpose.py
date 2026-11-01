"""No-task outbound calls have an explicit fallback; assigned and inbound tasks retain their purpose.

These check the instructions delivered to the brain, not a model's compliance with them.
"""
from __future__ import annotations

import asyncio

import pytest

from ai_phone.config import InboundConfig
from ai_phone.pipeline import prompts
from tests.unit.test_daemon_calls import CallTel, FakeTC, daemon, run_ctx


def assert_no_task_policy(composed):
    assert composed.endswith(prompts.CALL_SECTION_HEADER + prompts.DEFAULT_CALL_SECTION)
    assert "No specific task was provided" in composed
    assert "respond to what the person says and answer their questions" in composed
    assert "say you were not given a specific purpose" in composed
    assert "Do not invent a reason, business, message from the owner, or a test" in composed
    assert "Do not keep reintroducing the call" in composed


@pytest.mark.parametrize("section", [None, "", " ", "\t\n"])
def test_absent_or_whitespace_sections_use_an_honest_conversation_fallback(section):
    assert_no_task_policy(prompts.system_prompt(prompts.fixed_instructions(), section))


@pytest.mark.asyncio
@pytest.mark.parametrize("task", ["", " \t\n", "Ask whether they are open on Sunday."])
async def test_outbound_brain_preparation_receives_the_selected_call_purpose(tmp_path, monkeypatch, task):
    from ai_phone import daemon as daemon_mod

    monkeypatch.setattr(daemon_mod, "TurnController", FakeTC)
    monkeypatch.setattr(FakeTC, "script", "agent")
    d, tel = daemon(tmp_path, monkeypatch), CallTel()
    rec = d.calls.new("agent", "out", "agents", "sip:recipient@127.0.0.1", task)
    rec.pj_call_id = 7
    context = run_ctx(tel)

    await asyncio.wait_for(d._run_call(rec, context), 2)

    prepared = next(m["system_prompt"] for m in context["llm"].sent if m["op"] == "llm.prepare")
    assert prepared == FakeTC.last.args[11]
    assert FakeTC.last.kw["generate_opening"] is True
    if task.strip():
        assert prepared == "FIXED" + prompts.CALL_SECTION_HEADER + task
        assert prompts.DEFAULT_CALL_SECTION not in prepared
    else:
        assert_no_task_policy(prepared)


@pytest.mark.asyncio
@pytest.mark.parametrize("instructions", ["", " \t\n", "Ask who is calling and take a message for Alex."])
async def test_inbound_calls_keep_their_own_default_or_explicit_instructions(tmp_path, monkeypatch, instructions):
    from ai_phone import daemon as daemon_mod

    monkeypatch.setattr(daemon_mod, "TurnController", FakeTC)
    monkeypatch.setattr(FakeTC, "script", "agent")
    d, tel = daemon(tmp_path, monkeypatch), CallTel()
    d.cfg.inbound.instructions = instructions
    rec = d.calls.new("agent", "in", "agents", "sip:caller@127.0.0.1")
    rec.pj_call_id = 7

    await asyncio.wait_for(d._run_call(rec, run_ctx(tel)), 2)

    composed = FakeTC.last.args[11]
    expected = instructions.strip() or InboundConfig().instructions
    assert composed == "FIXED" + prompts.CALL_SECTION_HEADER + expected
    assert prompts.DEFAULT_CALL_SECTION not in composed
    assert FakeTC.last.kw["generate_opening"] is False

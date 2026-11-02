"""Which backend the daemon starts for each kind of brain, what it tells the person at start-up, and what it writes
to state.json. The workers themselves are not started here (tests/integration does that)."""
import asyncio
from pathlib import Path
import platform
import socket
from unittest.mock import Mock

import pytest

from ai_phone.config import BrainConfig, Config, ConversationConfig, InboundConfig, ModelsConfig
from ai_phone.daemon import Daemon, brain_warning, drain_timeout_ms, llm_worker, subscription_first_token_ms
from ai_phone.pipeline import prompts

PROMPT = "You are an AI assistant speaking on a live phone call."
WIRE = {"wire_memory": True}


def spec(kind, *, fake=False, tier="t4", **brain):
    cfg = Config(brain=BrainConfig(kind=kind, **brain), models=ModelsConfig(keep_warm_sec=30))
    return llm_worker(cfg, tier, fake, PROMPT, {"AI_PHONE_BRAIN_API_KEY": "sk-secret"}, dict(WIRE))


def test_a_brain_on_this_mac_loads_the_weights_the_tier_names():
    s = spec("local", tier="t9")
    assert s["factory"] == "ai_phone.services.backends.mlx_lm_backend:make_llm"
    assert s["factory_args"]["model_id"].count("/") == 1 and s["warm_args"] == {"system_prompt": PROMPT, **WIRE}
    assert s["keep_warm_sec"] == 30


def test_a_claude_subscription_runs_claude_code():
    s = spec("claude", model="haiku", cli="/opt/homebrew/bin/claude")
    assert s["factory"] == "ai_phone.services.backends.cli_brain:make_claude"
    assert s["factory_args"] == {"model": "haiku", "cli": "/opt/homebrew/bin/claude", "first_token_timeout_ms": 12000,
                                 "stream_idle_timeout_ms": 4000, "max_response_chars": 2000, "drain_timeout_ms": 4000}
    assert s["warm_args"] == {"system_prompt": PROMPT}  # no wire_memory: there are no weights on this Mac
    assert s["keep_warm_sec"] == 0.0                   # and nothing to keep warm


def test_a_cancelled_turn_may_take_a_third_of_the_callers_patience_to_finish():
    """Claude Code cannot be interrupted, so the next turn waits for the one the caller talked over. What is spent
    waiting comes out of the time this turn has before the controller apologises."""
    assert drain_timeout_ms(Config()) == 4000  # reply_timeout_sec 12
    assert drain_timeout_ms(Config(conversation=ConversationConfig(reply_timeout_sec=6))) == 2000
    assert spec("claude")["factory_args"]["drain_timeout_ms"] == 4000


def test_a_chatgpt_subscription_runs_codex():
    s = spec("chatgpt", model="gpt-5.6-luna")
    assert s["factory"] == "ai_phone.services.backends.cli_brain:make_codex"
    assert s["factory_args"]["model"] == "gpt-5.6-luna" and s["factory_args"]["cli"] is None


def test_a_plan_gets_twelve_seconds_for_its_first_word_unless_the_person_said_otherwise():
    assert subscription_first_token_ms(BrainConfig(kind="claude")) == 12000
    assert subscription_first_token_ms(BrainConfig(kind="claude", first_token_timeout_ms=6000)) == 6000
    assert spec("claude", first_token_timeout_ms=6000)["factory_args"]["first_token_timeout_ms"] == 6000


def test_an_endpoint_brain_is_the_server_the_config_names_even_with_fake_models():
    """--fake stands in for what runs on this Mac; an endpoint is a server the person chose."""
    for fake in (False, True):
        s = spec("openai_compatible", fake=fake, base_url="http://127.0.0.1:1234/v1", model="qwen",
                 api_key_env="AI_PHONE_BRAIN_API_KEY")
        assert s["factory"] == "ai_phone.services.backends.openai_compat:make_openai"
        assert s["factory_args"]["api_key"] == "sk-secret" and s["factory_args"]["base_url"].endswith("/v1")


@pytest.mark.parametrize("kind", ["local", "claude", "chatgpt"])
def test_fake_models_never_spend_a_turn_of_the_plan(kind):
    s = spec(kind, fake=True)
    assert s["factory"] == "ai_phone.services.backends.fake:FakeLLM"
    assert s["factory_args"] is None and s["warm_args"] == {"system_prompt": PROMPT}


def test_the_start_up_warning_says_where_the_words_go():
    assert brain_warning(Config()) is None
    assert brain_warning(Config(brain=BrainConfig(kind="openai_compatible", base_url="http://127.0.0.1:1234/v1"))) is None
    endpoint = brain_warning(Config(brain=BrainConfig(kind="openai_compatible", base_url="https://api.example.com/v1")))
    assert "api.example.com" in endpoint and "not audio" in endpoint
    claude = brain_warning(Config(brain=BrainConfig(kind="claude")))
    assert claude == ("Brain: your Claude subscription. What callers say, the instructions and the conversation are "
                      "sent to Anthropic through Claude Code (not audio).")
    chatgpt = brain_warning(Config(brain=BrainConfig(kind="chatgpt")))
    assert "to OpenAI through Codex (not audio)." in chatgpt and "ChatGPT subscription" in chatgpt


HOST_CANARY = "private-owner-host-canary"
ARCH_CANARY = "private-machine-architecture-canary"
BRAIN_KINDS = ["local", "openai_compatible", "claude", "chatgpt"]


@pytest.fixture
def identity_readers(monkeypatch):
    readers = []
    for module, name, value in ((platform, "node", HOST_CANARY),
                                (platform, "machine", ARCH_CANARY),
                                (socket, "gethostname", HOST_CANARY)):
        reader = Mock(return_value=value)
        monkeypatch.setattr(module, name, reader)
        readers.append(reader)
    return readers


def brain_config(kind):
    options = {"base_url": "https://brain.example.com/v1"} if kind == "openai_compatible" else {}
    return BrainConfig(kind=kind, **options)


@pytest.mark.parametrize("kind", BRAIN_KINDS)
def test_every_brain_warms_with_phone_policy_without_reading_host_identity(tmp_path, monkeypatch, identity_readers, kind):
    monkeypatch.setenv("AI_PHONE_HOME", str(tmp_path))
    cfg = Config(brain=brain_config(kind))
    daemon = Daemon(cfg, "t4", tmp_path, lan=True, fake=True)
    fixed = daemon.fixed_prompt()
    # Construct the real backend specification, without starting a worker or contacting the configured brain.
    worker = llm_worker(cfg, "t4", False, fixed, {}, dict(WIRE))

    assert daemon.remote_brain is (kind != "local")
    assert worker["warm_args"]["system_prompt"] == fixed == prompts.PHONE_INSTRUCTIONS
    for canary in (HOST_CANARY, ARCH_CANARY):
        assert canary not in repr(worker["warm_args"])
    for reader in identity_readers:
        reader.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", BRAIN_KINDS)
@pytest.mark.parametrize("direction", ["in", "out"])
async def test_real_call_prompt_construction_never_reads_or_sends_host_identity(
        tmp_path, monkeypatch, identity_readers, kind, direction):
    from ai_phone import daemon as daemon_mod
    from tests.unit.test_daemon_calls import CallTel, FakeTC, run_ctx

    monkeypatch.setenv("AI_PHONE_HOME", str(tmp_path))
    monkeypatch.setattr(daemon_mod, "TurnController", FakeTC)
    monkeypatch.setattr(FakeTC, "script", "agent")
    inbound = "Ask who is calling and take a message for Alex."
    outbound = "Ask whether they are open on Sunday."
    cfg = Config(brain=brain_config(kind), inbound=InboundConfig(instructions=inbound))
    daemon = Daemon(cfg, "t4", tmp_path, lan=True, fake=True)
    context = run_ctx(CallTel())
    context["fixed"] = daemon.fixed_prompt()  # use the real policy instead of run_ctx's generic FIXED stub
    record = daemon.calls.new("agent", direction, "agents", "sip:peer@127.0.0.1", outbound if direction == "out" else "")
    record.pj_call_id = 7

    await asyncio.wait_for(daemon._run_call(record, context), 2)

    prepared = [message["system_prompt"] for message in context["llm"].sent if message["op"] == "llm.prepare"]
    expected = prompts.PHONE_INSTRUCTIONS + prompts.CALL_SECTION_HEADER + (outbound if direction == "out" else inbound)
    assert prepared == [expected]
    assert FakeTC.last.args[11] == expected
    assert FakeTC.last.kw["generate_opening"] is (direction == "out")
    assert record.is_ended
    for canary in (HOST_CANARY, ARCH_CANARY):
        assert canary not in repr(context["llm"].sent)
        assert canary not in FakeTC.last.args[11]
    for reader in identity_readers:
        reader.assert_not_called()


def test_state_json_names_the_brain_the_daemon_is_using(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AI_PHONE_HOME", str(tmp_path))
    d = Daemon(Config(brain=BrainConfig(kind="claude", model="haiku")), "t4", tmp_path, lan=True, fake=False)
    d._write_state()
    assert d.state_file.read(d.state_file.path)["brain"] == {"kind": "claude", "base_url": None, "model": "haiku"}
    d = Daemon(Config(brain=BrainConfig(kind="openai_compatible", base_url="http://127.0.0.1:1234/v1", model="qwen")),
               "t4", tmp_path, lan=True, fake=False)
    d._write_state()
    assert d.state_file.read(d.state_file.path)["brain"] == {"kind": "openai_compatible",
                                                            "base_url": "http://127.0.0.1:1234/v1", "model": "qwen"}

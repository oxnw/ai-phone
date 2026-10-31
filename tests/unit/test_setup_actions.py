"""What the setup flow's actions write and check, against a temporary home: the fake carrier, a fake player, a fake
daemon process and a mock HTTP transport. Nothing here touches the network, plays sound or downloads."""
from __future__ import annotations

import json
import logging
import os
import signal
import stat
import subprocess
import sys
import time
import warnings
import wave
from pathlib import Path

import httpx
import numpy as np
import pytest
import yaml

from ai_phone import models, voices
from ai_phone.config import InboundConfig, load_config
from ai_phone.lines.journal import Journal
from ai_phone.providers.base import Number, ProviderError, VerificationStep
from ai_phone.providers.fake import OWNED, FakeProvider
from ai_phone.secrets import load_secrets, write_secret
from ai_phone.setup import actions as act
from ai_phone.setup.actions import BRAIN_KEY_ENV, ActionError, Actions, Current, KokoroPlayer, PhononPlayer, LineResult, Machine
from ai_phone.telephony.state import LineStatus, StateFile

FAKE_KEY = {"FAKE_API_KEY": "fake-key-SECRET"}
BOUGHT = "+442079460000"  # the fake carrier's first search result


class FakePlayer:
    def __init__(self, error: Exception | None = None):
        self.played: list[tuple] = []
        self.error = error

    def play(self, voice, speed, text):
        self.played.append((voice, speed, text))
        if self.error:
            raise self.error


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_PHONE_HOME", str(tmp_path))
    monkeypatch.delenv("AI_PHONE_FAKE_PENDING", raising=False)
    monkeypatch.setattr(models, "memory_gb", lambda: 24)
    monkeypatch.setattr(models, "local_path", lambda repo: None)  # nothing downloaded, and never the real cache
    monkeypatch.setattr(voices, "installed_voices", lambda root=None, engine="kokoro": ["default"] if engine == "phonon" else ["af_heart", "bf_emma", "bm_george"])
    monkeypatch.setattr(act, "probe", lambda path: False)
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    (vendor / "_pjsua2.cpython-313-darwin.so").write_bytes(b"")
    monkeypatch.setattr(act, "VENDOR", vendor)
    return tmp_path


def make(home, config: str | None = None, **kw) -> Actions:
    if config is not None:
        (home / "config.yaml").write_text(config)
    kw.setdefault("player", FakePlayer())
    kw.setdefault("provider_loader", lambda name, key, secret=None: FakeProvider(key=key, secret=secret))
    return Actions(home / "config.yaml", **kw)


def raw(home) -> dict:
    return yaml.safe_load((home / "config.yaml").read_text())


# ---- machine and current ---------------------------------------------------------------------

def test_machine_and_current_for_a_new_config(home, monkeypatch):
    a = make(home)
    assert a.machine() == Machine(sip_library=True, memory_gb=24, config_exists=False,
                                  models_missing=[models.STT, models.TTS, models.LLM["t4"]])
    inbound = InboundConfig()
    assert a.current() == Current(exists=False, lines={}, brain_kind="local", tier="t4", base_url="http://127.0.0.1:1234/v1",
                                  model="local", brain_key_saved=False, brain_cli=None, voice="af_heart", speed=1.0,
                                  greeting=inbound.greeting, instructions=inbound.instructions, daily_limit=20)
    cached = [models.STT, models.TTS, models.PHONON, models.LLM["t9"]]
    monkeypatch.setattr(models, "local_path", lambda repo: home if repo in cached else None)
    assert a.machine().models_downloaded == cached  # includes alternate voices and brains, even before a config exists
    assert a.machine().models_missing == [models.LLM["t4"]]
    a.save_brain_local("example/custom-model")
    cached.append("example/custom-model")
    assert a.machine().models_downloaded == cached
    assert a.machine().models_missing == []
    for so in act.VENDOR.iterdir():
        so.unlink()
    assert a.machine().sip_library is False


def test_current_reads_the_config(home):
    write_secret("BRAIN_TOKEN_KEY", "sk-saved")
    a = make(home, "models: {tier: t9}\n"
                   "brain: {kind: openai_compatible, base_url: 'https://api.example.com/v1', model: qwen, api_key_env: BRAIN_TOKEN_KEY}\n"
                   "tts: {voice: bm_george, speed: 0.9}\n"
                   "inbound: {greeting: Hello there., instructions: Take a message.}\n"
                   "calls: {daily_limit: 5}\n"
                   "lines:\n  agent: {kind: sip, user: agent, aor: 'sip:agent@192.0.2.10', accept_invites_from: lan}\n")
    cur = a.current()
    assert cur == Current(exists=True, lines={"agent": {"kind": "sip", "user": "agent", "aor": "sip:agent@192.0.2.10",
                                                        "accept_invites_from": "lan"}},
                          brain_kind="openai_compatible", tier="t9", base_url="https://api.example.com/v1", model="qwen",
                          brain_key_saved=True, brain_cli=None, voice="bm_george", speed=0.9, greeting="Hello there.",
                          instructions="Take a message.", daily_limit=5)
    assert a.machine().models_missing == [models.STT, models.TTS]  # an endpoint brain needs no local language model
    assert a.machine().config_exists
    assert make(home, "calls: {daily_limit: 0}\n").current().daily_limit == 0


def test_current_survives_a_config_that_does_not_load(home):
    a = make(home, "brain: [not, a, mapping\n")
    cur = a.current()
    assert cur.exists and cur.brain_kind == "local" and cur.lines == {}
    assert "not valid YAML" in a.check_config()
    a = make(home, "lines:\n  agent: {kind: sip, aor: nope}\n")
    assert a.current().lines == {"agent": {"kind": "sip", "aor": "nope"}}
    assert a.check_config() == "config.lines.agent.aor: expected sip:user@host[:port]"
    assert make(home, "inbound: {greeting: Hi}\n").check_config() is None


# ---- brain -----------------------------------------------------------------------------------

def test_save_brain_local_writes_its_section_and_keeps_the_others(home):
    a = make(home, "inbound: {greeting: Hi}\nmodels: {keep_warm_sec: 10}\n"
                   "brain: {kind: openai_compatible, base_url: 'https://api.example.com/v1', model: q, connect_timeout_ms: 900}\n")
    a.save_brain_local("t9")
    # the endpoint's own keys go; a timeout the person set by hand is theirs and stays, whichever brain they pick
    assert raw(home) == {"inbound": {"greeting": "Hi"}, "models": {"keep_warm_sec": 10, "tier": "t9"},
                         "brain": {"kind": "local", "connect_timeout_ms": 900}}
    assert stat.S_IMODE((home / "config.yaml").stat().st_mode) == 0o600
    assert load_config(home / "config.yaml").models.tier == "t9"


def test_save_brain_endpoint_saves_the_key_and_rolls_back(home):
    a = make(home, "inbound: {greeting: Hi}\n")
    a.save_brain_endpoint("https://api.example.com/v1", "qwen", "sk-new-SECRET")
    assert raw(home)["brain"] == {"kind": "openai_compatible", "base_url": "https://api.example.com/v1", "model": "qwen",
                                  "api_key_env": BRAIN_KEY_ENV}
    assert load_secrets()[BRAIN_KEY_ENV] == "sk-new-SECRET"
    assert a.current().brain_key_saved
    before = (home / "config.yaml").read_text()
    with pytest.raises(ActionError) as e:
        a.save_brain_endpoint("http://203.0.113.5/v1", "qwen", "sk-other-SECRET")
    assert "https" in str(e.value) and "SECRET" not in str(e.value)
    assert (home / "config.yaml").read_text() == before
    assert load_secrets()[BRAIN_KEY_ENV] == "sk-new-SECRET"  # the key that worked is put back
    a.save_brain_endpoint("https://api.example.com/v2", "local-model", None)  # no key typed, same server: keep the saved one
    assert raw(home)["brain"] == {"kind": "openai_compatible", "base_url": "https://api.example.com/v2", "model": "local-model",
                                  "api_key_env": BRAIN_KEY_ENV}
    assert raw(home)["inbound"] == {"greeting": "Hi"}


def test_save_brain_endpoint_without_a_key_on_a_new_config(home):
    a = make(home)
    with pytest.raises(ActionError, match="must use https"):
        a.save_brain_endpoint("http://203.0.113.5/v1", "qwen", None)
    assert not (home / "config.yaml").exists()
    assert BRAIN_KEY_ENV not in load_secrets()
    a.save_brain_endpoint("http://localhost:1234/v1", "qwen", None)
    assert raw(home) == {"brain": {"kind": "openai_compatible", "base_url": "http://localhost:1234/v1", "model": "qwen"}}
    assert not a.current().brain_key_saved


def test_a_config_that_is_not_yaml_is_never_overwritten(home):
    a = make(home, "brain: [not, a, mapping\n")
    with pytest.raises(ActionError, match="not valid YAML"):
        a.save_brain_local("t4")
    assert (home / "config.yaml").read_text() == "brain: [not, a, mapping\n"


def endpoint(home, handler, **kw):
    return make(home, http=httpx.Client(transport=httpx.MockTransport(handler)), **kw)


def test_endpoint_test_sends_one_short_request(home):
    seen = []

    def handler(req):
        seen.append(req)
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "OK"}}]})

    a = endpoint(home, handler)
    assert a.test_endpoint("https://api.example.com/v1/", "qwen", "sk-test-SECRET") == "OK"
    req = seen[0]
    assert str(req.url) == "https://api.example.com/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer sk-test-SECRET"
    assert json.loads(req.content) == {"model": "qwen", "messages": [{"role": "user", "content": "Say OK."}], "max_tokens": 5}
    assert a.test_endpoint("http://127.0.0.1:1234/v1", "qwen", None) == "OK"
    assert "authorization" not in seen[1].headers  # nothing saved: no key
    write_secret(BRAIN_KEY_ENV, "sk-saved-SECRET")
    (home / "config.yaml").write_text(f"brain: {{kind: openai_compatible, base_url: 'http://127.0.0.1:1234/v1', api_key_env: {BRAIN_KEY_ENV}}}\n")
    a.test_endpoint("http://127.0.0.1:1234/v1", "qwen", None)
    assert seen[2].headers["authorization"] == "Bearer sk-saved-SECRET"


def test_a_saved_key_goes_only_to_the_server_it_was_saved_for(home):
    seen = []

    def handler(req):
        seen.append(req)
        return httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}]})

    write_secret(BRAIN_KEY_ENV, "sk-saved-SECRET")
    a = endpoint(home, handler)
    (home / "config.yaml").write_text(f"brain: {{kind: openai_compatible, base_url: 'https://API.example.com/v1', "
                                      f"model: qwen, api_key_env: {BRAIN_KEY_ENV}}}\n")
    a.test_endpoint("https://api.example.com:443/v2", "qwen", None)  # the same host: the saved key
    a.test_endpoint("https://other.example.org/v1", "qwen", None)  # another host: no key
    assert seen[0].headers["authorization"] == "Bearer sk-saved-SECRET" and "authorization" not in seen[1].headers
    a.save_brain_endpoint("https://other.example.org/v1", "qwen", None)
    assert raw(home)["brain"] == {"kind": "openai_compatible", "base_url": "https://other.example.org/v1", "model": "qwen"}
    assert not a.current().brain_key_saved
    (home / "config.yaml").write_text(f"brain: {{kind: openai_compatible, base_url: 'https://api.example.com/v1', "
                                      f"model: qwen, api_key_env: {BRAIN_KEY_ENV}}}\n")
    a.save_brain_endpoint("https://api.example.com/v2", "qwen-2", None)
    assert raw(home)["brain"]["api_key_env"] == BRAIN_KEY_ENV


def test_endpoint_test_reports_what_the_endpoint_said_without_the_key(home):
    key = "sk-test-SECRET"
    a = endpoint(home, lambda req: httpx.Response(401, json={"error": {"message": f"Incorrect API key provided: {key}. " + "x" * 300}}))
    with pytest.raises(ActionError) as e:
        a.test_endpoint("https://api.example.com/v1", "qwen", key)
    msg = str(e.value)
    assert msg.startswith("the endpoint said 401: Incorrect API key provided: ***.")
    assert "SECRET" not in msg and len(msg) <= len("the endpoint said 401: ") + 160
    a = endpoint(home, lambda req: httpx.Response(404, text="no such route"))
    with pytest.raises(ActionError, match="^the endpoint said 404: no such route$"):
        a.test_endpoint("https://api.example.com/v1", "qwen", key)
    a = endpoint(home, lambda req: httpx.Response(200, text="<html>hello</html>"))
    with pytest.raises(ActionError, match="not like an OpenAI chat API"):
        a.test_endpoint("https://api.example.com/v1", "qwen", key)


def test_endpoint_test_never_sends_to_another_machine_over_plain_http(home):
    a = endpoint(home, lambda req: pytest.fail("nothing may be sent"))
    for key in ("sk-test-SECRET", None):
        with pytest.raises(ActionError) as e:
            a.test_endpoint("http://203.0.113.5/v1", "qwen", key)
        assert str(e.value) == "a brain that is not on this machine must use https"
    with pytest.raises(ActionError, match="must use https"):
        a.test_endpoint("http://api.example.com:8080/v1", "qwen", "sk-test-SECRET")


def test_endpoint_test_that_cannot_connect(home):
    def refuse(req):
        raise httpx.ConnectError("connection refused", request=req)

    a = endpoint(home, refuse)
    with pytest.raises(ActionError) as e:
        a.test_endpoint("http://127.0.0.1:1234/v1", "qwen", "sk-test-SECRET")
    assert str(e.value) == "could not reach 127.0.0.1: ConnectError"


# ---- carriers --------------------------------------------------------------------------------

def test_saved_keys_only_when_all_are_there(home):
    a = make(home)
    assert a.saved_keys("twilio") is None
    write_secret("TWILIO_ACCOUNT_SID", "ACsid")
    assert a.saved_keys("twilio") is None
    write_secret("TWILIO_AUTH_TOKEN", "tok")
    assert a.saved_keys("twilio") == {"TWILIO_ACCOUNT_SID": "ACsid", "TWILIO_AUTH_TOKEN": "tok"}
    assert a.saved_keys("telnyx") is None


def test_owned_and_search_use_the_typed_credentials(home):
    made = []

    def loader(name, key, secret=None):
        made.append((name, key, secret))
        return FakeProvider(key=key, secret=secret)

    a = make(home, provider_loader=loader)
    assert [n.e164 for n in a.owned_numbers("fake", FAKE_KEY)] == [OWNED]
    found = a.search_numbers("fake", FAKE_KEY, "GB", None)
    assert len(found) == 9 and found[0].e164 == BOUGHT
    make(home, provider_loader=loader).owned_numbers("twilio", {"TWILIO_ACCOUNT_SID": "ACsid", "TWILIO_AUTH_TOKEN": "tok"})
    assert made == [("fake", "fake-key-SECRET", None)] * 2 + [("twilio", "ACsid", "tok")]
    assert not (home / "secrets.env").exists()  # checking a key saves nothing


def test_a_key_that_does_not_work(home):
    class Refused(FakeProvider):
        def __init__(self, key="", secret=None):
            super().__init__(key=key, secret=secret)
            self.key = key

        def owned_numbers(self):
            raise ProviderError(f"telnyx GET /phone_numbers -> 401: bad key {self.key}", status=401)

        def search_numbers(self, country, area, limit):
            raise ProviderError("telnyx GET /available_phone_numbers: ConnectError", retryable=True)

    a = make(home, provider_loader=lambda name, key, secret=None: Refused(key=key))
    a_key = {"TELNYX_API_KEY": "KEY-SECRET"}
    with pytest.raises(ActionError) as e:
        a.owned_numbers("telnyx", a_key)
    assert str(e.value) == "That key did not work: telnyx GET /phone_numbers -> 401: bad key ***"
    with pytest.raises(ActionError, match="^Could not reach Telnyx: telnyx GET /available_phone_numbers: ConnectError$"):
        a.search_numbers("telnyx", a_key, "US", "212")

    def refused_token(name, key, secret=None):
        raise ProviderError("twilio rejected the Auth Token")

    with pytest.raises(ActionError, match="That key did not work"):
        make(home, provider_loader=refused_token).owned_numbers("twilio",
            {"TWILIO_ACCOUNT_SID": "AC", "TWILIO_AUTH_TOKEN": "invalid-token"})
    with pytest.raises(ActionError, match="Enter all Twilio credential fields"):
        make(home, provider_loader=refused_token).owned_numbers("twilio", {"TWILIO_ACCOUNT_SID": "AC"})


def test_buy_number_writes_a_ready_line_quietly(home, capsys):
    a = make(home, "inbound: {greeting: Hi}\n")
    n = a.search_numbers("fake", FAKE_KEY, "GB", None)[0]
    result = a.buy_number("fake", FAKE_KEY, n)
    assert result.line_id == "fake" and result.code == 0
    assert any(f"Line fake is ready: {BOUGHT}" in m for m in result.messages)
    ln = load_config(home / "config.yaml").lines["fake"]
    assert ln.number == BOUGHT and ln.provider.state == "ready" and ln.password_env == "AI_PHONE_LINE_FAKE_PASSWORD"
    saved = load_secrets()
    assert saved["FAKE_API_KEY"] == "fake-key-SECRET" and saved["AI_PHONE_LINE_FAKE_PASSWORD"]
    assert Journal.open(home / "lines", "fake").state == "ready"
    assert raw(home)["inbound"] == {"greeting": "Hi"}
    assert capsys.readouterr() == ("", "")
    other = a.search_numbers("fake", FAKE_KEY, "GB", None)[1]
    assert (a.buy_number("fake", FAKE_KEY, other).line_id, load_config(home / "config.yaml").lines["fake-2"].number) == \
        ("fake-2", other.e164)


def test_a_pending_or_failed_number_comes_back_with_its_messages(home):
    a = make(home, provider_loader=lambda name, key, secret=None: FakeProvider(key=key, pending_polls=1))
    n = a.search_numbers("fake", FAKE_KEY, "GB", None)[0]
    pending = a.buy_number("fake", FAKE_KEY, n)
    assert pending.code == 2 and any("pending" in m for m in pending.messages)
    assert load_config(home / "config.yaml").lines["fake"].provider.state == "pending"
    a = make(home, provider_loader=lambda name, key, secret=None: FakeProvider(key=key, crash_after="connection"))
    failed = a.buy_number("fake", FAKE_KEY, a.search_numbers("fake", FAKE_KEY, "GB", None)[1])
    assert failed == LineResult("fake-2", 3, failed.messages)
    assert "simulated crash after connection" in failed.messages[0] and "lines resume fake-2" in failed.messages[0]


def unfinished(home, line_id, number, brought):
    journal = Journal.open(home / "lines", line_id, "fake")
    journal.start("target", phone_number=number, brought=brought)
    journal.finish("target")


def test_a_stale_journal_never_blocks_a_new_line(home):
    """An earlier attempt that stopped part-way keeps its journal (for `lines resume` or `lines remove`); a new
    number gets an id of its own instead of being refused."""
    unfinished(home, "fake", "+442079460055", brought=False)
    (home / "lines" / "fake-2.journal.json").write_text("{not json")
    early = Journal.open(home / "lines", "fake-3", "fake")  # stopped before it recorded a number
    early.start("profile", name="ai-phone:fake-3")
    Journal.open(home / "lines", "fake-4", "fake")  # empty: it records nothing, so its id can be used
    a = make(home)
    result = a.buy_number("fake", FAKE_KEY, Number(BOUGHT, 1.0, "GBP", False, "src-0"))
    assert (result.line_id, result.code) == ("fake-4", 0)
    assert json.loads((home / "lines" / "fake.journal.json").read_text())["steps"][0]["planned"]["phone_number"] == "+442079460055"


def test_an_unfinished_connection_of_the_same_number_is_picked_up(home):
    unfinished(home, "other", OWNED, brought=True)  # any id: the journal is found by its number
    a = make(home)
    result = a.attach_number("fake", FAKE_KEY, a.owned_numbers("fake", FAKE_KEY)[0])
    assert (result.line_id, result.code) == ("other", 0)
    assert load_config(home / "config.yaml").lines["other"].number == OWNED


def test_a_number_set_up_the_other_way_is_never_set_up_again(home):
    """A purchase that stopped part-way, then the same number connected under a new id: `lines remove` of the first id
    would release the number the new line depends on. So the wizard stops with the way out instead."""
    unfinished(home, "fake", OWNED, brought=False)
    a = make(home)
    with pytest.raises(ActionError) as e:
        a.attach_number("fake", FAKE_KEY, a.owned_numbers("fake", FAKE_KEY)[0])
    assert str(e.value) == (f"line fake has an unfinished setup for {OWNED}; run `ai-phone lines resume fake` to finish it "
                            f"or `ai-phone lines remove fake` to undo it, then try again")
    unfinished(home, "fake-2", "+442079460001", brought=True)
    with pytest.raises(ActionError, match="^line fake-2 has an unfinished setup for \\+442079460001;"):
        a.buy_number("fake", FAKE_KEY, Number("+442079460001", 1.0, "GBP", False, "src-1"))
    old = home / "lines" / "old.journal.json"  # an older journal: no target step, so it cannot be picked up
    old.write_text(json.dumps({"line_id": "old", "provider": "fake", "state": "pending",
                               "steps": [{"name": "number", "status": "done", "planned": {"name": "+442079460002"}}]}))
    with pytest.raises(ActionError, match="^line old has an unfinished setup for \\+442079460002;"):
        a.buy_number("fake", FAKE_KEY, Number("+442079460002", 1.0, "GBP", False, "src-2"))
    assert not (home / "config.yaml").exists() and FakeProvider().created == {}


def test_a_setup_that_could_not_be_undone_is_never_reused(home):
    unfinished(home, "fake", OWNED, brought=True)
    Journal.open(home / "lines", "fake", "fake").set_state("rollback_failed")
    a = make(home)
    with pytest.raises(ActionError) as e:
        a.attach_number("fake", FAKE_KEY, a.owned_numbers("fake", FAKE_KEY)[0])
    assert str(e.value) == (f"An earlier setup of {OWNED} (line fake) could not be fully undone. Finish removing it in your "
                            f"Fake account, then run `ai-phone lines remove fake --local-only`.")


def test_unfinished_numbers(home):
    a = make(home)
    assert a.unfinished_numbers() == set()
    unfinished(home, "fake", OWNED, brought=True)
    unfinished(home, "gone", "+442079460003", brought=False)
    Journal.open(home / "lines", "empty", "fake")
    (home / "lines" / "broken.journal.json").write_text("{not json")
    a.buy_number("fake", FAKE_KEY, Number(BOUGHT, 1.0, "GBP", False, "src-0"))  # a journal with its line: not unfinished
    assert a.unfinished_numbers() == {OWNED, "+442079460003"}


def test_a_refused_number_comes_back_with_the_reason(home):
    a = make(home)
    n = Number(BOUGHT, 1.0, "GBP", False, "src-0")
    assert a.buy_number("fake", FAKE_KEY, n).code == 0
    refused = a.attach_number("fake", FAKE_KEY, n)
    assert (refused.line_id, refused.code) == ("fake-2", 3)
    assert refused.messages == [f"error: {BOUGHT} is already line fake"]
    assert not (home / "lines" / "fake-2.journal.json").exists()


def test_a_purchase_that_breaks_after_its_journal_says_how_to_finish_it(home, monkeypatch):
    import ai_phone.setup.actions as actions_module

    def full(*a, **k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(actions_module, "provision_flow", full)
    a = make(home)
    with pytest.raises(ActionError) as e:
        a.buy_number("fake", FAKE_KEY, Number(BOUGHT, 1.0, "GBP", False, "src-0"))
    assert str(e.value) == ("[Errno 28] No space left on device\nThe journal is kept: run `ai-phone lines resume fake` to "
                            "finish it, or `ai-phone lines remove fake` to undo it.")
    assert e.value.line_id == "fake"


def test_nothing_is_bought_into_a_config_that_does_not_load(home):
    providers = []

    def loader(name, key, secret=None):
        providers.append(FakeProvider(key=key))
        return providers[-1]

    a = make(home, "lines:\n  agent: {kind: sip, aor: nope}\n", provider_loader=loader)
    with pytest.raises(ActionError, match="config.lines.agent.aor"):
        a.buy_number("fake", FAKE_KEY, Number(BOUGHT, 1.0, "GBP", False, "src-0"))
    assert providers[0].created == {}
    assert not (home / "lines" / "fake.journal.json").exists()


def test_attach_number_connects_the_owned_number(home):
    a = make(home)
    owned = a.owned_numbers("fake", FAKE_KEY)[0]
    result = a.attach_number("fake", FAKE_KEY, owned)
    assert (result.line_id, result.code) == ("fake", 0)
    assert load_config(home / "config.yaml").lines["fake"].number == OWNED
    assert Journal.open(home / "lines", "fake").resources()["number_brought"] is True
    stranger = a.attach_number("fake", FAKE_KEY, Number("+15550100009", None, None, False, "x"))
    assert stranger.code == 3 and "not on this fake account" in stranger.messages[0]


def test_forward_adds_the_mobile_and_returns_the_codes(home):
    a = make(home)
    a.buy_number("fake", FAKE_KEY, Number(BOUGHT, 1.0, "GBP", False, "src-0"))
    codes = a.forward("fake", "+447700900123", "busy")
    assert codes == {"activate": f"**67*{BOUGHT}#", "check": "*#67#", "cancel": "##67#", "cancel_all": "##002#"}
    a.forward("fake", "+447700900123", "all")
    a.forward("fake", "+447700900111", "no-answer")
    assert raw(home)["lines"]["fake"]["diverted_from"] == ["+447700900111", "+447700900123"]
    a.add_sip_line("agent", "192.0.2.10")
    with pytest.raises(ActionError, match="no phone number"):
        a.forward("agent", "+447700900123", "busy")


def test_caller_id_verification_records_the_number_and_calls_out_as_it(home):
    """The point of the step is that people see the person's own number, so the verified number becomes the
    line's caller_id as well as one of its verified ids."""
    a = make(home)
    a.buy_number("fake", FAKE_KEY, Number(BOUGHT, 1.0, "GBP", False, "src-0"))
    step = a.start_caller_id("fake", FAKE_KEY, "+447700900123")
    assert step.kind == "enter_code_in_cli" and "+447700900123" in step.instructions
    assert a.check_caller_id("fake", FAKE_KEY, step, "000000") is False
    assert a.check_caller_id("fake", FAKE_KEY, step, "123456") is True
    a.record_caller_id("fake", "+447700900123", "fake")
    ln = load_config(home / "config.yaml").lines["fake"]
    assert ln.caller_id == "+447700900123" and ln.number == BOUGHT
    verified = raw(home)["lines"]["fake"]["verified_caller_ids"]
    assert [(v["number"], v["provider"]) for v in verified] == [("+447700900123", "fake")]
    assert verified[0]["verified_at"].endswith("Z")
    a.record_caller_id("fake", "+447700900123", "fake")  # again: the entry is replaced, never doubled
    assert len(raw(home)["lines"]["fake"]["verified_caller_ids"]) == 1


def test_a_carrier_that_refuses_the_verification_says_so_without_the_key(home):
    class Refuses(FakeProvider):
        def start_caller_id_verification(self, e164):
            raise ProviderError(f"rejected (key {FAKE_KEY['FAKE_API_KEY']})")

        def complete_caller_id_verification(self, step, code):
            raise ProviderError(f"gone (key {FAKE_KEY['FAKE_API_KEY']})", retryable=True)

    a = make(home, provider_loader=lambda name, key, secret=None: Refuses(key=key))
    with pytest.raises(ActionError) as start:
        a.start_caller_id("fake", FAKE_KEY, "+447700900123")
    assert "Could not start the check" in str(start.value) and FAKE_KEY["FAKE_API_KEY"] not in str(start.value)
    with pytest.raises(ActionError) as check:
        a.check_caller_id("fake", FAKE_KEY, VerificationStep("enter_code_in_cli", "", {"e164": "+447700900123"}), "123456")
    assert "Could not reach Fake" in str(check.value) and FAKE_KEY["FAKE_API_KEY"] not in str(check.value)


def test_record_caller_id_needs_a_line(home):
    a = make(home, "inbound: {greeting: Hi}\n")
    with pytest.raises(ActionError, match="no line ghost"):
        a.record_caller_id("ghost", "+447700900123", "fake")


def caller_id_actions(home, **line_settings):
    line = {"kind": "trunk", "number": BOUGHT, "caller_id": "+15555550100", "user": "mainuser",
            "registrar": "sip:trunk.example.org", "dial_domain": "trunk.example.org",
            "provider": {"name": "fake", "state": "ready", "resources": {"number_id": "number-1"}},
            "verified_caller_ids": [{"number": "+15555550100", "provider": "fake", "verified_at": "2026-09-10T12:00:00Z"}],
            **line_settings}
    return make(home, yaml.safe_dump({
        "lines": {"main": line, "agent": {"kind": "sip", "user": "agent", "aor": "sip:agent@example.org",
                                            "accept_invites_from": "lan"}},
        "inbound": {"greeting": "Hello there."}, "calls": {"daily_limit": 7, "allow_inbound": False},
    }))


def test_use_caller_id_switches_both_ways_without_keys_or_carrier_requests(home, monkeypatch):
    a = caller_id_actions(home)
    before = raw(home)

    def forbidden(*args, **kwargs):
        pytest.fail("selecting an existing caller ID must not read secrets or contact a carrier")

    monkeypatch.setattr(act, "load_secrets", forbidden)
    monkeypatch.setattr(a, "_provider", forbidden)
    a.use_caller_id("main", BOUGHT)
    expected = before.copy()
    expected["lines"]["main"]["caller_id"] = BOUGHT
    assert raw(home) == expected  # includes verification history, carrier resources, other lines and settings
    assert stat.S_IMODE((home / "config.yaml").stat().st_mode) == 0o600
    a.use_caller_id("main", "+15555550100")
    expected["lines"]["main"]["caller_id"] = "+15555550100"
    assert raw(home) == expected
    assert load_config(home / "config.yaml").lines["main"].caller_id == "+15555550100"


@pytest.mark.parametrize("action", ["use_caller_id", "record_caller_id"])
def test_account_from_cannot_save_an_ineffective_caller_id(home, action):
    a = caller_id_actions(home, from_user="account")
    before = (home / "config.yaml").read_bytes()
    args = ("main", BOUGHT, "fake") if action == "record_caller_id" else ("main", BOUGHT)
    with pytest.raises(ActionError, match="configure the caller ID with your provider"):
        getattr(a, action)(*args)
    assert (home / "config.yaml").read_bytes() == before


@pytest.mark.parametrize("provider", [None, {"name": "fake", "state": "ready"}])
def test_use_caller_id_keeps_an_omitted_verification_list_omitted(home, provider):
    a = caller_id_actions(home, provider=provider)
    content = raw(home)
    del content["lines"]["main"]["verified_caller_ids"]
    (home / "config.yaml").write_text(yaml.safe_dump(content))
    a.use_caller_id("main", BOUGHT)
    content["lines"]["main"]["caller_id"] = BOUGHT
    assert raw(home) == content


@pytest.mark.parametrize("provider", [{}, {"name": "fake"}, {"name": "fake", "state": "pending"},
                                    {"name": "fake", "state": "failed"}, {"name": "fake", "state": "rollback_failed"}])
def test_use_caller_id_refuses_a_line_whose_provider_is_not_ready(home, provider):
    a = caller_id_actions(home, provider=provider)
    before = (home / "config.yaml").read_bytes()
    with pytest.raises(ActionError, match="not ready; finish its setup"):
        a.use_caller_id("main", BOUGHT)
    assert (home / "config.yaml").read_bytes() == before


@pytest.mark.parametrize(("line_id", "number", "message"), [
    ("missing", BOUGHT, "no line missing"),
    ("agent", BOUGHT, "uses a SIP address"),
    ("main", "+15555550101", "not verified for line main"),
    ("main", "sip:agent@example.org", "E.164 format"),
    ("main", "", "E.164 format"),
])
def test_use_caller_id_rejects_invalid_selections_without_writing(home, line_id, number, message):
    a = caller_id_actions(home)
    before = (home / "config.yaml").read_bytes()
    with pytest.raises(ActionError, match=message):
        a.use_caller_id(line_id, number)
    assert (home / "config.yaml").read_bytes() == before


@pytest.mark.parametrize("verified", [[{"number": "+15555550100", "provider": "telnyx"}],
                                    [{"number": "+15555550100"}], []])
def test_use_caller_id_requires_verification_for_this_lines_carrier_even_when_already_selected(home, verified):
    a = caller_id_actions(home, verified_caller_ids=verified)
    before = (home / "config.yaml").read_bytes()
    with pytest.raises(ActionError, match="not verified for line main"):
        a.use_caller_id("main", "+15555550100")
    assert (home / "config.yaml").read_bytes() == before


def test_use_caller_id_does_not_borrow_verification_from_another_line(home):
    a = caller_id_actions(home, verified_caller_ids=[])
    content = raw(home)
    content["lines"]["other"] = {**content["lines"]["main"], "user": "otheruser",
                                  "verified_caller_ids": [{"number": "+15555550100", "provider": "fake"}]}
    (home / "config.yaml").write_text(yaml.safe_dump(content))
    with pytest.raises(ActionError, match="not verified for line main"):
        a.use_caller_id("main", "+15555550100")
    assert raw(home) == content


@pytest.mark.parametrize("explicit_null", [False, True])
def test_use_caller_id_on_a_manual_trunk_matches_empty_provider_verification(home, explicit_null):
    a = caller_id_actions(home, provider=None, verified_caller_ids=[{"number": "+15555550100"}])
    content = raw(home)
    if not explicit_null:
        del content["lines"]["main"]["provider"]
    (home / "config.yaml").write_text(yaml.safe_dump(content))
    a.use_caller_id("main", BOUGHT)
    a.use_caller_id("main", "+15555550100")
    assert raw(home) == content
    content["lines"]["main"]["verified_caller_ids"][0]["provider"] = "fake"
    (home / "config.yaml").write_text(yaml.safe_dump(content))
    with pytest.raises(ActionError, match="not verified for line main"):
        a.use_caller_id("main", "+15555550100")
    assert raw(home) == content


def test_use_caller_id_rolls_back_if_the_rest_of_the_config_is_invalid(home):
    a = caller_id_actions(home)
    content = raw(home)
    content["calls"]["daily_limit"] = -1
    (home / "config.yaml").write_text(yaml.safe_dump(content))
    with pytest.raises(ActionError, match="daily_limit"):
        a.use_caller_id("main", BOUGHT)
    assert raw(home) == content


# ---- lines by hand ---------------------------------------------------------------------------

def test_lan_address(home, monkeypatch):
    a = make(home)
    address = a.lan_address()
    assert address.count(".") == 3

    def unreachable(self, addr):
        raise OSError("Network is unreachable")

    monkeypatch.setattr(act.socket.socket, "connect", unreachable)
    assert a.lan_address() == "127.0.0.1"


def test_add_sip_line_refuses_an_address_that_is_taken(home):
    """Two SIP lines with one address: the endpoint gives every call to the first, so the second would never ring."""
    a = make(home, "lines:\n  desk: {kind: trunk, number: '+15550100001', registrar: 'sip:sip.example.com', "
                   "dial_domain: sip.example.com, user: desk1234}\n")
    assert a.add_sip_line("agent", "192.0.2.10") == "agent"
    before = (home / "config.yaml").read_text()
    with pytest.raises(ActionError) as e:
        a.add_sip_line("agent", "192.0.2.11")
    assert str(e.value) == "sip:agent@192.0.2.10 is already line agent; pick another name"
    assert (home / "config.yaml").read_text() == before
    assert a.add_sip_line("desk", "192.0.2.10") == "desk-2"  # a trunk's id is not an address: only the id moves on
    cfg = load_config(home / "config.yaml")
    assert cfg.lines["agent"].aor == "sip:agent@192.0.2.10" and cfg.lines["desk-2"].aor == "sip:desk@192.0.2.10"
    (home / "config.yaml").write_text(before.replace("lines:", "lines:\n  old: {kind: sip, aor: 'sip:front@192.0.2.10', "
                                                     "accept_invites_from: lan}"))
    with pytest.raises(ActionError, match="^sip:front@192.0.2.10 is already line old; pick another name$"):
        a.add_sip_line("front", "192.0.2.10")  # a line without `user` is known by its address
    before = raw(home)
    with pytest.raises(ActionError):
        a.add_sip_line("desk", "not a host")
    assert raw(home) == before


def test_add_trunk_line(home):
    a = make(home)
    assert a.add_trunk_line("sip:sip.example.com", "alex1234", "pw-SECRET", "+15550100000", "tls") == "trunk"
    ln = load_config(home / "config.yaml").lines["trunk"]
    assert (ln.registrar, ln.dial_domain, ln.user, ln.number, ln.transport) == \
        ("sip:sip.example.com", "sip.example.com", "alex1234", "+15550100000", "tls")
    assert ln.password_env == "AI_PHONE_LINE_TRUNK_PASSWORD"
    assert load_secrets()["AI_PHONE_LINE_TRUNK_PASSWORD"] == "pw-SECRET"
    assert "pw-SECRET" not in (home / "config.yaml").read_text()


@pytest.mark.parametrize("number", ["+15555550100", ""])
def test_add_trunk_with_account_from_can_omit_a_phone_number(home, number):
    a = make(home)
    assert a.add_trunk_line("sip:trunk.example.org", "account1", "pw-SECRET", number, "udp",
                            from_user="account") == "trunk"
    line = load_config(home / "config.yaml").lines["trunk"]
    assert line.from_user == "account" and line.number == (number or None)
    assert raw(home)["lines"]["trunk"]["from_user"] == "account"


def test_a_bad_trunk_leaves_no_secret_behind(home):
    a = make(home)
    with pytest.raises(ActionError) as e:
        a.add_trunk_line("sip:bad host", "alex1234", "pw-SECRET", "+15550100000", "udp")
    assert "registrar" in str(e.value) and "SECRET" not in str(e.value)
    assert "AI_PHONE_LINE_TRUNK_PASSWORD" not in load_secrets()
    assert raw(home) == {"lines": {}}
    with pytest.raises(ActionError, match="user"):
        a.add_trunk_line("sip:sip.example.com", "a", "pw-SECRET", "+15550100000", "udp")
    assert "AI_PHONE_LINE_TRUNK_PASSWORD" not in load_secrets()


# ---- voice -----------------------------------------------------------------------------------

def test_voices_and_the_sample(home):
    player = FakePlayer()
    a = make(home, player=player)
    assert a.voices() == ["af_heart", "bf_emma", "bm_george"]
    a.play_sample("bf_emma", 1.1, "Hello.")
    assert player.played == [("bf_emma", 1.1, "Hello.")]
    a = make(home, player=FakePlayer(subprocess.CalledProcessError(1, ["afplay"])))
    with pytest.raises(ActionError, match="^could not play the sample: "):
        a.play_sample("bf_emma", 1.0, "Hello.")
    a = make(home, player=FakePlayer(ActionError("could not play the sample: the speech model is not downloaded")))
    with pytest.raises(ActionError, match="^could not play the sample: the speech model is not downloaded$"):
        a.play_sample("bf_emma", 1.0, "Hello.")


def test_pull_voice_model(home, monkeypatch):
    pulls = []
    monkeypatch.setattr(models, "pull", lambda repos, report=None: pulls.append((repos, report)))
    report = lambda done, total: None  # noqa: E731
    make(home).pull_voice_model(report)
    assert pulls == [([models.TTS], report)]

    def fail(repos, report=None):
        raise models.ModelsError("not enough disk space: 0.4 GB to download and 0.1 GB free (keep 1 GB spare)")

    monkeypatch.setattr(models, "pull", fail)
    with pytest.raises(ActionError, match="^not enough disk space"):
        make(home).pull_voice_model(report)


def test_save_voice(home):
    a = make(home, "tts: {speed: 1.2}\ninbound: {greeting: Hi}\n")
    a.save_voice("bm_george", 0.9)
    assert raw(home) == {"tts": {"engine": "kokoro", "speed": 0.9, "voice": "bm_george"}, "inbound": {"greeting": "Hi"}}
    before = (home / "config.yaml").read_text()
    with pytest.raises(ActionError, match="bm_nobody is not installed"):
        a.save_voice("bm_nobody", 1.0)
    assert (home / "config.yaml").read_text() == before


def test_phonon_selection_replaces_voice_and_downloads_only_the_selected_model(home, monkeypatch):
    a = make(home, "tts: {voice: bm_george, speed: 0.9}\nbrain: {kind: local}\nmodels: {tier: t4}\n")
    a.save_voice("default", 1.0, engine="phonon")
    assert raw(home)["tts"] == {"engine": "phonon", "voice": "default", "speed": 1.0}
    assert a.current().tts_engine == "phonon" and a.voices(engine="phonon") == ["default"]
    assert a.missing_models() == [models.STT, models.PHONON, models.LLM["t4"]]
    pulled = []
    monkeypatch.setattr(models, "pull", lambda repos, report=None: pulled.append(repos))
    a.pull_voice_model(lambda done, total: None, engine="phonon")
    a.pull_models(lambda done, total: None)
    assert pulled == [[models.PHONON], [models.STT, models.PHONON, models.LLM["t4"]]]
    before = (home / "config.yaml").read_text()
    with pytest.raises(ActionError, match="speed"):
        a.save_voice("default", 0.9, engine="phonon")
    assert (home / "config.yaml").read_text() == before
    a.save_voice("bf_emma", 1.1)
    assert a.current().tts_engine == "kokoro" and a.current().voice == "bf_emma"


def test_phonon_preview_routes_to_its_player(home):
    a = make(home)
    phonon = FakePlayer()
    a.phonon_player = phonon
    a.play_sample("default", 1.0, "Hello.", engine="phonon")
    assert phonon.played == [("default", 1.0, "Hello.")]
    assert not a.player.played


def test_save_callers_merges_into_the_sections(home):
    a = make(home, "calls: {relay_filler_after_ms: 1000, allow_inbound: true, allow_outbound: true}\ninbound: {greeting: Old}\n")
    a.save_callers("Hello there.", "Take a message.", 7)
    assert raw(home)["calls"] == {"relay_filler_after_ms": 1000, "daily_limit": 7,
                                  "allow_inbound": True, "allow_outbound": True}
    assert raw(home)["inbound"] == {"greeting": "Hello there.", "instructions": "Take a message."}
    assert a.current().daily_limit == 7
    before = (home / "config.yaml").read_text()
    with pytest.raises(ActionError, match="greeting"):
        a.save_callers("x" * 501, "Take a message.", 7)
    assert (home / "config.yaml").read_text() == before


@pytest.mark.parametrize("inbound,outbound", [(False, True), (True, False)])
@pytest.mark.parametrize("explicit", [False, True])
def test_save_call_settings_preserves_inactive_values_and_absent_fields(home, inbound, outbound, explicit):
    a = make(home, "calls: {daily_limit: 0}\ninbound: {greeting: Old, instructions: Keep this.}\n" if explicit else None)
    a.save_call_directions(inbound, outbound)
    before = raw(home)
    # A missing daily_limit is displayed as 20 in setup but still defaults to 0 in the runtime.
    a.save_callers("New greeting.", "New instructions.", 20)
    after = raw(home)
    if inbound:
        assert after["calls"] == before["calls"]
        assert after["inbound"] == {"greeting": "New greeting.", "instructions": "New instructions."}
    else:
        assert after.get("inbound") == before.get("inbound")
        assert after["calls"]["daily_limit"] == 20


@pytest.mark.parametrize("inbound,outbound", [(True, True), (False, True), (True, False)])
def test_save_call_directions_preserves_other_settings(home, inbound, outbound):
    a = make(home, "calls: {daily_limit: 7, relay_filler_after_ms: 1000}\n"
                   "inbound: {greeting: Hello., instructions: Take a message.}\nmodels: {tier: qwen2b}\n")
    before = raw(home)
    a.save_call_directions(inbound, outbound)
    expected = {**before, "calls": {**before["calls"], "allow_inbound": inbound, "allow_outbound": outbound}}
    assert raw(home) == expected
    assert (a.current().allow_inbound, a.current().allow_outbound) == (inbound, outbound)


@pytest.mark.parametrize("existing", [False, True])
def test_save_call_directions_rolls_back_invalid_combination(home, existing):
    a = make(home, "calls: {daily_limit: 7}\n" if existing else None)
    before = raw(home) if existing else None
    with pytest.raises(ActionError, match="at least one"):
        a.save_call_directions(False, False)
    assert a.config_path.exists() == existing
    if existing:
        assert raw(home) == before


# ---- models ----------------------------------------------------------------------------------

def test_missing_models_download_size_and_pull(home, monkeypatch):
    a = make(home, "brain: {kind: local}\nmodels: {tier: t9}\n")
    missing = [models.STT, models.TTS, models.LLM["t9"]]
    assert a.missing_models() == missing
    sizes, pulls = [], []
    monkeypatch.setattr(models, "download_size", lambda repos: sizes.append(repos) or 5_000_000_000)
    monkeypatch.setattr(models, "pull", lambda repos, report=None: pulls.append((repos, report)))
    assert a.download_size() == 5_000_000_000 and sizes == [missing]
    report = lambda done, total: None  # noqa: E731
    a.pull_models(report)
    assert pulls == [(missing, report)]
    monkeypatch.setattr(models, "pull", lambda repos, report=None: (_ for _ in ()).throw(models.ModelsError("could not download x: OSError")))
    with pytest.raises(ActionError, match="^could not download x: OSError$"):
        a.pull_models(report)


# ---- checking the lines ----------------------------------------------------------------------

def status(lid, kind, state, error=None):
    return LineStatus(lid, kind, "", state, None, None, error, time.time())


def write_state(home, *lines):
    StateFile(home / "state.json").write({ln.id: ln for ln in lines}, None, {"kind": "local", "base_url": None})


def test_check_lines_reads_a_running_daemon(home, monkeypatch):
    monkeypatch.setattr(act, "probe", lambda path: True)
    monkeypatch.setattr(act.subprocess, "Popen", None)  # a running daemon is never started again
    write_state(home, status("main", "trunk", "registered"), status("office", "trunk", "degraded", "403 Forbidden"),
                status("new", "trunk", "registering"), status("agent", "sip", "unregistered"),
                status("spare", "trunk", "skipped", "AI_PHONE_LINE_SPARE_PASSWORD not set"),
                status("old", "trunk", "unregistered", "0 OK"))
    assert make(home).check_lines() == [
        ("main", "registered", ""), ("office", "degraded", "403 Forbidden"), ("new", "registering", ""),
        ("agent", "ready", "listening"), ("spare", "skipped", "AI_PHONE_LINE_SPARE_PASSWORD not set"),
        ("old", "unregistered", "0 OK")]
    (home / "state.json").unlink()
    assert make(home).check_lines() == [("lines", "failed", "ai-phone is running but has not written its status yet")]


def test_check_lines_when_the_control_socket_is_unsafe(home, monkeypatch):
    def unsafe():
        raise RuntimeError("/private/tmp/ai-phone-501 is a symlink; refusing to place the control socket there")

    monkeypatch.setattr(act.paths, "socket_path", unsafe)
    monkeypatch.setattr(act.subprocess, "Popen", None)
    assert make(home).check_lines() == [("lines", "failed", "/private/tmp/ai-phone-501 is a symlink; refusing to place "
                                                            "the control socket there")]


def test_check_lines_when_the_control_socket_cannot_be_made(home, monkeypatch):
    def blocked():
        raise FileExistsError(17, "File exists", "/private/tmp/ai-phone-501")

    monkeypatch.setattr(act.paths, "socket_path", blocked)
    monkeypatch.setattr(act.subprocess, "Popen", None)
    assert make(home).check_lines() == [("lines", "failed", "could not use the control socket: File exists")]


def test_check_lines_when_ai_phone_cannot_start(home, monkeypatch):
    def missing(*a, **k):
        raise FileNotFoundError(2, "No such file or directory")

    monkeypatch.setattr(act.subprocess, "Popen", missing)
    assert make(home).check_lines() == [("lines", "failed", "could not start ai-phone: No such file or directory")]


def test_check_lines_without_the_sip_library(home, monkeypatch):
    for so in act.VENDOR.iterdir():
        so.unlink()
    monkeypatch.setattr(act.subprocess, "Popen", None)
    assert make(home).check_lines() == [("lines", "skipped", "SIP library not built · run ./scripts/install.sh")]


class FakeDaemon:
    """Stands in for `ai-phone run --fake`: each poll writes the next state.json (None: nothing yet); SIGINT stops it."""

    def __init__(self, home, states=(), exit_with=None, log=b"", ignores_sigint=False):
        self.home, self.states, self.exit_with, self.log_text = home, list(states), exit_with, log
        self.ignores_sigint = ignores_sigint
        self.argv = self.signals = None
        self.returncode, self.killed, self.env = None, False, None

    def __call__(self, argv, stdout=None, stderr=None, **kw):
        self.argv, self.signals, self.kw = argv, [], kw
        stdout.write(self.log_text)
        stdout.flush()
        return self

    def poll(self):
        if self.exit_with is not None:
            self.returncode = self.exit_with
        elif self.states:
            lines = self.states.pop(0)
            if lines is not None:
                write_state(self.home, *lines)
        return self.returncode

    def send_signal(self, sig):
        self.signals.append(sig)
        if not self.ignores_sigint:
            self.returncode = -sig

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired(self.argv, timeout)
        return self.returncode

    def kill(self):
        self.killed, self.returncode = True, -9


@pytest.fixture
def quick(monkeypatch):
    monkeypatch.setattr(act, "POLL_SEC", 0.01)
    monkeypatch.setattr(act, "CHECK_SEC", 0.5)
    monkeypatch.setattr(act, "STOP_SEC", 0.05)
    monkeypatch.setattr(act, "EXIT_STOP_SEC", 0.05)


class FakeAtexit:
    def __init__(self):
        self.hooks = []

    def register(self, fn):
        self.hooks.append(fn)
        return fn

    def unregister(self, fn):
        self.hooks = [h for h in self.hooks if h != fn]


def test_check_lines_starts_ai_phone_briefly(home, monkeypatch, quick):
    daemon = FakeDaemon(home, states=[None, [status("main", "trunk", "registering"), status("agent", "sip", "unregistered")],
                                      [status("main", "trunk", "registered"), status("agent", "sip", "unregistered")]])
    monkeypatch.setattr(act.subprocess, "Popen", daemon)
    StateFile(home / "state.json").write({"main": status("main", "trunk", "degraded", "stale")}, None, {})
    doc = json.loads((home / "state.json").read_text())
    doc["written_at"] -= 3600  # a state.json from an earlier run is never read
    (home / "state.json").write_text(json.dumps(doc))
    assert make(home).check_lines() == [("main", "registered", ""), ("agent", "ready", "listening")]
    assert daemon.argv == [sys.executable, "-c", "import sys; from ai_phone.cli import main; sys.exit(main(sys.argv[1:]))",
                           "run", "--fake"]
    assert daemon.signals == [signal.SIGINT] and not daemon.killed


@pytest.mark.parametrize("stubborn", [False, True])
def test_the_trial_daemon_is_stopped_if_setup_exits_while_it_runs(home, monkeypatch, quick, stubborn):
    """A forced quit skips the flow thread's cleanup (it is a daemon thread), so an exit hook stops the trial start:
    SIGINT, a short wait, then kill. Stopping twice is harmless."""
    exits = FakeAtexit()
    monkeypatch.setattr(act, "atexit", exits)
    registered = []

    class Quitting(FakeDaemon):
        def poll(self):
            registered.append(len(exits.hooks))
            if len(registered) == 2:
                exits.hooks[0]()  # the interpreter exits now
            return super().poll()

    daemon = Quitting(home, states=[[status("main", "trunk", "registering")]] * 10, ignores_sigint=stubborn)
    monkeypatch.setattr(act.subprocess, "Popen", daemon)
    assert make(home).check_lines() == [("lines", "failed", "ai-phone stopped before it reported its lines")]
    assert registered[:2] == [1, 1] and exits.hooks == []  # registered while it ran, removed once it was stopped
    assert daemon.signals == [signal.SIGINT] and daemon.killed is stubborn


def test_check_lines_reports_a_daemon_that_exits(home, monkeypatch, quick):
    log = b"2026-09-17 INFO starting\n" + b"2026-09-17 ERROR could not bind 0.0.0.0:5060: " + b"x" * 300 + b"\n\n"
    daemon = FakeDaemon(home, exit_with=1, log=log)
    monkeypatch.setattr(act.subprocess, "Popen", daemon)
    rows = make(home).check_lines()
    assert rows == [("lines", "failed", ("2026-09-17 ERROR could not bind 0.0.0.0:5060: " + "x" * 300)[:160])]
    assert not daemon.signals  # it already stopped


def test_check_lines_gives_up_after_the_wait_and_stops_it_hard(home, monkeypatch, quick):
    daemon = FakeDaemon(home, states=[[status("main", "trunk", "registering")]] * 1000, ignores_sigint=True)
    monkeypatch.setattr(act.subprocess, "Popen", daemon)
    assert make(home).check_lines() == [("main", "registering", "")]
    assert daemon.signals == [signal.SIGINT] and daemon.killed


def test_check_lines_when_ai_phone_never_reports(home, monkeypatch, quick):
    daemon = FakeDaemon(home)
    monkeypatch.setattr(act.subprocess, "Popen", daemon)
    assert make(home).check_lines() == [("lines", "registering", "ai-phone did not report its lines within 0.5 seconds")]


# ---- the player ------------------------------------------------------------------------------

class Result:
    def __init__(self, audio):
        self.audio = audio


class FakeKokoro:
    """Talks like mlx-audio: a line on stdout, one on stderr and a warning while it generates."""

    def __init__(self):
        self.calls = []

    def generate(self, **kw):
        self.calls.append(kw)
        print("Creating new KokoroPipeline for language: b")
        print("some warning", file=sys.stderr)
        warnings.warn("words count mismatch", UserWarning)
        yield Result(np.full(2400, 0.5, dtype=np.float32))
        yield Result(np.full(1200, -2.0, dtype=np.float32))


def test_kokoro_player_writes_a_wav_and_plays_it(tmp_path, monkeypatch, capsys, recwarn):
    snapshot = tmp_path / "kokoro"
    (snapshot / "voices").mkdir(parents=True)
    (snapshot / "voices" / "bf_emma.safetensors").write_bytes(b"")
    monkeypatch.setattr(models, "local_path", lambda repo: snapshot if repo == models.TTS else None)
    monkeypatch.setattr(voices, "voices_dir", lambda: snapshot / "voices")
    kokoro, loads, played = FakeKokoro(), [], []

    def run(argv, timeout=None, check=False, capture_output=False):
        path = argv[1]
        with wave.open(path, "rb") as w:
            played.append((argv[0], w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes(), timeout, check))
            frames = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        assert frames[0] == 16383 and frames[-1] == -32767  # scaled, and clipped
        played.append(path)

    def load(path, **kw):
        loads.append((path, kw))
        print("fetching 1 file")
        return kokoro

    player = KokoroPlayer(load=load, run=run)
    player.play("bf_emma", 1.1, "Hello.")
    player.play("bf_emma", 1.0, "Again.")
    assert loads == [(str(snapshot), {"model_type": "kokoro"})]  # loaded once; the snapshot folder name is a hash
    assert capsys.readouterr() == ("", "") and len(recwarn) == 0  # plain mode stays clean
    assert played[0] == ("afplay", 1, 2, 24000, 3600, 30, True)
    assert kokoro.calls[0] == {"text": "Hello.", "voice": str(snapshot / "voices" / "bf_emma.safetensors"), "speed": 1.1,
                               "lang_code": "b"}
    assert len(played) == 4 and not any(Path(p).exists() for p in played[1::2])  # the temporary files are gone


def test_quiet_leaves_the_streams_alone_off_the_main_thread(capsys):
    """Under the full-screen app the flow runs in its own thread, and Textual owns the process-wide streams."""
    import threading
    seen = {}

    def worker():
        before = sys.stdout
        with act._quiet():
            seen["same"] = sys.stdout is before
            print("from the flow thread")

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert seen["same"] and "from the flow thread" in capsys.readouterr().out
    with act._quiet():
        print("from the main thread")
    assert capsys.readouterr().out == ""


def test_kokoro_player_needs_the_model_and_the_voice(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "local_path", lambda repo: None)
    player = KokoroPlayer(load=lambda path, **kw: pytest.fail("nothing to load"), run=lambda *a, **k: pytest.fail("nothing to play"))
    with pytest.raises(ActionError, match="^could not play the sample: the speech model is not downloaded$"):
        player.play("af_heart", 1.0, "Hi.")
    monkeypatch.setattr(models, "local_path", lambda repo: tmp_path)
    monkeypatch.setattr(voices, "voices_dir", lambda: None)
    with pytest.raises(ActionError, match="^could not play the sample: the voice bf_emma is not installed$"):
        player.play("bf_emma", 1.0, "Hi.")


def test_phonon_player_previews_the_daemons_frames_and_reuses_the_backend(tmp_path, monkeypatch):
    (tmp_path / "default-voice.safetensors").write_bytes(b"voice")
    monkeypatch.setattr(models, "local_path", lambda repo: tmp_path if repo == models.PHONON else None)
    calls, played = [], []
    frame = np.full(320, 1234, dtype="<i2").tobytes()

    class Backend:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def warmup(self):
            calls.append(("warmup",))

        def stream(self, gen, seq, text):
            calls.append(("stream", gen, seq, text))
            yield frame
            yield b""

    def play(argv, **kwargs):
        with wave.open(argv[1], "rb") as wav:
            assert (wav.getframerate(), wav.getsampwidth(), wav.getnchannels()) == (16000, 2, 1)
            assert wav.readframes(wav.getnframes()) == frame
        played.append(Path(argv[1]))

    player = PhononPlayer(factory=Backend, run=play)
    player.play("default", 1.0, "Hello.")
    player.play("default", 1.0, "Again.")
    assert calls == [("init", {"voice": "default", "speed": 1.0}), ("warmup",),
                     ("stream", 1, 0, "Hello."), ("stream", 2, 0, "Again.")]
    assert len(played) == 2 and not any(path.exists() for path in played)


def test_phonon_player_refuses_missing_assets_before_loading(tmp_path, monkeypatch):
    player = PhononPlayer(factory=lambda **kw: pytest.fail("must not load"))
    monkeypatch.setattr(models, "local_path", lambda repo: None)
    with pytest.raises(ActionError, match="speech model is not downloaded"):
        player.play("default", 1.0, "Hello.")
    monkeypatch.setattr(models, "local_path", lambda repo: tmp_path)
    with pytest.raises(ActionError, match="voice default is not installed"):
        player.play("default", 1.0, "Hello.")


# ---- a Claude or ChatGPT plan as the brain -------------------------------------------------------

@pytest.fixture
def fake_cli(tmp_path, monkeypatch):
    """The fake CLI the backend tests use, with a script of its own: setup never runs the real claude or codex.

    `as_name` also installs it under the name a backend looks for (`claude`, `codex`) on a PATH with nothing else on
    it, for the paths where the config names no CLI of its own and the lookup is what has to be covered.
    """
    from tests.unit.test_cli_brain import FAKE_CLI, fake_codex_home

    fake_codex_home(tmp_path, monkeypatch)
    exe = tmp_path / "fake-cli"
    exe.write_text(FAKE_CLI)
    exe.chmod(0o755)

    def script(protocol, turns=None, as_name=None, **extra):
        from ai_phone.services.backends.cli_brain import CODEX_TOOLS_OFF

        spec = tmp_path / "script.json"
        extra.setdefault("features", [f"{f} stable true" for f in CODEX_TOOLS_OFF])
        spec.write_text(json.dumps({"protocol": protocol, "log": str(tmp_path / "cli.jsonl"),
                                    "turns": turns or [{}], **extra}))
        monkeypatch.setenv("AI_PHONE_FAKE_CLI", str(spec))
        if as_name:
            bin_dir = tmp_path / "bin"
            bin_dir.mkdir(exist_ok=True)
            (bin_dir / as_name).write_text(FAKE_CLI)
            (bin_dir / as_name).chmod(0o755)
            # the fake and the interpreter its shebang needs, and nothing else: the real CLI is not on this PATH
            monkeypatch.setenv("PATH", os.pathsep.join([str(bin_dir), str(Path(sys.executable).parent)]))
        return str(exe)

    script.log = lambda: ([json.loads(l) for l in (tmp_path / "cli.jsonl").read_text().splitlines()]
                          if (tmp_path / "cli.jsonl").exists() else [])
    return script


def test_a_logged_in_cli_is_proved_with_one_turn_and_remembered(home, fake_cli):
    exe = fake_cli("claude", [{"deltas": ["OK"]}], version="2.1.286 (Claude Code)")
    a = make(home, f"brain: {{kind: claude, model: sonnet, cli: {exe}}}\n")
    assert a.cli_installed("claude") is True
    probe = a.test_subscription("claude")
    assert probe == {"model": "sonnet", "cli_version": "2.1.286 (Claude Code)", "warm_ms": probe["warm_ms"]}
    assert probe["warm_ms"] > 0
    assert a.test_subscription("claude") == probe              # the Check page asks again
    assert [r["event"] for r in fake_cli.log()].count("turn") == 1  # one turn of the plan for the whole run
    assert [r["text"] for r in fake_cli.log() if r["event"] == "turn"] == ["Say OK."]


def test_the_probe_uses_the_plans_own_model_when_the_config_names_another_brain(home, fake_cli):
    """The config is on the local brain, so its `cli` is not Claude Code's path and is rightly ignored: the backend
    looks the command up on PATH instead. The fake is installed there under that name, because the point of this test
    is the lookup, and a unit test that reaches the real claude spends a turn of the owner's plan on every suite run.
    """
    exe = fake_cli("claude", [{"deltas": ["OK"]}], as_name="claude")
    a = make(home, f"brain: {{kind: local, cli: {exe}}}\n")
    assert a.test_subscription("claude")["model"] == "haiku"   # the fastest, for a probe and for a call
    assert [r["event"] for r in fake_cli.log()].count("turn") == 1, "the fake handled the probe, not the real claude"


def test_the_login_is_established_without_spending_a_turn(home, fake_cli):
    """What the wizard asks before it offers a model: whether there is a login, and whether it is a plan."""
    exe = fake_cli("claude", [{"deltas": ["OK"]}], login={"loggedIn": True, "authMethod": "claude.ai",
                                                          "subscriptionType": "max"})
    a = make(home, f"brain: {{kind: claude, cli: {exe}}}\n")
    assert a.login_status("claude") == {"plan": "max"}
    assert [r["event"] for r in fake_cli.log()] == ["login"], "no session, no turn"
    assert a.login_status("claude") == {"plan": "max"}  # remembered for the run
    assert [r["event"] for r in fake_cli.log()] == ["login"]


def test_a_login_that_is_an_api_key_says_so_before_any_turn(home, fake_cli):
    """ANTHROPIC_API_KEY, or a key saved in the CLI's own settings, is billed per request and beats the plan login. A
    brain sold as "no API key" must not quietly become one."""
    exe = fake_cli("claude", [{"deltas": ["OK"]}],
                   login={"loggedIn": True, "authMethod": "api_key", "apiKeySource": "ANTHROPIC_API_KEY"})
    a = make(home, f"brain: {{kind: claude, cli: {exe}}}\n")
    with pytest.raises(ActionError) as e:
        a.login_status("claude")
    # the page names the method, because a refusal nobody can read is a refusal nobody can fix
    assert str(e.value) == ("claude is logged in as 'api_key', which is billed per request. ai-phone runs your "
                           "subscription instead: remove the key, then log in with: claude, then type /login")
    assert [r["event"] for r in fake_cli.log()] == ["login"]


def test_a_failed_login_check_is_never_remembered(home, fake_cli):
    exe = fake_cli("claude", login={"loggedIn": False})
    a = make(home, f"brain: {{kind: claude, cli: {exe}}}\n")
    with pytest.raises(ActionError, match="not logged in"):
        a.login_status("claude")
    fake_cli("claude", login={"loggedIn": True, "authMethod": "claude.ai", "subscriptionType": "max"})
    assert a.login_status("claude") == {"plan": "max"}


def test_the_probe_is_against_the_model_that_was_picked_and_is_remembered_as_such(home, fake_cli):
    """The wizard probes after the menu, with the model the person chose; the Check page then asks for the model the
    config names, which is that one, and finds it proved. One request of the plan for the whole run."""
    exe = fake_cli("claude", [{"deltas": ["OK"]}, {"deltas": ["OK"]}])
    a = make(home, f"brain: {{kind: claude, model: haiku, cli: {exe}}}\n")
    assert a.test_subscription("claude", "opus")["model"] == "opus"
    a.save_brain_subscription("claude", "opus")
    assert a.test_subscription("claude")["model"] == "opus"          # the Check page, with no model of its own
    assert [r["event"] for r in fake_cli.log()].count("turn") == 1
    assert [r["model"] for r in fake_cli.log() if r["event"] == "turn"] == ["opus"]


def test_a_cli_that_is_not_installed_says_how_to_install_it(home, fake_cli):
    fake_cli("claude")
    a = make(home, f"brain: {{kind: chatgpt, cli: {home / 'nowhere' / 'codex'}}}\n")
    assert a.cli_installed("chatgpt") is False
    with pytest.raises(ActionError, match="codex is not installed. Install it with: npm install -g @openai/codex"):
        a.test_subscription("chatgpt")


def test_a_cli_that_is_not_logged_in_says_how_to_log_in(home, fake_cli):
    exe = fake_cli("claude", [{"deltas": [], "fail": "Not logged in"}])
    a = make(home, f"brain: {{kind: claude, cli: {exe}}}\n")
    with pytest.raises(ActionError, match=r"claude is installed but not logged in. Log in with: claude, then type /login"):
        a.test_subscription("claude")
    exe = fake_cli("codex", [{"deltas": [], "fail": "401 Unauthorized"}])
    a = make(home, f"brain: {{kind: chatgpt, cli: {exe}}}\n")
    with pytest.raises(ActionError, match="codex is installed but not logged in. Log in with: codex login"):
        a.test_subscription("chatgpt")


def test_a_failed_login_is_never_remembered(home, fake_cli):
    """"Check again" has to mean it: a plan that has just been logged into must be allowed to answer."""
    exe = fake_cli("claude", [{"deltas": [], "fail": "Not logged in"}, {"deltas": ["OK"]}])
    a = make(home, f"brain: {{kind: claude, cli: {exe}}}\n")
    with pytest.raises(ActionError):
        a.test_subscription("claude")
    assert a.test_subscription("claude")["model"] == "haiku"


def test_claude_offers_its_three_models_without_asking_anything(home, fake_cli):
    exe = fake_cli("claude")
    a = make(home, f"brain: {{kind: claude, cli: {exe}}}\n")
    found = a.brain_models("claude")
    assert [m.id for m in found] == ["haiku", "sonnet", "opus"]
    assert [m.name for m in found] == ["Haiku", "Sonnet", "Opus"] and all(m.note for m in found)
    assert not (home / "cli.jsonl").exists(), "the CLI was started to list three names it always takes"


def test_chatgpt_models_are_the_plans_own_with_the_fastest_first(home, fake_cli):
    models_ = [{"id": "gpt-6-astra", "displayName": "GPT-6-Astra", "description": "the default", "isDefault": True},
               {"id": "gpt-5.6-luna", "displayName": "GPT-5.6-Luna", "description": "fast"},
               {"id": "secret", "displayName": "Hidden", "description": "", "hidden": True}]
    exe = fake_cli("codex", models=models_)
    a = make(home, f"brain: {{kind: chatgpt, cli: {exe}}}\n")
    found = a.brain_models("chatgpt")
    assert [m.id for m in found] == ["gpt-5.6-luna", "gpt-6-astra"]   # the measured default first; the hidden one is out
    assert found[0].name == "GPT-5.6-Luna" and found[1].note == "the default"
    assert [r["event"] for r in fake_cli.log()].count("turn") == 0    # asking costs no turn of the plan
    assert (home / "brain" / "codex" / "auth.json").readlink() == Path(os.environ["CODEX_HOME"]) / "auth.json"


def test_a_plan_that_lists_no_models_is_said_so(home, fake_cli):
    exe = fake_cli("codex", models=[])
    a = make(home, f"brain: {{kind: chatgpt, cli: {exe}}}\n")
    with pytest.raises(ActionError, match="Codex listed no models for this account"):
        a.brain_models("chatgpt")


def test_save_brain_subscription_writes_the_kind_and_the_model(home):
    a = make(home, "brain: {kind: openai_compatible, base_url: 'https://api.example.com/v1', model: qwen, "
                   "api_key_env: BRAIN_TOKEN_KEY, cli: /opt/homebrew/bin/claude, first_token_timeout_ms: 20000}\n")
    a.save_brain_subscription("claude", "sonnet")
    # the endpoint's url and key variable go with it, and so does a cli path set while it was the brain (it belongs to
    # the kind it was set for); a deadline the person chose is theirs whichever brain they pick
    assert raw(home)["brain"] == {"kind": "claude", "model": "sonnet", "first_token_timeout_ms": 20000}
    assert load_config(home / "config.yaml").brain.kind == "claude"
    cur = a.current()
    assert cur.brain_kind == "claude" and cur.model == "sonnet" and cur.brain_cli is None
    a.save_brain_subscription("chatgpt", "gpt-5.6-luna")
    assert raw(home)["brain"]["kind"] == "chatgpt" and raw(home)["brain"]["model"] == "gpt-5.6-luna"


def test_a_subscription_brain_needs_no_model_of_its_own(home):
    a = make(home, "brain: {kind: chatgpt, model: gpt-5.6-luna}\n")
    assert a.machine().models_missing == [models.STT, models.TTS]  # speech and voice only: the brain is not here


def test_a_failure_nobody_has_guidance_for_keeps_the_cli_s_own_words_off_the_page(home, fake_cli, caplog):
    """Whatever a provider says — a rate limit, an org policy, an incident — is unbounded third-party prose; the
    person gets a sentence and the kind, and the detail goes to the log."""
    said = "Overloaded: policy 1234 says this ORGANISATION may not do that"
    exe = fake_cli("claude", [{"deltas": [], "fail": said}])
    a = make(home, f"brain: {{kind: claude, cli: {exe}}}\n")
    # both loggers: the backend puts the CLI's own words in the debug log of "ai-phone" as it classifies them, and
    # the wizard adds its own line there too. Neither is above debug.
    with caplog.at_level(logging.DEBUG, logger="ai-phone"):
        with pytest.raises(ActionError) as e:
            a.test_subscription("claude")
    assert str(e.value) == "claude could not answer (failed). Try again, or pick another brain."
    assert said not in str(e.value) and "policy 1234" in caplog.text  # the detail is kept, out of sight
    above = [r for r in caplog.records if r.levelno > logging.DEBUG and said in r.getMessage()]
    assert above == [], "the provider's own words belong at debug and nowhere else"


def test_a_path_set_for_one_plans_cli_is_never_used_for_the_other(home, fake_cli):
    """`brain.cli` belongs to the kind it was set for: the Claude Code binary driven as `codex app-server` would fail
    with nothing the person could act on."""
    exe = fake_cli("claude")
    a = make(home, f"brain: {{kind: claude, cli: {exe}}}\n")
    assert a._cli_path("claude") == exe and a.cli_installed("claude") is True
    assert a._cli_path("chatgpt") is None, "the Claude Code path would have been run as Codex"


def test_choosing_the_other_plan_drops_the_path_to_the_first_ones_cli(home):
    a = make(home, "brain: {kind: claude, cli: /opt/homebrew/bin/claude, first_token_timeout_ms: 20000}\n")
    a.save_brain_subscription("claude", "sonnet")
    assert raw(home)["brain"]["cli"] == "/opt/homebrew/bin/claude"  # same plan: the path is still theirs
    a.save_brain_subscription("chatgpt", "gpt-5.6-luna")
    assert "cli" not in raw(home)["brain"], "a path to Claude Code was left behind to run Codex"
    assert raw(home)["brain"] == {"kind": "chatgpt", "model": "gpt-5.6-luna", "first_token_timeout_ms": 20000}
    assert a._cli_path("chatgpt") is None


SIGNALWIRE_CREDS = {"SIGNALWIRE_PROJECT_ID": "project-test", "SIGNALWIRE_API_TOKEN": "token-SECRET",
                    "SIGNALWIRE_SPACE": "example.signalwire.com"}


class ManagedCallerIdProvider:
    """Only the CLI/action boundary: real API request shapes are covered by the provider suite."""
    name = "signalwire"

    def __init__(self):
        self.events = []
        self.refuse = False
        self.wrong_context = False

    def validate_context(self, journal):
        self.events.append("context")
        if self.wrong_context:
            raise ProviderError("This journal belongs to another SignalWire Space or Project.")
        assert journal.doc["signalwire"] == {"space": "example.signalwire.com", "project_id": "project-test"}

    def pending_caller_id(self, journal):
        change = journal.doc.get("caller_id_change", {})
        return change.get("desired") if not change.get("local_committed") else None

    def apply_caller_id(self, journal, number):
        self.events.append(("apply", number))
        if self.refuse:
            raise ProviderError("Caller ID rejected: token-SECRET")
        journal.doc["caller_id_change"] = {"desired": number, "previous": BOUGHT,
                                           "remote_applied": True, "local_committed": False}
        journal._save()
        return number

    def finish_caller_id(self, journal):
        self.events.append("finish")
        journal.doc["caller_id_change"]["local_committed"] = True
        journal._save()

    def start_caller_id_verification(self, number):
        self.events.append(("verify", number))
        return VerificationStep("enter_code_in_cli", "Enter code", {})


def managed_caller_id_actions(home, provider=None, *, verification=True):
    provider = provider or ManagedCallerIdProvider()
    resources = {"sip_resource_id": "resource-1", "sip_endpoint_id": "endpoint-1", "brought_number_id": "number-1"}
    line = {"kind": "trunk", "number": BOUGHT, "caller_id": BOUGHT, "user": "account1", "from_user": "account",
            "registrar": "sip:example.sip.signalwire.com:5061", "dial_domain": "example.sip.signalwire.com:5061",
            "transport": "tls", "srtp": "required",
            "provider": {"name": "signalwire", "state": "ready", "resources": resources}}
    if verification:
        line["verified_caller_ids"] = [{"number": "+15555550100", "provider": "signalwire",
                                        "verified_at": "2026-10-01T12:00:00Z"}]
    journal = Journal.open(home / "lines", "main", "signalwire")
    journal.start("endpoint")
    journal.finish("endpoint", **resources)
    journal.doc["signalwire"] = {"space": "example.signalwire.com", "project_id": "project-test"}
    journal.set_state("ready")
    a = make(home, yaml.safe_dump({"inbound": {"greeting": "Hi"}, "lines": {"main": line}}),
             provider_loader=lambda name, **kwargs: provider)
    return a, provider


def test_signalwire_credentials_map_space_project_and_token_explicitly(home):
    seen = []

    def loader(name, **kwargs):
        seen.append((name, kwargs))
        return FakeProvider()

    a = make(home, provider_loader=loader)
    a.owned_numbers("signalwire", SIGNALWIRE_CREDS)
    assert seen == [("signalwire", {"key": "project-test", "secret": "token-SECRET", "space": "example.signalwire.com"})]
    with pytest.raises(ActionError, match="all SignalWire credential fields"):
        a.owned_numbers("signalwire", {"SIGNALWIRE_PROJECT_ID": "project-test", "SIGNALWIRE_API_TOKEN": "token-SECRET"})
    assert len(seen) == 1


def test_signalwire_purchase_is_refused_before_constructing_provider_or_saving_keys(home):
    a = make(home, provider_loader=lambda *args, **kw: pytest.fail("purchase must not reach provider"))
    with pytest.raises(ActionError, match="existing number"):
        a.buy_number("signalwire", SIGNALWIRE_CREDS, Number(BOUGHT, None, None, False, "number-1"))
    assert not (home / "secrets.env").exists() and not (home / "config.yaml").exists()


def test_signalwire_manual_preset_keeps_credentials_separate_and_number_optional(home):
    a = make(home)
    line_id = a.add_trunk_line("sip:example.sip.signalwire.com:5061", "account1", "sip-SECRET", "", "tls",
                               preset="signalwire", srtp="required")
    line = load_config(home / "config.yaml").lines[line_id]
    assert line_id == "signalwire" and line.from_user == "account" and line.number is None
    assert line.tls_hostname_policy == "verified_wildcard"
    assert line.registrar.endswith(";transport=tls")
    assert line.transport == "tls" and line.srtp == "required" and line.outbound_proxy == line.registrar
    assert line.provider is None and line.accept_invites_from == "registered_trunk"
    assert "sip-SECRET" not in (home / "config.yaml").read_text()
    assert load_secrets() == {"AI_PHONE_LINE_SIGNALWIRE_PASSWORD": "sip-SECRET"}


def test_signalwire_caller_id_requires_credentials_and_matching_journal_before_remote_change(home):
    a, provider = managed_caller_id_actions(home)
    before = (home / "config.yaml").read_bytes()
    with pytest.raises(ActionError, match="credentials"):
        a.use_caller_id("main", "+15555550100")
    assert provider.events == []
    (home / "lines" / "main.journal.json").unlink()
    with pytest.raises(ActionError, match="journal is missing"):
        a.use_caller_id("main", "+15555550100", SIGNALWIRE_CREDS)
    assert provider.events == [] and (home / "config.yaml").read_bytes() == before


def test_signalwire_context_mismatch_stops_verification_and_caller_id_change(home):
    a, provider = managed_caller_id_actions(home)
    provider.wrong_context = True
    before = (home / "config.yaml").read_bytes()
    with pytest.raises(ActionError, match="another SignalWire Space"):
        a.start_caller_id("signalwire", SIGNALWIRE_CREDS, "+15555550100", line_id="main")
    with pytest.raises(ActionError, match="another SignalWire Space"):
        a.use_caller_id("main", "+15555550100", SIGNALWIRE_CREDS)
    assert provider.events == ["context", "context"]
    assert (home / "config.yaml").read_bytes() == before


def test_signalwire_failed_remote_caller_id_change_preserves_local_config_and_redacts_token(home):
    a, provider = managed_caller_id_actions(home)
    provider.refuse = True
    before = (home / "config.yaml").read_bytes()
    with pytest.raises(ActionError) as exc:
        a.use_caller_id("main", "+15555550100", SIGNALWIRE_CREDS)
    assert "token-SECRET" not in str(exc.value) and "Caller ID rejected" in str(exc.value)
    assert (home / "config.yaml").read_bytes() == before and "finish" not in provider.events


def test_signalwire_own_number_change_does_not_invent_verification_history(home):
    a, provider = managed_caller_id_actions(home, verification=False)
    a.use_caller_id("main", BOUGHT, SIGNALWIRE_CREDS)
    assert raw(home)["lines"]["main"]["caller_id"] == BOUGHT
    assert "verified_caller_ids" not in raw(home)["lines"]["main"]
    assert provider.events == ["context", ("apply", BOUGHT), "finish"]


def test_signalwire_verified_caller_id_is_applied_remotely_before_local_commit(home):
    a, provider = managed_caller_id_actions(home, verification=False)
    a.record_caller_id("main", "+15555550102", "signalwire", SIGNALWIRE_CREDS)
    line = raw(home)["lines"]["main"]
    assert line["caller_id"] == "+15555550102"
    assert [(v["number"], v["provider"]) for v in line["verified_caller_ids"]] == [("+15555550102", "signalwire")]
    assert provider.events == ["context", ("apply", "+15555550102"), "finish"]
    assert Journal.open(home / "lines", "main").doc["caller_id_change"]["local_committed"]


def test_signalwire_local_write_failure_leaves_remote_change_resumable(home, monkeypatch):
    from ai_phone import cli_lines
    from ai_phone.cli import main
    from ai_phone.lines import caller_id
    a, provider = managed_caller_id_actions(home, verification=False)
    before = (home / "config.yaml").read_bytes()
    original = caller_id.update_line

    def disk_failure(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(caller_id, "update_line", disk_failure)
    with pytest.raises(ActionError, match="lines resume main"):
        a.record_caller_id("main", "+15555550102", "signalwire", SIGNALWIRE_CREDS)
    assert (home / "config.yaml").read_bytes() == before
    assert provider.pending_caller_id(Journal.open(home / "lines", "main")) == "+15555550102"
    monkeypatch.setattr(caller_id, "update_line", original)
    monkeypatch.setattr(cli_lines, "_provider", lambda args: provider)
    assert main(["lines", "resume", "main", "--key-env", "PROJECT_ID", "--space", "example.signalwire.com"]) == 0
    assert raw(home)["lines"]["main"]["caller_id"] == "+15555550102"
    assert raw(home)["lines"]["main"]["verified_caller_ids"][0]["number"] == "+15555550102"
    assert provider.pending_caller_id(Journal.open(home / "lines", "main")) is None


def pending_signalwire(home, *, provider="signalwire", brought=True, state="pending", target=True, line_id="retry"):
    journal = Journal.open(home / "lines", line_id, provider)
    journal.start("target" if target else "claim", phone_number=OWNED, brought=brought, source_id="owned-1")
    journal.finish("target" if target else "claim")
    journal.start("credentials", user="originaluser", password_env="AI_PHONE_LINE_RETRY_PASSWORD")
    journal.finish("credentials", user="originaluser", password_env="AI_PHONE_LINE_RETRY_PASSWORD")
    journal.doc["signalwire"] = {"space": "example.signalwire.com", "project_id": "11111111-1111-4111-8111-111111111111"}
    journal.set_state(state)
    write_secret("AI_PHONE_LINE_RETRY_PASSWORD", "original-sip-SECRET")
    return journal


def test_signalwire_resumable_numbers_reads_pending_attachment_without_writes(home):
    journal = pending_signalwire(home)
    a = make(home)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (home / "lines").glob("*.json")}
    assert a.resumable_numbers("signalwire") == {OWNED: "retry"}
    assert a.resumable_numbers("twilio") == {}
    assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before} == before
    assert journal.doc["provider"] == "signalwire"


@pytest.mark.parametrize("settings", [
    {"provider": "twilio"}, {"brought": False}, {"state": "rollback_failed"}, {"state": "ready"},
    {"state": "failed"}, {"target": False},
])
def test_signalwire_resume_picker_excludes_ineligible_journals(home, settings):
    pending_signalwire(home, **settings)
    assert make(home).resumable_numbers("signalwire") == {}


@pytest.mark.parametrize("line_id", ["retry", "another-line"])
def test_signalwire_resume_picker_excludes_configured_lines_and_numbers(home, line_id):
    pending_signalwire(home)
    a = make(home, yaml.safe_dump({"lines": {line_id: {"number": OWNED}}}))
    assert a.resumable_numbers("signalwire") == {}


def test_signalwire_resume_picker_excludes_ambiguous_number_claims(home):
    pending_signalwire(home)
    pending_signalwire(home, provider="twilio", line_id="other")
    assert make(home).resumable_numbers("signalwire") == {}


def test_signalwire_wizard_retry_reuses_line_and_sip_credentials(home):
    pending_signalwire(home)
    received = []

    class RetryProvider(FakeProvider):
        name = "signalwire"

        def attach(self, journal, number, line_id, user, password):
            received.append((line_id, user, password, number.source_id))
            return super().attach(journal, number, line_id, user, password)

    provider = RetryProvider()
    a = make(home, provider_loader=lambda name, **kwargs: provider)
    number = Number(OWNED, None, None, False, "owned-1")
    assert a.resumable_numbers("signalwire") == {OWNED: "retry"}
    result = a.attach_number("signalwire", SIGNALWIRE_CREDS, number)
    assert (result.line_id, result.code) == ("retry", 0)
    assert received == [("retry", "originaluser", "original-sip-SECRET", "owned-1")]
    assert load_secrets()["AI_PHONE_LINE_RETRY_PASSWORD"] == "original-sip-SECRET"
    assert set(load_config(home / "config.yaml").lines) == {"retry"}
    assert sorted(p.name for p in (home / "lines").glob("*.journal.json")) == ["retry.journal.json"]
    assert provider.created == {"profile": 1, "connection": 1}
    assert a.resumable_numbers("signalwire") == {}


@pytest.mark.parametrize("changed", ["space", "project"])
def test_signalwire_wizard_retry_rejects_wrong_context_before_http(home, changed):
    from ai_phone.providers.signalwire import SignalWireProvider
    pending_signalwire(home)
    creds = {"SIGNALWIRE_PROJECT_ID": "11111111-1111-4111-8111-111111111111", "SIGNALWIRE_API_TOKEN": "token-SECRET",
             "SIGNALWIRE_SPACE": "example.signalwire.com"}
    creds["SIGNALWIRE_SPACE" if changed == "space" else "SIGNALWIRE_PROJECT_ID"] = (
        "another.signalwire.com" if changed == "space" else "22222222-2222-4222-8222-222222222222")
    http = httpx.Client(transport=httpx.MockTransport(lambda req: pytest.fail("context mismatch must precede every request")))
    a = make(home, provider_loader=lambda name, **kwargs: SignalWireProvider(**kwargs, http=http))
    result = a.attach_number("signalwire", creds, Number(OWNED, None, None, False, "owned-1"))
    assert result.code == 3 and result.line_id == "retry"
    assert "different Space or Project" in result.messages[0]
    assert not (home / "config.yaml").exists()
    journal = Journal.open(home / "lines", "retry")
    assert journal.steps_done() == ["target", "credentials"]
    assert load_secrets()["AI_PHONE_LINE_RETRY_PASSWORD"] == "original-sip-SECRET"

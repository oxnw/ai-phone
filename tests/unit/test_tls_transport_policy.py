"""Per-line TLS factory selection and failure cleanup, without native sockets."""
import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("ai_phone.vendor.pjsua2")
from ai_phone.config import Config, LineConfig
from ai_phone.telephony import endpoint as module
from ai_phone.telephony.endpoint import LineSpec, Telephony


def spec(line_id="guarded", policy="verified_wildcard", **changes):
    values = dict(kind="trunk", user="synthetic", registrar="sip:tenant.carrier.test:5061;transport=tls",
                  outbound_proxy="sip:tenant.carrier.test:5061;transport=tls", transport="tls",
                  from_user="account", tls_hostname_policy=policy)
    values.update(changes)
    return LineSpec.from_config(line_id, LineConfig(**values), "synthetic-only")


@pytest.fixture
def harness(monkeypatch):
    events, transports, accounts = [], [], []
    audio = SimpleNamespace(setNullDev=lambda: None, setNoDev=lambda: events.append("clock_stopped"))

    class Endpoint:
        def __init__(self, _):
            events.append("construct")
        def libCreate(self):
            events.append("create")
        def libInit(self, _):
            events.append("init")
        def audDevManager(self):
            return audio
        def transportCreate(self, kind, config):
            transports.append((kind, config))
            return len(transports)
        def libStart(self):
            events.append("start")
        def codecEnum2(self):
            return []
        def libDestroy(self):
            events.append("destroy")

    class Account:
        def __init__(self, *args):
            pass
        def create(self, config):
            accounts.append(config)
        def shutdown(self):
            events.append("account_shutdown")

    def create_verified(config, hostname):
        events.append(("verified", hostname, config))
        return 6 + sum(isinstance(e, tuple) for e in events)

    def reset():
        assert events[-1] == "destroy", "callback storage released before libDestroy"
        events.append("reset")
        return True

    monkeypatch.setattr(module, "AiEndpoint", Endpoint)
    monkeypatch.setattr(module, "LineAccount", Account)
    monkeypatch.setattr(module, "RingMediaPort", lambda *args, **kw: object())
    monkeypatch.setattr(module.Telephony, "_peers_for", lambda *args: None)
    monkeypatch.setattr(module.pj, "ai_phone_set_keepalive", lambda *args: None)
    monkeypatch.setattr(module.pj, "ai_phone_tls_verifier_version", lambda: 1, raising=False)
    monkeypatch.setattr(module.pj, "ai_phone_create_verified_tls_transport", create_verified, raising=False)
    monkeypatch.setattr(module.pj, "ai_phone_reset_tls_verifier", reset, raising=False)
    monkeypatch.setattr(module.pj, "ai_phone_deactivate_tls_verifier", lambda: events.append("deactivate"), raising=False)
    return SimpleNamespace(events=events, transports=transports, accounts=accounts, audio=audio)


def telephony(lines):
    cfg = Config()
    cfg.sip.tls.ca_file = "/synthetic/roots.pem"
    return Telephony(cfg, lines, None, asyncio.Queue(), None, None)


def test_guarded_lines_have_dedicated_factories_and_strict_line_keeps_standard_verification(harness):
    tel = telephony([spec(), spec("other"), spec("strict", "strict")])
    tel.start()
    assert tel.line_transport_ids == {"guarded": 7, "other": 8}
    assert [a.sipConfig.transportId for a in harness.accounts] == [7, 8, 3]
    tls = [(kind, config) for kind, config in harness.transports if kind == module.pj.PJSIP_TRANSPORT_TLS]
    assert len(tls) == 1 and tls[0][1].tlsConfig.verifyServer
    for event in [e for e in harness.events if isinstance(e, tuple)]:
        assert event[1] == "tenant.carrier.test"
        assert event[2].port == 0
        assert event[2].tlsConfig.CaListFile == "/synthetic/roots.pem"
    tel.stop()
    assert harness.events[-2:] == ["destroy", "reset"]
    assert harness.events.index("deactivate") < harness.events.index("account_shutdown")
    assert harness.events.index("account_shutdown") < harness.events.index("clock_stopped") < harness.events.index("destroy")
    assert not tel.line_transport_ids and not tel.transport_ids


def test_guarded_only_does_not_create_unused_shared_tls_listener(harness):
    tel = telephony([spec()])
    tel.start()
    assert "tls" not in tel.transport_ids
    tel.stop()


@pytest.mark.parametrize("capability", [None, lambda: 0])
def test_stale_binding_fails_before_creating_an_endpoint(harness, monkeypatch, capability):
    monkeypatch.setattr(module.pj, "ai_phone_tls_verifier_version", capability)
    with pytest.raises(RuntimeError, match=r"\./scripts/install.sh"):
        telephony([spec()]).start()
    assert harness.events == []


def test_strict_line_does_not_require_new_native_capability(harness, monkeypatch):
    monkeypatch.setattr(module.pj, "ai_phone_tls_verifier_version", None)
    tel = telephony([spec(policy="strict")])
    tel.start()
    tel.stop()
    assert "reset" not in harness.events
    assert harness.events.index("clock_stopped") < harness.events.index("destroy")


def test_native_failure_destroys_endpoint_before_releasing_policy(harness, monkeypatch):
    monkeypatch.setattr(module.pj, "ai_phone_create_verified_tls_transport", lambda *args: -70010)
    tel = telephony([spec()])
    with pytest.raises(RuntimeError, match="guarded.*70010"):
        tel.start()
    assert harness.events[-2:] == ["destroy", "reset"]
    assert tel.ep is None and not tel.line_transport_ids
    assert not harness.accounts


def test_policy_is_retained_when_native_destruction_fails(harness, monkeypatch):
    tel = telephony([spec()])
    tel.start()
    def fail():
        raise RuntimeError("destruction failure")
    tel.ep.libDestroy = fail
    with pytest.raises(RuntimeError, match="destruction failure"):
        tel.stop()
    assert "reset" not in harness.events
    assert tel._uses_verified_tls and tel.ep is not None


def test_media_clock_failure_retains_port_endpoint_and_policy_for_retry(harness, monkeypatch):
    tel = telephony([spec()])
    tel.start()
    endpoint, port = tel.ep, tel.port
    transport_ids, line_transport_ids = dict(tel.transport_ids), dict(tel.line_transport_ids)
    stop_clock = harness.audio.setNoDev

    def fail():
        assert tel.ep is endpoint and tel.port is port
        harness.events.append("clock_stop_failed")
        raise RuntimeError("clock stop failure")

    monkeypatch.setattr(harness.audio, "setNoDev", fail)
    with pytest.raises(RuntimeError, match="clock stop failure"):
        tel.stop()
    assert tel.ep is endpoint and tel.port is port and tel._uses_verified_tls
    assert tel.transport_ids == transport_ids and tel.line_transport_ids == line_transport_ids
    assert "destroy" not in harness.events and "reset" not in harness.events

    monkeypatch.setattr(harness.audio, "setNoDev", stop_clock)
    tel.stop()
    assert harness.events[-3:] == ["clock_stopped", "destroy", "reset"]
    assert tel.ep is None and tel.port is None and not tel._uses_verified_tls
    assert not tel.transport_ids and not tel.line_transport_ids
    events = list(harness.events)
    tel.stop()
    assert harness.events == events


def test_mutated_direct_spec_is_checked_before_endpoint_creation(harness):
    line = spec()
    line.register_via_proxy = False
    with pytest.raises(ValueError, match="register_via_proxy"):
        telephony([line]).start()
    assert harness.events == []

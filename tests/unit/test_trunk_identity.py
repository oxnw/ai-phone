"""A trunk's authentication identity and call identity need not be the same."""
import asyncio

import pytest
import yaml

pytest.importorskip("ai_phone.vendor.pjsua2")

from ai_phone.config import Config, ConfigError, LineConfig, load_config, validate_config
from ai_phone.telephony.endpoint import DialError, LineSpec, Telephony, check_caller_id, id_uri, line_for_request, local_uri


def trunk(**changes):
    values = dict(kind="trunk", user="testuser", registrar="sip:sip.example.org",
                  dial_domain="sip.example.org", number="+15555550100", display_name="Test line",
                  trusted_cidrs=["127.0.0.1/32"])
    values.update(changes)
    return LineConfig(**values)


def spec(**changes):
    return LineSpec.from_config("main", trunk(**changes), "synthetic-password")


def test_omitted_mode_preserves_phone_number_identity():
    line = spec()
    assert line.from_user == "number"
    assert local_uri(line) == '"Test line" <sip:+15555550100@sip.example.org>'


@pytest.mark.parametrize("mode,expected", [("number", "+15555550101"), ("account", "testuser")])
def test_call_identity_is_selected_per_line_without_changing_registration_or_authentication(mode, expected):
    line = spec(from_user=mode, caller_id="+15555550101")
    assert local_uri(line) == f'"Test line" <sip:{expected}@sip.example.org>'
    assert id_uri(line) == '"Test line" <sip:testuser@sip.example.org>'
    tel = Telephony(Config(), [line], None, asyncio.Queue(), None, None)
    tel.transport_ids = {"udp": 1}
    account = tel._account_config(line)
    assert account.idUri == id_uri(line)
    assert account.regConfig.registrarUri == "sip:sip.example.org"
    assert account.sipConfig.authCreds[0].username == "testuser"
    assert account.sipConfig.authCreds[0].data == "synthetic-password"


@pytest.mark.parametrize("username", ["usuario_12345", "Desk.Sales", "office-2", "x", "a" * 64],
                         ids=["underscore", "dot-and-case", "hyphen", "short", "maximum-length"])
@pytest.mark.parametrize("mode", ["number", "account"])
def test_provider_supplied_username_survives_config_registration_and_call_identity(tmp_path, username, mode):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"lines": {"main": {
        "kind": "trunk", "user": username, "registrar": "sip:sip.example.org",
        "dial_domain": "sip.example.org", "number": "+15555550100", "caller_id": "+15555550101",
        "display_name": "Test line", "from_user": mode, "trusted_cidrs": ["127.0.0.1/32"],
    }}}))
    loaded = load_config(path).lines["main"]
    assert loaded.user == username
    line = LineSpec.from_config("main", loaded, "synthetic-password")
    # Programmatic callers receive the same validation without passing through YAML.
    assert spec(user=username, from_user=mode, caller_id="+15555550101") == line
    assert id_uri(line) == f'"Test line" <sip:{username}@sip.example.org>'
    expected_from = username if mode == "account" else "+15555550101"
    assert local_uri(line) == f'"Test line" <sip:{expected_from}@sip.example.org>'
    tel = Telephony(Config(), [line], None, asyncio.Queue(), None, None)
    tel.transport_ids = {"udp": 1}
    account = tel._account_config(line)
    assert account.idUri == id_uri(line)
    assert account.sipConfig.authCreds[0].username == username
    assert account.sipConfig.authCreds[0].data == "synthetic-password"


@pytest.mark.parametrize("username", [
    "", "a" * 65, "bad user", "user\tname", "user\n", "user\r\nX-Header: injected",
    'user"name', "user\\name", "user@example.org", "user;transport=tcp", "user>", "usér",
], ids=["empty", "overlong", "space", "tab", "trailing-newline", "header-injection",
        "quote", "backslash", "at-sign", "parameter", "angle-bracket", "non-ascii"])
@pytest.mark.parametrize("mode", ["number", "account"])
def test_invalid_trunk_usernames_are_rejected_before_header_assembly(username, mode):
    line = trunk(user=username, from_user=mode)
    with pytest.raises(ConfigError, match=r"config\.lines\.main\.user"):
        validate_config(Config(lines={"main": line}), env={})
    with pytest.raises(ValueError, match="line user"):
        LineSpec.from_config("main", line, None)


@pytest.mark.parametrize("mode", ["number", "account"])
def test_sip_username_is_not_accepted_as_a_phone_number_caller_id(mode):
    line = trunk(user="usuario_12345", caller_id="usuario_12345", from_user=mode)
    with pytest.raises(ConfigError, match=r"caller_id: expected E\.164"):
        validate_config(Config(lines={"main": line}), env={})
    with pytest.raises(ValueError, match="line caller_id"):
        LineSpec.from_config("main", line, None)


def test_incoming_request_selects_the_trunk_with_a_punctuated_username():
    line = spec(user="trunk.test_user-1", from_user="account")
    other = LineSpec.from_config("other", trunk(user="other"), None)
    request = "INVITE sip:trunk.test_user-1@127.0.0.1 SIP/2.0\r\n"
    assert line_for_request([other, line], {}, request, "203.0.113.10") == "main"
    assert line_for_request([other], {}, request, "203.0.113.10") is None


@pytest.mark.parametrize("metadata", [{"number": None}, {"caller_id": "+15555550199"}])
def test_account_mode_does_not_require_or_verify_a_number_it_does_not_send(metadata):
    line = spec(from_user="account", **metadata)
    check_caller_id(line)
    assert local_uri(line) == '"Test line" <sip:testuser@sip.example.org>'


def test_number_mode_still_requires_a_number_and_verified_caller_id():
    with pytest.raises(DialError, match="neither"):
        check_caller_id(spec(from_user="number", number=None))
    with pytest.raises(DialError, match="verified"):
        check_caller_id(spec(from_user="number", caller_id="+15555550199"))


@pytest.mark.parametrize("field,value", [("number", "not-a-number"), ("caller_id", "+15555550100\r\nX: y"),
                                        ("user", "bad user")])
def test_account_mode_keeps_header_and_metadata_validation(field, value):
    line = trunk(from_user="account", **{field: value})
    with pytest.raises(ConfigError, match=field):
        validate_config(Config(lines={"main": line}), env={})
    with pytest.raises(ValueError, match=field):
        LineSpec.from_config("main", line, None)


@pytest.mark.parametrize("mode", ["account", "number"])
def test_yaml_loads_each_mode(tmp_path, mode):
    path = tmp_path / "config.yaml"
    path.write_text("lines:\n  main:\n    kind: trunk\n    user: testuser\n"
                    "    registrar: sip:sip.example.org\n    dial_domain: sip.example.org\n"
                    f"    from_user: {mode}\n")
    assert load_config(path).lines["main"].from_user == mode


@pytest.mark.parametrize("bad", ["auto", "ACCOUNT", None, True, 1])
def test_invalid_modes_are_rejected_at_both_validation_boundaries(bad):
    line = trunk(from_user=bad)
    with pytest.raises(ConfigError, match="from_user"):
        validate_config(Config(lines={"main": line}), env={})
    with pytest.raises(ValueError, match="from_user"):
        LineSpec.from_config("main", line, None)


def test_account_mode_is_rejected_on_a_sip_address_line():
    line = LineConfig(kind="sip", user="agents", aor="sip:agents@example.org",
                      accept_invites_from="lan", from_user="account")
    with pytest.raises(ConfigError, match="from_user"):
        validate_config(Config(lines={"agents": line}), env={})
    with pytest.raises(ValueError, match="from_user"):
        LineSpec.from_config("agents", line, None)


@pytest.mark.parametrize("mode", ["number", "account"])
def test_outbound_identity_does_not_widen_inbound_trust(mode):
    line = spec(from_user=mode, registrar="sip:127.0.0.1", trusted_cidrs=[])
    tel = Telephony(Config(), [line], None, asyncio.Queue(), None, None)
    peers = tel._peers_for(line)
    assert line.accept_invites_from == "registered_trunk"
    assert peers.contains("127.0.0.1")
    assert not peers.contains("203.0.113.1")


def test_sip_address_line_keeps_its_configured_uri():
    config = LineConfig(kind="sip", user="agents", aor="sip:agents@example.org",
                        caller_id="sip:caller@example.org", accept_invites_from="lan")
    line = LineSpec.from_config("agents", config, None)
    check_caller_id(line)
    assert line.from_user == "number"
    assert id_uri(line) == '"ai-phone" <sip:agents@example.org>'
    assert local_uri(line) == "sip:caller@example.org"

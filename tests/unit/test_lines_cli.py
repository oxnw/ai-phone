import json
from pathlib import Path

import pytest

from ai_phone.cli import main
from ai_phone.config import load_config
from ai_phone.lines.journal import Journal
from ai_phone.secrets import load_secrets


@pytest.fixture
def home(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AI_PHONE_HOME", str(tmp_path))
    monkeypatch.setenv("FAKE_KEY", "k")
    monkeypatch.setattr("builtins.input", lambda *a: "y")
    (tmp_path / "config.yaml").write_text("inbound:\n  greeting: Hi\n")
    return tmp_path


def test_add_fake_provider_provisions_writes_line_and_secret(home: Path, capsys):
    rc = main(["lines", "add", "fake", "--key-env", "FAKE_KEY", "--id", "main", "--yes"])
    assert rc == 0
    cfg = load_config(home / "config.yaml")
    ln = cfg.lines["main"]
    assert ln.kind == "trunk" and ln.registrar == "sip:sip.fake.invalid" and ln.provider.name == "fake" and ln.provider.state == "ready"
    assert ln.password_env == "AI_PHONE_LINE_MAIN_PASSWORD"
    pw = load_secrets(home / "secrets.env")["AI_PHONE_LINE_MAIN_PASSWORD"]
    assert len(pw) == 24
    assert Journal.open(home / "lines", "main").state == "ready"
    out = capsys.readouterr().out
    assert "+442079460000" in out and "ai-phone run" in out and pw not in out


def test_add_refuses_duplicate_id(home: Path):
    assert main(["lines", "add", "fake", "--key-env", "FAKE_KEY", "--id", "main", "--yes"]) == 0
    assert main(["lines", "add", "fake", "--key-env", "FAKE_KEY", "--id", "main", "--yes"]) == 3


def test_pending_then_resume(home: Path, monkeypatch):
    monkeypatch.setenv("AI_PHONE_FAKE_PENDING", "2")
    assert main(["lines", "add", "fake", "--key-env", "FAKE_KEY", "--id", "main", "--yes"]) == 2
    assert load_config(home / "config.yaml").lines["main"].provider.state == "pending"
    monkeypatch.setenv("AI_PHONE_FAKE_PENDING", "0")
    assert main(["lines", "resume", "main", "--key-env", "FAKE_KEY"]) == 0
    assert load_config(home / "config.yaml").lines["main"].provider.state == "ready"


def test_remove_releases_and_deletes_secret(home: Path):
    main(["lines", "add", "fake", "--key-env", "FAKE_KEY", "--id", "main", "--yes"])
    assert main(["lines", "remove", "main", "--key-env", "FAKE_KEY", "--yes"]) == 0
    assert load_config(home / "config.yaml").lines == {}
    assert "AI_PHONE_LINE_MAIN_PASSWORD" not in load_secrets(home / "secrets.env")
    assert not (home / "lines" / "main.journal.json").exists()


def test_remove_refuses_to_release_without_the_journal(home: Path, capsys):
    """A lost journal releases nothing at the carrier, so removing the line would delete the only local record
    of resources that are still billing. Refuse, name the ids from the config, and keep everything local."""
    main(["lines", "add", "fake", "--key-env", "FAKE_KEY", "--id", "main", "--yes"])
    (home / "lines" / "main.journal.json").unlink()
    assert main(["lines", "remove", "main", "--key-env", "FAKE_KEY", "--yes"]) == 3
    out = capsys.readouterr().out
    assert "journal" in out and "number_id=number-1" in out and "--local-only" in out
    assert "main" in load_config(home / "config.yaml").lines
    assert "AI_PHONE_LINE_MAIN_PASSWORD" in load_secrets(home / "secrets.env")
    assert json.loads((home / "fake-carrier.json").read_text())["carrier"]["number"]["name"] == "+442079460000"
    assert main(["lines", "remove", "main", "--local-only", "--yes"]) == 0  # the way out, with the ids to release by hand
    assert "number_id=number-1" in capsys.readouterr().out


def test_remove_refuses_when_the_journal_lost_its_steps(home: Path, capsys):
    """Same defect from the other direction: a journal that no longer accounts for the ids the config recorded."""
    main(["lines", "add", "fake", "--key-env", "FAKE_KEY", "--id", "main", "--yes"])
    j = home / "lines" / "main.journal.json"
    j.write_text(json.dumps({**json.loads(j.read_text()), "steps": []}))
    assert main(["lines", "remove", "main", "--key-env", "FAKE_KEY", "--yes"]) == 3
    assert "number_id=number-1" in capsys.readouterr().out
    assert "main" in load_config(home / "config.yaml").lines


def test_remove_refuses_after_a_failed_resume_recreated_the_journal(home: Path, capsys):
    """`Journal.open` writes the file it invents, so a failed `lines resume` leaves an empty journal behind:
    the file existing must not be read as 'a release here would delete something'."""
    main(["lines", "add", "fake", "--key-env", "FAKE_KEY", "--id", "main", "--yes"])
    (home / "lines" / "main.journal.json").unlink()
    assert main(["lines", "resume", "main", "--key-env", "FAKE_KEY"]) == 3
    assert (home / "lines" / "main.journal.json").exists()  # invented by the failed resume, no steps in it
    assert main(["lines", "remove", "main", "--key-env", "FAKE_KEY", "--yes"]) == 3
    assert "records no carrier steps" in capsys.readouterr().out
    assert "main" in load_config(home / "config.yaml").lines


def test_remove_refuses_when_neither_journal_nor_config_kept_the_ids(home: Path, capsys):
    """The narrow case where nothing local knows the ids: still refuse, because the carrier may hold them."""
    from ai_phone.lines.config_writer import read_raw, write_raw
    main(["lines", "add", "fake", "--key-env", "FAKE_KEY", "--id", "main", "--yes"])
    raw = read_raw(home / "config.yaml")
    raw["lines"]["main"]["provider"]["resources"] = {}  # a hand-edited config, or `_commit`'s failed-state fallback
    write_raw(home / "config.yaml", raw)
    j = home / "lines" / "main.journal.json"
    j.write_text(json.dumps({**json.loads(j.read_text()), "steps": []}))
    assert main(["lines", "remove", "main", "--key-env", "FAKE_KEY", "--yes"]) == 3
    assert "find them by +442079460000" in capsys.readouterr().out
    assert "main" in load_config(home / "config.yaml").lines
    assert json.loads((home / "fake-carrier.json").read_text())["carrier"]["number"]["name"] == "+442079460000"


def test_remove_without_key_needs_local_only(home: Path, capsys):
    main(["lines", "add", "fake", "--key-env", "FAKE_KEY", "--id", "main", "--yes"])
    assert main(["lines", "remove", "main", "--yes"]) == 3
    assert "--local-only" in capsys.readouterr().out
    assert main(["lines", "remove", "main", "--local-only", "--yes"]) == 0


def test_add_trunk_and_sip_by_hand(home: Path):
    assert main(["lines", "add", "trunk", "--id", "byo", "--number", "+442079460001", "--registrar", "sip:sip.example.com",
                 "--user", "byouser1", "--password-env", "BYO_PW", "--trusted-cidr", "198.51.100.0/24"]) == 0
    assert main(["lines", "add", "sip", "--id", "agents", "--user", "agents", "--host", "home.example.org", "--trusted", "203.0.113.7"]) == 0
    cfg = load_config(home / "config.yaml")
    assert cfg.lines["byo"].dial_domain == "sip.example.com" and cfg.lines["byo"].caller_id == "+442079460001"
    assert cfg.lines["agents"].aor == "sip:agents@home.example.org" and cfg.lines["agents"].accept_invites_from == "trusted"


@pytest.mark.parametrize("mode,number", [(None, "+15555550100"), ("number", "+15555550100"),
                                         ("account", "+15555550100"), ("account", None)])
def test_manual_trunk_from_user_is_saved_and_defaults_to_number(home, mode, number):
    from ai_phone.lines.config_writer import read_raw
    args = ["lines", "add", "trunk", "--id", "byo", "--registrar", "sip:trunk.example.org",
            "--user", "account1", "--password-env", "BYO_PW"]
    if mode is not None:
        args += ["--from-user", mode]
    if number is not None:
        args += ["--number", number]
    assert main(args) == 0
    line = load_config(home / "config.yaml").lines["byo"]
    assert line.from_user == (mode or "number")
    assert line.number == number
    if mode == "account":
        assert read_raw(home / "config.yaml")["lines"]["byo"]["from_user"] == "account"
        assert line.caller_id is None


@pytest.mark.parametrize("provider", ["sip", "fake", "telnyx", "twilio"])
@pytest.mark.parametrize("mode", ["number", "account"])
def test_from_user_option_refuses_non_manual_trunks_before_provider_access(home, monkeypatch, capsys, provider, mode):
    monkeypatch.setattr("ai_phone.cli_lines._provider", lambda *a: pytest.fail("must not contact a carrier"))
    before = (home / "config.yaml").read_bytes()
    assert main(["lines", "add", provider, "--from-user", mode]) == 3
    assert "--from-user only applies to a trunk" in capsys.readouterr().out
    assert (home / "config.yaml").read_bytes() == before


@pytest.mark.parametrize("mode,number", [("number", None), ("account", "not-a-number")])
def test_manual_trunk_from_user_keeps_number_validation(home, mode, number):
    args = ["lines", "add", "trunk", "--id", "byo", "--registrar", "sip:trunk.example.org",
            "--user", "account1", "--password-env", "BYO_PW", "--from-user", mode]
    if number is not None:
        args += ["--number", number]
    assert main(args) == 3
    assert not load_config(home / "config.yaml").lines


def test_account_from_refuses_caller_id_verification_before_provider_access(home, monkeypatch, capsys):
    from ai_phone.lines.config_writer import update_line
    assert main(["lines", "add", "fake", "--key-env", "FAKE_KEY", "--id", "main", "--yes"]) == 0
    update_line(home / "config.yaml", "main", from_user="account")
    before = (home / "config.yaml").read_bytes()
    monkeypatch.setattr("ai_phone.cli_lines._provider", lambda *a: pytest.fail("must not contact a carrier"))
    assert main(["lines", "verify-caller-id", "main", "+15555550101", "--key-env", "FAKE_KEY"]) == 3
    assert "configure the caller ID with your provider" in capsys.readouterr().out
    assert (home / "config.yaml").read_bytes() == before


@pytest.mark.parametrize("mode,kind", [("invalid", "trunk"), ("account", "sip")])
def test_manual_line_refuses_invalid_identity_policy(mode, kind):
    from ai_phone.cli_lines import manual_line
    with pytest.raises(ValueError, match="from_user"):
        manual_line(kind, user="account1", host="example.org", from_user=mode)


def test_forward_prints_codes_and_records_divert(home: Path, capsys):
    main(["lines", "add", "fake", "--key-env", "FAKE_KEY", "--id", "main", "--yes"])
    assert main(["lines", "forward", "main", "--when", "no-answer", "--from", "+447700900123"]) == 0
    out = capsys.readouterr().out
    assert "**61*+442079460000#" in out and "*#61#" in out and "##002#" in out
    assert load_config(home / "config.yaml").lines["main"].diverted_from == ["+447700900123"]


def test_verify_caller_id_records_it(home: Path, monkeypatch):
    main(["lines", "add", "fake", "--key-env", "FAKE_KEY", "--id", "main", "--yes"])
    monkeypatch.setattr("builtins.input", lambda *a: "123456")
    assert main(["lines", "verify-caller-id", "main", "+447700900123", "--key-env", "FAKE_KEY"]) == 0
    v = load_config(home / "config.yaml").lines["main"].verified_caller_ids
    assert v and v[0].number == "+447700900123" and v[0].provider == "fake"


def test_persona_option_is_gone(home: Path):
    with pytest.raises(SystemExit) as e:
        main(["lines", "add", "fake", "--key-env", "FAKE_KEY", "--id", "main", "--persona", "x", "--yes"])
    assert e.value.code == 2


def test_added_lines_carry_no_persona(home: Path):
    from ai_phone.lines.config_writer import read_raw
    assert main(["lines", "add", "fake", "--key-env", "FAKE_KEY", "--id", "main", "--yes"]) == 0
    assert "persona" not in read_raw(home / "config.yaml")["lines"]["main"]


def test_unloadable_result_restores_the_config(home: Path, monkeypatch):
    from ai_phone.providers.base import ProvisionResult
    monkeypatch.setattr("ai_phone.providers.fake.FakeProvider.provision",
                        lambda self, journal, number, line_id, user, pw: ProvisionResult("ready", "ok", {"kind": "trunk", "registrar": "not a uri", "dial_domain": "x", "user": user}))
    from ai_phone.lines.config_writer import read_raw
    before = read_raw(home / "config.yaml")
    assert main(["lines", "add", "fake", "--key-env", "FAKE_KEY", "--id", "main", "--yes"]) == 3
    after = read_raw(home / "config.yaml")
    assert (after.get("lines") or {}) == {} and after["inbound"] == before["inbound"]  # restored (an empty lines: {} is equivalent)
    assert (home / "lines" / "main.journal.json").exists()  # the carrier side is kept for resume/remove


def test_failed_order_marks_an_existing_pending_line(home: Path, monkeypatch):
    monkeypatch.setenv("AI_PHONE_FAKE_PENDING", "5")
    assert main(["lines", "add", "fake", "--key-env", "FAKE_KEY", "--id", "main", "--yes"]) == 2
    from ai_phone.providers.base import ProvisionResult
    monkeypatch.setattr("ai_phone.providers.fake.FakeProvider.provision",
                        lambda self, journal, number, line_id, user, pw: ProvisionResult("failed", "the carrier rejected the order"))
    assert main(["lines", "resume", "main", "--key-env", "FAKE_KEY"]) == 3
    assert load_config(home / "config.yaml").lines["main"].provider.state == "failed"


def test_list(home: Path, capsys):
    main(["lines", "add", "fake", "--key-env", "FAKE_KEY", "--id", "main", "--yes"])
    assert main(["lines", "list"]) == 0
    assert "main" in capsys.readouterr().out


def test_manual_duplicate_id_keeps_the_existing_line(home: Path, capsys):
    main(["lines", "add", "fake", "--key-env", "FAKE_KEY", "--id", "main", "--yes"])
    assert main(["lines", "add", "sip", "--id", "main", "--user", "agents", "--host", "home.example.org"]) == 3
    cfg = load_config(home / "config.yaml")
    assert cfg.lines["main"].kind == "trunk" and cfg.lines["main"].provider.name == "fake"


def test_manual_trunk_without_registrar_is_a_clean_error(home: Path, capsys):
    assert main(["lines", "add", "trunk", "--id", "byo", "--number", "+442079460001", "--user", "byouser1"]) == 3
    assert "--registrar" in capsys.readouterr().out


def test_resume_after_a_crash_mid_provision(home: Path, monkeypatch):
    monkeypatch.setenv("AI_PHONE_FAKE_CRASH_AFTER", "connection")
    assert main(["lines", "add", "fake", "--key-env", "FAKE_KEY", "--id", "main", "--yes"]) == 3
    assert "main" not in (load_config(home / "config.yaml").lines)
    monkeypatch.delenv("AI_PHONE_FAKE_CRASH_AFTER")
    assert main(["lines", "resume", "main", "--key-env", "FAKE_KEY"]) == 0
    ln = load_config(home / "config.yaml").lines["main"]
    assert ln.provider.state == "ready" and ln.number == "+442079460000"
    j = Journal.open(home / "lines", "main")
    assert "carrier_ready" in j.steps_done() and "local_committed" in j.steps_done()


def test_provider_errors_exit_3_not_traceback(home: Path, monkeypatch, capsys):
    monkeypatch.setenv("AI_PHONE_FAKE_CRASH_AFTER", "profile")
    assert main(["lines", "add", "fake", "--key-env", "FAKE_KEY", "--id", "main", "--yes"]) == 3
    assert "error:" in capsys.readouterr().out


def test_resume_works_for_a_journal_without_the_target_step(home: Path, monkeypatch):
    monkeypatch.setenv("AI_PHONE_FAKE_CRASH_AFTER", "number")
    assert main(["lines", "add", "fake", "--key-env", "FAKE_KEY", "--id", "main", "--yes"]) == 3
    j = Journal.open(home / "lines", "main")
    j.doc["steps"] = [s for s in j.doc["steps"] if s["name"] != "target"]  # a journal from before the target step existed
    j._save()
    monkeypatch.delenv("AI_PHONE_FAKE_CRASH_AFTER")
    assert main(["lines", "resume", "main", "--key-env", "FAKE_KEY"]) == 0
    assert load_config(home / "config.yaml").lines["main"].number == "+442079460000"


def test_pick_number_stops_cleanly_on_a_non_numeric_answer(monkeypatch, capsys):
    from types import SimpleNamespace

    from ai_phone import cli_lines

    class Provider:
        name = "fake"

        def search_numbers(self, country, area, limit):
            return [SimpleNamespace(e164="+15550000001", monthly_cost=1.0, currency="USD", requires_regulatory=False)]

    args = SimpleNamespace(number=None, country="US", area=None, yes=False)
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    with pytest.raises(SystemExit):
        cli_lines._pick_number(Provider(), args)
    assert "stopped; nothing was bought" in capsys.readouterr().out
    monkeypatch.setattr("builtins.input", lambda prompt="": "7")
    with pytest.raises(SystemExit):
        cli_lines._pick_number(Provider(), args)
    assert "from 1 to 1" in capsys.readouterr().out


def test_bare_system_exit_message_is_printed_not_raised(home: Path, capsys, monkeypatch):
    """`--key-env` names a variable that is not set: one line on stderr and exit 2, never a traceback."""
    monkeypatch.delenv("NOPE_KEY", raising=False)
    rc = main(["lines", "--config", str(home / "config.yaml"), "add", "telnyx", "--key-env", "NOPE_KEY", "--yes"])
    assert rc == 2
    assert "NOPE_KEY" in capsys.readouterr().err


def test_missing_config_is_one_line(home: Path, capsys):
    rc = main(["lines", "--config", str(home / "absent.yaml"), "list"])
    assert rc != 0
    out = capsys.readouterr()
    assert "does not exist" in out.out + out.err


# ---- a number the person already has ------------------------------------------------------------

def _fake_carrier(home: Path) -> dict:
    return json.loads((home / "fake-carrier.json").read_text())


def test_add_existing_connects_without_buying(home: Path, monkeypatch, capsys):
    asked = []
    monkeypatch.setattr("builtins.input", lambda prompt="": asked.append(prompt) or "y")
    assert main(["lines", "add", "fake", "--key-env", "FAKE_KEY", "--id", "main", "--existing", "+442079460099"]) == 0
    assert "Connect +442079460099 on your fake account to ai-phone?" in asked[0] and "Nothing is bought" in asked[0]
    ln = load_config(home / "config.yaml").lines["main"]
    assert ln.number == "+442079460099" and ln.provider.state == "ready" and ln.password_env == "AI_PHONE_LINE_MAIN_PASSWORD"
    carrier = _fake_carrier(home)
    assert carrier["carrier"]["owned"]["name"] == "+442079460099" and "number" not in carrier["carrier"] and "number" not in carrier["created"]
    assert "Line main is ready: +442079460099" in capsys.readouterr().out
    assert Journal.open(home / "lines", "main").doc["steps"][0]["planned"] == {"phone_number": "+442079460099", "brought": True, "source_id": "owned-1"}


def test_add_existing_stops_when_the_answer_is_no(home: Path, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    assert main(["lines", "add", "fake", "--key-env", "FAKE_KEY", "--id", "main", "--existing", "+442079460099"]) == 3
    assert "main" not in load_config(home / "config.yaml").lines
    assert not (home / "lines" / "main.journal.json").exists() and not (home / "fake-carrier.json").exists()


def test_add_existing_refuses_a_number_not_on_the_account(home: Path, capsys):
    assert main(["lines", "add", "fake", "--key-env", "FAKE_KEY", "--id", "main", "--existing", "+442079460001", "--yes"]) == 3
    assert "is not on your fake account" in capsys.readouterr().out
    assert not (home / "lines" / "main.journal.json").exists()


def test_add_existing_checks_its_arguments(home: Path, capsys):
    assert main(["lines", "add", "fake", "--key-env", "FAKE_KEY", "--existing", "+442079460099", "--number", "+442079460000", "--yes"]) == 3
    assert "--existing cannot be combined with --number or --area" in capsys.readouterr().out
    assert main(["lines", "add", "fake", "--key-env", "FAKE_KEY", "--existing", "+442079460099", "--area", "20", "--yes"]) == 3
    assert "--existing cannot be combined with --number or --area" in capsys.readouterr().out
    assert main(["lines", "add", "fake", "--key-env", "FAKE_KEY", "--existing", "02079460099", "--yes"]) == 3
    assert "--existing must be E.164, like +15550100000" in capsys.readouterr().out
    assert load_config(home / "config.yaml").lines == {}


def test_remove_keeps_a_brought_number(home: Path, monkeypatch):
    assert main(["lines", "add", "fake", "--key-env", "FAKE_KEY", "--id", "main", "--existing", "+442079460099", "--yes"]) == 0
    asked = []
    monkeypatch.setattr("builtins.input", lambda prompt="": asked.append(prompt) or "y")
    assert main(["lines", "remove", "main", "--key-env", "FAKE_KEY"]) == 0
    assert asked == ["Disconnect +442079460099 from ai-phone and delete what ai-phone created on your fake account? "
                     "The number stays on your account. [y/N] "]
    carrier = _fake_carrier(home)["carrier"]
    assert carrier["owned"]["name"] == "+442079460099" and carrier["owned"]["connection"] is None
    assert load_config(home / "config.yaml").lines == {}


def test_remove_of_a_bought_number_still_warns_it_is_released(home: Path, monkeypatch):
    main(["lines", "add", "fake", "--key-env", "FAKE_KEY", "--id", "main", "--yes"])
    asked = []
    monkeypatch.setattr("builtins.input", lambda prompt="": asked.append(prompt) or "n")
    assert main(["lines", "remove", "main", "--key-env", "FAKE_KEY"]) == 3
    assert asked == ["Release +442079460000 from your fake account? This cannot be undone. [y/N] "]


def test_resume_of_a_brought_number_attaches_again(home: Path, monkeypatch):
    monkeypatch.setenv("AI_PHONE_FAKE_CRASH_AFTER", "connection")
    assert main(["lines", "add", "fake", "--key-env", "FAKE_KEY", "--id", "main", "--existing", "+442079460099", "--yes"]) == 3
    assert "main" not in load_config(home / "config.yaml").lines
    monkeypatch.delenv("AI_PHONE_FAKE_CRASH_AFTER")
    assert main(["lines", "resume", "main", "--key-env", "FAKE_KEY"]) == 0
    ln = load_config(home / "config.yaml").lines["main"]
    assert ln.provider.state == "ready" and ln.number == "+442079460099"
    carrier = _fake_carrier(home)
    assert "number" not in carrier["carrier"] and "number" not in carrier["created"] and carrier["created"]["connection"] == 1


def test_key_env_is_read_from_secrets_env(home: Path, monkeypatch, capsys):
    from ai_phone.secrets import write_secret
    monkeypatch.delenv("FAKE_KEY2", raising=False)
    write_secret("FAKE_KEY2", "k", home / "secrets.env")
    assert main(["lines", "add", "fake", "--key-env", "FAKE_KEY2", "--id", "main", "--yes"]) == 0
    assert load_config(home / "config.yaml").lines["main"].provider.state == "ready"
    monkeypatch.delenv("FAKE_KEY3", raising=False)
    assert main(["lines", "add", "fake", "--key-env", "FAKE_KEY3", "--id", "other", "--yes"]) == 2
    err = capsys.readouterr().err
    assert f"--key-env: FAKE_KEY3 is not set in the environment or in {home / 'secrets.env'}" in err


def test_provision_flow_sends_messages_to_say(home: Path, capsys):
    from ai_phone import paths
    from ai_phone.cli_lines import EXIT_ERROR, EXIT_OK, provision_flow
    from ai_phone.providers.fake import FakeProvider
    p = FakeProvider()
    msgs = []
    rc = provision_flow(p, Journal.open(paths.lines_dir(), "main", "fake"), home / "config.yaml", "main",
                        p.search_numbers("GB", None, 1)[0], say=msgs.append)
    assert rc == EXIT_OK and msgs == ["Line main is ready: +442079460000. Run `ai-phone run`; it will register."]
    msgs.clear()
    rc = provision_flow(p, Journal.open(paths.lines_dir(), "home", "fake"), home / "config.yaml", "home",
                        p.owned_numbers()[0], existing=True, say=msgs.append)
    assert rc == EXIT_OK and msgs == ["Line home is ready: +442079460099. Run `ai-phone run`; it will register."]
    msgs.clear()
    broken = FakeProvider(crash_after="profile")
    rc = provision_flow(broken, Journal.open(paths.lines_dir(), "work", "fake"), home / "config.yaml", "work",
                        broken.search_numbers("GB", None, 2)[1], say=msgs.append)
    assert rc == EXIT_ERROR and len(msgs) == 1 and msgs[0].startswith("error: simulated crash after profile")
    assert capsys.readouterr().out == ""


def test_next_line_id_keeps_its_old_name():
    from ai_phone import cli_lines
    assert cli_lines.next_line_id({"lines": {"fake": {}, "fake-2": {}}}, "fake") == "fake-3"
    assert cli_lines.next_line_id({}, "fake") == "fake" and cli_lines._next_id is cli_lines.next_line_id


def test_manual_line_and_write_line(home: Path):
    from ai_phone.cli_lines import manual_line, write_line
    from ai_phone.config import ConfigError
    from ai_phone.lines.config_writer import read_raw
    cfg = home / "config.yaml"
    write_line(cfg, "alex", manual_line("sip", user="alex", host="192.0.2.10"))
    ln = load_config(cfg).lines["alex"]
    assert ln.kind == "sip" and ln.aor == "sip:alex@192.0.2.10" and ln.accept_invites_from == "lan"
    trunk = manual_line("trunk", number="+442079460000", registrar="sip:sip.example.com:5061;transport=tls", user="sam1",
                        password_env="SAM_PW", transport="tls", trusted_cidrs=["203.0.113.0/24"])
    assert trunk["dial_domain"] == "sip.example.com" and trunk["caller_id"] == "+442079460000" and "outbound_proxy" not in trunk
    assert trunk["trusted_cidrs"] == ["203.0.113.0/24"] and trunk["accept_invites_from"] == "registered_trunk"
    with pytest.raises(ValueError, match="the number must be E.164, like \\+442079460000"):
        manual_line("trunk", number="02079460000", registrar="sip:sip.example.com", user="sam1", password_env="SAM_PW")
    with pytest.raises(ValueError, match="a trunk line needs a registrar, a user and a password"):
        manual_line("trunk", number="+442079460000", registrar="sip:sip.example.com", user="sam1")
    with pytest.raises(ValueError, match="a sip line needs a user and a host"):
        manual_line("sip", user="alex")
    before = read_raw(cfg)
    with pytest.raises(ConfigError):
        write_line(cfg, "byo", manual_line("trunk", number="+442079460000", registrar="nonsense", user="abcd", password_env="X_PW"))
    assert read_raw(cfg) == before
    with pytest.raises(ValueError, match="already exists"):
        write_line(cfg, "alex", manual_line("sip", user="sam", host="192.0.2.11"))
    with pytest.raises(ValueError, match="line ids must match"):
        write_line(cfg, "Not Valid", manual_line("sip", user="sam", host="192.0.2.11"))
    assert read_raw(cfg) == before


# ---- review round 1 ------------------------------------------------------------------------------

def test_an_unfinished_buy_blocks_a_bring_under_the_same_id(home: Path, monkeypatch, capsys):
    """A journal left by a failed purchase must not be reused for a different number: its release would mix the two."""
    monkeypatch.setenv("AI_PHONE_FAKE_CRASH_AFTER", "number")
    assert main(["lines", "add", "fake", "--key-env", "FAKE_KEY", "--id", "main", "--yes"]) == 3
    monkeypatch.delenv("AI_PHONE_FAKE_CRASH_AFTER")
    capsys.readouterr()
    assert main(["lines", "add", "fake", "--key-env", "FAKE_KEY", "--id", "main", "--existing", "+442079460099", "--yes"]) == 3
    out = capsys.readouterr().out
    assert "line main has an unfinished setup for +442079460000" in out
    assert "ai-phone lines resume main" in out and "ai-phone lines remove main" in out
    assert "owned" not in _fake_carrier(home)["carrier"]
    steps = Journal.open(home / "lines", "main").doc["steps"]
    assert steps[0]["planned"] == {"phone_number": "+442079460000", "brought": False, "source_id": "src-0"} and "claim" not in [s["name"] for s in steps]
    # the same request again is a retry, not a mix: it finishes the purchase
    assert main(["lines", "add", "fake", "--key-env", "FAKE_KEY", "--id", "main", "--yes"]) == 0
    assert load_config(home / "config.yaml").lines["main"].number == "+442079460000"


def test_an_unfinished_bring_blocks_a_buy_under_the_same_id(home: Path, monkeypatch, capsys):
    monkeypatch.setenv("AI_PHONE_FAKE_CRASH_AFTER", "connection")
    assert main(["lines", "add", "fake", "--key-env", "FAKE_KEY", "--id", "main", "--existing", "+442079460099", "--yes"]) == 3
    monkeypatch.delenv("AI_PHONE_FAKE_CRASH_AFTER")
    capsys.readouterr()
    assert main(["lines", "add", "fake", "--key-env", "FAKE_KEY", "--id", "main", "--yes"]) == 3
    assert "line main has an unfinished setup for +442079460099" in capsys.readouterr().out
    assert "number" not in _fake_carrier(home)["carrier"] and "number" not in _fake_carrier(home)["created"]
    assert main(["lines", "resume", "main", "--key-env", "FAKE_KEY"]) == 0  # the way forward still works
    assert load_config(home / "config.yaml").lines["main"].number == "+442079460099"


def test_a_number_that_is_already_a_line_is_refused(home: Path, capsys):
    from ai_phone import paths
    from ai_phone.cli_lines import EXIT_ERROR, provision_flow
    from ai_phone.providers.fake import FakeProvider
    assert main(["lines", "add", "fake", "--key-env", "FAKE_KEY", "--id", "main", "--existing", "+442079460099", "--yes"]) == 0
    capsys.readouterr()
    assert main(["lines", "add", "fake", "--key-env", "FAKE_KEY", "--id", "other", "--existing", "+442079460099", "--yes"]) == 3
    assert "error: +442079460099 is already line main" in capsys.readouterr().out
    assert not (home / "lines" / "other.journal.json").exists(), "a refusal leaves no journal behind"
    assert _fake_carrier(home)["carrier"]["owned"]["connection"] == "connection-1"
    assert "AI_PHONE_LINE_OTHER_PASSWORD" not in load_secrets(home / "secrets.env")
    p = FakeProvider()  # the wizard's way in: a bought number that is already a line is refused the same way
    main(["lines", "add", "fake", "--key-env", "FAKE_KEY", "--id", "shop", "--yes"])
    msgs = []
    rc = provision_flow(p, Journal.open(paths.lines_dir(), "shop-2", "fake"), home / "config.yaml", "shop-2",
                        p.search_numbers("GB", None, 1)[0], say=msgs.append)
    assert rc == EXIT_ERROR and msgs == ["error: +442079460000 is already line shop"]


def _brought_line(home: Path) -> None:
    assert main(["lines", "add", "fake", "--key-env", "FAKE_KEY", "--id", "main", "--existing", "+442079460099", "--yes"]) == 0
    j = Journal.open(home / "lines", "main")
    claim = next(s for s in j.doc["steps"] if s["name"] == "claim")
    claim["result"]["previous_voice_url"] = "https://sam:hunter22@hooks.example.com/voice"  # a URL can carry credentials
    j._save()


def test_keeping_a_brought_line_local_never_says_to_release_the_number(home: Path, capsys):
    _brought_line(home)
    capsys.readouterr()
    assert main(["lines", "remove", "main", "--local-only", "--yes"]) == 0
    out = capsys.readouterr().out
    assert "Kept in your carrier account (release them in the portal): profile_id=profile-1, connection_id=connection-1\n" in out
    assert ("+442079460099 stays on your fake account; reset its voice URL / connection in the portal "
            "if you want its calls back") in out
    for leaked in ("owned-1", "hunter22", "previous", "number_brought", "True", "password_env", "AI_PHONE_LINE"):
        assert leaked not in out


def test_a_brought_line_without_its_journal_never_says_to_release_the_number(home: Path, capsys):
    _brought_line(home)
    (home / "lines" / "main.journal.json").unlink()
    capsys.readouterr()
    assert main(["lines", "remove", "main", "--key-env", "FAKE_KEY", "--yes"]) == 3
    out = capsys.readouterr().out
    assert "Release them on your fake account first (profile_id=profile-1, connection_id=connection-1)" in out
    assert "+442079460099 stays on your fake account; reset its voice URL / connection in the portal" in out
    assert "owned-1" not in out and "main" in load_config(home / "config.yaml").lines


def test_keeping_a_bought_line_local_prints_carrier_ids_only(home: Path, capsys):
    main(["lines", "add", "fake", "--key-env", "FAKE_KEY", "--id", "main", "--yes"])
    capsys.readouterr()
    assert main(["lines", "remove", "main", "--local-only", "--yes"]) == 0
    out = capsys.readouterr().out
    assert "Kept in your carrier account (release them in the portal): profile_id=profile-1, connection_id=connection-1, number_id=number-1\n" in out
    assert "stays on your" not in out and "password_env" not in out and "user=" not in out


def test_remove_after_a_refused_claim_asks_the_disconnect_question_with_the_number(home: Path, monkeypatch):
    from ai_phone import paths
    from ai_phone.cli_lines import EXIT_ERROR, provision_flow
    from ai_phone.providers.base import Number
    from ai_phone.providers.fake import FakeProvider
    msgs = []
    rc = provision_flow(FakeProvider(), Journal.open(paths.lines_dir(), "main", "fake"), home / "config.yaml", "main",
                        Number("+442079460001", None, None, False, "x"), existing=True, say=msgs.append)
    assert rc == EXIT_ERROR and "is not on this fake account" in msgs[0]
    asked = []
    monkeypatch.setattr("builtins.input", lambda prompt="": asked.append(prompt) or "y")
    assert main(["lines", "remove", "main", "--key-env", "FAKE_KEY"]) == 0
    assert asked == ["Disconnect +442079460001 from ai-phone and delete what ai-phone created on your fake account? "
                     "The number stays on your account. [y/N] "]
    assert not (home / "lines" / "main.journal.json").exists()


def test_an_unfinished_setup_under_another_id_blocks_the_same_number(home: Path, monkeypatch, capsys):
    """One number, two line ids: `lines remove` of the first would delete (or re-route) the number the second uses."""
    monkeypatch.setenv("AI_PHONE_FAKE_CRASH_AFTER", "connection")
    assert main(["lines", "add", "fake", "--key-env", "FAKE_KEY", "--id", "shop", "--existing", "+442079460099", "--yes"]) == 3
    monkeypatch.setenv("AI_PHONE_FAKE_CRASH_AFTER", "number")
    assert main(["lines", "add", "fake", "--key-env", "FAKE_KEY", "--id", "desk", "--number", "+442079460001", "--yes"]) == 3
    monkeypatch.delenv("AI_PHONE_FAKE_CRASH_AFTER")
    capsys.readouterr()
    assert main(["lines", "add", "fake", "--key-env", "FAKE_KEY", "--id", "main", "--existing", "+442079460099", "--yes"]) == 3
    assert main(["lines", "add", "fake", "--key-env", "FAKE_KEY", "--id", "main", "--number", "+442079460001", "--yes"]) == 3
    out = capsys.readouterr().out
    assert ("error: line shop has an unfinished setup for +442079460099; run `ai-phone lines resume shop` to finish it "
            "or `ai-phone lines remove shop` to undo it, then try again") in out
    assert "error: line desk has an unfinished setup for +442079460001" in out
    assert not (home / "lines" / "main.journal.json").exists() and "main" not in load_config(home / "config.yaml").lines
    assert main(["lines", "resume", "shop", "--key-env", "FAKE_KEY"]) == 0  # the way forward still works


def test_journal_setups_reads_every_journal_without_writing(home: Path):
    from ai_phone.cli_lines import JournalSetup, journal_setups
    assert journal_setups() == {}
    j = Journal.open(home / "lines", "shop", "fake")
    j.start("target", phone_number="+442079460099", brought=True)
    Journal.open(home / "lines", "empty", "fake")
    early = Journal.open(home / "lines", "early", "fake")
    early.start("profile", name="ai-phone:early")
    old = Journal.open(home / "lines", "old", "fake")
    old.start("number", name="+442079460002")
    old.set_state("rollback_failed")
    (home / "lines" / "broken.journal.json").write_text("{not json")
    before = {p.name: p.stat().st_mtime_ns for p in (home / "lines").iterdir()}
    assert journal_setups() == {
        "broken": JournalSetup("?", False, "?", False),
        "early": JournalSetup("?", False, "pending", False, "fake"),
        "empty": JournalSetup(None, False, "pending", False, "fake"),
        "old": JournalSetup("+442079460002", False, "rollback_failed", False, "fake"),
        "shop": JournalSetup("+442079460099", True, "pending", True, "fake"),
    }
    assert {p.name: p.stat().st_mtime_ns for p in (home / "lines").iterdir()} == before


def test_list_shows_configured_lines_then_unfinished_setups(home: Path, capsys):
    """An interrupted purchase leaves a journal and no config line; `lines list` is where the stop message
    sends people, so it has to show those too, with the command that finishes or undoes each one."""
    assert main(["lines", "add", "fake", "--key-env", "FAKE_KEY", "--id", "main", "--yes"]) == 0
    shop = Journal.open(home / "lines", "shop", "fake")
    shop.start("target", phone_number="+15550100001", brought=False)
    desk = Journal.open(home / "lines", "desk", "fake")
    desk.start("target", phone_number="+442079460000", brought=True)
    desk.set_state("rollback_failed")
    capsys.readouterr()
    assert main(["lines", "list"]) == 0
    assert capsys.readouterr().out == (
        "main         trunk +442079460000 (fake: ready)\n"
        "\n"
        "Unfinished setups\n"
        "  desk         +442079460000  brought · rollback_failed  "
        "finish in the carrier portal, then: ai-phone lines remove desk --local-only\n"
        "  shop         +15550100001   bought · pending           "
        "resume: ai-phone lines resume shop --key-env <NAME>   undo: ai-phone lines remove shop --key-env <NAME>\n"
    )


def test_list_with_no_configured_lines_shows_the_journals_instead_of_the_empty_hint(home: Path, capsys):
    Journal.open(home / "lines", "early", "fake")  # opened, nothing created at the carrier yet
    capsys.readouterr()
    assert main(["lines", "list"]) == 0
    assert capsys.readouterr().out == (
        "Unfinished setups\n"
        "  early        nothing created yet · remove: ai-phone lines remove early --local-only\n"
    )
    (home / "lines" / "early.journal.json").write_text("{not json")
    assert main(["lines", "list"]) == 0
    assert capsys.readouterr().out == (
        "Unfinished setups\n"
        "  early        unreadable · remove: ai-phone lines remove early --local-only\n"
    )


def test_list_never_offers_to_resume_a_journal_that_has_no_number(home: Path, capsys):
    """`target` is the first step provisioning writes, before any carrier call, so a journal without it reached no
    carrier step and has nothing for `resume` to pick up. Claiming "bought" and offering `resume` would send people
    to a command that fails at once."""
    odd = Journal.open(home / "lines", "odd", "fake")
    odd.start("profile", name="ai-phone:odd")
    capsys.readouterr()
    assert main(["lines", "list"]) == 0
    assert capsys.readouterr().out == (
        "Unfinished setups\n"
        "  odd          nothing created yet · remove: ai-phone lines remove odd --local-only\n"
    )


def test_list_keeps_the_undo_for_a_legacy_journal_that_names_a_bought_number(home: Path, capsys):
    """Journals from older versions record the number under `order`/`number` and have no `target` step, so they are
    not resumable — but the number is bought and billing. Calling that "nothing created yet" and offering only
    `--local-only` would drop the one local record of a number that keeps costing money."""
    old = Journal.open(home / "lines", "old", "fake")
    old.start("number", name="+442079460002")
    capsys.readouterr()
    assert main(["lines", "list"]) == 0
    assert capsys.readouterr().out == (
        "Unfinished setups\n"
        "  old          +442079460002  bought · pending           "
        "undo: ai-phone lines remove old --key-env <NAME>\n"
    )
    old.set_state("rollback_failed")  # and the carrier-portal guidance is still reachable for them
    assert main(["lines", "list"]) == 0
    assert capsys.readouterr().out == (
        "Unfinished setups\n"
        "  old          +442079460002  bought · rollback_failed   "
        "finish in the carrier portal, then: ai-phone lines remove old --local-only\n"
    )


def test_list_without_lines_or_journals_still_suggests_adding_one(home: Path, capsys):
    assert main(["lines", "list"]) == 0
    assert capsys.readouterr().out == "no lines configured; try: ai-phone lines add telnyx --key-env TELNYX_API_KEY\n"


def test_remove_refuses_to_release_a_number_another_line_uses(home: Path, monkeypatch, capsys):
    monkeypatch.setenv("AI_PHONE_FAKE_CRASH_AFTER", "number")
    assert main(["lines", "add", "fake", "--key-env", "FAKE_KEY", "--id", "shop", "--yes"]) == 3  # bought, then stopped
    monkeypatch.delenv("AI_PHONE_FAKE_CRASH_AFTER")
    assert main(["lines", "add", "trunk", "--id", "main", "--number", "+442079460000", "--registrar", "sip:sip.example.com",
                 "--user", "main1234", "--password-env", "MAIN_PW"]) == 0  # the same number, typed in by hand
    capsys.readouterr()
    assert main(["lines", "remove", "shop", "--key-env", "FAKE_KEY", "--yes"]) == 3
    out = capsys.readouterr().out
    assert ("error: +442079460000 is also line main, so releasing line shop would delete the number that line uses. "
            "Remove line main first, or run `ai-phone lines remove shop --local-only` to remove only line shop "
            "and keep its carrier resources.") in out
    assert "number" in _fake_carrier(home)["carrier"] and (home / "lines" / "shop.journal.json").exists()

    assert main(["lines", "remove", "main"]) == 0
    other = Journal.open(home / "lines", "other", "fake")  # an unfinished setup elsewhere that names the number
    other.start("target", phone_number="+442079460000", brought=True)
    capsys.readouterr()
    assert main(["lines", "remove", "shop", "--key-env", "FAKE_KEY", "--yes"]) == 3
    assert "error: +442079460000 is also claimed by the unfinished setup of line other" in capsys.readouterr().out
    assert "number" in _fake_carrier(home)["carrier"]
    assert main(["lines", "remove", "shop", "--local-only"]) == 0  # keeping everything is still allowed
    other.delete()


def test_remove_releases_a_bought_number_nobody_else_uses(home: Path, monkeypatch, capsys):
    monkeypatch.setenv("AI_PHONE_FAKE_CRASH_AFTER", "number")
    assert main(["lines", "add", "fake", "--key-env", "FAKE_KEY", "--id", "shop", "--yes"]) == 3
    monkeypatch.delenv("AI_PHONE_FAKE_CRASH_AFTER")
    assert main(["lines", "add", "sip", "--id", "agent", "--user", "agent", "--host", "192.0.2.10"]) == 0
    assert main(["lines", "remove", "shop", "--key-env", "FAKE_KEY", "--yes"]) == 0
    assert "number" not in _fake_carrier(home)["carrier"]


def test_an_old_format_brought_number_is_never_listed_to_release(home: Path, capsys):
    from ai_phone.cli_lines import _carrier_ids
    assert _carrier_ids({"number_id": "owned-1", "number_brought": True, "connection_id": "c1"}) == ({"connection_id": "c1"}, "owned-1")
    assert _carrier_ids({"number_sid": "PN2", "brought_number_sid": "PN9", "number_brought": True}) == ({"number_sid": "PN2"}, "PN9")


def test_signalwire_purchase_requires_portal_before_any_provider_access(home, monkeypatch, capsys):
    monkeypatch.setattr("ai_phone.cli_lines._provider", lambda *args: pytest.fail("purchase must not construct a provider"))
    assert main(["lines", "add", "signalwire", "--id", "main", "--yes"]) == 3
    assert "Buy a number in your SignalWire Space" in capsys.readouterr().out
    assert not load_config(home / "config.yaml").lines and not (home / "secrets.env").exists()


def test_signalwire_manual_preset_is_numberless_encrypted_and_lists_the_account(home, capsys):
    assert main(["lines", "add", "trunk", "--preset", "signalwire", "--id", "main",
                 "--registrar", "sip:example.sip.signalwire.com:5061", "--user", "account1",
                 "--password-env", "SIP_PASSWORD"]) == 0
    line = load_config(home / "config.yaml").lines["main"]
    assert line.from_user == "account" and line.number is None and line.caller_id is None
    assert line.tls_hostname_policy == "verified_wildcard"
    assert line.registrar.endswith(";transport=tls")
    assert line.transport == "tls" and line.srtp == "required" and line.outbound_proxy == line.registrar
    assert line.accept_invites_from == "registered_trunk" and line.trusted_cidrs == []
    assert main(["lines", "list"]) == 0
    assert "sip:account1@example.sip.signalwire.com:5061" in capsys.readouterr().out


@pytest.mark.parametrize("provider", ["sip", "twilio", "telnyx", "signalwire"])
def test_signalwire_preset_rejects_nonmanual_lines_without_api_access(home, monkeypatch, provider):
    monkeypatch.setattr("ai_phone.cli_lines._provider", lambda *args: pytest.fail("must not construct a provider"))
    assert main(["lines", "add", provider, "--preset", "signalwire"]) == 3


def test_signalwire_add_resume_remove_keep_space_and_original_number_id(home, monkeypatch):
    from ai_phone.providers.base import Number
    from ai_phone.providers.fake import FakeProvider, OWNED
    from ai_phone.lines.config_writer import read_raw
    source = "12345678-1234-4123-8123-123456789012"
    seen = []
    attaches = []

    class SourceSensitiveProvider(FakeProvider):
        name = "signalwire"

        def owned_numbers(self):
            return [Number(OWNED, None, None, False, source)]

        def attach(self, journal, number, line_id, user, password):
            assert number.source_id == source, "resume must retain the carrier's ID rather than substitute E.164"
            attaches.append(number.source_id)
            return super().attach(journal, number, line_id, user, password)

    def loader(name, **kwargs):
        seen.append((name, kwargs))
        return SourceSensitiveProvider()

    monkeypatch.setattr("ai_phone.providers.load_provider", loader)
    monkeypatch.setenv("PROJECT_ID", "project-test")
    monkeypatch.setenv("API_TOKEN", "token-test")
    context = ["--key-env", "PROJECT_ID", "--secret-env", "API_TOKEN", "--space", "example.signalwire.com"]
    assert main(["lines", "add", "signalwire", "--existing", OWNED, "--id", "main", "--yes", *context]) == 0
    assert read_raw(home / "config.yaml")["lines"]["main"]["provider"]["name"] == "signalwire"
    assert main(["lines", "resume", "main", *context]) == 0
    assert main(["lines", "remove", "main", "--yes", *context]) == 0
    assert attaches == [source, source]
    assert seen == [("signalwire", {"key": "project-test", "secret": "token-test", "space": "example.signalwire.com"})] * 3
    assert not load_config(home / "config.yaml").lines


def test_set_caller_id_uses_only_line_eligible_numbers_without_carrier_access(home, monkeypatch):
    from ai_phone.lines.config_writer import read_raw, update_line
    main(["lines", "add", "fake", "--key-env", "FAKE_KEY", "--id", "main", "--yes"])
    monkeypatch.setattr("ai_phone.cli_lines._provider", lambda *args: pytest.fail("number From mode changes locally"))
    assert main(["lines", "set-caller-id", "main", "+15555550102"]) == 3
    update_line(home / "config.yaml", "main", verified_caller_ids=[
        {"number": "+15555550102", "provider": "fake", "verified_at": "2026-10-01T12:00:00Z"}])
    assert main(["lines", "set-caller-id", "main", "+15555550102"]) == 0
    assert read_raw(home / "config.yaml")["lines"]["main"]["caller_id"] == "+15555550102"
    assert main(["lines", "set-caller-id", "main", "+442079460000"]) == 0


def test_set_caller_id_without_verification_history_preserves_its_absence(home):
    from ai_phone.lines.config_writer import read_raw
    assert main(["lines", "add", "trunk", "--id", "main", "--number", "+15555550100", "--user", "account1",
                 "--registrar", "sip:trunk.example.org", "--password-env", "SIP_PASSWORD"]) == 0
    assert main(["lines", "set-caller-id", "main", "+15555550100"]) == 0
    assert "verified_caller_ids" not in read_raw(home / "config.yaml")["lines"]["main"]


@pytest.mark.parametrize("saved_policy", [None, "strict", "verified_wildcard"])
def test_provider_resume_preserves_existing_tls_policy(home, saved_policy):
    from ai_phone.cli_lines import _commit
    from ai_phone.lines.config_writer import read_raw, write_raw
    from ai_phone.providers.base import ProvisionResult
    line = {"kind": "trunk", "user": "synthetic", "from_user": "account", "transport": "tls",
            "dial_domain": "tenant.carrier.test:5061",
            "registrar": "sip:tenant.carrier.test:5061;transport=tls",
            "outbound_proxy": "sip:tenant.carrier.test:5061;transport=tls"}
    previous = dict(line)
    if saved_policy is not None:
        previous["tls_hostname_policy"] = saved_policy
    write_raw(home / "config.yaml", {"lines": {"main": previous}})
    line["tls_hostname_policy"] = "verified_wildcard"
    journal = Journal.open(home / "lines", "main", "signalwire")
    messages = []
    assert _commit(journal, home / "config.yaml", "main", ProvisionResult("ready", "ready", line),
                   "SYNTHETIC_PASSWORD", say=messages.append) == 0, messages
    actual = read_raw(home / "config.yaml")["lines"]["main"]
    assert actual.get("tls_hostname_policy") == saved_policy
    if saved_policy is None:
        assert "tls_hostname_policy" not in actual


@pytest.mark.parametrize("transport", ["tcp", "udp"])
def test_non_tls_signalwire_manual_preset_does_not_enable_wildcard_policy(home, transport):
    assert main(["lines", "add", "trunk", "--preset", "signalwire", "--id", "main",
                 "--registrar", "sip:tenant.carrier.test", "--user", "synthetic",
                 "--password-env", "SYNTHETIC_PASSWORD", "--transport", transport]) == 0
    line = load_config(home / "config.yaml").lines["main"]
    assert line.transport == transport and line.tls_hostname_policy == "strict"


def test_tls_preset_does_not_rewrite_explicitly_conflicting_proxy_transport(home):
    assert main(["lines", "add", "trunk", "--preset", "signalwire", "--id", "main",
                 "--registrar", "sip:tenant.carrier.test;transport=tcp", "--user", "synthetic",
                 "--password-env", "SYNTHETIC_PASSWORD"]) == 3
    assert not load_config(home / "config.yaml").lines

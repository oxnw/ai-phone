"""`ai-phone lines …`: buy, wire, resume, release, list, divert and verify. Carrier calls happen here, never in the daemon."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import NamedTuple

from ai_phone import paths
from ai_phone.carriers import CARRIERS, managed_caller_id
from ai_phone.lines.caller_id import apply_caller_id, caller_id_journal, caller_id_line
from ai_phone.config import E164_RE, LINE_ID_RE, SIP_URI_RE, ConfigError, host_of, line_password_env, load_config
from ai_phone.lines.config_writer import add_line, read_raw, remove_line, update_line, write_raw
from ai_phone.lines.credentials import generate_sip_password, generate_sip_user
from ai_phone.lines.forward import WHEN, divert_codes
from ai_phone.lines.journal import Journal, line_lock
from ai_phone.secrets import SecretsError, load_secrets, remove_secret, write_secret

EXIT_OK, EXIT_PENDING, EXIT_ERROR = 0, 2, 3


def _say(msg: str) -> None:
    print(msg, flush=True)


def _ask(prompt: str, yes: bool) -> bool:
    if yes:
        return True
    return input(f"{prompt} [y/N] ").strip().lower() == "y"


def _env(name: str | None, what: str) -> str | None:
    if not name:
        return None
    v = os.environ.get(name) or load_secrets(paths.secrets_path()).get(name)
    if not v:
        raise SystemExit(f"{what}: {name} is not set in the environment or in {paths.secrets_path()}")
    return v


PROVIDER_CHOICES = tuple(CARRIERS)  # mirrors ai_phone.providers.PROVIDERS without importing it at CLI start


def _provider(a):
    from ai_phone.providers import load_provider  # here, not at import: `ai-phone run` never loads the providers package
    kw = {}
    if a.provider == "signalwire":
        kw["space"] = getattr(a, "space", None) or _env("SIGNALWIRE_SPACE", "--space")
    elif getattr(a, "space", None):
        _fail("--space only applies to SignalWire")
    if a.provider == "fake":  # test hooks for the in-memory carrier
        kw["pending_polls"] = int(os.environ.get("AI_PHONE_FAKE_PENDING", "0"))
        kw["crash_after"] = os.environ.get("AI_PHONE_FAKE_CRASH_AFTER") or None
    return load_provider(a.provider, key=_env(a.key_env, "--key-env") or "", secret=_env(getattr(a, "secret_env", None), "--secret-env"), **kw)


def _config_path(a) -> Path:
    return Path(a.config).expanduser() if getattr(a, "config", None) else paths.config_path()


def next_line_id(raw: dict, base: str) -> str:
    lines = raw.get("lines") or {}
    if base not in lines:
        return base
    n = 2
    while f"{base}-{n}" in lines:
        n += 1
    return f"{base}-{n}"


_next_id = next_line_id


# ---- add <provider> ----------------------------------------------------------------------------

def _pick_number(provider, a):
    if a.number:
        return next((n for n in provider.search_numbers(a.country, None, 50) if n.e164 == a.number), None) or \
            _fail(f"{a.number} is not available from {provider.name}")
    nums = provider.search_numbers(a.country, a.area, 10)
    if not nums:
        _fail("no numbers found; try another --area or --country")
    for i, n in enumerate(nums, 1):
        cost = f"{n.monthly_cost:.2f} {n.currency}/month" if n.monthly_cost is not None else "price: see portal"
        _say(f"{i:2d}. {n.e164}   {cost}{'   (regulatory documents required)' if n.requires_regulatory else ''}")
    if a.yes:
        return nums[0]
    choice = input("Which one? [1] ").strip() or "1"
    if not choice.isdigit() or not 1 <= int(choice) <= len(nums):
        _fail(f"answer with a number from 1 to {len(nums)} (or q to stop)" if choice.lower() not in ("q", "n", "no") else "stopped; nothing was bought")
    return nums[int(choice) - 1]


def _fail(msg: str):
    _say(f"error: {msg}")
    raise SystemExit(EXIT_ERROR)


def provision_flow(provider, journal: Journal, config_path: Path, line_id: str, number, existing: bool = False, say=_say) -> int:
    """Everything after the person said yes: the line's SIP password, then `provision` (buy and wire the number) or,
    for a number already on the account, `attach` (wire it only), then the config. Every message goes through `say`."""
    refusal = _refusal(journal, config_path, line_id, number, existing)
    if refusal:
        if not journal.doc["steps"]:
            journal.delete()  # the empty journal Journal.open() wrote: a refusal leaves nothing behind
        say(f"error: {refusal}")
        return EXIT_ERROR
    # first thing in the journal, so resume always knows the number and whether it was bought or brought
    journal.start("target", phone_number=number.e164, brought=existing, source_id=number.source_id)
    journal.finish("target")
    env_name = line_password_env(line_id)
    pw_step, planned = journal.start("credentials", user=generate_sip_user(line_id), password_env=env_name)
    if pw_step == "new":
        write_secret(env_name, generate_sip_password(), paths.secrets_path())
        journal.finish("credentials", **planned)
    creds = journal.resources() if pw_step == "done" else planned
    password = load_secrets(paths.secrets_path())[env_name]
    from ai_phone.providers.base import ProviderError
    run = provider.attach if existing else provider.provision
    try:
        result = run(journal, number, line_id, creds["user"], password)
    except ProviderError as e:
        say(f"error: {e}\nThe journal is kept at {journal.path}; run `ai-phone lines resume {line_id}` to continue or `ai-phone lines remove {line_id}` to clean up.")
        return EXIT_ERROR
    return _commit(journal, config_path, line_id, result, env_name, say=say)


def _refusal(journal: Journal, config_path: Path, line_id: str, number, existing: bool) -> str | None:
    """Why this request must not run: the number is already another line (two lines would fight over its calls,
    and removing one would release or re-route the other's number), or the journal holds an unfinished setup for
    something else (one journal for two numbers lets `release` mix a bought number up with a brought one)."""
    for lid, ln in (read_raw(config_path).get("lines") or {}).items():
        if lid != line_id and (ln or {}).get("number") == number.e164:
            return f"{number.e164} is already line {lid}"
    for lid, setup in journal_setups().items():  # an earlier attempt under another id: that id finishes or undoes it
        if lid != line_id and setup.number == number.e164:
            return (f"line {lid} has an unfinished setup for {number.e164}; run `ai-phone lines resume {lid}` to finish it "
                    f"or `ai-phone lines remove {lid}` to undo it, then try again")
    if journal.doc["steps"]:
        target = next((s["planned"] for s in journal.doc["steps"] if s["name"] == "target"), None)
        if target is None or target.get("phone_number") != number.e164 or bool(target.get("brought")) != existing:
            before = _journal_number(journal) or "another number"
            return (f"line {line_id} has an unfinished setup for {before}; run `ai-phone lines resume {line_id}` to finish it "
                    f"or `ai-phone lines remove {line_id}` to undo it, then try again")
    return None


def _commit(journal: Journal, config_path: Path, line_id: str, result, env_name: str, say=_say) -> int:
    if result.state == "failed":
        raw = read_raw(config_path)
        existing = (raw.get("lines") or {}).get(line_id)
        if existing is not None:  # a pending line whose order the carrier finally rejected: say so in the config too
            provider = dict(existing.get("provider") or {"name": journal.doc["provider"], "resources": {}})
            provider["state"] = "failed"
            update_line(config_path, line_id, provider=provider)
        say(f"error: {result.message}\nRun `ai-phone lines remove {line_id} --key-env <NAME>` to release what was created.")
        return EXIT_ERROR
    if not result.line:  # nothing loadable to write yet; the journal keeps the state
        say(f"{result.message}\nResume with: ai-phone lines resume {line_id} --key-env <NAME>")
        return EXIT_PENDING
    if result.state == "ready":
        journal.start("carrier_ready")
        journal.finish("carrier_ready")
    line = dict(result.line)
    line.update({"password_env": env_name, "accept_invites_from": "registered_trunk"})
    line["provider"] = dict(line.get("provider") or {"name": journal.doc["provider"], "resources": {}})
    line["provider"]["state"] = result.state
    raw = read_raw(config_path)
    previous = (raw.get("lines") or {}).get(line_id)
    if previous is not None:
        # Resuming setup must not silently change an existing line's trust policy.
        line.pop("tls_hostname_policy", None)
        if "tls_hostname_policy" in previous:
            line["tls_hostname_policy"] = previous["tls_hostname_policy"]
        update_line(config_path, line_id, **line)
    else:
        add_line(config_path, line_id, line)
    try:
        load_config(config_path)  # what we wrote must load, pending or ready
    except ConfigError as e:  # never leave an unloadable config behind: restore what was there, keep the journal
        if previous is not None:
            raw2 = read_raw(config_path)
            raw2["lines"][line_id] = previous
            write_raw(config_path, raw2)
        else:
            remove_line(config_path, line_id)
        say(f"error: the provisioned line does not validate ({e}); the config was restored. The carrier resources exist: "
             f"fix the cause, then `ai-phone lines resume {line_id}`, or `ai-phone lines remove {line_id}` to release them.")
        return EXIT_ERROR
    if result.state == "ready":
        journal.start("local_committed")
        journal.finish("local_committed")  # the step's own timestamp records when; no result keys, so resources() stays carrier ids
    if result.state == "pending":
        say(f"{result.message}\nLine {line_id} is written but pending. Resume with: ai-phone lines resume {line_id} --key-env <NAME>")
        return EXIT_PENDING
    number = line.get("number", line_id)
    if paths.socket_path().exists():
        say(f"Line {line_id} is ready: {number}. Restart ai-phone to bring it live.")
    else:
        say(f"Line {line_id} is ready: {number}. Run `ai-phone run`; it will register.")
    return EXIT_OK


def cmd_add(a) -> int:
    config_path = _config_path(a)
    if getattr(a, "preset", None):
        if a.provider != "trunk":
            _fail("--preset only applies to lines add trunk")
        if a.from_user not in (None, "account"):
            _fail("the SignalWire preset uses --from-user account")
        a.from_user = "account"
    if a.from_user is not None and a.provider != "trunk":
        _fail("--from-user only applies to a trunk added by hand: lines add trunk")
    if a.provider in ("trunk", "sip"):
        if a.existing:
            _fail("--existing works with a carrier: telnyx, twilio or signalwire")
        return _add_manual(a, config_path)
    if a.existing:
        if a.number or a.area:
            _fail("--existing cannot be combined with --number or --area")
        if not E164_RE.fullmatch(a.existing):
            _fail("--existing must be E.164, like +15550100000")
    if not a.existing and not CARRIERS[a.provider].purchase:
        _fail("SignalWire number purchasing is not available here. Buy a number in your SignalWire Space, "
              "then connect it with --existing +15555550100.")
    provider = _provider(a)
    raw = read_raw(config_path)
    line_id = a.id or next_line_id(raw, a.provider)
    if not LINE_ID_RE.fullmatch(line_id):
        _fail(f"--id must match {LINE_ID_RE.pattern}")
    if line_id in (raw.get("lines") or {}):
        _say(f"error: line {line_id!r} already exists (use --id to pick another name)")
        return EXIT_ERROR
    # the config the finished line is written into must load now, before a single carrier request spends money
    if config_path.exists():
        load_config(config_path)
    if a.existing:
        number = next((n for n in provider.owned_numbers() if n.e164 == a.existing), None) or \
            _fail(f"{a.existing} is not on your {provider.name} account")
        if not _ask(f"Connect {number.e164} on your {provider.name} account to ai-phone? Its incoming calls will ring ai-phone. "
                    "Nothing is bought; `ai-phone lines remove` disconnects it and keeps the number.", a.yes):
            return EXIT_ERROR
        journal = Journal.open(paths.lines_dir(), line_id, provider.name)
        return provision_flow(provider, journal, config_path, line_id, number, existing=True)
    number = _pick_number(provider, a)
    cost = f"{number.monthly_cost:.2f} {number.currency}/month" if number.monthly_cost is not None else "the carrier's monthly price"
    if not _ask(f"Buy {number.e164} for {cost} on your {provider.name} account?", a.yes):
        return EXIT_ERROR
    journal = Journal.open(paths.lines_dir(), line_id, provider.name)
    return provision_flow(provider, journal, config_path, line_id, number)


def manual_line(kind: str, *, number: str | None = None, registrar: str | None = None, user: str | None = None,
                password_env: str | None = None, transport: str = "udp", caller_id: str | None = None, dial_domain: str | None = None,
                outbound_proxy: str | None = None, trusted_cidrs=(), host: str | None = None, trusted=(),
                from_user: str = "number", srtp: str = "off", tls_hostname_policy: str = "strict") -> dict:
    """The config entry for a line typed in by hand: a SIP trunk (`trunk`) or a SIP address (`sip`).
    Raises ValueError for what is missing; `write_line` checks everything else by loading the config."""
    if tls_hostname_policy not in ("strict", "verified_wildcard"):
        raise ValueError("tls_hostname_policy must be strict or verified_wildcard")
    if tls_hostname_policy != "strict":
        if kind != "trunk" or transport != "tls":
            raise ValueError("verified_wildcard requires a TLS trunk line")
        # The TLS preset accepts the usual bare registrar input, but saves an
        # explicit TLS next hop. An explicitly conflicting transport still fails validation.
        if registrar and SIP_URI_RE.fullmatch(registrar) and ";transport=" not in registrar:
            registrar += ";transport=tls"
        if outbound_proxy and SIP_URI_RE.fullmatch(outbound_proxy) and ";transport=" not in outbound_proxy:
            outbound_proxy += ";transport=tls"
    if from_user not in ("number", "account"):
        raise ValueError("from_user must be number or account")
    if from_user == "account" and kind != "trunk":
        raise ValueError("from_user account only applies to trunk lines")
    if kind == "trunk":
        if (number is not None or from_user == "number") and not E164_RE.fullmatch(number or ""):
            raise ValueError("the number must be E.164, like +442079460000")
        if not registrar or not user or not password_env:
            raise ValueError("a trunk line needs a registrar, a user and a password")
        # host_of needs a well-formed URI; a malformed registrar is left for the config check to reject
        line = {"kind": "trunk", "number": number, "caller_id": caller_id or (number if from_user == "number" else None), "registrar": registrar,
                "dial_domain": dial_domain or (host_of(registrar) if ":" in registrar else registrar), "user": user,
                "password_env": password_env, "transport": transport, "accept_invites_from": "registered_trunk",
                "trusted_cidrs": list(trusted_cidrs or [])}
        if tls_hostname_policy != "strict":
            line["tls_hostname_policy"] = tls_hostname_policy
        if srtp != "off":
            line["srtp"] = srtp
        if from_user == "account":
            line["from_user"] = from_user
        if outbound_proxy:
            line["outbound_proxy"] = outbound_proxy
        return line
    if kind == "sip":
        if not user or not host:
            raise ValueError("a sip line needs a user and a host")
        return {"kind": "sip", "user": user, "aor": f"sip:{user}@{host}", "caller_id": f"sip:{user}@{host}",
                "display_name": user, "transport": transport,
                "accept_invites_from": "trusted" if trusted else "lan", "trusted_peers": list(trusted or [])}
    raise ValueError(f"a line added by hand is a trunk or a sip line, not {kind!r}")


def write_line(config_path: Path, line_id: str, line: dict) -> None:
    """Add a line and prove the config still loads; if it does not, remove the line again and re-raise the ConfigError.
    A bad or taken line id is a ValueError."""
    if not LINE_ID_RE.fullmatch(line_id):
        raise ValueError(f"line ids must match {LINE_ID_RE.pattern}")
    with line_lock(paths.lines_dir()):
        add_line(config_path, line_id, line)
        try:
            load_config(config_path)
        except ConfigError:
            remove_line(config_path, line_id)  # only undo what this call wrote
            raise


def _add_manual(a, config_path: Path) -> int:
    if not LINE_ID_RE.fullmatch(a.id or ""):
        _fail(f"--id is required and must match {LINE_ID_RE.pattern}")
    raw = read_raw(config_path)
    if a.id in (raw.get("lines") or {}):
        _fail(f"line {a.id!r} already exists; pick another --id")
    if a.provider == "trunk":
        if (a.number is not None or a.from_user != "account") and not E164_RE.fullmatch(a.number or ""):
            _fail("--number must be E.164, like +442079460000")
        if not a.registrar or not a.user or not a.password_env:
            _fail("--registrar, --user and --password-env are required for a trunk line")
    elif not a.user or not a.host:
        _fail("--user and --host are required for a sip line")
    try:
        write_line(config_path, a.id, manual_line(
            a.provider, number=a.number, registrar=a.registrar, user=a.user, password_env=a.password_env, transport=a.transport or ("tls" if a.preset else "udp"),
            caller_id=a.caller_id, dial_domain=a.dial_domain, outbound_proxy=a.outbound_proxy or (a.registrar if a.preset else None),
            trusted_cidrs=a.trusted_cidr or (), host=a.host, trusted=a.trusted or (), from_user=a.from_user or "number",
            srtp=a.srtp or ("required" if a.preset and a.transport in (None, "tls") else "off"),
            tls_hostname_policy="verified_wildcard" if a.preset == "signalwire" and a.transport in (None, "tls") else "strict"))
    except ValueError as e:  # a ConfigError is a ValueError too
        _fail(str(e))
    _say(f"Line {a.id} added. Restart ai-phone (or start it) to bring it up.")
    return EXIT_OK


# ---- resume / remove / list ------------------------------------------------------------------

def _journal_for(line_id: str, a):
    raw = read_raw(_config_path(a))
    ln = (raw.get("lines") or {}).get(line_id) or {}
    pname = (ln.get("provider") or {}).get("name")
    jpath = paths.lines_dir() / f"{line_id}.journal.json"
    if pname is None and jpath.exists():
        pname = Journal.open(paths.lines_dir(), line_id).doc["provider"]
    return pname, ln


def _planned(journal: Journal, step: str, key: str):
    return next((s["planned"].get(key) for s in journal.doc["steps"] if s["name"] == step), None)


def _journal_number(journal: Journal) -> str | None:
    return (_planned(journal, "target", "phone_number")                      # journals from this version
            or _planned(journal, "order", "phone_number")                    # older Telnyx journals
            or _planned(journal, "number", "phone_number")                   # older Twilio journals
            or _planned(journal, "number", "name"))                          # older fake-carrier journals


class JournalSetup(NamedTuple):
    """What a provisioning journal records."""
    number: str | None   # None: nothing yet (an empty journal); "?": it cannot be read, or names no number
    brought: bool        # the number was already on the account (connected, not bought)
    state: str
    resumable: bool      # it has the `target` step provisioning needs to pick it up
    provider: str | None = None


def journal_setups(d: Path | None = None) -> dict[str, JournalSetup]:
    """Every provisioning journal under lines/, by line id. Read only: Journal.open() would write the file."""
    d = Path(d) if d else paths.lines_dir()
    if not d.is_dir():
        return {}
    return {p.name.removesuffix(".journal.json"): _journal_setup(p) for p in sorted(d.glob("*.journal.json"))}


def _journal_setup(path: Path) -> JournalSetup:
    try:
        if path.is_symlink():
            raise ValueError("a symlink")
        journal = Journal(path, json.loads(path.read_text()))
        steps = journal.doc["steps"]
        number = (_journal_number(journal) or "?") if steps else None
        return JournalSetup(number, bool(_planned(journal, "target", "brought")), str(journal.doc.get("state")),
                            any(s["name"] == "target" for s in steps),
                            journal.doc.get("provider") if isinstance(journal.doc.get("provider"), str) else None)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return JournalSetup("?", False, "?", False)


def _number_in_use(journal: Journal, line_id: str, raw: dict) -> tuple[str, str] | None:
    """(why, other line) when the number this journal would release belongs to another line or another setup."""
    number = _journal_number(journal)
    if not number or _planned(journal, "target", "brought"):  # a brought number is only disconnected, never deleted
        return None
    for lid, ln in (raw.get("lines") or {}).items():
        if lid != line_id and isinstance(ln, dict) and ln.get("number") == number:
            return f"{number} is also line {lid}", lid
    for lid, setup in journal_setups().items():
        if lid != line_id and setup.number == number:
            return f"{number} is also claimed by the unfinished setup of line {lid}", lid
    return None


def cmd_resume(a) -> int:
    """Re-runs provisioning from the journal: finished steps are skipped, half-done ones reconciled, a pending order polled."""
    pname, ln = _journal_for(a.line_id, a)
    if not pname:
        _fail(f"line {a.line_id} has no provider journal to resume")
    a.provider = pname
    provider = _provider(a)
    journal = Journal.open(paths.lines_dir(), a.line_id, pname)
    if managed_caller_id(ln):
        provider.validate_context(journal)
        pending = provider.pending_caller_id(journal)
        if pending:
            apply_caller_id(_config_path(a), a.line_id, pending, provider=provider, resume=True)
            _say(f"Line {a.line_id} calls out as {pending}. Restart ai-phone to update its local settings.")
            return EXIT_OK
    e164 = _journal_number(journal)
    user = _planned(journal, "credentials", "user")
    env_name = ln.get("password_env") or _planned(journal, "credentials", "password_env") or line_password_env(a.line_id)
    if not e164 or not user:
        _fail(f"the journal for {a.line_id} has no number or credentials yet; run `ai-phone lines add` again or `lines remove --local-only`")
    password = load_secrets(paths.secrets_path()).get(env_name)
    if not password:
        _fail(f"{env_name} is missing from {paths.secrets_path()}; run `ai-phone lines remove {a.line_id} --local-only` and add the line again")
    from ai_phone.providers.base import Number, ProviderError
    run = provider.attach if _planned(journal, "target", "brought") else provider.provision
    try:
        source_id = _planned(journal, "target", "source_id") or e164
        result = run(journal, Number(e164, None, None, False, source_id), a.line_id, user, password)
    except ProviderError as e:
        _say(f"error: {e}")
        return EXIT_ERROR
    return _commit(journal, _config_path(a), a.line_id, result, env_name)


def _recorded_resources(ln: dict) -> dict:
    """The carrier ids `_commit` copied into the config; the journal is authoritative, this is what outlives losing it."""
    return dict((ln.get("provider") or {}).get("resources") or {})


BROUGHT_KEYS = ("brought_number_sid", "brought_number_id")  # a number the person already had: kept, never released


def _brought_keys(res: dict) -> dict:
    """An older journal (e4a8c67) flagged `number_brought` and kept the brought number under number_sid/number_id."""
    res = dict(res)
    if res.get("number_brought") and not any(res.get(k) for k in BROUGHT_KEYS):
        for key, brought in (("number_sid", "brought_number_sid"), ("number_id", "brought_number_id")):
            if res.get(key):
                res[brought] = res.pop(key)
    return res


def _carrier_ids(res: dict) -> tuple[dict, str | None]:
    """The ids of what ai-phone created, and the id of a brought number. Journal flags and `previous_*` values are
    never shown: they are not ids, and a previous voice URL can carry credentials. A shared profile is the account's."""
    res = _brought_keys(res)
    ids = {k: v for k, v in res.items() if isinstance(v, str) and v and k.endswith(("_id", "_sid"))
           and not k.startswith("previous_") and k not in BROUGHT_KEYS}
    if res.get("profile_shared"):
        ids.pop("outbound_voice_profile_id", None)
    return ids, next((res[k] for k in BROUGHT_KEYS if res.get(k)), None)


def _ids(res: dict, ln: dict, line_id: str) -> str:
    return ", ".join(f"{k}={v}" for k, v in res.items()) or f"no ids were kept here; find them by {ln.get('number') or line_id}"


def _brought_note(journal: Journal | None, ln: dict, brought_id: str | None, pname: str, line_id: str) -> str | None:
    if not brought_id and not (journal and _planned(journal, "target", "brought")):
        return None
    number = ln.get("number") or (journal and _planned(journal, "target", "phone_number")) or line_id
    return f"{number} stays on your {pname} account; reset its voice URL / connection in the portal if you want its calls back."


def cmd_remove(a) -> int:
    config_path = _config_path(a)
    pname, ln = _journal_for(a.line_id, a)
    if not ln and not pname:
        _fail(f"no line {a.line_id}")
    jpath = paths.lines_dir() / f"{a.line_id}.journal.json"
    if pname and not a.local_only:
        # A release with no resource ids reports success without deleting anything, which would drop the local record
        # while the carrier keeps billing. An empty journal is as good as none: Journal.open() writes the one it
        # invents, so a failed `lines resume` leaves a stepless file behind, and provisioning's first act is a step.
        journal = Journal.open(paths.lines_dir(), a.line_id, pname) if jpath.exists() else None
        journalled = journal.resources() if journal and journal.doc["steps"] else None
        stranded = {k: v for k, v in _recorded_resources(ln).items() if k not in (journalled or {})}
        if journalled is None or stranded:
            why = "is missing" if journal is None else "records no carrier steps" if journalled is None else "is incomplete"
            ids, brought = _carrier_ids(stranded)
            note = _brought_note(journal, ln, brought, pname, a.line_id)
            _say(f"error: the provisioning journal for {a.line_id} {why} ({jpath}), "
                 f"so releasing here would leave carrier resources live.\n"
                 f"Release them on your {pname} account first ({_ids(ids, ln, a.line_id)}), "
                 f"then run `ai-phone lines remove {a.line_id} --local-only`." + (f"\n{note}" if note else ""))
            return EXIT_ERROR
        clash = _number_in_use(journal, a.line_id, read_raw(config_path))
        if clash:
            why, other = clash
            _say(f"error: {why}, so releasing line {a.line_id} would delete the number that line uses. Remove line {other} "
                 f"first, or run `ai-phone lines remove {a.line_id} --local-only` to remove only line {a.line_id} and keep "
                 f"its carrier resources.")
            return EXIT_ERROR
        if not a.key_env:
            _say(f"line {a.line_id} was provisioned on {pname}. Pass --key-env and the carrier context (--secret-env for Twilio/SignalWire; --space for SignalWire) to release its carrier resources, "
                 f"or --local-only to remove it from this machine and keep them.")
            return EXIT_ERROR
        a.provider = pname
        provider = _provider(a)
        number = ln.get("number") or _planned(journal, "target", "phone_number") or a.line_id
        if _planned(journal, "target", "brought"):
            question = (f"Disconnect {number} from ai-phone and delete what ai-phone created on your {pname} account? "
                        "The number stays on your account.")
        else:
            question = f"Release {number} from your {pname} account? This cannot be undone."
        if not _ask(question, a.yes):
            return EXIT_ERROR
        res = provider.release(journal)
        if not res.ok:
            _say("error: some resources could not be released; finish in the carrier portal:\n" + "\n".join(f"  {k}: {v}" for k, v in res.leftover.items()))
            update_line(config_path, a.line_id, provider={**(ln.get("provider") or {}), "state": "rollback_failed"})
            return EXIT_ERROR
        journal.delete()
    elif pname and a.local_only:
        journal = Journal.open(paths.lines_dir(), a.line_id, pname) if jpath.exists() else None
        ids, brought = _carrier_ids(journal.resources() if journal else {})
        if not ids and not brought:
            ids, brought = _carrier_ids(_recorded_resources(ln))
        _say("Kept in your carrier account (release them in the portal): " + _ids(ids, ln, a.line_id))
        note = _brought_note(journal, ln, brought, pname, a.line_id)
        if note:
            _say(note)
        jpath.unlink(missing_ok=True)
    remove_line(config_path, a.line_id)
    env_name = ln.get("password_env") or line_password_env(a.line_id)
    if env_name.startswith("AI_PHONE_LINE_"):
        remove_secret(env_name, paths.secrets_path())
    _say(f"Line {a.line_id} removed. Restart ai-phone if it is running.")
    return EXIT_OK


def _unfinished_row(line_id: str, setup: JournalSetup) -> str:
    """One row of the unfinished block: how far the journal got, and the command that finishes or undoes it."""
    local_only = f"ai-phone lines remove {line_id} --local-only"
    if setup.state == "?":  # the file could not be read: claim nothing about what it holds
        return f"  {line_id:12s} unreadable · remove: {local_only}"
    if setup.number is None or setup.number == "?":
        # No number under any key this or an older version wrote, so nothing was bought and there is nothing to
        # resume. (`target` is the first step provision_flow writes, before any carrier call.)
        return f"  {line_id:12s} nothing created yet · remove: {local_only}"
    where = setup.number
    got = f"{'brought' if setup.brought else 'bought'} · {setup.state}"
    undo = f"undo: ai-phone lines remove {line_id} --key-env <NAME>"
    if setup.state == "rollback_failed":  # the carrier refused the undo, so only a person can clear it
        how = f"finish in the carrier portal, then: {local_only}"
    elif not setup.resumable:
        # a journal from an older version: it names a bought number but has no `target` step for `resume` to
        # pick up. The number is still billing, so the way out is the undo, never --local-only.
        how = undo
    else:
        how = f"resume: ai-phone lines resume {line_id} --key-env <NAME>   {undo}"
    return f"  {line_id:12s} {where:14s} {got:26s} {how}"


def cmd_list(a) -> int:
    cfg = load_config(_config_path(a))
    unfinished = {lid: s for lid, s in journal_setups().items() if lid not in cfg.lines}
    if not cfg.lines and not unfinished:
        _say("no lines configured; try: ai-phone lines add telnyx --key-env TELNYX_API_KEY")
        return EXIT_OK
    for lid, ln in cfg.lines.items():
        registrar = (ln.registrar or "").removeprefix("sip:").split(";", 1)[0]
        where = ln.number or ln.aor or f"sip:{ln.user}@{registrar}"
        prov = f" ({ln.provider.name}: {ln.provider.state})" if ln.provider else ""
        _say(f"{lid:12s} {ln.kind:5s} {where}{prov}")
    if unfinished:  # a stopped purchase leaves a journal and no config line; this is where the stop message sends people
        if cfg.lines:
            _say("")
        _say("Unfinished setups")
        for lid, setup in unfinished.items():
            _say(_unfinished_row(lid, setup))
    return EXIT_OK


# ---- forward / verify ------------------------------------------------------------------------

def cmd_forward(a) -> int:
    cfg = load_config(_config_path(a))
    ln = cfg.lines.get(a.line_id) or _fail(f"no line {a.line_id}")
    if ln.kind != "trunk" or not ln.number:
        _fail(f"line {a.line_id} has no phone number to divert to")
    if not E164_RE.fullmatch(a.from_number):
        _fail("--from must be your mobile number in E.164, like +447700900123")
    c = divert_codes(a.when, ln.number)
    _say(f"On your iPhone, open the dialler, type {c['activate']} and press call.\n"
         f"Check it with {c['check']}. Cancel this divert with {c['cancel']}; cancel every divert with {c['cancel_all']}.\n"
         f"(Settings → Phone → Call Forwarding only does 'all calls'. Some pay-as-you-go plans do not allow diverts.)")
    diverted = sorted(set(ln.diverted_from) | {a.from_number})
    update_line(_config_path(a), a.line_id, diverted_from=diverted)
    return EXIT_OK


def cmd_verify(a) -> int:
    config_path = _config_path(a)
    pname, ln = _journal_for(a.line_id, a)
    if ln.get("from_user", "number") == "account" and not managed_caller_id(ln):
        _fail(f"line {a.line_id} uses its SIP account in From; configure the caller ID with your provider")
    if not pname:
        _fail(f"line {a.line_id} was not provisioned by a carrier module; verify the number in your carrier's portal")
    if not E164_RE.fullmatch(a.number):
        _fail("number must be E.164")
    a.provider = pname
    provider = _provider(a)
    if managed_caller_id(ln):
        ln = caller_id_line(config_path, a.line_id)
        provider.validate_context(caller_id_journal(a.line_id, ln))
    step = provider.start_caller_id_verification(a.number)
    _say(step.instructions)
    ok = False
    if step.kind == "enter_code_in_cli":
        for _ in range(3):
            code = input("Code: ").strip()
            if provider.complete_caller_id_verification(step, code):
                ok = True
                break
            _say("That code was not accepted.")
    else:
        deadline = time.time() + 120
        while time.time() < deadline and not ok:
            ok = provider.complete_caller_id_verification(step, None)
            if not ok:
                time.sleep(3)
    if not ok:
        _say("Verification did not complete.")
        return EXIT_ERROR
    if managed_caller_id(ln):
        apply_caller_id(config_path, a.line_id, a.number, provider=provider, verified=True)
        _say(f"Line {a.line_id} now calls out as {a.number}. What the recipient sees is up to the carriers.")
        return EXIT_OK
    verified = [v for v in (ln.get("verified_caller_ids") or []) if v.get("number") != a.number]
    verified.append({"number": a.number, "provider": pname, "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    update_line(config_path, a.line_id, verified_caller_ids=verified)
    _say(f"{a.number} can now be used as caller_id on line {a.line_id} (set `caller_id` in the config to use it). "
         "What the callee's handset shows is up to the carriers.")
    return EXIT_OK


def cmd_set_caller_id(a) -> int:
    config_path = _config_path(a)
    line = caller_id_line(config_path, a.line_id)
    provider = None
    if managed_caller_id(line):
        a.provider = line["provider"]["name"]
        provider = _provider(a)
    apply_caller_id(config_path, a.line_id, a.number, provider=provider)
    _say(f"Line {a.line_id} calls out as {a.number}. Restart ai-phone to apply its local settings.")
    return EXIT_OK


def _guarded(fn):
    """Every lines command runs under the lines lock for its whole duration (config, secrets and journal writes
    included, so two commands never interleave), and provider/secrets failures are reported, never traced (exit 3)."""
    def run(a) -> int:
        from ai_phone.providers.base import ProviderError
        try:
            with line_lock(paths.lines_dir()):
                return fn(a)
        except (ProviderError, SecretsError, ValueError, OSError, RuntimeError) as e:
            _say(f"error: {e}")
            return EXIT_ERROR
    return run


# ---- argparse ------------------------------------------------------------------------------------

def register(sub) -> None:
    p = sub.add_parser("lines", help="phone numbers and SIP addresses this daemon answers on")
    p.add_argument("--config", default=None)
    s = p.add_subparsers(dest="lines_cmd", required=True)

    add = s.add_parser("add", help="add a line: telnyx | twilio | signalwire | trunk | sip")
    add.add_argument("provider", choices=sorted(set(PROVIDER_CHOICES) | {"trunk", "sip"}))
    add.add_argument("--id")
    add.add_argument("--yes", action="store_true")
    add.add_argument("--key-env"); add.add_argument("--secret-env"); add.add_argument("--space", help="SignalWire Space host, such as example.signalwire.com")
    add.add_argument("--country", default="GB"); add.add_argument("--area"); add.add_argument("--number")
    add.add_argument("--existing", metavar="E164", help="connect a number already on your carrier account instead of buying one")
    add.add_argument("--registrar"); add.add_argument("--outbound-proxy"); add.add_argument("--dial-domain")
    add.add_argument("--user"); add.add_argument("--password-env"); add.add_argument("--caller-id")
    add.add_argument("--from-user", choices=["number", "account"],
                     help="manual trunks: use the phone number (default) or SIP account username in From")
    add.add_argument("--transport", choices=["udp", "tcp", "tls"])
    add.add_argument("--srtp", choices=["off", "optional", "required"], help="manual trunk media encryption; optional/required need TLS")
    add.add_argument("--preset", choices=["signalwire"], help="manual trunk: SignalWire account identity; TLS/SRTP by default")
    add.add_argument("--trusted-cidr", action="append"); add.add_argument("--host"); add.add_argument("--trusted", action="append")
    add.set_defaults(func=_guarded(cmd_add))

    r = s.add_parser("resume"); r.add_argument("line_id"); r.add_argument("--key-env", required=True); r.add_argument("--secret-env"); r.add_argument("--space"); r.set_defaults(func=_guarded(cmd_resume))
    rm = s.add_parser("remove"); rm.add_argument("line_id"); rm.add_argument("--key-env"); rm.add_argument("--secret-env"); rm.add_argument("--space")
    rm.add_argument("--local-only", action="store_true"); rm.add_argument("--yes", action="store_true"); rm.set_defaults(func=_guarded(cmd_remove))
    s.add_parser("list").set_defaults(func=_guarded(cmd_list))
    f = s.add_parser("forward"); f.add_argument("line_id"); f.add_argument("--when", choices=list(WHEN), default="no-answer")
    f.add_argument("--from", dest="from_number", required=True); f.set_defaults(func=_guarded(cmd_forward))
    v = s.add_parser("verify-caller-id"); v.add_argument("line_id"); v.add_argument("number"); v.add_argument("--key-env", required=True)
    v.add_argument("--secret-env"); v.add_argument("--space"); v.set_defaults(func=_guarded(cmd_verify))
    cid = s.add_parser("set-caller-id", help="use the line number or a caller ID already verified for this line")
    cid.add_argument("line_id"); cid.add_argument("number"); cid.add_argument("--key-env"); cid.add_argument("--secret-env"); cid.add_argument("--space")
    cid.set_defaults(func=_guarded(cmd_set_caller_id))

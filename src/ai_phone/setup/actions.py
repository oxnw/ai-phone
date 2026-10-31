"""Everything the setup flow changes or checks. The flow never writes a file or calls a carrier itself; it calls these.
A failure the person can fix raises ActionError with a message that is already safe to show (no secrets).

Config writes take the lines lock, change one section, write, and load the result; a config that no longer loads is
put back as it was and the ConfigError becomes the ActionError."""
from __future__ import annotations

import atexit
import contextlib
import copy
import functools
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import warnings
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

import httpx
import yaml

import ai_phone
from ai_phone import models, paths, voices
from ai_phone.carriers import CARRIERS, CARRIER_KEYS, managed_caller_id, provider_kwargs
from ai_phone.lines.caller_id import apply_caller_id, caller_id_journal, caller_id_line
from ai_phone.cli_lines import journal_setups, manual_line, next_line_id, provision_flow, write_line
from ai_phone.config import E164_RE, SUBSCRIPTION_MODELS, Config, ConfigError, line_password_env, load_config
from ai_phone.control.client import probe
from ai_phone.lines.config_writer import read_raw, write_raw
from ai_phone.lines.forward import divert_codes
from ai_phone.lines.journal import Journal, line_lock
from ai_phone.secrets import SecretsError, load_secrets, redact, remove_secret, write_secret
from ai_phone.services.backends.cli_brain import CLAUDE_MODELS, ClaudeCodeLLM, CliModel, CodexLLM, codex_models
from ai_phone.services.errors import BrainError
from ai_phone.telephony.state import StateError, StateFile

log = logging.getLogger("ai-phone.setup")
BRAIN_KEY_ENV = "AI_PHONE_BRAIN_API_KEY"  # a secret name by the daemon's rules (it contains KEY), so workers never inherit it
VENDOR = Path(ai_phone.__file__).parent / "vendor"  # where install.sh puts the PJSIP binding
DAEMON = [sys.executable, "-c", "import sys; from ai_phone.cli import main; sys.exit(main(sys.argv[1:]))", "run", "--fake"]
CHECK_SEC, POLL_SEC, STOP_SEC = 20, 0.5, 10  # how long a trial start may take to register, how often to look, how long to stop
EXIT_STOP_SEC = 2  # how long a trial start may take to stop when setup itself is exiting
SAMPLE_RATE = 24000  # both speech models' native rate
LOOPBACK = ("127.0.0.1", "localhost", "::1")  # the config's rule: plain http only to this machine
BRAINS = {"claude": ClaudeCodeLLM, "chatgpt": CodexLLM}  # the subscription brains, by the config's kind
INSTALL = {"claude": "npm install -g @anthropic-ai/claude-code", "chatgpt": "npm install -g @openai/codex"}
LOGIN = {"claude": "claude, then type /login", "chatgpt": "codex login"}
PROBE_MS = 30000  # how long the login probe may take: a CLI that has to start, then the provider's own first word


@dataclass
class Machine:
    sip_library: bool           # the PJSIP binding is built into src/ai_phone/vendor
    memory_gb: int
    models_missing: list[str]   # models still to download for the current config (defaults when there is none)
    config_exists: bool
    models_downloaded: list[str] = field(default_factory=list)  # complete cached models known to ai-phone


@dataclass
class Current:
    """What the config says now (defaults for a new config)."""
    exists: bool
    lines: dict[str, dict]      # raw line entries by id
    brain_kind: str             # "local" | "openai_compatible" | "claude" | "chatgpt"
    tier: str                   # resolved: a key from models.CATALOG, or a Hub repo id named by hand
    base_url: str
    model: str
    brain_key_saved: bool       # the brain's api_key_env has a value in secrets.env
    brain_cli: str | None       # brain.cli: where a subscription brain's CLI is, when the config says
    voice: str
    speed: float
    greeting: str
    instructions: str
    daily_limit: int            # calls.daily_limit, or 20 when the config has none
    tts_engine: str = "kokoro"
    allow_inbound: bool = True
    allow_outbound: bool = True


@dataclass
class LineResult:
    line_id: str
    code: int                   # cli_lines EXIT_OK / EXIT_PENDING / EXIT_ERROR
    messages: list[str]


class ActionError(RuntimeError):
    pass


def sip_library_built() -> bool:
    return any(VENDOR.glob("_pjsua2*.so"))


def _section(raw: dict, name: str) -> dict:
    value = raw.get(name)
    return dict(value) if isinstance(value, dict) else {}


BRAIN_KIND_KEYS = ("kind", "base_url", "model", "api_key_env")  # what choosing a brain sets; the rest is the person's


def _kept_brain(raw: dict, kind: str) -> dict:
    """The brain settings that are the person's own — the timeouts, a hand-set path to a CLI — kept through a change of
    brain, because `config.example.yaml` promises that a number they set themselves is the one used.

    `cli` belongs to the kind it was set for: a path to Claude Code left behind would be run as Codex, so it goes
    when the kind changes.
    """
    brain = _section(raw, "brain")
    drop = set(BRAIN_KIND_KEYS) if brain.get("kind") == kind else {*BRAIN_KIND_KEYS, "cli"}
    return {k: v for k, v in brain.items() if k not in drop}


def _saved() -> dict[str, str]:
    try:
        return load_secrets()
    except SecretsError:
        return {}


class Actions:
    def __init__(self, config_path: Path, player=None, provider_loader=None, http: httpx.Client | None = None):
        self.config_path = Path(config_path)
        self.player = player or KokoroPlayer()
        self.phonon_player = player or PhononPlayer()
        self.provider_loader = provider_loader
        self.http = http  # tests pass a client with a mock transport
        self._probed: dict[tuple[str, str], dict] = {}  # a model already proved in this run, by (kind, model)
        self._logged_in: dict[str, dict] = {}           # a login already established in this run, by kind

    # ---- what is there ---------------------------------------------------------------------------

    def machine(self) -> Machine:
        cur = self.current()
        needed = models.needed(cur.tier, cur.brain_kind, cur.tts_engine)
        candidates = dict.fromkeys([models.STT, models.TTS, models.PHONON, *models.LLM.values(), *needed])
        downloaded = [repo for repo in candidates if models.local_path(repo) is not None]
        return Machine(sip_library=sip_library_built(), memory_gb=models.memory_gb(),
                       models_missing=[repo for repo in needed if repo not in downloaded], config_exists=cur.exists,
                       models_downloaded=downloaded)

    def current(self) -> Current:
        exists = self.config_path.exists()
        raw, cfg = {}, Config()
        if exists:
            try:
                raw = read_raw(self.config_path)
                cfg = load_config(self.config_path)
            except (ConfigError, yaml.YAMLError, OSError):
                pass  # the Check step shows why; setup still runs, so the person can change what is wrong
        raw = raw if isinstance(raw, dict) else {}
        lines = {k: dict(v) for k, v in _section(raw, "lines").items() if isinstance(v, dict)}
        limit = _section(raw, "calls").get("daily_limit", 20)
        limit = limit if isinstance(limit, int) and not isinstance(limit, bool) and limit >= 0 else 20
        key_env = cfg.brain.api_key_env
        return Current(exists=exists, lines=lines, brain_kind=cfg.brain.kind, tier=models.resolve_tier(cfg.models.tier),
                       base_url=cfg.brain.base_url, model=cfg.brain.model, brain_key_saved=bool(key_env and _saved().get(key_env)),
                       brain_cli=cfg.brain.cli,
                       voice=cfg.tts.voice, speed=cfg.tts.speed, greeting=cfg.inbound.greeting,
                       instructions=cfg.inbound.instructions, daily_limit=limit, tts_engine=cfg.tts.engine,
                       allow_inbound=cfg.calls.allow_inbound, allow_outbound=cfg.calls.allow_outbound)

    def _read(self) -> dict:
        """The raw config for a change. A file that is not a YAML mapping is never overwritten."""
        try:
            raw = read_raw(self.config_path)
        except yaml.YAMLError:
            raise ActionError(f"{self.config_path.name} is not valid YAML; fix it or move it away, then run setup again") from None
        if not isinstance(raw, dict):
            raise ActionError(f"{self.config_path.name} must be a mapping; fix it or move it away, then run setup again")
        return raw

    def _edit(self, change: Callable[[dict], None]) -> None:
        with line_lock(paths.lines_dir()):
            existed = self.config_path.exists()
            raw = self._read()
            before = copy.deepcopy(raw)
            change(raw)
            write_raw(self.config_path, raw)
            try:
                load_config(self.config_path)
            except ConfigError as e:
                if existed:
                    write_raw(self.config_path, before)
                else:
                    self.config_path.unlink()
                raise ActionError(str(e)) from None

    # ---- call directions -------------------------------------------------------------------------

    def save_call_directions(self, inbound: bool, outbound: bool) -> None:
        def change(raw: dict) -> None:
            raw["calls"] = {**_section(raw, "calls"), "allow_inbound": inbound, "allow_outbound": outbound}
        self._edit(change)

    # ---- brain -----------------------------------------------------------------------------------

    def save_brain_local(self, tier: str) -> None:
        def change(raw: dict) -> None:
            raw["brain"] = {"kind": "local", **_kept_brain(raw, "local")}
            raw["models"] = {**_section(raw, "models"), "tier": tier}
        self._edit(change)

    def _brain_key_env(self, url: str) -> str | None:
        """The key variable the current endpoint brain uses, when `url` is on that same host: a key saved for one
        server is never sent to another."""
        try:
            brain = _section(self._read(), "brain")
        except ActionError:
            return None
        env = brain.get("api_key_env")
        if brain.get("kind") != "openai_compatible" or not isinstance(env, str):
            return None
        return env if _host(url) is not None and _host(brain.get("base_url")) == _host(url) else None

    def test_endpoint(self, url: str, model: str, key: str | None) -> str:
        try:
            parts = urlsplit(url)
            host = parts.hostname or url
        except ValueError:
            parts, host = None, url
        if parts is not None and parts.scheme == "http" and parts.hostname not in LOOPBACK:
            raise ActionError("a brain that is not on this machine must use https")  # before a key can cross the network
        if key is None:
            env = self._brain_key_env(url)
            key = _saved().get(env) if env else None
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        body = {"model": model, "messages": [{"role": "user", "content": "Say OK."}], "max_tokens": 5}
        try:
            r = (self.http or httpx).post(f"{url.rstrip('/')}/chat/completions", json=body, headers=headers, timeout=15)
        except (httpx.HTTPError, httpx.InvalidURL, ValueError) as e:
            raise ActionError(f"could not reach {host}: {type(e).__name__}") from None
        if r.status_code >= 400:
            try:
                said = r.json()["error"]["message"]
            except (ValueError, KeyError, TypeError):
                said = r.text
            said = redact(" ".join(str(said).split()), [key])[:160]
            raise ActionError(f"the endpoint said {r.status_code}: {said}")
        try:
            return r.json()["choices"][0]["message"].get("content") or ""
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            raise ActionError(f"{host} answered, but not like an OpenAI chat API (is the URL right? it usually ends in /v1)") from None

    # ---- a Claude or ChatGPT plan as the brain ----------------------------------------------------

    def cli_installed(self, kind: str) -> bool:
        """Is the CLI this plan is reached through on PATH (or where the config says it is)?"""
        return shutil.which(self._cli_path(kind) or BRAINS[kind].cli_name) is not None

    def _cli_path(self, kind: str) -> str | None:
        """Where the config says this plan's CLI is — only while the config is that plan. A path set for Claude Code
        is no use for running Codex, and it outlives a change of kind in a hand-written config."""
        cur = self.current()
        return cur.brain_cli if cur.brain_kind == kind else None

    def _brain(self, kind: str, model: str):
        return BRAINS[kind](model=model, cli=self._cli_path(kind), first_token_timeout_ms=PROBE_MS,
                            stream_idle_timeout_ms=PROBE_MS)

    def _brain_model(self, kind: str) -> str:
        """Which model to probe with: the one the config already names for this plan, or the plan's own default."""
        cur = self.current()
        return cur.model if cur.brain_kind == kind else SUBSCRIPTION_MODELS[kind]

    def _brain_trouble(self, kind: str, e: BrainError) -> ActionError:
        """What the person can do about it, in their own words. Never the CLI's own output."""
        cli = BRAINS[kind].cli_name
        if e.kind == "cli_missing":
            return ActionError(f"{cli} is not installed. Install it with: {INSTALL[kind]}")
        if e.kind == "not_logged_in":
            return ActionError(f"{cli} is installed but not logged in. Log in with: {LOGIN[kind]}")
        if e.kind in ("first_token_timeout", "idle_timeout"):
            return ActionError(f"{cli} did not answer within {PROBE_MS // 1000} seconds. Try again, or pick another brain.")
        if e.kind == "model_not_allowed":
            return ActionError("Your plan does not allow that model. Choose another one.")
        if e.kind == "api_key_login":
            return ActionError(f"{cli} is logged in as {e.detail!r}, which is billed per request. ai-phone runs your "
                               f"subscription instead: remove the key, then log in with: {LOGIN[kind]}")
        if e.kind == "login_unknown":
            # the login method is a value from the CLI's own structured output, so it is safe to show, and showing it is
            # the only way the person can tell which of their settings is in the way
            return ActionError(f"{cli} is logged in as {e.detail!r}, which ai-phone cannot run a subscription on. "
                               f"Log in with: {LOGIN[kind]}")
        if e.kind == "tools_renamed":
            return ActionError(f"This {cli} no longer has the settings ai-phone switches its tools off with "
                               f"({e.detail}). Update ai-phone, or pick another brain.")
        if e.kind in ("tool_use", "tools_unknown"):
            return ActionError(f"{cli} gives a call's session tools of its own, and what callers say must not reach "
                               f"anything that can read this computer. Update {cli}, then check again.")
        if e.kind == "rate_limited":
            return ActionError(f"Your {cli} plan has reached its limit for now. Try again later, or pick another brain.")
        # anything else (an org policy, a provider incident) was classified from the CLI's own text, which is
        # unbounded third-party prose: the kind is enough for the person, the detail is already in the debug log
        log.debug("%s brain probe failed: %s", cli, e)
        return ActionError(f"{cli} could not answer ({e.kind}). Try again, or pick another brain.")

    def login_status(self, kind: str) -> dict:
        """Is the person logged in on this plan, and is it a plan and not an API key? No inference, so no turn.

        This is what the wizard establishes before it offers models, so that choosing one is the only thing a setup
        run spends. Remembered for the run; a failure is not, because "Check again" has to mean it.
        """
        if kind in self._logged_in:
            return self._logged_in[kind]
        brain = self._brain(kind, self._brain_model(kind))
        try:
            said = brain.login()
        except BrainError as e:
            raise self._brain_trouble(kind, e) from None
        finally:
            brain.close()
        self._logged_in[kind] = {"plan": str(said.get("plan") or "")}
        return self._logged_in[kind]

    def test_subscription(self, kind: str, model: str | None = None) -> dict:
        """One tiny turn through the person's own CLI, against the model that will take the calls: it proves the plan
        answers with that model and that the session comes up with no tools.

        A proved model is remembered for the rest of the run, so a setup run spends one turn of their plan and not one
        per page: the Check step asks for the model the config now names, which is this one. A failure is never
        remembered: the point of "Check again" is to try once more.
        """
        model = model or self._brain_model(kind)
        if (kind, model) in self._probed:
            return self._probed[(kind, model)]
        brain = self._brain(kind, model)
        try:
            timings = brain.warmup()
        except BrainError as e:
            raise self._brain_trouble(kind, e) from None
        finally:
            brain.close()
        self._probed[(kind, model)] = {"model": model, "cli_version": timings.get("cli_version", ""),
                                       "warm_ms": timings.get("warm_ms", 0.0)}
        return self._probed[(kind, model)]

    def brain_models(self, kind: str) -> list[CliModel]:
        """What setup offers as the model: Claude Code's aliases, or the models this ChatGPT plan lists (no turn is
        spent on the asking), fastest-measured first."""
        if kind == "claude":
            return list(CLAUDE_MODELS)
        try:
            found = codex_models(cli=self._cli_path(kind))
        except BrainError as e:
            raise self._brain_trouble(kind, e) from None
        if not found:
            raise ActionError("Codex listed no models for this account. Check your plan, then try again.")
        fastest = SUBSCRIPTION_MODELS[kind]
        return sorted(found, key=lambda m: m.id != fastest)

    def save_brain_subscription(self, kind: str, model: str) -> None:
        def change(raw: dict) -> None:
            raw["brain"] = {"kind": kind, "model": model, **_kept_brain(raw, kind)}
        self._edit(change)

    def save_brain_endpoint(self, url: str, model: str, key: str | None) -> None:
        env = BRAIN_KEY_ENV if key else self._brain_key_env(url)
        previous = _saved().get(BRAIN_KEY_ENV)

        def change(raw: dict) -> None:
            brain = {"kind": "openai_compatible", "base_url": url, "model": model,
                     **_kept_brain(raw, "openai_compatible")}
            if env:
                brain["api_key_env"] = env
            raw["brain"] = brain

        try:
            if key:
                write_secret(BRAIN_KEY_ENV, key)
            self._edit(change)
        except (ActionError, SecretsError) as e:
            if key:  # put back the key that was there
                try:
                    if previous:
                        write_secret(BRAIN_KEY_ENV, previous)
                    else:
                        remove_secret(BRAIN_KEY_ENV)
                except SecretsError:
                    pass
            raise ActionError(redact(str(e), [key])) from None

    # ---- carriers --------------------------------------------------------------------------------

    def saved_keys(self, carrier: str) -> dict[str, str] | None:
        saved = _saved()
        keys = {name: saved.get(name) for name in CARRIER_KEYS[carrier]}
        return keys if all(keys.values()) else None

    def _provider(self, carrier: str, creds: dict[str, str]):
        from ai_phone.providers.base import ProviderError
        loader = self.provider_loader
        if loader is None:
            from ai_phone.providers import load_provider as loader
        try:
            return loader(carrier, **provider_kwargs(carrier, creds))
        except KeyError:
            raise ActionError(f"Enter all {CARRIERS[carrier].label} credential fields.") from None
        except ProviderError as e:
            raise ActionError(redact(f"That key did not work: {e}", creds.values())) from None

    def _carrier_error(self, carrier: str, creds: dict[str, str], e, otherwise: str) -> ActionError:
        """A request that never got an answer (or a carrier outage) is not the key's fault."""
        text = f"Could not reach {carrier.capitalize()}: {e}" if e.retryable else f"{otherwise}: {e}"
        return ActionError(redact(text, creds.values()))

    def owned_numbers(self, carrier: str, creds: dict[str, str]) -> list:
        from ai_phone.providers.base import ProviderError
        try:
            return self._provider(carrier, creds).owned_numbers()
        except ProviderError as e:
            raise self._carrier_error(carrier, creds, e, "That key did not work") from None

    def search_numbers(self, carrier: str, creds: dict[str, str], country: str, area: str | None) -> list:
        from ai_phone.providers.base import ProviderError
        try:
            return self._provider(carrier, creds).search_numbers(country, area, 9)
        except ProviderError as e:
            raise self._carrier_error(carrier, creds, e, "Could not search for numbers") from None

    def unfinished_numbers(self) -> set[str]:
        """Numbers an earlier attempt left half set up (a journal with no line in the config)."""
        try:
            lines = _section(self._read(), "lines")
        except ActionError:
            lines = {}
        return {j.number for line_id, j in journal_setups().items() if line_id not in lines and j.number not in (None, "?")}

    def resumable_numbers(self, carrier: str) -> dict[str, str]:
        """SignalWire existing-number attempts that setup can safely offer to finish, by number."""
        if carrier != "signalwire":
            return {}
        lines = _section(self._read(), "lines")
        connected = {ln.get("number") for ln in lines.values() if isinstance(ln, dict)}
        journals = journal_setups()
        return {j.number: line_id for line_id, j in journals.items()
                if line_id not in lines and j.number not in connected and j.number not in (None, "?")
                and j.provider == carrier and j.brought and j.resumable and j.state == "pending"
                and sum(other.number == j.number for other in journals.values()) == 1}

    def _line_id(self, raw: dict, carrier: str, number, existing: bool) -> str:
        """The id for setting up `number`. An earlier attempt at this number (a journal with no line, under any id) is
        picked up where it stopped when it was set up the same way (bought, or brought); otherwise the person is told how
        to finish or undo it. A second id for one number would let `lines remove` of the first release the number the
        second depends on. Anything else gets twilio, twilio-2, …: the first id with neither a line nor a journal."""
        lines, journals = _section(raw, "lines"), journal_setups()
        for line_id, j in journals.items():
            if line_id in lines or j.number != number.e164:
                continue
            if j.state == "rollback_failed":
                raise ActionError(f"An earlier setup of {number.e164} (line {line_id}) could not be fully undone. Finish "
                                  f"removing it in your {carrier.capitalize()} account, then run "
                                  f"`ai-phone lines remove {line_id} --local-only`.")
            if j.resumable and j.brought == existing:
                return line_id
            raise ActionError(f"line {line_id} has an unfinished setup for {number.e164}; run `ai-phone lines resume "
                              f"{line_id}` to finish it or `ai-phone lines remove {line_id}` to undo it, then try again")
        n = 1
        while True:
            line_id = carrier if n == 1 else f"{carrier}-{n}"
            n += 1
            if line_id not in lines and (line_id not in journals or journals[line_id].number is None):
                return line_id  # an empty journal records nothing, so it can be used

    def _connect(self, carrier: str, creds: dict[str, str], number, existing: bool) -> LineResult:
        provider = self._provider(carrier, creds)
        messages: list[str] = []
        line_id = ""
        try:
            for name, value in creds.items():
                write_secret(name, value)
            with line_lock(paths.lines_dir()):
                raw = self._read()
                line_id = self._line_id(raw, carrier, number, existing)
                if self.config_path.exists():
                    load_config(self.config_path)  # the line is written into this config: it must load before anything is bought
                journal = Journal.open(paths.lines_dir(), line_id, provider.name)
                code = provision_flow(provider, journal, self.config_path, line_id, number, existing=existing, say=messages.append)
        except (RuntimeError, ValueError, OSError) as e:  # ConfigError, SecretsError, ProviderError, a journal problem
            error = ActionError(redact(str(e), creds.values()))
            if line_id and (paths.lines_dir() / f"{line_id}.journal.json").exists():  # carrier work may have begun
                error = ActionError(f"{error}\nThe journal is kept: run `ai-phone lines resume {line_id}` to finish it, "
                                    f"or `ai-phone lines remove {line_id}` to undo it.")
                error.line_id = line_id
            raise error from None
        return LineResult(line_id, code, [redact(m, creds.values()) for m in messages])

    def buy_number(self, carrier: str, creds: dict[str, str], number) -> LineResult:
        if not CARRIERS[carrier].purchase:
            raise ActionError(f"Buy a number in your {CARRIERS[carrier].label} Space, then connect it as an existing number.")
        return self._connect(carrier, creds, number, existing=False)

    def attach_number(self, carrier: str, creds: dict[str, str], number) -> LineResult:
        return self._connect(carrier, creds, number, existing=True)

    def forward(self, line_id: str, mobile: str, when: str) -> dict:
        with line_lock(paths.lines_dir()):
            line = _section(self._read(), "lines").get(line_id)
            number = line.get("number") if isinstance(line, dict) else None
            if not number:
                raise ActionError(f"line {line_id} has no phone number to divert to")
            codes = divert_codes(when, number)

            def change(raw: dict) -> None:
                ln = raw["lines"][line_id]
                ln["diverted_from"] = sorted(set(ln.get("diverted_from") or []) | {mobile})
            self._edit(change)
        return codes

    def start_caller_id(self, carrier: str, creds: dict[str, str], number: str, *, line_id: str | None = None):
        """Ask the carrier to prove the person owns `number`. The step says how they finish it."""
        from ai_phone.providers.base import ProviderError
        try:
            provider = self._provider(carrier, creds)
            if CARRIERS[carrier].managed_caller_id:
                if not line_id:
                    raise ActionError("Select a line before verifying its caller ID.")
                line = caller_id_line(self.config_path, line_id)
                if (line.get("provider") or {}).get("name") != carrier:
                    raise ActionError("The selected line belongs to another carrier.")
                provider.validate_context(caller_id_journal(line_id, line))
            return provider.start_caller_id_verification(number)
        except ProviderError as e:
            raise self._carrier_error(carrier, creds, e, "Could not start the check") from None
        except (ValueError, OSError, RuntimeError) as e:
            raise ActionError(redact(str(e), creds.values())) from None

    def check_caller_id(self, carrier: str, creds: dict[str, str], step, code: str | None) -> bool:
        """Whether the carrier has accepted it yet: with the code they texted, or None while the call is waiting."""
        from ai_phone.providers.base import ProviderError
        try:
            return self._provider(carrier, creds).complete_caller_id_verification(step, code)
        except ProviderError as e:
            raise self._carrier_error(carrier, creds, e, "Could not check the code") from None

    def record_caller_id(self, line_id: str, number: str, carrier: str, creds: dict[str, str] | None = None) -> None:
        """The carrier accepted it: remember the verified number and call out as it, which is the point of the step
        (`ai-phone lines verify-caller-id` only records it for carriers using the phone number in From)."""
        line = _section(self._read(), "lines").get(line_id) or {}
        if managed_caller_id(line):
            if (line.get("provider") or {}).get("name") != carrier:
                raise ActionError("The selected line belongs to another carrier.")
            self._apply_carrier_caller_id(line_id, number, carrier, creds, verified=True)
            return
        def change(raw: dict) -> None:
            lines = raw.get("lines")
            ln = lines.get(line_id) if isinstance(lines, dict) else None  # the line itself, to change in place
            if not isinstance(ln, dict):
                raise ActionError(f"no line {line_id} to verify a number for")
            if ln.get("from_user", "number") == "account":
                raise ActionError(f"line {line_id} uses its SIP account in From; configure the caller ID with your provider")
            verified = [v for v in (ln.get("verified_caller_ids") or []) if v.get("number") != number]
            verified.append({"number": number, "provider": carrier,
                             "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            ln["verified_caller_ids"] = verified
            ln["caller_id"] = number
        self._edit(change)

    def use_caller_id(self, line_id: str, number: str, creds: dict[str, str] | None = None) -> None:
        """Select an eligible caller ID, applying it at the carrier when that endpoint controls it."""
        line = _section(self._read(), "lines").get(line_id) or {}
        if managed_caller_id(line):
            self._apply_carrier_caller_id(line_id, number, line["provider"]["name"], creds)
            return
        def change(raw: dict) -> None:
            lines = raw.get("lines")
            ln = lines.get(line_id) if isinstance(lines, dict) else None
            if not isinstance(ln, dict):
                raise ActionError(f"no line {line_id}; add a phone line before choosing its caller ID")
            if ln.get("kind", "trunk") != "trunk":
                raise ActionError(f"line {line_id} uses a SIP address; choose a phone line to change its caller ID")
            if ln.get("from_user", "number") == "account":
                raise ActionError(f"line {line_id} uses its SIP account in From; configure the caller ID with your provider")
            provider = ln.get("provider")
            if provider is not None and (not isinstance(provider, dict) or provider.get("state", "pending") != "ready"):
                raise ActionError(f"line {line_id} is not ready; finish its setup before choosing its caller ID")
            carrier = provider.get("name", "") if provider else ""
            if not isinstance(number, str) or not E164_RE.fullmatch(number):
                raise ActionError("choose a phone number in E.164 format, like +15555550100")
            verified = ln.get("verified_caller_ids") or []
            allowed = isinstance(verified, list) and any(
                isinstance(v, dict) and v.get("number") == number and v.get("provider", "") == carrier
                for v in verified
            )
            if number != ln.get("number") and not allowed:
                raise ActionError(f"{number} is not verified for line {line_id}; verify it first or choose the line's own number")
            ln["caller_id"] = number
        self._edit(change)

    def _apply_carrier_caller_id(self, line_id: str, number: str, carrier: str,
                                 creds: dict[str, str] | None, *, verified: bool = False) -> None:
        if not creds:
            raise ActionError(f"Enter the {CARRIERS[carrier].label} credentials for this line to change its caller ID.")
        try:
            provider = self._provider(carrier, creds)
            apply_caller_id(self.config_path, line_id, number, provider=provider, verified=verified)
        except (RuntimeError, ValueError, OSError) as e:
            raise ActionError(redact(f"{e}\nIf the carrier change was applied, finish saving it with: "
                                     f"ai-phone lines resume {line_id} --key-env SIGNALWIRE_PROJECT_ID "
                                     "--secret-env SIGNALWIRE_API_TOKEN --space <space-host>", creds.values())) from None

    # ---- lines by hand ---------------------------------------------------------------------------

    def lan_address(self) -> str:
        """The address other devices reach this computer on: the source address of the default route (nothing is sent)."""
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("192.0.2.1", 9))
            address = s.getsockname()[0]
        except OSError:
            return "127.0.0.1"
        finally:
            s.close()
        return address if address != "0.0.0.0" else "127.0.0.1"

    def add_sip_line(self, name: str, host: str) -> str:
        with line_lock(paths.lines_dir()):
            raw = self._read()
            # two SIP lines with one address: the endpoint gives every call to the first, so the other never rings
            for other, ln in _section(raw, "lines").items():
                if not isinstance(ln, dict) or ln.get("kind") != "sip":
                    continue
                aor = str(ln.get("aor") or "")
                if name == aor.removeprefix("sip:").split("@", 1)[0]:
                    raise ActionError(f"{aor} is already line {other}; pick another name")
                if name == ln.get("user"):
                    raise ActionError(f"sip:{name}@… is already line {other}; pick another name")
            line_id = next_line_id(raw, name)
            try:
                write_line(self.config_path, line_id, manual_line("sip", user=name, host=host))
            except ValueError as e:  # a ConfigError is a ValueError too
                raise ActionError(str(e)) from None
        return line_id

    def add_trunk_line(self, registrar: str, user: str, password: str, number: str, transport: str,
                       from_user: str = "number", *, preset: str | None = None, srtp: str = "off") -> str:
        if preset not in (None, "signalwire"):
            raise ActionError("Unknown SIP trunk preset.")
        if preset == "signalwire":
            from_user = "account"
        with line_lock(paths.lines_dir()):
            line_id = next_line_id(self._read(), preset or "trunk")
            env = line_password_env(line_id)
            try:
                write_secret(env, password)
                write_line(self.config_path, line_id, manual_line("trunk", number=number or None, registrar=registrar, user=user,
                                                                  password_env=env, transport=transport, from_user=from_user,
                                                                  outbound_proxy=registrar if preset else None, srtp=srtp,
                                                                  tls_hostname_policy="verified_wildcard" if preset == "signalwire" and transport == "tls" else "strict"))
            except (ValueError, SecretsError) as e:
                try:
                    remove_secret(env)
                except SecretsError:
                    pass
                raise ActionError(redact(str(e), [password])) from None
        return line_id

    # ---- voice -----------------------------------------------------------------------------------

    def voices(self, *, engine: str = "kokoro") -> list[str]:
        return voices.installed_voices(engine=engine)

    def play_sample(self, voice: str, speed: float, text: str, *, engine: str = "kokoro") -> None:
        try:
            (self.phonon_player if engine == "phonon" else self.player).play(voice, speed, text)
        except ActionError:
            raise
        except Exception as e:  # the audio stack can fail in many ways; none of them should end setup
            raise ActionError(f"could not play the sample: {type(e).__name__}: {e}"[:200]) from None

    def pull_voice_model(self, report: Callable[[int, int], None], *, engine: str = "kokoro") -> None:
        try:
            models.pull([models.tts_repo(engine)], report)
        except models.ModelsError as e:
            raise ActionError(str(e)) from None

    def save_voice(self, voice: str, speed: float, *, engine: str = "kokoro") -> None:
        def change(raw: dict) -> None:
            raw["tts"] = {**_section(raw, "tts"), "engine": engine, "voice": voice, "speed": speed}
        self._edit(change)

    # ---- callers ---------------------------------------------------------------------------------

    def save_callers(self, greeting: str, instructions: str, daily_limit: int) -> None:
        def change(raw: dict) -> None:
            calls = _section(raw, "calls")
            if calls.get("allow_inbound", True):
                raw["inbound"] = {**_section(raw, "inbound"), "greeting": greeting, "instructions": instructions}
            if calls.get("allow_outbound", True):
                raw["calls"] = {**calls, "daily_limit": daily_limit}
        self._edit(change)

    # ---- check -----------------------------------------------------------------------------------

    def missing_models(self) -> list[str]:
        cur = self.current()
        return models.missing(cur.tier, cur.brain_kind, cur.tts_engine)

    def download_size(self) -> int | None:
        return models.download_size(self.missing_models())

    def pull_models(self, report: Callable[[int, int], None]) -> None:
        try:
            models.pull(self.missing_models(), report)
        except models.ModelsError as e:
            raise ActionError(str(e)) from None

    def check_config(self) -> str | None:
        try:
            load_config(self.config_path)
        except ConfigError as e:
            return str(e)
        return None

    def check_lines(self) -> list[tuple[str, str, str]]:
        """(line id, state, detail) for each line: from the running ai-phone, or from a short trial start of it."""
        try:
            running = probe(paths.socket_path())
        except RuntimeError as e:  # socket_path refuses an unsafe directory under /tmp (a symlink, another owner)
            return [("lines", "failed", str(e))]
        except OSError as e:  # or cannot make it (a file in the way)
            return [("lines", "failed", f"could not use the control socket: {e.strerror or type(e).__name__}")]
        if running:
            try:
                return _line_rows(StateFile.read(paths.state_path()))
            except (StateError, ValueError):
                return [("lines", "failed", "ai-phone is running but has not written its status yet")]
        if not sip_library_built():
            return [("lines", "skipped", "SIP library not built · run ./scripts/install.sh")]
        return self._trial_start()

    def _trial_start(self) -> list[tuple[str, str, str]]:
        """Start `ai-phone run --fake` (no models load) until every trunk has an answer from its registrar or CHECK_SEC
        pass, then stop it. Only a state.json written by this start is read."""
        started = time.time()
        doc = None
        with tempfile.TemporaryFile() as log:
            try:
                proc = subprocess.Popen(DAEMON, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            except OSError as e:
                return [("lines", "failed", f"could not start ai-phone: {e.strerror or e}")]
            # a forced quit ends the flow thread (a daemon thread) without the finally below: stop it at exit too, so no
            # trial ai-phone is left holding registrations and a control socket that a later check would believe
            stop_at_exit = functools.partial(_stop, proc, EXIT_STOP_SEC)
            atexit.register(stop_at_exit)
            try:
                while True:
                    if proc.poll() is not None:
                        return [("lines", "failed", _last_line(log))]
                    doc = _fresh_state(started) or doc
                    if doc is not None and _settled(doc):
                        break
                    if time.time() - started >= CHECK_SEC:
                        break
                    time.sleep(POLL_SEC)
            finally:
                _stop(proc)
                atexit.unregister(stop_at_exit)
        if doc is None:
            return [("lines", "registering", f"ai-phone did not report its lines within {CHECK_SEC:g} seconds")]
        return _line_rows(doc)


def _fresh_state(since: float) -> dict | None:
    try:
        doc = StateFile.read(paths.state_path())
    except (StateError, ValueError):
        return None
    return doc if isinstance(doc, dict) and doc.get("written_at", 0) >= since else None


def _settled(doc: dict) -> bool:
    return not any(ln.get("kind") == "trunk" and ln.get("state") in ("registering", "unregistered")
                   for ln in (doc.get("lines") or {}).values())


def _line_rows(doc: dict) -> list[tuple[str, str, str]]:
    rows = []
    for line_id, ln in (doc.get("lines") or {}).items():
        state = ln.get("state", "")
        if ln.get("kind") == "sip":
            rows.append((line_id, "ready", "listening"))
        elif state in ("registered", "registering"):
            rows.append((line_id, state, ""))
        else:
            rows.append((line_id, state, ln.get("last_error") or ""))
    return rows


def _stop(proc, wait: float | None = None) -> None:
    """SIGINT, then kill if it has not stopped within `wait` seconds (STOP_SEC by default). Stopping twice is harmless."""
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=STOP_SEC if wait is None else wait)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _last_line(log) -> str:
    log.seek(0)
    lines = [ln.strip() for ln in log.read().decode(errors="replace").splitlines() if ln.strip()]
    text = lines[-1] if lines else "ai-phone stopped before it reported its lines"
    return redact(text, _saved().values())[:160]


def _host(url) -> str | None:
    try:
        return urlsplit(str(url)).hostname or None
    except ValueError:
        return None


@contextlib.contextmanager
def _quiet():
    """Silence what mlx-audio prints, in plain mode. Off the main thread (the flow under the full-screen app) the
    process-wide streams belong to Textual, which captures prints itself; swapping them there could lose its own."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink), \
            warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _samples(audio):
    import numpy as np
    if type(audio).__module__.startswith("mlx"):
        import mlx.core as mx
        audio = audio.astype(mx.float32)  # numpy has no bfloat16
    return np.asarray(audio, dtype=np.float32).reshape(-1)


class KokoroPlayer:
    """Speaks a sample through this computer's speakers with the downloaded Kokoro model (loaded on first use)."""

    def __init__(self, load=None, run=subprocess.run):
        self._load, self._run = load, run
        self._model = None

    def play(self, voice: str, speed: float, text: str) -> None:
        snapshot = models.local_path(models.TTS)
        if snapshot is None:
            raise ActionError("could not play the sample: the speech model is not downloaded")
        path = voices.voice_path(voice)
        if path is None:  # a bare voice name would make mlx-audio fetch it from another repo
            raise ActionError(f"could not play the sample: the voice {voice} is not installed")
        with _quiet():  # mlx-audio prints progress and warnings; plain mode and the terminal stay clean
            if self._model is None:
                load = self._load
                if load is None:
                    from mlx_audio.tts.utils import load_model as load
                # the snapshot folder is named by a hash, so the model type cannot be read from the path
                self._model = load(str(snapshot), model_type="kokoro")
            parts = [_samples(r.audio) for r in self._model.generate(text=text, voice=str(path), speed=speed,
                                                                      lang_code=voices.lang_code(voice))]
        _play_audio(parts, self._run)


class PhononPlayer:
    """Preview the bundled voice through the same offline backend the daemon uses."""

    def __init__(self, factory=None, run=subprocess.run):
        self._factory, self._run = factory, run
        self._backend = None
        self._gen = 0

    def play(self, voice: str, speed: float, text: str) -> None:
        if models.local_path(models.PHONON) is None:
            raise ActionError("could not play the sample: the speech model is not downloaded")
        if voices.voice_path(voice, engine="phonon") is None:
            raise ActionError(f"could not play the sample: the voice {voice} is not installed")
        with _quiet():
            if self._backend is None:
                factory = self._factory
                if factory is None:
                    from ai_phone.services.backends.mlx_phonon import PhononTTS as factory
                backend = factory(voice=voice, speed=speed)
                backend.warmup()
                self._backend = backend
            self._gen += 1
            pcm = b"".join(self._backend.stream(self._gen, 0, text))
        _play_pcm(pcm, self._run, 16000)


def _play_audio(parts, run) -> None:
    import numpy as np
    parts = [part for part in parts if part.size]
    if not parts:
        raise ActionError("could not play the sample: the speech model returned no audio")
    pcm = (np.clip(np.concatenate(parts), -1.0, 1.0) * 32767).astype("<i2")
    _play_pcm(pcm.tobytes(), run, SAMPLE_RATE)


def _play_pcm(pcm: bytes, run, sample_rate: int) -> None:
    if not pcm:
        raise ActionError("could not play the sample: the speech model returned no audio")
    fd, wav = tempfile.mkstemp(prefix="ai-phone-voice-", suffix=".wav")
    os.close(fd)
    try:
        with wave.open(wav, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(pcm)
        run(["afplay", wav], timeout=30, check=True, capture_output=True)
    finally:
        os.unlink(wav)

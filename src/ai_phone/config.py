"""Typed loader for ~/.ai-phone/config.yaml — Milestone 2 schema (architecture.md §8, §14a).

sip:        transports and NAT for the whole daemon
lines:      one entry per line (trunk or sip); each is a pjsua2 Account
inbound:    what the agent says and does when someone calls and no AI is listening
calls:      outbound limits and relay-mode timing
brain:      a model on this Mac, an OpenAI-compatible endpoint, or the person's own Claude / ChatGPT plan
tts:        the voice ai-phone speaks with, and how fast
"""
from __future__ import annotations

import re
import typing
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, Mapping
from urllib.parse import urlsplit

import yaml

from ai_phone import paths


class ConfigError(ValueError):
    """A config file problem; the message names the offending key and the fix."""


LINE_ID_RE = re.compile(r"^[a-z0-9-]{1,32}$")
USER_RE = re.compile(r"^[A-Za-z0-9]{4,32}$")          # trunk credentials: what Telnyx and Twilio accept
SIP_USER_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")     # sip lines: the user part of an AOR
E164_RE = re.compile(r"^\+[1-9][0-9]{6,14}$")
ENV_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
CLI_RE = re.compile(r"^[A-Za-z0-9._~/-]{1,256}$")     # brain.cli: a name to look for on PATH, or a path to one
MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,128}$")
_HOST = r"[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?"
HOST_RE = re.compile(rf"^{_HOST}(?::[0-9]{{1,5}})?$")
SIP_URI_RE = re.compile(rf"^sip:{_HOST}(?::[0-9]{{1,5}})?(?:;transport=(?:udp|tcp|tls))?$")
AOR_RE = re.compile(rf"^sip:[A-Za-z0-9._-]{{1,64}}@{_HOST}(?::[0-9]{{1,5}})?$")
DISPLAY_RE = re.compile(r"^[\x20\x21\x23-\x5b\x5d-\x7e]{1,64}$")  # printable ASCII without '"' and '\'
CIDR_RE = re.compile(r"^[0-9a-fA-F:.]+(/[0-9]{1,3})?$")
DNS_LABEL_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?")
TLS_PROXY_RE = re.compile(r"sip:(?P<host>[A-Za-z0-9.-]+)(?::(?P<port>[0-9]{1,5}))?;transport=tls")
TtsEngine = Literal["kokoro", "phonon"]
TTS_ENGINES = typing.get_args(TtsEngine)


@dataclass
class BindConfig:
    address: str = "0.0.0.0"
    udp_port: int = 5060
    tcp_port: int = 5060
    tls_port: int = 5061


@dataclass
class TlsConfig:
    cert_file: str | None = None
    key_file: str | None = None
    ca_file: str | None = None


@dataclass
class SipConfig:
    bind: BindConfig = field(default_factory=BindConfig)
    tls: TlsConfig = field(default_factory=TlsConfig)
    public_addr: str | None = None
    stun_server: str | None = None
    nameservers: list[str] = field(default_factory=list)
    tcp_keepalive_sec: int = 30


@dataclass
class RtpConfig:
    port_min: int = 10000
    port_max: int = 10100


@dataclass
class ModelsConfig:
    tier: str = "auto"  # auto, a key from ai_phone.models.CATALOG, or a Hub repo id someone named by hand
    home: str = field(default_factory=lambda: str(paths.home() / "models"))
    wire_memory: bool = True  # keep model weights resident (macOS 15+); off = the pager may evict them while idle
    keep_warm_sec: int = 30  # idle workers run one tiny inference this often so the pager evicts other things first; 0 = off


@dataclass
class ProviderRef:
    name: str = ""
    state: Literal["pending", "ready", "failed", "rollback_failed"] = "pending"
    resources: dict[str, str] = field(default_factory=dict)


@dataclass
class VerifiedCallerId:
    number: str = ""
    provider: str = ""
    verified_at: str = ""


@dataclass
class LineConfig:
    kind: Literal["trunk", "sip"] = "trunk"
    number: str | None = None
    aor: str | None = None
    display_name: str = "ai-phone"
    registrar: str | None = None
    outbound_proxy: str | None = None
    dial_domain: str | None = None
    register_via_proxy: bool = True
    transport: Literal["udp", "tcp", "tls"] = "udp"
    tls_hostname_policy: Literal["strict", "verified_wildcard"] = "strict"
    user: str = "aiphone"
    password_env: str | None = None
    register_expiry_sec: int = 300
    srtp: Literal["off", "optional", "required"] = "off"
    accept_invites_from: Literal["registered_trunk", "trusted", "lan", "any"] = "registered_trunk"
    trusted_cidrs: list[str] = field(default_factory=list)
    trusted_peers: list[str] = field(default_factory=list)
    caller_id: str | None = None
    from_user: Literal["number", "account"] = "number"
    diverted_from: list[str] = field(default_factory=list)
    verified_caller_ids: list[VerifiedCallerId] = field(default_factory=list)
    provider: ProviderRef | None = None


@dataclass
class InboundConfig:
    greeting: str = "Hi, you've reached an AI assistant. How can I help?"
    instructions: str = ("Someone has called this number. Find out who they are and what they need, and offer to take "
                         "a message: their name, their number and what it is about.")


@dataclass
class CallsConfig:
    daily_limit: int = 0                 # outbound calls per local day; 0 = no limit
    relay_filler_after_ms: int = 1500
    relay_second_filler_sec: int = 10
    relay_reply_timeout_sec: int = 25
    allow_inbound: bool = True
    allow_outbound: bool = True


# What a subscription brain runs when the config names no model: Claude Code's fastest alias, and the fastest model
# a ChatGPT plan lists (measured; see README "Any brain").
SUBSCRIPTION_MODELS = {"claude": "haiku", "chatgpt": "gpt-5.6-luna"}
SUBSCRIPTION_KINDS = tuple(SUBSCRIPTION_MODELS)
# Every brain a config may name, in one place: the annotation below and `BRAIN_KINDS` are the same object read two ways,
# so the tuple cannot drift from the type the loader validates against. Anything that reads brain.kind out of a config
# too broken to load (models pull's recovery) takes the kinds from here, and a fifth one is never read as the default.
BrainKind = Literal["local", "openai_compatible", "claude", "chatgpt"]
BRAIN_KINDS = typing.get_args(BrainKind)


@dataclass
class BrainConfig:
    kind: BrainKind = "local"
    base_url: str = "http://127.0.0.1:1234/v1"   # openai_compatible only; ignored by the other kinds
    model: str = "local"
    api_key_env: str | None = None               # openai_compatible only; a subscription brain needs no key
    cli: str | None = None                       # claude / chatgpt: where the CLI is; null = look for it on PATH
    connect_timeout_ms: int = 2000
    # a subscription brain's first word takes seconds (its own CLI, then the provider), so the daemon raises this to
    # 12000 for those kinds while it is still the default below; set it yourself and your number is the one used
    first_token_timeout_ms: int = 4000
    stream_idle_timeout_ms: int = 4000
    max_response_chars: int = 2000


@dataclass
class TtsConfig:
    voice: str = "af_heart"  # an English Kokoro voice: af_/am_ American, bf_/bm_ British
    speed: float = 1.0
    engine: TtsEngine = "kokoro"


@dataclass
class ConversationConfig:
    endpoint_silence_ms: int = 300
    bargein_min_speech_ms: int = 250
    bargein_playback_gain: float = 3.0
    history_turns: int = 6
    max_tokens: int = 120
    silence_prompt_sec: int = 8
    silence_hangup_sec: int = 8
    max_call_sec: int = 600
    rtp_timeout_sec: int = 10
    reply_timeout_sec: int = 12  # no audible reply this long after the caller stops = the brain has stalled: apologise


@dataclass
class LogConfig:
    level: str = "info"
    pjsip_level: int = 3
    dir: str = "~/Library/Logs/ai-phone"


@dataclass
class Config:
    sip: SipConfig = field(default_factory=SipConfig)
    rtp: RtpConfig = field(default_factory=RtpConfig)
    codecs: list[str] = field(default_factory=lambda: ["PCMU", "PCMA"])
    models: ModelsConfig = field(default_factory=ModelsConfig)
    lines: dict[str, LineConfig] = field(default_factory=dict)
    inbound: InboundConfig = field(default_factory=InboundConfig)
    calls: CallsConfig = field(default_factory=CallsConfig)
    brain: BrainConfig = field(default_factory=BrainConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    conversation: ConversationConfig = field(default_factory=ConversationConfig)
    log: LogConfig = field(default_factory=LogConfig)
    warnings: list[str] = field(default_factory=list, init=False, repr=False)


# ---- structural loading ---------------------------------------------------------------------

def _coerce(value: Any, hint: Any, path: str) -> Any:
    origin = typing.get_origin(hint)
    if isinstance(hint, type) and is_dataclass(hint):
        return _build(hint, value, path)
    if origin is Literal:
        allowed = typing.get_args(hint)
        if value not in allowed:
            raise ConfigError(f"{path}: must be one of {list(allowed)}")
        return value
    if origin is typing.Union or (origin is not None and origin.__name__ == "UnionType"):
        if value is None and type(None) in typing.get_args(hint):
            return None  # `provider: null` stays None instead of becoming a default ProviderRef()
        errors = []
        for member in typing.get_args(hint):
            try:
                return _coerce(value, member, path)
            except ConfigError as e:
                errors.append(str(e))
        # report the member that matches the value's shape (a mapping → the dataclass), not the trailing "expected null"
        raise ConfigError(errors[0] if isinstance(value, (dict, list)) else f"{path}: {errors[-1]}")
    if origin is list:
        if not isinstance(value, list):
            raise ConfigError(f"{path}: expected a list")
        (elem,) = typing.get_args(hint) or (Any,)
        return [_coerce(v, elem, f"{path}[{i}]") for i, v in enumerate(value)]
    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a mapping")
        _kt, vt = typing.get_args(hint) or (Any, Any)
        out = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise ConfigError(f"{path}: keys must be strings, got {k!r}")
            out[k] = _coerce(v, vt, f"{path}.{k}")
        return out
    if hint is type(None):
        if value is not None:
            raise ConfigError(f"{path}: expected null")
        return None
    if hint is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected true or false")
        return value
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: expected an integer, got {value!r}")
        return value
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected a number, got {value!r}")
        return value
    if hint is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected a string, got {value!r}")
        return value
    return value


def _build(cls: type, data: Any, path: str) -> Any:
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping")
    hints = typing.get_type_hints(cls)
    known = {f.name: f for f in fields(cls) if f.init}
    unknown = set(data) - set(known)
    if unknown:
        raise ConfigError(f"{path}: unknown key(s): {', '.join(sorted(unknown))}")
    kwargs = {name: _coerce(data[name], hints[name], f"{path}.{name}") for name in known if name in data}
    return cls(**kwargs)


# ---- legacy (Milestone 1) migration ---------------------------------------------------------

_LEGACY_SIP_KEYS = {"user", "domain", "password", "password_env", "trusted_peers", "accept_invites_from"}


def _migrate_legacy(raw: dict) -> tuple[dict, list[str]]:
    raw = dict(raw)
    warnings: list[str] = []
    sip = raw.get("sip") or {}
    if not isinstance(sip, dict):
        raise ConfigError("config.sip: expected a mapping")
    sip = dict(sip)
    bind = sip.get("bind") or {}
    if not isinstance(bind, dict):
        raise ConfigError("config.sip.bind: expected a mapping")
    bind = dict(bind)
    if _LEGACY_SIP_KEYS & set(sip) or "addr" in bind or "port" in bind:
        if sip.get("password") is not None:
            raise ConfigError("config.sip.password: inline passwords are no longer accepted; use password_env")
        warnings.append("config.sip: Milestone 1 keys found; mapped to sip.bind and lines.main. "
                        "Update your config: this mapping is removed in the next release.")
        user = sip.get("user", "aiphone")
        mode = sip.get("accept_invites_from", "registered_trunk")
        line: dict[str, Any] = {"user": user, "trusted_peers": list(sip.get("trusted_peers", []))}
        if sip.get("domain"):
            line.update({"kind": "trunk", "registrar": f"sip:{sip['domain']}", "dial_domain": sip["domain"],
                         "accept_invites_from": mode})
            if sip.get("password_env"):
                line["password_env"] = sip["password_env"]
        else:
            host = bind.get("addr", "0.0.0.0")
            host = "127.0.0.1" if host == "0.0.0.0" else host
            if mode == "registered_trunk":
                mode = "trusted" if line["trusted_peers"] else "lan"
            line.update({"kind": "sip", "aor": f"sip:{user}@{host}", "accept_invites_from": mode})
        port = bind.get("port", 5060)
        if isinstance(port, bool) or not isinstance(port, int):
            raise ConfigError(f"config.sip.bind.port: expected an integer, got {port!r}")
        if not isinstance(bind.get("addr", "0.0.0.0"), str) or not isinstance(user, str):
            raise ConfigError("config.sip: user and bind.addr must be strings")
        new_sip = {k: v for k, v in sip.items() if k not in _LEGACY_SIP_KEYS and k != "bind"}
        new_sip["bind"] = {"address": bind.get("addr", "0.0.0.0"), "udp_port": port, "tcp_port": port,
                           "tls_port": 5061 if port != 5061 else 5062}
        raw["sip"] = new_sip
        lines = dict(raw.get("lines") or {})
        lines.setdefault("main", line)
        raw["lines"] = lines
    conv = raw.get("conversation") or {}
    if not isinstance(conv, dict):
        raise ConfigError("config.conversation: expected a mapping")
    conv = dict(conv)
    if "greeting" in conv or "system_prompt" in conv:
        inbound = _inbound_of(raw)
        if "greeting" in conv:
            inbound.setdefault("greeting", conv.pop("greeting"))
        if "system_prompt" in conv:
            inbound.setdefault("instructions", conv.pop("system_prompt"))
        raw["inbound"], raw["conversation"] = inbound, conv
        warnings.append("config.conversation: greeting/system_prompt moved to inbound")
    raw, more = _migrate_personas(raw)
    return raw, warnings + more


def _inbound_of(raw: dict) -> dict:
    inbound = raw.get("inbound") or {}
    if not isinstance(inbound, dict):
        raise ConfigError("config.inbound: expected a mapping")
    return dict(inbound)


def _migrate_personas(raw: dict) -> tuple[dict, list[str]]:
    """Personas were replaced by one `inbound:` section (outbound calls carry their own prompt)."""
    lines = raw.get("lines") if isinstance(raw.get("lines"), dict) else {}   # a malformed `lines` fails later, in _build
    line_personas = [ln.get("persona") for ln in lines.values() if isinstance(ln, dict) and ln.get("persona")]
    if "personas" not in raw and not line_personas:
        return raw, []
    raw = dict(raw)
    personas = raw.pop("personas", None) or {}
    if not isinstance(personas, dict) or not all(v is None or isinstance(v, dict) for v in personas.values()):
        raise ConfigError("config.personas: expected a mapping of persona names to mappings")
    first = next(iter(lines.values()), None)
    name = first.get("persona", "default") if isinstance(first, dict) else "default"   # the first line's persona
    name = name if isinstance(name, str) and name in personas else "default"
    chosen = personas.get(name) or personas.get("default") or {}
    inbound = _inbound_of(raw)
    if "greeting" in chosen:
        inbound.setdefault("greeting", chosen["greeting"])
    if "system_prompt" in chosen:
        inbound.setdefault("instructions", chosen["system_prompt"])
    raw["inbound"] = inbound
    if lines:
        raw["lines"] = {lid: ({k: v for k, v in ln.items() if k != "persona"} if isinstance(ln, dict) else ln)
                        for lid, ln in lines.items()}
    dropped = sorted(n for n in personas if n != name)
    if "outbound_opening" in chosen:
        dropped.append(f"{name}.outbound_opening")
    note = f"; dropped: {', '.join(dropped)}" if dropped else ""
    return raw, [f"config.personas: replaced by inbound (greeting and instructions taken from persona {name!r}){note}; "
                 f"line persona settings are ignored and outbound calls carry their own --prompt"]


# ---- semantic validation --------------------------------------------------------------------

def host_of(uri: str) -> str:
    """'sip:host:port;transport=tls' -> 'host'. Grammar is validated before this is called."""
    body = uri.split(":", 1)[1]
    body = body.split(";", 1)[0]
    return body.rsplit("@", 1)[-1].split(":", 1)[0]


def line_password_env(line_id: str) -> str:
    return f"AI_PHONE_LINE_{line_id.upper().replace('-', '_')}_PASSWORD"


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


def tls_proxy_hostname(line: LineConfig, path: str = "line") -> str | None:
    """Validate the explicit TLS policy; return its fixed proxy DNS name without mutating the line.

    Shared with runtime validation because callers may construct line dataclasses directly.
    Strict lines retain the existing transport behavior and need no fixed proxy identity.
    """
    policy = line.tls_hostname_policy
    policy_path = f"{path}.tls_hostname_policy"
    _require(isinstance(policy, str) and policy in ("strict", "verified_wildcard"),
             f"{policy_path}: expected strict or verified_wildcard")
    if policy == "strict":
        return None
    _require(line.kind == "trunk" and line.transport == "tls",
             f"{policy_path}: verified_wildcard requires kind: trunk and transport: tls")
    _require(line.register_via_proxy is True,
             f"{policy_path}: verified_wildcard requires register_via_proxy: true")
    proxy = line.outbound_proxy
    match = TLS_PROXY_RE.fullmatch(proxy) if isinstance(proxy, str) else None
    _require(match is not None,
             f"{path}.outbound_proxy: verified_wildcard requires sip:DNS-host[:port];transport=tls")
    host, port = match.group("host"), match.group("port")
    _require(port is None or 1 <= int(port) <= 65535,
             f"{path}.outbound_proxy: port must be between 1 and 65535")
    # Numeric/dotted numeric forms include ordinary IP literals and legacy short IPv4 forms.
    _require(len(host) <= 253 and not host.replace(".", "").isdigit()
             and all(DNS_LABEL_RE.fullmatch(label) for label in host.split(".")),
             f"{path}.outbound_proxy: expected an ASCII DNS name with labels of 1 to 63 characters; no IP or wildcard")
    return host.lower()


def validate_config(cfg: Config, env: Mapping[str, str] | None = None) -> list[str]:
    """Hard errors raise ConfigError; soft problems (a line whose password is not set) come back as warnings."""
    import os
    env = os.environ if env is None else env
    warnings: list[str] = []
    _require(cfg.rtp.port_min < cfg.rtp.port_max, "config.rtp: port_min must be less than port_max")
    for c in cfg.codecs:
        _require(c in ("PCMU", "PCMA"), f"config.codecs: only PCMU and PCMA are allowed, got {c}")
    b = cfg.sip.bind
    _require(b.tcp_port != b.tls_port, "config.sip.bind.tls_port: TLS cannot share the TCP port")
    for ns in cfg.sip.nameservers:
        _require(bool(HOST_RE.fullmatch(ns)), f"config.sip.nameservers: {ns!r} is not host[:port]")
    if cfg.sip.stun_server is not None:
        _require(bool(HOST_RE.fullmatch(cfg.sip.stun_server)), "config.sip.stun_server: expected host[:port]")
    c = cfg.calls
    _require(c.allow_inbound or c.allow_outbound,
             "config.calls: at least one of allow_inbound or allow_outbound must be true")
    _require(c.daily_limit >= 0, "config.calls.daily_limit: must be 0 (no limit) or more")
    _require(c.relay_filler_after_ms > 0, "config.calls.relay_filler_after_ms: must be more than 0")
    _require(c.relay_second_filler_sec * 1000 > c.relay_filler_after_ms,
             "config.calls.relay_second_filler_sec: must come after relay_filler_after_ms")
    _require(c.relay_reply_timeout_sec > c.relay_second_filler_sec,
             "config.calls.relay_reply_timeout_sec: must be longer than relay_second_filler_sec")
    from ai_phone import models  # here, not at import: `config` is loaded by every command
    _require(models.known_tier(cfg.models.tier), f"config.models.tier: must be {models.tier_help()}")
    _require(0 < len(cfg.inbound.greeting.strip()) <= 500, "config.inbound.greeting: 1 to 500 characters")
    _require(len(cfg.inbound.instructions) <= 4000, "config.inbound.instructions: at most 4000 characters")
    users_seen: dict[str, str] = {}
    uses_tls = False
    for lid, ln in cfg.lines.items():
        p = f"config.lines.{lid}"
        _require(bool(LINE_ID_RE.fullmatch(lid)), f"{p}: line ids must match {LINE_ID_RE.pattern}")
        tls_proxy_hostname(ln, p)
        _require(ln.from_user in ("number", "account"), f"{p}.from_user: expected number or account")
        _require(ln.kind == "trunk" or ln.from_user == "number", f"{p}.from_user: account is only valid on a trunk line")
        if ln.kind == "trunk":
            _require(bool(USER_RE.fullmatch(ln.user)), f"{p}.user: must match {USER_RE.pattern} (letters and digits, 4 to 32)")
        else:
            _require(bool(SIP_USER_RE.fullmatch(ln.user)), f"{p}.user: must match {SIP_USER_RE.pattern}")
        _require(bool(DISPLAY_RE.fullmatch(ln.display_name)), f"{p}.display_name: printable ASCII, 1 to 64 characters, no quotes or backslashes")
        _require(ln.srtp == "off" or ln.transport == "tls", f"{p}.srtp: SRTP needs transport: tls")
        if ln.password_env is not None:
            _require(bool(ENV_RE.fullmatch(ln.password_env)), f"{p}.password_env: must match {ENV_RE.pattern}")
        for c in ln.trusted_cidrs:
            _require(bool(CIDR_RE.fullmatch(c)), f"{p}.trusted_cidrs: {c!r} is not an IP or CIDR")
        for n in ln.diverted_from:
            _require(bool(E164_RE.fullmatch(n)), f"{p}.diverted_from: {n!r} is not E.164")
        for v in ln.verified_caller_ids:
            _require(bool(E164_RE.fullmatch(v.number)), f"{p}.verified_caller_ids: {v.number!r} is not E.164")
        if ln.accept_invites_from == "trusted":
            _require(bool(ln.trusted_peers), f"{p}.trusted_peers: accept_invites_from: trusted needs at least one peer")
        if cfg.calls.allow_inbound and ln.accept_invites_from == "any":
            warnings.append(f"{p}.accept_invites_from is 'any': every INVITE from anywhere on the internet will be answered")
        uses_tls = uses_tls or ln.transport == "tls"
        if ln.kind == "trunk":
            _require(ln.registrar is not None and bool(SIP_URI_RE.fullmatch(ln.registrar)), f"{p}.registrar: expected sip:host[:port][;transport=…]")
            _require(ln.dial_domain is not None and bool(HOST_RE.fullmatch(ln.dial_domain)), f"{p}.dial_domain: expected host[:port]")
            if ln.outbound_proxy is not None:
                _require(bool(SIP_URI_RE.fullmatch(ln.outbound_proxy)), f"{p}.outbound_proxy: expected sip:host[:port]")
            if ln.number is not None:
                _require(bool(E164_RE.fullmatch(ln.number)), f"{p}.number: expected E.164 like +442079460000")
            if ln.caller_id is not None:
                _require(bool(E164_RE.fullmatch(ln.caller_id)), f"{p}.caller_id: expected E.164 on a trunk line")
            _require(ln.accept_invites_from != "trusted" or bool(ln.trusted_peers), f"{p}.trusted_peers: required")
            if ln.accept_invites_from == "registered_trunk" and not ln.trusted_cidrs:
                warnings.append(f"{p}.trusted_cidrs is empty: only the registrar's resolved addresses will be accepted")
            _require(ln.user not in users_seen, f"{p}.user: {ln.user!r} is not unique (also used by line {users_seen.get(ln.user)!r})")
            users_seen[ln.user] = lid
            env_name = ln.password_env or line_password_env(lid)
            if not env.get(env_name):
                warnings.append(f"{p}: password variable {env_name} is not set; this line will be skipped at start")
        else:
            _require(ln.aor is not None and bool(AOR_RE.fullmatch(ln.aor)), f"{p}.aor: expected sip:user@host[:port]")
            _require(ln.accept_invites_from != "registered_trunk", f"{p}.accept_invites_from: a sip line cannot use registered_trunk")
            if ln.caller_id is not None:
                _require(bool(AOR_RE.fullmatch(ln.caller_id)), f"{p}.caller_id: expected a sip: URI on a sip line")
            if ln.transport == "tls":
                _require(bool(cfg.sip.tls.cert_file and cfg.sip.tls.key_file), f"{p}.transport: tls on a sip line needs sip.tls.cert_file and key_file")
    if uses_tls:
        for f_ in (cfg.sip.tls.cert_file, cfg.sip.tls.key_file, cfg.sip.tls.ca_file):
            if f_ is not None:
                _require(Path(f_).expanduser().exists(), f"config.sip.tls: file not found: {f_}")
    br = cfg.brain
    if br.kind == "openai_compatible":
        try:
            u = urlsplit(br.base_url)
            host = u.hostname
        except ValueError:
            raise ConfigError("config.brain.base_url: not a valid URL") from None
        _require(u.scheme in ("http", "https") and bool(host), "config.brain.base_url: expected http(s)://host[:port]/…")
        loopback = host in ("127.0.0.1", "localhost", "::1")
        _require(loopback or u.scheme == "https", "config.brain.base_url: a brain that is not on this machine must use https")
        if br.api_key_env is not None:
            _require(bool(ENV_RE.fullmatch(br.api_key_env)), "config.brain.api_key_env: must match " + ENV_RE.pattern)
    if br.kind in SUBSCRIPTION_KINDS:
        _require(bool(MODEL_RE.fullmatch(br.model)), f"config.brain.model: {br.model!r} is not a model name")
        if br.cli is not None:
            _require(bool(CLI_RE.fullmatch(br.cli)), "config.brain.cli: expected a command name or a path to one")
    from ai_phone import voices
    t = cfg.tts
    if t.engine == "phonon":
        _require(t.voice == voices.PHONON_DEFAULT, "config.tts.voice: Phonon uses the bundled voice 'default'")
        _require(t.speed == 1.0, "config.tts.speed: Phonon supports only 1.0")
    else:
        _require(bool(voices.VOICE_RE.fullmatch(t.voice)), f"config.tts.voice: {t.voice!r} is not an English voice (like af_heart or bm_george)")
        _require(0.5 <= t.speed <= 2.0, "config.tts.speed: must be between 0.5 and 2.0")
    if t.engine == "kokoro" and t.voice != voices.DEFAULT:  # only Kokoro has a choice of installed voices
        installed = voices.installed_voices()
        if not installed:
            warnings.append(f"config.tts.voice: {t.voice} cannot be checked until the speech model is downloaded (ai-phone models pull)")
        else:
            _require(t.voice in installed, f"config.tts.voice: {t.voice} is not installed; installed: {', '.join(installed)}")
    return warnings


def fill_defaults(cfg: Config) -> None:
    """Defaults that depend on another field, filled in as the config loads (validation only reads; it never writes).
    A subscription brain's `model` means something different per kind, so the placeholder becomes that kind's own."""
    b = cfg.brain
    if b.kind in SUBSCRIPTION_KINDS and b.model == BrainConfig().model:
        b.model = SUBSCRIPTION_MODELS[b.kind]


def load_config(path: Path, env: Mapping[str, str] | None = None) -> Config:
    """`env` is where line passwords are looked up for the warnings (the daemon passes os.environ merged with secrets.env)."""
    p = Path(path).expanduser()
    if not p.exists():
        raise ConfigError(f"config: {p} does not exist")
    try:
        raw = yaml.safe_load(p.read_text()) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"config: {p} is not valid YAML: {e}") from None
    if not isinstance(raw, dict):
        raise ConfigError("config: expected a mapping at the top level")
    raw, warnings = _migrate_legacy(raw)
    cfg = _build(Config, raw, "config")
    fill_defaults(cfg)
    cfg.warnings = warnings + validate_config(cfg, env)
    return cfg

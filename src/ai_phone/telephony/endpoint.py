"""PJSIP endpoint lifecycle: transports, null audio device, configured codecs, one Account per line."""
from __future__ import annotations

import asyncio
import logging
import threading
import time
import re
import weakref
from dataclasses import dataclass, field

from ai_phone.config import AOR_RE, DISPLAY_RE, E164_RE, SIP_URI_RE, SIP_USER_RE, Config, LineConfig, host_of, tls_proxy_hostname
from ai_phone.telephony import events as ev
from ai_phone.telephony.call import AiCall, LineAccount
from ai_phone.telephony.media_port import ReceiveMediaPort, RingMediaPort
from ai_phone.telephony.pj import pj
from ai_phone.telephony.policy import PeerSet, build_peer_set
from ai_phone.telephony.rings import FrameRing

log = logging.getLogger("ai-phone.telephony")

# pjsip/sip_errno.h; this error constant is not exposed by the pjsua2 SWIG binding.
_SESSION_TERMINATED = 171140


@dataclass
class LineSpec:
    id: str
    kind: str
    user: str
    password: str | None
    display_name: str
    number: str | None
    aor: str | None
    registrar: str | None
    outbound_proxy: str | None
    dial_domain: str | None
    register_via_proxy: bool
    transport: str
    register_expiry_sec: int
    srtp: str
    accept_invites_from: str
    trusted_cidrs: list[str] = field(default_factory=list)
    trusted_peers: list[str] = field(default_factory=list)
    caller_id: str | None = None
    verified_caller_ids: list[str] = field(default_factory=list)
    from_user: str = "number"
    tls_hostname_policy: str = "strict"

    @property
    def registrar_host(self) -> str | None:
        return host_of(self.registrar) if self.registrar else None

    @property
    def aor_host(self) -> str:
        return host_of(self.aor) if self.aor else "127.0.0.1"

    @classmethod
    def from_config(cls, line_id: str, c: LineConfig, password: str | None) -> "LineSpec":
        """Builds the spec and re-checks every part that ends up inside a SIP header, so a LineConfig built in code
        (not through load_config) can never smuggle quotes or delimiters into a name-addr."""
        _grammar(c)
        return cls(id=line_id, kind=c.kind, user=c.user, password=password, display_name=c.display_name, number=c.number,
                   aor=c.aor, registrar=c.registrar, outbound_proxy=c.outbound_proxy, dial_domain=c.dial_domain,
                   register_via_proxy=c.register_via_proxy, transport=c.transport, register_expiry_sec=c.register_expiry_sec,
                   srtp=c.srtp, accept_invites_from=c.accept_invites_from, trusted_cidrs=list(c.trusted_cidrs),
                   trusted_peers=list(c.trusted_peers), caller_id=c.caller_id,
                   verified_caller_ids=[v.number for v in c.verified_caller_ids], from_user=c.from_user,
                   tls_hostname_policy=c.tls_hostname_policy)


def _grammar(c: LineConfig) -> None:
    tls_proxy_hostname(c)
    if c.from_user not in ("number", "account"):
        raise ValueError("line from_user must be number or account")
    if c.kind != "trunk" and c.from_user != "number":
        raise ValueError("line from_user: account is only valid on a trunk line")
    checks = [(DISPLAY_RE, c.display_name, "display_name"), (SIP_USER_RE, c.user, "user")]
    if c.kind == "trunk":
        checks.append((SIP_URI_RE, c.registrar or "", "registrar"))
        if c.number is not None:
            checks.append((E164_RE, c.number, "number"))
        if c.caller_id is not None:
            checks.append((E164_RE, c.caller_id, "caller_id"))
        if c.outbound_proxy is not None:
            checks.append((SIP_URI_RE, c.outbound_proxy, "outbound_proxy"))
    else:
        checks.append((AOR_RE, c.aor or "", "aor"))
        if c.caller_id is not None:
            checks.append((AOR_RE, c.caller_id, "caller_id"))
    for rx, value, name in checks:
        if not rx.fullmatch(value):  # fullmatch: `$` alone would accept a trailing newline
            raise ValueError(f"line {name} {value!r} does not match {rx.pattern}")


def id_uri(spec: LineSpec) -> str:
    host = spec.registrar_host if spec.kind == "trunk" else spec.aor_host
    user = spec.user if spec.kind == "trunk" else spec.aor.split(":", 1)[1].split("@", 1)[0]
    return f'"{spec.display_name}" <sip:{user}@{host}>'


def local_uri(spec: LineSpec) -> str:
    """Per-call From: the trunk's selected identity, or a sip line's caller_id URI verbatim."""
    if spec.kind == "trunk":
        user = spec.user if spec.from_user == "account" else spec.caller_id or spec.number
        return f'"{spec.display_name}" <sip:{user}@{spec.registrar_host}>'
    return spec.caller_id or spec.aor


_URI_RE = re.compile(r"sips?:(?:(?P<user>[^@\s;>]+)@)?(?P<host>\[[^\]]+\]|[^\s;:>]+)", re.I)
_TO_RE = re.compile(r"^(?:To|t):\s*(?P<v>.*)$", re.M | re.I)


def _user_host(fragment: str) -> tuple[str, str] | None:
    m = _URI_RE.search(fragment)
    return (m.group("user") or "", m.group("host").strip("[]").lower()) if m else None


def request_target(msg: str) -> tuple[str, str] | None:
    """(user, host) the request is addressed to: the request URI, else the To header."""
    first = msg.split("\n", 1)[0]
    target = _user_host(first) if first.upper().startswith(("INVITE", "OPTIONS", "MESSAGE", "SUBSCRIBE", "NOTIFY", "REFER", "INFO", "UPDATE")) else None
    if target is None:
        m = _TO_RE.search(msg)
        target = _user_host(m.group("v")) if m else None
    return target


def line_for_request(specs: list[LineSpec], peers: dict[str, PeerSet], msg: str, source_ip: str) -> str | None:
    """Which line an incoming request belongs to. PJSIP scores accounts by transport, domain and user, and with a
    registered trunk next to a sip line a LAN INVITE for `sip:ai@127.0.0.1` can land on the trunk (whose policy then
    rejects it). We look at what the request asks for and where it comes from:
      1. a sip line whose AOR user and host match the target;
      2. a trunk whose registrar/trusted proxies sent it, or whose SIP user the target names;
      3. a sip line whose AOR user matches (the host may be any address of this machine).
    None = keep PJSIP's choice."""
    target = request_target(msg)
    if target:
        for s in specs:
            if s.kind == "sip" and s.aor and (aor_user(s), (s.aor_host or "").lower()) == target:
                return s.id
    for s in specs:
        if s.kind == "trunk":
            if s.accept_invites_from == "registered_trunk" and s.id in peers and peers[s.id].contains(source_ip):
                return s.id
            if target and s.user and target[0] == s.user:
                return s.id
    if target:
        for s in specs:
            if s.kind == "sip" and s.aor and aor_user(s) == target[0]:
                return s.id
    return None


def aor_user(spec: LineSpec) -> str:
    return (spec.aor or "").split(":", 1)[-1].split("@", 1)[0]


class AiEndpoint(pj.Endpoint):
    """pjsua2 endpoint that picks the line for incoming requests with `line_for_request` (see there)."""

    def __init__(self, telephony: "Telephony"):
        super().__init__()
        self.telephony = telephony

    def onSelectAccount(self, prm) -> None:  # PJSIP thread, before onIncomingCall
        try:
            t = self.telephony
            src = str(prm.rdata.srcAddress).rsplit(":", 1)[0].strip("[]")
            line_id = line_for_request(t.lines, {i: a.peers for i, a in t.accounts.items()}, str(prm.rdata.wholeMsg), src)
            acc = t.accounts.get(line_id) if line_id else None
            if acc is not None and acc.getId() != prm.accountIndex:
                log.info("incoming request for line %s (PJSIP had picked account %d)", line_id, prm.accountIndex)
                prm.accountIndex = acc.getId()
        except Exception as e:  # never let a parse error take the PJSIP thread down; PJSIP's own choice stands
            log.warning("account selection failed (%s); keeping PJSIP's choice", e)


def srtp_values(spec: LineSpec) -> tuple[int, int]:
    use = {"off": pj.PJMEDIA_SRTP_DISABLED, "optional": pj.PJMEDIA_SRTP_OPTIONAL, "required": pj.PJMEDIA_SRTP_MANDATORY}[spec.srtp]
    return use, (0 if spec.srtp == "off" else 1)


class BusyError(RuntimeError):
    pass


class DialError(ValueError):
    pass


SIP_TARGET_RE = re.compile(r"^sip:[A-Za-z0-9._+-]{1,64}@[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?(?::[0-9]{1,5})?$")


def request_uri(spec: LineSpec, target: str) -> str:
    if target.startswith("sip:"):
        if not SIP_TARGET_RE.fullmatch(target):
            raise DialError(f"target {target!r} is not a plain sip:user@host[:port] address")
        return target
    if not E164_RE.fullmatch(target):
        raise DialError(f"target {target!r} is neither a sip: URI nor an E.164 number like +447700900123")
    if spec.kind != "trunk" or not spec.dial_domain:
        raise DialError(f"line {spec.id} cannot dial phone numbers (no dial_domain)")
    return f"sip:{target}@{spec.dial_domain}"


def check_caller_id(spec: LineSpec) -> None:
    if spec.kind != "trunk" or spec.from_user == "account":
        return
    cid = spec.caller_id or spec.number
    if cid is None:
        raise DialError(f"line {spec.id} has neither a number nor a caller_id; nothing can be presented as From")
    if cid != spec.number and cid not in spec.verified_caller_ids:
        raise DialError(f"caller_id {cid} is neither line {spec.id}'s number nor a verified caller id (run `ai-phone lines verify-caller-id`)")


def _system_resolvers() -> list[str]:
    try:
        return [l.split()[1] for l in open("/etc/resolv.conf") if l.startswith("nameserver")][:4]
    except OSError:
        return []


class Telephony:
    def __init__(self, cfg: Config, lines: list[LineSpec], loop: asyncio.AbstractEventLoop, events: asyncio.Queue,
                 rx: FrameRing, tx: FrameRing, echo_mode: bool = False):
        self.cfg, self.lines, self.loop, self.events, self.rx, self.tx = cfg, lines, loop, events, rx, tx
        self.echo_mode = echo_mode
        self.ep: pj.Endpoint | None = None
        self.port: RingMediaPort | None = None
        self.receive_port: ReceiveMediaPort | RingMediaPort | None = None
        self.accounts: dict[str, LineAccount] = {}
        self.transport_ids: dict[str, int] = {}
        self.line_transport_ids: dict[str, int] = {}
        self._uses_verified_tls = False
        self.current_call: AiCall | None = None
        self.peer_warnings: list[str] = []
        self._call_lock = threading.Lock()  # make_call (asyncio thread) and onIncomingCall (PJSIP thread) both claim current_call
        self._stopping = False  # once stop() begins, nothing may claim a call

    @property
    def current_call_id(self) -> int | None:
        return self.current_call.getId() if self.current_call is not None else None

    def try_claim_call(self, call: AiCall) -> bool:
        """The one-call-at-a-time gate, safe across the asyncio and PJSIP threads. The claim lasts from the INVITE
        (or makeCall) until the daemon calls release_call() after its own teardown, so a call that PJSIP already
        reports as disconnected still blocks the next one while the controller drains and flushes."""
        with self._call_lock:
            if self.current_call is not None or self._stopping:
                return False
            self.current_call = call
            return True

    def release_call(self, call_id: int) -> bool:
        """Called by the daemon when it has finished with the call (rings flushed, controller stopped)."""
        with self._call_lock:
            c = self.current_call
            if c is not None and c.getId() == call_id:
                self._drop_answer_timeout(c)
                self.current_call = None
                return True
            return False

    def line_of(self, call_id: int) -> str | None:
        c = self.current_call
        return c.acc.line_id if c is not None and c.getId() == call_id else None

    # ---- start --------------------------------------------------------------------------------
    def start(self) -> None:
        if self.echo_mode and self.cfg.playback_rate != 16000:
            raise ValueError("the diagnostic echo mode requires 16 kHz playback; remove opus from codecs")
        if "opus" in self.cfg.codecs:
            version = getattr(pj, "ai_phone_opus_version", None)
            if (not callable(version) or version() != 1
                    or not callable(getattr(pj, "ai_phone_enable_opus", None))):
                raise RuntimeError("Opus requires a rebuilt native binding; run ./scripts/install.sh from the ai-phone checkout")
        verified_hosts = {}
        for line in self.lines:
            hostname = tls_proxy_hostname(line, f"line {line.id}")
            if hostname is not None:
                verified_hosts[line.id] = hostname
        if verified_hosts:
            version = getattr(pj, "ai_phone_tls_verifier_version", None)
            if (not callable(version) or version() != 1
                    or not callable(getattr(pj, "ai_phone_create_verified_tls_transport", None))
                    or not callable(getattr(pj, "ai_phone_reset_tls_verifier", None))
                    or not callable(getattr(pj, "ai_phone_deactivate_tls_verifier", None))):
                raise RuntimeError("verified TLS hostname policy requires a rebuilt native binding; "
                                   "run ./scripts/install.sh from the ai-phone checkout")
        ep = AiEndpoint(self)
        ep.libCreate()
        self.ep = ep  # partial transport/account failures must also destroy the native endpoint
        self._stopping = False
        try:
            self._start_endpoint(ep, verified_hosts)
        except BaseException:
            self.stop()
            raise

    def _start_endpoint(self, ep: AiEndpoint, verified_hosts: dict[str, str]) -> None:
        ec = pj.EpConfig()
        ec.uaConfig.threadCnt = 1
        ec.medConfig.clockRate = ec.medConfig.sndClockRate = self.cfg.playback_rate
        ec.medConfig.audioFramePtime = 20
        ec.medConfig.ecTailLen = 0
        ec.medConfig.noVad = True
        ec.logConfig.level = self.cfg.log.pjsip_level
        s = self.cfg.sip
        if s.stun_server and not s.public_addr:
            ec.uaConfig.stunServer.append(s.stun_server)
        for ns in (s.nameservers or _system_resolvers()):
            ec.uaConfig.nameserver.append(ns)
        ep.libInit(ec)
        if "opus" in self.cfg.codecs:
            status = pj.ai_phone_enable_opus()
            if status != 0:
                raise RuntimeError(f"could not initialize Opus (PJSIP status {status})")
        ep.audDevManager().setNullDev()
        pj.ai_phone_set_keepalive(s.tcp_keepalive_sec, s.tcp_keepalive_sec)
        self.transport_ids["udp"] = ep.transportCreate(pj.PJSIP_TRANSPORT_UDP, self._tc(s.bind.udp_port))
        self.transport_ids["tcp"] = ep.transportCreate(pj.PJSIP_TRANSPORT_TCP, self._tc(s.bind.tcp_port))
        if any(l.transport == "tls" and l.id not in verified_hosts for l in self.lines):
            tc = self._tls_config(s.bind.tls_port)
            tc.tlsConfig.verifyServer = True
            self.transport_ids["tls"] = ep.transportCreate(pj.PJSIP_TRANSPORT_TLS, tc)
        for line_id, hostname in verified_hosts.items():
            # Each account selects its own factory. The native handshake callback
            # authenticates this fixed proxy before PJSIP can send queued SIP bytes.
            self._uses_verified_tls = True
            transport_id = pj.ai_phone_create_verified_tls_transport(self._tls_config(0), hostname)
            if transport_id < 0:
                raise RuntimeError(f"line {line_id}: could not create verified TLS transport "
                                   f"(PJSIP status {-transport_id}); no fallback was used")
            self.line_transport_ids[line_id] = transport_id
        ep.libStart()
        self._configure_codecs(ep)
        if self.cfg.playback_rate == 16000:
            self.port = RingMediaPort(self.rx, self.tx, echo=self.echo_mode)
            self.receive_port = self.port
        else:
            self.port = RingMediaPort(None, self.tx, sample_rate=self.cfg.playback_rate)
            self.receive_port = ReceiveMediaPort(self.rx)
        for spec in self.lines:
            peers = self._peers_for(spec)
            acc = LineAccount(self, spec, peers)
            acc.create(self._account_config(spec))
            self.accounts[spec.id] = acc
            log.info("line %s: %s %s transport=%s policy=%s", spec.id, spec.kind, spec.number or spec.aor, spec.transport, spec.accept_invites_from)

    def _configure_codecs(self, ep) -> None:
        prefixes = {"PCMU": "PCMU/8000/", "PCMA": "PCMA/8000/", "G722": "G722/16000/", "opus": "opus/48000/"}
        available = ep.codecEnum2()
        codecs = [str(c.codecId) for c in available]
        priorities = {}
        for index, name in enumerate(self.cfg.codecs):
            matches = [codec for codec in codecs if codec.startswith(prefixes[name])]
            if not matches:
                raise RuntimeError(f"configured codec {name} is unavailable; rebuild with ./scripts/install.sh")
            priorities.update({codec: 254 - index for codec in matches})
        for codec in codecs:
            ep.codecSetPriority(codec, priorities.get(codec, 0))

    def _tls_config(self, port: int) -> "pj.TransportConfig":
        import certifi
        tc = self._tc(port)
        tc.tlsConfig.CaListFile = self.cfg.sip.tls.ca_file or certifi.where()
        tc.tlsConfig.verifyServer = True
        if self.cfg.sip.tls.cert_file:
            tc.tlsConfig.certFile = self.cfg.sip.tls.cert_file
            tc.tlsConfig.privKeyFile = self.cfg.sip.tls.key_file
        return tc

    def _tc(self, port: int) -> "pj.TransportConfig":
        tc = pj.TransportConfig()
        tc.port = port
        tc.boundAddress = self.cfg.sip.bind.address
        if self.cfg.sip.public_addr:
            tc.publicAddress = self.cfg.sip.public_addr
        return tc

    def _peers_for(self, spec: LineSpec) -> PeerSet:
        if spec.accept_invites_from == "registered_trunk":
            entries = list(spec.trusted_cidrs) + ([spec.registrar_host] if spec.registrar_host else [])
        elif spec.accept_invites_from == "trusted":
            entries = list(spec.trusted_peers)
        else:
            entries = []
        peers, warnings = build_peer_set(entries)
        self.peer_warnings.extend(f"line {spec.id}: {w}" for w in warnings)
        return peers

    def _account_config(self, spec: LineSpec) -> "pj.AccountConfig":
        ac = pj.AccountConfig()
        ac.idUri = id_uri(spec)
        ac.sipConfig.transportId = (self.line_transport_ids[spec.id] if spec.id in self.line_transport_ids
                                    else self.transport_ids[spec.transport])
        if spec.kind == "trunk":
            ac.regConfig.registrarUri = spec.registrar
            ac.regConfig.timeoutSec = spec.register_expiry_sec
            ac.regConfig.retryIntervalSec = 30
            ac.regConfig.firstRetryIntervalSec = 5
            ac.regConfig.randomRetryIntervalSec = 10
            ac.regConfig.dropCallsOnFail = False
            ac.regConfig.unregWaitMsec = 1000
            ac.regConfig.registerOnAdd = True
            ac.regConfig.proxyUse = (pj.PJSUA_REG_USE_OUTBOUND_PROXY | pj.PJSUA_REG_USE_ACC_PROXY) if spec.register_via_proxy else 0
            ac.sipConfig.authCreds.append(pj.AuthCredInfo("digest", "*", spec.user, 0, spec.password or ""))
            if spec.outbound_proxy:
                ac.sipConfig.proxies.append(spec.outbound_proxy)
        else:
            ac.regConfig.registerOnAdd = False
        ac.natConfig.contactRewriteUse = 1
        ac.natConfig.viaRewriteUse = 1
        ac.natConfig.sdpNatRewriteUse = 1
        ac.natConfig.udpKaIntervalSec = 15 if spec.transport == "udp" else 0
        ac.natConfig.iceEnabled = False
        ac.natConfig.turnEnabled = False
        ac.mediaConfig.transportConfig.port = self.cfg.rtp.port_min
        ac.mediaConfig.transportConfig.portRange = self.cfg.rtp.port_max - self.cfg.rtp.port_min
        if self.cfg.sip.public_addr:
            ac.mediaConfig.transportConfig.publicAddress = self.cfg.sip.public_addr
        ac.mediaConfig.srtpUse, ac.mediaConfig.srtpSecureSignaling = srtp_values(spec)
        return ac

    # ---- call control -------------------------------------------------------------------------
    def answer(self, call_id: int) -> bool:
        """False when the caller cancelled; the queued ended event still releases the claim."""
        call = self.current_call
        if call is None or call.getId() != call_id or call.ended or not call.isActive():
            return False
        op = pj.CallOpParam()
        op.statusCode = pj.PJSIP_SC_OK
        try:
            call.answer(op)
        except pj.Error as error:
            if error.status != _SESSION_TERMINATED:
                raise  # unrelated failures must not masquerade as a caller cancellation
            # CANCEL can win after isActive(). Handle it here so its SWIG traceback
            # never unwinds the daemon into endpoint destruction while holding Call.
            return False
        return True

    def hangup(self, call_id: int, code: int = 200) -> None:
        if self.current_call is not None and self.current_call.getId() == call_id:
            op = pj.CallOpParam()
            op.statusCode = code
            self.current_call.hangup(op)

    def make_call(self, line_id: str, target: str, timeout_sec: float = 30) -> int:
        if not self.cfg.calls.allow_outbound:
            raise DialError("outbound calls are disabled (calls.allow_outbound is false)")
        acc = self.accounts[line_id]
        spec = acc.spec
        if not acc.registered:
            raise DialError(f"line {line_id} is not registered (degraded); try again once `ai-phone status` shows it registered")
        check_caller_id(spec)
        uri = request_uri(spec, target)
        op = pj.CallOpParam(True)
        op.txOption.localUri = local_uri(spec)
        call = AiCall(acc, pj.PJSUA_INVALID_ID, "out")
        if not self.try_claim_call(call):
            raise BusyError("a call is already active")
        # "dialling" goes on the queue before makeCall, so no PJSIP callback can queue "ringing" ahead of it
        self.events.put_nowait(ev.OutboundProgress(-1, line_id, "dialling", 0, uri))
        try:
            call.makeCall(uri, op)
        except Exception as e:
            with self._call_lock:
                if self.current_call is call:  # never erase a claim that another call made meanwhile
                    self.current_call = None
            # close the sequence that "dialling" opened, so consumers never see a dangling start
            self.events.put_nowait(ev.OutboundProgress(-1, line_id, "rejected", 0, f"could not place the call: {e}"))
            raise DialError(f"could not place the call: {e}") from e
        cid = call.getId()
        call.answer_timeout = self.loop.call_later(timeout_sec, self._cancel_if_unanswered, call)
        log.info("line %s: dialling %s", line_id, uri)
        return cid

    def _cancel_if_unanswered(self, call: AiCall) -> None:
        call.answer_timeout = None
        if self.current_call is call and not call.answered:
            call.cancelled = True
            try:
                call.hangup(pj.CallOpParam())  # early dialog: PJSIP sends CANCEL; the disconnect arrives as 487
            except Exception:
                pass

    @staticmethod
    def _drop_answer_timeout(call: AiCall) -> None:
        """A pending timer would keep the pjsua2 Call alive past libDestroy, which PJSIP aborts on. Per call, so a
        new call's timer is never cancelled by an old call's cleanup."""
        handle = getattr(call, "answer_timeout", None)
        if handle is not None:
            handle.cancel()
            call.answer_timeout = None

    def _call_ended(self, call: AiCall) -> None:
        """PJSIP thread: DISCONNECTED. Only the answer timer is dropped here; release_call() ends the claim."""
        # The loop may drain this callback after stop() destroys PJSIP. Its queue
        # must not keep a native Call alive beyond that destruction.
        self.loop.call_soon_threadsafe(self._drop_ended_timeout, weakref.ref(call))

    @staticmethod
    def _drop_ended_timeout(call_ref: weakref.ReferenceType[AiCall]) -> None:
        call = call_ref()
        if call is not None:
            Telephony._drop_answer_timeout(call)

    def stop(self) -> None:
        if self._uses_verified_tls:
            pj.ai_phone_deactivate_tls_verifier()
        with self._call_lock:
            self._stopping = True  # no new claim from either thread from here on
            call = self.current_call
        if call is not None:
            self._drop_answer_timeout(call)
            try:
                if not call.ended:
                    self.hangup(call.getId())
            except Exception:
                pass
            for _ in range(40):  # up to 2 s for PJSIP to report the release before the library goes away
                if call.ended:
                    break
                time.sleep(0.05)
        call = None  # drop our own reference too: a pjsua2 Call that outlives libDestroy aborts the process
        if self.ep is not None:
            with self._call_lock:
                self.current_call = None
            for acc in self.accounts.values():
                try:
                    acc.shutdown()
                except Exception:
                    pass
            self.accounts = {}
            if self.port is not None or self.receive_port is not None:
                # Stop and join the null-device media clock while its Python callback target
                # and rings are still owned. If this fails, retain them and the endpoint.
                self.ep.audDevManager().setNoDev()
            self.receive_port = None
            self.port = None
            self.ep.libDestroy()
            self.ep = None
            # Slots can be released only after all factories and child sockets are gone.
            if self._uses_verified_tls:
                if not pj.ai_phone_reset_tls_verifier():
                    raise RuntimeError("verified TLS state could not be released after endpoint destruction")
                self._uses_verified_tls = False
            self.transport_ids.clear()
            self.line_transport_ids.clear()

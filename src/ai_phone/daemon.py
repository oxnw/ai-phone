"""Supervisor: spawns the three model workers, starts PJSIP with one account per line, serves the control socket,
and runs one TurnController per call (inbound or outbound)."""
from __future__ import annotations

import asyncio
import logging
import os
import platform
import signal
import socket
import sys
import time
from pathlib import Path

from ai_phone import models, paths
from ai_phone.call_guard import DailyCounter, is_emergency
from ai_phone.calls import CallRecord, CallRegistry, ListenerBusy
from ai_phone.config import SUBSCRIPTION_KINDS, BrainConfig, Config, InboundConfig, LineConfig, line_password_env
from ai_phone.control.protocol import EXIT
from ai_phone.control.server import ControlServer
from ai_phone.pipeline import prompts
from ai_phone.pipeline.replies import ModelReplies, Phrase, RelayPhrases, RelayReplies
from ai_phone.pipeline.timing import TurnLog
from ai_phone.pipeline.turn import TurnController
from ai_phone.pipeline.vad import Endpointer, EnergyVad, SileroVad
from ai_phone.secrets import load_secrets, redact
from ai_phone.services.client import WorkerClient, is_secret_env
from ai_phone.services.ipc import STT_IN_FRAME, tts_out_frame_bytes, untag_frame
from ai_phone.services.tts import tts_factory
from ai_phone.telephony import events as ev
from ai_phone.telephony.endpoint import BusyError, DialError, LineSpec, Telephony
from ai_phone.telephony.rings import FrameRing
from ai_phone.telephony.state import LineStatus, StateFile, next_state

log = logging.getLogger("ai-phone")
FAKE_LLM = "ai_phone.services.backends.fake:FakeLLM"  # the fake brain is the same whatever the config asks for
FACTORIES = {
    "real": {"stt": "ai_phone.services.backends.mlx_parakeet:ParakeetSTT",
             "llm": {"local": "ai_phone.services.backends.mlx_lm_backend:make_llm",  # models.tier names the weights
                     "openai_compatible": "ai_phone.services.backends.openai_compat:make_openai",
                     "claude": "ai_phone.services.backends.cli_brain:make_claude",
                     "chatgpt": "ai_phone.services.backends.cli_brain:make_codex"},
             "tts": "ai_phone.services.backends.mlx_kokoro:KokoroTTS"},
    "fake": {"stt": "ai_phone.services.backends.fake:FakeSTT",
             # openai_compatible is here for completeness only: llm_worker takes that kind before it reads this map,
             # because an endpoint is a server the person chose and --fake must not stand in for it
             "llm": {"local": FAKE_LLM, "openai_compatible": FAKE_LLM, "claude": FAKE_LLM, "chatgpt": FAKE_LLM},
             "tts": "ai_phone.services.backends.fake:FakeTTS"},
}
SUBSCRIPTION_FIRST_TOKEN_MS = 12000  # a plan's first word takes seconds: its own CLI starts, then the provider answers
DRAIN_SHARE = 3  # a turn the caller talked over may take this fraction of their patience to finish, and no more
LOOPBACK = ("127.0.0.1", "localhost", "::1", "[::1]")
BRAIN_WARNINGS = {
    "claude": ("Brain: your Claude subscription. What callers say, the instructions and the conversation are sent to "
               "Anthropic through Claude Code (not audio)."),
    "chatgpt": ("Brain: your ChatGPT subscription. What callers say, the instructions and the conversation are sent to "
                "OpenAI through Codex (not audio)."),
}


def subscription_first_token_ms(b: BrainConfig) -> int:
    """The deadline for a plan's first word. Four seconds is the default for a model on this Mac; a number the person
    set themselves is the one that counts."""
    default = BrainConfig().first_token_timeout_ms
    return SUBSCRIPTION_FIRST_TOKEN_MS if b.first_token_timeout_ms == default else b.first_token_timeout_ms


def drain_timeout_ms(cfg: Config) -> int:
    """How long a cancelled turn may take to finish before its session is dropped. Claude Code cannot be interrupted,
    and whatever is spent waiting for the old turn comes out of the time the new one has before the controller
    apologises, so the wait gets a third of that budget; past it, a new session (about a second) is the better trade."""
    return cfg.conversation.reply_timeout_sec * 1000 // DRAIN_SHARE


def brain_warning(cfg: Config) -> str | None:
    """What the person is told at start-up about where the words of their calls go. None for a brain on this Mac."""
    b = cfg.brain
    if b.kind in BRAIN_WARNINGS:
        return BRAIN_WARNINGS[b.kind]
    if b.kind == "openai_compatible":
        host = b.base_url.split("://", 1)[-1].split("/", 1)[0].split(":", 1)[0]
        if host not in LOOPBACK:
            return (f"Brain: {b.base_url}. Transcripts of what callers say, the system prompt and the conversation "
                    "history are sent there (not audio). What that service keeps is up to that service.")
    return None


def llm_worker(cfg: Config, tier: str, fake: bool, system_prompt: str, secrets: dict, wire: dict) -> dict:
    """What the brain worker needs: which backend, how to warm it, how to build it, how often to touch it when idle.

    `--fake` stands in for whatever would run on this Mac and for a subscription plan (a trial start must never spend
    a turn of the person's plan). An OpenAI-compatible endpoint is a server the config names, so that one is used
    either way.
    """
    b = cfg.brain
    warm = {"system_prompt": system_prompt}
    if b.kind == "openai_compatible":
        key = secrets.get(b.api_key_env) if b.api_key_env else None
        return {"factory": FACTORIES["real"]["llm"][b.kind], "warm_args": warm, "keep_warm_sec": 0.0,
                "factory_args": {"base_url": b.base_url, "model": b.model, "api_key": key,
                                 "connect_timeout_ms": b.connect_timeout_ms,
                                 "first_token_timeout_ms": b.first_token_timeout_ms,
                                 "stream_idle_timeout_ms": b.stream_idle_timeout_ms,
                                 "max_response_chars": b.max_response_chars}}
    if fake:
        return {"factory": FACTORIES["fake"]["llm"][b.kind], "warm_args": warm, "factory_args": None, "keep_warm_sec": 0.0}
    if b.kind in SUBSCRIPTION_KINDS:
        return {"factory": FACTORIES["real"]["llm"][b.kind], "warm_args": warm, "keep_warm_sec": 0.0,
                "factory_args": {"model": b.model, "cli": b.cli,
                                 "first_token_timeout_ms": subscription_first_token_ms(b),
                                 "stream_idle_timeout_ms": b.stream_idle_timeout_ms,
                                 "max_response_chars": b.max_response_chars,
                                 "drain_timeout_ms": drain_timeout_ms(cfg)}}
    return {"factory": FACTORIES["real"]["llm"]["local"], "warm_args": {**warm, **wire},
            "factory_args": {"model_id": models.repo_for(tier)}, "keep_warm_sec": cfg.models.keep_warm_sec}


APOLOGY = "Sorry, I lost my train of thought. Could you say that again?"
GOODBYE = "I'm having trouble thinking right now. Please call back later. Goodbye."
HANGUP_FIRST_WAIT_SEC = 30.0   # `hangup`: how long the call (and a `--say` goodbye) may take to end before hanging up again
HANGUP_SECOND_WAIT_SEC = 10.0  # then how long PJSIP may take to report the end before the command gives up
_RECORD_ENDED = object()       # `_next_or_connected`: the record ended with no event on the queue


class Daemon:
    def __init__(self, cfg: Config, tier: str, log_dir: Path, lan: bool, fake: bool = False):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        self.cfg, self.tier, self.log_dir, self.fake, self.lan = cfg, tier, Path(log_dir).expanduser(), fake, lan
        self.secrets = load_secrets(paths.secrets_path())
        self.remote_brain = cfg.brain.kind != "local"  # the brain is outside this Mac: an endpoint, or a plan through its CLI
        self._take_secrets_from_environment()
        self._drop_secret_environment()
        self.line_status: dict[str, LineStatus] = {}
        self.state_file = StateFile(paths.state_path())
        self.active_call: dict | None = None
        self._call_waiters: list[asyncio.Queue] = []
        self._outbound: dict | None = None   # {"call_id", "record", "send"} while a control-socket call is in flight
        self._bg_tasks: set[asyncio.Task] = set()  # fire-and-forget tasks the loop only holds weakly
        self.calls = CallRegistry()
        self.counter = DailyCounter(paths.call_counter_path(), cfg.calls.daily_limit)
        self._controllers: dict[str, TurnController] = {}  # call ID -> the controller running it

    def _secret_env_names(self) -> list[str]:
        names = [lc.password_env or line_password_env(lid) for lid, lc in self.cfg.lines.items() if lc.kind == "trunk"]
        if self.cfg.brain.api_key_env:
            names.append(self.cfg.brain.api_key_env)
        return names

    def _take_secrets_from_environment(self) -> None:
        """Secrets the user exported in the shell move from os.environ into self.secrets before any worker is
        spawned, so no child process inherits them (spawned workers copy the daemon's environment)."""
        for name in self._secret_env_names():
            value = os.environ.pop(name, None)
            if value and name not in self.secrets:
                self.secrets[name] = value

    def _drop_secret_environment(self) -> None:
        """Every other secret-looking variable the shell exported (unrelated API keys, tokens) leaves the daemon's
        environment for good. The daemon needs none of them, and every child it will ever spawn inherits what is
        left: the model workers, and the multiprocessing resource tracker that appears with the first shared ring,
        before any worker's own scrub runs (seen on the gate: the tracker held a shell's ELEVENLABS_API_KEY)."""
        for name in [n for n in os.environ if is_secret_env(n)]:
            del os.environ[name]

    # ---- system prompt and cached phrases -------------------------------------------------------
    def fixed_prompt(self) -> str:
        """The cached part of every call's instructions. The host line is for a local brain only."""
        host = None if self.remote_brain else f"You are running on {socket.gethostname()} ({platform.machine()})."
        return prompts.fixed_instructions(host)

    async def _cache_phrase(self, tts: WorkerClient, ring: FrameRing, text: str) -> list[bytes]:
        gen = tts.next_gen()
        tts.send({"op": "tts.synth", "gen": gen, "seq": 0, "text": text})
        frames = []
        while True:
            m = await asyncio.wait_for(tts.events.get(), 120)
            if m["ev"] == "error":
                raise RuntimeError(f"TTS worker failed while caching a phrase: {m.get('message', '')[:200]}")
            while (f := ring.pop()) is not None:
                g, payload = untag_frame(f)
                if g == (gen & 0xFFFFFFFF):
                    frames.append(payload)
            if m["ev"] == "done" and m["gen"] == gen:
                break
        return frames

    # ---- lines -----------------------------------------------------------------------------------
    def _line_specs(self) -> list[LineSpec]:
        lines = dict(self.cfg.lines)
        if not lines:
            lines["local"] = LineConfig(kind="sip", user="ai", aor="sip:ai@127.0.0.1", display_name="ai", accept_invites_from="lan", caller_id="sip:ai@127.0.0.1")
        specs = []
        for lid, lc in lines.items():
            if self.lan:
                lc.accept_invites_from = "lan"
            password = None
            if lc.kind == "trunk":
                env_name = lc.password_env or line_password_env(lid)
                password = self.secrets.get(env_name)
                if not password:
                    log.error("line %s skipped: password variable %s is not set (secrets.env or the environment)", lid, env_name)
                    self.line_status[lid] = LineStatus(lid, lc.kind, lc.number or "", "skipped", lc.registrar, None, f"{env_name} not set", time.time())
                    continue
            specs.append(LineSpec.from_config(lid, lc, password))
            self.line_status[lid] = LineStatus(lid, lc.kind, lc.number or lc.aor or "", "registering" if lc.kind == "trunk" else "unregistered",
                                               lc.registrar, None, None, time.time())
        return specs

    def _write_state(self) -> None:
        """state.json is for `ai-phone status`; a write that fails (a full disk) is logged and never stops a call."""
        b = self.cfg.brain
        brain = {"kind": b.kind, "base_url": b.base_url if b.kind == "openai_compatible" else None,
                 "model": "fake" if self.fake else (b.model if self.remote_brain else models.repo_for(self.tier))}
        try:
            self.state_file.write(self.line_status, self.active_call, brain)
        except OSError as e:
            log.warning("could not write %s: %s", self.state_file.path, e)

    def _on_reg(self, e: ev.RegState) -> None:
        st = self.line_status.get(e.line_id)
        if st is None:
            return
        new = next_state(st.state, e.active, e.code)
        if new != st.state:
            st.since = time.time()
        st.state, st.last_error = new, (None if e.active else f"{e.code} {e.reason}")
        st.expires_at = time.time() + e.expires_sec if e.active else None
        log.info("registration line=%s state=%s code=%s reason=%s", e.line_id, new, e.code, e.reason)
        self._write_state()

    # ---- main --------------------------------------------------------------------------------------
    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        events: asyncio.Queue = asyncio.Queue()
        playback_rate = self.cfg.playback_rate
        rx, tx = FrameRing(None, 100), FrameRing(None, 400, frame_bytes=playback_rate // 50 * 2)
        stt_in = FrameRing(None, 400, frame_bytes=STT_IN_FRAME)
        tts_out = FrameRing(None, 400, frame_bytes=tts_out_frame_bytes(playback_rate))
        F = FACTORIES["fake" if self.fake else "real"]
        default_prompt = self.fixed_prompt()
        wa = {} if self.fake else {"wire_memory": self.cfg.models.wire_memory}  # the MLX backends pin their weights
        kw = self.cfg.models.keep_warm_sec
        stt = WorkerClient("stt", F["stt"], loop, stt_in, None, dict(wa), keep_warm_sec=kw)
        warning = brain_warning(self.cfg)
        if warning:
            log.warning("%s", warning)
        spec = llm_worker(self.cfg, self.tier, self.fake, default_prompt, self.secrets, wa)
        llm = WorkerClient("llm", spec["factory"], loop, None, None, spec["warm_args"],
                           factory_args=spec["factory_args"], keep_warm_sec=spec["keep_warm_sec"])
        tts_args = {"output_rate": playback_rate}
        if not self.fake:
            tts_args.update(voice=self.cfg.tts.voice, speed=self.cfg.tts.speed)
        tts = WorkerClient("tts", F["tts"] if self.fake else tts_factory(self.cfg.tts.engine), loop, None, tts_out, dict(wa),
                           factory_args=tts_args, keep_warm_sec=kw)
        tel: Telephony | None = None
        control: ControlServer | None = None
        started: list[WorkerClient] = []
        try:  # from here on every failure tears down what exists: workers, rings, endpoint, socket
            warm = []
            for client in (stt, llm, tts):  # one at a time, so a failure part-way stops exactly the workers that exist
                try:
                    warm.append(await client.start())
                except RuntimeError as e:
                    log.error("%s", redact(str(e), self.secrets.values()))
                    raise SystemExit(1)
                started.append(client)
            cached = {"prompt": await self._cache_phrase(tts, tts_out, "Still there?"),
                      "warning": await self._cache_phrase(tts, tts_out, "This call will end in thirty seconds."),
                      "apology": await self._cache_phrase(tts, tts_out, APOLOGY),
                      "goodbye": await self._cache_phrase(tts, tts_out, GOODBYE)}
            disclosure = await self._cache_phrase(tts, tts_out, prompts.DISCLOSURE)
            greeting = await self._cache_phrase(tts, tts_out, self.cfg.inbound.greeting)
            relay_phrases = RelayPhrases(
                Phrase(prompts.FILLER, tuple(await self._cache_phrase(tts, tts_out, prompts.FILLER))),
                Phrase(prompts.STILL_HERE, tuple(await self._cache_phrase(tts, tts_out, prompts.STILL_HERE))),
                Phrase(prompts.RELAY_GOODBYE, tuple(await self._cache_phrase(tts, tts_out, prompts.RELAY_GOODBYE))))
            specs = self._line_specs()
            tel = Telephony(self.cfg, specs, loop, events, rx, tx)
            tel.start()
            for w in tel.peer_warnings:
                log.warning(w)
            self._write_state()
            turn_log = TurnLog(self.log_dir / "turns.jsonl")
            stop = asyncio.Event()
            for s in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(s, stop.set)
            ctx = dict(tel=tel, stt=stt, llm=llm, tts=tts, rx=rx, tx=tx, stt_in=stt_in, tts_out=tts_out, cached=cached,
                       disclosure=disclosure, greeting=greeting, relay_phrases=relay_phrases, fixed=default_prompt,
                       turn_log=turn_log, events=events, stop=stop, specs={s.id: s for s in specs})
            control = ControlServer(paths.socket_path(), lambda msg, send: self._control(msg, send, ctx))
            await control.start()
            # what actually started, not what the config asked for: --fake loads FakeLLM whatever the tier says
            brain = "fake" if self.fake else (self.cfg.brain.model if self.remote_brain else models.repo_for(self.tier))
            log.info("ready lines=%s brain=%s model=%s warm=%s", [s.id for s in specs], self.cfg.brain.kind, brain, warm)
            print("ready", flush=True)
            while not stop.is_set():
                try:
                    e = await asyncio.wait_for(events.get(), 0.2)
                except asyncio.TimeoutError:
                    continue
                if isinstance(e, ev.IncomingCall):
                    self._on_incoming(e, tel)
                elif isinstance(e, ev.CallRejected):
                    log.info("rejected on %s: %s with %s", e.line_id, e.source_ip, e.code)
                elif isinstance(e, ev.MediaActive):
                    await self._on_media_active(e, ctx)
                elif isinstance(e, ev.RegState):
                    self._on_reg(e)
                elif isinstance(e, ev.OutboundProgress):
                    log.info("outbound line=%s call=%s state=%s code=%s %s", e.line_id, e.call_id, e.state, e.code, e.reason)
                    for q in self._call_waiters:
                        q.put_nowait(e)
                elif isinstance(e, ev.CallStateChanged) and e.state == "ended":
                    self._on_call_ended(e, tel)
        finally:
            if control is not None:
                await control.stop()
            if tel is not None:
                tel.stop()
            if started:
                await asyncio.gather(*(c.stop() for c in started))
            for r in (rx, tx, stt_in, tts_out):
                r.close()
                r.unlink()

    # ---- call events (main loop) ------------------------------------------------------------------------
    def _on_incoming(self, e: ev.IncomingCall, tel) -> CallRecord | None:
        """Relay mode when an AI is waiting in `listen` and no call is in progress; phone agent mode otherwise."""
        log.info("incoming on %s from %s (%s)", e.line_id, e.from_uri, e.source_ip)
        if not tel.answer(e.call_id):
            log.info("incoming call %s on %s ended before it could be answered", e.call_id, e.line_id)
            return None
        mode = "relay" if self.calls.listening and self.calls.current is None else "agent"
        rec = self.calls.new(mode, "in", e.line_id, e.from_uri)
        rec.pj_call_id = e.call_id
        if mode == "relay":
            self.calls.deliver_incoming(rec)
        self.active_call = {"call_id": e.call_id, "call": rec.id, "mode": mode, "line_id": e.line_id,
                            "remote": e.from_uri, "started_at": time.time(), "direction": "in"}
        self._write_state()
        return rec

    async def _on_media_active(self, e: ev.MediaActive, ctx: dict) -> None:
        """The call's audio is up: run the conversation. The control handler (if any) waits on its queue; the main
        loop is the only consumer of `events`."""
        ob = self._outbound if self._outbound and self._outbound["call_id"] == e.call_id else None
        rec = self.calls.by_pj(e.call_id) or (ob["record"] if ob else None)
        if rec is None or rec.is_ended:  # cannot happen (every answered call has a live record); never run a call without one
            log.error("media on call %s without a live call record; hanging up", e.call_id)
            ctx["tel"].hangup(e.call_id)
            return
        on_turn = None
        if ob is not None and rec.mode == "agent":
            send = ob["send"]
            on_turn = lambda r: self._send_quiet(
                send, {"event": "turn", **{k: r[k] for k in ("turn", "first_audible_ms", "bargein")}})
        await self._run_call(rec, ctx, on_turn_event=on_turn)
        for q in self._call_waiters:
            q.put_nowait(("call_done", e.call_id))
        if ob is not None and self._outbound is ob:
            self._outbound = None

    def _on_call_ended(self, e: ev.CallStateChanged, tel) -> None:
        """An `ended` no controller consumed (the controller runs inside the main loop and eats its own): the call
        ended before its conversation started. Outbound records are finished by their `call` handler."""
        rec = self.calls.by_pj(e.call_id)
        if rec is not None and rec.direction == "in":  # an inbound call that ended before its audio started
            rec.finish("completed", ended_by="them", detail="ended before the conversation started")
            self.calls.end_current(rec)
        if self.active_call is not None and self.active_call.get("call_id") == e.call_id:
            # the active call ended before media became active (caller hung up while ringing, busy, rejected…);
            # a rejected second caller's own "ended" carries a different id and must not touch it
            self.active_call = None
            self._write_state()
        # nothing else will release the claim; release_call is a no-op for any id that is not the claimed call
        tel.release_call(e.call_id)
        for q in self._call_waiters:
            q.put_nowait(e)

    # ---- control socket -------------------------------------------------------------------------------
    async def _control(self, msg: dict, send, ctx: dict) -> None:
        cmd = msg["cmd"]
        if cmd == "status":
            cur = self.calls.current
            await self._send_quiet(send, {"ok": True, "lines": {k: v.__dict__ for k, v in self.line_status.items()},
                                          "active_call": self.active_call, "call": cur.summary() if cur else None,
                                          "relay_listener": self.calls.listening})
        elif cmd == "listen":
            await self._cmd_listen(msg, send)
        elif cmd == "say":
            await self._cmd_say(msg, send)
        elif cmd == "hangup":
            await self._cmd_hangup(msg, send, ctx)
        elif cmd == "call":
            await self._cmd_call(msg, send, ctx)
        else:  # never let an unrouted command reach the dial path
            await self._err(send, "usage", f"unknown command {cmd!r}")

    async def _err(self, send, code: str, text: str) -> None:
        await self._send_quiet(send, {"ok": False, "error": text, "code": EXIT[code]})

    def _resolve(self, call_id: str | None) -> tuple[CallRecord | None, str | None]:
        if call_id:
            rec = self.calls.get(call_id)
            return rec, (None if rec else f"no call {call_id} (only the last {self.calls.retain} calls are kept)")
        if self.calls.current is not None:
            return self.calls.current, None
        last = self.calls.unread_ended()
        hint = f" (call {last.id} has ended; `ai-phone listen` returns its result)" if last else ""
        return None, "no call in progress" + hint

    async def _reply_event(self, send, rec: CallRecord, event: dict | None) -> None:
        """Hand one of a call's events to `listen`. A listen cancelled before its reply went out (its client left)
        puts the event back, so the next listen gets it."""
        try:
            await self._send_quiet(send, {"ok": True, **event} if event else {"ok": True, "kind": "timeout"})
        except asyncio.CancelledError:
            if event:
                rec.put_back(event)
            raise

    async def _cmd_listen(self, msg: dict, send) -> None:
        call_id, timeout = msg.get("call_id"), float(msg.get("timeout_sec", 60))
        if call_id is None and self.calls.current is None:
            ended = self.calls.unread_ended()
            if ended is not None:  # a relay call ended since anyone last listened: report that first
                return await self._reply_event(send, ended, await ended.next_event(0))
            if not self.cfg.calls.allow_inbound:
                return await self._err(send, "rejected", "inbound calls are disabled (calls.allow_inbound is false)")
            try:
                fut = self.calls.start_listening()
            except ListenerBusy as e:
                return await self._err(send, "listener_busy", str(e))
            rec = None
            try:
                rec = await asyncio.wait_for(asyncio.shield(fut), timeout)
            except asyncio.TimeoutError:
                pass
            finally:
                self.calls.stop_listening(fut)  # also runs when the server cancels a listen whose client left
            if rec is None and fut.done() and not fut.cancelled():
                rec = fut.result()  # a call arrived as the wait timed out: it is relay mode now, so report it
            if rec is None:
                return await self._send_quiet(send, {"ok": True, "kind": "timeout"})
            # a reply cancelled from here on cannot be undone: that call runs with nobody listening and ends
            # politely at the relay timeout
            return await self._send_quiet(send, {"ok": True, "kind": "incoming", "call_id": rec.id,
                                                 "line": rec.line_id, "remote": rec.remote})
        rec, err = self._resolve(call_id)
        if err:
            return await self._err(send, "no_call", err)
        if rec.mode != "relay" and not rec.is_ended:
            return await self._err(send, "no_call", f"call {rec.id} is in phone agent mode; listen follows relay calls, "
                                                    "or waits for incoming calls when no call is in progress")
        await self._reply_event(send, rec, await rec.next_event(timeout))

    async def _cmd_say(self, msg: dict, send) -> None:
        rec, err = self._resolve(msg.get("call_id"))
        if err:
            return await self._err(send, "no_call", err)
        if rec.is_ended:
            return await self._err(send, "no_call", f"call {rec.id} has ended; `ai-phone listen --call {rec.id}` returns its result")
        if rec.mode != "relay":
            return await self._err(send, "no_call", f"call {rec.id} is in phone agent mode; say works on relay calls")
        if not await rec.wait_connected(10) or rec.id not in self._controllers:
            if rec.is_ended:
                return await self._err(send, "no_call", f"call {rec.id} has ended; `ai-phone listen --call {rec.id}` returns its result")
            return await self._err(send, "no_call", f"call {rec.id} is not connected")
        result = await self._controllers[rec.id].say(msg["text"])
        if result == "ended":
            return await self._err(send, "no_call", f"call {rec.id} ended before the text was spoken")
        await self._send_quiet(send, {"ok": True, "result": result, "call_id": rec.id})

    async def _cmd_hangup(self, msg: dict, send, ctx: dict) -> None:
        rec, err = self._resolve(msg.get("call_id"))
        if err:
            return await self._err(send, "no_call", err)
        if not rec.is_ended:
            rec.ended_by = rec.ended_by or "hangup_command"
            tc = self._controllers.get(rec.id)
            if msg.get("say") and tc is not None:
                tc.say(msg["say"], end_reason="hangup_command")
            elif rec.pj_call_id is not None:
                ctx["tel"].hangup(rec.pj_call_id)
            if not await rec.wait_ended(HANGUP_FIRST_WAIT_SEC) and rec.pj_call_id is not None:
                ctx["tel"].hangup(rec.pj_call_id)  # the goodbye never got its turn: hang up anyway
                await rec.wait_ended(HANGUP_SECOND_WAIT_SEC)
            if not rec.is_ended:
                return await self._err(send, "daemon_gone", f"call {rec.id} did not end after the hangup")
        rec.end_reported = True
        await self._send_quiet(send, {"ok": True, "result": rec.result()})

    async def _cmd_call(self, msg: dict, send, ctx: dict) -> None:
        if not self.cfg.calls.allow_outbound:
            return await self._err(send, "rejected", "outbound calls are disabled (calls.allow_outbound is false)")
        tel, to = ctx["tel"], msg["to"]
        relay, prompt = bool(msg.get("relay")), msg.get("prompt")
        if is_emergency(to):
            return await self._err(send, "blocked", f"{to} matches an emergency number; ai-phone never dials those")
        line_id = msg.get("line") or self._default_line_for(to, ctx["specs"], tel)
        if line_id not in ctx["specs"]:
            return await self._err(send, "rejected", f"no such line {line_id!r}" if msg.get("line") else
                                   "no line can dial that target (no registered trunk for a phone number, or no sip line for a sip: address)")
        if not self.counter.allowed():
            return await self._err(send, "daily_limit", f"the daily limit of {self.counter.limit} outbound calls is reached")
        q: asyncio.Queue = asyncio.Queue()
        self._call_waiters.append(q)
        cid: int | None = None     # set once PJSIP has a call; cleanup below must not assume it
        rec: CallRecord | None = None
        mine: dict | None = None   # our entry in self._outbound
        handed_over = False        # a relay call that carries on after this command returns
        outcome, detail = "rejected", ""
        try:
            try:
                cid = tel.make_call(line_id, to, float(msg.get("timeout_sec", 30)))
            except BusyError:
                return await self._err(send, "busy", "busy")
            except DialError as e:
                return await self._send_quiet(send, {"event": "ended", "reason": "rejected", "detail": str(e),
                                                     "exit_code": EXIT["rejected"], "call_id": None, "result": None})
            rec = self.calls.new("relay" if relay else "agent", "out", line_id, to, prompt)
            rec.pj_call_id = cid
            try:
                self.counter.record()  # every accepted dial attempt counts: busy and unanswered calls can cost too
            except Exception as e:  # an attempt the daily limit cannot see is never let through
                error = f"could not update the daily call counter: {e}"
                log.error("control: call %s refused: %s", rec.id, error)
                rec.finish("rejected", ended_by="error", detail=error)
                self.calls.end_current(rec)
                self._hangup_quietly(tel, cid)
                return await self._err(send, "rejected", error)
            log.info("control: %s call %s on line %s to %s -> pjsip call %s", rec.mode, rec.id, line_id, to, cid)
            mine = self._outbound = {"call_id": cid, "record": rec, "send": send}
            self.active_call = {"call_id": cid, "call": rec.id, "mode": rec.mode, "line_id": line_id, "remote": to,
                                "started_at": time.time(), "direction": "out"}
            self._write_state()
            await self._send_quiet(send, {"event": "created", "call_id": rec.id})
            while True:  # progress arrives via the main loop, which is the only consumer of the telephony queue
                if relay and outcome == "answered":
                    # answered: wait for the audio, but an `ended` that comes first must end the record at once
                    e = await self._next_or_connected(q, rec)
                    if e is None:
                        handed_over = True  # the conversation runs; _run_call finishes the record
                        await self._send_quiet(send, {"ok": True, "event": "answered", "call_id": rec.id, "mode": "relay"})
                        return  # the call carries on; listen/say/hangup take over
                else:
                    e = await q.get()
                if isinstance(e, ev.OutboundProgress) and (e.call_id == cid or (e.call_id == -1 and e.line_id == line_id)):
                    # -1: "dialling", before PJSIP assigns the id
                    state = e.state
                    terminal = state in ("busy", "no_answer", "rejected", "redirected")
                    if terminal:  # PJSIP posts this before `ended`; a `hangup` before the answer shows up here (487)
                        state = "cancelled" if rec.ended_by == "hangup_command" else state
                        outcome, detail = state, f"{e.code} {e.reason}".strip()
                    await self._send_quiet(send, {"event": state, "code": e.code, "uri": e.reason if state == "dialling" else "",
                                                  "reason": "" if state == "dialling" else e.reason})
                    if terminal:
                        break
                    if state == "answered":
                        outcome = "answered"
                        rec.answered = rec.answered or time.time()
                elif isinstance(e, tuple) and e[0] == "call_done" and e[1] == cid:
                    outcome = "answered"  # the conversation ran, whatever order PJSIP delivered media and CONFIRMED in
                    break
                elif e is _RECORD_ENDED:
                    break
                elif isinstance(e, ev.CallStateChanged) and e.call_id == cid and e.state == "ended" and not rec.is_ended:
                    if rec.ended_by == "hangup_command" and outcome != "answered":
                        outcome = "cancelled"
                    break  # ended before media became active
            if outcome == "rejected" and detail and detail.split(" ", 1)[0] in ("403", "603"):
                if not any(w in detail.lower() for w in ("whitelist", "account level", "verified")):  # the carrier said why already
                    detail += (" (the carrier refused the call: either the caller ID is not one of the line's numbers or a verified one,"
                               " or the destination is not allowed for this account or its outbound voice profile)")
            if rec.is_ended:  # the conversation ran and _run_call finished the record
                code = EXIT["completed"]
            elif outcome == "answered":  # answered, then gone before any audio
                rec.finish("completed", ended_by="them", detail="ended before the conversation started")
                self.calls.end_current(rec)
                code = EXIT["completed"]
            else:
                rec.finish(outcome, detail=detail)
                self.calls.end_current(rec)
                code = EXIT.get(outcome, EXIT["rejected"])
            rec.end_reported = True  # the result goes to this command: `listen` need not report the end again
            await self._send_quiet(send, {"event": "ended", "reason": rec.outcome, "detail": rec.detail, "exit_code": code,
                                          "call_id": rec.id, "result": rec.result()})
        finally:
            failure = sys.exception()  # None when the handler returned normally
            stopped = isinstance(failure, asyncio.CancelledError)
            ended_before = rec is not None and rec.is_ended  # the call's own end was already recorded
            if mine is not None and self._outbound is mine:
                self._outbound = None  # only our own: a `call` refused as busy must not touch the call in progress
            self._call_waiters.remove(q)
            running = rec is not None and rec.id in self._controllers
            if rec is not None and not rec.is_ended and not handed_over and not running:
                # left early, with no conversation running that would finish the record: cancelled when the daemon
                # stops (ControlServer.stop), or a failure in this handler
                rec.finish("completed" if outcome == "answered" else ("cancelled" if stopped else "rejected"), ended_by="error",
                           detail="ai-phone stopped" if stopped else f"the call command failed ({type(failure).__name__})")
                self.calls.end_current(rec)
            if cid is not None and failure is not None and not stopped and outcome != "answered" and not handed_over \
                    and not running and not ended_before:
                self._hangup_quietly(tel, cid)  # this handler failed: nobody is behind the call, so it must not ring on
            if cid is not None and outcome != "answered":
                if rec is not None and self.active_call is not None and self.active_call.get("call") == rec.id:
                    self.active_call = None  # never reached media, so _run_call never cleared it
                    self._write_state()
                # the main loop releases on `ended` too; both are idempotent, so the claim can never be left behind
                if getattr(tel, "current_call", None) is not None and tel.current_call.getId() == cid and tel.current_call.ended:
                    tel.release_call(cid)

    @staticmethod
    def _hangup_quietly(tel, cid: int) -> None:
        """Hang up a call nobody is behind. PJSIP may refuse one it has just ended; that changes nothing, and must not
        replace the failure being handled."""
        try:
            tel.hangup(cid)
        except Exception as e:
            log.warning("hangup of pjsip call %s failed: %s", cid, e)

    @staticmethod
    async def _next_or_connected(q: asyncio.Queue, rec: CallRecord):
        """An answered relay call waiting for its audio: None once the audio is up, otherwise the next event on `q`
        (so an `ended` that comes before any audio is handled at once instead of after a timeout)."""
        getter = asyncio.ensure_future(q.get())
        connected = asyncio.ensure_future(rec.wait_connected(None))
        try:
            await asyncio.wait((getter, connected), return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in (getter, connected):
                t.cancel()
            await asyncio.gather(getter, connected, return_exceptions=True)
        if connected.done() and not connected.cancelled() and connected.result():
            return None  # an event that arrived at the same moment is not needed: _run_call owns the call now
        if getter.done() and not getter.cancelled():
            return getter.result()
        return _RECORD_ENDED  # the record was finished elsewhere and no event is waiting

    @staticmethod
    def _default_line_for(target: str, specs: dict, tel) -> str | None:
        """No --line given: a sip: address goes out on the first sip line, or nowhere;
        a phone number goes out on the first registered trunk line that has a dial domain."""
        if target.startswith("sip:"):
            sip_lines = [lid for lid, s in specs.items() if s.kind == "sip"]
            return sip_lines[0] if sip_lines else None  # never a trunk by accident: say --line if that is really wanted
        for lid, s in specs.items():
            acc = tel.accounts.get(lid) if hasattr(tel, "accounts") else None
            if s.kind == "trunk" and s.dial_domain and (acc is None or acc.registered):
                return lid
        return None

    async def _send_quiet(self, send, msg: dict) -> None:
        try:
            await send(msg)
        except ConnectionError:
            pass  # the CLI went away; the call continues

    # ---- one call --------------------------------------------------------------------------------------
    @staticmethod
    def _ended_by(end_reason: str | None, remote_ended: bool) -> str:
        if remote_ended:
            return "them"
        return {"agent": "agent", "relay_timeout": "timeout", "silence": "silence", "max_duration": "max_duration",
                "hangup_command": "hangup_command"}.get(end_reason or "", "error")

    async def _run_call(self, rec: CallRecord, ctx: dict, on_turn_event=None) -> None:
        tel, stt, llm, tts = ctx["tel"], ctx["stt"], ctx["llm"], ctx["tts"]
        rx, tx, stt_in, tts_out, cached = ctx["rx"], ctx["tx"], ctx["stt_in"], ctx["tts_out"], ctx["cached"]
        call_id, line_id, out = rec.pj_call_id, rec.line_id, rec.direction == "out"
        rec.answered = rec.answered or time.time()
        vad = EnergyVad() if self.fake else SileroVad()
        n = [0]
        turn_log, events, stop = ctx["turn_log"], ctx["events"], ctx["stop"]

        def on_turn(t):
            r = t.to_record(str(call_id), n[0])
            r.update(line_id=line_id, call=rec.id, mode=rec.mode)
            turn_log.write(r)
            n[0] += 1
            log.info("turn %s call=%s line=%s first_audible=%s bargein=%s brain_error=%s", n[0], rec.id, line_id,
                     t.first_audible_ms(), t.bargein, t.brain_error)
            if on_turn_event is not None:
                task = asyncio.get_running_loop().create_task(on_turn_event(r))
                self._bg_tasks.add(task)
                task.add_done_callback(self._bg_tasks.discard)

        # an outbound call without a task uses prompts.DEFAULT_CALL_SECTION (an honest no-purpose conversation);
        # an incoming call never does: blank inbound instructions mean the take-a-message default
        section = rec.prompt if out else (self.cfg.inbound.instructions.strip() or InboundConfig().instructions)
        system = prompts.system_prompt(ctx["fixed"], section)
        # the instructions are settled before the greeting plays: a brain that holds a session starts it now, so the
        # first reply of the call does not wait for a CLI to come up (every other brain ignores this)
        llm.send({"op": "llm.prepare", "gen": llm.current_gen, "system_prompt": system})
        tc = TurnController(self.cfg.conversation, tel.port, stt, llm, tts, rx, tx, stt_in, tts_out,
                            Endpointer(vad, self.cfg.conversation.endpoint_silence_ms), ctx["greeting"],
                            system, on_turn,
                            prompt_frames=cached["prompt"], warning_frames=cached["warning"],
                            rtp_last_rx_ns=lambda: tel.receive_port.last_rx_ns, apology_frames=cached["apology"],
                            goodbye_frames=cached["goodbye"], opening_frames=ctx["disclosure"] if out else None,
                            direction=rec.direction,
                            replies=RelayReplies(rec.heard) if rec.mode == "relay" else ModelReplies(),
                            calls=self.cfg.calls, relay_phrases=ctx["relay_phrases"], on_line=rec.add_line,
                            opening_text=prompts.DISCLOSURE if out else self.cfg.inbound.greeting,
                            generate_opening=out and rec.mode == "agent")
        self._controllers[rec.id] = tc
        rec.mark_connected()
        remote_ended = False
        call_stop = asyncio.Event()
        runner = asyncio.create_task(tc.run(call_stop))
        try:
            while not call_stop.is_set() and not stop.is_set():
                if tc.hangup_requested.is_set():
                    log.info("hangup: call %s %s", rec.id, tc.end_reason)
                    tel.hangup(call_id)
                    break
                try:
                    e = await asyncio.wait_for(events.get(), 0.1)
                except asyncio.TimeoutError:
                    continue
                if isinstance(e, ev.CallStateChanged) and e.state == "ended":
                    if e.call_id == call_id:
                        remote_ended = True
                        break
                    log.info("ignoring end of unrelated call %s while call %s is active", e.call_id, call_id)
                elif isinstance(e, ev.CallRejected):
                    log.info("rejected on %s: %s with %s", e.line_id, e.source_ip, e.code)
                elif isinstance(e, ev.OutboundProgress):
                    # PJSIP can report media active before CONFIRMED, so "answered" may arrive while the call already runs
                    log.info("outbound line=%s call=%s state=%s code=%s %s", e.line_id, e.call_id, e.state, e.code, e.reason)
                    for q in self._call_waiters:
                        q.put_nowait(e)
                elif isinstance(e, ev.RegState):
                    self._on_reg(e)   # registration changes keep flowing during a call
        finally:
            call_stop.set()
            try:
                await runner
            finally:  # the teardown runs even if the controller failed: the record ends and the claim is released
                for c in (stt, llm, tts):
                    c.cancel(c.current_gen)
                # the call is over: a brain that holds a CLI session lets it go, so no process of ours idles between
                # calls (every other brain ignores this, as it ignores llm.prepare)
                llm.send({"op": "llm.release", "gen": llm.current_gen})
                if tc.timer is not None and tc.timer.t0 is not None:
                    r = tc.incomplete_record(str(call_id), n[0])  # keeps what the agent had said, and the counters
                    r.update(incomplete=True, state_at_end=tc.state, line_id=line_id, call=rec.id, mode=rec.mode)
                    turn_log.write(r)
                ended_by = self._ended_by(tc.end_reason, remote_ended)
                detail = "" if ended_by != "error" else (tc.end_reason or ("ai-phone stopped" if stop.is_set() else "unknown"))
                rec.finish("completed", ended_by=ended_by, detail=detail)
                self._controllers.pop(rec.id, None)
                port = tel.port
                if port is not None and hasattr(port, "cancel_flush"):
                    port.cancel_flush()  # a flush the PJSIP thread never performed must not hit the next call's greeting
                for r_ in (rx, tx, tts_out):
                    r_.flush()
                self.active_call = None
                tel.release_call(call_id)  # only now may the next call be claimed
                self.calls.end_current(rec)
                self._write_state()  # last: nothing it raises may keep the line claimed or the record current

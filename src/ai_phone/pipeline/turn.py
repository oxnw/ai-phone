"""TurnController: owns one call's conversation. GREETING → LISTENING → THINKING → SPEAKING → LISTENING.

Runs a 5 ms asyncio loop that drains caller audio into the endpointer, pumps TTS frames to the
outbound ring, and applies worker events. Every generation carries a gen_id shared by all three
workers; anything tagged with a stale gen_id is dropped on receipt by the WorkerClient.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from typing import Callable, Protocol

import numpy as np

from ai_phone.config import CallsConfig, ConversationConfig
from ai_phone.pipeline.bargein import BargeInDetector
from ai_phone.pipeline.chunker import ClauseChunker, remove_end_marker
from ai_phone.pipeline.ending import is_closing_reply
from ai_phone.pipeline.prompts import OPENING_CUE
from ai_phone.pipeline.replies import ModelReplies, Phrase, RelayPhrases, ReplySource
from ai_phone.pipeline.timing import TurnTimer
from ai_phone.pipeline.vad import Endpointer, SpeechChunk, SpeechEnd, SpeechStart
from ai_phone.services.client import WorkerClient
from ai_phone.services.ipc import tag_frame, untag_frame
from ai_phone.telephony.rings import FrameRing


_log = logging.getLogger("ai-phone.turn")
OPENING = ("GREETING", "OUTBOUND_OPENING")  # the cached line that opens a call, then LISTENING; only a greeting can be cut
GREETING_MAX_SEC = 1.5  # held speech shorter than this that ended during the disclosure is a greeting, not a turn
_DEBUG_RX = os.environ.get("AI_PHONE_DEBUG_RX")  # path prefix: every call's inbound audio is written as <prefix>-<n>.raw (16 kHz int16)


def _without_marker(text: str) -> str:
    """The reply as it was spoken: the end marker (any case) is an instruction, never part of the transcript."""
    return remove_end_marker(text)[0]


class PortLike(Protocol):
    first_nonsilent_ns: int | None

    def arm_t5(self) -> None: ...


class TurnController:
    def __init__(self, cfg: ConversationConfig, port: PortLike, stt: WorkerClient, llm: WorkerClient, tts: WorkerClient,
                 rx: FrameRing, tx: FrameRing, stt_in: FrameRing, tts_out: FrameRing, endpointer: Endpointer,
                 greeting_frames: list[bytes], system_prompt: str, on_turn: Callable[[TurnTimer], None],
                 prompt_frames: list[bytes] = (), warning_frames: list[bytes] = (),
                 rtp_last_rx_ns: Callable[[], int] | None = None, apology_frames: list[bytes] = (), goodbye_frames: list[bytes] = (),
                 opening_frames: list[bytes] | None = None, direction: str = "in",
                 replies: ReplySource | None = None, calls: CallsConfig | None = None,
                 relay_phrases: RelayPhrases | None = None,
                 on_line: Callable[[str, str, bool], None] | None = None, opening_text: str = "",
                 generate_opening: bool = False):
        self.cfg, self.port, self.stt, self.llm, self.tts = cfg, port, stt, llm, tts
        self.rx, self.tx, self.stt_in, self.tts_out, self.ep = rx, tx, stt_in, tts_out, endpointer
        self.greeting, self.system_prompt, self.on_turn = greeting_frames, system_prompt, on_turn
        self.state = "GREETING"
        self.history: list[dict] = []
        self.timer: TurnTimer | None = None
        self.chunker = ClauseChunker()
        self._pending_stt = 0
        self._seq = 0
        self._tts_done_seq = -1
        self._llm_done = False
        self._reply_parts: list[str] = []
        self.end_reason: str | None = None
        self.last_activity_ns = time.monotonic_ns()
        self.hangup_requested = asyncio.Event()
        self.bargein = BargeInDetector(cfg.bargein_min_speech_ms, cfg.bargein_playback_gain)
        self._interrupt_t: int | None = None
        self.prompt_frames, self.warning_frames, self.rtp_last_rx_ns = list(prompt_frames), list(warning_frames), rtp_last_rx_ns
        self._prompted = False
        self._warned = False
        self._start_ns = 0
        self._candidate: list[np.ndarray] = []  # speech chunks seen while not listening; replayed on barge-in
        self.apology_frames, self.goodbye_frames = list(apology_frames), list(goodbye_frames)
        self.opening_frames, self.direction = opening_frames, direction
        self._brain_errors = 0  # consecutive brain failures in this call; reset by a completed turn
        self._after_flush: list[bytes] = []  # cached frames waiting for a flush or room in the playback ring
        self._tts_pending: bytes | None = None  # one tagged frame waiting for room, checked again after a gen change
        self._drain_deadline_ns: int | None = None  # ENDING_DRAIN: hang up once tx is empty or this passes
        self._rx_dump = open(f"{_DEBUG_RX}-{int(time.time())}.raw", "wb") if _DEBUG_RX else None
        self.replies = replies or ModelReplies()
        self.calls = calls or CallsConfig()
        self.relay_phrases = relay_phrases
        self.on_line, self.opening_text, self.generate_opening = on_line, opening_text, generate_opening
        self._say_queue: list[tuple[str, str | None, asyncio.Future]] = []
        self._say_active: asyncio.Future | None = None  # the say being spoken now
        self._any_say = False                           # relay: the calling AI has spoken at least once on this call
        self._end_after_reply: str | None = None        # end reason to apply once the current say has played
        self._relay_wait_from: int | None = None        # relay: waiting for the calling AI's say since this instant
        self._fillers_armed = False
        self._fillers_played = 0
        self._held_end: tuple[int, int] | None = None   # relay: (t0, t0_end) of speech that ended while transcribing
        self._opening_pending = False                   # the model's opening sentence is in flight
        self._held: list[bytes] = []                    # its audio, held until the disclosure has played
        self._closed = False                            # run() has returned

    # ---- generation bookkeeping ---------------------------------------------------------------
    def _new_gen(self) -> int:
        """One shared generation across all three workers (their counters may differ: cached phrases use TTS alone)."""
        g = max(self.stt.current_gen, self.llm.current_gen, self.tts.current_gen) + 1
        for c in (self.stt, self.llm, self.tts):
            c.set_gen(g)
        self._seq, self._tts_done_seq, self._llm_done, self._reply_parts = 0, -1, False, []
        self._tts_pending = None
        self._pending_stt = 0
        return g

    def request_end(self, reason: str) -> None:
        self.end_reason = reason
        self.state = "ENDING"
        self.hangup_requested.set()

    # ---- transcript, say, relay wait ------------------------------------------------------------
    def _line(self, who: str, text: str, interrupted: bool = False) -> None:
        if self.on_line is not None and text and text.strip():
            self.on_line(who, text.strip(), interrupted)

    def generate(self) -> None:
        msgs = [{"role": "system", "content": self.system_prompt}] + self.history[-2 * self.cfg.history_turns:]
        self.llm.send({"op": "llm.generate", "gen": self.timer.gen, "messages": msgs, "max_tokens": self.cfg.max_tokens})

    def say(self, text: str, end_reason: str | None = None) -> asyncio.Future:
        """Queue text to be spoken once nobody is talking. Resolves to "said", "interrupted" or "ended"."""
        fut = asyncio.get_running_loop().create_future()
        if self._closed or self.state in ("ENDING", "ENDING_DRAIN"):
            fut.set_result("ended")
        else:
            self._say_queue.append((text, end_reason, fut))
        return fut

    def _resolve_active_say(self, result: str) -> None:
        if self._say_active is not None and not self._say_active.done():
            self._say_active.set_result(result)
        self._say_active = None

    def _resolve_says(self, result: str) -> None:
        for _, _, fut in self._say_queue:
            if not fut.done():
                fut.set_result(result)
        self._say_queue = []
        self._resolve_active_say(result)

    def _pump_says(self, now: int) -> None:
        """Speak the next queued text when the line is free: listening, nobody mid-utterance, nothing being said."""
        if not self._say_queue or self._say_active is not None or self.state != "LISTENING" or self.timer is not None:
            return
        text, end_reason, fut = self._say_queue.pop(0)
        if fut.done():
            return
        self._any_say = True
        self._relay_wait_from, self._fillers_armed, self._fillers_played = None, False, 0
        self._say_active, self._end_after_reply = fut, end_reason
        t = self.timer = self._timer()
        # t0_end arms the reply timeout for a stalled TTS; no speech and no model, so no t0, t1 or t2 (and no latency
        # figures for this turn)
        t.t0_end = now
        self.state = "THINKING"
        self.port.arm_t5()
        self.chunker = ClauseChunker()
        self._reply_parts.append(text)
        for tok in re.findall(r"\S+\s*", text):
            for unit in self.chunker.feed(tok):
                self._send_tts(unit, now)
        for unit in self.chunker.flush():
            self._send_tts(unit, now)
        self._llm_done = True
        if self._seq == 0:
            self._finish_turn()  # nothing speakable (only the marker)

    def _holding_for_transcript(self) -> bool:
        """Relay, waiting for the transcript of what they just said: new speech is kept for later, never barge-in."""
        return self.replies.relay and self.state == "THINKING" and self._say_active is None

    def _heard(self, now: int, delivered: bool) -> None:
        """Relay mode: the transcript is with the calling AI (or was empty). Keep listening until its say arrives."""
        t = self.timer
        if delivered:
            t.rx_overruns, t.tx_underruns = self.rx.overruns, self.tx.underruns
            self.on_turn(t)
        self.timer = None
        self._new_gen()
        self.state = "LISTENING"
        self.last_activity_ns = now
        if delivered and not (self.direction == "out" and not self._any_say):
            if self._relay_wait_from is None:
                self._fillers_played = 0
            self._relay_wait_from, self._fillers_armed = now, True
        candidate, self._candidate = self._candidate, []
        held_end, self._held_end = self._held_end, None
        self.bargein.reset()
        self._interrupt_t = None
        if candidate:  # they went on talking while the last words were transcribed
            self.timer = self._timer()
            for pcm in candidate:
                self._push_stt(pcm)
            if held_end is not None:  # and they finished: transcribe that too
                self._finalize_replayed(held_end[0], now)

    def _holding_speech(self) -> bool:
        """Speech now is kept for later, never barge-in: the outbound disclosure always plays in full, and in relay mode
        the transcript of what they said before is still pending."""
        return self.state == "OUTBOUND_OPENING" or self._holding_for_transcript()

    def _finalize_replayed(self, t0: int, now: int) -> None:
        """Held speech that had already ended has just been replayed: transcribe it now. The reply timeout counts from
        now, when transcription can start; t0 stays when they stopped talking, for the latency record."""
        self._flush_stt_push()
        self.timer.t0, self.timer.t0_end = t0, now
        self.stt.send({"op": "stt.finalize", "gen": self.stt.current_gen})
        self.state = "THINKING"

    def _arm_outbound_relay_wait(self) -> None:
        """Outbound relay call, before the calling AI's first say: the reply timeout counts from the answer, no fillers."""
        if self.replies.relay and self.direction == "out" and not self._any_say and self._relay_wait_from is None:
            self._relay_wait_from, self._fillers_armed = self._start_ns, False

    def _play_phrase(self, phrase: Phrase) -> None:
        self._queue_cached(list(phrase.frames))
        self._line("agent", phrase.text)

    def _relay_timers(self, now: int) -> None:
        waited = (now - self._relay_wait_from) / 1e9
        if waited >= self.calls.relay_reply_timeout_sec:
            self._relay_timeout(now)
            return
        p = self.relay_phrases
        if not self._fillers_armed or p is None:
            return
        if self._fillers_played == 0 and waited * 1000 >= self.calls.relay_filler_after_ms:
            self._play_phrase(p.filler)
            self._fillers_played = 1
        elif self._fillers_played == 1 and waited >= self.calls.relay_second_filler_sec:
            self._play_phrase(p.still_here)
            self._fillers_played = 2

    def _relay_timeout(self, now: int) -> None:
        _log.warning("relay: no reply from the calling AI in %ss; ending the call", self.calls.relay_reply_timeout_sec)
        self._relay_wait_from = None
        if self.relay_phrases is not None:
            self._play_phrase(self.relay_phrases.goodbye)
        self.end_reason = "relay_timeout"
        self.state = "ENDING_DRAIN"
        self._drain_deadline_ns = now + int(5e9)

    # ---- the model's opening sentence (phone agent mode, outbound) ------------------------------------
    def _start_opening(self) -> None:
        self.history.append({"role": "user", "content": OPENING_CUE})
        t = self.timer = self._timer()
        t.t0_end = time.monotonic_ns()  # the reply timeout counts from the answer; no speech, so no latency figure
        self.chunker = ClauseChunker()
        self._opening_pending = True
        self.generate()

    def _cancel_opening(self) -> None:
        t = self.timer
        if t is not None:
            for c in (self.llm, self.tts):
                c.cancel(t.gen)
        self.tts_out.flush()
        self._held = []
        self.history.append({"role": "assistant", "content": "[interrupted]"})
        self._opening_pending = False
        self.timer = None
        self._new_gen()

    # ---- main loop ----------------------------------------------------------------------------
    def _timer(self) -> TurnTimer:
        t = TurnTimer(gen=self.stt.current_gen)
        t.direction = self.direction
        return t

    async def run(self, stop: asyncio.Event) -> None:
        self.state = "OUTBOUND_OPENING" if self.opening_frames is not None else "GREETING"
        self._queue_cached(self.opening_frames if self.opening_frames is not None else self.greeting)
        self._line("agent", self.opening_text)
        self._new_gen()
        self._start_ns = time.monotonic_ns()
        self._rtp_baseline = self._start_ns
        if self.generate_opening:
            self._start_opening()
        try:
            while not stop.is_set() and self.state != "ENDING":
                now = time.monotonic_ns()
                while (f := self.rx.pop()) is not None:
                    self._on_frame(f, now)
                self._push_after_flush()
                self._pump_tts_out()
                self._drain_worker_events()
                self._pump_says(now)
                if self.state in OPENING and self.tx.available() == 0 and not self._after_flush:
                    self._greeting_over(now)
                playing = getattr(self.port, "playing", None)
                if playing is not None:
                    self.port.playing = self.state in OPENING or self.state == "SPEAKING"
                if (self.state == "SPEAKING" and self._llm_done and self._tts_done_seq >= self._seq - 1
                        and self.tx.available() == 0 and self.tts_out.available() == 0
                        and self._tts_pending is None and not self._after_flush):
                    self._finish_turn()
                if self.state == "ENDING_DRAIN" and ((self.tx.available() == 0 and not self._after_flush and not self._flush_pending())
                                                     or now >= self._drain_deadline_ns):
                    self.state = "ENDING"  # the goodbye has been handed to PJSIP (or we waited long enough): now the BYE
                    self.hangup_requested.set()
                self._check_timers(now)
                await asyncio.sleep(0.005)
        finally:
            self._closed = True
            self._resolve_says("ended")

    def _check_timers(self, now: int) -> None:
        """All call timers live here, none in PJSIP. Warning fires 30 s before the cap (0.5 s for caps <= 30 s)."""
        idle_s = (now - self.last_activity_ns) / 1e9
        if self.state == "LISTENING" and not self._prompted and idle_s >= self.cfg.silence_prompt_sec \
                and self._relay_wait_from is None:
            self._queue_cached(self.prompt_frames)
            self._prompted = True
            self.last_activity_ns = now
        elif self.state == "LISTENING" and self._prompted and idle_s >= self.cfg.silence_hangup_sec \
                and self._relay_wait_from is None:
            self.request_end("silence")
        if self._relay_wait_from is not None and self.state == "LISTENING" and self.timer is None:
            self._relay_timers(now)
        t = self.timer
        if self.state == "THINKING" and t is not None and t.t0_end is not None \
                and (now - t.t0_end) / 1e9 >= self.cfg.reply_timeout_sec:
            # STT, the model or TTS has gone quiet for longer than any caller waits: same recovery as a brain error
            _log.warning("no reply %ss after the caller finished (stt=%s llm=%s); treating the brain as stalled",
                         self.cfg.reply_timeout_sec, t.t1 is not None, t.t2 is not None)
            self._brain_error(now, "timeout")
        elapsed_s = (now - self._start_ns) / 1e9
        warn_at = self.cfg.max_call_sec - (30 if self.cfg.max_call_sec > 30 else 0.5)
        if not self._warned and elapsed_s >= warn_at:
            self._queue_cached(self.warning_frames)
            self._warned = True
        if elapsed_s >= self.cfg.max_call_sec:
            self.request_end("max_duration")
        if self.rtp_last_rx_ns is not None:
            last_rx = max(self.rtp_last_rx_ns(), self._rtp_baseline)  # a stale or zero stamp must not fire the timeout
            if (now - last_rx) / 1e9 >= self.cfg.rtp_timeout_sec:
                self.request_end("rtp_timeout")

    # ---- inbound audio ------------------------------------------------------------------------
    def _on_frame(self, frame: bytes, t_ns: int) -> None:
        if self._rx_dump is not None:
            self._rx_dump.write(frame)
        for e in self.ep.feed(frame, t_ns):
            if _DEBUG_RX and isinstance(e, (SpeechStart, SpeechEnd)):
                _log.info("endpointer %s state=%s%s", type(e).__name__, self.state,
                          f" utter={len(e.utterance_f32) / 16000:.2f}s" if isinstance(e, SpeechEnd) else "")
            if isinstance(e, SpeechChunk):
                rms = float(np.sqrt(np.mean(e.pcm_f32 * e.pcm_f32)))
                playing = self.tx.available() > 0 or self.state in OPENING or self.state == "SPEAKING"
                if self.state in OPENING or self.state in ("THINKING", "SPEAKING"):
                    # not listening: remember the audio so the interrupting utterance keeps its onset
                    if self._interrupt_t is None:
                        self._interrupt_t = e.t_ns
                    self._candidate.append(e.pcm_f32)
                    if _DEBUG_RX and len(self._candidate) <= 40:
                        _log.info("bargein-in %s rms=%.4f noise=%.4f playing=%s run=%d tx_avail=%d", self.state, rms, self.ep.noise_rms,
                                  playing, self.bargein._run, self.tx.available())
                    if self.bargein.update(True, rms, self.ep.noise_rms, playing) and not self._holding_speech():
                        self._interrupt(t_ns)  # replays the candidate audio, including this chunk
                    continue
                self.bargein.update(True, rms, self.ep.noise_rms, playing)
            else:
                self.bargein.update(False, 0.0, self.ep.noise_rms, False)
                self._interrupt_t = None
                if isinstance(e, SpeechEnd) and self.state == "GREETING" and self._candidate:
                    self._flush_tx()  # they have spoken; answering beats finishing the greeting
                    self._greeting_over(t_ns)  # replays what was heard, then the SpeechEnd below finalises it
                elif self._holding_speech():  # keep all of it: replayed after the disclosure or once the transcript is in
                    if isinstance(e, SpeechEnd) and self._candidate:
                        self._held_end = (e.t0_ns, t_ns)  # _greeting_over or _heard replays and finalises it
                    elif isinstance(e, SpeechStart):
                        self._held_end = None  # they went on: finalise only once this utterance has ended too
                else:
                    self._candidate = []
            if isinstance(e, SpeechStart):
                self.last_activity_ns = t_ns
                self._prompted = False  # the caller is back
                if self.state == "LISTENING":
                    self.timer = self._timer()
            elif isinstance(e, SpeechChunk) and self.state == "LISTENING":
                self._push_stt(e.pcm_f32)
            elif isinstance(e, SpeechEnd) and self.state == "LISTENING":
                self._flush_stt_push()
                if self.timer is None:
                    self.timer = self._timer()
                self.timer.t0 = e.t0_ns
                self.timer.t0_end = t_ns
                self.stt.send({"op": "stt.finalize", "gen": self.stt.current_gen})
                self.state = "THINKING"

    def _greeting_over(self, now: int) -> None:
        """The opening finished (an inbound greeting may also have been cut). Anything said meanwhile becomes the start
        of their turn, and is transcribed at once if they had already finished."""
        candidate, self._candidate = self._candidate, []
        held_end, self._held_end = self._held_end, None
        self.last_activity_ns = now
        self._interrupt_t = None
        self.bargein.reset()
        self._arm_outbound_relay_wait()
        held_sec = sum(len(pcm) for pcm in candidate) / 16000
        if candidate and self._opening_pending and self.state == "OUTBOUND_OPENING" and held_end is not None \
                and held_sec < GREETING_MAX_SEC:
            # the "Hello?" at pickup: the model's opening answers it, so the held audio is dropped, not transcribed
            _log.info("short speech during the disclosure treated as a greeting (%.1f s of audio)", held_sec)
            candidate = []
        if candidate:
            if self._opening_pending:
                self._cancel_opening()  # they said more, or are still talking: answer them, not the cue
            self.state = "LISTENING"
            self.timer = self._timer()
            for pcm in candidate:
                self._push_stt(pcm)
            if held_end is not None:
                self._finalize_replayed(held_end[0], now)
            return
        if self._opening_pending:  # the model's first sentence follows the disclosure without a gap
            self.port.arm_t5()
            had_audio = bool(self._held)
            self._queue_cached(self._held)
            self._held = []
            if self._llm_done and self._seq == 0:
                self._finish_turn()  # the model had nothing to add
            else:
                self.state = "SPEAKING" if had_audio else "THINKING"
            return
        self.state = "LISTENING"

    def _push_stt(self, pcm_f32: np.ndarray) -> None:
        self.stt_in.push(tag_frame(self.stt.current_gen, (np.clip(pcm_f32, -1, 1) * 32767).astype("<i2").tobytes()))
        self._pending_stt += 1
        if self._pending_stt >= 10:
            self._flush_stt_push()

    def _interrupt(self, now: int) -> None:
        """They talked over us (opening, reply or say): cancel this generation, flush audio, keep the partial reply,
        and replay the interrupting speech into the new generation so its first 250 ms are not lost."""
        was_greeting = self.state in OPENING
        t = self.timer
        if t is not None and (not was_greeting or self._opening_pending):
            for c in (self.stt, self.llm, self.tts):
                c.cancel(t.gen)
        self._flush_tx()
        self.tts_out.flush()  # we are its consumer
        self._held = []
        if t is not None and not was_greeting:
            t.bargein = True
            t.bargein_stop_ns = now - (self._interrupt_t or now)
            t.reply_text = _without_marker("".join(self._reply_parts)).strip()
            self.history.append({"role": "assistant", "content": (t.reply_text + " [interrupted]").strip()})
            t.rx_overruns, t.tx_underruns = self.rx.overruns, self.tx.underruns
            self.on_turn(t)
            self._line("agent", t.reply_text, interrupted=True)
        elif was_greeting and self._opening_pending:
            self.history.append({"role": "assistant", "content": "[interrupted]"})
        self._opening_pending = False
        self._resolve_active_say("interrupted")
        end_now, self._end_after_reply = self._end_after_reply == "hangup_command", None
        candidate, self._candidate = self._candidate, []
        self._new_gen()
        self.bargein.reset()
        self._interrupt_t = None
        self.timer = self._timer()
        self.state = "LISTENING"
        for pcm in candidate:
            self._push_stt(pcm)
        if was_greeting:
            self._arm_outbound_relay_wait()  # leaving an opening by barge-in skips _greeting_over (today only inbound)
        if end_now:
            self.request_end("hangup_command")  # `hangup --say` hangs up even when the goodbye is cut

    def _flush_pending(self) -> bool:
        return bool(getattr(self.port, "flush_pending", False))

    def _queue_cached(self, frames: list[bytes]) -> None:
        """Preserve cached audio until the flush is acknowledged and playback has room for it."""
        self._after_flush.extend(frames)
        self._push_after_flush()

    def _push_after_flush(self) -> None:
        if self._after_flush and not self._flush_pending():
            count = min(len(self._after_flush), self.tx.frames - self.tx.available())
            for f in self._after_flush[:count]:
                self.tx.push(f)
            del self._after_flush[:count]

    def _flush_tx(self) -> None:
        """The PJSIP thread is the tx ring's consumer, so it performs the flush; a stub port flushes directly."""
        self._after_flush = []
        self._tts_pending = None
        req = getattr(self.port, "request_flush", None)
        if req is not None:
            req()
        else:
            self.tx.flush()

    def _flush_stt_push(self) -> None:
        if self._pending_stt:
            self.stt.send({"op": "stt.push", "gen": self.stt.current_gen, "nframes": self._pending_stt})
            self._pending_stt = 0

    # ---- worker events ------------------------------------------------------------------------
    def _drain_worker_events(self) -> None:
        for client in (self.stt, self.llm, self.tts):
            while not client.events.empty():
                self._on_event(client.events.get_nowait())

    def _on_event(self, m: dict) -> None:
        ev, now = m["ev"], time.monotonic_ns()
        if ev == "error":
            self._worker_died(now, m.get("kind", "?"), str(m.get("message", ""))[:200])
            return
        if self.timer is None or m.get("gen") != self.timer.gen:
            return
        if ev == "brain_error":
            self._brain_error(now, m["kind"])
            return
        if ev == "final":
            self.timer.t1 = now
            self.timer.stt_text = m["text"]
            self.timer.worker_timings.update(m.get("timings", {}))
            text = m["text"].strip()
            if not text:
                if self.replies.relay:
                    self._heard(now, delivered=False)  # nothing to hand over; replay anything said meanwhile
                else:
                    self.state = "LISTENING"
                    self.timer = None
                return
            self.history.append({"role": "user", "content": m["text"]})
            self._line("them", text)
            self.chunker = ClauseChunker()
            if self.replies.relay:
                self.replies.on_heard(self, text)
                self._heard(now, delivered=True)
                return
            self.port.arm_t5()
            self.replies.on_heard(self, text)
        elif ev == "token":
            if self.timer.t2 is None:
                self.timer.t2 = now
            self._reply_parts.append(m["text"])
            for unit in self.chunker.feed(m["text"]):
                self._send_tts(unit, now)
        elif ev == "done" and m["kind"] == "llm":
            self.timer.worker_timings.update({f"llm_{k}": v for k, v in m.get("timings", {}).items()})
            for unit in self.chunker.flush():
                self._send_tts(unit, now)
            self._llm_done = True
            if self._seq == 0 and self.state not in OPENING:
                self._finish_turn()  # the model said nothing
        elif ev == "chunk":
            if self.timer.t4 is None:
                self.timer.t4 = now
            if self.state not in OPENING:
                self.state = "SPEAKING"  # during the opening the audio is held; _greeting_over switches state
        elif ev == "done" and m["kind"] == "tts":
            self._tts_done_seq = max(self._tts_done_seq, m["seq"])

    def _worker_died(self, now: int, kind: str, message: str) -> None:
        """A model worker is gone mid-call (crash, kill). The supervisor will restart it for the next call; this call
        cannot continue, but the caller gets the goodbye line, not a silent drop."""
        _log.error("%s worker died mid-call: %s", kind, message)
        t = self.timer
        if t is not None:  # record the turn that was in flight, marked
            t.brain_error = f"worker_{kind}"
            t.reply_text = _without_marker("".join(self._reply_parts)).strip()
            t.rx_overruns, t.tx_underruns = self.rx.overruns, self.tx.underruns
            self.on_turn(t)
            self.timer = None
        for c in (self.stt, self.llm, self.tts):
            if c.kind != kind:
                c.cancel(c.current_gen)
        if self.state not in OPENING:
            self._flush_tx()  # an opening (the disclosure above all) plays in full; the model's held audio is dropped
        self.tts_out.flush()
        self._new_gen()
        self.end_reason = f"worker error: {kind}"
        if self.goodbye_frames:  # the goodbye is cached audio: it needs no worker, only the tx ring
            self._queue_cached(self.goodbye_frames)
            self.state = "ENDING_DRAIN"
            # time for everything queued (an unflushed opening, then the goodbye) at 20 ms a frame, plus 1 s; a flush
            # still owed by the PJSIP thread empties tx, and the goodbye then waits in _after_flush
            queued = len(self._after_flush) + (0 if self._flush_pending() else self.tx.available())
            self._drain_deadline_ns = now + queued * 20_000_000 + int(1e9)
        else:
            self.state = "ENDING"
            self.hangup_requested.set()

    def _brain_error(self, now: int, kind: str) -> None:
        """The brain failed mid-turn. Cancel exactly like a barge-in (nothing half-generated may be spoken),
        then apologise; twice in a row means the brain is gone and the call ends politely."""
        t = self.timer
        self._opening_pending = False
        self._held = []
        self._resolve_active_say("interrupted")
        end_now, self._end_after_reply = self._end_after_reply == "hangup_command", None
        for c in (self.stt, self.llm, self.tts):
            c.cancel(t.gen)
        if self.state not in OPENING:
            self._flush_tx()  # during an opening tx holds only the cached line; the model's audio is held, dropped above
        self.tts_out.flush()
        t.brain_error = kind
        t.reply_text = _without_marker("".join(self._reply_parts)).strip()
        t.rx_overruns, t.tx_underruns = self.rx.overruns, self.tx.underruns
        if self.state in OPENING:  # the opening's audio was held, so none of its text was heard: none of it is kept
            self.history.append({"role": "assistant", "content": "[interrupted]"})
        elif t.reply_text:  # what was already said stays in history and the transcript, marked, exactly as after a barge-in
            self.history.append({"role": "assistant", "content": (t.reply_text + " [interrupted]").strip()})
            self._line("agent", t.reply_text, interrupted=True)
        self.on_turn(t)
        self.timer = None
        self._new_gen()
        if end_now:
            self.request_end("hangup_command")
            return
        self._brain_errors += 1
        _log.warning("brain error %s (%d in a row)", kind, self._brain_errors)
        if self._brain_errors >= 2:
            self._queue_cached(self.goodbye_frames)
            self.end_reason = "brain_error"
            self.state = "ENDING_DRAIN"  # hangup_requested is set once the goodbye has drained (see run())
            self._drain_deadline_ns = now + int(3e9)
            return
        self._queue_cached(self.apology_frames)
        self.last_activity_ns = now
        if self.state not in OPENING:
            self.state = "LISTENING"

    def _send_tts(self, unit: str, now: int) -> None:
        if self.timer.t3 is None:
            self.timer.t3 = now
        self.tts.send({"op": "tts.synth", "gen": self.timer.gen, "seq": self._seq, "text": unit})
        self._seq += 1

    def _pump_tts_out(self) -> None:
        cur = self.tts.current_gen & 0xFFFFFFFF
        while True:
            f, self._tts_pending = self._tts_pending, None
            if f is None:
                f = self.tts_out.pop()
            if f is None:
                break
            g, payload = untag_frame(f)
            if g != cur:
                continue  # audio from a cancelled generation is dropped here, whatever its timing
            if self.timer is not None and self.timer.t4 is None:
                # the frames are available the moment they are here. The worker pushes them before it sends `chunk`,
                # so the port can stamp t5 before that event is drained; taking t4 from the event made egress_ms
                # negative (and t5 < t4) whenever the pump won that race.
                self.timer.t4 = time.monotonic_ns()
            if self.state in OPENING:
                self._held.append(payload)  # the model's opening waits for the disclosure to finish
            else:
                if self._flush_pending() or self._after_flush or self.tx.available() >= self.tx.frames:
                    self._tts_pending = f
                    break
                self.tx.push(payload)
        # accept the port's stamp only once this turn has sent TTS (t3), so a previous turn's stamp is never reused
        if (self.timer is not None and self.timer.t5 is None and self.timer.t3 is not None
                and self.port.first_nonsilent_ns and self.port.first_nonsilent_ns >= self.timer.t3):
            self.timer.t5 = self.port.first_nonsilent_ns

    def incomplete_record(self, call_id: str, turn: int) -> dict:
        """The turn that was still running when the call ended, recorded the way every finished one is: what the agent
        had already said, and the ring counters. A call that ends mid-sentence must not lose the reply."""
        t = self.timer
        t.reply_text = _without_marker("".join(self._reply_parts)).strip()
        t.rx_overruns, t.tx_underruns = self.rx.overruns, self.tx.underruns
        return t.to_record(call_id, turn)

    def _finish_turn(self) -> None:
        t = self.timer
        t.reply_text = _without_marker("".join(self._reply_parts)).strip()
        t.rx_overruns, t.tx_underruns = self.rx.overruns, self.tx.underruns
        if t.reply_text:
            self.history.append({"role": "assistant", "content": t.reply_text})
            self._line("agent", t.reply_text)
        self.on_turn(t)
        self.timer = None
        self._brain_errors = 0  # a completed turn ends the streak
        self.last_activity_ns = time.monotonic_ns()
        # A stray marker in an ordinary answer must not disconnect the caller. A marker in say text
        # is still removed without being obeyed; explicit hangup commands retain their own end reason.
        model_ended = (self.chunker.ended and self._say_active is None
                       and is_closing_reply("".join(self._reply_parts)))
        end = self._end_after_reply or ("agent" if model_ended else None)
        self._resolve_active_say("said")
        self._end_after_reply = None
        self._opening_pending = False
        self._new_gen()
        self.state = "LISTENING"
        if end:
            self.request_end(end)  # the reply has played in full (tx is empty): now the BYE

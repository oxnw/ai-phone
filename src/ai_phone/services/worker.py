"""Generic model-worker process loop: one backend, JSON control over a pipe, audio over shared-memory rings.

Cancellation is exact: every request carries a gen_id; a cancelled gen_id is remembered and any
later work for it is dropped, mid-stream checks included.
"""
from __future__ import annotations

import importlib
import os
import sys
import time
import traceback

import numpy as np

from ai_phone.services import ipc
from ai_phone.services.errors import BrainError
from ai_phone.services.ipc import STT_IN_FRAME, tag_frame, tts_out_frame_bytes, untag_frame
from ai_phone.telephony.rings import FrameRing


def _load(factory: str, factory_args: dict | None = None):
    mod, _, name = factory.partition(":")
    return getattr(importlib.import_module(mod), name)(**(factory_args or {}))


def _drain_stt(ring: FrameRing, gen: int) -> list[bytes]:
    """Pop every queued STT frame; keep the payloads tagged with `gen`, discard the rest."""
    out = []
    want = gen & 0xFFFFFFFF
    seen: dict[int, int] = {}
    while (f := ring.pop()) is not None:
        g, payload = untag_frame(f)
        seen[g] = seen.get(g, 0) + 1
        if g == want:
            out.append(payload)
    if os.environ.get("AI_PHONE_DEBUG_RX"):
        print(f"[stt-worker] drain gen={want} kept={len(out)} seen_by_tag={seen} ring_over={ring.overruns}", file=sys.stderr, flush=True)
    return out


def _drain_cancel(conn, cancelled: set[int], backend, gen: int, pending: list[dict]) -> bool:
    """Apply any cancel messages waiting on the pipe; queue every other message for the main loop.
    Returns True if `gen` is now cancelled."""
    while conn.poll(0):
        m = ipc.recv(conn)
        if m.get("op") == "cancel":
            cancelled.add(m["gen"])
            backend.cancel(m["gen"])
        else:
            pending.append(m)
    return gen in cancelled


def _tts_stopped(conn, cancelled: set[int], backend, gen: int, pending: list[dict]) -> bool:
    if conn.poll(0):
        _drain_cancel(conn, cancelled, backend, gen, pending)
    return gen in cancelled or any(m.get("op") == "shutdown" for m in pending)


def _synth(conn, backend, out: FrameRing, msg: dict, cancelled: set[int], pending: list[dict]) -> None:
    """Apply backpressure in this worker, keeping the media thread free to consume at its own pace."""
    gen, seq = msg["gen"], msg["seq"]
    if _tts_stopped(conn, cancelled, backend, gen, pending):
        return
    source = backend.stream(gen, seq, msg["text"])
    unannounced = 0
    batch = min(5, out.frames)

    def announce() -> None:
        nonlocal unannounced
        if unannounced:
            ipc.send(conn, {"ev": "chunk", "gen": gen, "seq": seq, "nframes": unannounced})
            unannounced = 0

    try:
        for frame in source:
            if _tts_stopped(conn, cancelled, backend, gen, pending):
                break
            while out.available() >= out.frames:
                # Startup phrase caching drains in response to events, including when the ring is smaller
                # than the usual batch. Announce every queued frame before waiting for the consumer.
                announce()
                conn.poll(0.01)
                if _tts_stopped(conn, cancelled, backend, gen, pending):
                    break
            if _tts_stopped(conn, cancelled, backend, gen, pending):
                break
            out.push(tag_frame(gen, frame))
            unannounced += 1
            if unannounced >= batch:
                announce()
    finally:
        close = getattr(source, "close", None)
        if close is not None:
            close()
    announce()
    ipc.send(conn, {"ev": "done", "gen": gen, "seq": seq, "timings": getattr(backend, "last_timings", {})})


def _keep_warm(kind: str, backend) -> float:
    """One tiny inference on an idle worker. On a host that is short of memory macOS compresses whatever was used least
    recently; a model that answers a call every few hours is exactly that, and the first turn then waits seconds for
    page-ins (measured 9-10 s to the first transcript). Touching the weights every keep_warm_sec keeps them recent."""
    fn = getattr(backend, "keep_warm", None)
    if fn is None:
        return 0.0
    t = time.monotonic_ns()
    try:
        fn()
    except Exception as e:  # a failed keep-warm is not a failed worker
        print(f"[{kind}-worker] keep_warm failed: {e!r}", file=sys.stderr, flush=True)
        return 0.0
    _trim_memory()
    took = (time.monotonic_ns() - t) / 1e9
    if took > 1.0 or os.environ.get("AI_PHONE_DEBUG_RX"):
        print(f"[{kind}-worker] keep_warm {took * 1e3:.0f} ms", file=sys.stderr, flush=True)
    return took


def _trim_memory(first: bool = False) -> None:
    """Hand MLX's cached scratch buffers back to the OS. After warm-up the STT worker still holds the float32 copy of
    its checkpoint (2.5 GB) in that cache and the LLM worker its warm-up KV scratch; on a 24 GB host that is the
    difference between the models staying resident and the pager compressing them between calls. Idle workers must
    hold the weights and little else, so we cap the cache and clear it after every request."""
    try:
        import mlx.core as mx
    except Exception:  # Linux backends: nothing to trim
        return
    if first:
        mx.set_cache_limit(64 * 1024 * 1024)
    mx.clear_cache()


def _release(kind: str, backend) -> None:
    """End whatever the backend holds open (a CLI session). Everything else has nothing to release."""
    close = getattr(backend, "close", None)
    if close is None:
        return
    try:
        close()
    except Exception as e:  # a backend that will not let go must not take the worker down with it
        print(f"[{kind}-worker] close failed: {e!r}", file=sys.stderr, flush=True)


def run_worker(kind: str, backend_factory: str, conn, audio_in_name, audio_out_name, warm_args: dict, factory_args: dict | None = None,
               keep_warm_sec: float = 0.0) -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from ai_phone.services.client import is_secret_env
    for name in [n for n in os.environ if is_secret_env(n)]:
        del os.environ[name]  # belt and braces: a worker never holds a key, token or password, whatever the parent exported
    ain = aout = None
    cancelled: set[int] = set()
    pending: list[dict] = []  # messages read during a mid-stream cancel check, processed next
    interval = float(keep_warm_sec)  # current keep-warm cadence (backs off while ticks are slow)
    try:
        ain = FrameRing.attach(audio_in_name, frame_bytes=STT_IN_FRAME if kind == "stt" else 640) if audio_in_name else None
        output_frame_bytes = tts_out_frame_bytes((factory_args or {}).get("output_rate", 16000)) if kind == "tts" else 640
        aout = FrameRing.attach(audio_out_name, frame_bytes=output_frame_bytes) if audio_out_name else None
        backend = _load(backend_factory, factory_args)
        timings = backend.warmup(**warm_args) if warm_args else backend.warmup()
        _trim_memory(first=True)
        ipc.send(conn, {"ev": "ready", "timings": timings})
        while True:
            if pending:
                msg = pending.pop(0)
            else:
                while keep_warm_sec > 0 and not conn.poll(interval):
                    took = _keep_warm(kind, backend)  # idle: touch the model so the pager evicts something else first
                    # a tick that took seconds means the host is thrashing; a request arriving mid-tick waits for it,
                    # so back off (up to 10 min) and return to the configured cadence once ticks are quick again
                    interval = min(interval * 2, 600.0) if took > 1.0 else keep_warm_sec
                msg = ipc.recv(conn)
            op = msg.get("op")
            if op == "shutdown":
                _release(kind, backend)
                break
            if op == "cancel":
                cancelled.add(msg["gen"])
                backend.cancel(msg["gen"])
                continue
            gen = msg.get("gen")
            if op == "stt.push":
                # drain everything queued (counts cannot desynchronise); the tag decides which frames are ours
                frames = b"".join(_drain_stt(ain, gen))
                if gen in cancelled or not frames:
                    if os.environ.get("AI_PHONE_DEBUG_RX"):
                        print(f"[stt-worker] push gen={gen} skipped (cancelled={gen in cancelled}, frames={len(frames)})", file=sys.stderr, flush=True)
                    continue
                pcm = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
                partial = backend.push(gen, pcm)
                if partial:
                    ipc.send(conn, {"ev": "partial", "gen": gen, "text": partial})
                continue
            if op == "llm.prepare":
                # A call's instructions are known before its greeting plays: a brain that starts a session (a
                # subscription CLI) starts it now, so the first reply does not pay the startup. Deliberately before
                # the cancelled check: the gen is the one the last call ended on, which was cancelled as it ended.
                prepare = getattr(backend, "prepare", None)  # the local and endpoint brains need no preparing
                if prepare is not None:
                    try:
                        prepare(msg["system_prompt"])
                    except BrainError as e:
                        # the controller has no turn to attach this to yet, so say it here too: a login that has gone
                        # or a CLI that has been uninstalled is worth one line in the log at the moment it is found
                        print(f"[{kind}-worker] llm.prepare failed: {e.kind}", file=sys.stderr, flush=True)
                        ipc.send(conn, {"ev": "brain_error", "gen": gen, "kind": e.kind})
                continue
            if op == "llm.release":
                # the call is over: a brain that holds a session (a subscription CLI) ends it now, so no process of
                # ours sits idle between calls. The next call's llm.prepare starts a fresh one.
                _release(kind, backend)
                continue
            if op == "llm.generate" and conn.poll(0):
                # Everything above is synchronous, and `prepare()` of a subscription brain is a second of it: a
                # caller who hangs up during session startup has their generate, cancel and release waiting in the
                # pipe together. Reading the cancels first is what makes the hangup free — one turn of a plan is sent
                # the moment backend.stream() is asked for a token, and the mid-stream check comes too late for it.
                _drain_cancel(conn, cancelled, backend, gen, pending)
            if gen in cancelled:
                continue
            if op == "llm.generate" and any(m.get("op") in ("llm.release", "shutdown") for m in pending):
                continue  # the call is already over; the release waiting behind this turn is the proof
            if op == "stt.finalize":
                tail = b"".join(_drain_stt(ain, gen))  # frames pushed after the last announcement
                if tail:
                    backend.push(gen, np.frombuffer(tail, dtype="<i2").astype(np.float32) / 32768.0)
                text, t = backend.finalize(gen)
                ipc.send(conn, {"ev": "final", "gen": gen, "text": text, "timings": t})
                _trim_memory()
            elif op == "llm.generate":
                try:
                    for tok in backend.stream(gen, msg["messages"], msg["max_tokens"]):
                        if gen in cancelled or (conn.poll(0) and _drain_cancel(conn, cancelled, backend, gen, pending)):
                            break
                        ipc.send(conn, {"ev": "token", "gen": gen, "text": tok})
                except BrainError as e:  # recoverable: tell the controller, keep serving
                    ipc.send(conn, {"ev": "brain_error", "gen": gen, "kind": e.kind})
                    continue
                ipc.send(conn, {"ev": "done", "gen": gen, "timings": getattr(backend, "last_timings", {})})
                _trim_memory()
            elif op == "tts.synth":
                _synth(conn, backend, aout, msg, cancelled, pending)
                _trim_memory()
    except BrainError as e:
        # A brain that will not warm up or will not prepare fails here, outside the per-turn handler. Only the kind
        # travels: the traceback would carry the provider's own words into the daemon's log, where the already-fixed
        # wizard page no longer puts them. The exit code is what the supervisor reads.
        print(f"[{kind}-worker] the brain failed: {e.kind}", file=sys.stderr, flush=True)
        ipc.send(conn, {"ev": "error", "message": f"the brain failed: {e.kind}"})
        raise SystemExit(1) from None
    except Exception as e:  # report, then exit non-zero so the supervisor restarts us
        ipc.send(conn, {"ev": "error", "message": f"{e}\n{traceback.format_exc()}"})
        raise
    finally:
        for ring in (ain, aout):
            if ring is not None:
                ring.close()  # the main process owns and unlinks the shared memory

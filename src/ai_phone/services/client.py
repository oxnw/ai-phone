"""Main-process handle to one worker: spawns it, delivers replies to an asyncio queue, tracks gen_ids."""
from __future__ import annotations

import asyncio
import contextlib
import multiprocessing as mp
import os
import re

from ai_phone.services import ipc
from ai_phone.services.worker import run_worker
from ai_phone.telephony.rings import FrameRing


SECRET_ENV_RE = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH)", re.IGNORECASE)
SECRET_ENV_PREFIXES = ("AI_PHONE_LINE_", "TELNYX_", "TWILIO_", "SIGNALWIRE_")


def is_secret_env(name: str) -> bool:
    return name.startswith(SECRET_ENV_PREFIXES) or bool(SECRET_ENV_RE.search(name))


@contextlib.contextmanager
def scrubbed_environment():
    """Spawned children copy the parent's environment at start(); hide anything that looks like a secret meanwhile."""
    hidden = {k: os.environ.pop(k) for k in list(os.environ) if is_secret_env(k)}
    try:
        yield
    finally:
        os.environ.update(hidden)


class WorkerClient:
    def __init__(self, kind, backend_factory, loop, audio_in: FrameRing | None, audio_out: FrameRing | None, warm_args: dict,
                 factory_args: dict | None = None, keep_warm_sec: float = 0.0):
        self.kind, self.factory, self.loop, self.warm_args = kind, backend_factory, loop, warm_args
        self.factory_args = factory_args
        self.keep_warm_sec = keep_warm_sec
        self.audio_in, self.audio_out = audio_in, audio_out
        self.events: asyncio.Queue = asyncio.Queue()
        self._gen = 0
        self._ctx = mp.get_context("spawn")
        self._parent, self._child = self._ctx.Pipe()
        self._proc = None

    @property
    def current_gen(self) -> int:
        return self._gen

    def next_gen(self) -> int:
        self._gen += 1
        return self._gen

    def gen_is_current(self, gen: int) -> bool:
        return gen == self._gen

    def set_gen(self, gen: int) -> None:
        """Align this worker's current generation with the others (the controller owns the shared counter)."""
        self._gen = gen

    async def start(self) -> dict:
        self._proc = self._ctx.Process(
            target=run_worker, name=f"{self.kind}-worker", daemon=True,
            args=(self.kind, self.factory, self._child,
                  self.audio_in.name if self.audio_in else None,
                  self.audio_out.name if self.audio_out else None, self.warm_args, self.factory_args, self.keep_warm_sec))
        with scrubbed_environment():  # no carrier key, line password or token reaches a model worker
            self._proc.start()
        self._child.close()  # the child owns its end now
        ready = await self.loop.run_in_executor(None, ipc.recv, self._parent)
        if ready.get("ev") != "ready":
            raise RuntimeError(f"{self.kind} worker failed: {ready}")
        self.loop.add_reader(self._parent.fileno(), self._on_readable)
        return ready["timings"]

    def _on_readable(self) -> None:
        while self._parent.poll(0):
            try:
                msg = ipc.recv(self._parent)
            except (EOFError, OSError) as e:  # the worker died: stop reading a dead pipe and report it once
                try:
                    self.loop.remove_reader(self._parent.fileno())
                except Exception:
                    pass
                self.events.put_nowait({"ev": "error", "kind": self.kind, "message": f"{self.kind} worker disconnected: {e!r}"})
                return
            msg.setdefault("kind", self.kind)  # an event's own kind wins: brain_error carries the failure kind
            if "gen" in msg and msg["gen"] != self._gen and msg["ev"] != "error":
                continue  # stale generation: drop on receipt
            self.events.put_nowait(msg)

    def send(self, msg: dict) -> None:
        ipc.send(self._parent, msg)

    def cancel(self, gen: int) -> None:
        self.send({"op": "cancel", "gen": gen})

    async def stop(self) -> None:
        try:
            self.loop.remove_reader(self._parent.fileno())
            self.send({"op": "shutdown"})
        except Exception:
            pass
        if self._proc:
            self._proc.join(5)
            if self._proc.is_alive():
                self._proc.kill()
        try:
            self._parent.close()
        except Exception:
            pass

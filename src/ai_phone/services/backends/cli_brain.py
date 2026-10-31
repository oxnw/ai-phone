"""A person's own Claude or ChatGPT plan as the brain, through the CLI they are already logged into.

`ClaudeCodeLLM` drives `claude -p` in stream-json mode; `CodexLLM` drives `codex app-server`, which speaks JSON-RPC
2.0 over stdio. Neither needs an API key: the CLI uses the login the person already has, and the plan's limits are
the limits.

One session per call. The session remembers the instructions it was started with and the conversation it has been
given, so the next turn of the same call sends only the newest thing the caller said; different instructions, or a
history that is not the continuation of what this session has heard, start a new session. `prepare()` starts the
session for a call before the greeting plays, so the first reply does not pay the startup.

A caller's words are model input, so a session that can reach a tool is a way into this Mac from the phone. Both
sessions are started tool-free and both refuse to run otherwise: Claude Code takes `--tools ""` and says what it came
up with, and Codex is given a home of its own and every tool feature turned off, then watched: a tool call, an MCP
server or an approval request ends the turn as a `tool_use` failure instead of answering the caller.

Failures raise BrainError with a kind the controller logs, exactly like the endpoint brain; there are no retries,
because the caller is waiting on the line. A provider's own words never travel in the exception: the kind is what the
daemon and the wizard see, the text goes to the debug log. What leaves this Mac is the instructions, what callers say
and the conversation (not audio), through the person's own CLI to their provider, on the login they already have and
never on an API key (the variables that would re-route either CLI are dropped at the subprocess boundary).
"""
from __future__ import annotations

import json
import logging
import os
import queue
import re
import shutil
import signal
import stat
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ai_phone import paths
from ai_phone.lines.journal import private_mkdir
from ai_phone.services.errors import BrainError

if TYPE_CHECKING:  # only for the annotations: the setup wizard imports this module and needs no array library
    from ai_phone.services.base import Message

log = logging.getLogger("ai-phone")

PROBE = "Say OK."              # the one tiny turn warm-up spends to prove the login works
PROBE_PROMPT = "You are a phone assistant. Answer in one short sentence."
CODEX_EFFORT = "low"           # measured: the first word takes about four seconds at any reasoning effort, so take
                               # the one every current model accepts rather than the person's own configured default
DRAIN_MS = 2000                # how long a turn we stopped reading may take to finish before the session is dropped;
                               # the daemon sets this from the caller's own patience (conversation.reply_timeout_sec)
STDOUT_BUDGET = 1 << 20        # bytes of output one call may print (plus the reply cap) before we take it for a runaway
HANDSHAKE_SEC = 10.0           # how long starting a session may take, when that is longer than the token deadline
CLOSE_SEC = 2.0                # how long the CLI may take to exit after its stdin closes
VERSION_SEC = 15.0             # how long `<cli> --version` may take, and `claude auth status`
STDERR_WAIT_SEC = 0.3          # how long a CLI that has already exited may take to finish saying why
STDERR_CHARS = 400             # how much of one of the CLI's own stderr lines reaches the log

# The logins ai-phone will run on. `claude auth status --json` reports exactly one of none | third_party | claude.ai |
# api_key_helper | oauth_token | api_key (read from the CLI 2.1.286 that builds that payload), and only the first is a
# subscription this feature can promise: an API key and a key helper are billed per request, third_party is somebody
# else's gateway, and oauth_token covers a token in the environment or an AWS profile, neither of which is the login the
# person is being told ai-phone uses. An allowlist, so a value added in a later version is refused rather than accepted.
CLAUDE_SUBSCRIPTION_LOGINS = frozenset({"claude.ai"})
CLAUDE_BILLED_LOGINS = frozenset({"api_key", "api_key_helper"})
CLAUDE_LOGINS = CLAUDE_SUBSCRIPTION_LOGINS | CLAUDE_BILLED_LOGINS | {"none", "third_party", "oauth_token"}
# `account/read`'s own enum, from the app-server schema: an account is one of these three and nothing else.
CODEX_ACCOUNTS = frozenset({"chatgpt", "apiKey", "amazonBedrock"})
UNNAMED = "unrecognised"   # what a value that is not one of the above is called on a page: never the value itself

# Anything that would move a CLI off the login the person has: a key or token it would bill instead of the plan, and a
# base URL or token file that would send the call somewhere else. `ANTHROPIC_API_KEY` is the one that matters most --
# Claude Code prefers it over the keychain login, so with it set the whole premise of this brain is quietly false and
# every call is charged per token. None of these crosses into either CLI.
PROVIDER_ENV_PREFIXES = ("ANTHROPIC_", "OPENAI_", "AWS_", "AZURE_", "GOOGLE_", "GCLOUD_", "VERTEX_", "BEDROCK_", "SIGNALWIRE_")
PROVIDER_ENV_SUFFIXES = ("_API_KEY", "_API_TOKEN", "_AUTH_TOKEN", "_ACCESS_TOKEN", "_BASE_URL", "_TOKEN_FILE", "_API_BASE")

# Codex: the feature flags that carry its tools (`codex features list`; `--disable <name>` is `-c features.<name>=false`).
# There is no single switch and no `tools` setting for them -- the app-server's own `tools` table holds only web search
# -- so a phone brain names every one of them.
CODEX_TOOLS_OFF = ("shell_tool", "unified_exec", "unified_exec_tty", "view_image", "image_generation", "sleep_tool",
                   "browser_use", "browser_use_external", "browser_use_full_cdp_access", "in_app_browser",
                   "computer_use", "apps", "plugins", "hooks", "multi_agent", "goals", "worktrees", "shell_snapshot",
                   "skill_search", "skill_mcp_dependency_install", "tool_suggest", "tool_call_mcp_elicitation",
                   "memories", "realtime_conversation", "code_mode", "code_mode_host")
# The only thread items a tool-free session produces; every other kind in the protocol is a tool having been called.
# `plan` is not here because the one tool that could produce it is off in the config above (measured: Codex warns about
# a setting it does not recognise, and it warns about none of these).
CODEX_SAID = frozenset({"userMessage", "agentMessage", "reasoning", "contextCompaction"})
# Notifications that mean something ran. These sit outside the item lifecycle, so the item allowlist below does not
# see them: they are a second way to notice a tool that the feature flags and the private home were supposed to prevent.
CODEX_RAN = frozenset({"hook/started", "hook/completed", "command/exec/outputDelta", "process/outputDelta",
                       "process/exited", "item/commandExecution/outputDelta", "item/commandExecution/terminalInteraction",
                       "item/fileChange/outputDelta", "item/fileChange/patchUpdated", "item/mcpToolCall/progress",
                       "item/plan/delta", "item/autoApprovalReview/started", "item/autoApprovalReview/completed"})
# requests the app-server makes of its client on a tool's behalf. Each one blocks the turn until it is answered, so
# each gets an answer and ends the turn. (`account/chatgptAuthTokens/refresh` and `attestation/generate` are the
# server's own business and are deliberately not here.)
CODEX_APPROVALS = frozenset({"item/commandExecution/requestApproval", "item/fileChange/requestApproval",
                             "item/permissions/requestApproval"})
CODEX_ASKS = CODEX_APPROVALS | {"item/tool/call", "item/tool/requestUserInput", "mcpServer/elicitation/request",
                                "openai/form"}
# the home a session is given: the person's own holds their MCP servers, hooks, plugins, skills and AGENTS.md, and
# `-c mcp_servers={}` does not remove them (measured: ten of the owner's started inside our thread), so a session gets
# a home with nothing in it but this file and a link to their login.
CODEX_CONFIG = """# written by ai-phone for its own phone sessions; not the person's own Codex configuration
web_search = "disabled"
project_doc_max_bytes = 0
include_environment_context = false
include_apps_instructions = false
include_permissions_instructions = false
include_collaboration_mode_instructions = false
check_for_update_on_startup = false

[tools]
web_search = false
update_plan = { enabled = false }
experimental_request_user_input = { enabled = false }
"""


@dataclass(frozen=True)
class CliModel:
    """A model the person's plan offers, for the setup wizard's menu."""
    id: str
    name: str
    note: str


# Claude Code takes an alias or a full model id; these three are what setup offers (the measured first-word times are
# in README "Any brain").
CLAUDE_MODELS = (CliModel("haiku", "Haiku", "fastest to answer · the one for a phone call"),
                 CliModel("sonnet", "Sonnet", "thinks more, about two seconds slower to the first word"),
                 CliModel("opus", "Opus", "the strongest, and the slowest to start talking"))


def _client_version() -> str:
    try:
        from importlib.metadata import version
        return version("ai-phone")
    except Exception:
        return "0"


def _split(messages: list[Message]) -> tuple[str, list[Message], Message]:
    """The instructions, what the session has already been given, and the newest thing the caller said."""
    system = next((str(m.get("content") or "") for m in messages if m.get("role") == "system"), "")
    rest = [m for m in messages if m.get("role") != "system"]
    if not rest or rest[-1].get("role") != "user":
        raise BrainError("malformed", "the newest message is not the caller's, so there is no turn to send")
    return system, rest[:-1], rest[-1]


def _same(sent: Message, shown: Message) -> bool:
    """Is `shown` (what the controller shows now) the message this session was already given?

    A reply the caller talked over comes back marked, and a long one comes back trimmed, so an assistant turn is
    matched by its place in the conversation: the session itself holds what it really said.
    """
    if sent.get("role") != shown.get("role"):
        return False
    return sent.get("role") == "assistant" or sent.get("content") == shown.get("content")


def _pump(stream, q: queue.Queue, budget: int) -> None:
    """Every line the CLI prints, onto the queue, until it stops or prints more than one call's worth. `max_tokens`
    cannot be imposed on a CLI, so a runaway reply is possible; the producer stops long before the queue could eat
    the worker's memory (the endpoint brain's producer has the same guard)."""
    try:
        for line in stream:
            budget -= len(line)
            if budget < 0:
                q.put(("error", BrainError("too_long", "the CLI printed more than one call's worth of output")))
                break
            q.put(("line", line))
    except (OSError, ValueError):
        pass
    q.put(("eof", None))


def _pump_stderr(stream, name: str, fatal: list) -> None:
    """The CLI's own stderr, at debug and truncated. A line that explains why the process will not run at all is also
    kept: it is the difference between telling the owner `failed` and telling them which flag is gone."""
    try:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            log.debug("%s: %s", name, line[:STDERR_CHARS])
            m = re.search(r"Unknown feature flag:\s*([A-Za-z0-9_.-]{1,60})", line)
            if m:
                fatal.append(BrainError("tools_renamed", _one_of(m.group(1), frozenset(CODEX_TOOLS_OFF))))
    except (OSError, ValueError):
        pass


def _classify(cli: str, said: str) -> BrainError:
    """A failure the CLI reported, as a kind, and only a kind.

    Whatever a provider says is unbounded third-party prose that may quote the instructions, what a caller said, or a
    key. It decides the kind here and then goes to the debug log; the exception that travels to the controller, the
    worker's parent, the daemon's log and the wizard's page carries none of it.
    """
    low = " ".join(said.split()).lower()
    log.debug("%s brain: the CLI reported a failure: %s", cli, " ".join(said.split())[:400])
    if "not supported when using" in low or "does not have access to" in low or "model_not_found" in low:
        return BrainError("model_not_allowed", "that model is not on this plan")
    if any(w in low for w in ("401", "unauthorized", "not logged in", "codex login", "log in", "/login", "authenticat")):
        return BrainError("not_logged_in", f"{cli} is not logged in")
    if any(w in low for w in ("rate limit", "429", "usage limit", "quota")):
        return BrainError("rate_limited", f"{cli} says the plan's limit is reached")
    return BrainError("failed", f"{cli} ended the turn with an error")


def _version(cli: str, said: str) -> str:
    """A CLI's version, or "unknown" -- and nothing else.

    This string rides the worker's `ready` event into the daemon's log at INFO, which is the one line a person is
    guaranteed to read, and it is read from whatever the CLI printed on stdout *or stderr*. That is the same channel
    finding 4 was about, so it is classified like the rest: a version is a short dull string with a number in it, and
    anything that is not one leaves its text in the debug log and reports nothing.
    """
    line = " ".join(said.split())[:60]
    if line and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 ._()+/-]*", line) and any(c.isdigit() for c in line):
        return line
    if line:
        log.debug("%s: what it printed for --version does not read as one: %s", cli, line)
    return "unknown"


def _one_of(value, known: frozenset) -> str:
    """A value that may be shown to the person, which is to say: one of the values it is allowed to be.

    `detail` reaches a wizard page, and nothing reaches a screen on the strength of a charset filter -- stripping
    everything but `[A-Za-z0-9_.-]` leaves a token like an API key entirely intact. These fields are single enums whose
    members this module already holds, so the check is membership, and anything else is unnamed.
    """
    said = str(value or "")
    if said not in known:
        log.debug("a value outside the %d this field may hold: %r", len(known), said[:80])
        return UNNAMED
    return said


def _routes_elsewhere(name: str) -> bool:
    """Would this variable bill someone's API account, or send the call to an endpoint that is not their provider?"""
    return name.startswith(PROVIDER_ENV_PREFIXES) or name.endswith(PROVIDER_ENV_SUFFIXES)


def _kill_group(p: subprocess.Popen) -> None:
    """Kill the CLI and whatever it started (a Node runtime starts children of its own), not only the process we hold."""
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except (OSError, PermissionError):
        try:
            p.kill()
        except OSError:
            pass


class CliBrain:
    """The part both CLIs share: the process, the deadlines, and what makes a session reusable.

    A subclass says how to start the CLI (`_args`, `_opened`), how to send a turn (`_send`), what a line of its
    output means (`_event`) and how to stop a turn early (`_interrupt`).
    """

    cli_name = ""      # the executable to look for on PATH
    asks_at_warmup = True  # False when the subclass establishes the login as part of starting a session

    def __init__(self, model: str, cli: str | None = None, first_token_timeout_ms: int = 12000,
                 stream_idle_timeout_ms: int = 4000, max_response_chars: int = 2000, drain_timeout_ms: int = DRAIN_MS):
        self.model = model
        self.cli = cli or self.cli_name
        self.first_token_s = first_token_timeout_ms / 1000
        self.idle_s = stream_idle_timeout_ms / 1000
        self.drain_s = drain_timeout_ms / 1000
        self.max_chars = max_response_chars
        self.last_timings: dict = {}
        self._cancelled: set[int] = set()
        self._proc: subprocess.Popen | None = None
        self._q: queue.Queue | None = None
        self._system: str | None = None    # the instructions this session was started with
        self._seen: list[Message] = []     # the conversation this session has been given, as the controller gave it
        self._in_flight = False            # a turn we stopped reading: drain it before sending the next one
        self._spent = False                # this session failed in a way that makes it unsafe to reuse
        self._plan = ""                    # what the CLI calls the plan this login is on
        self._fatal: list = []             # why the CLI would not run at all, when its stderr said so
        self._err: threading.Thread | None = None  # the thread reading that stderr

    # ---- the brain protocol ----------------------------------------------------------------------
    def _preflight(self) -> None:
        """Whatever has to be true of this CLI before a session is worth starting. Costs no turn."""

    def login(self) -> dict:
        """What the CLI says about the login, without inference: {"plan": <what it calls the plan>}.

        Spends no turn, so the wizard can establish that a person is logged in before it offers them a model. A login
        that is an API key is refused here, not billed: this brain exists to run on a subscription.
        """
        return {"plan": ""}

    def warmup(self, system_prompt: str = "") -> dict:
        """Prove the CLI is installed, that the person is logged in on their plan, and that a session comes up
        tool-free, with one tiny turn. The probe is thrown away with its session, so no call inherits it."""
        t = time.monotonic_ns()
        version = self.cli_version()
        self._preflight()
        if self.asks_at_warmup:
            self._plan = str(self.login().get("plan", ""))
        self.close()
        try:
            said = "".join(self.stream(0, [{"role": "system", "content": system_prompt or PROBE_PROMPT},
                                           {"role": "user", "content": PROBE}], 16))
        finally:
            self.close()
        if not said.strip():
            raise BrainError("failed", f"{self.cli} answered nothing")
        return {"warm_ms": (time.monotonic_ns() - t) / 1e6, "model": self.model, "cli_version": version,
                "plan": self._plan}

    def prepare(self, system_prompt: str) -> None:
        """Have a session ready for a call with these instructions, without spending a turn.

        A session that has heard nothing yet is pristine, so one already waiting with these instructions is kept (a
        restart would cost the startup this call is trying to avoid). Any session that has taken a turn is closed, so
        nothing of the last call can carry into this one.
        """
        if self._alive() and self._system == system_prompt and not self._seen and not self._in_flight and not self._spent:
            return
        self.close()
        self._start(system_prompt)

    def stream(self, gen_id: int, messages: list[Message], max_tokens: int):
        """One turn. `max_tokens` cannot be imposed on a CLI, so a reply past max_response_chars ends the turn
        instead; the first delta must arrive within first_token_timeout_ms and each later one within
        stream_idle_timeout_ms."""
        system, prior, newest = _split(messages)
        proc = None
        t0, t_first, n, chars, said, done = time.monotonic_ns(), None, 0, 0, [], False
        try:
            if gen_id in self._cancelled:
                return  # cancelled before it began: never spend a turn of the person's plan on it
            self._begin(system, prior, newest)
            proc = self._proc
            deadline = time.monotonic() + self.first_token_s
            while True:
                if gen_id in self._cancelled:
                    return
                what, value = self._event(self._take(deadline - time.monotonic(),
                                                     "first_token_timeout" if t_first is None else "idle_timeout"))
                if what == "end":
                    done = True
                    return
                if what == "error":
                    done = True
                    raise value
                if what != "delta" or not value:
                    continue
                if t_first is None:
                    t_first = time.monotonic_ns()
                n, chars = n + 1, chars + len(value)
                said.append(value)
                deadline = time.monotonic() + self.idle_s
                yield value
                if chars >= self.max_chars:
                    log.info("%s brain: the reply passed %d characters; ending the turn", self.cli_name, self.max_chars)
                    return
        finally:
            if proc is not None and self._proc is proc:
                self._in_flight = not done
                if not done:
                    self._interrupt()
                self._seen.append({"role": "assistant", "content": "".join(said)})
            t_end = time.monotonic_ns()
            first = t_first or t_end
            self.last_timings = {"prompt_eval_ms": (first - t0) / 1e6, "first_token_ms": (first - t0) / 1e6,
                                 "gen_ms": (t_end - first) / 1e6, "tokens": n}
            self._cancelled.discard(gen_id)

    def cancel(self, gen_id: int) -> None:
        self._cancelled.add(gen_id)

    def close(self, hard: bool = False) -> None:
        """End the session. The CLI exits when its stdin closes; one that does not is killed.

        `hard` skips the asking: a CLI that is still generating will not exit on EOF, and a session we have already
        given up on must not cost the caller the wait as well as the drain.
        """
        p, self._proc, self._q, self._system = self._proc, None, None, None
        self._seen, self._in_flight, self._spent = [], False, False
        if p is None:
            return
        if hard:
            _kill_group(p)
        else:
            try:
                if p.stdin is not None:
                    p.stdin.close()
            except (OSError, ValueError):
                pass
            try:
                p.wait(timeout=CLOSE_SEC)
            except subprocess.TimeoutExpired:
                _kill_group(p)
        try:
            p.wait(timeout=CLOSE_SEC)  # a killed process is reaped at once; this is only so none is left a zombie
        except subprocess.TimeoutExpired:
            pass
        for pipe in (p.stdout, p.stderr):  # the pump threads see the end of their stream and stop
            try:
                if pipe is not None:
                    pipe.close()
            except (OSError, ValueError):
                pass

    # ---- sessions --------------------------------------------------------------------------------
    def _begin(self, system: str, prior: list[Message], newest: Message) -> None:
        theirs = self._alive() and system == self._system and self._continues(prior) and not self._spent
        # `not self._spent` again, and after the drain: `_settle()` reads the abandoned turn, and the tool watch lives
        # in that read. A turn that reaches for a tool while nobody is listening to it still loses its session.
        if theirs and self._settle() and not self._spent:
            self._send(str(newest.get("content") or ""))
            self._seen.append(dict(newest))
            return
        # A session we gave up on mid-turn is killed outright: its drain budget is already spent, and the polite
        # close (stdin, then wait) would come straight out of the time this turn has to answer.
        self.close(hard=self._in_flight)
        self._start(system)
        self._send(str(newest.get("content") or ""))
        # The new session has heard only this message. Anchoring to what the controller shows keeps the rest of the
        # call in one session: without it every later turn's history is longer than what we sent, so every later turn
        # would start again and the agent would forget each sentence as it answered the next.
        self._seen = [*(dict(m) for m in prior), dict(newest)]

    def _continues(self, prior: list[Message]) -> bool:
        """Is this the next turn of the conversation this session has had? The controller keeps only the last few
        turns, so what it shows is the end of what we sent, not all of it."""
        if len(prior) > len(self._seen):
            return False
        if not prior:
            return not self._seen  # nothing sent yet: the first turn of a session that is waiting for one
        tail = self._seen[len(self._seen) - len(prior):]
        return all(_same(sent, shown) for sent, shown in zip(tail, prior))

    def _settle(self) -> bool:
        """Let a turn we stopped reading finish before the next one starts, so two are never in flight (Claude Code
        cannot be interrupted at all). False means this session is no good and a new one should be started.

        The wait is bounded by `drain_timeout_ms`, a slice of the caller's own patience: whatever is spent here comes
        out of the time this turn has to answer before the controller apologises, so a turn that will not finish in
        that slice loses its session (about a second to start a new one) instead of the caller losing the call. That
        is the whole cost: a session dropped here is killed outright, not asked to leave (see `close`).
        """
        if not self._in_flight:
            return True
        end = time.monotonic() + self.drain_s
        try:
            while self.drain_s > 0:
                what, _ = self._event(self._take(end - time.monotonic(), "idle_timeout"))
                if what in ("end", "error"):
                    self._in_flight = False
                    return True
        except BrainError:
            pass
        log.info("%s brain: a cancelled turn did not finish within %.0f ms; starting a new session",
                 self.cli_name, self.drain_s * 1000)
        return False

    def _start(self, system_prompt: str | None) -> None:
        self._spawn(system_prompt)
        t = time.monotonic_ns()
        try:
            self._opened()
        except BrainError:
            self.close()
            raise
        log.info("%s brain session ready in %.0f ms model=%s", self.cli_name, (time.monotonic_ns() - t) / 1e6, self.model)

    def _spawn(self, system_prompt: str | None) -> None:
        exe = self._executable()
        try:
            # its own session: close() then kills the CLI and anything it started, so nothing of ours is left running
            p = subprocess.Popen([exe, *self._args(system_prompt or "")], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True, bufsize=1, env=self._env(), cwd=str(self._workdir()),
                                 start_new_session=True)
        except OSError as e:
            raise BrainError("cli_missing", f"{self.cli} would not start: {type(e).__name__}") from None
        self._proc, self._q, self._system, self._seen = p, queue.Queue(), system_prompt, []
        self._in_flight, self._spent, self._fatal = False, False, []
        threading.Thread(target=_pump, args=(p.stdout, self._q, STDOUT_BUDGET + 100 * self.max_chars),
                         name=f"{self.cli_name}-brain", daemon=True).start()
        self._err = threading.Thread(target=_pump_stderr, args=(p.stderr, self.cli_name, self._fatal),
                                    name=f"{self.cli_name}-brain-err", daemon=True)
        self._err.start()

    def _executable(self) -> str:
        found = shutil.which(self.cli)
        if found is None:
            raise BrainError("cli_missing", f"{self.cli} is not installed (not on PATH)")
        return found

    def _workdir(self) -> Path:
        """An empty private directory of our own: nothing of the person's (a CLAUDE.md, an AGENTS.md, project
        settings) is read from the place a session runs in. This may be the first thing to create the ai-phone home
        (a plan is probed before any config is written), so the home itself is made 0700 too, not the umask's 0755."""
        d = paths.home() / "brain"
        private_mkdir(d)
        if stat.S_IMODE(d.stat().st_mode) != 0o700:
            d.chmod(0o700)
        return d

    def _env(self) -> dict[str, str]:
        """The child's environment, without the variables that would take it off the person's own login.

        The worker already drops anything that looks like a secret, but the setup wizard starts a backend in its own
        process, and `ANTHROPIC_API_KEY` is not a secret by that rule and is set on plenty of developer machines. It
        wins over the keychain login, so leaving it in makes "no API key" false and bills every call.
        """
        return {k: v for k, v in os.environ.items() if not _routes_elsewhere(k) and not self._own_env(k)}

    def _own_env(self, name: str) -> bool:
        """A variable of this CLI's own that a session must not inherit (a nested run, someone else's home)."""
        return False

    def _alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def cli_version(self) -> str:
        r = self._run(["--version"])
        said = ((r.stdout if r else "") or (r.stderr if r else "") or "").strip()
        return _version(self.cli_name, said.splitlines()[0] if said else "")

    def _run(self, args: list[str]) -> subprocess.CompletedProcess | None:
        """One short, non-streaming run of the CLI (its version, its login status). None means it would not run."""
        exe = self._executable()
        try:
            return subprocess.run([exe, *args], capture_output=True, text=True, timeout=VERSION_SEC,
                                  env=self._env(), cwd=str(self._workdir()))
        except (OSError, subprocess.SubprocessError):
            return None

    # ---- talking to the CLI ----------------------------------------------------------------------
    def _write(self, line: str) -> None:
        p = self._proc
        if p is None or p.stdin is None:
            raise BrainError("failed", f"{self.cli} has no session")
        try:
            p.stdin.write(line + "\n")
            p.stdin.flush()
        except (OSError, ValueError):
            self.close()
            raise BrainError("failed", f"{self.cli} stopped") from None

    def _take(self, timeout_s: float, on_timeout: str) -> dict:
        """The next object the CLI printed. A line that is not JSON is not protocol, so it counts as nothing."""
        q = self._q
        if q is None:
            raise BrainError("failed", f"{self.cli} has no session")
        try:
            kind, payload = q.get(timeout=max(0.0, timeout_s))
        except queue.Empty:
            raise BrainError(on_timeout) from None
        if kind == "error":  # the pump stopped it: this session has printed enough
            self.close()
            raise payload
        if kind == "eof":
            why = self._why_it_stopped()
            self.close()
            raise why or BrainError("failed", f"{self.cli} stopped")
        try:
            obj = json.loads(payload)
        except ValueError:
            return {}
        return obj if isinstance(obj, dict) else {}

    def _why_it_stopped(self) -> BrainError | None:
        """A CLI that exits before it says anything usually said why on stderr, and both pipes close together, so the
        reason may still be in flight. The wait is only for a process that has already gone -- a caller on the line is
        not kept waiting for a reason about a CLI that is still running."""
        p, err = self._proc, self._err
        if not self._fatal and err is not None and p is not None and p.poll() is not None:
            err.join(STDERR_WAIT_SEC)
        return self._fatal[0] if self._fatal else None

    # ---- what a subclass provides ----------------------------------------------------------------
    def _args(self, system_prompt: str) -> list[str]:
        raise NotImplementedError

    def _opened(self) -> None:
        """Whatever the protocol needs before the first turn."""

    def _send(self, text: str) -> None:
        raise NotImplementedError

    def _event(self, obj: dict) -> tuple[str, object]:
        """One object from the CLI as ("delta", text) | ("end", None) | ("error", BrainError) | ("note", None)."""
        raise NotImplementedError

    def _interrupt(self) -> None:
        """Stop the turn that is running, if the protocol can."""


class ClaudeCodeLLM(CliBrain):
    """Claude Code in print mode with stream-json on both sides: one process for the call, one line per turn.

    `--bare` must not be used (it skips the keychain login) and every CLAUDE* variable leaves the child's
    environment, or the CLI takes itself for a nested session.
    """

    cli_name = "claude"

    def login(self) -> dict:
        """`claude auth status --json`: it says whether there is a login and what kind, and costs no turn."""
        r = self._run(["auth", "status", "--json"])
        if r is None:
            raise BrainError("cli_missing", "claude would not run")
        try:
            said = json.loads(r.stdout or "{}")
        except ValueError:
            said = {}
        if not isinstance(said, dict) or not said.get("loggedIn"):
            raise BrainError("not_logged_in", "claude is not logged in")
        how = _one_of(said.get("authMethod") or "none", CLAUDE_LOGINS)
        if how == "none":  # logged in by one reckoning and not by the other: the CLI's own word for no login wins
            raise BrainError("not_logged_in", "claude is not logged in")
        if how in CLAUDE_BILLED_LOGINS:
            # the variable itself is already gone from _env(), so this is a key saved in the CLI's own settings
            raise BrainError("api_key_login", how)
        if how not in CLAUDE_SUBSCRIPTION_LOGINS:
            # named, because a login method is not a secret and a refusal nobody can read is a refusal nobody can fix
            raise BrainError("login_unknown", how)
        return {"plan": str(said.get("subscriptionType") or "")[:40]}

    def _args(self, system_prompt: str) -> list[str]:
        return ["-p", "--verbose", "--no-session-persistence", "--setting-sources", "", "--strict-mcp-config",
                "--settings", json.dumps({"alwaysThinkingEnabled": False}),
                "--input-format", "stream-json", "--output-format", "stream-json", "--include-partial-messages",
                "--system-prompt", system_prompt, "--model", self.model,
                "--tools", ""]  # last: --tools takes every argument that follows it

    def _own_env(self, name: str) -> bool:
        return name.startswith("CLAUDE")

    def _send(self, text: str) -> None:
        self._write(json.dumps({"type": "user", "message": {"role": "user", "content": text}}))

    def _event(self, obj: dict) -> tuple[str, object]:
        what = obj.get("type")
        if what == "stream_event":
            delta = (obj.get("event") or {}).get("delta") or {}
            text = delta.get("text")
            if delta.get("type") == "text_delta" and isinstance(text, str):
                return "delta", text
            return "note", None
        if what == "result":
            if obj.get("is_error"):
                return "error", _classify("claude", str(obj.get("result") or obj.get("subtype") or ""))
            return "end", None
        if what == "system" and obj.get("subtype") == "init":
            if obj.get("tools"):
                return "error", BrainError("tool_use", "the CLI came up with tools of its own; a phone brain gets none")
        return "note", None


class CodexLLM(CliBrain):
    """`codex app-server`: JSON-RPC 2.0 over stdio, one ephemeral thread per call.

    The model is always named, because the person's own default may be one their plan does not allow.

    Codex has no single setting that turns the model's tools off, so a phone session gets all three of what it does
    have: every tool feature disabled by name (`--disable shell_tool`, …), a CODEX_HOME of ai-phone's own so none of
    the person's MCP servers, hooks, plugins, skills or AGENTS.md is in it, and a read-only sandbox that approves
    nothing. Then it is watched: a thread that did not come up inside that sandbox does not take a turn, and a tool
    call, an MCP server or an approval request during one is a `tool_use` failure, not an answer. A caller's words
    are model input, and nothing a caller can say may reach a tool.
    """

    cli_name = "codex"
    asks_at_warmup = False  # every session starts with account/read, so warm-up needs no app-server of its own

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._id = 0
        self._thread: str | None = None
        self._turn_req: int | None = None
        self._turn_id: str | None = None
        self._past: set[str] = set()   # turns this session has finished with; their late events are not the next turn's
        self._orphans: set[int] = set()  # turn/start requests we moved on from before their reply named the turn

    def _args(self, system_prompt: str) -> list[str]:
        off = [a for feature in CODEX_TOOLS_OFF for a in ("--disable", feature)]
        return ["app-server", *off, "-c", f'model_reasoning_effort="{CODEX_EFFORT}"']

    def _own_env(self, name: str) -> bool:
        return name.startswith("CODEX_")

    def _env(self) -> dict[str, str]:
        return {**super()._env(), "CODEX_HOME": str(self._codex_home())}

    def _codex_home(self) -> Path:
        """A CODEX_HOME of ai-phone's own: the config below, and a link to the login the person already has.

        The link is why no copy of their tokens is made: Codex reads and refreshes the one file they logged in to. If
        it ever leaves a file of its own here instead, that file goes (ai-phone's home is not a place to keep someone
        else's credentials) and the link is made again.
        """
        d = paths.home() / "brain" / "codex"
        private_mkdir(d)
        if stat.S_IMODE(d.stat().st_mode) != 0o700:
            d.chmod(0o700)
        cfg = d / "config.toml"
        if not cfg.exists() or cfg.read_text() != CODEX_CONFIG:
            cfg.write_text(CODEX_CONFIG)
        theirs = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex") / "auth.json"
        link = d / "auth.json"
        if link.is_symlink() and link.readlink() == theirs:
            return d
        if link.is_symlink() or link.exists():
            log.debug("codex brain: replacing what was left at %s with the link to the login", link)
            link.unlink()
        if not theirs.exists():
            raise BrainError("not_logged_in", "codex is not logged in")
        link.symlink_to(theirs)
        return d

    def _preflight(self) -> None:
        """Every tool feature ai-phone switches off must still be one this Codex has.

        A renamed flag makes `codex app-server` exit with a single line on stderr and nothing else, which is the least
        diagnosable failure in this design and a likely one (much of `codex features list` is already marked removed).
        Asking the list first names the flags instead, from a local command that spends no turn. If the list itself
        cannot be read, the session start is still fail-closed.
        """
        r = self._run(["features", "list"])
        if r is None or r.returncode != 0 or not (r.stdout or "").strip():
            log.debug("codex brain: could not read `codex features list`; the flags are unchecked")
            return
        known = {line.split()[0] for line in r.stdout.splitlines() if line.split()}
        gone = [f for f in CODEX_TOOLS_OFF if f not in known]
        if gone:
            raise BrainError("tools_renamed", " ".join(gone[:6]))

    def login(self) -> dict:
        """`account/read` on a fresh app-server: it says whose login this is, and starts no thread, so no turn."""
        self.close()
        self._spawn(None)
        try:
            self._handshake()
            return self._account()
        finally:
            self.close()

    def _account(self) -> dict:
        r = self._await(self._request("account/read", {}))
        account = ((r.get("result") or {}).get("account") or {})
        kind = str(account.get("type") or "")
        if not kind:
            raise BrainError("not_logged_in", "codex is not logged in")
        if kind != "chatgpt":
            raise BrainError("api_key_login", _one_of(kind, CODEX_ACCOUNTS))
        self._plan = str(account.get("planType") or "")[:40]
        return {"plan": self._plan}

    def _opened(self) -> None:
        self._handshake()
        self._account()  # before the thread, so a session is never started on an account that bills per token
        r = self._await(self._request("thread/start", {"ephemeral": True, "cwd": str(self._workdir()),
                                                      "baseInstructions": self._system or PROBE_PROMPT,
                                                      "sandbox": "read-only", "approvalPolicy": "never",
                                                      "model": self.model,
                                                      "config": {"sandbox_mode": "read-only",
                                                                 "web_search": "disabled",
                                                                 "tools": {"web_search": False}}}))
        result = r.get("result") or {}
        self._thread = (result.get("thread") or {}).get("id")
        self._turn_req = self._turn_id = None
        self._past, self._orphans = set(), set()
        if not self._thread:
            raise BrainError("failed", "codex started no thread")
        self._confined(result)

    def _confined(self, result: dict) -> None:
        """What the thread says it came up as. A requirements file, an MDM policy or a version that reads one of these
        settings differently can hand back a thread with more reach than was asked for; that thread does not take a
        turn. This is the Claude side's `tools == []` for a protocol that reports no tool list of its own."""
        err = self._widened(result.get("sandbox") or {}, result.get("approvalPolicy"))
        if err is not None:
            raise err

    def _widened(self, sandbox: dict, approval) -> BrainError | None:
        """Is this more reach than a phone brain gets? Read at thread start and again whenever the thread says its
        settings changed, because `thread/start` is not the only moment a sandbox can be set."""
        if sandbox.get("type") != "readOnly" or sandbox.get("networkAccess") or approval != "never":
            log.debug("codex brain: the session reports sandbox=%r approval=%r", sandbox, approval)
            return BrainError("tools_unknown", "codex gave the session more reach than a phone brain gets")
        return None

    def _handshake(self) -> None:
        self._await(self._request("initialize", {"clientInfo": {"name": "ai-phone", "version": _client_version()}}))
        self._notify("initialized", {})

    def list_models(self) -> list[CliModel]:
        """The models this plan allows. No thread is started, so no turn is spent."""
        self.close()
        self._spawn(None)
        try:
            self._handshake()
            r = self._await(self._request("model/list", {}))
            data = (r.get("result") or {}).get("data") or []
            return [CliModel(m["id"], str(m.get("displayName") or m["id"]), str(m.get("description") or ""))
                    for m in data if isinstance(m, dict) and m.get("id") and not m.get("hidden")]
        finally:
            self.close()

    def _request(self, method: str, params: dict) -> int:
        self._id += 1
        self._write(json.dumps({"jsonrpc": "2.0", "id": self._id, "method": method, "params": params}))
        return self._id

    def _notify(self, method: str, params: dict) -> None:
        self._write(json.dumps({"jsonrpc": "2.0", "method": method, "params": params}))

    def _await(self, rid: int) -> dict:
        end = time.monotonic() + max(self.first_token_s, HANDSHAKE_SEC)
        while True:
            obj = self._take(end - time.monotonic(), "first_token_timeout")
            if obj.get("id") == rid:
                if obj.get("error"):
                    raise self._failure(obj["error"])
                return obj
            what, value = self._event(obj)
            if what == "error":
                raise value

    def _send(self, text: str) -> None:
        if self._turn_id:
            self._past.add(self._turn_id)
        elif self._turn_req is not None:
            # A turn we are leaving whose id we never read. The request id is known the moment it is sent, and the
            # stream is ordered, so its reply still has to arrive before any notification of its own: when it does,
            # `_event` learns the id and `_past` can recognise the late events. This is what closes the hole the
            # permissive default in `_ours` leaves, without touching that default -- which is there to honour a
            # completion that arrives before its own reply, and whose removal is how a silent turn gets shipped.
            self._orphans.add(self._turn_req)
        self._turn_id = None
        self._turn_req = self._request("turn/start", {"threadId": self._thread, "input": [{"type": "text", "text": text}]})

    def _event(self, obj: dict) -> tuple[str, object]:
        method, prm = obj.get("method"), obj.get("params") or {}
        if method == "mcpServer/startupStatus/updated":
            # the person's own servers, attaching to our thread: their tools would be in the model's list
            log.info("codex brain: an MCP server joined the session; a phone brain gets no tools")
            self._spent = True
            return "error", BrainError("tool_use", "an MCP server joined the session; a phone brain gets no tools")
        if method == "thread/settings/updated":
            # a sandbox or an approval policy changed under the session; `_confined()` only sees thread/start
            settings = prm.get("threadSettings") or {}
            err = self._widened(settings.get("sandboxPolicy") or {}, settings.get("approvalPolicy"))
            if err is not None:
                self._spent = True
                return "error", err
        if method == "configWarning":
            log.debug("codex brain: codex ignored settings: %s", str(prm.get("summary"))[:STDERR_CHARS])
            self._spent = True
            return "error", BrainError("tools_unknown", "codex ignored settings a phone brain depends on")
        if method in CODEX_ASKS and obj.get("id") is not None:
            self._deny(obj["id"], method)  # it waits on an answer; one tool is one too many, so it gets a no and an end
            self._spent = True
            return "error", BrainError("tool_use", "the session asked to run something; a phone brain gets no tools")
        if method in CODEX_RAN:
            log.info("codex brain: the session ran something (%s); ending the turn", method)
            self._spent = True
            return "error", BrainError("tool_use", "the session ran something; a phone brain gets no tools")
        if method in ("item/started", "item/completed"):
            # both ends of the lifecycle: the two are separate notifications carrying the same item, and an item seen
            # only as completed would otherwise let the turn finish. A type we cannot read is not a reason to carry on.
            item = str((prm.get("item") or {}).get("type") or "")
            if item not in CODEX_SAID:
                log.info("codex brain: the session reached for a tool (%s); ending the turn", item or "unreadable")
                self._spent = True
                return "error", BrainError("tool_use", "the session reached for a tool; a phone brain gets none")
        if method == "item/agentMessage/delta":
            if not self._ours(prm):
                return "note", None  # an earlier turn's words: they are not an answer to what was just asked
            return "delta", prm.get("delta") or ""
        if method == "turn/completed":
            turn = prm.get("turn") or {}
            if not self._ours(prm, turn.get("id")):
                return "note", None
            status = turn.get("status")
            if status in ("completed", "interrupted"):  # interrupted is our own doing: the turn is over either way
                return "end", None
            return "error", self._failure(turn.get("error") or {"message": f"the turn {status}"})
        if method == "error":
            if not self._ours(prm):
                return "note", None
            if prm.get("willRetry"):
                # a Codex error is not the end of a turn: this one is being retried and its own turn/completed is
                # still to come. Treating it as the end would leave that completion to be read as the next turn's.
                log.info("codex brain: the turn failed and codex is retrying it")
                return "note", None
            self._spent = True  # whatever still arrives for this turn must not be read as the next turn's
            return "error", self._failure(prm.get("error") or {})
        if obj.get("id") is not None and obj.get("id") == self._turn_req:
            if obj.get("error"):
                return "error", self._failure(obj["error"])
            self._turn_id = ((obj.get("result") or {}).get("turn") or {}).get("id")
        elif obj.get("id") in self._orphans:
            self._orphans.discard(obj["id"])
            late = ((obj.get("result") or {}).get("turn") or {}).get("id")
            if late:
                self._past.add(late)  # now its own completion can be told apart from this turn's
        return "note", None

    def _ours(self, prm: dict, turn_id: str | None = None) -> bool:
        """Is this event the turn that is running? Codex names the thread and the turn on everything it sends; a
        session that has had one turn fail can still be sent another, and the first one's events must not be read as
        the second one's answer."""
        thread = prm.get("threadId")
        if thread is not None and self._thread is not None and thread != self._thread:
            return False
        turn = turn_id if turn_id is not None else prm.get("turnId")
        if turn is not None and turn in self._past:
            return False  # a turn this session is already done with: a duplicate completion, a trailing delta
        if turn is None or self._turn_id is None:  # before turn/start is answered there is nothing else to compare
            return True
        return turn == self._turn_id

    def _deny(self, rid: int, method: str) -> None:
        answer = ({"result": {"decision": "denied"}} if method in CODEX_APPROVALS
                  else {"error": {"code": -32601, "message": "ai-phone grants no tools"}})
        try:
            self._write(json.dumps({"jsonrpc": "2.0", "id": rid, **answer}))
        except BrainError:
            pass  # the session is going anyway

    def _failure(self, err: dict) -> BrainError:
        return _classify(self.cli_name, str(err.get("message") if isinstance(err, dict) else err or ""))

    def _interrupt(self) -> None:
        if self._thread and self._turn_id:
            try:
                self._request("turn/interrupt", {"threadId": self._thread, "turnId": self._turn_id})
            except BrainError:
                pass  # a session that will not take an interrupt is dropped by the next turn anyway


def codex_models(cli: str | None = None) -> list[CliModel]:
    """What `ai-phone setup` offers as the ChatGPT brain's model: the models this plan lists."""
    return CodexLLM(model="", cli=cli).list_models()


def make_claude(**kw) -> ClaudeCodeLLM:
    return ClaudeCodeLLM(**kw)


def make_codex(**kw) -> CodexLLM:
    return CodexLLM(**kw)

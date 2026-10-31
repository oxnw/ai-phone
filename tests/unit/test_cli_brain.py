"""The subscription brains, driven against a fake CLI that speaks the real protocols from a scripted list of turns.

Nothing here runs the real `claude` or `codex` (tests/integration/test_cli_brains.py does that once, for one turn):
the fake records everything it is asked to do, so the tests can state what the backends send, what they reuse and
what they refuse.
"""
from __future__ import annotations

import json
import logging
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ai_phone.services.backends.cli_brain import (CODEX_TOOLS_OFF, ClaudeCodeLLM, CodexLLM, codex_models,
                                                  make_claude, make_codex)
from ai_phone.services.errors import BrainError

FAKE_CLI = r'''#!/usr/bin/env python3
"""A fake Claude Code / Codex CLI. It follows the JSON script named by AI_PHONE_FAKE_CLI and writes everything it was
asked to do to that script's log file, one JSON object per line."""
import json
import os
import queue
import sys
import threading
import time

SPEC = json.load(open(os.environ["AI_PHONE_FAKE_CLI"]))
TURNS = SPEC.get("turns") or [{}]


def note(**what):
    with open(SPEC["log"], "a") as f:
        f.write(json.dumps(what) + "\n")


def out(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def directive():
    """The next scripted turn. The count is kept on disk, so a session that starts again carries on down the list."""
    path = SPEC["log"] + ".turns"
    n = int(open(path).read()) if os.path.exists(path) else 0
    with open(path, "w") as f:
        f.write(str(n + 1))
    return TURNS[n] if n < len(TURNS) else TURNS[-1]


def flag(name, default=None):
    argv = sys.argv[1:]
    return argv[argv.index(name) + 1] if name in argv else default


def init():
    """Claude Code's own `system`/`init` line. The real CLI prints nothing until the first turn arrives and then one
    of these per turn (measured); `init_at_start` in the script asks for the other ordering, where the line is already
    queued by the time the first turn is sent."""
    out({"type": "system", "subtype": "init", "model": flag("--model"), "tools": SPEC.get("tools", [])})


def deltas(d, emit, interrupted=lambda: False):
    """The turn's text, chunk by chunk, with the script's pauses. Returns what was said."""
    time.sleep(d.get("before_ms", 0) / 1000)
    if d.get("die"):
        os._exit(7)
    said = []
    for i, chunk in enumerate(d.get("deltas", ["OK"])):
        if i:
            time.sleep(d.get("gap_ms", 0) / 1000)
        if interrupted():
            break
        said.append(chunk)
        emit(chunk)
    return "".join(said)


if "--version" in sys.argv:
    print(SPEC.get("version", "9.9.9 (fake)"))
    raise SystemExit(0)

if sys.argv[1:3] == ["features", "list"]:
    """`codex features list`: the flag names ai-phone checks its tool switches against."""
    note(event="features")
    for line in SPEC.get("features", ["shell_tool stable true"]):
        print(line)
    raise SystemExit(0)

if SPEC.get("die_with") and "app-server" in sys.argv:
    """A CLI that will not run at all: one line on stderr, nothing on stdout, non-zero."""
    note(event="start", argv=sys.argv[1:], cwd=os.getcwd(), entries=[], claude_env=[], provider_env=[],
         codex_home=os.environ.get("CODEX_HOME"))
    sys.stderr.write(SPEC["die_with"] + "\n")
    sys.stderr.flush()
    raise SystemExit(1)

if sys.argv[1:3] == ["auth", "status"]:
    """`claude auth status --json`, which the backend reads before it spends a turn."""
    note(event="login", argv=sys.argv[1:])
    print(json.dumps(SPEC.get("login", {"loggedIn": True, "authMethod": "claude.ai", "subscriptionType": "max"})))
    raise SystemExit(0)

note(event="start", argv=sys.argv[1:], cwd=os.getcwd(), entries=sorted(os.listdir(".")),
     claude_env=sorted(k for k in os.environ if k.startswith("CLAUDE")),
     provider_env=sorted(k for k in os.environ if k.startswith(("ANTHROPIC", "OPENAI", "AWS", "SIGNALWIRE_"))
                         or k.endswith(("_API_KEY", "_API_TOKEN", "_BASE_URL"))),
     codex_home=os.environ.get("CODEX_HOME"))

if SPEC["protocol"] == "claude":
    if SPEC.get("init_at_start"):
        init()
    for line in sys.stdin:
        if not line.strip():
            continue
        note(event="turn", text=json.loads(line).get("message", {}).get("content"),
             system=flag("--system-prompt"), model=flag("--model"))
        d = directive()
        if not SPEC.get("init_at_start"):
            init()
        said = deltas(d, lambda c: out({"type": "stream_event", "event": {"type": "content_block_delta",
                                                                         "delta": {"type": "text_delta", "text": c}}}))
        out({"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": said}]}})
        if d.get("fail"):
            out({"type": "result", "subtype": "error_during_execution", "is_error": True, "result": d["fail"]})
        else:
            out({"type": "result", "subtype": "success", "is_error": False, "result": said})
else:
    lines: queue.Queue = queue.Queue()
    threading.Thread(target=lambda: ([lines.put(l) for l in sys.stdin], lines.put(None)), daemon=True).start()
    n, waiting, stop = 0, [], False
    while not stop:
        msg = json.loads(waiting.pop(0)) if waiting else None
        if msg is None:
            line = lines.get()
            if line is None:
                break
            msg = json.loads(line)
    # ---- one request or notification ----
        method, mid, prm = msg.get("method"), msg.get("id"), msg.get("params") or {}
        if method == "initialize":
            note(event="initialize", client=prm.get("clientInfo"))
            out({"jsonrpc": "2.0", "id": mid, "result": {"userAgent": "fake-codex"}})
        elif method == "initialized":
            note(event="initialized")
        elif method == "model/list":
            out({"jsonrpc": "2.0", "id": mid, "result": {"data": SPEC.get("models", [])}})
        elif method == "account/read":
            note(event="account")
            out({"jsonrpc": "2.0", "id": mid,
                 "result": {"account": SPEC.get("account", {"type": "chatgpt", "planType": "pro"})}})
        elif method == "thread/start":
            time.sleep(SPEC.get("thread_ms", 0) / 1000)  # a session that takes a while to come up, as a real one does
            note(event="thread", system=prm.get("baseInstructions"), model=prm.get("model"), cwd=prm.get("cwd"),
                 sandbox=prm.get("sandbox"), approval=prm.get("approvalPolicy"), ephemeral=prm.get("ephemeral"),
                 config=prm.get("config"))
            if SPEC.get("thread_error"):
                out({"jsonrpc": "2.0", "id": mid, "error": {"code": -32603, "message": SPEC["thread_error"]}})
            else:
                # what the thread says it came up as; `came_up` in the script asks for a thread with more reach
                came = SPEC.get("came_up") or {"sandbox": {"type": "readOnly", "networkAccess": False},
                                               "approvalPolicy": "never"}
                out({"jsonrpc": "2.0", "id": mid, "result": {"thread": {"id": "th1"}, "model": prm.get("model"), **came}})
                for name in SPEC.get("mcp_servers") or []:
                    out({"jsonrpc": "2.0", "method": "mcpServer/startupStatus/updated",
                         "params": {"threadId": "th1", "name": name, "status": "ready"}})
                if SPEC.get("config_warning"):
                    out({"jsonrpc": "2.0", "method": "configWarning", "params": {"summary": SPEC["config_warning"]}})
        elif method == "turn/interrupt":
            note(event="interrupt", turn=prm.get("turnId"))
            out({"jsonrpc": "2.0", "id": mid, "result": {}})
        elif method is None and mid is not None:
            note(event="answer", id=mid, result=msg.get("result"))  # what the client said to a request of ours
        elif method == "turn/start":
            note(event="turn", text=(prm.get("input") or [{}])[0].get("text"), effort=prm.get("effort"))
            d, n = directive(), n + 1
            turn_id = f"t{n}"
            reply = {"jsonrpc": "2.0", "id": mid, "result": {"turn": {"id": turn_id, "items": [], "status": "inProgress"}}}
            if not d.get("reply_last"):
                out(reply)          # normally the reply comes first; `reply_last` puts it after the turn's own events
            if d.get("widened"):
                out({"jsonrpc": "2.0", "method": "thread/settings/updated",
                     "params": {"threadId": "th1", "threadSettings": {"sandboxPolicy": d["widened"],
                                                                      "approvalPolicy": d.get("widened_approval", "never")}}})
            cut = []

            def interrupted():
                while True:
                    try:
                        line = lines.get_nowait()
                    except queue.Empty:
                        return bool(cut)
                    if line is None:
                        return True
                    if json.loads(line).get("method") == "turn/interrupt":
                        cut.append(line)
                    else:
                        waiting.append(line)

            if d.get("tool"):  # the session reaching for one of the model's own tools
                out({"jsonrpc": "2.0", "method": "item/started",
                     "params": {"threadId": "th1", "turnId": turn_id,
                                "item": {"type": d["tool"], "id": "x1", "command": "cat /etc/hosts"}}})
            if d.get("approval"):  # a tool asking the client's permission: it blocks until it is answered
                out({"jsonrpc": "2.0", "id": 9001, "method": "item/commandExecution/requestApproval",
                     "params": {"threadId": "th1", "turnId": turn_id, "command": "cat /etc/hosts"}})
            out({"jsonrpc": "2.0", "method": "item/started",
                 "params": {"threadId": "th1", "turnId": turn_id, "item": {"type": "agentMessage", "id": "i1"}}})
            said = deltas(d, lambda c: out({"jsonrpc": "2.0", "method": "item/agentMessage/delta",
                                            "params": {"threadId": "th1", "turnId": turn_id, "itemId": "i1", "delta": c}}),
                          interrupted)
            if d.get("tool_late"):
                # a tool item that arrives after the words, which is the realistic ordering, and the one that lands in
                # the drain when the caller has talked over the reply
                out({"jsonrpc": "2.0", "method": "item/started",
                     "params": {"threadId": "th1", "turnId": turn_id,
                                "item": {"type": d["tool_late"], "id": "x2", "command": "cat /etc/hosts"}}})
            if d.get("tool_completed_only"):
                # the same item reported only as completed: item/started and item/completed are separate notifications
                out({"jsonrpc": "2.0", "method": "item/completed",
                     "params": {"threadId": "th1", "turnId": turn_id,
                                "item": {"type": d["tool_completed_only"], "id": "x3"}}})
            if d.get("ran"):
                out({"jsonrpc": "2.0", "method": d["ran"], "params": {"threadId": "th1", "turnId": turn_id}})
            for line in cut:
                waiting.append(line)
            if d.get("retry"):
                # the real shape of a retryable failure: an error that is NOT the end of the turn, then the turn's own
                # completion. `retry_deltas` is what the retry says before it completes.
                out({"jsonrpc": "2.0", "method": "error",
                     "params": {"threadId": "th1", "turnId": turn_id, "willRetry": True,
                                "error": {"message": d["retry"]}}})
                for chunk in d.get("retry_deltas") or []:
                    out({"jsonrpc": "2.0", "method": "item/agentMessage/delta",
                         "params": {"threadId": "th1", "turnId": turn_id, "itemId": "i1", "delta": chunk}})
                out({"jsonrpc": "2.0", "method": "turn/completed",
                     "params": {"threadId": "th1", "turn": {"id": turn_id, "items": [], "status": "completed"}}})
            elif d.get("fail"):
                out({"jsonrpc": "2.0", "method": "error",
                     "params": {"threadId": "th1", "turnId": turn_id, "willRetry": False, "error": {"message": d["fail"]}}})
                if d.get("then_completed"):
                    # what the review found: a failed turn's own completion still arrives, and must not be read as
                    # the answer to whatever is asked next
                    out({"jsonrpc": "2.0", "method": "turn/completed",
                         "params": {"threadId": "th1", "turn": {"id": turn_id, "items": [], "status": "failed",
                                                                "error": {"message": d["fail"]}}}})
            else:
                status = "interrupted" if cut else "completed"
                out({"jsonrpc": "2.0", "method": "turn/completed",
                     "params": {"threadId": "th1", "turn": {"id": turn_id, "items": [], "status": status}}})
                if d.get("reply_last"):
                    out(reply)
                if d.get("extra_completion"):
                    # a turn whose completion arrives twice, so the second one is on the wire when the next turn starts
                    out({"jsonrpc": "2.0", "method": "turn/completed",
                         "params": {"threadId": "th1", "turn": {"id": turn_id, "items": [], "status": status}}})
'''

# A long value of exactly the shape a charset filter preserves, and the reason `detail` is shown from an allowlist
# instead: every character of this survives stripping everything but [A-Za-z0-9_.-]. Not a credential of any kind.
TOKEN_SHAPED = "aaaa-bbbb01-CcDdEe_123.xyz"
PROMPT = "You are a phone assistant."
OTHER = "You are a phone assistant. This call: take a message."
MODELS = [{"id": "gpt-5.6-luna", "displayName": "GPT-5.6-Luna", "description": "fast", "hidden": False, "isDefault": False},
          {"id": "gpt-6-astra", "displayName": "GPT-6-Astra", "description": "the default", "hidden": False, "isDefault": True},
          {"id": "secret", "displayName": "Hidden", "description": "", "hidden": True, "isDefault": False}]


def fake_codex_home(tmp_path, monkeypatch):
    """Keep fake CLI runs independent of any real login, including in spawned workers.

    The backend requires an auth file before launching the fake. Its empty JSON is only an existence placeholder;
    the fake's scripted account/read response decides whether a test is logged in. No real credentials are copied.
    """
    user_home = tmp_path / "user-home"
    user_home.mkdir(mode=0o700, exist_ok=True)
    codex_home = tmp_path / "fake-codex-home"
    codex_home.mkdir(mode=0o700, exist_ok=True)
    auth = codex_home / "auth.json"
    auth.write_text("{}\n")
    auth.chmod(0o600)
    monkeypatch.setenv("HOME", str(user_home))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    return codex_home


@pytest.fixture
def cli(tmp_path, monkeypatch):
    """A fake CLI on disk, the home directory the session's private folder lives in, and the script it follows."""
    monkeypatch.setenv("AI_PHONE_HOME", str(tmp_path / "home"))
    login_home = fake_codex_home(tmp_path, monkeypatch)
    exe = tmp_path / "fake-cli"
    exe.write_text(FAKE_CLI)
    exe.chmod(0o755)

    class Fake:
        path = str(exe)
        codex_home = login_home
        spec_path = tmp_path / "script.json"
        log_path = tmp_path / "log.jsonl"

        def script(self, protocol, turns=None, **extra):
            # by default the fake knows every flag the backend switches off, as the real CLI does; a test that wants a
            # flag to have gone passes its own `features`
            extra.setdefault("features", [f"{f} stable true" for f in CODEX_TOOLS_OFF])
            self.spec_path.write_text(json.dumps({"protocol": protocol, "log": str(self.log_path),
                                                  "turns": turns or [{}], **extra}))
            monkeypatch.setenv("AI_PHONE_FAKE_CLI", str(self.spec_path))
            return self

        @property
        def log(self) -> list[dict]:
            return [json.loads(l) for l in self.log_path.read_text().splitlines()] if self.log_path.exists() else []

        def events(self, what: str) -> list[dict]:
            return [r for r in self.log if r["event"] == what]

        def wait(self, what: str, count: int, timeout: float = 10.0) -> list[dict]:
            """Starting a session does not wait for the CLI itself, so a test that counts sessions waits here."""
            end = time.monotonic() + timeout
            while time.monotonic() < end and len(self.events(what)) < count:
                time.sleep(0.02)
            return self.events(what)

    return Fake()


def talk(*pairs: str) -> list[dict]:
    """[system, user, assistant, user, …] the way the turn controller builds it."""
    out = [{"role": "system", "content": PROMPT}]
    for i, text in enumerate(pairs):
        out.append({"role": "user" if i % 2 == 0 else "assistant", "content": text})
    return out


def claude_brain(cli, **kw):
    kw.setdefault("first_token_timeout_ms", 4000)
    kw.setdefault("stream_idle_timeout_ms", 4000)
    return ClaudeCodeLLM(model="haiku", cli=cli.path, **kw)


def codex_brain(cli, **kw):
    kw.setdefault("first_token_timeout_ms", 4000)
    kw.setdefault("stream_idle_timeout_ms", 4000)
    return CodexLLM(model="gpt-5.6-luna", cli=cli.path, **kw)


# ---- Claude Code --------------------------------------------------------------------------------

def test_claude_runs_the_command_the_spike_proved(cli, monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "cli")  # this session's own variables must not reach the child
    cli.script("claude", [{"deltas": ["Hel", "lo."]}])
    brain = claude_brain(cli)
    try:
        assert "".join(brain.stream(1, talk("Hi"), 120)) == "Hello."
    finally:
        brain.close()
    argv = cli.events("start")[0]["argv"]
    assert argv[:3] == ["-p", "--verbose", "--no-session-persistence"]
    assert argv[argv.index("--setting-sources") + 1] == "" and "--strict-mcp-config" in argv
    assert json.loads(argv[argv.index("--settings") + 1]) == {"alwaysThinkingEnabled": False}
    assert argv[argv.index("--input-format") + 1] == "stream-json"
    assert argv[argv.index("--output-format") + 1] == "stream-json"
    assert "--include-partial-messages" in argv and "--bare" not in argv
    assert argv[argv.index("--system-prompt") + 1] == PROMPT
    assert argv[argv.index("--model") + 1] == "haiku"
    assert argv[-2:] == ["--tools", ""]  # last: --tools takes everything that follows it
    assert cli.events("start")[0]["claude_env"] == []
    assert cli.events("turn") == [{"event": "turn", "text": "Hi", "system": PROMPT, "model": "haiku"}]


def test_a_second_turn_in_the_same_call_sends_only_the_newest_thing_the_caller_said(cli):
    cli.script("claude", [{"deltas": ["One."]}, {"deltas": ["Two."]}])
    brain = claude_brain(cli)
    try:
        assert "".join(brain.stream(1, talk("First"), 120)) == "One."
        assert "".join(brain.stream(2, talk("First", "One.", "Second"), 120)) == "Two."
    finally:
        brain.close()
    assert len(cli.events("start")) == 1, "the session was restarted"
    assert [r["text"] for r in cli.events("turn")] == ["First", "Second"]


def test_a_reply_the_caller_talked_over_still_continues_the_session(cli):
    """The controller marks a reply it cut short; the session itself holds what it really said, so the mark is no
    reason to start again."""
    cli.script("claude", [{"deltas": ["One."]}, {"deltas": ["Two."]}])
    brain = claude_brain(cli)
    try:
        list(brain.stream(1, talk("First"), 120))
        assert "".join(brain.stream(2, talk("First", "One. [interrupted]", "Second"), 120)) == "Two."
    finally:
        brain.close()
    assert len(cli.events("start")) == 1


def test_a_window_that_has_scrolled_past_the_oldest_turns_still_continues(cli):
    cli.script("claude", [{"deltas": ["a"]}, {"deltas": ["b"]}, {"deltas": ["c"]}])
    brain = claude_brain(cli)
    try:
        list(brain.stream(1, talk("one"), 120))
        list(brain.stream(2, talk("one", "a", "two"), 120))
        # the controller keeps only the last few turns: "one"/"a" have scrolled out of the window
        assert "".join(brain.stream(3, talk("two", "b", "three"), 120)) == "c"
    finally:
        brain.close()
    assert len(cli.events("start")) == 1
    assert [r["text"] for r in cli.events("turn")] == ["one", "two", "three"]


def test_a_new_system_prompt_starts_a_new_session(cli):
    cli.script("claude", [{"deltas": ["One."]}, {"deltas": ["Two."]}])
    brain = claude_brain(cli)
    try:
        list(brain.stream(1, talk("First"), 120))
        messages = [{"role": "system", "content": OTHER}, {"role": "user", "content": "Second"}]
        assert "".join(brain.stream(2, messages, 120)) == "Two."
    finally:
        brain.close()
    starts = cli.events("start")
    assert len(starts) == 2
    assert [s["argv"][s["argv"].index("--system-prompt") + 1] for s in starts] == [PROMPT, OTHER]


def test_a_history_that_is_not_a_continuation_starts_a_new_session(cli):
    cli.script("claude", [{"deltas": ["One."]}, {"deltas": ["Two."]}])
    brain = claude_brain(cli)
    try:
        list(brain.stream(1, talk("First"), 120))
        assert "".join(brain.stream(2, talk("Something else", "Not ours", "Second"), 120)) == "Two."
    finally:
        brain.close()
    assert len(cli.events("start")) == 2
    assert [r["text"] for r in cli.events("turn")] == ["First", "Second"]


def test_prepare_starts_the_session_without_spending_a_turn(cli):
    cli.script("claude", [{"deltas": ["Hello."]}])
    brain = claude_brain(cli)
    try:
        brain.prepare(PROMPT)
        assert len(cli.wait("start", 1)) == 1 and cli.events("turn") == []
        assert "".join(brain.stream(1, talk("Hi"), 120)) == "Hello."
    finally:
        brain.close()
    assert len(cli.events("start")) == 1, "the prepared session was not the one used"


def test_prepare_starts_a_fresh_session_for_the_next_call(cli):
    """The same instructions on a new call: the words of the last call must not be in the session."""
    cli.script("claude", [{"deltas": ["One."]}, {"deltas": ["Two."]}])
    brain = claude_brain(cli)
    try:
        list(brain.stream(1, talk("First"), 120))
        brain.prepare(PROMPT)
        assert len(cli.wait("start", 2)) == 2
        assert "".join(brain.stream(2, talk("Second"), 120)) == "Two."
    finally:
        brain.close()
    assert len(cli.events("start")) == 2


def test_a_cancelled_turn_is_drained_before_the_next_one(cli):
    """Claude Code cannot be interrupted, so the turn runs on; the next one waits for it (turns are never mixed)."""
    cli.script("claude", [{"deltas": ["One", " more", " word."], "gap_ms": 150}, {"deltas": ["Two."]}])
    brain = claude_brain(cli)
    try:
        said = []
        for piece in brain.stream(1, talk("First"), 120):
            said.append(piece)
            brain.cancel(1)
        assert said == ["One"]
        assert "".join(brain.stream(2, talk("First", "One", "Second"), 120)) == "Two."
    finally:
        brain.close()
    assert len(cli.events("start")) == 1
    assert [r["text"] for r in cli.events("turn")] == ["First", "Second"]


def test_a_consumer_that_walks_away_mid_turn_leaves_the_session_usable(cli):
    cli.script("claude", [{"deltas": ["One", " more."], "gap_ms": 100}, {"deltas": ["Two."]}])
    brain = claude_brain(cli)
    try:
        stream = brain.stream(1, talk("First"), 120)
        assert next(stream) == "One"
        stream.close()  # the worker breaks out of the loop: the generator is closed at the yield
        assert "".join(brain.stream(2, talk("First", "One", "Second"), 120)) == "Two."
    finally:
        brain.close()
    assert len(cli.events("start")) == 1


def test_a_reply_longer_than_the_cap_ends_the_turn(cli):
    cli.script("claude", [{"deltas": ["x" * 40] * 10}, {"deltas": ["Next."]}])
    brain = claude_brain(cli, max_response_chars=100)
    try:
        said = "".join(brain.stream(1, talk("Go on"), 120))
        assert 100 <= len(said) <= 140
        assert "".join(brain.stream(2, talk("Go on", said, "And now?"), 120)) == "Next."
    finally:
        brain.close()
    assert len(cli.events("start")) == 1


def test_no_first_delta_within_the_deadline_is_a_timeout(cli):
    cli.script("claude", [{"before_ms": 2000, "deltas": ["Late."]}, {"deltas": ["Soon."]}])
    brain = claude_brain(cli, first_token_timeout_ms=300, drain_timeout_ms=8000)
    try:
        with pytest.raises(BrainError) as e:
            list(brain.stream(1, talk("Hi"), 120))
        assert e.value.kind == "first_token_timeout"
        # the session survives a slow turn: the next one waits for the one still running, then answers
        assert "".join(brain.stream(2, talk("Hi", "Late.", "Again"), 120)) == "Soon."
    finally:
        brain.close()
    assert len(cli.events("start")) == 1


def test_a_turn_that_will_not_finish_in_the_drain_budget_loses_its_session(cli):
    """Claude Code cannot be interrupted, so a cancelled turn runs on. Waiting it out comes out of the time this turn
    has before the caller is apologised to, so a turn that is too slow to finish loses the session instead."""
    cli.script("claude", [{"deltas": ["One", " two", " three"], "gap_ms": 400}, {"deltas": ["Fresh."]}])
    brain = claude_brain(cli, drain_timeout_ms=100)
    try:
        for piece in brain.stream(1, talk("First"), 120):
            brain.cancel(1)
        assert "".join(brain.stream(2, talk("First", "One", "Second"), 120)) == "Fresh."
    finally:
        brain.close()
    assert len(cli.events("start")) == 2, "the session that was still talking was kept"


def test_the_rest_of_the_call_stays_in_the_session_the_restart_made(cli):
    """A barge-in on a slow model costs the session. It must not cost one per turn after that: the new session is
    anchored to what the controller shows, so the turns that follow are continuations of it."""
    cli.script("claude", [{"deltas": ["One", " more", " word."], "gap_ms": 400}, {"deltas": ["Two."]},
                          {"deltas": ["Three."]}])
    brain = claude_brain(cli, drain_timeout_ms=100)
    try:
        for _ in brain.stream(1, talk("First"), 120):
            brain.cancel(1)
        assert "".join(brain.stream(2, talk("First", "One", "Second"), 120)) == "Two."
        # the third turn's history is longer than the two messages the new session was given; it is still its turn
        assert "".join(brain.stream(3, talk("First", "One", "Second", "Two.", "Third"), 120)) == "Three."
    finally:
        brain.close()
    assert len(cli.events("start")) == 2, "a session per turn for the rest of the call"
    assert [r["text"] for r in cli.events("turn")] == ["First", "Second", "Third"]


def test_a_session_given_up_on_is_killed_rather_than_asked_to_leave(cli):
    """The drain budget is the whole cost: a CLI that is still generating will not exit when its stdin closes, and
    waiting CLOSE_SEC for it would come out of the time this turn has to answer."""
    cli.script("claude", [{"deltas": ["One"] + [" on"] * 30, "gap_ms": 300}, {"deltas": ["Two."]}])
    brain = claude_brain(cli, drain_timeout_ms=200)
    try:
        for _ in brain.stream(1, talk("First"), 120):
            brain.cancel(1)
        start = time.monotonic()
        assert "".join(brain.stream(2, talk("First", "One", "Second"), 120)) == "Two."
        took = time.monotonic() - start
    finally:
        brain.close()
    # the budget (0.2 s) plus a fake CLI's start; a polite close would have added CLOSE_SEC (2 s) on its own
    assert took < 1.5, f"the turn took {took:.1f} s: the lost session was waited on"


def test_the_drain_budget_is_the_daemon_s_to_set(cli):
    assert claude_brain(cli, drain_timeout_ms=3500).drain_s == 3.5
    assert CodexLLM(model="m").drain_s == 2.0  # the backend's own default, for anything that does not pass one


def test_a_cli_that_floods_its_stdout_is_stopped(cli):
    """`max_tokens` cannot be imposed on a CLI, so the producer stops a runaway long before the queue eats the
    worker's memory."""
    cli.script("claude", [{"deltas": ["x" * 2_000_000]}, {"deltas": ["Fresh."]}])
    brain = claude_brain(cli, max_response_chars=100)
    try:
        with pytest.raises(BrainError) as e:
            list(brain.stream(1, talk("Go on"), 120))
        assert e.value.kind == "too_long"
        assert "".join(brain.stream(2, talk("Again"), 120)) == "Fresh."  # a new session; that one is gone
    finally:
        brain.close()
    assert len(cli.events("start")) == 2


def test_a_stream_that_stops_mid_reply_is_an_idle_timeout(cli):
    cli.script("claude", [{"deltas": ["Half", " a sentence"], "gap_ms": 1500}])
    brain = claude_brain(cli, first_token_timeout_ms=4000, stream_idle_timeout_ms=300)
    try:
        with pytest.raises(BrainError) as e:
            list(brain.stream(1, talk("Hi"), 120))
        assert e.value.kind == "idle_timeout"
    finally:
        brain.close()


def test_a_cli_that_is_not_logged_in_says_so(cli):
    cli.script("claude", [{"deltas": [], "fail": "Not logged in · run claude and /login"}])
    brain = claude_brain(cli)
    try:
        with pytest.raises(BrainError) as e:
            list(brain.stream(1, talk("Hi"), 120))
        assert e.value.kind == "not_logged_in"
    finally:
        brain.close()


def test_a_cli_that_is_not_installed_says_so(cli, tmp_path):
    cli.script("claude")
    for brain in (ClaudeCodeLLM(model="haiku", cli=str(tmp_path / "no-such-cli")), CodexLLM(model="m", cli="no-such-cli-at-all")):
        with pytest.raises(BrainError) as e:
            list(brain.stream(1, talk("Hi"), 120))
        assert e.value.kind == "cli_missing"
        assert brain.cli in str(e.value)


def test_a_session_whose_process_died_fails_the_turn_and_the_next_one_restarts_it(cli):
    cli.script("claude", [{"die": True}, {"deltas": ["Back."]}])
    brain = claude_brain(cli)
    try:
        with pytest.raises(BrainError) as e:
            list(brain.stream(1, talk("Hi"), 120))
        assert e.value.kind == "failed"
        assert "".join(brain.stream(2, talk("Hi", "", "Again"), 120)) == "Back."
    finally:
        brain.close()
    assert len(cli.events("start")) == 2


def test_an_init_line_queued_before_the_turn_is_read_in_its_turn(cli):
    """prepare() starts the CLI, so anything it prints is waiting on the queue when the first turn is sent. Whether
    the init line comes then or with the turn, the reply is the same and tools are still refused."""
    cli.script("claude", [{"deltas": ["Hello."]}], init_at_start=True)
    brain = claude_brain(cli)
    try:
        brain.prepare(PROMPT)
        cli.wait("start", 1)
        assert "".join(brain.stream(1, talk("Hi"), 120)) == "Hello."
    finally:
        brain.close()
    cli.script("claude", [{"deltas": ["Hello."]}], init_at_start=True, tools=["Bash"])
    brain = claude_brain(cli)
    try:
        with pytest.raises(BrainError) as e:
            list(brain.stream(1, talk("Hi"), 120))
        assert e.value.kind == "tool_use" and "tools" in str(e.value)
    finally:
        brain.close()


def test_a_cli_that_came_up_with_tools_of_its_own_is_refused(cli):
    cli.script("claude", [{"deltas": ["Hello."]}], tools=["Bash", "Read"])
    brain = claude_brain(cli)
    try:
        with pytest.raises(BrainError) as e:
            list(brain.stream(1, talk("Hi"), 120))
        assert e.value.kind == "tool_use" and "tools" in str(e.value)
    finally:
        brain.close()


def test_the_session_runs_in_a_private_empty_directory(cli, tmp_path):
    cli.script("claude")
    brain = claude_brain(cli)
    try:
        brain.prepare(PROMPT)
    finally:
        brain.close()
    where = Path(cli.events("start")[0]["cwd"]).resolve()
    assert where == (tmp_path / "home" / "brain").resolve()
    assert cli.events("start")[0]["entries"] == [], "a CLAUDE.md or AGENTS.md in there would be read"
    assert stat.S_IMODE(where.stat().st_mode) == 0o700
    # this is often the first thing to create the ai-phone home (a plan is probed before any config is written), so
    # the home must not be left at the umask's 0755 beside the config and the secrets file
    assert stat.S_IMODE(where.parent.stat().st_mode) == 0o700


def test_warmup_proves_the_login_and_reports_the_model_and_version(cli):
    cli.script("claude", [{"deltas": ["OK"]}], version="2.1.286 (Claude Code)")
    brain = claude_brain(cli)
    try:
        t = brain.warmup(PROMPT)
    finally:
        brain.close()
    assert t["model"] == "haiku" and t["cli_version"] == "2.1.286 (Claude Code)" and t["warm_ms"] > 0
    assert [r["text"] for r in cli.events("turn")] == ["Say OK."]


def test_a_warmup_that_is_not_logged_in_says_so(cli):
    cli.script("claude", [{"deltas": [], "fail": "Not logged in"}])
    brain = claude_brain(cli)
    try:
        with pytest.raises(BrainError) as e:
            brain.warmup(PROMPT)
        assert e.value.kind == "not_logged_in"
    finally:
        brain.close()


def test_the_probe_turn_is_never_part_of_a_call(cli):
    cli.script("claude", [{"deltas": ["OK"]}, {"deltas": ["Hello."]}])
    brain = claude_brain(cli)
    try:
        brain.warmup(PROMPT)
        assert "".join(brain.stream(1, talk("Hi"), 120)) == "Hello."
    finally:
        brain.close()
    starts, turns = cli.events("start"), cli.events("turn")
    assert len(starts) == 2 and [r["text"] for r in turns] == ["Say OK.", "Hi"]


def test_nothing_anyone_said_reaches_the_log(cli, caplog):
    cli.script("claude", [{"deltas": ["A secret the caller told us."]}])
    brain = claude_brain(cli)
    with caplog.at_level(logging.DEBUG, logger="ai-phone"):
        try:
            said = "".join(brain.stream(1, talk("My card number is 4111 1111 1111 1111"), 120))
        finally:
            brain.close()
    assert said and "4111" not in caplog.text and "secret" not in caplog.text


def test_timings_are_reported_for_the_turn_log(cli):
    cli.script("claude", [{"deltas": ["One", " two"], "before_ms": 60}])
    brain = claude_brain(cli)
    try:
        list(brain.stream(1, talk("Hi"), 120))
    finally:
        brain.close()
    t = brain.last_timings
    assert t["tokens"] == 2 and t["first_token_ms"] >= 60 and t["gen_ms"] >= 0


# ---- Codex -------------------------------------------------------------------------------------

def test_codex_speaks_the_handshake_and_one_turn_per_message(cli):
    cli.script("codex", [{"deltas": ["Hel", "lo."]}, {"deltas": ["Again."]}])
    brain = codex_brain(cli)
    try:
        assert "".join(brain.stream(1, talk("Hi"), 120)) == "Hello."
        assert "".join(brain.stream(2, talk("Hi", "Hello.", "More"), 120)) == "Again."
    finally:
        brain.close()
    assert [r["event"] for r in cli.log][:5] == ["start", "initialize", "initialized", "account", "thread"]
    assert cli.events("initialize")[0]["client"]["name"] == "ai-phone"
    thread = cli.events("thread")[0]
    assert thread["system"] == PROMPT and thread["model"] == "gpt-5.6-luna" and thread["ephemeral"] is True
    assert thread["sandbox"] == "read-only" and thread["approval"] == "never"
    argv = cli.events("start")[0]["argv"]
    assert argv[0] == "app-server"
    assert 'model_reasoning_effort="low"' in argv
    assert thread["config"] == {"sandbox_mode": "read-only", "web_search": "disabled", "tools": {"web_search": False}}
    assert [r["text"] for r in cli.events("turn")] == ["Hi", "More"]
    assert len(cli.events("thread")) == 1


def test_codex_asks_for_the_model_every_time_because_the_saved_default_may_not_be_allowed(cli):
    cli.script("codex", [{"deltas": ["Hi."]}])
    brain = codex_brain(cli)
    try:
        list(brain.stream(1, talk("Hi"), 120))
    finally:
        brain.close()
    assert cli.events("thread")[0]["model"] == "gpt-5.6-luna"


def test_codex_interrupts_a_cancelled_turn(cli):
    cli.script("codex", [{"deltas": ["One", " more", " word."], "gap_ms": 150}, {"deltas": ["Two."]}])
    brain = codex_brain(cli)
    try:
        said = []
        for piece in brain.stream(1, talk("First"), 120):
            said.append(piece)
            brain.cancel(1)
        assert said == ["One"]
        assert "".join(brain.stream(2, talk("First", "One", "Second"), 120)) == "Two."
    finally:
        brain.close()
    assert cli.events("interrupt") and cli.events("interrupt")[0]["turn"] == "t1"
    assert len(cli.events("thread")) == 1


def test_codex_restarts_its_thread_for_a_new_system_prompt(cli):
    cli.script("codex", [{"deltas": ["One."]}, {"deltas": ["Two."]}])
    brain = codex_brain(cli)
    try:
        list(brain.stream(1, talk("First"), 120))
        list(brain.stream(2, [{"role": "system", "content": OTHER}, {"role": "user", "content": "Second"}], 120))
    finally:
        brain.close()
    assert [r["system"] for r in cli.events("thread")] == [PROMPT, OTHER]
    assert len(cli.events("start")) == 2


def test_codex_prepare_starts_the_thread_without_a_turn(cli):
    cli.script("codex", [{"deltas": ["Hello."]}])
    brain = codex_brain(cli)
    try:
        brain.prepare(PROMPT)
        assert len(cli.wait("thread", 1)) == 1 and cli.events("turn") == []
        assert "".join(brain.stream(1, talk("Hi"), 120)) == "Hello."
    finally:
        brain.close()
    assert len(cli.events("thread")) == 1


def test_codex_model_that_the_plan_does_not_allow_says_which_one(cli):
    cli.script("codex", [{"deltas": [], "fail": "400 model is not supported when using Codex with a ChatGPT account"}])
    brain = codex_brain(cli)
    try:
        with pytest.raises(BrainError) as e:
            list(brain.stream(1, talk("Hi"), 120))
        # the kind is what the wizard turns into a sentence naming the model; the provider's own words (the status
        # code, whatever it quoted back) stay in the debug log
        assert e.value.kind == "model_not_allowed" and "400" not in str(e.value)
    finally:
        brain.close()


def test_codex_that_is_not_logged_in_says_so(cli):
    cli.script("codex", [{"deltas": [], "fail": "401 Unauthorized: please run codex login"}])
    brain = codex_brain(cli)
    try:
        with pytest.raises(BrainError) as e:
            list(brain.stream(1, talk("Hi"), 120))
        assert e.value.kind == "not_logged_in"
    finally:
        brain.close()


def test_codex_whose_thread_cannot_start_fails_the_turn(cli):
    cli.script("codex", thread_error="401 Unauthorized")
    brain = codex_brain(cli)
    try:
        with pytest.raises(BrainError) as e:
            list(brain.stream(1, talk("Hi"), 120))
        assert e.value.kind == "not_logged_in"
    finally:
        brain.close()


def test_codex_warmup_proves_the_login(cli):
    cli.script("codex", [{"deltas": ["OK"]}], version="codex-cli 0.157.1")
    brain = codex_brain(cli)
    try:
        t = brain.warmup(PROMPT)
    finally:
        brain.close()
    assert sorted(t) == ["cli_version", "model", "plan", "warm_ms"]
    assert t["model"] == "gpt-5.6-luna" and t["cli_version"] == "codex-cli 0.157.1" and t["warm_ms"] > 0
    assert [r["text"] for r in cli.events("turn")] == ["Say OK."]


def test_codex_models_are_the_ones_the_plan_lists(cli):
    cli.script("codex", models=MODELS)
    found = codex_models(cli=cli.path)
    assert [m.id for m in found] == ["gpt-5.6-luna", "gpt-6-astra"], "a hidden model is not offered"
    assert found[0].name == "GPT-5.6-Luna" and found[1].note == "the default"
    assert cli.events("turn") == [], "listing the models never spends a turn"


def test_models_from_a_cli_that_is_not_installed_say_so(cli):
    cli.script("codex", models=MODELS)
    with pytest.raises(BrainError) as e:
        codex_models(cli="no-such-cli-at-all")
    assert e.value.kind == "cli_missing"


def test_the_factories_build_what_the_daemon_asks_for(cli):
    claude = make_claude(model="sonnet", cli=cli.path, first_token_timeout_ms=12000, stream_idle_timeout_ms=4000,
                         max_response_chars=2000)
    codex = make_codex(model="gpt-6-astra", cli=None, first_token_timeout_ms=12000, stream_idle_timeout_ms=4000,
                       max_response_chars=2000)
    assert isinstance(claude, ClaudeCodeLLM) and claude.model == "sonnet" and claude.first_token_s == 12.0
    assert isinstance(codex, CodexLLM) and codex.model == "gpt-6-astra" and codex.cli == "codex"
    # a subscription brain's own default deadline is the one a plan needs, not the four seconds a local model gets
    assert CodexLLM(model="m").first_token_s == 12.0 and ClaudeCodeLLM(model="haiku").first_token_s == 12.0
    claude.close()
    codex.close()


def test_a_turn_with_nothing_for_the_caller_to_say_is_refused(cli):
    cli.script("claude")
    brain = claude_brain(cli)
    try:
        with pytest.raises(BrainError) as e:
            list(brain.stream(1, [{"role": "system", "content": PROMPT}], 120))
        assert e.value.kind == "malformed"
        assert cli.events("start") == [], "no session is started for a turn that cannot be sent"
    finally:
        brain.close()


def test_closing_the_brain_ends_the_cli(cli):
    cli.script("claude")
    brain = claude_brain(cli)
    brain.prepare(PROMPT)
    proc = brain._proc
    assert os.getpgid(proc.pid) != os.getpgid(0), "a session of its own, so closing it takes its children too"
    brain.close()
    assert proc.poll() is not None, "the CLI is still running"
    assert brain._proc is None and brain.close() is None  # closing twice is how the worker shuts down


# ---- the tool boundary: a caller's words must not reach anything that can touch this Mac ----------

def test_fake_codex_uses_only_its_synthetic_login(cli, tmp_path):
    assert Path.home() == tmp_path / "user-home"
    assert not (Path.home() / ".codex" / "auth.json").exists()
    cli.script("codex", [{"deltas": ["Hi."]}])
    brain = codex_brain(cli)
    try:
        assert "".join(brain.stream(1, talk("Hi"), 120)) == "Hi."
    finally:
        brain.close()
    private_home = Path(cli.events("start")[0]["codex_home"])
    assert (private_home / "auth.json").readlink() == cli.codex_home / "auth.json"


def test_missing_synthetic_login_is_refused_before_the_fake_starts(cli):
    (cli.codex_home / "auth.json").unlink()
    cli.script("codex", [{"deltas": ["Hi."]}])
    brain = codex_brain(cli)
    try:
        with pytest.raises(BrainError) as error:
            brain.prepare(PROMPT)
        assert error.value.kind == "not_logged_in"
        assert cli.log == []
    finally:
        brain.close()


def test_codex_is_started_with_every_tool_feature_off_and_a_home_of_its_own(cli, tmp_path):
    """What the review found: `-c mcp_servers={}` does not remove the person's MCP servers (measured against the real
    CLI: ten of them started inside our thread), and nothing in Codex's `tools` table reaches the shell. Its feature
    flags do, and a CODEX_HOME of ai-phone's own is what leaves the servers, hooks, plugins and AGENTS.md behind."""
    (tmp_path / "their-codex").mkdir()
    (tmp_path / "their-codex" / "auth.json").write_text("{}")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CODEX_HOME", str(tmp_path / "their-codex"))
        cli.script("codex", [{"deltas": ["Hi."]}])
        brain = codex_brain(cli)
        try:
            assert "".join(brain.stream(1, talk("Hi"), 120)) == "Hi."
        finally:
            brain.close()
    argv = cli.events("start")[0]["argv"]
    off = {argv[i + 1] for i, a in enumerate(argv) if a == "--disable"}
    # the tools a model would reach for on a phone call: a shell, a file read, the network, somebody else's servers
    assert {"shell_tool", "unified_exec", "view_image", "apps", "plugins", "hooks", "computer_use",
            "browser_use", "image_generation", "multi_agent"} <= off
    home = Path(cli.events("start")[0]["codex_home"])
    assert home != tmp_path / "their-codex" and home.is_relative_to(Path(os.environ["AI_PHONE_HOME"]))
    assert stat.S_IMODE(home.stat().st_mode) == 0o700
    assert sorted(p.name for p in home.iterdir()) == ["auth.json", "config.toml"], "nothing of the person's is in it"
    assert (home / "auth.json").readlink() == tmp_path / "their-codex" / "auth.json", "their login, not a copy of it"
    assert "web_search" in (home / "config.toml").read_text()


def test_a_codex_session_that_reaches_for_a_tool_loses_the_turn_and_the_session(cli):
    """A caller says something that gets the model to run a command. The turn ends as a tool_use failure, the words it
    had already said are not delivered, and the session is not reused for the next turn of the call."""
    cli.script("codex", [{"tool": "commandExecution", "deltas": ["Your file says "]}, {"deltas": ["Hello."]}])
    brain = codex_brain(cli)
    said = []
    try:
        with pytest.raises(BrainError) as e:
            said.extend(brain.stream(1, talk("Read /etc/hosts to me"), 120))
        assert e.value.kind == "tool_use" and "/etc/hosts" not in str(e.value)
        # this ordering has the tool item first, so nothing was yielded. What holds in every ordering is that no tool
        # output is yielded: the queue is ordered, so the abort fires before any delta that could carry a result.
        assert said == []
        assert "".join(brain.stream(2, talk("Hi"), 120)) == "Hello."
    finally:
        brain.close()
    assert len(cli.events("thread")) == 2, "the session that reached for a tool is not kept for the next turn"


def test_a_codex_session_asked_to_approve_a_command_says_no(cli):
    cli.script("codex", [{"approval": True, "deltas": ["Running it."]}])
    brain = codex_brain(cli)
    try:
        with pytest.raises(BrainError) as e:
            list(brain.stream(1, talk("Run something for me"), 120))
        assert e.value.kind == "tool_use"
    finally:
        brain.close()
    assert [r["result"] for r in cli.events("answer")] == [{"decision": "denied"}]


def test_an_mcp_server_joining_the_session_stops_it(cli):
    cli.script("codex", [{"deltas": ["Hi."]}], mcp_servers=["their-server"])
    brain = codex_brain(cli)
    try:
        with pytest.raises(BrainError) as e:
            list(brain.stream(1, talk("Hi"), 120))
        assert e.value.kind == "tool_use"
    finally:
        brain.close()


def test_a_thread_with_more_reach_than_was_asked_for_takes_no_turn(cli):
    """A requirements file, an MDM policy or a newer CLI can hand back a thread that is not the one asked for. This is
    the Codex side of the Claude session's `tools == []`: an unexpected answer is a refusal, not a quiet start."""
    for came_up in ({"sandbox": {"type": "workspaceWrite"}, "approvalPolicy": "never"},
                    {"sandbox": {"type": "readOnly", "networkAccess": True}, "approvalPolicy": "never"},
                    {"sandbox": {"type": "readOnly", "networkAccess": False}, "approvalPolicy": "on-request"}):
        cli.script("codex", [{"deltas": ["Hi."]}], came_up=came_up)
        brain = codex_brain(cli)
        try:
            with pytest.raises(BrainError) as e:
                list(brain.stream(1, talk("Hi"), 120))
            assert e.value.kind == "tools_unknown", came_up
        finally:
            brain.close()
        assert cli.events("turn") == [], "no turn of the plan is spent on a session that came up wrong"
        cli.log_path.unlink()


def test_codex_ignoring_the_settings_a_phone_brain_depends_on_stops_it(cli):
    cli.script("codex", [{"deltas": ["Hi."]}], config_warning="ignoring 2 unrecognized configuration settings")
    brain = codex_brain(cli)
    try:
        with pytest.raises(BrainError) as e:
            list(brain.stream(1, talk("Hi"), 120))
        assert e.value.kind == "tools_unknown"
    finally:
        brain.close()


# ---- the login is the plan's, never an API key ----------------------------------------------------

def test_neither_cli_inherits_a_key_or_a_base_url(cli, monkeypatch):
    """ANTHROPIC_API_KEY beats the keychain login, so a "subscription" brain set up with it in the environment is
    billed per token and the feature's whole premise is false. Nothing that re-routes either CLI crosses over."""
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "OPENAI_API_KEY",
                 "OPENAI_BASE_URL", "AWS_BEARER_TOKEN_BEDROCK", "SOMETHING_ELSE_API_KEY"):
        monkeypatch.setenv(name, "must-not-cross")
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "cli")
    monkeypatch.setenv("LANG", "en_GB.UTF-8")  # an ordinary variable the CLI still needs
    cli.script("claude", [{"deltas": ["Hi."]}])
    brain = claude_brain(cli)
    try:
        list(brain.stream(1, talk("Hi"), 120))
    finally:
        brain.close()
    started = cli.events("start")[0]
    assert started["provider_env"] == [] and started["claude_env"] == []
    assert "LANG" not in started["provider_env"]


@pytest.mark.parametrize("kind", ["claude", "codex"])
def test_subscription_cli_never_receives_signalwire_account_context(cli, monkeypatch, kind):
    names = ("SIGNALWIRE_API_TOKEN", "SIGNALWIRE_PROJECT_ID", "SIGNALWIRE_SPACE", "CUSTOM_CARRIER_API_TOKEN")
    for name in names:
        monkeypatch.setenv(name, "synthetic-carrier-context")
    cli.script(kind, [{"deltas": ["Hi."]}])
    brain = claude_brain(cli) if kind == "claude" else codex_brain(cli)
    try:
        list(brain.stream(1, talk("Hi"), 120))
    finally:
        brain.close()
    assert cli.events("start")[0]["provider_env"] == []
    assert all(os.environ[name] == "synthetic-carrier-context" for name in names)


def test_a_cli_logged_in_with_an_api_key_is_refused_before_a_turn(cli):
    cli.script("claude", [{"deltas": ["Hi."]}],
               login={"loggedIn": True, "authMethod": "api_key", "apiKeySource": "ANTHROPIC_API_KEY"})
    brain = ClaudeCodeLLM(model="haiku", cli=cli.path)
    try:
        with pytest.raises(BrainError) as e:
            brain.warmup(PROMPT)
        assert e.value.kind == "api_key_login"
    finally:
        brain.close()
    assert cli.events("turn") == [], "a key would be billed per token, so no turn is sent to find out"


def test_a_cli_with_no_login_at_all_is_refused_before_a_turn(cli):
    cli.script("claude", [{"deltas": ["Hi."]}], login={"loggedIn": False})
    brain = ClaudeCodeLLM(model="haiku", cli=cli.path)
    try:
        with pytest.raises(BrainError) as e:
            brain.warmup(PROMPT)
        assert e.value.kind == "not_logged_in"
    finally:
        brain.close()
    assert cli.events("turn") == []


def test_codex_on_an_api_key_account_is_refused_before_a_turn(cli):
    cli.script("codex", [{"deltas": ["Hi."]}], account={"type": "apiKey"})
    brain = codex_brain(cli)
    try:
        with pytest.raises(BrainError) as e:
            brain.warmup(PROMPT)
        assert e.value.kind == "api_key_login"
    finally:
        brain.close()
    assert cli.events("turn") == [] and cli.events("thread") == []


def test_warmup_reports_what_the_plan_is_called(cli):
    cli.script("claude", [{"deltas": ["OK"]}],
               login={"loggedIn": True, "authMethod": "claude.ai", "subscriptionType": "max"})
    brain = claude_brain(cli)
    try:
        assert brain.warmup(PROMPT)["plan"] == "max"
    finally:
        brain.close()
    assert [r["argv"] for r in cli.events("login")] == [["auth", "status", "--json"]]


# ---- a provider's own words stay out of the exception and out of the log -------------------------

SENTINEL = "zq7-sentinel-the-provider-said-this"


def test_a_failing_warmup_carries_no_provider_text(cli, caplog):
    """The worker's outer handler sends and re-raises a warm-up failure, so it reaches the daemon's log. Only the kind
    may travel: a provider's error can quote the instructions, what a caller said, or a key."""
    cli.script("claude", [{"deltas": [], "fail": f"Overloaded: {SENTINEL}"}])
    brain = claude_brain(cli)
    with caplog.at_level(logging.INFO, logger="ai-phone"):
        try:
            with pytest.raises(BrainError) as e:
                brain.warmup(PROMPT)
        finally:
            brain.close()
    assert e.value.kind == "failed"
    assert SENTINEL not in str(e.value) and SENTINEL not in repr(e.value.args)
    assert SENTINEL not in caplog.text


def test_what_a_cli_writes_to_stderr_stays_out_of_the_log_at_info(cli, caplog):
    cli.script("codex", [{"deltas": [], "fail": f"500 {SENTINEL}"}])
    brain = codex_brain(cli)
    with caplog.at_level(logging.INFO, logger="ai-phone"):
        try:
            with pytest.raises(BrainError) as e:
                list(brain.stream(1, talk("Hi"), 120))
        finally:
            brain.close()
    assert e.value.kind == "failed"
    assert SENTINEL not in str(e.value) and SENTINEL not in caplog.text


def test_the_detail_is_kept_at_debug_for_whoever_is_debugging(cli, caplog):
    cli.script("codex", [{"deltas": [], "fail": f"500 {SENTINEL}"}])
    brain = codex_brain(cli)
    with caplog.at_level(logging.DEBUG, logger="ai-phone"):
        try:
            with pytest.raises(BrainError):
                list(brain.stream(1, talk("Hi"), 120))
        finally:
            brain.close()
    assert SENTINEL in caplog.text


# ---- a Codex error is not the end of a turn ------------------------------------------------------

def test_a_retryable_codex_error_is_followed_by_the_turns_own_completion(cli):
    """A Codex `error` carries willRetry and a turn id. Taking it for the end of the turn left that turn's real
    completion on the wire, to be read as the answer to whatever the caller said next."""
    cli.script("codex", [{"deltas": ["Hel"], "retry": "502 upstream hiccup", "retry_deltas": ["lo."]},
                         {"deltas": ["Again."]}])
    brain = codex_brain(cli)
    try:
        assert "".join(brain.stream(1, talk("Hi"), 120)) == "Hello."
        assert "".join(brain.stream(2, talk("Hi", "Hello.", "More"), 120)) == "Again."
    finally:
        brain.close()
    assert len(cli.events("thread")) == 1, "a retried turn did not cost the call its session"


def test_a_failed_codex_turns_completion_is_not_the_next_turns_answer(cli):
    cli.script("codex", [{"deltas": ["Half"], "fail": "500 upstream", "then_completed": True},
                         {"deltas": ["Hello."]}])
    brain = codex_brain(cli)
    try:
        with pytest.raises(BrainError) as e:
            list(brain.stream(1, talk("Hi"), 120))
        assert e.value.kind == "failed"
        # the failed turn's completion is still on the wire; the next turn must be answered by the next turn
        assert "".join(brain.stream(2, talk("Hi"), 120)) == "Hello."
    finally:
        brain.close()
    assert len(cli.events("thread")) == 2, "the session that failed is not reused"


def test_an_older_turns_words_are_not_read_as_this_turns_answer(cli):
    """Straight at the correlation: a delta tagged with a turn that is not the one running is not an answer."""
    brain = codex_brain(cli)
    brain._thread, brain._turn_id = "th1", "t2"
    assert brain._event({"method": "item/agentMessage/delta",
                         "params": {"threadId": "th1", "turnId": "t1", "delta": "stale"}}) == ("note", None)
    assert brain._event({"method": "item/agentMessage/delta",
                         "params": {"threadId": "th1", "turnId": "t2", "delta": "fresh"}}) == ("delta", "fresh")
    what, _ = brain._event({"method": "turn/completed",
                            "params": {"threadId": "th1", "turn": {"id": "t1", "status": "completed"}}})
    assert what == "note", "an older turn's completion does not end this one"
    assert brain._event({"method": "turn/completed",
                         "params": {"threadId": "th1", "turn": {"id": "t2", "status": "completed"}}}) == ("end", None)


# ---- the guard that keeps the suite off the owner's plans -----------------------------------------

def test_the_suite_refuses_to_run_the_real_subscription_clis(cli):
    """tests/conftest.py. A unit test that reaches the real claude or codex spends a turn of someone's plan and fails
    on a machine where nobody is logged in, so a green suite must not be able to depend on one."""
    for name in ("claude", "codex"):
        with pytest.raises(AssertionError, match="spends a turn of the owner's plan"):
            subprocess.run([f"/opt/homebrew/bin/{name}", "--version"])
        with pytest.raises(AssertionError):
            subprocess.Popen([name])
    # the fakes, and everything else, still run: it is the name from outside the test's own directory that is refused
    cli.script("claude")
    assert subprocess.run([cli.path, "--version"], capture_output=True).returncode == 0
    assert subprocess.run([sys.executable, "-c", "pass"]).returncode == 0


def test_a_tool_during_the_drain_does_not_keep_the_session(cli):
    """A barge-in leaves a turn in flight, and the next turn drains it before sending. The tool watch lives in that
    drain, so a turn that reaches for a tool while nobody is reading it must still lose its session: `_spent` has to be
    read after the drain, not before it."""
    cli.script("codex", [{"deltas": ["One", " more", " word."], "gap_ms": 150, "tool_late": "commandExecution"},
                         {"deltas": ["Two."]}])
    brain = codex_brain(cli)
    try:
        said = []
        for piece in brain.stream(1, talk("First"), 120):
            said.append(piece)
            brain.cancel(1)  # the caller talks over the reply; we stop reading and leave the turn in flight
        assert said == ["One"]
        assert "".join(brain.stream(2, talk("First", "One", "Second"), 120)) == "Two."
    finally:
        brain.close()
    assert len(cli.events("thread")) == 2, "the next turn ran on the session that had reached for a tool"


def test_a_finished_turns_late_completion_does_not_silence_the_next_turn(cli):
    """A turn's id is not known until `turn/start` is answered, so anything still on the wire for the turn before it
    would be read as this turn's. A stale completion read that way ends the turn at once: the caller gets silence."""
    cli.script("codex", [{"deltas": ["One."], "extra_completion": True}, {"deltas": ["Two."]}])
    brain = codex_brain(cli)
    try:
        assert "".join(brain.stream(1, talk("First"), 120)) == "One."
        assert "".join(brain.stream(2, talk("First", "One.", "Second"), 120)) == "Two."
    finally:
        brain.close()
    assert len(cli.events("thread")) == 1, "nothing was wrong with the session"


def test_a_tool_item_seen_only_as_completed_still_stops_the_turn(cli):
    """`item/started` and `item/completed` are separate notifications carrying the same item. Watching only the first
    left a tool that reports only its completion free to finish the turn and answer the caller."""
    cli.script("codex", [{"tool_completed_only": "commandExecution", "deltas": ["The file says "]}])
    brain = codex_brain(cli)
    try:
        with pytest.raises(BrainError) as e:
            list(brain.stream(1, talk("Read it to me"), 120))
        assert e.value.kind == "tool_use"
    finally:
        brain.close()


def test_an_item_whose_type_cannot_be_read_is_not_taken_for_a_safe_one(cli):
    """The allowlist has to fail closed on an item it cannot classify, the way every other check here does."""
    brain = codex_brain(cli)
    brain._thread, brain._turn_id = "th1", "t1"
    for item in ({}, {"id": "x"}, {"type": ""}, {"type": "somethingCodexAddedLater"}):
        what, value = brain._event({"method": "item/started", "params": {"threadId": "th1", "turnId": "t1", "item": item}})
        assert (what, value.kind) == ("error", "tool_use"), item
    # and the four benign ones still pass
    for kind in ("userMessage", "agentMessage", "reasoning", "contextCompaction"):
        assert brain._event({"method": "item/started",
                             "params": {"threadId": "th1", "turnId": "t1", "item": {"type": kind}}}) == ("note", None)


@pytest.mark.parametrize("ran", ["hook/started", "command/exec/outputDelta", "item/commandExecution/outputDelta",
                                 "item/mcpToolCall/progress", "process/exited", "item/plan/delta"])
def test_a_notification_that_means_something_ran_stops_the_turn(cli, ran):
    """Execution signals that live outside the item lifecycle. Nothing should produce them once the tools are off, so
    one arriving means a layer failed and the turn is over."""
    cli.script("codex", [{"ran": ran, "deltas": ["Done."]}])
    brain = codex_brain(cli)
    try:
        with pytest.raises(BrainError) as e:
            list(brain.stream(1, talk("Hi"), 120))
        assert e.value.kind == "tool_use"
    finally:
        brain.close()


# ---- the login must be one ai-phone can actually run on -------------------------------------------

@pytest.mark.parametrize("how", ["api_key", "api_key_helper"])
def test_a_login_that_is_billed_per_request_is_refused_and_named(cli, how):
    cli.script("claude", [{"deltas": ["Hi."]}], login={"loggedIn": True, "authMethod": how})
    brain = claude_brain(cli)
    try:
        with pytest.raises(BrainError) as e:
            brain.warmup(PROMPT)
        assert (e.value.kind, e.value.detail) == ("api_key_login", how)
    finally:
        brain.close()
    assert cli.events("turn") == []


@pytest.mark.parametrize("how", ["third_party", "oauth_token"])
def test_a_login_ai_phone_does_not_know_is_refused_and_named(cli, how):
    """An allowlist, not a denylist. `claude auth status --json` reports one of none, third_party, claude.ai,
    api_key_helper, oauth_token or api_key, and only claude.ai is the subscription this feature promises; a value a
    later version adds has to be refused rather than run on."""
    cli.script("claude", [{"deltas": ["Hi."]}], login={"loggedIn": True, "authMethod": how})
    brain = claude_brain(cli)
    try:
        with pytest.raises(BrainError) as e:
            brain.warmup(PROMPT)
        assert (e.value.kind, e.value.detail) == ("login_unknown", how)
    finally:
        brain.close()
    assert cli.events("turn") == [], "no turn is spent finding out"


def test_a_login_method_outside_the_six_is_refused_without_repeating_it(cli):
    """The detail reaches a wizard page, so it is shown from the enumeration this module holds, not by filtering the
    characters of whatever arrived: `[A-Za-z0-9_.-]` leaves a token like a key entirely intact."""
    cli.script("claude", [{"deltas": ["Hi."]}],
               login={"loggedIn": True, "authMethod": TOKEN_SHAPED})
    brain = claude_brain(cli)
    try:
        with pytest.raises(BrainError) as e:
            brain.warmup(PROMPT)
        assert (e.value.kind, e.value.detail) == ("login_unknown", "unrecognised")
        assert TOKEN_SHAPED not in str(e.value)
    finally:
        brain.close()


@pytest.mark.parametrize("how", ["none", "", None])
def test_a_cli_that_says_its_login_is_none_reads_as_not_logged_in(cli, how):
    """`loggedIn` true with no method is a contradiction; the CLI's own word for having no login is the one to believe,
    and it has a message of its own that tells the person what to run."""
    cli.script("claude", [{"deltas": ["Hi."]}], login={"loggedIn": True, "authMethod": how})
    brain = claude_brain(cli)
    try:
        with pytest.raises(BrainError) as e:
            brain.warmup(PROMPT)
        assert e.value.kind == "not_logged_in"
    finally:
        brain.close()
    assert cli.events("turn") == []


def test_a_claude_ai_login_is_the_one_that_runs(cli):
    cli.script("claude", [{"deltas": ["OK"]}], login={"loggedIn": True, "authMethod": "claude.ai",
                                                      "subscriptionType": "max"})
    brain = claude_brain(cli)
    try:
        assert brain.warmup(PROMPT)["plan"] == "max"
    finally:
        brain.close()


# ---- a feature flag that Codex has renamed ------------------------------------------------------

def test_a_tool_flag_codex_no_longer_has_is_named_at_warmup(cli, monkeypatch):
    """The flags are how the tools are switched off, and `codex features list` is where they are named. A renamed one
    would otherwise make the app-server exit with one line on stderr, which only a debug log would show."""
    from ai_phone.services.backends import cli_brain as mod

    monkeypatch.setattr(mod, "CODEX_TOOLS_OFF", ("shell_tool", "a_flag_codex_dropped"))
    cli.script("codex", [{"deltas": ["OK"]}], features=["shell_tool stable true", "apps stable true"])
    brain = codex_brain(cli)
    try:
        with pytest.raises(BrainError) as e:
            brain.warmup(PROMPT)
        assert (e.value.kind, e.value.detail) == ("tools_renamed", "a_flag_codex_dropped")
    finally:
        brain.close()
    assert cli.events("turn") == [] and cli.events("thread") == []


def test_a_codex_that_will_not_start_because_of_a_flag_says_which_one(cli):
    """The other half: a CLI already running the renamed flag exits before it says anything on stdout. The one line it
    does write is the whole diagnosis, so it becomes the kind rather than staying in the debug log."""
    # the real shape: Codex drops a flag ai-phone names, so the name in its message is one of ai-phone's own
    cli.script("codex", [{"deltas": ["OK"]}], die_with="Error: Unknown feature flag: shell_tool")
    brain = codex_brain(cli)
    try:
        with pytest.raises(BrainError) as e:
            list(brain.stream(1, talk("Hi"), 120))
        assert (e.value.kind, e.value.detail) == ("tools_renamed", "shell_tool")
    finally:
        brain.close()
    # and a name that is not one of ours is not repeated onto a page either
    cli.log_path.unlink()
    cli.script("codex", [{"deltas": ["OK"]}], die_with=f"Error: Unknown feature flag: {TOKEN_SHAPED}")
    brain = codex_brain(cli)
    try:
        with pytest.raises(BrainError) as e:
            list(brain.stream(1, talk("Hi"), 120))
        assert (e.value.kind, e.value.detail) == ("tools_renamed", "unrecognised")
    finally:
        brain.close()


def test_a_completion_that_arrives_before_its_own_reply_still_ends_the_turn(cli):
    """Why `_ours` is permissive while the turn's id is unknown, and why that default is kept: a `turn/completed` can
    reach us before the `turn/start` reply that names the turn, and dropping it would hang the turn to its timeout and
    leave the caller with silence. This is the case any tightening of that line has to keep passing."""
    cli.script("codex", [{"deltas": ["Hello."], "reply_last": True}])
    brain = codex_brain(cli)
    try:
        assert "".join(brain.stream(1, talk("Hi"), 120)) == "Hello."
    finally:
        brain.close()
    assert len(cli.events("thread")) == 1


def test_a_turn_abandoned_before_its_reply_cannot_silence_the_next_one(cli):
    """The hole the permissive default leaves, closed from the other side. A turn whose id was never read records
    nothing, so its late completion used to pass as the live turn's. The request id is known at send time, so its reply
    -- which the ordered stream still puts before its own notifications -- is what names it afterwards."""
    brain = codex_brain(cli)
    brain._thread, brain._turn_req, brain._turn_id = "th1", 7, None
    brain._request = lambda method, params: 8   # the real _send, with only the write to a process stubbed out
    brain._send("what did it say?")
    assert brain._orphans == {7} and brain._turn_id is None, "the request id is what it had to go on"
    # the abandoned turn's reply finally arrives and names it
    assert brain._event({"id": 7, "result": {"turn": {"id": "t1"}}}) == ("note", None)
    assert "t1" in brain._past
    # so its completion is no longer read as the live turn's
    assert brain._event({"method": "turn/completed",
                         "params": {"threadId": "th1", "turn": {"id": "t1", "status": "completed"}}}) == ("note", None)
    # and the live turn, once its own reply arrives, still ends on its own completion
    assert brain._event({"id": 8, "result": {"turn": {"id": "t2"}}}) == ("note", None)
    assert brain._event({"method": "turn/completed",
                         "params": {"threadId": "th1", "turn": {"id": "t2", "status": "completed"}}}) == ("end", None)


@pytest.mark.parametrize("widened,approval", [
    ({"type": "workspaceWrite"}, "never"),
    ({"type": "dangerFullAccess"}, "never"),
    ({"type": "readOnly", "networkAccess": True}, "never"),
    ({"type": "readOnly", "networkAccess": False}, "on-request"),
])
def test_a_sandbox_widened_after_the_thread_started_stops_the_turn(cli, widened, approval):
    """`_confined()` reads thread/start, which is not the only moment a sandbox is set: `thread/settings/updated` can
    report one widened under a running session, and that is the same thing as a tool appearing."""
    cli.script("codex", [{"widened": widened, "widened_approval": approval, "deltas": ["Hello."]}])
    brain = codex_brain(cli)
    try:
        with pytest.raises(BrainError) as e:
            list(brain.stream(1, talk("Hi"), 120))
        assert e.value.kind == "tools_unknown"
    finally:
        brain.close()


def test_settings_that_change_without_widening_anything_are_nothing(cli):
    cli.script("codex", [{"widened": {"type": "readOnly", "networkAccess": False}, "deltas": ["Hello."]}])
    brain = codex_brain(cli)
    try:
        assert "".join(brain.stream(1, talk("Hi"), 120)) == "Hello."
    finally:
        brain.close()


def test_what_a_cli_prints_for_its_version_is_a_version_or_nothing(cli, caplog):
    """The version rides the worker's `ready` event into the daemon's log at INFO, read from whatever the CLI put on
    stdout or stderr. That is finding 4's channel, so it is classified like the rest."""
    sentinel = f"Error: key {TOKEN_SHAPED} rejected"   # short enough that the 60-char debug line keeps all of it
    cli.script("claude", version=sentinel)
    brain = claude_brain(cli)
    with caplog.at_level(logging.INFO, logger="ai-phone"):
        assert brain.cli_version() == "unknown"
    assert sentinel not in caplog.text and TOKEN_SHAPED not in caplog.text
    with caplog.at_level(logging.DEBUG, logger="ai-phone"):
        brain.cli_version()
    assert TOKEN_SHAPED in caplog.text, "still there for whoever is debugging, one level down"
    for real in ("2.1.286 (Claude Code)", "codex-cli 0.157.1"):
        cli.script("claude", version=real)
        assert brain.cli_version() == real
    brain.close()

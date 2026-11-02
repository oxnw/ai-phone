"""The setup conversation driven by scripted answers against fake actions: which questions come, in what order, and
what each answer saves."""
from __future__ import annotations

from pathlib import Path

import pytest

from ai_phone import models
from ai_phone.config import SUBSCRIPTION_MODELS, InboundConfig
from ai_phone.lines.forward import divert_codes
from ai_phone.pipeline.prompts import DISCLOSURE
from ai_phone.providers.base import Number, VerificationStep
from ai_phone.services.backends.cli_brain import CLAUDE_MODELS, CliModel
from ai_phone.setup import flow
from ai_phone.setup.actions import ActionError, Current, LineResult, Machine
from ai_phone.setup.flow import BRAIN, CALLERS, CHECK, DONE, LINE, MODE, VOICE, WELCOME
from ai_phone.setup.ui import BaseUI, Cancel, Check, Outcome, Page

GREETING, INSTRUCTIONS = InboundConfig().greeting, InboundConfig().instructions
VOICES = ["af_bella", "af_heart", "am_adam", "bf_emma", "bm_george"]
OWNED = [Number("+15550100007", None, None, False, "PN7"), Number("+15550100008", None, None, False, "PN8")]
FOUND = [Number("+15550100001", 1.15, "USD", False, "n1"), Number("+15550100002", 1.15, "USD", True, "n2")]
TWILIO = {"TWILIO_ACCOUNT_SID": "ACsid", "TWILIO_AUTH_TOKEN": "tokSECRET"}
CODEX_MODELS = [CliModel("gpt-5.6-luna", "gpt-5.6-luna", "fastest to answer"),
                CliModel("gpt-6-astra", "gpt-6-astra", "the one your plan starts with")]
CLI_VERSION = {"claude": "2.1.286 (Claude Code)", "chatgpt": "codex-cli 0.157.1"}
SIP_LINE = {"kind": "sip", "user": "agent", "aor": "sip:agent@192.0.2.10"}
NUMBER_LINE = {"kind": "trunk", "number": "+15550100001", "provider": {"name": "twilio", "state": "ready"}}


class Seq:
    """Successive results of one action: each call takes the next one (an exception is raised)."""

    def __init__(self, *items):
        self.items = list(items)


class ScriptError(BaseException):
    """A script that does not match the flow. A BaseException, so flow.run's own catch-all cannot swallow it."""


def expect(ok, message):
    if not ok:
        raise ScriptError(message)


class ScriptedUI(BaseUI):
    """Answers are (question substring, answer) pairs, consumed in order. An answer is an option label (choose),
    a list of strings (ask; "" takes the default), "<default>" (Enter), "<back>" or "<quit>"."""

    def __init__(self, answers):
        super().__init__()
        self.answers = list(answers)
        self.pages: list[Page] = []
        self.menus: list[tuple[str, list, int, bool]] = []
        self.fields = []
        self.labels: list[str] = []
        self.steps: list[int] = []
        self.closed = False

    def begin(self, step):
        super().begin(step)
        self.steps.append(step)

    def _answer(self, page, back=True):
        self.next_question()
        self.pages.append(page)
        expect(self.answers, f"no answer left for {page.question!r}")
        want, answer = self.answers.pop(0)
        expect(want in page.question, f"expected a question with {want!r}, got {page.question!r}")
        if answer == "<back>":
            expect(back, f"{page.question!r} has no Back")
            raise self.back()
        if answer == "<quit>":
            raise Cancel()
        return answer

    def choose(self, page, options, default=0, back=True):
        self.menus.append((page.question, options, default, back))
        answer = self._answer(page, back)
        if answer == "<default>":
            if default == len(options):  # Enter on a menu whose default is Back
                expect(back, f"{page.question!r} defaults to a Back it does not show")
                raise self.back()
            return default
        labels = [o.label for o in options]
        expect(answer in labels, f"{answer!r} is not one of {labels} on {page.question!r}")
        return labels.index(answer)

    def ask(self, page, fields):
        self.fields.extend(fields)
        answer = self._answer(page)
        expect(isinstance(answer, list) and len(answer) == len(fields), f"{page.question!r} needs {len(fields)} answers")
        values = []
        for typed, f in zip(answer, fields, strict=True):
            value = typed or f.default
            error = f.validate(value) if f.validate else None
            expect(error is None, f"{f.label}={value!r}: {error}")
            values.append(value)
        return values

    def busy(self, page, label, work):
        self.pages.append(page)
        self.labels.append(label)
        return work()

    def progress(self, page, label, work):
        self.pages.append(page)
        self.labels.append(label)
        return work(lambda done, total: None)

    def close(self):
        self.closed = True

    def menu(self, question):
        return next(m for m in self.menus if question in m[0])

    def questions(self, text):
        return [p for p in self.pages if text in p.question]


class FakeActions:
    def __init__(self, *, exists=False, memory=24, missing=(), downloaded=None, sip_library=True, lines=None,
                 allow_inbound=True, allow_outbound=True, **results):
        self.calls: list[tuple[str, tuple]] = []
        if downloaded is None:
            downloaded = [r for r in (models.STT, models.TTS, models.PHONON, models.LLM["t4"]) if r not in missing]
        self.m = Machine(sip_library=sip_library, memory_gb=memory, models_missing=list(missing), config_exists=exists,
                         models_downloaded=downloaded)
        self.cur = Current(exists=exists, lines=dict(lines or {}), brain_kind="local", tier="t4",
                           base_url="http://127.0.0.1:1234/v1", model="local", brain_key_saved=False, brain_cli=None,
                           voice="af_heart", speed=1.0, greeting=GREETING, instructions=INSTRUCTIONS, daily_limit=20,
                           allow_inbound=allow_inbound, allow_outbound=allow_outbound)
        self.results = {"saved_keys": None, "owned_numbers": OWNED, "search_numbers": FOUND, "lan_address": "192.0.2.10",
                        "voices": VOICES, "phonon_voices": ["default"], "check_config": None, "download_size": 4_300_000_000, "test_endpoint": "OK"}
        self.results.update(results)

    def _call(self, name, *args):
        self.calls.append((name, args))
        r = self.results.get(name)
        if isinstance(r, Seq):
            r = r.items.pop(0)
        if isinstance(r, BaseException):
            raise r
        return r

    def args(self, name):
        return [a for n, a in self.calls if n == name]

    def changes(self):
        return [n for n, _ in self.calls if n.startswith(("save_", "add_", "buy_", "attach_", "forward", "record_", "use_"))]

    # what the flow reads
    def machine(self):
        return self.m

    def current(self):
        return self.cur

    def saved_keys(self, carrier):
        return self._call("saved_keys", carrier)

    def lan_address(self):
        return self._call("lan_address")

    def unfinished_numbers(self):
        return self._call("unfinished_numbers") or set()

    def resumable_numbers(self, carrier):
        return self._call("resumable_numbers", carrier) or {}

    def voices(self, *, engine="kokoro"):
        return self._call("voices", engine) if engine == "kokoro" else self._call("phonon_voices", engine)

    def missing_models(self):
        return self._call("missing_models") or self.m.models_missing

    def download_size(self):
        return self._call("download_size")

    def check_config(self):
        return self._call("check_config")

    def check_lines(self):
        r = self._call("check_lines")
        return r if r is not None else [(lid, "ready", "listening") for lid in self.cur.lines]

    # what the flow tries
    def test_endpoint(self, url, model, key):
        return self._call("test_endpoint", url, model, key)

    def cli_installed(self, kind):
        found = self._call("cli_installed", kind)
        return True if found is None else found

    def login_status(self, kind):
        return self._call("login_status", kind) or {"plan": "max" if kind == "claude" else "pro"}

    def test_subscription(self, kind, model=None):
        probe = self._call("test_subscription", kind, model)
        model = model or (self.cur.model if self.cur.brain_kind == kind else SUBSCRIPTION_MODELS[kind])
        return probe or {"model": model, "cli_version": CLI_VERSION[kind], "warm_ms": 1200.0}

    def brain_models(self, kind):
        found = self._call("brain_models", kind)
        return list(found if found is not None else (CLAUDE_MODELS if kind == "claude" else CODEX_MODELS))

    def owned_numbers(self, carrier, creds):
        return self._call("owned_numbers", carrier, dict(creds))

    def search_numbers(self, carrier, creds, country, area):
        return self._call("search_numbers", carrier, dict(creds), country, area)

    def play_sample(self, voice, speed, text, *, engine="kokoro"):
        return self._call("play_sample", voice, speed, text, engine)

    def pull_voice_model(self, report, *, engine="kokoro"):
        self._call("pull_voice_model", report, engine)
        report(1, 1)

    def pull_models(self, report):
        self._call("pull_models", report)
        report(1, 2)

    # what the flow saves
    def save_brain_local(self, tier):
        self._call("save_brain_local", tier)
        self.cur.brain_kind, self.cur.tier = "local", tier

    def save_brain_subscription(self, kind, model):
        self._call("save_brain_subscription", kind, model)
        self.cur.brain_kind, self.cur.model = kind, model

    def save_brain_endpoint(self, url, model, key):
        self._call("save_brain_endpoint", url, model, key)
        self.cur.brain_kind, self.cur.base_url, self.cur.model = "openai_compatible", url, model

    def _line(self, name, carrier, creds, number):
        r = self._call(name, carrier, dict(creds), number) or LineResult(carrier, 0, [f"Line {carrier} is ready: {number.e164}."])
        if r.code in (0, 2):
            self.cur.lines[r.line_id] = {"kind": "trunk", "number": number.e164, "provider": {"name": carrier, "state": "ready"}}
        return r

    def buy_number(self, carrier, creds, number):
        return self._line("buy_number", carrier, creds, number)

    def attach_number(self, carrier, creds, number):
        return self._line("attach_number", carrier, creds, number)

    def forward(self, line_id, mobile, when):
        self._call("forward", line_id, mobile, when)
        return divert_codes(when, self.cur.lines[line_id]["number"])

    def start_caller_id(self, carrier, creds, number, *, line_id=None):
        """The fake carrier's step: a code texted to the number, and "123456" is the one it sent."""
        return self._call("start_caller_id", carrier, dict(creds), number, *((line_id,) if line_id else ())) or \
            VerificationStep("enter_code_in_cli", f"Enter the 6-digit code sent to {number}", {"e164": number})

    def check_caller_id(self, carrier, creds, step, code):
        seeded = self._call("check_caller_id", carrier, dict(creds), step, code)
        return code == "123456" if seeded is None else seeded

    def record_caller_id(self, line_id, number, carrier, creds=None):
        self._call("record_caller_id", line_id, number, carrier, *((creds,) if creds else ()))
        self.cur.lines[line_id]["caller_id"] = number

    def use_caller_id(self, line_id, number, creds=None):
        self._call("use_caller_id", line_id, number, *((creds,) if creds else ()))
        self.cur.lines[line_id]["caller_id"] = number

    def add_sip_line(self, name, host):
        line_id = self._call("add_sip_line", name, host) or name
        self.cur.lines[line_id] = {"kind": "sip", "user": name, "aor": f"sip:{name}@{host}"}
        return line_id

    def add_trunk_line(self, registrar, user, password, number, transport, from_user="number", *, preset=None, srtp="off"):
        line_id = self._call("add_trunk_line", registrar, user, password, number, transport, from_user,
                             *((preset, srtp) if preset else ())) or preset or "trunk"
        self.cur.lines[line_id] = {"kind": "trunk", "number": number, "user": user, "from_user": from_user, "registrar": registrar}
        return line_id

    def save_voice(self, voice, speed, *, engine="kokoro"):
        self._call("save_voice", voice, speed, engine)
        self.cur.voice, self.cur.speed, self.cur.tts_engine = voice, speed, engine

    def save_call_directions(self, inbound, outbound):
        self._call("save_call_directions", inbound, outbound)
        self.cur.allow_inbound, self.cur.allow_outbound = inbound, outbound
        self.cur.exists = True

    def save_callers(self, greeting, instructions, daily_limit):
        self._call("save_callers", greeting, instructions, daily_limit)
        self.cur.greeting, self.cur.instructions, self.cur.daily_limit = greeting, instructions, daily_limit


def page_text(page: Page) -> str:
    return repr(page)


def drive(ui, actions):
    """flow.run, and the whole script was used; an unexpected error (Outcome 3) fails the test with its summary."""
    outcome = flow.run(ui, actions)
    assert outcome.code != 3, outcome.summary
    assert not ui.answers, f"unused answers: {ui.answers}"
    return outcome


@pytest.fixture(autouse=True)
def default_home(monkeypatch):
    """The flow only shows paths (the fake actions write nothing), so the default home is safe and gives `~` paths."""
    monkeypatch.delenv("AI_PHONE_HOME", raising=False)


WELCOME_NEW = ("Give your AI a phone number", "Continue")
MODE_BOTH = ("How should ai-phone work?", "Receive and place calls")
FINISH = ("You're set up", "Finish")


def test_new_setup_local_brain_sip_line_default_voice():
    ui = ScriptedUI([
        WELCOME_NEW,
        MODE_BOTH,
        ("Where should your agent think?", "On this computer"),
        ("Which model?", "<default>"),
        ("Which connection should ai-phone use?", "A SIP address"),
        ("Name your SIP address", ["agent", "192.0.2.10"]),
        ("Which speech model?", "Kokoro"),
        ("How should your agent sound?", "American · female"),
        ("American · female", "<default>"),
        ("Heart", "Use this voice"),
        ("How fast should it talk?", [""]),
        ("What should callers hear?", ["", "", ""]),
        ("Everything works", "Continue"),
        FINISH,
    ])
    actions = FakeActions()
    outcome = drive(ui, actions)
    assert outcome.code == 0 and ui.closed and not ui.answers
    assert ui.steps == [WELCOME, MODE, BRAIN, LINE, VOICE, CALLERS, CHECK, DONE]
    _question, options, default, _back = ui.menu("Which model?")
    assert options[default].label == "Qwen3.5 4B"
    assert ui.questions("Which model?")[0].detail == \
        "This computer has 24 GB of memory. Models that need more may swap and answer slowly."
    assert [o.label for o in ui.menu("Which connection should ai-phone use")[1]][3] == "Skip"
    assert ui.menu("Which connection should ai-phone use")[1][3].hint == "add one later"
    assert ui.menu("Which connection should ai-phone use")[1][0].hint == "Twilio, Telnyx or SignalWire"
    group = ui.menu("American · female")
    assert [o.label for o in group[1]] == ["Bella", "Heart"] and group[2] == 1 and group[1][1].hint == "current"
    assert ui.questions("Heart")[0].detail == "American · female"
    speed = next(f for f in ui.fields if f.label == "Speed")
    assert speed.default == "1.0"
    assert actions.changes() == ["save_call_directions", "save_brain_local", "add_sip_line", "save_voice", "save_callers"]
    assert actions.args("save_brain_local") == [("t4",)]
    assert actions.args("add_sip_line") == [("agent", "192.0.2.10")]
    assert actions.args("save_voice") == [("af_heart", 1.0, "kokoro")]
    assert actions.args("save_callers") == [(GREETING, INSTRUCTIONS, 20)]
    assert actions.args("play_sample") == [("af_heart", 1.0, DISCLOSURE, "kokoro")]
    check = ui.questions("Everything works")[0]
    assert check.checks == [Check("Models", "Parakeet 0.6B v2\nKokoro 82M\nQwen3.5 4B", True), Check("Config", "loads", True),
                            Check("Line agent", "ready · listening", True), Check("Voice", "Heart · played", True)]
    assert "ai-phone run" in outcome.summary
    assert "pjsua --local-port 5070 sip:agent@127.0.0.1" in outcome.summary
    assert ui.questions("You're set up")[0].body == outcome.summary


def test_welcome_shows_the_machine_checks():
    ui = ScriptedUI([("Give your AI", "Cancel")])
    outcome = drive(ui, FakeActions(memory=8, sip_library=False, missing=["a/b"], downloaded=[]))
    assert outcome.code == 1
    assert outcome.summary == "Setup cancelled. Nothing was changed."
    page = ui.pages[0]
    assert page.detail == "Calls answered and placed by models running on this computer."
    assert page.note == "Picks a line, a brain and a voice, then checks it all works."
    assert page.checks == [
        Check("SIP library", "not built yet · run ./scripts/install.sh", False),
        Check("Memory", "8 GB · Qwen3.5 2B recommended", True),
        Check("Models", "none downloaded yet\nMissing: a/b · setup can download them", None),
        Check("Config", "new · ~/.ai-phone/", True),
    ]
    assert [(o.label, o.hint) for o in ui.menus[0][1]] == [("Continue", ""), ("Cancel", "exit setup, nothing is saved")]

    ui = ScriptedUI([("Give your AI", "Cancel")])
    drive(ui, FakeActions(memory=36, exists=True))
    assert ui.pages[0].checks == [
        Check("SIP library", "built", True),
        Check("Memory", "36 GB · Qwen3.5 9B recommended", True),
        Check("Models", "Parakeet 0.6B v2\nKokoro 82M\nPhonon 31M\nQwen3.5 4B", True),
        Check("Config", "found · ~/.ai-phone/ (setup changes only what you pick)", True),
    ]
    assert [o.label for o in ui.menus[0][1]] == ["Add a line", "Change the brain", "Change the voice",
                                                "Change call directions", "Change call settings", "Change caller ID",
                                                "Run all steps", "Cancel"]


@pytest.mark.parametrize("label,inbound,outbound", [
    ("Receive and place calls", True, True),
    ("Place calls only", False, True),
    ("Receive calls only", True, False),
])
def test_first_run_saves_call_directions_before_the_brain(label, inbound, outbound):
    ui = ScriptedUI([
        WELCOME_NEW,
        ("How should ai-phone work?", label),
        ("Where should your agent think?", "<quit>"),
    ])
    actions = FakeActions()
    outcome = drive(ui, actions)
    assert outcome.code == 1
    assert ui.steps == [WELCOME, MODE, BRAIN]
    assert actions.changes() == ["save_call_directions"]
    assert actions.args("save_call_directions") == [(inbound, outbound)]
    _, options, default, _ = ui.menu("How should ai-phone work?")
    assert [o.label for o in options] == ["Receive and place calls", "Place calls only", "Receive calls only"]
    assert default == 0
    assert "running" in page_text(ui.questions("How should ai-phone work?")[0]).lower()


@pytest.mark.parametrize("label,inbound,outbound,question,answers,field_labels,expected", [
    ("Receive and place calls", True, True, "What should callers hear?",
     ["New greeting.", "Take a message.", "7"], ["Greeting", "Instructions", "Outbound calls per day"],
     ("New greeting.", "Take a message.", 7)),
    ("Place calls only", False, True, "How many calls may ai-phone place?",
     ["7"], ["Outbound calls per day"], ("Saved greeting.", "Saved instructions.", 7)),
    ("Receive calls only", True, False, "What should callers hear?",
     ["New greeting.", "Take a message."], ["Greeting", "Instructions"],
     ("New greeting.", "Take a message.", 13)),
])
def test_changing_directions_asks_only_active_settings_and_preserves_others(
        label, inbound, outbound, question, answers, field_labels, expected):
    actions = FakeActions(exists=True, lines={"agent": SIP_LINE}, allow_inbound=inbound, allow_outbound=outbound)
    actions.cur.greeting, actions.cur.instructions, actions.cur.daily_limit = "Saved greeting.", "Saved instructions.", 13
    ui = ScriptedUI([
        ("Give your AI", "Change call directions"),
        ("How should ai-phone work?", "<default>"),
        (question, answers),
        ("Everything works", "Continue"),
        FINISH,
    ])
    assert drive(ui, actions).code == 0
    assert ui.steps == [WELCOME, MODE, CALLERS, CHECK, DONE]
    _, options, default, _ = ui.menu("How should ai-phone work?")
    assert options[default].label == label
    assert [f.label for f in ui.fields] == field_labels
    assert actions.changes() == ["save_call_directions", "save_callers"]
    assert actions.args("save_call_directions") == [(inbound, outbound)]
    assert actions.args("save_callers") == [expected]


def test_a_call_direction_save_error_retries_without_running_later_steps():
    ui = ScriptedUI([
        ("Give your AI", "Change call directions"),
        ("How should ai-phone work?", "Place calls only"),
        ("How should ai-phone work?", "Place calls only"),
        ("How many calls may ai-phone place?", "<quit>"),
    ])
    actions = FakeActions(exists=True, save_call_directions=Seq(ActionError("config could not be saved"), None))
    assert drive(ui, actions).code == 1
    first, retry = ui.questions("How should ai-phone work?")
    assert not first.error and retry.error == "config could not be saved"
    assert actions.args("save_call_directions") == [(False, True), (False, True)]
    assert not actions.args("save_callers")


def test_back_from_brain_returns_to_the_saved_call_directions():
    ui = ScriptedUI([
        WELCOME_NEW, MODE_BOTH,
        ("Where should your agent think?", "<back>"),
        ("How should ai-phone work?", "Place calls only"),
        ("Where should your agent think?", "<quit>"),
    ])
    actions = FakeActions()
    assert drive(ui, actions).code == 1
    assert ui.steps == [WELCOME, MODE, BRAIN, MODE, BRAIN]
    assert actions.args("save_call_directions") == [(True, True), (False, True)]
    assert actions.changes() == ["save_call_directions", "save_call_directions"]


@pytest.mark.parametrize("inbound,outbound", [(False, True), (True, False)])
def test_number_setup_skips_actions_for_disabled_call_directions(inbound, outbound):
    ui = ScriptedUI([
        ("Give your AI", "Add a line"),
        ("Which connection", "A phone number"),
        ("Which carrier?", "Telnyx"),
        ("Use the saved Telnyx key?", "Use the saved key"),
        ("A new number, or one already on", "One already on my Telnyx account"),
        ("Which number?", "+15550100007"),
        ("Connect +15550100007 to ai-phone?", "Connect +15550100007"),
        *([("Also forward your own mobile to it?", "No")] if inbound else []),
        *([CALLER_ID_NO] if outbound else []),
        ("Everything works", "<quit>"),
    ])
    actions = FakeActions(exists=True, saved_keys={"TELNYX_API_KEY": "saved-test-key"},
                          allow_inbound=inbound, allow_outbound=outbound)
    assert drive(ui, actions).code == 1
    assert bool(ui.questions("Also forward your own mobile")) is inbound
    assert bool(ui.questions("Show your own number when the agent calls out?")) is outbound
    assert actions.changes() == ["attach_number"]


def test_endpoint_brain_retries_after_a_failed_test():
    ui = ScriptedUI([
        WELCOME_NEW,
        MODE_BOTH,
        ("Where should your agent think?", "My own endpoint"),
        ("Your endpoint", ["http://127.0.0.1:8080/v1", "qwen", "sk-test"]),
        ("Your endpoint", ["", "", "sk-test"]),
        ("Which connection should ai-phone use?", "<quit>"),
    ])
    actions = FakeActions(test_endpoint=Seq(ActionError("could not reach 127.0.0.1: ConnectError"), "OK"))
    assert drive(ui, actions).code == 1
    first, _busy, second, _busy2 = ui.questions("Your endpoint")
    assert first.error == "" and second.error == "could not reach 127.0.0.1: ConnectError"
    assert ui.labels == ["Sending one short test request to 127.0.0.1…"] * 2
    assert actions.args("test_endpoint") == [("http://127.0.0.1:8080/v1", "qwen", "sk-test")] * 2
    assert actions.args("save_brain_endpoint") == [("http://127.0.0.1:8080/v1", "qwen", "sk-test")]
    retry = ui.fields[3:]
    assert [f.default for f in retry] == ["http://127.0.0.1:8080/v1", "qwen", ""]
    assert retry[2].secret and retry[2].placeholder == "leave empty for none"
    assert all("sk-test" not in page_text(p) for p in ui.pages)
    assert all("sk-test" not in repr(f) for f in ui.fields)


def test_endpoint_brain_keeps_the_saved_key_when_left_empty():
    ui = ScriptedUI([
        ("Give your AI", "Change the brain"),
        ("Where should your agent think?", "<default>"),
        ("Your endpoint", ["https://api.example.com/v1", "", ""]),
        ("A few things need attention", "<quit>"),
    ])
    actions = FakeActions(exists=True)
    actions.cur.brain_kind, actions.cur.model, actions.cur.brain_key_saved = "openai_compatible", "qwen", True
    assert drive(ui, actions).code == 1
    assert ui.menus[1][2] == 3  # the current brain is the default: the endpoint is the last of the four
    assert ui.fields[2].placeholder == "leave empty to keep the saved key (same server)"
    assert actions.args("test_endpoint") == [("https://api.example.com/v1", "qwen", None)]
    assert actions.args("save_brain_endpoint") == [("https://api.example.com/v1", "qwen", None)]
    assert ui.labels[0] == "Sending one short test request to api.example.com…"


def test_endpoint_brain_that_cannot_be_saved_is_asked_again():
    ui = ScriptedUI([
        ("Give your AI", "Change the brain"),
        ("Where should your agent think?", "My own endpoint"),
        ("Your endpoint", ["https://203.0.113.5/v1", "qwen", ""]),
        ("Your endpoint", ["https://203.0.113.6/v1", "", ""]),
        ("A few things need attention", "<quit>"),
    ])
    broken = "config.lines.agent.aor: expected sip:user@host[:port]"
    actions = FakeActions(exists=True, save_brain_endpoint=Seq(ActionError(broken), None))
    drive(ui, actions)
    assert ui.questions("Your endpoint")[2].error == broken
    assert actions.args("save_brain_endpoint")[-1] == ("https://203.0.113.6/v1", "qwen", None)


def test_a_config_that_does_not_load_is_shown_where_the_answer_is_saved():
    broken = "config.lines.agent.aor: expected sip:user@host[:port]"
    ui = ScriptedUI([
        ("Give your AI", "Change the brain"),
        ("Where should your agent think?", "On this computer"),
        ("Which model?", "Qwen3.5 9B"),
        ("Which model?", "<default>"),
        ("A few things need attention", "<quit>"),
    ])
    actions = FakeActions(exists=True, memory=32, save_brain_local=Seq(ActionError(broken), None))
    drive(ui, actions)
    first, second = ui.questions("Which model?")
    assert first.error == "" and second.error == broken
    assert ui.menus[-2][2] == 1  # the model that was picked stays picked
    assert actions.args("save_brain_local") == [("t9",), ("t9",)]


def add_a_line(carrier="Twilio"):
    return [("Give your AI", "Add a line"), ("Which connection should ai-phone use?", "A phone number"),
            ("Which carrier?", carrier)]


def test_twilio_new_number_buys_only_after_confirmation():
    choose_a_number = [
        ("Your Twilio credentials", ["ACsid", "tokSECRET"]),
        ("A new number, or one already on", "A new number"),
        ("Where should the number be?", ["", ""]),
        ("Pick a number", "+15550100001"),
    ]
    ui = ScriptedUI([
        *add_a_line(), *choose_a_number,
        ("Buy +15550100001 for $1.15/month?", "<default>"),  # Enter goes back: it never buys
        ("Which connection should ai-phone use?", "A phone number"), ("Which carrier?", "Twilio"), *choose_a_number,
        ("Buy +15550100001 for $1.15/month?", "Buy +15550100001"),
        ("Also forward your own mobile to it?", "No"),
        CALLER_ID_NO,
        ("Everything works", "Continue"),
        FINISH,
    ])
    actions = FakeActions(exists=True, lines={}, check_lines=[("twilio", "registered", "")])
    outcome = drive(ui, actions)
    assert outcome.code == 0
    assert ui.steps == [WELCOME, LINE, LINE, CHECK, DONE]
    assert actions.args("buy_number") == [("twilio", TWILIO, FOUND[0])]
    assert actions.args("owned_numbers") == [("twilio", TWILIO)] * 2
    assert actions.args("search_numbers") == [("twilio", TWILIO, "US", None)] * 2
    credentials = ui.questions("Your Twilio credentials")[0]
    assert credentials.detail == "Saved to ~/.ai-phone/secrets.env, readable only by you. Never shown or sent anywhere else."
    assert [(f.label, f.secret, f.placeholder) for f in ui.fields[:2]] == [("Account SID", True, "AC…"), ("Auth Token", True, "32 characters")]
    assert ui.questions("A new number, or one already on")[0].checks == [Check("Twilio key", "works · 2 numbers on this account", True)]
    assert [(o.label, o.hint) for o in ui.menu("A new number, or one already on")[1]] == [
        ("A new number", "search and buy one"), ("One already on my Twilio account", "2 not yet connected")]
    assert [(o.label, o.hint) for o in ui.menu("Pick a number")[1]] == [
        ("+15550100001", "$1.15/month"), ("+15550100002", "$1.15/month · documents required")]
    confirm = ui.questions("Buy +15550100001")[0]
    assert confirm.detail == "Twilio bills your account every month until you remove the line (ai-phone lines remove)."
    assert [(o.label, o.hint) for o in ui.menu("Buy +15550100001")[1]] == [("Buy +15550100001", "$1.15/month")]
    assert ui.menu("Buy +15550100001")[2:] == (1, True)  # the default is Back
    assert ui.labels[-4:-2] == ["Looking for numbers…", "Buying +15550100001 and connecting it…"]
    assert ui.questions("Setting up +15550100001")
    assert all("tokSECRET" not in page_text(p) and "ACsid" not in page_text(p) for p in ui.pages)
    assert "call +15550100001 from any phone (your normal call rates)" in outcome.summary


def test_no_numbers_found_asks_for_another_area():
    ui = ScriptedUI([
        *add_a_line(),
        ("Your Twilio credentials", ["ACsid", "tokSECRET"]),
        ("A new number, or one already on", "A new number"),
        ("Where should the number be?", ["gb", "999"]),
        ("Where should the number be?", ["", ""]),
        ("Pick a number", "<quit>"),
    ])
    actions = FakeActions(exists=True, search_numbers=Seq([], FOUND))
    drive(ui, actions)
    first, _, second, _ = ui.questions("Where should the number be?")
    assert second.error == "No numbers found there. Try another area code, or leave it empty."
    assert actions.args("search_numbers") == [("twilio", TWILIO, "GB", "999"), ("twilio", TWILIO, "GB", None)]


def test_telnyx_existing_number_and_forward():
    ui = ScriptedUI([
        *add_a_line("Telnyx"),
        ("Use the saved Telnyx key?", "Use the saved key"),
        ("A new number, or one already on", "One already on my Telnyx account"),
        ("Which number?", "+15550100008"),
        ("Connect +15550100008 to ai-phone?", "Connect +15550100008"),
        ("Also forward your own mobile to it?", "Yes"),
        ("Your mobile number", ["+447700900123"]),
        ("When should calls go to your agent?", "When I'm busy"),
        ("Dial this on your mobile", "Done"),
        CALLER_ID_NO,
        ("Everything works", "Continue"),
        FINISH,
    ])
    saved = {"TELNYX_API_KEY": "KEYsaved"}
    actions = FakeActions(exists=True, saved_keys=saved)
    assert drive(ui, actions).code == 0
    assert not ui.questions("credentials")
    assert ui.questions("Use the saved Telnyx key?")[0].detail == "It is in ~/.ai-phone/secrets.env."
    assert actions.args("attach_number") == [("telnyx", saved, OWNED[1])]
    assert not actions.args("buy_number")
    assert [(o.label, o.hint) for o in ui.menu("Connect +15550100008")[1]] == [("Connect +15550100008", "free")]
    assert ui.menu("Connect +15550100008")[2:] == (1, True)  # the default is Back
    assert ui.labels[:2] == ["Checking your Telnyx account…", "Connecting +15550100008…"]
    assert actions.args("forward") == [("telnyx", "+447700900123", "busy")]
    codes = ui.questions("Dial this on your mobile")[0]
    assert "**67*+15550100008#" in codes.body and "##002#" in codes.body
    assert codes.body.splitlines() == ["Turn on                 **67*+15550100008#",
                                       "Check                   *#67#",
                                       "Turn off                ##67#",
                                       "Turn off every divert   ##002#"]
    assert all("KEYsaved" not in page_text(p) for p in ui.pages)


def test_back_in_the_forward_questions_never_returns_to_the_purchase():
    ui = ScriptedUI([
        *add_a_line("Telnyx"),
        ("Use the saved Telnyx key?", "Use the saved key"),
        ("A new number, or one already on", "One already on my Telnyx account"),
        ("Which number?", "+15550100007"),
        ("Connect +15550100007 to ai-phone?", "Connect +15550100007"),
        ("Also forward your own mobile to it?", "Yes"),
        ("Your mobile number", "<back>"),
        ("Also forward your own mobile to it?", "No"),
        CALLER_ID_NO,
        ("Everything works", "<quit>"),
    ])
    actions = FakeActions(exists=True, saved_keys={"TELNYX_API_KEY": "k"})
    drive(ui, actions)
    assert len(actions.args("attach_number")) == 1 and not actions.args("forward")
    assert ui.menu("Also forward")[3] is False  # the number is connected: there is no going back to it


CALLER_ID_NO = ("Show your own number when the agent calls out?", "No")


def connect_a_number(*after, forwarding=(("Also forward your own mobile to it?", "No"),)):
    """Connect +15550100007 on the saved Telnyx key, skip forwarding, then whatever the test is about."""
    return ScriptedUI([
        *add_a_line("Telnyx"),
        ("Use the saved Telnyx key?", "Use the saved key"),
        ("A new number, or one already on", "One already on my Telnyx account"),
        ("Which number?", "+15550100007"),
        ("Connect +15550100007 to ai-phone?", "Connect +15550100007"),
        *forwarding,
        *after,
    ])


def test_caller_id_verification_saves_the_verified_mobile_and_shows_it_on_the_done_page():
    ui = connect_a_number(
        ("Show your own number when the agent calls out?", "Yes"),
        ("Your mobile number", ["+447700900123"]),
        ("Enter the code", ["123456"]),
        ("Everything works", "Continue"),
        FINISH,
    )
    actions = FakeActions(exists=True, saved_keys={"TELNYX_API_KEY": "KEYsaved"})
    outcome = drive(ui, actions)
    assert outcome.code == 0
    assert actions.args("start_caller_id") == [("telnyx", {"TELNYX_API_KEY": "KEYsaved"}, "+447700900123")]
    assert actions.args("record_caller_id") == [("telnyx", "+447700900123", "telnyx")]
    ask_page = ui.questions("Show your own number")[0]
    assert ask_page.detail == ("People you call see your mobile number instead of +15550100007. Telnyx sends one code "
                               "to your phone to prove it is yours; no paperwork.")
    assert ui.menu("Show your own number")[2:] == (0, False)  # No is the default, and there is no going back
    assert [o.label for o in ui.menu("Show your own number")[1]] == ["No", "Yes"]
    assert ui.questions("Enter the code")[0].detail == "Enter the 6-digit code sent to +447700900123"
    assert "Asking Telnyx to verify +447700900123…" in ui.labels
    assert "telnyx   +15550100007   Telnyx number · calls out as +447700900123" in outcome.summary
    assert all("KEYsaved" not in page_text(p) for p in ui.pages)


def test_the_mobile_typed_for_forwarding_is_offered_again_for_the_caller_id():
    ui = connect_a_number(
        ("Show your own number when the agent calls out?", "Yes"),
        ("Your mobile number", [""]),  # Enter takes the number typed for forwarding
        ("Enter the code", ["123456"]),
        ("Everything works", "<quit>"),
        forwarding=(("Also forward your own mobile to it?", "Yes"),
                    ("Your mobile number", ["+447700900123"]),
                    ("When should calls go to your agent?", "All calls"),
                    ("Dial this on your mobile", "Done")),
    )
    actions = FakeActions(exists=True, saved_keys={"TELNYX_API_KEY": "k"})
    drive(ui, actions)
    assert actions.args("start_caller_id") == [("telnyx", {"TELNYX_API_KEY": "k"}, "+447700900123")]
    assert [f.default for f in ui.fields if f.label == "Mobile number"] == ["", "+447700900123"]


def test_a_wrong_code_is_asked_again_and_the_right_one_still_saves_it():
    ui = connect_a_number(
        ("Show your own number when the agent calls out?", "Yes"),
        ("Your mobile number", ["+447700900123"]),
        ("Enter the code", ["000000"]),
        ("Enter the code", ["123456"]),
        ("Everything works", "<quit>"),
    )
    actions = FakeActions(exists=True, saved_keys={"TELNYX_API_KEY": "k"})
    drive(ui, actions)
    assert [p.error for p in ui.questions("Enter the code")] == ["", "", "That code was not accepted.", ""]
    assert [a[3] for a in actions.args("check_caller_id")] == ["000000", "123456"]
    assert actions.args("record_caller_id") == [("telnyx", "+447700900123", "telnyx")]


def test_three_wrong_codes_give_up_without_saving_anything():
    ui = connect_a_number(
        ("Show your own number when the agent calls out?", "Yes"),
        ("Your mobile number", ["+447700900123"]),
        ("Enter the code", ["000000"]),
        ("Enter the code", ["000001"]),
        ("Enter the code", ["000002"]),
        ("Verification did not complete", "Continue"),
        ("Everything works", "<quit>"),
    )
    actions = FakeActions(exists=True, saved_keys={"TELNYX_API_KEY": "k"})
    drive(ui, actions)
    assert len(actions.args("check_caller_id")) == 3 and not actions.args("record_caller_id")
    gave_up = ui.questions("Verification did not complete")[0]
    assert "ai-phone lines verify-caller-id telnyx +447700900123" in gave_up.detail
    assert ui.menu("Verification did not complete")[3] is False


def test_a_verification_the_config_will_not_record_is_never_silently_dropped():
    """The carrier has accepted the number by now. If the config write fails, say so and name the command that
    records it, instead of looping back as though nothing had happened."""
    broken = "config.yaml is not valid YAML; fix it or move it away, then run setup again"
    ui = connect_a_number(
        ("Show your own number when the agent calls out?", "Yes"),
        ("Your mobile number", ["+447700900123"]),
        ("Enter the code", ["123456"]),
        ("+447700900123 is verified, but not saved", "Continue"),
        ("Everything works", "<quit>"),
    )
    actions = FakeActions(exists=True, saved_keys={"TELNYX_API_KEY": "k"}, record_caller_id=ActionError(broken))
    outcome = drive(ui, actions)
    page = ui.questions("+447700900123 is verified, but not saved")[0]
    assert page.detail == ("+447700900123 is verified with Telnyx; run: ai-phone lines verify-caller-id "
                           "telnyx +447700900123 --key-env <NAME> to record it.")
    assert page.error == broken
    assert ui.menu("+447700900123 is verified, but not saved")[3] is False
    assert not ui.questions("Verification did not complete")
    assert "calls out as" not in outcome.summary  # nothing was saved, so nothing claims it was


def test_no_asks_the_carrier_nothing():
    ui = connect_a_number(CALLER_ID_NO, ("Everything works", "<quit>"))
    actions = FakeActions(exists=True, saved_keys={"TELNYX_API_KEY": "k"})
    drive(ui, actions)
    assert not actions.args("start_caller_id") and not actions.args("record_caller_id")
    assert actions.changes() == ["attach_number"]


def test_a_carrier_that_will_not_start_the_check_says_so_and_asks_again():
    ui = connect_a_number(
        ("Show your own number when the agent calls out?", "Yes"),
        ("Your mobile number", ["+447700900123"]),
        ("Show your own number when the agent calls out?", "No"),
        ("Everything works", "<quit>"),
    )
    actions = FakeActions(exists=True, saved_keys={"TELNYX_API_KEY": "k"},
                          start_caller_id=ActionError("Could not reach Telnyx: timed out"))
    drive(ui, actions)
    first, second = ui.questions("Show your own number")
    assert first.error == "" and second.error == "Could not reach Telnyx: timed out"
    assert not actions.args("record_caller_id")


def test_back_from_the_mobile_number_returns_to_the_caller_id_question():
    ui = connect_a_number(
        ("Show your own number when the agent calls out?", "Yes"),
        ("Your mobile number", "<back>"),
        ("Show your own number when the agent calls out?", "No"),
        ("Everything works", "<quit>"),
    )
    actions = FakeActions(exists=True, saved_keys={"TELNYX_API_KEY": "k"})
    drive(ui, actions)
    assert len(actions.args("attach_number")) == 1 and not actions.args("start_caller_id")


def test_a_carrier_that_calls_the_phone_waits_for_the_code_to_be_keyed_in(monkeypatch):
    monkeypatch.setattr(flow, "CALLER_ID_POLL_SEC", 0)
    on_phone = VerificationStep("enter_code_on_phone", "Telnyx is calling +447700900123. Key in 314159.", {"e164": "x"})
    ui = connect_a_number(
        ("Show your own number when the agent calls out?", "Yes"),
        ("Your mobile number", ["+447700900123"]),
        ("Everything works", "<quit>"),
    )
    actions = FakeActions(exists=True, saved_keys={"TELNYX_API_KEY": "k"}, start_caller_id=on_phone,
                          check_caller_id=Seq(False, False, True))
    drive(ui, actions)
    assert [a[3] for a in actions.args("check_caller_id")] == [None, None, None]
    assert actions.args("record_caller_id") == [("telnyx", "+447700900123", "telnyx")]
    assert "Waiting for you to enter the code on the call… (up to 2 minutes)" in ui.labels
    assert ui.questions("Telnyx is calling you")[0].body == on_phone.instructions


def test_a_call_the_person_never_answers_times_out(monkeypatch):
    monkeypatch.setattr(flow, "CALLER_ID_POLL_SEC", 0)
    monkeypatch.setattr(flow, "CALLER_ID_WAIT_SEC", 0)
    on_phone = VerificationStep("enter_code_on_phone", "Telnyx is calling. Key in 314159.", {"e164": "x"})
    ui = connect_a_number(
        ("Show your own number when the agent calls out?", "Yes"),
        ("Your mobile number", ["+447700900123"]),
        ("Verification did not complete", "Continue"),
        ("Everything works", "<quit>"),
    )
    actions = FakeActions(exists=True, saved_keys={"TELNYX_API_KEY": "k"}, start_caller_id=on_phone,
                          check_caller_id=False)
    drive(ui, actions)
    assert len(actions.args("check_caller_id")) == 1 and not actions.args("record_caller_id")


def test_the_caller_id_step_is_only_for_a_carrier_number_that_is_ready():
    """A SIP address has no carrier to ask, and a pending number is not usable yet."""
    ui = ScriptedUI([
        *add_a_line(), ("Your Twilio credentials", ["ACsid", "tokSECRET"]),
        ("A new number, or one already on", "A new number"),
        ("Where should the number be?", ["", ""]),
        ("Pick a number", "+15550100001"),
        ("Buy +15550100001 for $1.15/month?", "Buy +15550100001"),
        ("The number is not ready yet", "Continue"),
        ("Everything works", "<quit>"),
    ])
    pending = FakeActions(exists=True, buy_number=LineResult("twilio", 2, ["order pending"]))
    drive(ui, pending)
    assert not ui.questions("Show your own number") and not pending.args("start_caller_id")

    ui = ScriptedUI([
        ("Give your AI", "Add a line"), ("Which connection should ai-phone use?", "A SIP address"),
        ("Name your SIP address", ["agent", "192.0.2.10"]),
        ("Everything works", "<quit>"),
    ])
    sip = FakeActions(exists=True)
    drive(ui, sip)
    assert not ui.questions("Show your own number") and not sip.args("start_caller_id")


def caller_id_line(*, carrier="telnyx", number="+15555550100", caller="+15555550101"):
    return {"kind": "trunk", "number": number, "caller_id": caller,
            "provider": {"name": carrier, "state": "ready"} if carrier else None,
            "verified_caller_ids": [{"number": caller, "provider": carrier, "verified_at": "2026-10-27T00:00:00Z"}]}


CALLER_ID_OWNED = [Number("+15555550100", None, None, False, "PN-test")]


def edit_caller_id(*answers, line_id="main", number="+15555550100"):
    return ScriptedUI([("Give your AI", "Change caller ID"),
                       ("Which line's caller ID?", f"{line_id} · {number}"), *answers])


@pytest.mark.parametrize("selection,expected", [("Line's number · +15555550100", "+15555550100"),
                                                ("<default>", "+15555550101")])
def test_saved_caller_id_can_be_changed_without_contacting_the_carrier(selection, expected):
    line = caller_id_line()
    history = list(line["verified_caller_ids"])
    actions = FakeActions(exists=True, lines={"main": line})
    ui = edit_caller_id(("Which number should people see?", selection),
                        ("Everything works", "Continue"), FINISH)
    outcome = drive(ui, actions)
    assert outcome.code == 0
    assert actions.changes() == ["use_caller_id"]
    assert actions.args("use_caller_id") == [("main", expected)]
    assert line["caller_id"] == expected and line["verified_caller_ids"] == history
    assert not actions.args("saved_keys") and not actions.args("start_caller_id")
    assert not actions.args("owned_numbers") and not actions.args("attach_number")
    assert ui.steps == [WELCOME, CALLERS, CHECK, DONE]
    _, options, default, _ = ui.menu("Which number should people see?")
    assert options[default].label == "Your verified number · +15555550101"
    assert "restart" in outcome.summary.lower()


def test_caller_id_choices_filter_other_carriers_and_deduplicate_numbers():
    line = caller_id_line()
    line["verified_caller_ids"] += [dict(line["verified_caller_ids"][0]),
                                    {"number": "+15555550100", "provider": "telnyx"},
                                    {"number": "+15555550102", "provider": "twilio"}]
    ui = edit_caller_id(("Which number should people see?", "<quit>"))
    actions = FakeActions(exists=True, lines={"main": line})
    assert drive(ui, actions).code == 1
    assert [o.label for o in ui.menu("Which number should people see?")[1]] == [
        "Line's number · +15555550100", "Your verified number · +15555550101", "Verify another number you own"]
    assert not actions.changes()


def test_caller_id_line_picker_shows_ready_phone_lines_and_edits_only_the_selected_line():
    first, second = caller_id_line(), caller_id_line(carrier="twilio", number="+15555550102", caller="+15555550103")
    lines = {"main": first, "office": second, "agent": SIP_LINE,
             "pending": {**caller_id_line(), "provider": {"name": "telnyx", "state": "pending"}},
             "unknown": {**caller_id_line(), "provider": {"name": "telnyx"}},
             "no-number": {"kind": "trunk"}}
    actions = FakeActions(exists=True, lines=lines)
    ui = edit_caller_id(("Which number should people see?", "Line's number · +15555550102"),
                        ("Everything works", "<quit>"), line_id="office", number="+15555550102")
    drive(ui, actions)
    options = ui.menu("Which line's caller ID?")[1]
    assert [o.label for o in options] == ["main · +15555550100", "office · +15555550102"]
    assert "+15555550101" in options[0].hint and "+15555550103" in options[1].hint
    assert actions.args("use_caller_id") == [("office", "+15555550102")]
    assert first["caller_id"] == "+15555550101" and second["caller_id"] == "+15555550102"


@pytest.mark.parametrize("provider", [None, "absent"])
def test_manual_trunks_can_reuse_caller_ids_but_do_not_offer_carrier_verification(provider):
    line = caller_id_line(carrier="")
    if provider == "absent":
        del line["provider"]
    actions = FakeActions(exists=True, lines={"main": line})
    ui = edit_caller_id(("Which number should people see?", "Your verified number · +15555550101"),
                        ("Everything works", "<quit>"))
    drive(ui, actions)
    assert [o.label for o in ui.menu("Which number should people see?")[1]] == [
        "Line's number · +15555550100", "Your verified number · +15555550101"]
    assert actions.args("use_caller_id") == [("main", "+15555550101")]
    assert not actions.args("saved_keys")


@pytest.mark.parametrize("number", ["+15555550100", None])
def test_account_from_explains_provider_control_without_offering_caller_id_edits(number):
    line = {**caller_id_line(number=number), "user": "account1", "from_user": "account"}
    before = dict(line)
    actions = FakeActions(exists=True, lines={"main": line, "normal": caller_id_line()})
    ui = edit_caller_id(("Your provider controls this caller ID", "Continue"),
                       ("Everything works", "Continue"), FINISH, number=number or "account1")
    outcome = drive(ui, actions)
    assert outcome.code == 0
    assert line == before and not actions.changes()
    assert not actions.args("saved_keys") and not actions.args("start_caller_id")
    assert not ui.questions("Which number should people see?")
    options = ui.menu("Which line's caller ID?")[1]
    assert options[0].hint == "caller ID set by your provider"
    assert "+15555550101" in options[1].hint
    main_row = next(row for row in outcome.summary.splitlines() if "main" in row)
    assert main_row.endswith("caller ID set by your provider")
    assert "calls out as" not in main_row and "+15555550101" not in main_row


@pytest.mark.parametrize("outbound,lines,question", [
    (False, {"main": caller_id_line()}, "Outgoing calls are off"),
    (True, {"agent": SIP_LINE}, "No phone-number lines are ready"),
])
def test_unavailable_caller_id_editor_explains_why_without_changing_settings(outbound, lines, question):
    actions = FakeActions(exists=True, allow_outbound=outbound, lines=lines)
    ui = ScriptedUI([("Give your AI", "Change caller ID"), (question, "Continue"),
                     ("Everything works", "<quit>")])
    drive(ui, actions)
    assert not actions.changes() and not actions.args("saved_keys")
    assert not ui.questions("Which line's caller ID?")


def test_back_from_caller_id_choices_returns_to_lines_then_welcome_and_clears_the_shortcut():
    actions = FakeActions(exists=True, lines={"main": caller_id_line()})
    ui = edit_caller_id(("Which number should people see?", "<back>"),
                        ("Which line's caller ID?", "<back>"),
                        ("Give your AI", "Change call settings"),
                        ("What should callers hear?", ["", "", ""]),
                        ("Everything works", "<quit>"))
    drive(ui, actions)
    assert actions.changes() == ["save_callers"]
    assert len(ui.questions("Which line's caller ID?")) == 2


@pytest.mark.parametrize("saved", [True, False])
def test_existing_line_can_verify_another_number_with_saved_or_entered_credentials(saved):
    creds = {"TELNYX_API_KEY": "KEYtest"}
    actions = FakeActions(exists=True, lines={"main": caller_id_line()}, saved_keys=creds if saved else None,
                          owned_numbers=CALLER_ID_OWNED)
    credentials = [("Use the saved Telnyx key?", "Use the saved key")] if saved else [
        ("Your Telnyx credentials", ["KEYtest"])]
    ui = edit_caller_id(("Which number should people see?", "Verify another number you own"), *credentials,
                        ("Your mobile number", ["+15555550102"]), ("Enter the code", ["123456"]),
                        ("Everything works", "Continue"), FINISH)
    outcome = drive(ui, actions)
    assert outcome.code == 0
    assert actions.args("start_caller_id") == [("telnyx", creds, "+15555550102")]
    assert actions.args("record_caller_id") == [("main", "+15555550102", "telnyx")]
    assert not actions.args("use_caller_id")
    assert actions.args("owned_numbers") == [("telnyx", creds)]
    assert "calls out as +15555550102" in outcome.summary
    assert all(p.step == CALLERS for p in ui.pages if any(t in p.question for t in ("credentials", "mobile", "code", "Verifying")))
    assert all("KEYtest" not in page_text(p) for p in ui.pages)


def test_existing_twilio_line_uses_the_phone_verification_flow(monkeypatch):
    monkeypatch.setattr(flow, "CALLER_ID_POLL_SEC", 0)
    step = VerificationStep("enter_code_on_phone", "Key in 314159 when Twilio calls.", {"e164": "+15555550102"})
    actions = FakeActions(exists=True, lines={"main": caller_id_line(carrier="twilio")}, saved_keys=TWILIO,
                          start_caller_id=step, check_caller_id=Seq(False, True), owned_numbers=CALLER_ID_OWNED)
    ui = edit_caller_id(("Which number should people see?", "Verify another number you own"),
                        ("Use the saved Twilio key?", "Enter a different key"),
                        ("Your Twilio credentials", ["ACother", "OTHERtest"]),
                        ("Your mobile number", ["+15555550102"]), ("Everything works", "<quit>"))
    drive(ui, actions)
    assert actions.args("start_caller_id") == [("twilio", {"TWILIO_ACCOUNT_SID": "ACother", "TWILIO_AUTH_TOKEN": "OTHERtest"},
                                                "+15555550102")]
    assert [a[3] for a in actions.args("check_caller_id")] == [None, None]
    assert actions.args("record_caller_id") == [("main", "+15555550102", "twilio")]
    assert ui.questions("Twilio is calling you now")[0].step == CALLERS
    assert all("OTHERtest" not in page_text(p) and "tokSECRET" not in page_text(p) for p in ui.pages)


@pytest.mark.parametrize("leave", ["<back>", "<quit>"])
def test_leaving_existing_caller_id_verification_preserves_the_current_number(leave):
    line = caller_id_line()
    actions = FakeActions(exists=True, lines={"main": line}, saved_keys={"TELNYX_API_KEY": "KEYtest"},
                          owned_numbers=CALLER_ID_OWNED)
    after = [("Which number should people see?", "<quit>")] if leave == "<back>" else []
    ui = edit_caller_id(("Which number should people see?", "Verify another number you own"),
                        ("Use the saved Telnyx key?", "Use the saved key"),
                        ("Your mobile number", ["+15555550102"]), ("Enter the code", leave), *after)
    outcome = drive(ui, actions)
    assert outcome.code == 1 and "Nothing was changed" in outcome.summary
    assert line["caller_id"] == "+15555550101"
    assert not actions.changes()


def test_failed_existing_caller_id_verification_names_the_current_number():
    line = caller_id_line()
    actions = FakeActions(exists=True, lines={"main": line}, saved_keys={"TELNYX_API_KEY": "KEYtest"},
                          owned_numbers=CALLER_ID_OWNED)
    ui = edit_caller_id(("Which number should people see?", "Verify another number you own"),
                        ("Use the saved Telnyx key?", "Use the saved key"),
                        ("Your mobile number", ["+15555550102"]),
                        *[("Enter the code", ["000000"])] * 3,
                        ("Verification did not complete", "Continue"), ("Everything works", "<quit>"))
    drive(ui, actions)
    assert "still calls out as +15555550101" in ui.questions("Verification did not complete")[0].detail
    assert line["caller_id"] == "+15555550101" and not actions.changes()


@pytest.mark.parametrize("failure", ["start_caller_id", "use_caller_id"])
def test_caller_id_errors_return_to_the_choices_without_changing_the_saved_number(failure):
    line = caller_id_line()
    actions = FakeActions(exists=True, lines={"main": line}, saved_keys={"TELNYX_API_KEY": "KEYtest"},
                          owned_numbers=CALLER_ID_OWNED,
                          **{failure: ActionError("Could not update caller ID")})
    select = [("Which number should people see?", "Line's number · +15555550100")] if failure == "use_caller_id" else [
        ("Which number should people see?", "Verify another number you own"),
        ("Use the saved Telnyx key?", "Use the saved key"), ("Your mobile number", ["+15555550102"])]
    ui = edit_caller_id(*select, ("Which number should people see?", "<quit>"))
    drive(ui, actions)
    assert ui.questions("Which number should people see?")[-1].error == "Could not update caller ID"
    assert line["caller_id"] == "+15555550101" and not actions.args("record_caller_id")


def test_verification_with_another_carrier_account_stops_before_sending_a_code():
    line = caller_id_line()
    actions = FakeActions(exists=True, lines={"main": line}, saved_keys={"TELNYX_API_KEY": "KEYtest"}, owned_numbers=[])
    ui = edit_caller_id(("Which number should people see?", "Verify another number you own"),
                        ("Use the saved Telnyx key?", "Use the saved key"),
                        ("Which number should people see?", "<quit>"))
    drive(ui, actions)
    assert "Could not find +15555550100 on this Telnyx account" in ui.questions("Which number should people see?")[-1].error
    assert not ui.questions("Your mobile number") and not actions.args("start_caller_id")
    assert line["caller_id"] == "+15555550101" and not actions.changes()


def test_verification_can_retry_with_the_selected_lines_account():
    wrong, right = {"TELNYX_API_KEY": "KEYwrong"}, {"TELNYX_API_KEY": "KEYright"}
    actions = FakeActions(exists=True, lines={"main": caller_id_line()}, saved_keys=wrong,
                          owned_numbers=Seq([], CALLER_ID_OWNED))
    ui = edit_caller_id(("Which number should people see?", "Verify another number you own"),
                        ("Use the saved Telnyx key?", "Use the saved key"),
                        ("Which number should people see?", "Verify another number you own"),
                        ("Use the saved Telnyx key?", "Enter a different key"),
                        ("Your Telnyx credentials", ["KEYright"]),
                        ("Your mobile number", ["+15555550102"]), ("Enter the code", ["123456"]),
                        ("Everything works", "<quit>"))
    drive(ui, actions)
    assert actions.args("owned_numbers") == [("telnyx", wrong), ("telnyx", right)]
    assert actions.args("start_caller_id") == [("telnyx", right, "+15555550102")]
    assert actions.args("record_caller_id") == [("main", "+15555550102", "telnyx")]
    assert all("KEYwrong" not in page_text(p) and "KEYright" not in page_text(p) for p in ui.pages)


@pytest.mark.parametrize("at_mobile", [False, True])
def test_back_before_verifying_returns_to_the_existing_caller_id_choices(at_mobile):
    actions = FakeActions(exists=True, lines={"main": caller_id_line()}, owned_numbers=CALLER_ID_OWNED)
    credentials = [("Your Telnyx credentials", ["KEYtest"]), ("Your mobile number", "<back>")] if at_mobile else [
        ("Your Telnyx credentials", "<back>")]
    ui = edit_caller_id(("Which number should people see?", "Verify another number you own"), *credentials,
                        ("Which number should people see?", "<quit>"))
    drive(ui, actions)
    assert not actions.changes() and not actions.args("start_caller_id")


def test_no_owned_numbers_offers_to_buy_one():
    ui = ScriptedUI([
        *add_a_line("Telnyx"),
        ("Your Telnyx credentials", ["KEYnew"]),
        ("A new number, or one already on", "One already on my Telnyx account"),
        ("No numbers on this Telnyx account", "Buy a new number"),
        ("Where should the number be?", ["", ""]),
        ("Pick a number", "<quit>"),
    ])
    actions = FakeActions(exists=True, owned_numbers=[])
    drive(ui, actions)
    assert ui.questions("No numbers on this Telnyx account")[0].detail == \
        "A number on your phone's SIM cannot be connected; buy one here, or add a number in the Telnyx console and run setup again."
    assert ui.questions("A new number, or one already on")[0].checks == [Check("Telnyx key", "works · 0 numbers on this account", True)]
    assert [(f.label, f.placeholder) for f in ui.fields[:1]] == [("API key", "KEY…")]
    assert ui.menu("A new number, or one already on")[1][1].hint == "none on this account"


def test_numbers_already_connected_are_left_out():
    """Connecting a number twice would overwrite its routing, so numbers that are already a line are not offered."""
    ui = ScriptedUI([
        *add_a_line("Telnyx"),
        ("Use the saved Telnyx key?", "Use the saved key"),
        ("A new number, or one already on", "One already on my Telnyx account"),
        ("Which number?", "+15550100008"),
        ("Connect +15550100008 to ai-phone?", "<quit>"),
    ])
    actions = FakeActions(exists=True, saved_keys={"TELNYX_API_KEY": "k"},
                          lines={"telnyx": {"kind": "trunk", "number": "+15550100007"}, "agent": SIP_LINE})
    drive(ui, actions)
    assert [o.label for o in ui.menu("Which number?")[1]] == ["+15550100008"]
    assert ui.questions("A new number, or one already on")[0].checks == [Check("Telnyx key", "works · 2 numbers on this account", True)]
    assert ui.menu("A new number, or one already on")[1][1].hint == "1 not yet connected"

    ui = ScriptedUI([
        *add_a_line("Telnyx"),
        ("Use the saved Telnyx key?", "Use the saved key"),
        ("A new number, or one already on", "One already on my Telnyx account"),
        ("No other numbers on this Telnyx account", "Buy a new number"),
        ("Where should the number be?", ["", ""]),
        ("Pick a number", "<quit>"),
    ])
    lines = {"telnyx": {"kind": "trunk", "number": "+15550100007"}, "telnyx-2": {"kind": "trunk", "number": "+15550100008"}}
    actions = FakeActions(exists=True, saved_keys={"TELNYX_API_KEY": "k"}, lines=lines)
    drive(ui, actions)
    page = ui.questions("No other numbers")[0]
    assert page.detail == "A number on your phone's SIM cannot be connected; buy one here, or add a number in the Telnyx console and run setup again."
    assert [o.label for o in ui.menu("No other numbers")[1]] == ["Buy a new number"]
    assert ui.menu("A new number, or one already on")[1][1].hint == "none left to connect"
    assert not actions.args("attach_number")


def test_a_forward_that_cannot_be_saved_is_asked_again():
    broken = "config.yaml is not valid YAML; fix it or move it away, then run setup again"
    ui = ScriptedUI([
        *add_a_line("Telnyx"),
        ("Use the saved Telnyx key?", "Use the saved key"),
        ("A new number, or one already on", "One already on my Telnyx account"),
        ("Which number?", "+15550100007"),
        ("Connect +15550100007 to ai-phone?", "Connect +15550100007"),
        ("Also forward your own mobile to it?", "Yes"),
        ("Your mobile number", ["+447700900123"]),
        ("When should calls go to your agent?", "All calls"),
        ("Also forward your own mobile to it?", "No"),
        CALLER_ID_NO,
        ("Everything works", "<quit>"),
    ])
    actions = FakeActions(exists=True, saved_keys={"TELNYX_API_KEY": "k"}, forward=ActionError(broken))
    drive(ui, actions)
    first, second = ui.questions("Also forward")
    assert first.error == "" and second.error == broken
    assert actions.args("forward") == [("telnyx", "+447700900123", "all")]
    assert not ui.questions("Dial this")


def test_numbers_with_an_unfinished_setup_are_left_out():
    """A number an earlier attempt left half set up belongs to `lines resume` / `lines remove`, not to a new line."""
    ui = ScriptedUI([
        *add_a_line("Telnyx"),
        ("Use the saved Telnyx key?", "Use the saved key"),
        ("A new number, or one already on", "One already on my Telnyx account"),
        ("Which number?", "<quit>"),
    ])
    actions = FakeActions(exists=True, saved_keys={"TELNYX_API_KEY": "k"}, unfinished_numbers={"+15550100007"})
    drive(ui, actions)
    assert [o.label for o in ui.menu("Which number?")[1]] == ["+15550100008"]
    assert ui.menu("A new number, or one already on")[1][1].hint == "1 not yet connected"


def test_pasted_keys_lose_their_spaces():
    ui = ScriptedUI([
        *add_a_line("Twilio"),
        ("Your Twilio credentials", [" ACsid ", "tokSECRET\t "]),
        ("A new number, or one already on", "<quit>"),
    ])
    actions = FakeActions(exists=True)
    drive(ui, actions)
    assert actions.args("owned_numbers") == [("twilio", TWILIO)]
    ui = ScriptedUI([
        ("Give your AI", "Change the brain"),
        ("Where should your agent think?", "My own endpoint"),
        ("Your endpoint", ["https://api.example.com/v1", "qwen", "  sk-test  "]),
        ("A few things need attention", "Check again"),
        ("A few things need attention", "<quit>"),
    ])
    actions = FakeActions(exists=True)
    drive(ui, actions)
    assert actions.args("test_endpoint") == [("https://api.example.com/v1", "qwen", "sk-test")]
    ui = ScriptedUI([
        ("Give your AI", "Change the brain"),
        ("Where should your agent think?", "My own endpoint"),
        ("Your endpoint", ["https://api.example.com/v1", "qwen", "   "]),
        ("A few things need attention", "<quit>"),
    ])
    actions = FakeActions(exists=True)
    drive(ui, actions)
    assert actions.args("save_brain_endpoint") == [("https://api.example.com/v1", "qwen", None)]  # only spaces: no key


def test_bad_key_is_asked_again_with_the_error():
    ui = ScriptedUI([
        *add_a_line("Telnyx"),
        ("Your Telnyx credentials", ["KEYbad"]),
        ("Your Telnyx credentials", ["KEYgood"]),
        ("A new number, or one already on", "<quit>"),
    ])
    actions = FakeActions(exists=True, owned_numbers=Seq(ActionError("That key did not work: 401 Unauthorized"), OWNED))
    assert drive(ui, actions).code == 1
    first, _busy, second, _busy2 = ui.questions("Your Telnyx credentials")
    assert first.error == "" and second.error == "That key did not work: 401 Unauthorized"
    assert actions.args("owned_numbers") == [("telnyx", {"TELNYX_API_KEY": "KEYbad"}), ("telnyx", {"TELNYX_API_KEY": "KEYgood"})]
    assert all("KEYbad" not in page_text(p) for p in ui.pages)


def test_a_saved_key_that_fails_asks_for_a_new_one():
    ui = ScriptedUI([
        *add_a_line("Twilio"),
        ("Use the saved Twilio key?", "Use the saved key"),
        ("Your Twilio credentials", ["ACsid", "tokSECRET"]),
        ("A new number, or one already on", "<quit>"),
    ])
    old = {"TWILIO_ACCOUNT_SID": "ACold", "TWILIO_AUTH_TOKEN": "old"}
    actions = FakeActions(exists=True, saved_keys=old,
                          owned_numbers=Seq(ActionError("That key did not work: 401 Unauthorized"), OWNED))
    drive(ui, actions)
    assert ui.questions("Your Twilio credentials")[0].error == "That key did not work: 401 Unauthorized"
    assert actions.args("owned_numbers") == [("twilio", old), ("twilio", TWILIO)]


def test_enter_a_different_key_asks_for_it():
    ui = ScriptedUI([
        *add_a_line("Twilio"),
        ("Use the saved Twilio key?", "Enter a different key"),
        ("Your Twilio credentials", ["ACsid", "tokSECRET"]),
        ("A new number, or one already on", "<quit>"),
    ])
    actions = FakeActions(exists=True, saved_keys={"TWILIO_ACCOUNT_SID": "ACold", "TWILIO_AUTH_TOKEN": "old"})
    drive(ui, actions)
    assert actions.args("owned_numbers") == [("twilio", TWILIO)]
    assert [(o.label, o.hint) for o in ui.menu("Use the saved Twilio key?")[1]] == [
        ("Use the saved key", ""), ("Enter a different key", "replaces the saved key")]


def test_pending_number_shows_the_messages_and_skips_forwarding():
    messages = ["order pending", "Line twilio is written but pending. Resume with: ai-phone lines resume twilio --key-env <NAME>"]
    ui = ScriptedUI([
        *add_a_line(),
        ("Your Twilio credentials", ["ACsid", "tokSECRET"]),
        ("A new number, or one already on", "A new number"),
        ("Where should the number be?", ["", "555"]),
        ("Pick a number", "+15550100002"),
        ("Buy +15550100002 for $1.15/month?", "Buy +15550100002"),
        ("The number is not ready yet", "Continue"),
        ("A few things need attention", "Continue"),
        FINISH,
    ])
    actions = FakeActions(exists=True, buy_number=LineResult("twilio", 2, messages),
                          check_lines=[("twilio", "registering", "")])
    assert drive(ui, actions).code == 0
    pending = ui.questions("The number is not ready yet")[0]
    assert pending.body == "\n".join(messages)
    assert ui.menu("The number is not ready yet")[1][0].label == "Continue"
    assert not ui.questions("Also forward") and not actions.args("forward")
    assert ui.questions("A few things")[0].checks[2] == Check("Line twilio", "registering", None)


def test_failed_number_says_something_went_wrong():
    ui = ScriptedUI([
        *add_a_line(),
        ("Your Twilio credentials", ["ACsid", "tokSECRET"]),
        ("A new number, or one already on", "A new number"),
        ("Where should the number be?", ["", ""]),
        ("Pick a number", "+15550100001"),
        ("Buy +15550100001", "Buy +15550100001"),
        ("Something went wrong", "Continue"),
        ("A few things need attention", "<quit>"),
    ])
    actions = FakeActions(exists=True, buy_number=Seq(ActionError("could not save the key: disk full")))
    drive(ui, actions)
    assert ui.questions("Something went wrong")[0].body == "could not save the key: disk full"


def test_own_trunk():
    ui = ScriptedUI([
        ("Give your AI", "Add a line"),
        ("Which connection should ai-phone use?", "My own SIP trunk"),
        ("What should outgoing SIP From use?", "<default>"),
        ("Your SIP trunk", ["sip.example.com", "alex1234", "pwSECRET", "+15550100000"]),
        ("Which transport?", "TLS"),
        ("Your SIP trunk", ["", "", "pwSECRET2", ""]),
        ("Which transport?", "TLS"),
        ("Everything works", "Continue"),
        FINISH,
    ])
    bad = "config.lines.trunk.registrar: expected sip:host[:port][;transport=…]"
    actions = FakeActions(exists=True, add_trunk_line=Seq(ActionError(bad), "trunk"))
    assert drive(ui, actions).code == 0
    assert actions.args("add_trunk_line") == [("sip:sip.example.com", "alex1234", "pwSECRET", "+15550100000", "tls", "number"),
                                              ("sip:sip.example.com", "alex1234", "pwSECRET2", "+15550100000", "tls", "number")]
    retry = ui.questions("Your SIP trunk")[1]
    assert retry.error == bad
    assert [f.default for f in ui.fields[4:]] == ["sip.example.com", "alex1234", "", "+15550100000"]
    assert ui.fields[2].secret and not ui.fields[0].secret
    assert [(o.label, o.hint) for o in ui.menu("Which transport?")[1]] == [("UDP", ""), ("TCP", ""), ("TLS", "encrypted signalling")]
    assert all("pwSECRET" not in page_text(p) for p in ui.pages)


def test_own_trunk_keeps_sips_and_the_password_as_typed():
    ui = ScriptedUI([
        ("Give your AI", "Add a line"),
        ("Which connection should ai-phone use?", "My own SIP trunk"),
        ("What should outgoing SIP From use?", "Phone number"),
        ("Your SIP trunk", ["  sips:sip.example.com ", " alex1234 ", "  pw SECRET  ", " +15550100000 "]),
        ("Which transport?", "TLS"),
        ("Everything works", "<quit>"),
    ])
    actions = FakeActions(exists=True)
    drive(ui, actions)
    assert actions.args("add_trunk_line") == [("sips:sip.example.com", "alex1234", "  pw SECRET  ", "+15550100000", "tls", "number")]


def test_own_trunk_account_from_allows_no_number_and_explains_provider_caller_id():
    ui = ScriptedUI([
        ("Give your AI", "Add a line"),
        ("Which connection should ai-phone use?", "My own SIP trunk"),
        ("What should outgoing SIP From use?", "SIP username"),
        ("Your SIP trunk", ["trunk.example.org", "account1", "pwSECRET", ""]),
        ("Which transport?", "UDP"),
        ("Everything works", "Continue"), FINISH,
    ])
    actions = FakeActions(exists=True)
    outcome = drive(ui, actions)
    assert outcome.code == 0
    assert actions.args("add_trunk_line") == [("sip:trunk.example.org", "account1", "pwSECRET", "", "udp", "account")]
    assert "caller ID set by your provider" in outcome.summary
    assert "provider controls the displayed caller ID" in page_text(ui.questions("What should outgoing SIP From use?")[0])
    assert ui.fields[3].label == "Your number on it (optional)"
    assert not actions.args("start_caller_id")


def test_own_trunk_wizard_saves_an_underscored_username_with_real_validation(tmp_path, monkeypatch):
    from ai_phone.config import load_config
    from ai_phone.secrets import load_secrets
    from ai_phone.setup.actions import Actions

    monkeypatch.setenv("AI_PHONE_HOME", str(tmp_path))
    path = tmp_path / "config.yaml"
    ui = ScriptedUI([
        ("What should outgoing SIP From use?", "SIP username"),
        ("Your SIP trunk", ["trunk.example.org", "usuario_12345", "synthetic-password", ""]),
        ("Which transport?", "UDP"),
    ])
    # Exercise the real config/secret writer; no model, daemon or carrier check is involved.
    flow.own_trunk(ui, Actions(path, player=object()), flow.Session())
    assert not ui.answers
    line = load_config(path).lines["trunk"]
    assert line.user == "usuario_12345"
    assert line.from_user == "account" and line.number is None and line.caller_id is None
    assert load_secrets(tmp_path / "secrets.env")[line.password_env] == "synthetic-password"
    assert all("synthetic-password" not in page_text(p) for p in ui.pages)


def test_sip_address_name_taken_is_asked_again():
    taken = "sip:agent@192.0.2.10 is already line agent; pick another name"
    ui = ScriptedUI([
        ("Give your AI", "Add a line"),
        ("Which connection should ai-phone use?", "A SIP address"),
        ("Name your SIP address", ["", ""]),
        ("Name your SIP address", ["desk", ""]),
        ("Everything works", "<quit>"),
    ])
    actions = FakeActions(exists=True, add_sip_line=Seq(ActionError(taken), "desk"))
    drive(ui, actions)
    assert actions.args("add_sip_line") == [("agent", "192.0.2.10"), ("desk", "192.0.2.10")]
    assert ui.questions("Name your SIP address")[1].error == taken
    assert ui.questions("Name your SIP address")[0].detail == \
        "Softphones and other agents call sip:<name>@<this computer>. Free; no carrier."


def test_sip_address_offers_a_free_name():
    ui = ScriptedUI([
        ("Give your AI", "Add a line"),
        ("Which connection should ai-phone use?", "A SIP address"),
        ("Name your SIP address", ["", ""]),
        ("Everything works", "<quit>"),
    ])
    lines = {"agent": SIP_LINE, "agent-2": {"kind": "trunk", "number": "+15550100001"},
             "desk": {"kind": "sip", "aor": "sip:agent-3@192.0.2.10"}}
    actions = FakeActions(exists=True, lines=lines)
    drive(ui, actions)
    assert actions.args("add_sip_line") == [("agent-4", "192.0.2.10")]


def test_validators():
    assert flow.required("  ") == "Enter a value." and flow.required("x") is None
    assert flow.e164("+15550100000") is None and flow.e164("5550100000") == "Use international format, like +15550100000."
    assert flow.country("us") is None and flow.country("USA") == "Use a two-letter country code, like US or GB."
    assert flow.area("") is None and flow.area("212") is None and flow.area("1") and flow.area("12a")
    assert flow.sip_name("agent-2") is None and flow.sip_name("Agent") == "Use lowercase letters, digits and dashes."
    assert flow.sip_name("a.b") and flow.sip_name("x" * 33)
    assert flow.host("192.0.2.10") is None and flow.host("mac.example.com") is None and flow.host("my mac")
    assert flow.url("https://api.example.com/v1") is None and flow.url("api.example.com")
    for local in ("http://127.0.0.1:1234/v1", "http://localhost:8080/v1", "http://[::1]:8080/v1"):
        assert flow.url(local) is None
    assert flow.url("http://203.0.113.5/v1") == flow.url("http://api.example.com/v1") == \
        "Use https:// for a server that is not on this computer."
    assert flow.url("https://203.0.113.5/v1") is None and flow.url("http://") and flow.url("https://[bad")
    assert flow.speed("0.9") is None and flow.speed("2") is None
    assert flow.speed("2.5") == flow.speed("fast") == flow.speed("nan") == "Use a number from 0.5 to 2.0."
    assert flow.count("0") is None and flow.count("20") is None and flow.count("-1") and flow.count("1.5")


def test_price_size_and_plural_helpers():
    assert flow.price(Number("+1", 1.15, "USD", False, "")) == "$1.15/month"
    assert flow.price(Number("+1", 1.0, "GBP", False, "")) == "£1.00/month"
    assert flow.price(Number("+1", 1.0, "EUR", False, "")) == "€1.00/month"
    assert flow.price(Number("+1", 1.0, "XYZ", False, "")) == "1.00 XYZ/month"
    assert flow.price(Number("+1", None, None, False, "")) == "the carrier's monthly price"
    assert flow.gb(4_300_000_000) == "4.3 GB" and flow.gb(None) == "several GB"
    assert flow.plural(1, "number") == "1 number" and flow.plural(2, "number") == "2 numbers"


def test_the_brain_step_offers_the_whole_model_catalog():
    ui = ScriptedUI([
        WELCOME_NEW,
        MODE_BOTH,
        ("Where should your agent think?", "On this computer"),
        ("Which model?", "<default>"),
        ("Which connection should ai-phone use?", "<quit>"),
    ])
    actions = FakeActions(memory=24)
    drive(ui, actions)
    _q, options, default, _back = ui.menu("Which model?")
    assert ui.questions("Which model?")[0].detail == \
        "This computer has 24 GB of memory. Models that need more may swap and answer slowly."
    assert [(o.label, o.hint) for o in options] == [
        ("Qwen3.5 4B", "recommended for your RAM · needs 16 GB"),
        ("Qwen3.5 9B", "needs 32 GB · exceeds system RAM"),
        ("Qwen3.5 2B", "needs 8 GB"),
        ("Llama 3.2 3B", "needs 16 GB"),
        ("Gemma 3 4B", "needs 16 GB"),
        ("Mistral 7B", "needs 24 GB"),
        ("Gemma 3 12B", "needs 32 GB · exceeds system RAM"),
    ]
    assert options[default].label == "Qwen3.5 4B"  # 24 GB: the recommended one
    assert actions.args("save_brain_local") == [("t4",)]


def test_a_model_this_computer_cannot_hold_is_confirmed_first():
    ui = ScriptedUI([
        WELCOME_NEW,
        MODE_BOTH,
        ("Where should your agent think?", "On this computer"),
        ("Which model?", "Gemma 3 12B"),
        ("Use Gemma 3 12B anyway?", "<default>"),  # Enter is Back: nothing is saved
        ("Which model?", "Gemma 3 12B"),
        ("Use Gemma 3 12B anyway?", "Use it anyway"),
        ("Which connection should ai-phone use?", "<quit>"),
    ])
    actions = FakeActions(memory=24)
    drive(ui, actions)
    assert ui.questions("Use Gemma 3 12B anyway?")[0].detail == \
        "It needs 32 GB; this computer has 24 GB, so it may swap and answer slowly."
    assert ui.menu("Use Gemma 3 12B anyway?")[2:] == (1, True)  # the default is Back
    assert [o.label for o in ui.menu("Use Gemma 3 12B anyway?")[1]] == ["Use it anyway"]
    assert actions.args("save_brain_local") == [("gemma12b",)]
    assert [m[2] for m in ui.menus if "Which model?" in m[0]] == [0, 6]  # Back leaves the big one picked


@pytest.mark.parametrize("memory,recommended,default", [
    (0, None, "Qwen3.5 4B"), (4, None, "Qwen3.5 4B"),
    (8, "Qwen3.5 2B", "Qwen3.5 2B"), (16, "Qwen3.5 4B", "Qwen3.5 4B"),
    (24, "Qwen3.5 4B", "Qwen3.5 4B"), (32, "Qwen3.5 9B", "Qwen3.5 9B"),
])
def test_first_setup_recommends_only_a_model_that_fits_detected_ram(memory, recommended, default):
    ui = ScriptedUI([WELCOME_NEW, MODE_BOTH, ("Where should your agent think?", "On this computer"),
                     ("Which model?", "<quit>")])
    drive(ui, FakeActions(memory=memory))
    _, options, picked, _ = ui.menu("Which model?")
    assert [o.label for o in options if "recommended" in o.hint] == ([recommended] if recommended else [])
    assert options[picked].label == default
    if memory == 0:
        assert ui.pages[0].checks[1] == Check("Memory", "could not detect memory", None)
        assert "Memory could not be detected" in ui.questions("Which model?")[0].detail
        assert not any("exceeds" in o.hint for o in options)


def test_a_model_that_fits_is_saved_without_a_confirmation():
    ui = ScriptedUI([
        WELCOME_NEW,
        MODE_BOTH,
        ("Where should your agent think?", "On this computer"),
        ("Which model?", "Mistral 7B"),
        ("Which connection should ai-phone use?", "<quit>"),
    ])
    actions = FakeActions(memory=24)
    drive(ui, actions)
    assert not ui.questions("anyway?")
    assert actions.args("save_brain_local") == [("mistral7b",)]


def test_a_repo_named_by_hand_in_the_config_keeps_the_recommended_default():
    """models.tier may be any Hub repo; the wizard cannot name that one, so it offers the recommended one."""
    ui = ScriptedUI([
        ("Give your AI", "Change the brain"),
        ("Where should your agent think?", "On this computer"),
        ("Which model?", "<default>"),
        ("A few things need attention", "<quit>"),
    ])
    actions = FakeActions(exists=True, memory=32)
    actions.cur.tier = "org/my-own-4bit"
    drive(ui, actions)
    assert actions.args("save_brain_local") == [("t9",)]


def test_back_at_the_first_question_goes_to_the_previous_step():
    ui = ScriptedUI([
        WELCOME_NEW,
        MODE_BOTH,
        ("Where should your agent think?", "On this computer"),
        ("Which model?", "Qwen3.5 9B"),
        ("Which connection should ai-phone use?", "<back>"),
        ("Where should your agent think?", "On this computer"),
        ("Which model?", "Qwen3.5 4B"),
        ("Which connection should ai-phone use?", "Skip"),
        ("Which speech model?", "Kokoro"),
        ("How should your agent sound?", "<quit>"),
    ])
    actions = FakeActions(memory=32)
    assert drive(ui, actions).code == 1
    assert ui.steps == [WELCOME, MODE, BRAIN, LINE, BRAIN, LINE, VOICE]
    assert actions.args("save_brain_local") == [("t9",), ("t4",)]
    model_menus = [menu for menu in ui.menus if menu[0] == "Which model?"]
    assert model_menus[-1][2] == 1  # the model menu now defaults to what was saved


def test_back_later_in_a_step_restarts_it():
    ui = ScriptedUI([
        WELCOME_NEW,
        MODE_BOTH,
        ("Where should your agent think?", "My own endpoint"),
        ("Your endpoint", "<back>"),
        ("Where should your agent think?", "On this computer"),
        ("Which model?", "<default>"),
        ("Which connection should ai-phone use?", "<quit>"),
    ])
    actions = FakeActions()
    drive(ui, actions)
    assert ui.steps == [WELCOME, MODE, BRAIN, BRAIN, LINE]
    assert actions.changes() == ["save_call_directions", "save_brain_local"]


def test_back_from_the_first_step_returns_to_welcome():
    ui = ScriptedUI([
        WELCOME_NEW,
        ("How should ai-phone work?", "<back>"),
        WELCOME_NEW,
        ("How should ai-phone work?", "<quit>"),
    ])
    actions = FakeActions()
    assert drive(ui, actions).code == 1
    assert ui.steps == [WELCOME, MODE, WELCOME, MODE]
    assert not actions.changes()


def test_back_from_the_only_step_of_a_change_returns_to_welcome():
    ui = ScriptedUI([
        ("Give your AI", "Change call settings"),
        ("What should callers hear?", "<back>"),
        ("Give your AI", "Cancel"),
    ])
    actions = FakeActions(exists=True)
    assert drive(ui, actions).code == 1
    assert ui.steps == [WELCOME, CALLERS, WELCOME]
    assert not actions.changes()


def test_cancel_keeps_finished_steps_only():
    ui = ScriptedUI([
        WELCOME_NEW,
        MODE_BOTH,
        ("Where should your agent think?", "On this computer"),
        ("Which model?", "<default>"),
        ("Which connection should ai-phone use?", "A SIP address"),
        ("Name your SIP address", ["", ""]),
        ("Which speech model?", "Kokoro"),
        ("How should your agent sound?", "<quit>"),
    ])
    actions = FakeActions()
    outcome = drive(ui, actions)
    assert outcome.code == 1 and ui.closed
    assert outcome.summary == ("Setup cancelled. Saved in this run: call directions, the brain, line agent (sip:agent@192.0.2.10). "
                               "Nothing else was changed.")
    assert actions.changes() == ["save_call_directions", "save_brain_local", "add_sip_line"]


def buy_then(*after):
    return ScriptedUI([
        *add_a_line(),
        ("Your Twilio credentials", ["ACsid", "tokSECRET"]),
        ("A new number, or one already on", "A new number"),
        ("Where should the number be?", ["", ""]),
        ("Pick a number", "+15550100001"),
        ("Buy +15550100001 for $1.15/month?", "Buy +15550100001"),
        *after,
    ])


def test_a_cancel_after_a_purchase_names_the_number():
    """People who read "nothing changed" do not go looking for a monthly charge."""
    outcome = drive(buy_then(("Also forward your own mobile to it?", "<quit>")), FakeActions(exists=True))
    assert outcome == Outcome(1, "Setup cancelled. Saved in this run: line twilio (+15550100001, a Twilio number billed "
                                 "monthly). Nothing else was changed.")
    messages = ["order pending"]
    actions = FakeActions(exists=True, buy_number=LineResult("twilio", 2, messages))
    outcome = drive(buy_then(("The number is not ready yet", "<quit>")), actions)
    assert outcome.summary == ("Setup cancelled. Saved in this run: line twilio (+15550100001, not finished: "
                               "ai-phone lines resume twilio, or ai-phone lines remove twilio). "
                               "Nothing else was changed.")
    ui = ScriptedUI([
        *add_a_line("Telnyx"),
        ("Use the saved Telnyx key?", "Use the saved key"),
        ("A new number, or one already on", "One already on my Telnyx account"),
        ("Which number?", "+15550100007"),
        ("Connect +15550100007 to ai-phone?", "Connect +15550100007"),
        ("Also forward your own mobile to it?", "Yes"),
        ("Your mobile number", ["+447700900123"]),
        ("When should calls go to your agent?", "All calls"),
        ("Dial this on your mobile", "Done"),
        CALLER_ID_NO,
        ("Everything works", "<quit>"),
    ])
    outcome = drive(ui, FakeActions(exists=True, saved_keys={"TELNYX_API_KEY": "k"}))
    assert outcome.summary == ("Setup cancelled. Saved in this run: line telnyx (+15550100007), call forwarding from "
                               "+447700900123 to line telnyx. Nothing else was changed.")


def test_a_failed_purchase_keeps_its_line_id_for_the_summary():
    error = ActionError("disk full\nThe journal is kept: run `ai-phone lines resume twilio` to finish it, or "
                        "`ai-phone lines remove twilio` to undo it.")
    error.line_id = "twilio"
    outcome = drive(buy_then(("Something went wrong", "<quit>")), FakeActions(exists=True, buy_number=Seq(error)))
    assert outcome.summary == ("Setup cancelled. Saved in this run: line twilio (+15550100001, not finished: "
                               "ai-phone lines resume twilio, or ai-phone lines remove twilio). "
                               "Nothing else was changed.")


def test_ctrl_c_during_a_purchase_says_it_may_be_half_done():
    """Plain mode: ctrl+c lands inside the carrier requests."""
    session = flow.Session()
    ui = buy_then()
    outcome = flow.run(ui, FakeActions(exists=True, save_brain_local=None, buy_number=Seq(KeyboardInterrupt())), session)
    assert outcome == Outcome(1, "\n".join([
        "Setup stopped during: Buying +15550100001 and connecting it…",
        "That carrier step may be half done. See your lines with: ai-phone lines list",
        "then finish one with: ai-phone lines resume <line>, or undo it with: ai-phone lines remove <line>",
    ]))
    assert ui.closed and not ui.answers and not session.left_download
    ui = ScriptedUI([
        WELCOME_NEW,
        MODE_BOTH,
        ("Where should your agent think?", "On this computer"),
        ("Which model?", "<default>"),
        ("Which connection should ai-phone use?", "A SIP address"),
        ("Name your SIP address", ["", ""]),
        ("Which speech model?", "Kokoro"),
        ("How should your agent sound?", "American · female"),
        ("American · female", "Heart"),
        ("Heart", "Hear it"),
    ])
    outcome = flow.run(ui, FakeActions(play_sample=KeyboardInterrupt()))
    assert outcome == Outcome(1, "Setup stopped during: Playing…\nSaved in this run: call directions, the brain, line agent "
                                 "(sip:agent@192.0.2.10). Nothing else was changed.")


def test_ctrl_c_during_a_download_says_it_resumes():
    session = flow.Session()
    ui = ScriptedUI([
        ("Give your AI", "Change call settings"),
        ("What should callers hear?", ["", "", ""]),
        ("Download the models now?", "Download now"),
    ])
    actions = FakeActions(exists=True, missing=["mlx-community/Qwen3.5-4B-MLX-4bit"], pull_models=KeyboardInterrupt())
    outcome = flow.run(ui, actions, session)
    assert outcome == Outcome(1, "Setup cancelled. The download stops now and picks up where it left off next time. "
                                 "Saved in this run: call settings.")
    assert session.left_download and flow.cancel_text(session) == outcome.summary
    ui = ScriptedUI([("Give your AI", "Change the voice"), ("Which speech model?", "Kokoro"), ("How should your agent sound?", "Download it now")])
    session = flow.Session()
    outcome = flow.run(ui, FakeActions(exists=True, voices=[], pull_voice_model=KeyboardInterrupt()), session)
    assert outcome == Outcome(1, "Setup cancelled. The download stops now and picks up where it left off next time.")
    assert session.left_download


def test_a_finished_or_failed_step_is_no_longer_the_one_running():
    session = flow.Session()
    ui = ScriptedUI([
        ("Give your AI", "Change call settings"),
        ("What should callers hear?", ["", "", ""]),
        ("Download the models now?", "Download now"),
        ("A few things need attention", "Continue"),
        ("You're set up", "<quit>"),
    ])
    actions = FakeActions(exists=True, missing=["m"], lines={"agent": SIP_LINE}, pull_models=ActionError("could not download m"))
    assert flow.run(ui, actions, session).code == 0
    assert session.working is None and not session.left_download
    assert session.saved == ["call settings"]
    ui = ScriptedUI([("Give your AI", "Change call settings"), ("What should callers hear?", ["", "", ""]),
                     ("Download the models now?", "Download now"), ("Everything works", "<quit>")])
    session = flow.Session()
    flow.run(ui, FakeActions(exists=True, missing=["m"], lines={"agent": SIP_LINE}), session)
    assert session.saved == ["call settings", "the downloaded models"]
    assert flow.cancel_text(session) == ("Setup cancelled. Saved in this run: call settings, the downloaded models. "
                                         "Nothing else was changed.")


def test_leaving_on_the_done_page_still_finishes():
    ui = ScriptedUI([
        ("Give your AI", "Change call settings"),
        ("What should callers hear?", ["", "", "0"]),
        ("A few things need attention", "Continue"),
        ("You're set up", "<quit>"),
    ])
    actions = FakeActions(exists=True)
    outcome = drive(ui, actions)
    assert outcome.code == 0 and "ai-phone run" in outcome.summary
    assert actions.args("save_callers") == [(GREETING, INSTRUCTIONS, 0)]
    assert ui.questions("A few things")[0].checks[2] == Check("Lines", "none yet · run setup again to add one", None)
    assert not actions.args("check_lines")  # nothing to check, so nothing is started


def test_callers_that_do_not_validate_are_asked_again():
    ui = ScriptedUI([
        ("Give your AI", "Change call settings"),
        ("What should callers hear?", ["x" * 501, "", ""]),
        ("What should callers hear?", ["Hello there.", "Take a message.", "5"]),
        ("A few things need attention", "<quit>"),
    ])
    too_long = "config.inbound.greeting: 1 to 500 characters"
    actions = FakeActions(exists=True, save_callers=Seq(ActionError(too_long), None))
    drive(ui, actions)
    assert ui.questions("What should callers hear?")[1].error == too_long
    assert actions.args("save_callers")[-1] == ("Hello there.", "Take a message.", 5)
    assert [f.placeholder for f in ui.fields[:3]] == ["", "", "0 means no limit"]


def test_existing_config_change_the_voice_only():
    ui = ScriptedUI([
        ("Give your AI", "Change the voice"),
        ("Which speech model?", "Kokoro"),
        ("How should your agent sound?", "British · male"),
        ("British · male", "George"),
        ("George", "Hear it"),
        ("George", "Hear it"),
        ("George", "Use this voice"),
        ("How fast should it talk?", ["0.9"]),
        ("Everything works", "Continue"),
        FINISH,
    ])
    actions = FakeActions(exists=True, lines={"agent": SIP_LINE},
                          play_sample=Seq(ActionError("could not play the sample: afplay not found"), None, None))
    assert drive(ui, actions).code == 0
    assert ui.steps == [WELCOME, VOICE, CHECK, DONE]
    assert actions.changes() == ["save_voice"]
    assert actions.args("save_voice") == [("bm_george", 0.9, "kokoro")]
    assert actions.args("play_sample") == [("bm_george", 1.0, flow.SAMPLE, "kokoro")] * 2 + [("bm_george", 0.9, DISCLOSURE, "kokoro")]
    groups = ui.menu("How should your agent sound?")
    assert [(o.label, o.hint) for o in groups[1]] == [("American · female", "2 voices"), ("American · male", "1 voice"),
                                                      ("British · female", "1 voice"), ("British · male", "1 voice")]
    assert groups[2] == 0  # the current voice is American · female
    george = [p for p in ui.pages if p.question == "George"]
    assert [p.error for p in george] == ["", "", "could not play the sample: afplay not found", "", ""]
    assert "Playing…" in ui.labels
    assert [(o.label, o.hint) for o in ui.menu("George")[1]] == [
        ("Use this voice", ""), ("Hear it", "plays a short sample on this computer's speakers")]
    assert "Voice    Kokoro · George · British · male · 0.9×" in ui.questions("You're set up")[0].body
    question, options, default, _ = ui.menu("Which speech model?")
    assert options[default].label == "Kokoro"  # no silent change for an existing config


def test_new_setup_recommends_phonon_previews_then_downloads_the_remaining_models():
    ui = ScriptedUI([
        WELCOME_NEW,
        MODE_BOTH,
        ("Where should your agent think?", "On this computer"),
        ("Which model?", "<default>"),
        ("Which connection should ai-phone use?", "A SIP address"),
        ("Name your SIP address", ["agent", "192.0.2.10"]),
        ("Which speech model?", "<default>"),
        ("Phonon voice", "Hear it"),
        ("Phonon voice", "Use this voice"),
        ("What should callers hear?", ["", "", ""]),
        ("Download the models now?", "Download now"),
        ("Everything works", "Continue"),
        FINISH,
    ])
    actions = FakeActions(missing=["speech-to-text", "language-model"])
    outcome = drive(ui, actions)
    assert outcome.code == 0
    _, options, default, _ = ui.menu("Which speech model?")
    assert options[default].label == "Phonon" and "recommended" in options[default].hint
    assert actions.args("save_voice") == [("default", 1.0, "phonon")]
    assert actions.args("play_sample") == [("default", 1.0, flow.SAMPLE, "phonon"),
                                             ("default", 1.0, DISCLOSURE, "phonon")]
    names = [name for name, _ in actions.calls]
    assert names.index("play_sample") < names.index("pull_models")
    assert not any(field.label == "Speed" for field in ui.fields)
    assert "Voice    Phonon · English · bundled voice" in outcome.summary
    assert ui.questions("Everything works")[0].checks[0] == \
        Check("Models", "Parakeet 0.6B v2\nPhonon 31M\nQwen3.5 4B", True)


@pytest.mark.parametrize("leave", ["<quit>", "<back>"])
def test_leaving_a_phonon_preview_keeps_the_existing_voice(leave):
    answers = [("Give your AI", "Change the voice"), ("Which speech model?", "Phonon"),
               ("Phonon voice", "Hear it"), ("Phonon voice", leave)]
    if leave == "<back>":
        answers.append(("Which speech model?", "<quit>"))
    ui = ScriptedUI(answers)
    actions = FakeActions(exists=True, lines={"agent": SIP_LINE})
    actions.cur.voice, actions.cur.speed = "bm_george", 0.9
    assert drive(ui, actions).code == 1
    assert not actions.changes()
    assert (actions.cur.tts_engine, actions.cur.voice, actions.cur.speed) == ("kokoro", "bm_george", 0.9)


def test_existing_phonon_is_selected_and_switching_to_kokoro_uses_its_defaults():
    ui = ScriptedUI([
        ("Give your AI", "Change the voice"), ("Which speech model?", "Kokoro"),
        ("How should your agent sound?", "American · female"), ("American · female", "<default>"),
        ("Heart", "Use this voice"), ("How fast should it talk?", [""]),
        ("Everything works", "Continue"), FINISH,
    ])
    actions = FakeActions(exists=True, lines={"agent": SIP_LINE})
    actions.cur.tts_engine, actions.cur.voice = "phonon", "default"
    assert drive(ui, actions).code == 0
    _, options, default, _ = ui.menu("Which speech model?")
    assert options[default].label == "Phonon"
    assert actions.args("save_voice") == [("af_heart", 1.0, "kokoro")]


def test_missing_phonon_offers_its_own_download_before_preview():
    ui = ScriptedUI([
        ("Give your AI", "Change the voice"), ("Which speech model?", "Phonon"),
        ("Download Phonon?", "Download it now"), ("Phonon voice", "Hear it"),
        ("Phonon voice", "Use this voice"), ("Everything works", "Continue"), FINISH,
    ])
    actions = FakeActions(exists=True, lines={"agent": SIP_LINE}, phonon_voices=Seq([], ["default"], ["default"]))
    assert drive(ui, actions).code == 0
    assert actions.args("pull_voice_model")[0][1] == "phonon"
    assert actions.args("save_voice") == [("default", 1.0, "phonon")]
    assert actions.args("play_sample")[0] == ("default", 1.0, flow.SAMPLE, "phonon")


def test_phonon_download_can_wait_for_check_without_changing_speed():
    ui = ScriptedUI([
        ("Give your AI", "Change the voice"), ("Which speech model?", "Phonon"),
        ("Download Phonon?", "Use the bundled voice"), ("Everything works", "<quit>"),
    ])
    actions = FakeActions(exists=True, lines={"agent": SIP_LINE}, phonon_voices=[])
    actions.cur.voice, actions.cur.speed = "bm_george", 0.9
    assert drive(ui, actions).code == 1
    assert actions.args("save_voice") == [("default", 1.0, "phonon")]
    assert not actions.args("pull_voice_model") and not actions.args("play_sample")


def test_no_voices_offers_the_speech_model_download():
    ui = ScriptedUI([
        ("Give your AI", "Change the voice"),
        ("Which speech model?", "Kokoro"),
        ("How should your agent sound?", "Download it now"),
        ("How should your agent sound?", "American · female"),
        ("American · female", "Heart"),
        ("Heart", "Use this voice"),
        ("How fast should it talk?", [""]),
        ("Everything works", "<quit>"),
    ])
    actions = FakeActions(exists=True, lines={"agent": SIP_LINE}, voices=Seq([], ["af_heart"], ["af_heart"]))
    drive(ui, actions)
    first = ui.questions("How should your agent sound?")[0]
    assert first.detail == "The speech model is not downloaded yet, so there are no voices to pick from."
    assert [(o.label, o.hint) for o in ui.menu("How should your agent sound?")[1]] == [
        ("Download it now", "about 0.4 GB"), ("Keep the default voice", "Heart · American · female")]
    assert len(actions.args("pull_voice_model")) == 1 and callable(actions.args("pull_voice_model")[0][0])
    assert ui.questions("Downloading the speech model")
    assert actions.args("save_voice") == [("af_heart", 1.0, "kokoro")]


def test_no_voices_keep_the_default():
    ui = ScriptedUI([
        ("Give your AI", "Change the voice"),
        ("Which speech model?", "Kokoro"),
        ("How should your agent sound?", "Keep the default voice"),
        ("Everything works", "<quit>"),
    ])
    actions = FakeActions(exists=True, lines={"agent": SIP_LINE}, voices=[])
    drive(ui, actions)
    assert actions.args("save_voice") == [("af_heart", 1.0, "kokoro")]
    assert not actions.args("pull_voice_model")
    rows = ui.questions("Everything works")[0].checks
    assert [r.label for r in rows] == ["Models", "Config", "Line agent"]  # no speech model: no voice row


def test_no_voices_download_failure_is_shown_on_the_voice_page():
    failed = "could not download mlx-community/Kokoro-82M-bf16: OSError"
    broken = "config.yaml is not valid YAML; fix it or move it away, then run setup again"
    ui = ScriptedUI([
        ("Give your AI", "Change the voice"),
        ("Which speech model?", "Kokoro"),
        ("How should your agent sound?", "Download it now"),
        ("How should your agent sound?", "Keep the default voice"),
        ("How should your agent sound?", "Keep the default voice"),
        ("Everything works", "<quit>"),
    ])
    actions = FakeActions(exists=True, lines={"agent": SIP_LINE}, voices=[], pull_voice_model=ActionError(failed),
                          save_voice=Seq(ActionError(broken), None))
    drive(ui, actions)
    pages = [p for p in ui.pages if p.question == "How should your agent sound?"]
    assert [p.error for p in pages] == ["", failed, broken]
    assert actions.args("save_voice") == [("af_heart", 1.0, "kokoro")] * 2


def test_no_voices_after_the_download_says_so():
    ui = ScriptedUI([
        ("Give your AI", "Change the voice"),
        ("Which speech model?", "Kokoro"),
        ("How should your agent sound?", "Download it now"),
        ("How should your agent sound?", "<quit>"),
    ])
    actions = FakeActions(exists=True, voices=[])
    drive(ui, actions)
    pages = [p for p in ui.pages if p.question == "How should your agent sound?"]
    assert pages[1].error == "The speech model downloaded, but it has no voices to pick from. Try the download again."
    assert not actions.args("save_voice")


def test_a_voice_that_cannot_be_saved_is_asked_again():
    broken = "config.yaml is not valid YAML; fix it or move it away, then run setup again"
    ui = ScriptedUI([
        ("Give your AI", "Change the voice"),
        ("Which speech model?", "Kokoro"),
        ("How should your agent sound?", "<default>"),
        ("American · female", "<default>"),
        ("Heart", "Use this voice"),
        ("How fast should it talk?", ["1.2"]),
        ("How fast should it talk?", [""]),
        ("Everything works", "<quit>"),
    ])
    actions = FakeActions(exists=True, lines={"agent": SIP_LINE}, save_voice=Seq(ActionError(broken), None))
    drive(ui, actions)
    first, second = ui.questions("How fast")
    assert first.error == "" and second.error == broken
    assert [f.default for f in ui.fields] == ["1.0", "1.2"]
    assert actions.args("save_voice") == [("af_heart", 1.2, "kokoro")] * 2


def test_unexpected_errors_stop_setup_without_a_traceback():
    broken = "config.yaml is not valid YAML; fix it or move it away, then run setup again"
    ui = ScriptedUI([("Give your AI", "Change call settings"), ("What should callers hear?", ["", "", ""])])
    outcome = flow.run(ui, FakeActions(exists=True, check_config=ActionError(broken)))
    assert (outcome.code, outcome.summary) == (3, f"Setup stopped: {broken}\nSaved in this run: call settings. "
                                                  "Nothing else was changed.") and ui.closed
    ui = ScriptedUI([("Give your AI", "Change the voice"), ("Which speech model?", "Kokoro")])
    outcome = flow.run(ui, FakeActions(exists=True, voices=KeyError("af_secret_detail")))
    assert (outcome.code, outcome.summary) == (3, "Setup stopped: KeyError\nNothing was changed.") and ui.closed


def test_check_offers_the_download_and_reports():
    ui = ScriptedUI([
        ("Give your AI", "Change call settings"),
        ("What should callers hear?", ["", "", ""]),
        ("Download the models now?", "Download now"),
        ("Everything works", "Check again"),
        ("Download the models now?", "Later"),
        ("A few things need attention", "Check again"),
        ("Download the models now?", "Download now"),
        ("A few things need attention", "Continue"),
        FINISH,
    ])
    failed = "could not download mlx-community/Qwen3.5-4B-MLX-4bit: ConnectError"
    actions = FakeActions(exists=True, missing=["mlx-community/Qwen3.5-4B-MLX-4bit"], lines={"agent": SIP_LINE},
                          pull_models=Seq(None, ActionError(failed)))
    assert drive(ui, actions).code == 0
    assert ui.pages[0].checks[2] == Check("Models", "Parakeet 0.6B v2\nKokoro 82M\nPhonon 31M\n"
                                         "Missing: Qwen3.5 4B · setup can download them", None)
    offer = ui.questions("Download the models now?")[0]
    assert offer.detail == "About 4.3 GB, saved in the Hugging Face cache. ai-phone needs them to handle calls."
    assert [(o.label, o.hint) for o in ui.menu("Download the models now?")[1]] == [
        ("Download now", "4.3 GB"), ("Later", "run: ai-phone models pull")]
    assert "Checking the download size…" in ui.labels
    downloading = ui.questions("Downloading the models")[0]
    assert downloading.detail == "You can leave this running. It picks up where it stopped if interrupted."
    assert all(callable(a[0]) for a in actions.args("pull_models")) and len(actions.args("pull_models")) == 2
    ok, later, failed_page = [p for p in ui.pages if p.step == CHECK and p.question in ("Everything works", "A few things need attention")]
    assert ok.checks[0] == Check("Models", "Parakeet 0.6B v2\nKokoro 82M\nQwen3.5 4B", True)
    assert later.checks[0] == Check("Models", "not downloaded yet · ai-phone models pull", None)
    assert failed_page.checks[0] == Check("Models", failed, False)
    assert [(o.label) for o in ui.menu("Everything works")[1]] == ["Continue", "Check again"]


def test_endpoint_brain_check_lines_and_voice_rows():
    ui = ScriptedUI([
        ("Give your AI", "Change call settings"),
        ("What should callers hear?", ["", "", ""]),
        ("A few things need attention", "Continue"),
        FINISH,
    ])
    lines = {"main": {"kind": "trunk", "number": "+15550100001"}, "office": {"kind": "trunk", "number": "+15550100002"}}
    actions = FakeActions(exists=True, lines=lines,
                          check_lines=[("main", "registered", ""), ("office", "degraded", "403 Forbidden")])
    actions.cur.brain_kind, actions.cur.base_url, actions.cur.model = "openai_compatible", "https://api.example.com/v1", "qwen"
    outcome = drive(ui, actions)
    assert "Checking your lines… (up to 20 seconds)" in ui.labels and "Playing your agent's voice…" in ui.labels
    rows = ui.questions("A few things need attention")[0].checks
    assert rows == [Check("Models", "Parakeet 0.6B v2\nKokoro 82M", True), Check("Config", "loads", True),
                    Check("Line main", "registered", True), Check("Line office", "degraded · 403 Forbidden", False),
                    Check("Voice", "Heart · played", True)]
    assert "Brain    https://api.example.com/v1 · qwen" in outcome.summary


def test_check_skips_lines_when_the_config_does_not_load():
    ui = ScriptedUI([
        ("Give your AI", "Change call settings"),
        ("What should callers hear?", ["", "", ""]),
        ("A few things need attention", "<quit>"),
    ])
    actions = FakeActions(exists=True, lines={"agent": SIP_LINE}, check_config="config.lines.agent.aor: expected sip:user@host",
                          play_sample=ActionError("could not play the sample: no audio device"))
    drive(ui, actions)
    rows = ui.questions("A few things need attention")[0].checks
    assert rows == [Check("Models", "Parakeet 0.6B v2\nKokoro 82M\nQwen3.5 4B", True), Check("Config", "config.lines.agent.aor: expected sip:user@host", False),
                    Check("Voice", "could not play the sample: no audio device", False)]
    assert not actions.args("check_lines")


def test_check_shows_a_skipped_row_without_the_sip_library():
    ui = ScriptedUI([
        ("Give your AI", "Change call settings"),
        ("What should callers hear?", ["", "", ""]),
        ("A few things need attention", "<quit>"),
    ])
    actions = FakeActions(exists=True, lines={"agent": SIP_LINE},
                          check_lines=[("lines", "skipped", "SIP library not built · run ./scripts/install.sh")])
    drive(ui, actions)
    assert ui.questions("A few things")[0].checks[2] == \
        Check("Lines", "skipped · SIP library not built · run ./scripts/install.sh", None)


def test_done_text_fits_the_page(monkeypatch, tmp_path):
    monkeypatch.delenv("AI_PHONE_HOME", raising=False)
    secrets = tmp_path / "secrets.env"   # this run saved a key, so the page names the file it is in
    secrets.write_text("")
    monkeypatch.setattr(flow.paths, "secrets_path", lambda: secrets)
    lines = {"agent": SIP_LINE, "twilio": NUMBER_LINE, "office": {"kind": "trunk", "number": "+15550100002"},
             "telnyx": {"kind": "trunk", "number": "+15550100003", "provider": {"name": "telnyx", "state": "pending"}},
             "desk": {"kind": "sip", "user": "desk", "aor": "sip:desk@192.0.2.10"},
             "odd": {"kind": "trunk", "number": "+15550100004", "provider": "twilio"}}
    text = flow.done_text(FakeActions(exists=True, lines=lines), flow.Session())
    assert str(Path.home()) not in text
    assert text.splitlines() == [
        "Saved in ~/.ai-phone/   config.yaml and secrets.env (keys and passwords, private to you)",
        "",
        "Lines    agent    sip:agent@192.0.2.10   SIP address",
        "         twilio   +15550100001           Twilio number",
        "         office   +15550100002           SIP trunk",
        "         telnyx   +15550100003           Telnyx number · pending",
        "         desk     sip:desk@192.0.2.10    SIP address",
        "         odd      +15550100004           SIP trunk",
        "Calls    receive and place calls",
        "Brain    on this computer · Qwen3.5 4B",
        "Voice    Kokoro · Heart · American · female · 1.0×",
        "",
        "Start ai-phone",
        "  ai-phone run",
        "  Leave ai-phone running while you want to receive calls.",
        "  If already running, stop and restart to apply these settings.",
        "",
        "Try it",
        "  call +15550100001 from any phone (your normal call rates)",
        "  pjsua --local-port 5070 sip:agent@127.0.0.1   talk through this computer's mic, free",
        "",
        "Give your AI the phone: it can run these from any folder",
        '  ai-phone call +15550100000 --prompt "…" --json   ai-phone\'s agent talks',
        "  ai-phone call +15550100000 --relay --json        your AI talks (say, listen)",
        "  ai-phone listen --json                           wait for a call",
        "  ai-phone hangup                                  end the call",
    ]
    assert max(len(ln) for ln in text.splitlines() if not ln.startswith("Give your AI the phone")) <= 88


@pytest.mark.parametrize("inbound,outbound,label", [
    (True, True, "receive and place calls"),
    (False, True, "place calls only"),
    (True, False, "receive calls only"),
])
def test_done_instructions_match_the_enabled_call_directions(inbound, outbound, label):
    actions = FakeActions(exists=True, lines={"agent": SIP_LINE, "twilio": {**NUMBER_LINE, "caller_id": "+15550100009"}},
                          allow_inbound=inbound, allow_outbound=outbound)
    text = flow.done_text(actions, flow.Session())
    assert f"Calls    {label}" in text
    assert ("ai-phone call " in text) is outbound
    assert ("ai-phone listen --json" in text) is inbound
    assert ("calls out as +15550100009" in text) is outbound
    assert ("Try it" in text) is inbound
    assert ("pjsua --local-port" in text) is inbound
    if inbound:
        assert "Leave ai-phone running while you want to receive calls." in text
    else:
        assert "Start ai-phone when you want to place a call; stop it with Ctrl-C afterward." in text
        assert "ai-phone listen --call <id> --json" in text
    assert "ai-phone hangup" in text


def test_the_done_page_names_secrets_env_only_when_there_is_one(monkeypatch, tmp_path):
    monkeypatch.setattr(flow.paths, "secrets_path", lambda: tmp_path / "secrets.env")
    first = flow.done_text(FakeActions(exists=True), flow.Session()).splitlines()[0]
    assert first.endswith("config.yaml") and "secrets.env" not in first
    (tmp_path / "secrets.env").write_text("TWILIO_AUTH_TOKEN=x\n")
    first = flow.done_text(FakeActions(exists=True), flow.Session()).splitlines()[0]
    assert first.endswith("config.yaml and secrets.env (keys and passwords, private to you)")


def test_a_sip_address_does_not_say_it_calls_out_as_itself(monkeypatch, tmp_path):
    monkeypatch.setattr(flow.paths, "secrets_path", lambda: tmp_path / "secrets.env")
    aor = "sip:agent@192.0.2.10"
    lines = {"agent": {**SIP_LINE, "caller_id": aor},
             "twilio": {**NUMBER_LINE, "caller_id": "+15550100009"}}
    rows = [ln for ln in flow.done_text(FakeActions(exists=True, lines=lines), flow.Session()).splitlines()
            if "SIP address" in ln or "Twilio number" in ln]
    assert rows[0].endswith("SIP address"), rows[0]
    assert rows[1].endswith("Twilio number · calls out as +15550100009")  # a verified number is worth saying


def test_done_text_with_one_kind_of_line():
    text = flow.done_text(FakeActions(exists=True), flow.Session())
    lines = text.splitlines()
    assert "Lines    none yet · run setup again to add one" in lines
    assert lines[lines.index("Try it") + 1] == "  add a line first: ai-phone setup"
    assert not any("pjsua" in ln for ln in lines)
    long_name = {"a" * 32: {"kind": "sip", "user": "a" * 32, "aor": f"sip:{'a' * 32}@192.0.2.10"}}
    lines = flow.done_text(FakeActions(exists=True, lines=long_name), flow.Session()).splitlines()
    tried = lines[lines.index("Try it") + 1:lines.index("Try it") + 3]
    assert tried == [f"  pjsua --local-port 5070 sip:{'a' * 32}@127.0.0.1", "    talk through this computer's mic, free"]
    pending = {"twilio": dict(NUMBER_LINE, provider={"name": "twilio", "state": "pending"})}
    lines = flow.done_text(FakeActions(exists=True, lines=pending), flow.Session()).splitlines()
    assert lines[lines.index("Try it") + 1] == "  nothing to call yet: the number is still being set up"


def test_a_hand_edited_provider_is_read_as_a_plain_line_not_a_crash():
    """`provider:` is a mapping in every config ai-phone writes, but the file is the person's to edit: whatever
    they leave there, the Done page still names the line."""
    for odd in ("twilio", ["twilio"], 7, None):
        assert flow.line_kind({"kind": "trunk", "number": "+15550100001", "provider": odd}) == "SIP trunk"
        assert flow.line_kind({"kind": "sip", "user": "agent", "provider": odd}) == "SIP address"
    assert flow.line_kind(NUMBER_LINE) == "Twilio number"  # the shape ai-phone writes still reads as the carrier
    assert flow.line_kind(dict(NUMBER_LINE, provider={"name": "twilio"})) == "Twilio number"  # state defaults to ready
    assert flow.line_kind({"kind": "trunk", "number": "+15550100001"}) == "SIP trunk"


def test_the_done_page_survives_a_hand_edited_provider():
    lines = {"odd": {"kind": "trunk", "number": "+15550100001", "provider": "twilio"}}
    rows = flow.done_text(FakeActions(exists=True, lines=lines), flow.Session()).splitlines()
    assert any(row.endswith("SIP trunk") for row in rows)
    # no provider mapping to read a state from is the shape of a trunk somebody wired up by hand: it is ready
    assert rows[rows.index("Try it") + 1] == "  call +15550100001 from any phone (your normal call rates)"


# ---- a Claude or ChatGPT subscription as the brain ----------------------------------------------

def test_new_setup_with_a_claude_subscription():
    ui = ScriptedUI([
        WELCOME_NEW,
        MODE_BOTH,
        ("Where should your agent think?", "My Claude subscription"),
        ("Which model?", "<default>"),
        ("Which connection should ai-phone use?", "A SIP address"),
        ("Name your SIP address", ["agent", "192.0.2.10"]),
        ("Which speech model?", "Kokoro"),
        ("How should your agent sound?", "American · female"),
        ("American · female", "<default>"),
        ("Heart", "Use this voice"),
        ("How fast should it talk?", [""]),
        ("What should callers hear?", ["", "", ""]),
        ("Everything works", "Continue"),
        FINISH,
    ])
    actions = FakeActions()
    outcome = drive(ui, actions)
    assert [o.label for o in ui.menu("Where should your agent think")[1]] == \
        ["On this computer", "My Claude subscription", "My ChatGPT subscription", "My own endpoint"]
    assert ui.menu("Where should your agent think")[1][1].hint == "uses the Claude Code login · no API key"
    assert actions.args("cli_installed") == [("claude",)]
    assert "Checking your Claude login…" in ui.labels
    assert "Asking which models your Claude plan allows…" not in ui.labels  # Claude Code always takes these three
    model = ui.menu("Which model?")
    assert [o.label for o in model[1]] == ["Haiku", "Sonnet", "Opus"] and model[2] == 0
    assert model[1][0].hint == "fastest to answer · the one for a phone call"
    assert "The words of every call go to Claude through Claude Code" in ui.questions("Which model?")[0].detail
    assert actions.changes() == ["save_call_directions", "save_brain_subscription", "add_sip_line", "save_voice", "save_callers"]
    assert actions.args("save_brain_subscription") == [("claude", "haiku")]
    # one request of the plan for the whole run: the model that was picked, which the Check page then finds proved
    assert actions.args("test_subscription") == [("claude", "haiku"), ("claude", None)]
    assert actions.args("login_status") == [("claude",)]  # the login itself costs nothing
    assert "Asking Claude to answer once with Haiku…" in ui.labels
    assert Check("Brain", "Claude (Haiku) · logged in", True) in ui.questions("Everything works")[0].checks
    assert "Brain    your Claude subscription · Haiku" in outcome.summary
    assert "nothing leaves this computer" not in outcome.summary


def test_a_chatgpt_plan_is_asked_which_models_it_allows():
    ui = ScriptedUI([
        ("Give your AI", "Change the brain"),
        ("Where should your agent think?", "My ChatGPT subscription"),
        ("Which model?", "<default>"),
        ("A few things need attention", "Continue"),
        FINISH,
    ])
    actions = FakeActions(exists=True)
    outcome = drive(ui, actions)
    assert actions.args("brain_models") == [("chatgpt",)]
    assert "Asking which models your ChatGPT plan allows…" in ui.labels
    model = ui.menu("Which model?")
    assert [o.label for o in model[1]] == ["gpt-5.6-luna", "gpt-6-astra"] and model[2] == 0  # fastest measured first
    assert actions.args("save_brain_subscription") == [("chatgpt", "gpt-5.6-luna")]
    assert Check("Brain", "ChatGPT (gpt-5.6-luna) · logged in", True) in ui.questions("A few things need attention")[0].checks
    assert "Brain    your ChatGPT subscription · gpt-5.6-luna" in outcome.summary


def test_the_brain_the_config_already_names_is_the_default_everywhere():
    ui = ScriptedUI([
        ("Give your AI", "Change the brain"),
        ("Where should your agent think?", "<default>"),
        ("Which model?", "<default>"),
        ("A few things need attention", "Continue"),
        FINISH,
    ])
    actions = FakeActions(exists=True)
    actions.cur.brain_kind, actions.cur.model = "chatgpt", "gpt-6-astra"
    drive(ui, actions)
    assert ui.menus[1][2] == 2  # My ChatGPT subscription
    assert ui.menu("Which model?")[2] == 1  # the model it is already using
    assert actions.args("save_brain_subscription") == [("chatgpt", "gpt-6-astra")]


def test_a_cli_that_is_not_installed_says_how_to_install_it():
    ui = ScriptedUI([
        ("Give your AI", "Change the brain"),
        ("Where should your agent think?", "My ChatGPT subscription"),
        ("Codex is not installed", "Check again"),
        ("Which model?", "<default>"),
        ("A few things need attention", "Continue"),
        FINISH,
    ])
    actions = FakeActions(exists=True, cli_installed=Seq(False, True))
    drive(ui, actions)
    page = ui.questions("Codex is not installed")[0]
    assert page.body == "npm install -g @openai/codex\ncodex login"
    assert page.detail == "ai-phone reaches your ChatGPT plan through it. Install it, log in, then check again."
    assert actions.args("login_status") == [("chatgpt",)] and not page.error  # only once the CLI is there
    assert actions.args("test_subscription") == [("chatgpt", "gpt-5.6-luna"), ("chatgpt", None)]


def test_a_cli_that_is_not_logged_in_shows_what_to_run():
    told = "claude is installed but not logged in. Log in with: claude, then type /login"
    ui = ScriptedUI([
        ("Give your AI", "Change the brain"),
        ("Where should your agent think?", "My Claude subscription"),
        ("cannot use your Claude plan yet", "Check again"),
        ("Which model?", "<default>"),
        ("A few things need attention", "Continue"),
        FINISH,
    ])
    actions = FakeActions(exists=True, login_status=Seq(ActionError(told), None))
    drive(ui, actions)
    page = ui.questions("cannot use your Claude plan yet")[0]
    assert page.error == told and page.body == "npm install -g @anthropic-ai/claude-code\nclaude, then type /login"
    assert actions.args("save_brain_subscription") == [("claude", "haiku")]
    # the login was established twice without inference; the plan answered once, for the model that was picked
    assert actions.args("login_status") == [("claude",)] * 2
    assert actions.args("test_subscription") == [("claude", "haiku"), ("claude", None)]


def test_back_from_the_login_page_goes_back_to_the_brain_menu():
    ui = ScriptedUI([
        ("Give your AI", "Change the brain"),
        ("Where should your agent think?", "My Claude subscription"),
        ("Claude Code is not installed", "<back>"),
        ("Where should your agent think?", "On this computer"),
        ("Which model?", "<default>"),
        ("A few things need attention", "Continue"),
        FINISH,
    ])
    actions = FakeActions(exists=True, cli_installed=False)
    drive(ui, actions)
    assert actions.changes() == ["save_brain_local"]
    assert len(ui.questions("Where should your agent think?")) == 2


def test_a_subscription_brain_that_cannot_be_saved_is_asked_again():
    broken = "config.lines.agent.aor: expected sip:user@host[:port]"
    ui = ScriptedUI([
        ("Give your AI", "Change the brain"),
        ("Where should your agent think?", "My Claude subscription"),
        ("Which model?", "<default>"),
        ("Which model?", "Sonnet"),
        ("A few things need attention", "<quit>"),
    ])
    actions = FakeActions(exists=True, save_brain_subscription=Seq(ActionError(broken), None), check_config=broken)
    assert drive(ui, actions).code == 1
    assert ui.questions("Which model?")[1].error == broken and ui.menu("Which model?")[2] == 0
    assert actions.args("save_brain_subscription") == [("claude", "haiku"), ("claude", "sonnet")]


def test_choosing_a_different_model_still_spends_one_request():
    """The probe used to run before the menu, so a person who picked anything but the default paid again at the Check
    step: "one request per setup" was not true of the ordinary path. The request is now the chosen model's."""
    ui = ScriptedUI([
        ("Give your AI", "Change the brain"),
        ("Where should your agent think?", "My ChatGPT subscription"),
        ("Which model?", "gpt-6-astra"),          # not the default the menu offers
        ("A few things need attention", "Continue"),
        FINISH,
    ])
    actions = FakeActions(exists=True)
    drive(ui, actions)
    assert actions.args("save_brain_subscription") == [("chatgpt", "gpt-6-astra")]
    # once, for the model that was picked. The Check step asks for the model the config now names, and finds it proved.
    assert actions.args("test_subscription") == [("chatgpt", "gpt-6-astra"), ("chatgpt", None)]
    assert actions.args("login_status") == [("chatgpt",)]
    assert "Asking ChatGPT to answer once with gpt-6-astra…" in ui.labels
    assert "Picking one sends a single short request to your plan" in ui.questions("Which model?")[0].detail


def test_a_model_the_plan_does_not_allow_leaves_the_others_on_the_menu():
    """The saved or default model is one this plan refuses. The probe used to run before the menu, so the page said
    "choose another model" and the only way on was the same failing probe. The menu is now where the answer is."""
    told = "Your plan does not allow that model. Choose another one."
    ui = ScriptedUI([
        ("Give your AI", "Change the brain"),
        ("Where should your agent think?", "My ChatGPT subscription"),
        ("Which model?", "<default>"),           # the saved one: refused
        ("Which model?", "gpt-6-astra"),         # the one that works, straight from the same menu
        ("A few things need attention", "Continue"),
        FINISH,
    ])
    actions = FakeActions(exists=True, test_subscription=Seq(ActionError(told), None, None))
    actions.cur.brain_kind, actions.cur.model = "chatgpt", "gpt-5.6-luna"
    drive(ui, actions)
    second = ui.questions("Which model?")[1]
    assert second.error == told, "the sentence is on the menu that offers the others"
    again = [m for m in ui.menus if "Which model?" in m[0]][1]
    assert again[2] == 0, "the refused model is still the one the cursor is on"
    assert [o.label for o in again[1]] == ["gpt-5.6-luna", "gpt-6-astra"], "every model the plan lists is still offered"
    assert actions.args("save_brain_subscription") == [("chatgpt", "gpt-6-astra")]
    assert actions.args("test_subscription")[:2] == [("chatgpt", "gpt-5.6-luna"), ("chatgpt", "gpt-6-astra")]


def test_a_cli_logged_in_with_an_api_key_says_what_to_do_about_it():
    told = ("codex is set up to use an API key, which is billed per request. ai-phone runs your subscription instead: "
            "remove the key, then log in with: codex login")
    ui = ScriptedUI([
        ("Give your AI", "Change the brain"),
        ("Where should your agent think?", "My ChatGPT subscription"),
        ("cannot use your ChatGPT plan yet", "Check again"),
        ("Which model?", "<default>"),
        ("A few things need attention", "Continue"),
        FINISH,
    ])
    actions = FakeActions(exists=True, login_status=Seq(ActionError(told), None))
    drive(ui, actions)
    assert ui.questions("cannot use your ChatGPT plan yet")[0].error == told
    assert actions.args("test_subscription")[0] == ("chatgpt", "gpt-5.6-luna"), "no turn until the login is a plan"


def test_a_login_that_stops_working_is_a_failed_check_row():
    ui = ScriptedUI([
        ("Give your AI", "Change call settings"),
        ("What should callers hear?", ["", "", ""]),
        ("A few things need attention", "Continue"),
        FINISH,
    ])
    told = "claude is installed but not logged in. Log in with: claude, then type /login"
    actions = FakeActions(exists=True, test_subscription=ActionError(told))
    actions.cur.brain_kind, actions.cur.model = "claude", "sonnet"
    outcome = drive(ui, actions)
    assert Check("Brain", told, False) in ui.questions("A few things need attention")[0].checks
    assert "Brain    your Claude subscription · Sonnet" in outcome.summary


def test_a_subscription_brain_needs_no_model_download():
    """models.missing() already leaves the brain's weights out when the brain is not on this computer; the Check page then
    has nothing to download and says so once the speech models are there."""
    ui = ScriptedUI([
        ("Give your AI", "Change the brain"),
        ("Where should your agent think?", "My Claude subscription"),
        ("Which model?", "Opus"),
        ("A few things need attention", "Continue"),
        FINISH,
    ])
    actions = FakeActions(exists=True)
    drive(ui, actions)
    rows = ui.questions("A few things need attention")[0].checks
    assert rows[0] == Check("Models", "Parakeet 0.6B v2\nKokoro 82M", True) and not actions.args("pull_models")
    assert Check("Brain", "Claude (Opus) · logged in", True) in rows


SIGNALWIRE = {"SIGNALWIRE_PROJECT_ID": "project-test", "SIGNALWIRE_API_TOKEN": "token-SECRET",
              "SIGNALWIRE_SPACE": "example.signalwire.com"}


def test_signalwire_setup_maps_all_credentials_and_only_attaches_existing_number():
    actions = FakeActions()
    ui = ScriptedUI([
        ("Which carrier?", "SignalWire"), ("How should SignalWire connect?", "Connect with API"),
        ("Your SignalWire credentials", ["project-test", "token-SECRET", "example.signalwire.com"]),
        ("Which number?", OWNED[0].e164), ("Connect", f"Connect {OWNED[0].e164}"),
        ("Also forward", "No"), ("Show your own number", "No"),
    ])
    session = flow.Session()
    flow.phone_number(ui, actions, session)
    assert not ui.answers
    assert actions.args("owned_numbers") == [("signalwire", SIGNALWIRE)]
    assert actions.args("attach_number") == [("signalwire", SIGNALWIRE, OWNED[0])]
    assert not actions.args("buy_number") and not actions.args("search_numbers")
    assert [(f.label, f.secret) for f in ui.fields] == [("Project ID", True), ("API token", True), ("Space host", False)]
    assert all("token-SECRET" not in page_text(page) for page in ui.pages)


def test_signalwire_without_numbers_explains_portal_purchase_without_searching():
    actions = FakeActions(saved_keys=SIGNALWIRE, owned_numbers=[])
    ui = ScriptedUI([
        ("Which carrier?", "SignalWire"), ("How should SignalWire connect?", "Connect with API"),
        ("Use the saved SignalWire key?", "Use the saved key"),
        ("No unconnected numbers", "Continue"),
    ])
    flow.phone_number(ui, actions, flow.Session())
    assert not ui.answers and not actions.changes()
    assert not actions.args("search_numbers")
    assert "SignalWire dashboard" in page_text(ui.pages[-1])


def test_signalwire_manual_setup_needs_no_api_credentials_and_sets_account_identity():
    actions = FakeActions()
    ui = ScriptedUI([
        ("Which carrier?", "SignalWire"), ("How should SignalWire connect?", "Use SIP credentials"),
        ("Your SignalWire SIP trunk", ["example.sip.signalwire.com:5061", "account1", "sip-SECRET", ""]),
        ("Which transport?", "<default>"), ("Media encryption", "Required"),
    ])
    flow.phone_number(ui, actions, flow.Session())
    assert not ui.answers
    assert actions.args("add_trunk_line") == [("sip:example.sip.signalwire.com:5061", "account1", "sip-SECRET", "",
                                               "tls", "account", "signalwire", "required")]
    assert not actions.args("saved_keys") and not actions.args("owned_numbers")
    summary = flow.done_text(actions, flow.Session())
    assert "sip:account1@example.sip.signalwire.com:5061" in summary
    assert "number is still being set up" not in summary
    assert "call the number routed to your SIP account" in summary


def test_signalwire_caller_id_shortcut_collects_credentials_even_for_own_number():
    line = {**caller_id_line(carrier="signalwire"), "from_user": "account", "user": "account1"}
    actions = FakeActions(exists=True, lines={"main": line}, saved_keys=SIGNALWIRE,
                          owned_numbers=[Number(line["number"], None, None, False, "number-id")])
    ui = ScriptedUI([
        ("Which line's caller ID?", f"main · {line['number']}"),
        ("Which number should people see?", f"Line's number · {line['number']}"),
        ("Use the saved SignalWire key?", "Use the saved key"),
    ])
    session = flow.Session()
    flow.change_caller_id(ui, actions, session)
    assert not ui.answers
    assert actions.args("use_caller_id") == [("main", line["number"], SIGNALWIRE)]
    assert actions.args("owned_numbers") == [("signalwire", SIGNALWIRE)]
    assert "Updating the caller ID" in [p.question for p in ui.pages]


def test_signalwire_verified_caller_id_applies_with_same_credentials_and_line_context():
    line = {**caller_id_line(carrier="signalwire"), "from_user": "account"}
    actions = FakeActions(exists=True, lines={"main": line})
    ui = ScriptedUI([("Your mobile number", ["+15555550102"]), ("Enter the code", ["123456"])])
    flow.verify_caller_id(ui, actions, flow.Session(), "signalwire", "SignalWire", SIGNALWIRE, "main", line["number"])
    assert actions.args("start_caller_id") == [("signalwire", SIGNALWIRE, "+15555550102", "main")]
    assert actions.args("record_caller_id") == [("main", "+15555550102", "signalwire", SIGNALWIRE)]


def test_numberless_manual_account_trunk_has_a_useful_summary_and_test_instruction():
    line = {"kind": "trunk", "from_user": "account", "user": "account1", "registrar": "sip:sip.example.com"}
    summary = flow.done_text(FakeActions(lines={"trunk": line}), flow.Session())
    assert "sip:account1@sip.example.com" in summary
    assert "still being set up" not in summary
    assert "caller ID set by your provider" in summary


def test_signalwire_pending_number_is_offered_as_resume_and_uses_existing_attachment_path():
    actions = FakeActions(saved_keys=SIGNALWIRE, owned_numbers=OWNED[:1],
                          unfinished_numbers={OWNED[0].e164}, resumable_numbers={OWNED[0].e164: "retry"},
                          attach_number=LineResult("retry", 0, ["Line retry is ready."]))
    ui = ScriptedUI([
        ("Which carrier?", "SignalWire"), ("How should SignalWire connect?", "Connect with API"),
        ("Use the saved SignalWire key?", "Use the saved key"),
        ("Which number?", f"Resume {OWNED[0].e164}"), ("Resume setup", f"Resume {OWNED[0].e164}"),
        ("Also forward", "No"), ("Show your own number", "No"),
    ])
    flow.phone_number(ui, actions, flow.Session())
    assert not ui.answers
    option = ui.menu("Which number?")[1][0]
    assert option.label == f"Resume {OWNED[0].e164}" and option.hint == "unfinished line retry"
    assert actions.args("attach_number") == [("signalwire", SIGNALWIRE, OWNED[0])]
    assert actions.args("resumable_numbers") == [("signalwire",)]
    assert not actions.args("buy_number") and not actions.args("search_numbers")
    assert f"Resuming {OWNED[0].e164}…" in ui.labels
    assert "saved SIP credentials" in ui.questions("Resume setup")[0].detail


def test_signalwire_ineligible_unfinished_number_shows_recovery_instead_of_buying():
    actions = FakeActions(saved_keys=SIGNALWIRE, owned_numbers=OWNED[:1], unfinished_numbers={OWNED[0].e164})
    ui = ScriptedUI([
        ("Which carrier?", "SignalWire"), ("How should SignalWire connect?", "Connect with API"),
        ("Use the saved SignalWire key?", "Use the saved key"), ("An earlier setup needs attention", "Continue"),
    ])
    flow.phone_number(ui, actions, flow.Session())
    page = ui.questions("An earlier setup needs attention")[0]
    text = page_text(page)
    assert "Buy a" not in text and "lines list" in text
    assert "lines resume <line> --key-env SIGNALWIRE_PROJECT_ID --secret-env SIGNALWIRE_API_TOKEN" in text
    assert "lines remove <line> --key-env SIGNALWIRE_PROJECT_ID --secret-env SIGNALWIRE_API_TOKEN" in text
    assert "SIGNALWIRE_SPACE" in text and "original carrier account" in text
    assert not actions.changes() and not ui.answers


def test_signalwire_pending_number_does_not_expose_other_ineligible_numbers():
    actions = FakeActions(saved_keys=SIGNALWIRE, unfinished_numbers={n.e164 for n in OWNED},
                          resumable_numbers={OWNED[1].e164: "retry"})
    ui = ScriptedUI([
        ("Which carrier?", "SignalWire"), ("How should SignalWire connect?", "Connect with API"),
        ("Use the saved SignalWire key?", "Use the saved key"), ("Which number?", "<quit>"),
    ])
    with pytest.raises(Cancel):
        flow.phone_number(ui, actions, flow.Session())
    assert [o.label for o in ui.menu("Which number?")[1]] == [f"Resume {OWNED[1].e164}"]
    assert not actions.changes()


def test_signalwire_already_configured_numbers_do_not_prompt_another_purchase():
    actions = FakeActions(saved_keys=SIGNALWIRE, owned_numbers=OWNED[:1], lines={"main": {"number": OWNED[0].e164}})
    ui = ScriptedUI([
        ("Which carrier?", "SignalWire"), ("How should SignalWire connect?", "Connect with API"),
        ("Use the saved SignalWire key?", "Use the saved key"), ("No unconnected numbers", "Continue"),
    ])
    flow.phone_number(ui, actions, flow.Session())
    text = page_text(ui.questions("No unconnected numbers")[0])
    assert "already configured" in text and "Buy a" not in text and "lines list" in text
    assert not actions.changes()

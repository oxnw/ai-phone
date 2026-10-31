"""Plain numbered prompts: what is printed, how answers are read, and that a secret is never echoed."""
from __future__ import annotations

import sys
import types

import pytest

from ai_phone.setup import flow, run_setup
from ai_phone.setup.plain import BLANK_LIMIT, NO_ANSWER, PlainUI
from ai_phone.setup.ui import Back, Cancel, Check, Field, Option, Outcome, Page
from tests.unit.test_setup_flow import FakeActions, ScriptedUI, caller_id_line

LINE_PAGE = Page(3, "Which connection should ai-phone use?", "You can add more lines later by running setup again.",
                 checks=[Check("Twilio key", "works · 2 numbers on this account", True)])
LINE_OPTIONS = [Option("A phone number", "Twilio or Telnyx · a new number, or one already on that account"),
                Option("A SIP address", "free · softphones and other agents")]


@pytest.fixture
def typed(monkeypatch):
    """Scripted answers for input() and getpass.getpass(); the prompts are recorded. input() echoes its prompt like
    the real one does; getpass echoes nothing of what is typed."""
    answers: list = []
    prompts: list[str] = []

    def take(prompt, echo):
        prompts.append(prompt)
        if echo:
            print(prompt, end="")
        assert answers, f"no answer left for {prompt!r}"
        value = answers.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr("builtins.input", lambda prompt="": take(prompt, True))
    monkeypatch.setattr("getpass.getpass", lambda prompt="Password: ", stream=None: take(prompt, False))
    return types.SimpleNamespace(answers=answers, prompts=prompts)


def test_choose_prints_the_page_and_enter_takes_the_default(typed, capsys):
    typed.answers.append("")
    assert PlainUI().choose(LINE_PAGE, LINE_OPTIONS, default=1) == 1
    assert capsys.readouterr().out.splitlines() == [
        "",
        "S T E P  4  O F  8  ·  L I N E",
        "Which connection should ai-phone use?",
        "You can add more lines later by running setup again.",
        "  ✓ Twilio key   works · 2 numbers on this account",
        "  1) A phone number   Twilio or Telnyx · a new number, or one already on that account",
        "  2) A SIP address    free · softphones and other agents",
        "  b) Back",
        "Choose [2]: ",
    ]


def test_welcome_page_has_a_title_note_and_no_back(typed, capsys):
    typed.answers += ["b", "2"]
    page = Page(0, "Give your AI a phone number", "Calls answered and placed by models running on this computer.",
                note="Picks a line, a brain and a voice, then checks it all works.",
                checks=[Check("SIP library", "not built yet · run ./scripts/install.sh", False),
                        Check("Memory", "8 GB · 4B model recommended", None),
                        Check("Models", "Parakeet 0.6B v2\nPhonon 31M", True)])
    assert PlainUI().choose(page, [Option("Continue"), Option("Cancel", "exit setup, nothing is saved")], back=False) == 1
    out = capsys.readouterr().out.splitlines()
    assert out[:11] == [
        "",
        "ai-phone setup",
        "Give your AI a phone number",
        "Calls answered and placed by models running on this computer.",
        "Picks a line, a brain and a voice, then checks it all works.",
        "  ✗ SIP library   not built yet · run ./scripts/install.sh",
        "  ○ Memory        8 GB · 4B model recommended",
        "  ✓ Models        Parakeet 0.6B v2",
        "                  Phonon 31M",
        "  1) Continue",
        "  2) Cancel     exit setup, nothing is saved",
    ]
    assert "  b) Back" not in out
    assert out[-2] == "Choose [1]: Answer with a number from the list or q to quit."  # piped input echoes no newline


def test_b_goes_back_and_knows_whether_it_is_the_first_question(typed):
    ui = PlainUI()
    ui.begin(3)
    typed.answers.append("b")
    with pytest.raises(Back) as first:
        ui.choose(LINE_PAGE, LINE_OPTIONS)
    assert first.value.at_start
    ui.begin(3)
    typed.answers += ["1", "B"]
    ui.choose(LINE_PAGE, LINE_OPTIONS)
    with pytest.raises(Back) as later:
        ui.choose(LINE_PAGE, LINE_OPTIONS)
    assert not later.value.at_start


def test_enter_on_a_menu_whose_default_is_back_goes_back(typed, capsys):
    typed.answers += ["", "1"]
    ui = PlainUI()
    confirm = Page(3, "Buy +15550100001 for $1.15/month?")
    with pytest.raises(Back):
        ui.choose(confirm, [Option("Buy +15550100001", "$1.15/month")], default=1)
    assert ui.choose(confirm, [Option("Buy +15550100001", "$1.15/month")], default=1) == 0
    assert typed.prompts == ["Choose [b]: "] * 2
    assert capsys.readouterr().out.count("  b) Back\n") == 2


def test_q_quits(typed):
    typed.answers.append("q")
    with pytest.raises(Cancel):
        PlainUI().choose(LINE_PAGE, LINE_OPTIONS)


@pytest.mark.parametrize("stop", [EOFError(), KeyboardInterrupt()])
def test_end_of_input_or_ctrl_c_quits(typed, stop):
    typed.answers.append(stop)
    with pytest.raises(Cancel):
        PlainUI().choose(LINE_PAGE, LINE_OPTIONS)
    typed.answers.append(stop)
    with pytest.raises(Cancel):
        PlainUI().ask(LINE_PAGE, [Field("Name")])
    typed.answers.append(stop)
    with pytest.raises(Cancel):
        PlainUI().ask(LINE_PAGE, [Field("API key", secret=True)])


def test_an_answer_that_is_not_on_the_list_is_asked_again(typed, capsys):
    typed.answers += ["9", "x", "0", " 2 "]
    assert PlainUI().choose(LINE_PAGE, LINE_OPTIONS) == 1
    out = capsys.readouterr().out
    assert out.count("Answer with a number from the list, b to go back or q to quit.") == 3
    assert typed.prompts == ["Choose [1]: "] * 4


def test_ask_takes_defaults_and_asks_again_until_valid(typed, capsys):
    typed.answers += ["", "Agent", "desk", "192.0.2.10", "5"]
    must_be_lower = lambda v: None if v.islower() else "Use lowercase letters, digits and dashes."  # noqa: E731
    fields = [Field("Name", "agent", validate=must_be_lower), Field("Desk", validate=must_be_lower),
              Field("This computer's address", "", placeholder="192.0.2.1"), Field("Outbound calls per day", "20")]
    assert PlainUI().ask(Page(3, "Name your SIP address"), fields) == ["agent", "desk", "192.0.2.10", "5"]
    assert typed.prompts == ["Name [agent]: ", "Desk: ", "Desk: ", "This computer's address (192.0.2.1): ",
                             "Outbound calls per day [20]: "]
    assert capsys.readouterr().out.count("! Use lowercase letters, digits and dashes.\n") == 1


def test_endless_blank_input_gives_up_instead_of_asking_for_ever(typed, capsys):
    """A piped `yes ''` never ends and never answers: without a limit the required field is re-asked for ever."""
    typed.answers += [""] * 20
    required = lambda v: None if v else "Enter a value."  # noqa: E731
    with pytest.raises(Cancel, match="No answer after ten tries"):
        PlainUI().ask(LINE_PAGE, [Field("Name", validate=required)])
    assert len(typed.prompts) == BLANK_LIMIT
    out = capsys.readouterr().out
    assert out.count("! Enter a value.") == BLANK_LIMIT - 1  # the tenth blank gives up instead of complaining again
    assert NO_ANSWER in out


def test_the_blank_count_starts_again_for_each_field_and_after_an_answer(typed, capsys):
    """Nine blanks then a real answer is not a stuck prompt, and the next field starts its own count."""
    required = lambda v: None if v else "Enter a value."  # noqa: E731
    typed.answers += [""] * 9 + ["agent"] + [""] * 9 + ["desk"]
    fields = [Field("Name", validate=required), Field("Desk", validate=required)]
    assert PlainUI().ask(LINE_PAGE, fields) == ["agent", "desk"]
    assert NO_ANSWER not in capsys.readouterr().out


def test_a_blank_answer_a_field_accepts_is_not_counted_against_it(typed):
    """An optional field takes the first empty answer, so it never reaches the limit."""
    typed.answers += ["", ""]
    assert PlainUI().ask(LINE_PAGE, [Field("Caller id"), Field("Note")]) == ["", ""]


def test_a_secret_is_read_hidden_and_never_printed(typed, capsys):
    typed.answers += ["", "sk-live-SECRET", "  pw SECRET  "]
    required = lambda v: None if v.strip() else "Enter a value."  # noqa: E731
    page = Page(2, "Your endpoint", "Any server that speaks the OpenAI chat API.")
    ui = PlainUI()
    assert ui.ask(page, [Field("API key", secret=True, validate=required)]) == ["sk-live-SECRET"]
    # a secret is kept exactly as typed: a password may start or end with a space
    assert ui.ask(page, [Field("API key", secret=True, placeholder="leave empty for none")]) == ["  pw SECRET  "]
    out = capsys.readouterr()
    assert "SECRET" not in out.out and "SECRET" not in out.err
    assert typed.prompts == ["API key: ", "API key: ", "API key (leave empty for none): "]


def test_errors_and_bodies_are_printed(typed, capsys):
    typed.answers.append("")
    page = Page(3, "Dial this on your mobile", "Open the phone app, type the code and press call.",
                error="could not reach 127.0.0.1: ConnectError", body="Turn on     **67*+15550100001#\nCheck       *#67#")
    PlainUI().choose(page, [Option("Done")], back=False)
    out = capsys.readouterr().out.splitlines()
    assert out[3:7] == ["Open the phone app, type the code and press call.",
                        "! could not reach 127.0.0.1: ConnectError",
                        "  Turn on     **67*+15550100001#",
                        "  Check       *#67#"]


def test_busy_prints_the_label_and_returns_the_result(capsys):
    ui = PlainUI()
    assert ui.busy(LINE_PAGE, "Checking your Twilio account…", lambda: 42) == 42
    assert capsys.readouterr().out == "Checking your Twilio account…\n"
    with pytest.raises(RuntimeError):
        ui.busy(LINE_PAGE, "Playing…", lambda: (_ for _ in ()).throw(RuntimeError("boom")))


def test_progress_prints_at_most_once_per_five_percent(capsys):
    def work(report):
        for done in (0, 1, 2, 3, 5, 9, 10, 42, 43, 44, 100, 100):
            report(done, 100)
        report(5, 0)  # an unknown total prints nothing
        return "ok"

    assert PlainUI().progress(LINE_PAGE, "Downloading…", work) == "ok"
    assert capsys.readouterr().out.splitlines() == ["Downloading…", "  0%", "  5%", "  10%", "  42%", "  100%"]


def sip_setup_answers():
    return [
        "",   # Welcome: Continue
        "",   # Mode: Receive and place calls
        "",   # Brain: On this computer
        "",   # model: Qwen3.5 4B
        "2",  # Line: A SIP address
        "", "",  # name and address: defaults
        "2",  # speech model: Kokoro
        "", "", "",  # voice group, voice, Use this voice
        "",   # speed
        "", "", "",  # greeting, instructions, daily limit
        "",   # Check: Continue
        "",   # Done: Finish
    ]


def test_run_setup_plain_end_to_end(typed, capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("AI_PHONE_HOME", str(tmp_path))
    made = []
    monkeypatch.setattr("ai_phone.setup.actions.Actions", lambda path: made.append(path) or made_actions)
    made_actions = FakeActions()
    typed.answers += sip_setup_answers()
    assert run_setup(plain=True) == 0
    assert made == [tmp_path / "config.yaml"]
    assert made_actions.changes() == ["save_call_directions", "save_brain_local", "add_sip_line", "save_voice", "save_callers"]
    out = capsys.readouterr().out
    # Count the summary heading, since earlier guidance also mentions the run command.
    assert out.count("Start ai-phone\n") == 1  # the Done page body is printed once
    assert "  Start ai-phone\n    ai-phone run\n" in out
    assert "  pjsua --local-port 5070 sip:agent@127.0.0.1" in out
    assert out.splitlines()[-2:] == ["  1) Finish", "Choose [1]: "]
    assert not typed.answers


def test_plain_setup_can_change_an_existing_lines_caller_id(typed, capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("AI_PHONE_HOME", str(tmp_path))
    actions = FakeActions(exists=True, lines={"main": caller_id_line()})
    monkeypatch.setattr("ai_phone.setup.actions.Actions", lambda path: actions)
    # Change caller ID, main, use the line's own number, Check, Finish.
    typed.answers += ["6", "1", "1", "", ""]
    assert run_setup(plain=True) == 0
    assert actions.args("use_caller_id") == [("main", "+15555550100")]
    assert not actions.args("saved_keys") and not actions.args("start_caller_id")
    out = capsys.readouterr().out
    assert "6) Change caller ID" in out and "Which line's caller ID?" in out
    assert "1) Line's number · +15555550100" in out and "2) Your verified number · +15555550101" in out
    assert "S T E P  6  O F  8  ·  S E T T I N G S" in out
    assert "restart" in out.lower() and not typed.answers


class EndOfInput:
    """Blank lines (Enter), then the end of the input."""

    def __init__(self, answers):
        self.answers = list(answers)

    def __call__(self, prompt=""):
        print(prompt, end="")
        if not self.answers:
            raise EOFError
        return self.answers.pop(0)


def test_pressing_enter_never_buys_or_connects_a_number(capsys, monkeypatch):
    """Plain mode reads piped input: blank lines are no answer, so they must never reach a purchase."""
    monkeypatch.setattr("getpass.getpass", lambda prompt="": pytest.fail("the saved key is used"))
    saved = {"TWILIO_ACCOUNT_SID": "ACsid", "TWILIO_AUTH_TOKEN": "tok"}
    actions = FakeActions(exists=True, saved_keys=saved)
    monkeypatch.setattr("builtins.input", EndOfInput([""] * 60))
    assert flow.run(PlainUI(), actions).code == 1
    assert len(actions.args("search_numbers")) >= 3  # it went round the purchase questions several times
    assert not actions.args("buy_number") and not actions.args("attach_number")
    assert "Buy +15550100001 for $1.15/month?" in capsys.readouterr().out

    actions = FakeActions(exists=True, saved_keys=saved)
    # Add a line, A phone number, Twilio, Use the saved key, one already on the account, the first number, then Enter
    monkeypatch.setattr("builtins.input", EndOfInput(["", "", "", "", "2", ""] + [""] * 30))
    assert flow.run(PlainUI(), actions).code == 1
    assert "Connect +15550100007 to ai-phone?" in capsys.readouterr().out
    assert not actions.args("buy_number") and not actions.args("attach_number")


def test_run_setup_without_a_terminal_uses_plain_prompts(typed, capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("AI_PHONE_HOME", str(tmp_path))
    monkeypatch.setattr("ai_phone.setup.actions.Actions", lambda path: FakeActions())
    monkeypatch.setattr("ai_phone.setup.sys", types.SimpleNamespace(stdin=Terminal(False), stdout=Terminal(True)))
    typed.answers.append("q")
    assert run_setup() == 1
    out = capsys.readouterr().out
    assert "ai-phone setup" in out
    assert out.endswith("Setup cancelled. Nothing was changed.\n")


class Terminal:
    def __init__(self, tty):
        self.tty = tty

    def isatty(self):
        return self.tty


def test_run_setup_in_a_terminal_runs_the_full_screen_app(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("AI_PHONE_HOME", str(tmp_path))
    actions = FakeActions()
    monkeypatch.setattr("ai_phone.setup.actions.Actions", lambda path: actions)
    monkeypatch.setattr("ai_phone.setup.sys", types.SimpleNamespace(stdin=Terminal(True), stdout=Terminal(True)))
    seen = {}

    def run_tui(flow_fn, summary=None):
        seen["flow"] = flow_fn(ScriptedUI([("Give your AI", "Continue"), ("How should ai-phone work?", "Receive and place calls"),
                                           ("Where should your agent think?", "On this computer"),
                                           ("Which model?", "<default>"), ("Which connection should ai-phone use", "<quit>")]))
        seen["summary"] = summary()  # what a run the app had to leave early says: the same session
        return Outcome(0, "Saved\n  summary")

    monkeypatch.setitem(sys.modules, "ai_phone.setup.tui", types.SimpleNamespace(run_tui=run_tui))
    assert run_setup() == 0
    assert seen["flow"].code == 1 and actions.changes() == ["save_call_directions", "save_brain_local"]
    assert seen["summary"] == "Setup cancelled. Saved in this run: call directions, the brain. Nothing else was changed."
    assert capsys.readouterr().out == "Saved\n  summary\n"


def test_ctrl_c_in_plain_mode_ends_with_the_summary(typed, capsys, monkeypatch, tmp_path):
    """ctrl+c outside a prompt (here, while a line is saved) never ends in a traceback."""
    monkeypatch.setenv("AI_PHONE_HOME", str(tmp_path))
    monkeypatch.setattr("ai_phone.setup.actions.Actions", lambda path: actions)
    actions = FakeActions(add_sip_line=KeyboardInterrupt())
    typed.answers += ["", "", "", "", "2", "", ""]
    assert run_setup(plain=True) == 1
    assert capsys.readouterr().out.endswith("\nSetup cancelled. Saved in this run: call directions, the brain. Nothing else was changed.\n")


def test_quitting_after_a_purchase_in_plain_mode_names_it(typed, capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("AI_PHONE_HOME", str(tmp_path))
    actions = FakeActions(exists=True, saved_keys={"TWILIO_ACCOUNT_SID": "ACsid", "TWILIO_AUTH_TOKEN": "tok"})
    monkeypatch.setattr("ai_phone.setup.actions.Actions", lambda path: actions)
    # Add a line, A phone number, Twilio, the saved key, A new number, US, any area, the first number, Buy, then EOF
    typed.answers += ["1", "1", "1", "1", "1", "", "", "1", "1", EOFError()]
    assert run_setup(plain=True) == 1
    assert len(actions.args("buy_number")) == 1
    assert capsys.readouterr().out.endswith("Setup cancelled. Saved in this run: line twilio (+15550100001, a Twilio "
                                            "number billed monthly). Nothing else was changed.\n")


def test_ctrl_c_during_a_download_in_plain_mode_exits_at_once(typed, capsys, monkeypatch, tmp_path):
    """huggingface_hub joins its download threads at exit: a normal exit would wait for the whole download."""
    import os
    monkeypatch.setenv("AI_PHONE_HOME", str(tmp_path))
    actions = FakeActions(exists=True, missing=["mlx-community/Qwen3.5-4B-MLX-4bit"], pull_models=KeyboardInterrupt())
    monkeypatch.setattr("ai_phone.setup.actions.Actions", lambda path: actions)
    exits = []

    def fake_exit(code):
        exits.append(code)
        raise SystemExit(code)

    monkeypatch.setattr(os, "_exit", fake_exit)
    typed.answers += ["5", "", "", "", ""]  # Change call settings, three defaults, Download now
    with pytest.raises(SystemExit):
        run_setup(plain=True)
    assert exits == [1]
    assert capsys.readouterr().out.endswith("Downloading…\n\nSetup cancelled. The download stops now and picks up where it "
                                            "left off next time. Saved in this run: call settings.\n")


def test_a_subscription_brain_in_plain_mode(typed, capsys, monkeypatch, tmp_path):
    """The numbered prompts inherit the whole branch: the menu, the login check and the model."""
    monkeypatch.setenv("AI_PHONE_HOME", str(tmp_path))
    actions = FakeActions()
    monkeypatch.setattr("ai_phone.setup.actions.Actions", lambda path: actions)
    typed.answers += [
        "",    # Welcome: Continue
        "",    # Mode: Receive and place calls
        "2",   # Brain: My Claude subscription
        "",    # model: Haiku
        "2", "", "",          # a SIP address, with its defaults
        "2",                  # speech model: Kokoro
        "", "", "", "",       # voice group, voice, use it, speed
        "", "", "",           # greeting, instructions, daily limit
        "",    # Check: Continue
        "",    # Done: Finish
    ]
    assert run_setup(plain=True) == 0
    assert actions.changes() == ["save_call_directions", "save_brain_subscription", "add_sip_line", "save_voice", "save_callers"]
    assert actions.args("save_brain_subscription") == [("claude", "haiku")]
    out = capsys.readouterr().out
    assert "2) My Claude subscription" in out and "uses the Claude Code login · no API key" in out
    assert "Checking your Claude login…" in out
    assert "1) Haiku" in out and "fastest to answer · the one for a phone call" in out
    assert "✓ Brain" in out and "Claude (Haiku) · logged in" in out
    assert "Brain    your Claude subscription · Haiku" in out
    assert not typed.answers


def test_a_cli_to_install_in_plain_mode(typed, capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("AI_PHONE_HOME", str(tmp_path))
    actions = FakeActions(cli_installed=False)
    monkeypatch.setattr("ai_phone.setup.actions.Actions", lambda path: actions)
    typed.answers += ["", "", "3", "b", "q"]  # Continue, receive and place calls, My ChatGPT subscription, Back from the install page, quit
    assert run_setup(plain=True) == 1
    out = capsys.readouterr().out
    assert "Codex is not installed" in out and "npm install -g @openai/codex" in out and "codex login" in out
    assert "  1) Check again" in out and "  b) Back" in out
    assert actions.changes() == ["save_call_directions"] and not typed.answers

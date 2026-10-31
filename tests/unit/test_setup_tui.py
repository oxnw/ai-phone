"""The full-screen setup wizard, driven headless by Textual's pilot with small scripted flows."""
import asyncio
import dataclasses
import threading
import time
import types

import pytest
from rich.color import ColorType
from textual.widgets import Input, ProgressBar

from ai_phone.setup import look, tui
from ai_phone.setup.tui import FRAMES, PageScreen, SetupApp, run_tui
from ai_phone.setup.ui import Back, Cancel, Check, Field, Option, Outcome, Page

LINE = Page(3, "Which connection should ai-phone use?", "detail")
OPTIONS = [Option("A phone number", "hint"), Option("A SIP address")]
WELCOME = Page(0, "Give your AI a phone number", "Calls answered and placed by models running on this computer.",
               note="Picks a line, a brain and a voice, then checks it all works.",
               checks=[Check("SIP library", "built", True), Check("Memory", "16 GB", None),
                       Check("Models", "Parakeet 0.6B v2\nKokoro 82M\nPhonon 31M\nQwen3.5 4B", True),
                       Check("Config", "not saved yet", False)])
TWILIO = Page(3, "Your Twilio credentials",
              "Saved to ~/.ai-phone/secrets.env, readable only by you. Never shown or sent anywhere else.")
KEY_FIELDS = [Field("Account SID", secret=True, placeholder="AC…"),
              Field("Auth Token", secret=True, placeholder="32 characters")]


def screen_text(app) -> str:
    """What the terminal shows, one line per row (each row is a few strips of segments)."""
    rows = app.screen._compositor.render_full_update().strips
    return "\n".join("".join(seg.text for strip in row for seg in strip) for row in rows)


async def wait_for(pilot, app, predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting; the screen shows:\n{screen_text(app)}")
        await pilot.pause(0.01)


def segments(app):
    for row in app.screen._compositor.render_full_update().strips:
        for strip in row:
            yield from strip


def colour_of(app, text: str) -> str | None:
    """The foreground colour of the first drawn segment that contains `text`."""
    for seg in segments(app):
        if text in seg.text and seg.style and seg.style.color:
            return seg.style.color.get_truecolor().hex
    return None


def colour_at(app, row: int, col: int) -> str | None:
    """The foreground colour of one cell."""
    x = 0
    for strip in app.screen._compositor.render_full_update().strips[row]:
        for seg in strip:
            if x <= col < x + len(seg.text):
                return seg.style.color.get_truecolor().hex if seg.style and seg.style.color else None
            x += len(seg.text)
    return None


def headless(autopilot=None):
    """SetupApp run by run_tui without a terminal, driven by `autopilot`."""
    class Headless(SetupApp):
        def run(self, **kw):
            return super().run(headless=True, auto_pilot=autopilot, **kw)
    return Headless


def showing(app, text: str):
    """A predicate: a question screen is up and shows `text`."""
    return lambda: isinstance(app.screen, PageScreen) and app.screen.is_mounted and text in screen_text(app)


@pytest.mark.asyncio
async def test_choose_renders_and_returns_the_pick():
    got = {}

    def flow(ui):
        got["answer"] = ui.choose(LINE, OPTIONS)
        return Outcome(0)

    app = SetupApp(flow)
    async with app.run_test(size=(104, 40)) as pilot:
        await wait_for(pilot, app, showing(app, "[3] Back"))
        text = screen_text(app)
        for part in ("✓ Welcome", "● Line", "○ Voice", "S T E P   4   O F   8", "ai-phone setup",
                     "Which connection should ai-phone use?", "[1] A phone number", "hint", "[2] A SIP address",
                     "esc back"):
            assert part in text
        await pilot.press("down", "enter")
        await wait_for(pilot, app, lambda: app.outcome is not None)
    assert got["answer"] == 1
    assert app.outcome == Outcome(0)


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [70, 104])
async def test_call_directions_page_explains_runtime_and_all_choices_fit(width):
    from ai_phone.setup import flow
    from tests.unit.test_setup_flow import FakeActions

    actions = FakeActions()

    def choose_mode(ui):
        ui.begin(flow.MODE)
        flow.call_mode(ui, actions, flow.Session())
        return Outcome(0)

    app = SetupApp(choose_mode)
    async with app.run_test(size=(width, 30)) as pilot:
        await wait_for(pilot, app, showing(app, "[4] Back"))
        text = screen_text(app)
        for label in ("Receive and place calls", "Place calls only", "Receive calls only", "running"):
            assert label in text
        assert look.MENU_KEYS in text
        assert not app.screen.max_scroll_y
        assert max(len(row.rstrip()) for row in text.splitlines()) <= width
        await pilot.press("2")
        await wait_for(pilot, app, lambda: app.outcome is not None)
    assert actions.args("save_call_directions") == [(False, True)]


@pytest.mark.asyncio
async def test_existing_setup_caller_id_shortcut_is_visible_and_changes_the_selected_number():
    from ai_phone.setup import flow
    from tests.unit.test_setup_flow import FakeActions, caller_id_line

    actions = FakeActions(exists=True, lines={"main": caller_id_line()})
    app = SetupApp(lambda ui: flow.run(ui, actions))
    async with app.run_test(size=(104, 40)) as pilot:
        await wait_for(pilot, app, showing(app, "[6] Change caller ID"))
        await pilot.press("6")
        await wait_for(pilot, app, showing(app, "Which line's caller ID?"))
        assert "● Settings" in screen_text(app)
        await pilot.press("1")
        await wait_for(pilot, app, showing(app, "Which number should people see?"))
        text = screen_text(app)
        assert "Line's number · +15555550100" in text
        assert "Your verified number · +15555550101" in text
        assert "Verify another number you own" in text
        await pilot.press("1")
        await wait_for(pilot, app, showing(app, "Everything works"))
        await pilot.press("enter")
        await wait_for(pilot, app, showing(app, "You're set up"))
        await pilot.press("enter")
        await wait_for(pilot, app, lambda: app.outcome is not None)
    assert app.outcome.code == 0
    assert actions.args("use_caller_id") == [("main", "+15555550100")]
    assert not actions.args("saved_keys") and not actions.args("start_caller_id")


@pytest.mark.asyncio
async def test_digits_jump_and_escape_goes_back():
    got = {}

    def pick(ui):
        got["pick"] = ui.choose(LINE, OPTIONS)
        return Outcome(0)

    app = SetupApp(pick)
    async with app.run_test(size=(104, 40)) as pilot:
        await wait_for(pilot, app, showing(app, "[3] Back"))
        await pilot.press("2")
        await wait_for(pilot, app, lambda: app.outcome is not None)
    assert got["pick"] == 1

    backs = []

    def go_back(ui):
        ui.begin(3)
        for question in ("First", "Back option"):
            try:
                ui.choose(Page(3, question), OPTIONS)
            except Back as e:
                backs.append(e.at_start)
        ui.begin(3)
        ui.choose(Page(3, "Earlier"), OPTIONS)
        try:
            ui.choose(Page(3, "Later"), OPTIONS)
        except Back as e:
            backs.append(e.at_start)
        try:
            ui.ask(Page(3, "Your Telnyx credentials"), [Field("API key", secret=True)])
        except Back as e:
            backs.append(e.at_start)
        return Outcome(0)

    app = SetupApp(go_back)
    async with app.run_test(size=(104, 40)) as pilot:
        await wait_for(pilot, app, showing(app, "First"))
        await pilot.press("escape")
        await wait_for(pilot, app, showing(app, "Back option"))
        await pilot.press("3")  # the trailing Back option
        await wait_for(pilot, app, showing(app, "Earlier"))
        await pilot.press("enter")
        await wait_for(pilot, app, showing(app, "Later"))
        await pilot.press("escape")
        await wait_for(pilot, app, showing(app, "API key"))
        await pilot.press("escape")  # a form goes back too, even while its field has focus
        await wait_for(pilot, app, lambda: app.outcome is not None)
    # asked counts up since begin(): the Back option is the step's second question, so it is not at the start
    assert backs == [True, False, False, False]


@pytest.mark.asyncio
async def test_the_default_option_starts_selected():
    got = {}

    def flow(ui):
        got["answer"] = ui.choose(LINE, OPTIONS, default=1)
        return Outcome(0)

    app = SetupApp(flow)
    async with app.run_test(size=(104, 40)) as pilot:
        await wait_for(pilot, app, showing(app, "▸ [2] A SIP address"))
        await pilot.press("enter")
        await wait_for(pilot, app, lambda: app.outcome is not None)
    assert got["answer"] == 1


@pytest.mark.asyncio
async def test_a_digit_that_is_not_1_to_9_does_nothing():
    got = {}

    def flow(ui):
        got["answer"] = ui.choose(LINE, OPTIONS)
        return Outcome(0)

    app = SetupApp(flow)
    async with app.run_test(size=(104, 40)) as pilot:
        await wait_for(pilot, app, showing(app, "[3] Back"))
        await pilot.press("²", "0", "4")  # ² is its own key on French keyboards; "²".isdigit() is True
        await pilot.pause(0.1)
        assert app.return_code is None and "answer" not in got and "▸ [1]" in screen_text(app)
        await pilot.press("2")
        await wait_for(pilot, app, lambda: app.outcome is not None)
    assert got["answer"] == 1


@pytest.mark.asyncio
async def test_back_can_be_the_default():
    raised = []

    def flow(ui):
        try:
            ui.choose(LINE, OPTIONS, default=len(OPTIONS))
        except Back as e:
            raised.append(e)
        return Outcome(0)

    app = SetupApp(flow)
    async with app.run_test(size=(104, 40)) as pilot:
        await wait_for(pilot, app, showing(app, "▸ [3] Back"))
        await pilot.press("enter")
        await wait_for(pilot, app, lambda: app.outcome is not None)
    assert len(raised) == 1


@pytest.mark.asyncio
async def test_q_cancels():
    seen = []

    def flow(ui):
        try:
            ui.choose(LINE, OPTIONS)
        except Cancel:
            seen.append("cancel")
            raise
        return Outcome(0)

    app = SetupApp(flow)
    async with app.run_test(size=(104, 40)) as pilot:
        await wait_for(pilot, app, showing(app, "[3] Back"))
        await pilot.press("q")
        await wait_for(pilot, app, lambda: app.outcome is not None)
    assert seen == ["cancel"]
    assert app.outcome.code == 1


@pytest.mark.asyncio
async def test_form_hides_secrets_and_validates():
    got = {}
    checked = []

    def name(value):
        checked.append(value)
        return None if value == "agent" else "Use the name agent."

    fields = [Field("Account SID", secret=True, placeholder="AC…"), Field("Name", validate=name)]

    def flow(ui):
        got["values"] = ui.ask(Page(3, "Your Twilio credentials", "Saved to ~/.ai-phone/secrets.env."), fields)
        return Outcome(0)

    app = SetupApp(flow)
    async with app.run_test(size=(104, 40)) as pilot:
        seen = []

        async def press(*keys):
            for key in keys:
                await pilot.press(key)
                seen.append(screen_text(app))

        await wait_for(pilot, app, showing(app, "Account SID"))
        first, second = app.screen.query(Input)
        assert app.focused is first
        assert "⏎ next field   esc back   ctrl+c quit" in screen_text(app)
        await press(*"ACsecret123", "enter")
        assert app.focused is second
        await press(*"bad", "enter")
        await wait_for(pilot, app, lambda: "Use the name agent." in screen_text(app))
        assert colour_of(app, "Use the name agent.") == look.RED
        assert app.focused is second and "values" not in got
        await press("q")  # a letter in a field, not quit
        assert second.value == "badq" and app.outcome is None
        app.screen.set_focus(None)
        await press("q")  # not even when no field has focus
        assert app.outcome is None and not app.screen.answered
        second.focus()
        second.clear()
        await press(*"agent", "enter")
        await wait_for(pilot, app, lambda: app.outcome is not None)
        seen.append(screen_text(app))
    assert got["values"] == ["ACsecret123", "agent"]
    assert checked == ["bad", "agent"]  # once per Enter, even on the last field
    assert seen and not any("ACsecret123" in t or "secret123" in t for t in seen)


@pytest.mark.asyncio
async def test_a_validator_that_fails_cannot_take_the_screen_down():
    got = {}

    def fragile(value):
        if value.startswith("AC"):
            raise ValueError(f"cannot parse {value}")  # the value would be in a traceback's locals
        return None

    def flow(ui):
        got["values"] = ui.ask(TWILIO, [Field("Account SID", secret=True, validate=fragile)])
        return Outcome(0)

    app = SetupApp(flow)
    async with app.run_test(size=(104, 40)) as pilot:
        await wait_for(pilot, app, showing(app, "Account SID"))
        await pilot.press(*"ACsecret789", "enter")
        await wait_for(pilot, app, lambda: "That value could not be checked." in screen_text(app))
        assert colour_of(app, "That value could not be checked.") == look.RED
        assert app.return_code is None and "values" not in got and isinstance(app.focused, Input)
        assert "secret789" not in screen_text(app)
        app.focused.clear()
        await pilot.press(*"sk", "enter")
        await wait_for(pilot, app, lambda: app.outcome is not None)
    assert got["values"] == ["sk"]


@pytest.mark.asyncio
async def test_busy_spins_then_returns():
    release = threading.Event()
    got = {}

    def work():
        release.wait(5)
        return 7

    def flow(ui):
        got["result"] = ui.busy(Page(3, "Your Twilio credentials"), "Checking your Twilio account…", work)
        return Outcome(0)

    def spinner(app) -> str:
        line = next((l for l in screen_text(app).splitlines() if "Checking your Twilio account…" in l), "")
        return line.strip()[:1]

    app = SetupApp(flow)
    async with app.run_test(size=(104, 40)) as pilot:
        await wait_for(pilot, app, lambda: spinner(app) in tuple("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"))
        assert app.screen.busy == "Checking your Twilio account…"
        assert "ctrl+c quit" in screen_text(app) and "esc back" not in screen_text(app)
        assert colour_of(app, "Checking your Twilio account…") == look.MID
        row = next(l for l in screen_text(app).splitlines() if "Checking your Twilio account…" in l)
        assert row.index("Checking") == 22  # the frame at column 20, like a form's status line
        frame = spinner(app)
        await wait_for(pilot, app, lambda: spinner(app) != frame)  # it turns
        assert "result" not in got
        release.set()
        await wait_for(pilot, app, lambda: app.outcome is not None)
    assert got["result"] == 7


@pytest.mark.asyncio
async def test_busy_after_a_form_keeps_the_form_on_screen():
    checked, listed = threading.Event(), threading.Event()
    got = {}

    def check():
        checked.wait(5)
        raise ValueError("rejected")

    def flow(ui):
        ui.ask(TWILIO, KEY_FIELDS)
        try:
            ui.busy(dataclasses.replace(TWILIO), "Checking your Twilio account…", check)  # an equal page will do
        except ValueError:
            got["retry"] = ui.ask(dataclasses.replace(TWILIO, error="Twilio did not accept that key."), KEY_FIELDS)
        got["numbers"] = ui.busy(Page(3, "Which number?"), "Loading your numbers…", lambda: listed.wait(5))
        return Outcome(0)

    app = SetupApp(flow)
    async with app.run_test(size=(104, 40)) as pilot:
        seen = []

        async def press(*keys):
            for key in keys:
                await pilot.press(key)
                seen.append(screen_text(app))

        await wait_for(pilot, app, showing(app, "Account SID"))
        form = app.screen
        await press(*"ACsecret123", "enter", *"tokensecret456", "enter")
        await wait_for(pilot, app, lambda: "Checking your Twilio account…" in screen_text(app))
        assert app.screen is form
        text = screen_text(app)
        rows = text.splitlines()
        (row,) = [r for r, line in enumerate(rows) if "Checking your Twilio account…" in line]
        # the spinner is on the form's status line, under the fields, where the approved frame draws it
        assert (row, rows[row].index("Checking")) == (26, 22) and rows[row][20] in FRAMES
        assert "Account SID" in text and "Auth Token" in text and "•" * 11 in text and "•" * 14 in text
        assert all(field.disabled for field in form.query(Input))
        # read-only fields look as they did: not faded, and the one that had focus keeps its accent border
        tops = [(r, line.index("╭")) for r, line in enumerate(rows) if "╭" in line]
        assert [colour_at(app, r, c) for r, c in tops] == ["#2a2d4a", look.ACCENT]
        assert colour_of(app, "•" * 14) == look.WHITE
        assert "ctrl+c quit" in text and "esc back" not in text
        await press("x", "escape", "q", "enter")  # nothing reaches the flow while it works
        assert app.outcome is None and "retry" not in got and form.busy

        checked.set()
        await wait_for(pilot, app, showing(app, "Twilio did not accept that key."))
        assert app.screen is not form
        assert [(f.disabled, f.value) for f in app.screen.query(Input)] == [(False, ""), (False, "")]
        await press(*"AC9", "enter", *"tok9", "enter")
        await wait_for(pilot, app, lambda: "Loading your numbers…" in screen_text(app))
        assert "Account SID" not in screen_text(app)  # another page gets a busy screen of its own
        listed.set()
        await wait_for(pilot, app, lambda: app.outcome is not None)
    assert got == {"retry": ["AC9", "tok9"], "numbers": True}
    assert seen and not any("secret123" in t or "secret456" in t for t in seen)


@pytest.mark.asyncio
async def test_progress_bar_moves():
    release = threading.Event()
    got = {}

    def work(report):
        report(5, 10)
        report(10, 10)
        release.wait(5)
        return "pulled"

    def flow(ui):
        got["result"] = ui.progress(Page(2, "Where should your agent think?"), "Downloading models", work)
        return Outcome(0)

    def bar(app):
        bars = app.screen.query(ProgressBar) if isinstance(app.screen, PageScreen) else []
        return bars[0] if bars else None

    app = SetupApp(flow)
    async with app.run_test(size=(104, 40)) as pilot:
        await wait_for(pilot, app, lambda: bar(app) is not None and bar(app).progress == 10)
        assert bar(app).total == 10
        await wait_for(pilot, app, lambda: "100%" in screen_text(app))
        row = next(l for l in screen_text(app).splitlines() if "━" in l)
        # the bar and its percentage fill the 66-column row, from the status line's column to the form's right edge
        assert row.index("━") == 20 and len(row.rstrip()) == 85 and row.rstrip().endswith("100%")
        await wait_for(pilot, app, lambda: "Downloading models  0.0 of 0.0 GB" in screen_text(app))
        release.set()
        await wait_for(pilot, app, lambda: app.outcome is not None)
    assert got["result"] == "pulled"


async def welcome_screen(width: int) -> str:
    app = SetupApp(lambda ui: Outcome(ui.choose(WELCOME, [Option("Continue"), Option("Cancel")], back=False)))
    async with app.run_test(size=(width, 40)) as pilot:
        await wait_for(pilot, app, showing(app, "[2] Cancel"))
        text = screen_text(app)
        await pilot.press("enter")
        await wait_for(pilot, app, lambda: app.outcome is not None)
    return text


@pytest.mark.asyncio
async def test_welcome_shows_the_wordmark_when_wide_and_a_title_when_narrow():
    wide = await welcome_screen(104)
    assert "▀" in wide or "█" in wide
    assert all(step in wide for step in ("● Welcome ←", "○ Mode", "○ Settings", "○ Done"))
    for part in ("Give your AI a phone number", "Picks a line, a brain and a voice", "SIP library",
                 "Parakeet 0.6B v2", "Kokoro 82M", "Phonon 31M", "Qwen3.5 4B", "✓", "✗", "○"):
        assert part in wide
    assert "esc back" not in wide and "[3]" not in wide  # no Back on Welcome

    narrow = await welcome_screen(70)
    lines = [line.strip() for line in narrow.splitlines()]
    assert "ai-phone" in lines
    assert not any(block in narrow for block in "▀▄█")
    assert "● Welcome  1/8" in narrow and "○ Brain" not in narrow
    for model in ("Parakeet 0.6B v2", "Kokoro 82M", "Phonon 31M", "Qwen3.5 4B"):
        assert model in narrow


@pytest.mark.asyncio
async def test_resizing_the_terminal_refits_the_crumbs_and_the_wordmark():
    app = SetupApp(lambda ui: Outcome(ui.choose(WELCOME, [Option("Continue")], back=False)))
    async with app.run_test(size=(104, 40)) as pilot:
        await wait_for(pilot, app, showing(app, "○ Brain"))
        await pilot.resize_terminal(70, 40)
        await wait_for(pilot, app, lambda: "● Welcome  1/8" in screen_text(app))
        assert "█" not in screen_text(app) and "ai-phone" in [l.strip() for l in screen_text(app).splitlines()]
        await pilot.resize_terminal(104, 40)
        await wait_for(pilot, app, lambda: "○ Brain" in screen_text(app) and "█" in screen_text(app))
        await pilot.press("enter")
        await wait_for(pilot, app, lambda: app.outcome is not None)


# the shape flow.done_text really produces: long rows, and more of them than a 30-row terminal can hold
NARROW_DONE = "\n".join([
    "Saved in ~/.ai-phone/   config.yaml and secrets.env (keys and passwords, private to you)",
    "",
    "Lines    main    +15550100001           Twilio number",
    "         agent   sip:agent@192.0.2.10   SIP address",
    "Calls    receive and place calls",
    "Brain    on this computer · Qwen3.5 4B",
    "Voice    Heart · American · female · 1.0×",
    "",
    "Start ai-phone",
    "  ai-phone run",
    "  Leave ai-phone running while you want to receive calls.",
    "  If already running, stop and restart to apply these settings.",
    "",
    "Try it",
    "  call +15550100001 from any phone (your normal call rates)",
    "  pjsua --local-port 5070 sip:agent@127.0.0.1   talk through this computer's mic, free",
])
# the widest row of the summary, and one the page keeps in view: 81 columns, so at 70 it has to wrap
WIDEST = "pjsua --local-port 5070 sip:agent@127.0.0.1   talk through this computer's mic, free"


def clipped(text: str) -> list[str]:
    """Rows where a bordered box ran off the right edge: a top or bottom rule that lost its corner, or a middle row
    that lost its right wall. Only the input boxes draw these characters."""
    bad = []
    for row in text.splitlines():
        edge = row.strip()
        if edge.startswith(("╭", "╰")) and not edge.endswith(("╮", "╯")):
            bad.append(row)
        elif "│" in edge and edge.count("│") != 2:
            bad.append(row)
    return bad


FORM_PAGE = Page(3, "Your Twilio credentials", "Saved to ~/.ai-phone/secrets.env, readable only by you.")
# one page of each kind the wizard shows, with the text that says it is up: Welcome has no eyebrow and a wordmark,
# a question has a menu, a form has input boxes
PAGE_KINDS = {"welcome": (WELCOME, [Option("Continue"), Option("Cancel")], None, "[2] Cancel"),
              "question": (LINE, OPTIONS, None, "[2] A SIP address"),
              "form": (FORM_PAGE, None, KEY_FIELDS, "Auth Token")}


async def render(size, page, options=None, fields=None, marker=""):
    """Drive one page onto a terminal of `size` and return what it shows, with how it chose to lay itself out."""
    if options is not None:
        app = SetupApp(lambda ui: Outcome(ui.choose(page, options, back=page.step != 0)))
        keys = look.MENU_KEYS if page.step else look.MENU_KEYS.replace("esc back   ", "")  # no Back on Welcome
    else:
        app = SetupApp(lambda ui: Outcome(0) if ui.ask(page, fields) else Outcome(1))
        keys = look.FORM_KEYS
    async with app.run_test(size=size) as pilot:
        await wait_for(pilot, app, showing(app, marker))
        shown = types.SimpleNamespace(
            rows=[row.rstrip() for row in screen_text(app).splitlines()], keys=keys, width=size[0],
            scrolled=app.screen.show_vertical_scrollbar or app.screen.max_scroll_y > 0,
            dense=app.screen.has_class("dense"))
        if fields is None:
            await pilot.press("enter")
        else:
            await pilot.press("a", "enter", "b", "enter")
        await wait_for(pilot, app, lambda: app.outcome is not None)
    return shown


def eyebrow_row(shown) -> int:
    return next(i for i, row in enumerate(shown.rows) if "S T E P" in row)


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [70, 104])
async def test_a_narrow_terminal_clips_no_form_row(width):
    """A 70-column terminal is the narrowest the wizard is meant to look right in: the input boxes keep both
    borders and come out the same width as each other. 104 columns is the control."""
    shown = await render((width, 30), FORM_PAGE, fields=KEY_FIELDS, marker="Auth Token")
    assert clipped("\n".join(shown.rows)) == []
    assert max(len(row) for row in shown.rows) <= width
    tops = [row for row in shown.rows if row.strip().startswith("╭")]
    assert len(tops) == 2 and len({len(row.strip()) for row in tops}) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [70, 104])
@pytest.mark.parametrize("kind", list(PAGE_KINDS))
async def test_a_30_row_terminal_fits_the_page_and_keeps_its_key_hints(kind, width):
    """The approved look's spacing wants 40 rows; in a shorter terminal it would push the key hints below the
    bottom of the screen, so the wizard tightens the spacing by height as well as by body length."""
    page, options, fields, marker = PAGE_KINDS[kind]
    shown = await render((width, 30), page, options, fields, marker)
    assert shown.dense and not shown.scrolled  # the whole page is on screen, not one scroll below it
    assert any(shown.keys in row for row in shown.rows)  # and so is the whole key hint
    assert clipped("\n".join(shown.rows)) == []
    assert max(len(row) for row in shown.rows) <= width
    if kind != "welcome":  # Welcome has a wordmark where the others have an eyebrow
        assert eyebrow_row(shown) == 5  # one blank row above it, not five


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", list(PAGE_KINDS))
async def test_40_rows_still_gets_the_spacing_the_look_was_approved_with(kind):
    """The tightening is for short terminals only: at 40 rows nothing about the approved look moves."""
    page, options, fields, marker = PAGE_KINDS[kind]
    shown = await render((104, 40), page, options, fields, marker)
    assert not shown.dense and not shown.scrolled
    assert any(shown.keys in row for row in shown.rows)
    if kind != "welcome":
        assert eyebrow_row(shown) == 9  # five blank rows above it


@pytest.mark.asyncio
async def test_resizing_into_a_short_terminal_tightens_the_spacing_and_back():
    app = SetupApp(lambda ui: Outcome(ui.choose(LINE, OPTIONS)))
    async with app.run_test(size=(104, 40)) as pilot:
        await wait_for(pilot, app, showing(app, "[2] A SIP address"))
        assert not app.screen.has_class("dense")
        await pilot.resize_terminal(104, 30)
        await wait_for(pilot, app, lambda: app.screen.has_class("dense"))
        rows = [row.rstrip() for row in screen_text(app).splitlines()]
        assert not app.screen.show_vertical_scrollbar and any(look.MENU_KEYS in row for row in rows)
        await pilot.resize_terminal(104, 40)
        await wait_for(pilot, app, lambda: not app.screen.has_class("dense"))
        await pilot.press("enter")
        await wait_for(pilot, app, lambda: app.outcome is not None)


@pytest.mark.asyncio
async def test_a_narrow_terminal_wraps_the_done_body_instead_of_cutting_it():
    """The Done page's summary is wider than 70 columns and taller than 30 rows, so the page scrolls — but no row
    of it is cut off on the right, which is what a fixed width would have done."""
    done = Page(7, "You're set up", "Everything is saved. Here's how to use it.", body=NARROW_DONE)
    app = SetupApp(lambda ui: Outcome(ui.choose(done, [Option("Finish")], back=False)))
    async with app.run_test(size=(70, 30)) as pilot:
        await wait_for(pilot, app, showing(app, "Everything is saved."))
        rows = [row.rstrip() for row in screen_text(app).splitlines()]
        assert clipped("\n".join(rows)) == []
        assert max(len(row) for row in rows) <= 70
        flat = lambda text: " ".join(text.split())  # noqa: E731
        assert not any(WIDEST in row for row in rows)  # too wide for one row at 70 columns
        assert flat(WIDEST) in flat(" ".join(rows))  # so it wraps, whole, over two rows
        await pilot.press("enter")
        await wait_for(pilot, app, lambda: app.outcome is not None)


@pytest.mark.asyncio
async def test_a_long_body_fits_without_scrolling():
    # the Done page: a summary of ~22 lines, one option and the hint must fit in 40 rows
    lines = [f"line {i:02d}  " + "·" * 60 for i in range(22)]
    done = Page(7, "You're set up", "Everything is saved. Here's how to use it.", body="\n".join(lines))
    app = SetupApp(lambda ui: Outcome(ui.choose(done, [Option("Finish")], back=False)))
    async with app.run_test(size=(104, 40)) as pilot:
        await wait_for(pilot, app, showing(app, "[1] Finish"))
        rows = screen_text(app).splitlines()
        assert app.screen.max_scroll_y == 0 and not app.screen.show_vertical_scrollbar
        assert any("↑↓ move   ⏎ select   1-9 jump   q quit" in row for row in rows)
        assert "S T E P   8   O F   8" in rows[5]  # the eyebrow moves up
        body = [(row.index(line), row) for line in lines for row in rows if line in row]
        assert len(body) == 22
        detail = next(r for r, row in enumerate(rows) if "Everything is saved." in row)
        assert rows.index(body[0][1]) == detail + 2  # one blank line under the detail, no more
        left = body[0][0]
        assert {col for col, _ in body} == {left}
        assert abs((left - 5) - (99 - (left + len(lines[0])))) <= 1  # still centred in the page's 94 columns
        await pilot.press("enter")
        await wait_for(pilot, app, lambda: app.outcome is not None)


@pytest.mark.asyncio
async def test_no_color(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    app = SetupApp(lambda ui: Outcome(ui.choose(WELCOME, [Option("Continue")], back=False)))
    async with app.run_test(size=(104, 40)) as pilot:
        await wait_for(pilot, app, showing(app, "[1] Continue"))
        drawn = [seg for seg in segments(app) if seg.text.strip()]
        await pilot.press("enter")
        await wait_for(pilot, app, lambda: app.outcome is not None)
    assert drawn
    assert not [seg for seg in drawn if seg.style and seg.style.color and seg.style.color.type == ColorType.TRUECOLOR]

    # with no colours to tell it apart, the cursor in a field is drawn in reverse video
    app = SetupApp(lambda ui: Outcome(0, ui.ask(Page(3, "Your Telnyx credentials"), [Field("API key")])[0]))
    async with app.run_test(size=(104, 40)) as pilot:
        await wait_for(pilot, app, showing(app, "API key"))
        await pilot.press("k", "e", "y")
        cursor = [seg for seg in segments(app) if seg.style and seg.style.reverse]
        await pilot.press("enter")
        await wait_for(pilot, app, lambda: app.outcome is not None)
    assert cursor and app.outcome == Outcome(0, "key")


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["ctrl+c", "ctrl+q"])
async def test_ctrl_c_on_a_question_cancels_and_exits(key):
    seen = []

    def flow(ui):
        try:
            ui.ask(Page(3, "Your Telnyx credentials"), [Field("API key", secret=True)])
        except Cancel:
            seen.append("cancel")
            raise
        return Outcome(0)

    app = SetupApp(flow)
    async with app.run_test(size=(104, 40)) as pilot:
        await wait_for(pilot, app, showing(app, "API key"))
        assert isinstance(app.focused, Input)  # the key reaches the app even while a field has focus
        await pilot.press(key)
        await wait_for(pilot, app, lambda: app.outcome is not None)
        await wait_for(pilot, app, lambda: app.return_code is not None)
    assert seen == ["cancel"]
    assert app.outcome == Outcome(1, "Setup cancelled.")


@pytest.mark.asyncio
async def test_ctrl_c_between_an_answer_and_the_next_question_still_cancels():
    class LateQuit(SetupApp):
        def show(self, screen):
            if screen.page.question == "Second":
                self.action_request_quit()  # lands while the answered first question is still up
            return super().show(screen)

    seen = []

    def flow(ui):
        ui.choose(Page(3, "First"), OPTIONS)
        try:
            ui.choose(Page(3, "Second"), OPTIONS)
        except Cancel:
            seen.append("cancel")
            raise
        return Outcome(0)

    app = LateQuit(flow)
    async with app.run_test(size=(104, 40)) as pilot:
        await wait_for(pilot, app, showing(app, "First"))
        await pilot.press("enter")
        await wait_for(pilot, app, lambda: app.outcome is not None)
    assert seen == ["cancel"] and app.outcome == Outcome(1, "Setup cancelled.")


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["busy", "progress"])
async def test_ctrl_c_just_before_work_starts_cancels_the_work(kind):
    class LateQuit(SetupApp):
        def show(self, screen):
            if screen.busy:
                self.action_request_quit()  # lands while the answered question is still up
            return super().show(screen)

    ran = []

    def flow(ui):
        ui.choose(Page(4, "Download the voice now?"), OPTIONS)
        work_page = Page(4, "Downloading the voice")
        if kind == "busy":
            ui.busy(work_page, "Downloading…", lambda: ran.append("work"))
        else:
            ui.progress(work_page, "Downloading…", lambda report: ran.append("work"))
        return Outcome(0)

    app = LateQuit(flow)
    async with app.run_test(size=(104, 40)) as pilot:
        await wait_for(pilot, app, showing(app, "Download the voice now?"))
        await pilot.press("enter")
        await wait_for(pilot, app, lambda: app.outcome is not None)
    assert ran == [] and app.outcome == Outcome(1, "Setup cancelled.")


def stopping_line(app) -> str:
    return next((l.strip() for l in screen_text(app).splitlines() if "Stopping after this step…" in l), "")


@pytest.mark.asyncio
async def test_ctrl_c_while_working_lets_the_step_finish_then_stops():
    release = threading.Event()
    got = {}

    def flow(ui):
        got["step"] = ui.busy(Page(3, "Buying +15550100001"), "Buying the number…", lambda: release.wait(5))
        ui.choose(Page(3, "Which voice?"), OPTIONS)  # never shown: the person asked to quit
        return Outcome(0)

    app = SetupApp(flow)
    async with app.run_test(size=(104, 40)) as pilot:
        await wait_for(pilot, app, showing(app, "Buying the number…"))
        await pilot.press("ctrl+c")
        await wait_for(pilot, app, lambda: stopping_line(app)[:1] in tuple(FRAMES))
        assert "Buying the number…" not in screen_text(app)
        assert app.quit_requested and app.return_code is None and "step" not in got
        frame = stopping_line(app)[:1]
        await wait_for(pilot, app, lambda: stopping_line(app)[:1] != frame)  # still spinning
        release.set()
        await wait_for(pilot, app, lambda: app.outcome is not None)
        await wait_for(pilot, app, lambda: app.return_code is not None)
    assert got["step"] is True  # the step ran to its end
    assert app.outcome == Outcome(1, "Setup cancelled.")


@pytest.mark.asyncio
async def test_ctrl_c_while_a_form_is_checked_says_so_under_the_fields():
    release = threading.Event()

    def flow(ui):
        ui.ask(TWILIO, KEY_FIELDS)
        ui.busy(TWILIO, "Checking your Twilio account…", lambda: release.wait(5))
        ui.choose(Page(3, "Which number?"), OPTIONS)  # never shown
        return Outcome(0)

    app = SetupApp(flow)
    async with app.run_test(size=(104, 40)) as pilot:
        await wait_for(pilot, app, showing(app, "Account SID"))
        await pilot.press("a", "enter", "b", "enter")
        await wait_for(pilot, app, lambda: "Checking your Twilio account…" in screen_text(app))
        await pilot.press("ctrl+c")
        await wait_for(pilot, app, lambda: "Stopping after this step…" in screen_text(app))
        rows = screen_text(app).splitlines()
        assert rows[26].index("Stopping after this step…") == 22 and "Auth Token" in screen_text(app)
        assert app.return_code is None
        release.set()
        await wait_for(pilot, app, lambda: app.outcome is not None)
    assert app.outcome == Outcome(1, "Setup cancelled.")


@pytest.mark.asyncio
async def test_a_second_ctrl_c_while_working_leaves_at_once():
    release = threading.Event()

    def flow(ui):
        ui.busy(Page(4, "Which voice?"), "Loading voices…", lambda: release.wait(5))
        ui.choose(Page(4, "Which voice?"), OPTIONS)  # never shown: the person already asked to quit
        return Outcome(0)

    app = SetupApp(flow)
    async with app.run_test(size=(104, 40)) as pilot:
        await wait_for(pilot, app, showing(app, "Loading voices…"))
        await pilot.press("ctrl+c")
        await wait_for(pilot, app, lambda: "Stopping after this step…" in screen_text(app))
        assert app.return_code is None
        await pilot.press("ctrl+q")
        await wait_for(pilot, app, lambda: app.return_code is not None)
        assert app.outcome is None and not app.left_download
    # the work was left to finish on its own; the flow then stops at its next question
    release.set()
    deadline = time.monotonic() + 5
    while app.outcome is None and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
    assert app.outcome == Outcome(1, "Setup cancelled.")


@pytest.mark.asyncio
async def test_run_tui_returns_the_flow_outcome():
    app = SetupApp(lambda ui: Outcome(0, "done"))
    async with app.run_test() as pilot:
        await wait_for(pilot, app, lambda: app.outcome is not None)
    assert app.outcome == Outcome(0, "done")


def test_run_tui_reports_a_crash_without_its_message(monkeypatch):
    def crash(ui):
        raise RuntimeError("token AC123 rejected")

    def broken_screen(ui):  # a screen that cannot be drawn ends the app with an error
        ui.choose(Page(2, "Where should your agent think?", checks=[Check("Memory", None, True)]), OPTIONS)
        return Outcome(0)

    monkeypatch.setattr(tui, "SetupApp", headless())
    assert run_tui(lambda ui: Outcome(0, "done")) == Outcome(0, "done")
    assert run_tui(crash) == Outcome(3, "Setup stopped: RuntimeError")
    assert run_tui(broken_screen) == Outcome(3, "Setup stopped: the screen closed unexpectedly.")


def test_quitting_a_download_ends_the_process_without_waiting_for_it(monkeypatch, capsys):
    go, finished = threading.Event(), threading.Event()
    errors = []

    def download(report):
        report(1, 4)
        go.wait(5)
        try:
            report(2, 4)  # the person has quit: nothing to draw, and no error for the downloader
        except Exception as e:
            errors.append(e)
        finished.set()

    def flow(ui):
        ui.progress(Page(4, "Downloading the speech model"), "Downloading…", download)
        return Outcome(0)

    async def quit_while_downloading(pilot):
        await wait_for(pilot, pilot.app, showing(pilot.app, "Downloading…"))
        await pilot.press("ctrl+c")

    def fake_exit(code):
        raise SystemExit(code)

    monkeypatch.setattr(tui, "SetupApp", headless(quit_while_downloading))
    monkeypatch.setattr(tui.os, "_exit", fake_exit)
    with pytest.raises(SystemExit) as exited:
        run_tui(flow)
    assert exited.value.code == 1
    out = capsys.readouterr().out
    assert out == "Setup cancelled. The download stops now and picks up where it left off next time.\n"
    go.set()
    assert finished.wait(2) and errors == []


def test_quitting_other_work_returns_as_before(monkeypatch):
    release = threading.Event()
    exits = []

    def flow(ui):
        ui.busy(Page(4, "Which voice?"), "Loading voices…", lambda: release.wait(5))
        ui.choose(Page(4, "Which voice?"), OPTIONS)
        return Outcome(0)

    async def quit_while_loading(pilot):
        await wait_for(pilot, pilot.app, showing(pilot.app, "Loading voices…"))
        await pilot.press("ctrl+c")
        await wait_for(pilot, pilot.app, lambda: "Stopping after this step…" in screen_text(pilot.app))
        release.set()

    monkeypatch.setattr(tui, "SetupApp", headless(quit_while_loading))
    monkeypatch.setattr(tui.os, "_exit", exits.append)
    assert run_tui(flow) == Outcome(1, "Setup cancelled.")
    assert exits == []


@pytest.mark.asyncio
async def test_a_repeated_digit_never_confirms_a_purchase():
    """A double tap or a held key: the digit that picked the number lands again on "Buy …?". There a digit only moves
    the highlight; Enter confirms."""
    got = {}

    def flow(ui):
        got["pick"] = ui.choose(Page(3, "Pick a number"), [Option("+15550100001", "$1.15/month"), Option("+15550100002")])
        try:
            got["confirm"] = ui.choose(Page(3, "Buy +15550100001 for $1.15/month?"), [Option("Buy +15550100001")], default=1)
        except Back:
            got["confirm"] = "back"
        return Outcome(0)

    app = SetupApp(flow)
    async with app.run_test(size=(104, 40)) as pilot:
        await wait_for(pilot, app, showing(app, "Pick a number"))
        await pilot.press("1", "1")
        await wait_for(pilot, app, showing(app, "Buy +15550100001 for"))
        await pilot.press("1", "1")
        await pilot.pause(0.1)
        assert got == {"pick": 0} and "▸ [1] Buy +15550100001" in screen_text(app)
        await pilot.press("2")
        await pilot.pause(0.05)
        assert got == {"pick": 0} and "▸ [2] Back" in screen_text(app)
        await pilot.press("enter")
        await wait_for(pilot, app, lambda: app.outcome is not None)
    assert got == {"pick": 0, "confirm": "back"}

    got.clear()
    app = SetupApp(flow)
    async with app.run_test(size=(104, 40)) as pilot:
        await wait_for(pilot, app, showing(app, "Pick a number"))
        await pilot.press("enter")
        await wait_for(pilot, app, showing(app, "Buy +15550100001 for"))
        await pilot.press("1", "enter")  # two separate keys: a choice, then a confirmation
        await wait_for(pilot, app, lambda: app.outcome is not None)
    assert got == {"pick": 0, "confirm": 0}


@pytest.mark.asyncio
async def test_stopping_before_a_status_line_exists_is_harmless():
    app = SetupApp(lambda ui: Outcome(ui.choose(LINE, OPTIONS)))
    async with app.run_test(size=(104, 40)) as pilot:
        await wait_for(pilot, app, showing(app, "[3] Back"))
        waiting = PageScreen(Page(3, "Setting up +15550100001"), busy="Buying…")  # not mounted yet: no status line
        waiting.stopping()
        assert waiting.label == "Stopping after this step…"
        app.screen.stopping()  # a menu has no status line either
        await pilot.press("enter")
        await wait_for(pilot, app, lambda: app.outcome is not None)
    assert app.outcome == Outcome(0)


def body_lines(app) -> list[str]:
    body = app.screen.query_one("#body")
    return [body.render_line(y).text for y in range(body.size.height)]


@pytest.mark.asyncio
async def test_a_wide_body_wraps_on_a_narrow_terminal(tmp_path, monkeypatch):
    from ai_phone.setup import flow
    from tests.unit.test_setup_flow import NUMBER_LINE, SIP_LINE, FakeActions
    # Keep this layout fixture independent of checkout/home path lengths and any real saved settings.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("AI_PHONE_HOME", str(tmp_path / ".ai-phone"))
    summary = flow.done_text(FakeActions(exists=True, lines={"twilio": NUMBER_LINE, "agent": SIP_LINE}), flow.Session())
    done = Page(7, "You're set up", "Everything is saved. Here's how to use it.", body=summary)
    app = SetupApp(lambda ui: Outcome(ui.choose(done, [Option("Finish")], back=False)))
    async with app.run_test(size=(70, 30)) as pilot:
        await wait_for(pilot, app, lambda: isinstance(app.screen, PageScreen) and app.screen.is_mounted
                       and app.screen.query_one("#body").size.height > 0)
        await pilot.pause(0.05)
        lines = body_lines(app)
        assert max(len(line.rstrip()) for line in lines) <= 64  # the page's 70 columns less its padding
        assert " ".join(" ".join(lines).split()) == " ".join(summary.split())  # every word is there, none cut
        assert app.screen.query_one("#body").region.right <= 67
        await pilot.press("enter")
        await wait_for(pilot, app, lambda: app.outcome is not None)


def test_a_second_ctrl_c_ends_with_what_the_run_did(monkeypatch, capsys):
    release = threading.Event()

    def flow(ui):
        ui.busy(Page(3, "Setting up +15550100001"), "Buying +15550100001 and connecting it…", lambda: release.wait(5))
        return Outcome(0)

    async def quit_twice(pilot):
        await wait_for(pilot, pilot.app, showing(pilot.app, "Buying +15550100001"))
        await pilot.press("ctrl+c")
        await wait_for(pilot, pilot.app, lambda: "Stopping after this step…" in screen_text(pilot.app))
        await pilot.press("ctrl+c")

    monkeypatch.setattr(tui, "SetupApp", headless(quit_twice))
    assert run_tui(flow, summary=lambda: "what this run did") == Outcome(1, "what this run did")
    release.set()

    go = threading.Event()

    def download(ui):
        ui.progress(Page(6, "Downloading the models"), "Downloading…", lambda report: go.wait(5))
        return Outcome(0)

    async def quit_download(pilot):
        await wait_for(pilot, pilot.app, showing(pilot.app, "Downloading…"))
        await pilot.press("ctrl+c")

    def fake_exit(code):
        raise SystemExit(code)

    monkeypatch.setattr(tui, "SetupApp", headless(quit_download))
    monkeypatch.setattr(tui.os, "_exit", fake_exit)
    with pytest.raises(SystemExit):
        run_tui(download, summary=lambda: "the download and what this run did")
    assert capsys.readouterr().out == "the download and what this run did\n"
    go.set()


def test_run_setup_names_a_purchase_that_a_second_ctrl_c_left(monkeypatch, capsys, tmp_path):
    import types

    from ai_phone.setup import run_setup
    from tests.unit.test_setup_flow import FakeActions
    from tests.unit.test_setup_plain import Terminal

    release = threading.Event()

    class SlowCarrier(FakeActions):
        def buy_number(self, carrier, creds, number):
            release.wait(5)
            return super().buy_number(carrier, creds, number)

    actions = SlowCarrier(exists=True, saved_keys={"TWILIO_ACCOUNT_SID": "ACsid", "TWILIO_AUTH_TOKEN": "tok"})
    monkeypatch.setenv("AI_PHONE_HOME", str(tmp_path))
    monkeypatch.setattr("ai_phone.setup.actions.Actions", lambda path: actions)
    monkeypatch.setattr("ai_phone.setup.sys", types.SimpleNamespace(stdin=Terminal(True), stdout=Terminal(True)))

    async def buy_then_quit_twice(pilot):
        app = pilot.app
        # Add a line, A phone number, Twilio, the saved key, A new number
        for question in ("Give your AI", "Which connection should ai-phone use", "Which carrier?", "Use the saved Twilio key?",
                         "A new number, or one"):
            await wait_for(pilot, app, showing(app, question))
            await pilot.press("enter")
        await wait_for(pilot, app, showing(app, "Where should the number be?"))
        await pilot.press("enter", "enter")
        await wait_for(pilot, app, showing(app, "Pick a number"))
        await pilot.press("enter")
        await wait_for(pilot, app, showing(app, "Buy +15550100001"))
        await pilot.press("1", "enter")
        await wait_for(pilot, app, lambda: "Buying +15550100001 and connecting it…" in screen_text(app))
        await pilot.press("ctrl+c")
        await wait_for(pilot, app, lambda: "Stopping after this step…" in screen_text(app))
        await pilot.press("ctrl+c")

    monkeypatch.setattr(tui, "SetupApp", headless(buy_then_quit_twice))
    assert run_setup() == 1
    assert capsys.readouterr().out == (
        "Setup stopped during: Buying +15550100001 and connecting it…\n"
        "That carrier step may be half done. See your lines with: ai-phone lines list\n"
        "then finish one with: ai-phone lines resume <line>, or undo it with: ai-phone lines remove <line>\n")
    release.set()

"""The setup conversation: which questions to ask, in what order, and what each answer writes. It runs in its own
thread and talks to the person only through a UI (ui.py); everything it changes goes through Actions (actions.py).
A step writes its answers only when the step completes, so leaving part-way never leaves half an answer behind.
The one exception is a number: once it is bought or connected, its line is written at once, because it exists at
the carrier."""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from urllib.parse import urlsplit

from ai_phone import models, paths, voices
from ai_phone.carriers import CARRIERS as CARRIER_INFO, managed_caller_id
from ai_phone.config import E164_RE, HOST_RE, LINE_ID_RE, SIP_USER_RE
from ai_phone.pipeline.prompts import DISCLOSURE
from ai_phone.services.backends.cli_brain import CLAUDE_MODELS
from ai_phone.setup.actions import CARRIER_KEYS, INSTALL, LOGIN, LOOPBACK, ActionError, LineResult
from ai_phone.setup.ui import Back, Cancel, Check, Field, Option, Outcome, Page

WELCOME, MODE, BRAIN, LINE, VOICE, CALLERS, CHECK, DONE = range(8)
BRAIN_KINDS = ("local", "claude", "chatgpt", "openai_compatible")  # in the order the Brain menu offers them


@dataclass(frozen=True)
class Plan:
    """A subscription brain in the words the pages use."""
    name: str          # the plan: "Claude", "ChatGPT"
    cli: str           # the CLI it is reached through: "Claude Code", "Codex"
    label: str         # the Brain menu's option
    hint: str


PLANS = {"claude": Plan("Claude", "Claude Code", "My Claude subscription", "uses the Claude Code login · no API key"),
         "chatgpt": Plan("ChatGPT", "Codex", "My ChatGPT subscription", "uses the Codex login · no API key")}
SAMPLE = "Hi, this is your agent. This is how I sound on the phone."
CARRIERS = tuple((key, value.label) for key, value in CARRIER_INFO.items() if key != "fake")
VOICE_GROUPS = (("af", "American · female"), ("am", "American · male"), ("bf", "British · female"), ("bm", "British · male"))
SPEECH_ENGINES = ("phonon", "kokoro")
FORWARD_WHEN = ("no-answer", "busy", "unreachable", "all")
CALLER_ID_TRIES = 3        # codes the carrier lets a person type, as `ai-phone lines verify-caller-id` does
CALLER_ID_WAIT_SEC = 120   # how long to wait while the carrier's call asks for the code
CALLER_ID_POLL_SEC = 3
TRANSPORTS = ("udp", "tcp", "tls")
DOWNLOAD_LEFT = "Setup cancelled. The download stops now and picks up where it left off next time."
HALF_DONE = ("That carrier step may be half done. See your lines with: ai-phone lines list\n"
             "then finish one with: ai-phone lines resume <line>, or undo it with: ai-phone lines remove <line>")


@dataclass
class Session:
    """What this run did, so a run that stops early can say so. The full-screen app reads it from another thread."""
    summary: str = ""
    saved: list[str] = field(default_factory=list)  # in words: "the brain", "line twilio (+1555…, …)"
    working: tuple[str, str] | None = None  # the step running now: (its label, "carrier" | "download" | "")
    new_config: bool = False
    caller_id_only: bool = False  # the returning user's caller-ID shortcut uses the Settings step

    def note(self, what: str) -> None:
        if what not in self.saved:
            self.saved.append(what)

    @property
    def left_download(self) -> bool:
        return self.working is not None and self.working[1] == "download"


def saved_text(session: Session) -> str:
    done = ", ".join(session.saved)
    return f"Saved in this run: {done}. Nothing else was changed." if done else "Nothing was changed."


def cancel_text(session: Session) -> str:
    """What a run that stopped early tells the person: what it saved, and what the step it stopped in may have left."""
    label, kind = session.working or ("", "")
    done = ", ".join(session.saved)
    if kind == "carrier":
        return "\n".join([f"Setup stopped during: {label}", HALF_DONE] + ([f"Saved in this run before that: {done}."] if done else []))
    if kind == "download":
        return DOWNLOAD_LEFT + (f" Saved in this run: {done}." if done else "")
    return (f"Setup stopped during: {label}\n" if label else "Setup cancelled. ") + saved_text(session)


def run(ui, actions, session: Session | None = None) -> Outcome:
    session = Session() if session is None else session
    try:
        session.new_config = not actions.current().exists
        plan = welcome(ui, actions, session)
        i = 0
        while i < len(plan):
            ui.begin(plan[i])
            try:
                STEP[plan[i]](ui, actions, session)
                i += 1
            except Back as b:
                if not b.at_start:
                    continue  # a later question in the step: start the step again
                if i == 0:
                    plan, i = welcome(ui, actions, session), 0
                else:
                    i -= 1
        return Outcome(0, session.summary or done_text(actions, session))
    except (Cancel, KeyboardInterrupt):  # ctrl+c reaches plain mode as KeyboardInterrupt, wherever the flow is
        return Outcome(1, cancel_text(session))
    except ActionError as e:  # every step shows its own; this is one no step expected (a config that broke meanwhile)
        return Outcome(3, f"Setup stopped: {e}\n{saved_text(session)}")
    except Exception as e:  # a bug: say so without a traceback (or a message that might hold more than it should)
        return Outcome(3, f"Setup stopped: {type(e).__name__}\n{saved_text(session)}")
    finally:
        ui.close()


def busy(ui, session: Session, page: Page, label: str, work, kind: str = ""):
    """ui.busy, with the step noted as running while its work runs (a stop in the middle then names it)."""
    return ui.busy(page, label, lambda: _running(session, label, kind, work))


def progress(ui, session: Session, page: Page, label: str, work):
    return ui.progress(page, label, lambda report: _running(session, label, "download", work, report))


def _running(session: Session, label: str, kind: str, work, *args):
    session.working = (label, kind)
    try:
        result = work(*args)
    except ActionError:  # the step ended, and the flow carries on
        session.working = None
        raise
    session.working = None  # a stop (Cancel, KeyboardInterrupt) or a bug leaves it noted
    return result


def ask(ui, page: Page, fields: list[Field]) -> list[str]:
    """ui.ask, with the spaces around typed text removed. Secrets are kept exactly as typed."""
    return [v if f.secret else v.strip() for v, f in zip(ui.ask(page, fields), fields, strict=True)]


# ---- answers ---------------------------------------------------------------------------------

def required(v: str) -> str | None:
    return None if v.strip() else "Enter a value."


def e164(v: str) -> str | None:
    return None if E164_RE.fullmatch(v.strip()) else "Use international format, like +15550100000."


def six_digits(v: str) -> str | None:
    return None if re.fullmatch(r"\d{6}", v.strip()) else "Enter the 6-digit code."


def country(v: str) -> str | None:
    return None if re.fullmatch(r"[A-Za-z]{2}", v.strip()) else "Use a two-letter country code, like US or GB."


def area(v: str) -> str | None:
    return None if re.fullmatch(r"(\d{2,5})?", v.strip()) else "Use 2 to 5 digits, or leave it empty."


def sip_name(v: str) -> str | None:
    v = v.strip()
    ok = SIP_USER_RE.fullmatch(v) and LINE_ID_RE.fullmatch(v)
    return None if ok else "Use lowercase letters, digits and dashes."


def host(v: str) -> str | None:
    return None if HOST_RE.fullmatch(v.strip()) else "Use this computer's hostname or IP address, like 192.0.2.10."


def url(v: str) -> str | None:
    """http(s) with a host; plain http only to this computer, so a key never crosses the network unencrypted."""
    try:
        u = urlsplit(v.strip())
        where = u.hostname
    except ValueError:
        return "Use a URL that starts with http:// or https://."
    if u.scheme not in ("http", "https") or not where:
        return "Use a URL that starts with http:// or https://."
    if u.scheme == "http" and where not in LOOPBACK:
        return "Use https:// for a server that is not on this computer."
    return None


def speed(v: str) -> str | None:
    try:
        x = float(v)
    except ValueError:
        x = math.nan
    return None if 0.5 <= x <= 2.0 else "Use a number from 0.5 to 2.0."


def count(v: str) -> str | None:
    return None if v.strip().isdecimal() else "Use a whole number: 0 or more."


# ---- words -----------------------------------------------------------------------------------

def price(n) -> str:
    if n.monthly_cost is None:
        return "the carrier's monthly price"
    symbol = {"USD": "$", "GBP": "£", "EUR": "€"}.get((n.currency or "").upper())
    if symbol:
        return f"{symbol}{n.monthly_cost:.2f}/month"
    return f"{n.monthly_cost:.2f} {n.currency}/month" if n.currency else f"{n.monthly_cost:.2f}/month"


def gb(n: int | None) -> str:
    return "several GB" if n is None else f"{n / 1e9:.1f} GB"


def plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def speed_text(x: float) -> str:
    s = f"{x:g}"
    return s if "." in s else f"{s}.0"


def tilde(p: Path) -> str:
    """A path under the home directory, shown with ~ (never expanded)."""
    p, home = Path(p), Path.home()
    return "~/" + str(p.relative_to(home)) if p.is_relative_to(home) and p != home else str(p)


def home_dir() -> str:
    return tilde(paths.home())


def model_name(kind: str, model: str) -> str:
    """What to call a subscription brain's model on a page: Claude's aliases have names, a ChatGPT model is its id."""
    return next((m.name for m in CLAUDE_MODELS if m.id == model), model) if kind == "claude" else model


def install_lines(kind: str) -> str:
    """The two lines that put the CLI there and log it in, shown as a block the person can copy."""
    return f"{INSTALL[kind]}\n{LOGIN[kind]}"


def tier_name(tier: str) -> str:
    """What to call the local brain: its catalog name, or the repo id when the config names one by hand."""
    entry = models.CATALOG.get(tier)
    return entry.name if entry else tier


def model_names(repos: list[str]) -> str:
    names = {models.STT: "Parakeet 0.6B v2", models.TTS: "Kokoro 82M", models.PHONON: "Phonon 31M",
             **{entry.repo: entry.name for entry in models.CATALOG.values()}}
    return "\n".join(names.get(repo, repo) for repo in repos)


def recommended_local_model(mem_gb: int) -> str | None:
    for key in (models.recommended_tier(mem_gb), "qwen2b"):
        if mem_gb > 0 and models.fits(key, mem_gb):
            return key
    return None


def model_hint(entry, mem_gb: int) -> str:
    recommended = "recommended for your RAM · " if entry.key == recommended_local_model(mem_gb) else ""
    more = " · exceeds system RAM" if 0 < mem_gb < entry.needs_gb else ""
    return f"{recommended}needs {entry.needs_gb} GB{more}"


# ---- Welcome -----------------------------------------------------------------------------------

def welcome(ui, actions, session: Session) -> list[int]:
    ui.begin(WELCOME)
    session.caller_id_only = False
    m = actions.machine()
    mem = m.memory_gb
    recommended = recommended_local_model(mem)
    memory = (f"{mem} GB · {tier_name(recommended)} recommended" if recommended else
              f"{mem} GB · below the models' memory requirements" if mem > 0 else "could not detect memory")
    downloaded = model_names(m.models_downloaded) or "none downloaded yet"
    if m.models_missing:
        missing = model_names(m.models_missing).replace("\n", ", ")
        downloaded += f"\nMissing: {missing} · setup can download them"
    checks = [
        Check("SIP library", "built" if m.sip_library else "not built yet · run ./scripts/install.sh", m.sip_library),
        Check("Memory", memory, True if recommended else None),
        Check("Models", downloaded, None if m.models_missing else True),
        Check("Config", f"found · {home_dir()}/ (setup changes only what you pick)" if m.config_exists else f"new · {home_dir()}/", True),
    ]
    page = Page(WELCOME, "Give your AI a phone number", "Calls answered and placed by models running on this computer.",
                note="Picks a line, a brain and a voice, then checks it all works.", checks=checks)
    if not m.config_exists:
        if ui.choose(page, [Option("Continue"), Option("Cancel", "exit setup, nothing is saved")], back=False) == 1:
            raise Cancel()
        return [MODE, BRAIN, LINE, VOICE, CALLERS, CHECK, DONE]
    plans = [[LINE], [BRAIN], [VOICE], [MODE, CALLERS], [CALLERS], [CALLERS], [MODE, BRAIN, LINE, VOICE, CALLERS]]
    i = ui.choose(page, [Option("Add a line", "a phone number, a SIP address or your own trunk"),
                         Option("Change the brain", "on this computer, a plan you pay for, or your own endpoint"),
                         Option("Change the voice", "speech model, voice and speed"),
                         Option("Change call directions", "receive calls, place calls, or both"),
                         Option("Change call settings", "greeting, instructions, daily limit"),
                         Option("Change caller ID", "show your line's number or your own verified number"),
                         Option("Run all steps"),
                         Option("Cancel", "exit setup, nothing is changed")], back=False)
    if i == len(plans):
        raise Cancel()
    session.caller_id_only = i == 5
    return plans[i] + [CHECK, DONE]


# ---- Mode --------------------------------------------------------------------------------------

CALL_MODES = ((True, True, "receive and place calls"),
              (False, True, "place calls only"),
              (True, False, "receive calls only"))


def call_mode(ui, actions, session: Session) -> None:
    cur = actions.current()
    default = next(i for i, (incoming, outgoing, _) in enumerate(CALL_MODES)
                   if (incoming, outgoing) == (cur.allow_inbound, cur.allow_outbound))
    error = ""
    while True:
        default = ui.choose(Page(MODE, "How should ai-phone work?",
                                 "To receive calls, ai-phone must stay running. For outgoing calls only, start it "
                                 "when you need it and stop it afterward. You can change this later.", error=error),
                            [Option("Receive and place calls", "keep ai-phone running to receive calls"),
                             Option("Place calls only", "start ai-phone when you want to call"),
                             Option("Receive calls only", "keep ai-phone running; outgoing calls are blocked")],
                            default=default)
        incoming, outgoing, _ = CALL_MODES[default]
        try:
            actions.save_call_directions(incoming, outgoing)
            session.note("call directions")
            return
        except ActionError as e:
            error = str(e)


# ---- Brain -------------------------------------------------------------------------------------

def brain(ui, actions, session: Session) -> None:
    cur = actions.current()
    kinds = list(BRAIN_KINDS)
    where = ui.choose(Page(BRAIN, "Where should your agent think?",
                           "The model that writes what your agent says. Speech and voice always run on this computer."),
                      [Option("On this computer", "private · nothing leaves this computer"),
                       Option(PLANS["claude"].label, PLANS["claude"].hint),
                       Option(PLANS["chatgpt"].label, PLANS["chatgpt"].hint),
                       Option("My own endpoint", "any OpenAI-compatible URL: LM Studio, llama.cpp, a hosted API")],
                      default=kinds.index(cur.brain_kind) if cur.brain_kind in kinds else 0)
    kind = kinds[where]
    if kind in PLANS:
        return brain_subscription(ui, actions, session, cur, kind)
    if kind == "openai_compatible":
        return brain_endpoint(ui, actions, session, cur)
    mem = actions.machine().memory_gb
    catalog = list(models.CATALOG.values())
    keys = [m.key for m in catalog]
    default = keys.index(cur.tier if cur.brain_kind == "local" and cur.tier in keys else models.recommended_tier(mem))
    recommended = recommended_local_model(mem)
    if recommended and session.new_config and "the brain" not in session.saved:
        default = keys.index(recommended)
    memory = (f"This computer has {mem} GB of memory. Models that need more may swap and answer slowly." if mem > 0 else
              "Memory could not be detected. Compare the model requirements with your system RAM.")
    error = ""
    while True:
        default = ui.choose(Page(BRAIN, "Which model?", memory, error=error),
                            [Option(m.name, model_hint(m, mem)) for m in catalog], default=default)
        error = ""
        chosen = catalog[default]
        if mem > 0 and not models.fits(chosen.key, mem):
            try:
                if ui.choose(Page(BRAIN, f"Use {chosen.name} anyway?", f"It needs {chosen.needs_gb} GB; this computer has "
                                  f"{mem} GB, so it may swap and answer slowly."),
                             [Option("Use it anyway")], default=1) != 0:
                    continue
            except Back:  # Esc or the Back option: choose again, with this model still picked
                continue
        try:
            actions.save_brain_local(chosen.key)
            session.note("the brain")
            return
        except ActionError as e:
            error = str(e)


def brain_subscription(ui, actions, session: Session, cur, kind: str) -> None:
    """A plan the person already pays for: ai-phone drives the CLI they are logged into, so there is no key to type.
    What has to be true is that the CLI is installed and logged in, and the pages say how when it is not.

    The login is established without inference, so the only thing a setup run spends is the one request that proves
    the model they picked answers. That order matters both ways round: it is the truth of "one request per setup"
    (choosing a different model used to cost a second one at the Check step), and it means a model the plan does not
    allow is a sentence on the menu that offers the others, not a page that sends them back to the same failure.
    """
    plan = PLANS[kind]
    while True:
        if not actions.cli_installed(kind):
            try_again(ui, Page(BRAIN, f"{plan.cli} is not installed",
                               f"ai-phone reaches your {plan.name} plan through it. Install it, log in, then check again.",
                               body=install_lines(kind)))
            continue
        waiting = Page(BRAIN, f"Your {plan.name} subscription")
        try:
            busy(ui, session, waiting, f"Checking your {plan.name} login…", lambda: actions.login_status(kind))
            # Claude Code's models are three names it always takes; a ChatGPT plan is asked which ones it allows
            choices = (busy(ui, session, waiting, f"Asking which models your {plan.name} plan allows…",
                            lambda: actions.brain_models(kind)) if kind == "chatgpt" else actions.brain_models(kind))
        except ActionError as e:
            try_again(ui, Page(BRAIN, f"ai-phone cannot use your {plan.name} plan yet",
                               f"It talks to {plan.name} through {plan.cli}, with the login you already have.",
                               body=install_lines(kind), error=str(e)))
            continue
        break
    detail = (f"Logged in. The words of every call go to {plan.name} through {plan.cli}; a faster model starts "
              "talking sooner. Picking one sends a single short request to your plan, to check it answers.")
    default = next((i for i, m in enumerate(choices) if m.id == cur.model), 0) if cur.brain_kind == kind else 0
    error = ""
    while True:
        i = ui.choose(Page(BRAIN, "Which model?", detail, error=error),
                      [Option(m.name, m.note) for m in choices], default=default)
        chosen = choices[i]
        try:
            busy(ui, session, waiting, f"Asking {plan.name} to answer once with {chosen.name}…",
                 lambda: actions.test_subscription(kind, chosen.id))
            actions.save_brain_subscription(kind, chosen.id)
            session.note("the brain")
            return
        except ActionError as e:
            error, default = str(e), i


def try_again(ui, page: Page) -> None:
    """A page with nothing to do but check again. Back (the front-end's own option) starts the step over."""
    ui.choose(page, [Option("Check again")])


def brain_endpoint(ui, actions, session: Session, cur) -> None:
    base_url, model, error = cur.base_url, cur.model, ""
    keep = "leave empty to keep the saved key (same server)" if cur.brain_key_saved else "leave empty for none"
    while True:
        page = Page(BRAIN, "Your endpoint", "Any server that speaks the OpenAI chat API. The key is optional and is saved "
                    f"to {home_dir()}/secrets.env.", error=error)
        base_url, model, key = ask(ui, page, [
            Field("Base URL", base_url, validate=url),
            Field("Model", model, validate=required),
            Field("API key", "", secret=True, placeholder=keep),
        ])
        key = key.strip()  # a pasted key's spaces are never part of it (a trunk password's may be)
        try:
            where = urlsplit(base_url).hostname or base_url
        except ValueError:
            where = base_url
        try:
            busy(ui, session, replace(page, error=""), f"Sending one short test request to {where}…",
                 lambda: actions.test_endpoint(base_url, model, key or None))
            actions.save_brain_endpoint(base_url, model, key or None)
            session.note("the brain")
            return
        except ActionError as e:
            error = str(e)


# ---- Line --------------------------------------------------------------------------------------

def line(ui, actions, session: Session) -> None:
    cur = actions.current()
    i = ui.choose(Page(LINE, "Which connection should ai-phone use?", "You can add more lines later by running setup again."),
                  [Option("A phone number", "Twilio, Telnyx or SignalWire"),
                   Option("A SIP address", "free · softphones and other agents"),
                   Option("My own SIP trunk", "your PBX or provider"),
                   Option("Skip", "keep the lines you have" if cur.lines else "add one later")])
    if i == 0:
        phone_number(ui, actions, session)
    elif i == 1:
        sip_address(ui, actions, session)
    elif i == 2:
        own_trunk(ui, actions, session)


def phone_number(ui, actions, session: Session) -> None:
    carrier, name = CARRIERS[ui.choose(
        Page(LINE, "Which carrier?", "ai-phone uses your own account. Numbers and calls are billed by the carrier."),
        [Option(label, " + ".join(f.label for f in CARRIER_INFO[key].credentials)) for key, label in CARRIERS])]
    if carrier == "signalwire" and ui.choose(
        Page(LINE, "How should SignalWire connect?", "Connect a number already in your SignalWire Space, or enter existing SIP credentials."),
        [Option("Connect with API", "Space + Project ID + API token"),
         Option("Use SIP credentials", "configure a SIP endpoint in the SignalWire dashboard")]) == 1:
        own_trunk(ui, actions, session, preset="signalwire")
        return
    creds = actions.saved_keys(carrier)
    page = Page(LINE, f"Use the saved {name} key?", f"It is in {home_dir()}/secrets.env.")
    if creds and ui.choose(page, [Option("Use the saved key"), Option("Enter a different key", "replaces the saved key")]) == 1:
        creds = None
    owned, error = None, ""
    while owned is None:
        if not creds:
            page = Page(LINE, f"Your {name} credentials", f"Saved to {home_dir()}/secrets.env, readable only by you. "
                        "Never shown or sent anywhere else.", error=error)
            typed = [v.strip() for v in ask(ui, page, credential_fields(carrier))]  # pasted keys: spaces are not part of them
            creds = dict(zip(CARRIER_KEYS[carrier], typed, strict=True))
            page = replace(page, error="")
        try:
            owned = busy(ui, session, page, f"Checking your {name} account…", lambda: actions.owned_numbers(carrier, creds))
        except ActionError as e:
            creds, error = None, str(e)
    # Existing lines stay hidden. SignalWire can finish a matching pending attachment using its original journal;
    # purchases, other carriers and ambiguous/failed cleanup journals still belong to lines resume/remove.
    unfinished = actions.unfinished_numbers()
    resumable = actions.resumable_numbers(carrier) if carrier == "signalwire" else {}
    connected = {ln.get("number") for ln in actions.current().lines.values()}
    free = [n for n in owned if n.e164 not in connected and (n.e164 not in unfinished or n.e164 in resumable)]
    left = f"{len(free)} not yet connected" if free else "none left to connect" if owned else "none on this account"
    if not CARRIER_INFO[carrier].purchase:
        new = False
        if not free:
            if any(n.e164 in unfinished for n in owned):
                page = Page(LINE, "An earlier setup needs attention",
                            "Finish or undo the unfinished setup before connecting this number. Use the original carrier "
                            "account. A failed cleanup must be completed before retrying.",
                            body="See the line and its state: ai-phone lines list\n"
                                 "To resume a SignalWire line:\n"
                                 "ai-phone lines resume <line> --key-env SIGNALWIRE_PROJECT_ID "
                                 "--secret-env SIGNALWIRE_API_TOKEN\n"
                                 "To undo a SignalWire setup:\n"
                                 "ai-phone lines remove <line> --key-env SIGNALWIRE_PROJECT_ID "
                                 "--secret-env SIGNALWIRE_API_TOKEN\n"
                                 "The saved SIGNALWIRE_SPACE is used; other carriers need their own credentials.")
            elif owned:
                page = Page(LINE, f"No unconnected numbers on this {name} account",
                            "These numbers are already configured. Check them with: ai-phone lines list")
            else:
                page = Page(LINE, f"No unconnected numbers on this {name} account",
                            f"Buy a voice-capable number in your {name} Space, then run setup again to connect it. "
                            "Number purchases and pricing are managed in the SignalWire dashboard.")
            ui.choose(page, [Option("Continue")])
            return
    else:
        new = ui.choose(Page(LINE, f"A new number, or one already on your {name} account?",
                             checks=[Check(f"{name} key", f"works · {plural(len(owned), 'number')} on this account", True)]),
                        [Option("A new number", "search and buy one"), Option(f"One already on my {name} account", left)]) == 0
        if not new and not free:
            ui.choose(Page(LINE, f"No other numbers on this {name} account" if owned else f"No numbers on this {name} account",
                           "A number on your phone's SIM cannot be connected; buy one here, or add a number in the "
                           f"{name} console and run setup again."),
                      [Option("Buy a new number")])
            new = True
    if new:
        result, n = buy_number(ui, actions, session, carrier, name, creds)
    else:
        result, n = attach_number(ui, actions, session, carrier, name, creds, free, resumable=resumable)
    if result.line_id:  # written, or at least journaled: a stop from here on must name it
        session.note(line_note(result, n.e164, name, bought=new))
    if result.code != 0:
        ui.choose(Page(LINE, "The number is not ready yet" if result.code == 2 else "Something went wrong",
                       body="\n".join(result.messages)), [Option("Continue")], back=False)
        return
    cur = actions.current()
    mobile = forward(ui, actions, session, result.line_id) if cur.allow_inbound else ""
    if cur.allow_outbound:
        caller_id(ui, actions, session, carrier, name, creds, result.line_id, n.e164, mobile)


def line_note(result: LineResult, number: str, carrier: str, bought: bool) -> str:
    line_id = result.line_id
    if result.code != 0:
        return (f"line {line_id} ({number}, not finished: ai-phone lines resume {line_id}, "
                f"or ai-phone lines remove {line_id})")
    return f"line {line_id} ({number}, a {carrier} number billed monthly)" if bought else f"line {line_id} ({number})"


def credential_fields(carrier: str) -> list[Field]:
    return [Field(field.label, secret=field.secret, placeholder=field.placeholder, validate=required)
            for field in CARRIER_INFO[carrier].credentials]


def buy_number(ui, actions, session: Session, carrier: str, name: str, creds: dict):
    where, found, error = "US", [], ""
    while not found:
        page = Page(LINE, "Where should the number be?", "A two-letter country code, and an area code if you want one.", error=error)
        where, code = ask(ui, page, [
            Field("Country", where, validate=country),
            Field("Area code", "", placeholder="optional", validate=area),
        ])
        where = where.upper()
        try:
            found = busy(ui, session, replace(page, error=""), "Looking for numbers…",
                         lambda: actions.search_numbers(carrier, creds, where, code or None))
            error = "" if found else "No numbers found there. Try another area code, or leave it empty."
        except ActionError as e:
            error = str(e)
    n = found[ui.choose(Page(LINE, "Pick a number", f"Prices are {name}'s monthly price for the number. Calls are billed separately."),
                        [Option(n.e164, price(n) + (" · documents required" if n.requires_regulatory else "")) for n in found])]
    if ui.choose(Page(LINE, f"Buy {n.e164} for {price(n)}?",
                      f"{name} bills your account every month until you remove the line (ai-phone lines remove)."),
                 [Option(f"Buy {n.e164}", price(n))], default=1) != 0:  # Enter means Back: only choosing Buy buys
        raise Back()
    return run_line_action(ui, session, Page(LINE, f"Setting up {n.e164}"), f"Buying {n.e164} and connecting it…",
                           lambda: actions.buy_number(carrier, creds, n)), n


def attach_number(ui, actions, session: Session, carrier: str, name: str, creds: dict, owned: list,
                  *, resumable: dict[str, str] | None = None):
    resumable = resumable or {}
    shown = owned[:20]
    options = [Option(f"Resume {n.e164}", f"unfinished line {resumable[n.e164]}") if n.e164 in resumable
               else Option(n.e164) for n in shown]
    n = shown[ui.choose(Page(LINE, "Which number?", f"The numbers on your {name} account."), options)]
    resume = n.e164 in resumable
    verb = "Resume" if resume else "Connect"
    routing = ("Its incoming calls will ring ai-phone." if actions.current().allow_inbound else
               "Its incoming calls will be routed to ai-phone and rejected while receiving is disabled.")
    detail = (f"Continue setting up line {resumable[n.e164]} with its saved SIP credentials. " if resume else "")
    question = f"Resume setup for {n.e164}?" if resume else f"Connect {n.e164} to ai-phone?"
    if ui.choose(Page(LINE, question, detail + f"{routing} Nothing is bought. "
                      "Removing the line later disconnects it and keeps the number."),
                 [Option(f"{verb} {n.e164}", "free")], default=1) != 0:  # Enter means Back
        raise Back()
    return run_line_action(ui, session, Page(LINE, f"Setting up {n.e164}"),
                           f"Resuming {n.e164}…" if resume else f"Connecting {n.e164}…",
                           lambda: actions.attach_number(carrier, creds, n)), n


def run_line_action(ui, session: Session, page: Page, label: str, work) -> LineResult:
    try:
        return busy(ui, session, page, label, work, kind="carrier")
    except ActionError as e:  # the line id, when one was chosen and its journal kept
        return LineResult(getattr(e, "line_id", ""), 3, [str(e)])


def forward(ui, actions, session: Session, line_id: str) -> str:
    """The line exists now, so Back inside these questions (or a forward that cannot be saved) returns to the first
    one, never to the purchase. Returns the mobile number typed, or "" when forwarding was skipped."""
    error = ""
    while True:
        try:
            if ui.choose(Page(LINE, "Also forward your own mobile to it?",
                              "Calls to your mobile can ring your agent when you are busy or don't answer. "
                              "Your mobile carrier's divert charges apply.", error=error),
                         [Option("No"), Option("Yes")], back=False) == 0:
                return ""
            (mobile,) = ask(ui, Page(LINE, "Your mobile number", "In international format, like +447700900123."),
                            [Field("Mobile number", validate=e164)])
            when = FORWARD_WHEN[ui.choose(Page(LINE, "When should calls go to your agent?"),
                                          [Option("When I don't answer"), Option("When I'm busy"),
                                           Option("When my phone is off"), Option("All calls")])]
            codes = actions.forward(line_id, mobile, when)
            session.note(f"call forwarding from {mobile} to line {line_id}")
            break
        except Back:
            error = ""
        except ActionError as e:
            error = str(e)
    rows = [("Turn on", codes["activate"]), ("Check", codes["check"]), ("Turn off", codes["cancel"]),
            ("Turn off every divert", codes["cancel_all"])]
    body = "\n".join(f"{label.ljust(21)}   {code}" for label, code in rows)
    ui.choose(Page(LINE, "Dial this on your mobile", "Open the phone app, type the code and press call.", body=body),
              [Option("Done")], back=False)
    return mobile


def caller_id(ui, actions, session: Session, carrier: str, name: str, creds: dict, line_id: str, number: str,
              mobile: str) -> None:
    """Prove to the carrier that the person's own mobile is theirs, so the agent's outgoing calls show it instead of
    the line's number. The line exists now, so Back returns to the first question here, never to the purchase."""
    error = ""
    while True:
        if ui.choose(Page(LINE, "Show your own number when the agent calls out?",
                          f"People you call see your mobile number instead of {number}. {name} sends one code to "
                          "your phone to prove it is yours; no paperwork.", error=error),
                     [Option("No"), Option("Yes")], back=False) == 0:
            return
        error = ""
        try:
            verify_caller_id(ui, actions, session, carrier, name, creds, line_id, number, mobile)
        except Back:
            continue
        except ActionError as e:
            error = str(e)
            continue
        return


def verify_caller_id(ui, actions, session: Session, carrier: str, name: str, creds: dict, line_id: str,
                     current: str, mobile: str = "", stage: int = LINE) -> None:
    """Shared verification for a newly connected number and the existing-line settings shortcut."""
    (mine,) = ask(ui, Page(stage, "Your mobile number", "In international format, like +447700900123."),
                  [Field("Mobile number", mobile, validate=e164)])
    step = busy(ui, session, Page(stage, f"Verifying {mine}"), f"Asking {name} to verify {mine}…",
                lambda: actions.start_caller_id(carrier, creds, mine, line_id=line_id)
                if CARRIER_INFO[carrier].managed_caller_id else actions.start_caller_id(carrier, creds, mine))
    verified = (typed_code(ui, actions, session, carrier, creds, step, name, stage) if step.kind == "enter_code_in_cli"
                else keyed_code(ui, actions, session, carrier, creds, step, name, stage))
    if verified:
        try:
            if CARRIER_INFO[carrier].managed_caller_id:
                busy(ui, session, Page(stage, "Updating the caller ID"), f"Setting {name} caller ID to {mine}…",
                     lambda: actions.record_caller_id(line_id, mine, carrier, creds), kind="carrier")
            else:
                actions.record_caller_id(line_id, mine, carrier)
            session.note(f"line {line_id} calls out as {mine}")
        except ActionError as e:  # the carrier has accepted it by now: never lose that quietly
            recovery = (f"Finish the caller-ID change with: ai-phone lines resume {line_id} "
                        "--key-env SIGNALWIRE_PROJECT_ID --secret-env SIGNALWIRE_API_TOKEN --space <space-host>"
                        if CARRIER_INFO[carrier].managed_caller_id else
                        f"{mine} is verified with {name}; run: ai-phone lines verify-caller-id "
                        f"{line_id} {mine} --key-env <NAME> to record it.")
            ui.choose(Page(stage, f"{mine} is verified, but not saved", recovery, error=str(e)),
                      [Option("Continue")], back=False)
    else:
        ui.choose(Page(stage, "Verification did not complete",
                       f"Your agent still calls out as {current}. To try again: "
                       f"ai-phone lines verify-caller-id {line_id} {mine} --key-env <NAME>"),
                  [Option("Continue")], back=False)


def typed_code(ui, actions, session: Session, carrier: str, creds: dict, step, name: str, stage: int = LINE) -> bool:
    """The carrier texted a code: up to three tries at it, as the CLI's verify-caller-id allows."""
    error = ""
    for _ in range(CALLER_ID_TRIES):
        (typed,) = ask(ui, Page(stage, "Enter the code", step.instructions, error=error),
                       [Field("Code", placeholder="6 digits", validate=six_digits)])
        if busy(ui, session, Page(stage, "Enter the code", step.instructions), f"Asking {name} to check the code…",
                lambda: actions.check_caller_id(carrier, creds, step, typed)):
            return True
        error = "That code was not accepted."
    return False


def keyed_code(ui, actions, session: Session, carrier: str, creds: dict, step, name: str, stage: int = LINE) -> bool:
    """The carrier is calling the phone and the instructions hold the code to key in; wait for it to accept."""
    def wait() -> bool:
        deadline = time.monotonic() + CALLER_ID_WAIT_SEC
        while True:
            if actions.check_caller_id(carrier, creds, step, None):
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(CALLER_ID_POLL_SEC)

    return busy(ui, session, Page(stage, f"{name} is calling you now", body=step.instructions),
                "Waiting for you to enter the code on the call… (up to 2 minutes)", wait)


def free_sip_name(lines: dict[str, dict]) -> str:
    """agent, agent-2, …: the first name that is neither a line id nor the user of a SIP address."""
    taken = set(lines)
    for ln in lines.values():
        if ln.get("kind") == "sip":
            taken.add(ln.get("user"))
            taken.add(str(ln.get("aor") or "").removeprefix("sip:").split("@", 1)[0])
    name, n = "agent", 2
    while name in taken:
        name, n = f"agent-{n}", n + 1
    return name


def sip_address(ui, actions, session: Session) -> None:
    name, address, error = free_sip_name(actions.current().lines), actions.lan_address(), ""
    while True:
        name, address = ask(ui, Page(LINE, "Name your SIP address", "Softphones and other agents call sip:<name>@<this computer>. "
                                     "Free; no carrier.", error=error),
                            [Field("Name", name, validate=sip_name), Field("This computer's address", address, validate=host)])
        try:
            session.note(f"line {actions.add_sip_line(name, address)} (sip:{name}@{address})")
            return
        except ActionError as e:
            error = str(e)


def own_trunk(ui, actions, session: Session, *, preset: str | None = None) -> None:
    from_user = "account" if preset == "signalwire" else ("number", "account")[ui.choose(Page(LINE, "What should outgoing SIP From use?",
        "Use the identity your PBX or provider requires. With a SIP username, the provider controls the displayed caller ID."),
        [Option("Phone number", "the line's number or a verified caller ID"),
         Option("SIP username", "the same account username used to register")])]
    registrar = user = number = error = ""
    while True:
        detail = ("Use the SIP domain and endpoint credentials from your SignalWire dashboard, not your API token. "
                  "Set the endpoint caller ID (send_as) in that dashboard." if preset else
                  "The details your PBX or provider gave you.")
        registrar, user, password, number = ask(ui, Page(LINE, "Your SignalWire SIP trunk" if preset else "Your SIP trunk",
                                                         detail + f" The password is saved to {home_dir()}/secrets.env.", error=error),
                                                [Field("Registrar", registrar, placeholder="sip:sip.example.com", validate=required),
                                                 Field("Username", user, validate=required),
                                                 Field("Password", secret=True, validate=required),
                                                 Field("Your number on it" + (" (optional)" if from_user == "account" else ""),
                                                       number, placeholder="+15550100000",
                                                       validate=(lambda value: e164(value) if value else None)
                                                       if from_user == "account" else e164)])
        transport = TRANSPORTS[ui.choose(Page(LINE, "Which transport?",
                    "Match the SignalWire endpoint settings; TLS uses encrypted signalling." if preset else
                    "Use what your provider says; UDP is the usual one."),
                    [Option("UDP"), Option("TCP"), Option("TLS", "encrypted signalling")], default=2 if preset else 0)]
        srtp = "off"
        if preset and transport == "tls":
            srtp = ("required", "optional", "off")[ui.choose(Page(LINE, "Media encryption",
                    "Match the endpoint encryption setting in SignalWire."),
                    [Option("Required", "SRTP"), Option("Optional", "accept SRTP or plain audio"), Option("Off")])]
        uri = registrar if registrar.startswith(("sip:", "sips:")) else f"sip:{registrar}"
        try:
            if preset:
                line_id = actions.add_trunk_line(uri, user, password, number, transport, from_user=from_user,
                                                preset=preset, srtp=srtp)
            else:
                line_id = actions.add_trunk_line(uri, user, password, number, transport, from_user=from_user)
            session.note(f"line {line_id}" + (f" ({number})" if number else ""))
            return
        except ActionError as e:
            error = str(e)


# ---- Voice -------------------------------------------------------------------------------------

def english_voices(actions) -> list[str]:
    return [v for v in actions.voices() if v[:2] in dict(VOICE_GROUPS)]


def voice(ui, actions, session: Session) -> None:
    cur = actions.current()
    default = 0 if session.new_config and "the voice" not in session.saved else SPEECH_ENGINES.index(cur.tts_engine)
    engine = SPEECH_ENGINES[ui.choose(
        Page(VOICE, "Which speech model?", "Both run on this computer. You can hear a sample before saving."),
        [Option("Phonon", "recommended for new setups · one English voice"),
         Option("Kokoro", "American and British voices · adjustable speed")], default=default)]
    if engine == "phonon":
        phonon_voice(ui, actions, session)
        return
    # A switch from Phonon starts at Kokoro's normal speed and default voice.
    if cur.tts_engine != "kokoro":
        cur = replace(cur, voice=voices.DEFAULT, speed=1.0)
    names, error = english_voices(actions), ""
    while not names:
        i = ui.choose(Page(VOICE, "How should your agent sound?",
                           "The speech model is not downloaded yet, so there are no voices to pick from.", error=error),
                      [Option("Download it now", "about 0.4 GB"), Option("Keep the default voice", "Heart · American · female")])
        try:
            if i == 1:
                actions.save_voice(voices.DEFAULT, 1.0)
                session.note("the voice")
                return
            progress(ui, session, Page(VOICE, "Downloading the speech model"), "Downloading…", actions.pull_voice_model)
            session.note("the downloaded speech model")
            names = english_voices(actions)
            error = "" if names else "The speech model downloaded, but it has no voices to pick from. Try the download again."
        except ActionError as e:
            error = str(e)
    groups = [(label, [v for v in names if v.startswith(prefix)]) for prefix, label in VOICE_GROUPS]
    groups = [(label, vs) for label, vs in groups if vs]
    default = next((i for i, (_, vs) in enumerate(groups) if cur.voice in vs), 0)
    label, group = groups[ui.choose(
        Page(VOICE, "How should your agent sound?", "English voices that came with the speech model. You can listen before you pick."),
        [Option(label, plural(len(vs), "voice")) for label, vs in groups], default=default)]
    v = group[ui.choose(Page(VOICE, label, "Pick one to hear it or use it."),
                        [Option(voices.title(v), "current" if v == cur.voice else "") for v in group],
                        default=group.index(cur.voice) if cur.voice in group else 0)]
    error = ""
    while True:
        page = Page(VOICE, voices.title(v), voices.describe(v), error=error)
        if ui.choose(page, [Option("Use this voice"), Option("Hear it", "plays a short sample on this computer's speakers")]) == 0:
            break
        try:
            busy(ui, session, replace(page, error=""), "Playing…", lambda: actions.play_sample(v, cur.speed, SAMPLE))
            error = ""
        except ActionError as e:
            error = str(e)
    value, error = speed_text(cur.speed), ""
    while True:
        (value,) = ask(ui, Page(VOICE, "How fast should it talk?", "1.0 is normal. 0.9 is a little slower, 1.1 a little faster.",
                                error=error), [Field("Speed", value, validate=speed)])
        try:
            actions.save_voice(v, float(value))
            session.note("the voice")
            return
        except ActionError as e:
            error = str(e)


def phonon_voice(ui, actions, session: Session) -> None:
    name, error = voices.default_voice("phonon"), ""
    while not actions.voices(engine="phonon"):
        i = ui.choose(Page(VOICE, "Download Phonon?", "The model includes its English voice. You can download it now to hear a sample.",
                           error=error),
                      [Option("Download it now"), Option("Use the bundled voice", "download at the Check step")])
        try:
            if i == 1:
                actions.save_voice(name, 1.0, engine="phonon")
                session.note("the voice")
                return
            progress(ui, session, Page(VOICE, "Downloading Phonon"), "Downloading…",
                     lambda report: actions.pull_voice_model(report, engine="phonon"))
            session.note("the downloaded speech model")
            error = "The model downloaded, but its bundled voice is missing. Try the download again."
        except ActionError as e:
            error = str(e)
    error = ""
    while True:
        page = Page(VOICE, "Phonon voice", "One bundled English voice at its natural speed.", error=error)
        i = ui.choose(page, [Option("Use this voice"), Option("Hear it", "plays a short sample on this computer's speakers")])
        try:
            if i == 0:
                actions.save_voice(name, 1.0, engine="phonon")
                session.note("the voice")
                return
            busy(ui, session, replace(page, error=""), "Playing…",
                 lambda: actions.play_sample(name, 1.0, SAMPLE, engine="phonon"))
            error = ""
        except ActionError as e:
            error = str(e)


# ---- Callers -----------------------------------------------------------------------------------

def change_caller_id(ui, actions, session: Session) -> None:
    cur = actions.current()
    if not cur.allow_outbound:
        ui.choose(Page(CALLERS, "Outgoing calls are off",
                       "Choose Change call directions in setup to allow outgoing calls before choosing a caller ID."),
                  [Option("Continue")])
        return
    lines = [(line_id, ln) for line_id, ln in cur.lines.items()
             if ln.get("kind", "trunk") == "trunk" and (ln.get("number") or ln.get("from_user") == "account")
             and (ln.get("provider") is None or provider_of(ln).get("state", "pending") == "ready")]
    if not lines:
        ui.choose(Page(CALLERS, "No phone-number lines are ready",
                       "Add a phone-number line first, or finish setting up its pending order. "
                       "SIP addresses do not have a phone-number caller ID."), [Option("Continue")])
        return
    line_id, ln = lines[ui.choose(Page(CALLERS, "Which line's caller ID?",
                                      "Choose the line whose outgoing number you want to change."),
                                 [Option(f"{line_id} · {ln.get('number') or ln.get('user', 'SIP account')}",
                                         "caller ID set by your provider" if ln.get("from_user") == "account" and not managed_caller_id(ln) else
                                         f"currently shows {ln.get('caller_id') or ln['number']}")
                                  for line_id, ln in lines])]
    if ln.get("from_user") == "account" and not managed_caller_id(ln):
        ui.choose(Page(CALLERS, "Your provider controls this caller ID",
                       "This line sends its SIP account username in From. Set the displayed phone number with your provider; "
                       "changing or verifying a number here would not change outgoing calls."), [Option("Continue")])
        return
    number, current = ln["number"], ln.get("caller_id") or ln["number"]
    carrier = provider_of(ln).get("name", "")
    name = dict(CARRIERS).get(carrier)
    verified = dict.fromkeys(v["number"] for v in (ln.get("verified_caller_ids") or [])
                             if v.get("provider", "") == carrier and v.get("number") and v["number"] != number)
    numbers = [number, *verified]
    options = [Option(f"Line's number · {number}", "the number on this line"),
               *[Option(f"Your verified number · {n}", "already verified for this line") for n in verified]]
    if name:
        options.append(Option("Verify another number you own", f"{name} will text or call you with a verification code"))
    default = numbers.index(current) if current in numbers else 0
    error = ""
    while True:
        detail = f"Currently shows {current}. Your carrier controls what the recipient sees."
        if not name:
            detail += " To verify another number on this trunk, contact your provider."
        selected = ui.choose(Page(CALLERS, "Which number should people see?", detail, error=error), options, default=default)
        error = ""
        try:
            creds = None
            if selected >= len(numbers) or managed_caller_id(ln):
                creds = caller_id_credentials(ui, actions, session, carrier, name, number)
            if selected < len(numbers):
                if managed_caller_id(ln):
                    busy(ui, session, Page(CALLERS, "Updating the caller ID"), f"Setting {name} caller ID…",
                         lambda: actions.use_caller_id(line_id, numbers[selected], creds), kind="carrier")
                else:
                    actions.use_caller_id(line_id, numbers[selected])
                session.note(f"line {line_id} calls out as {numbers[selected]}")
            else:
                verify_caller_id(ui, actions, session, carrier, name, creds, line_id, current, stage=CALLERS)
            return
        except Back:
            continue
        except ActionError as e:
            error = str(e)


def caller_id_credentials(ui, actions, session: Session, carrier: str, name: str, number: str) -> dict:
    creds = actions.saved_keys(carrier)
    if creds and ui.choose(Page(CALLERS, f"Use the saved {name} key?", "Use the account this line belongs to."),
                           [Option("Use the saved key"), Option("Enter a different key")]) == 1:
        creds = None
    if not creds:
        typed = [v.strip() for v in ask(ui, Page(CALLERS, f"Your {name} credentials",
                                               "Use the account this line belongs to. These credentials are used only "
                                               "for this caller-ID change."), credential_fields(carrier))]
        creds = dict(zip(CARRIER_KEYS[carrier], typed, strict=True))
    owned = busy(ui, session, Page(CALLERS, f"Checking your {name} account"), f"Checking your {name} account…",
                 lambda: actions.owned_numbers(carrier, creds))
    if not any(n.e164 == number for n in owned):
        raise ActionError(f"Could not find {number} on this {name} account. "
                          "Enter credentials for the account this line belongs to.")
    return creds


def callers(ui, actions, session: Session) -> None:
    if session.caller_id_only:
        return change_caller_id(ui, actions, session)
    cur = actions.current()
    greeting, instructions, limit, error = cur.greeting, cur.instructions, str(cur.daily_limit), ""
    while True:
        fields = []
        if cur.allow_inbound:
            fields += [Field("Greeting", greeting, validate=required), Field("Instructions", instructions, validate=required)]
        if cur.allow_outbound:
            fields.append(Field("Outbound calls per day", limit, placeholder="0 means no limit", validate=count))
        question = "What should callers hear?" if cur.allow_inbound else "How many calls may ai-phone place?"
        detail = ("The greeting plays when ai-phone answers. The instructions tell ai-phone's own model what to do "
                  "when your AI isn't listening." if cur.allow_inbound else
                  "Set a daily limit for outgoing calls. Use 0 for no limit.")
        values = ask(ui, Page(CALLERS, question, detail, error=error), fields)
        if cur.allow_inbound:
            greeting, instructions = values[:2]
        if cur.allow_outbound:
            limit = values[-1]
        try:
            actions.save_callers(greeting, instructions, int(limit))
            session.note("call settings")
            return
        except ActionError as e:
            error = str(e)


# ---- Check -------------------------------------------------------------------------------------

def check(ui, actions, session: Session) -> None:
    while True:
        rows: list[Check] = []
        rows.append(check_models(ui, actions, session, checking(rows)))
        error = actions.check_config()
        rows.append(Check("Config", "loads", True) if error is None else Check("Config", error, False))
        cur = actions.current()
        if cur.brain_kind in PLANS:
            rows.append(check_brain(ui, actions, session, checking(rows), cur))
        if error is None:
            rows.extend(check_lines(ui, actions, session, checking(rows), cur))
        if actions.voices(engine=cur.tts_engine):  # the selected speech model is here
            try:
                busy(ui, session, checking(rows), "Playing your agent's voice…",
                     lambda: actions.play_sample(cur.voice, cur.speed, DISCLOSURE, engine=cur.tts_engine))
                rows.append(Check("Voice", f"{voices.title(cur.voice, engine=cur.tts_engine)} · played", True))
            except ActionError as e:
                rows.append(Check("Voice", str(e), False))
        fine = all(r.ok is True for r in rows)
        if ui.choose(Page(CHECK, "Everything works" if fine else "A few things need attention", checks=list(rows)),
                     [Option("Continue"), Option("Check again")]) == 0:
            return


def checking(rows: list[Check]) -> Page:
    """The page shown while a check runs: the rows found so far."""
    return Page(CHECK, "Checking that it all works", checks=list(rows))


def check_brain(ui, actions, session: Session, page: Page, cur) -> Check:
    """The plan answers, with the login the person already has. One tiny turn, and only if this run has not used one."""
    plan = PLANS[cur.brain_kind]
    try:
        probe = busy(ui, session, page, f"Checking your {plan.name} login…",
                     lambda: actions.test_subscription(cur.brain_kind))
    except ActionError as e:
        return Check("Brain", str(e), False)
    return Check("Brain", f"{plan.name} ({model_name(cur.brain_kind, probe['model'])}) · logged in", True)


def check_models(ui, actions, session: Session, page: Page) -> Check:
    cur = actions.current()
    downloaded = model_names(models.needed(cur.tier, cur.brain_kind, cur.tts_engine))
    if not actions.missing_models():
        return Check("Models", downloaded, True)
    size = busy(ui, session, page, "Checking the download size…", actions.download_size)
    later = ui.choose(Page(CHECK, "Download the models now?", f"About {gb(size)}, saved in the Hugging Face cache. "
                           "ai-phone needs them to handle calls."),
                      [Option("Download now", gb(size)), Option("Later", "run: ai-phone models pull")]) == 1
    if later:
        return Check("Models", "not downloaded yet · ai-phone models pull", None)
    try:
        progress(ui, session, Page(CHECK, "Downloading the models", "You can leave this running. It picks up where it "
                                   "stopped if interrupted."), "Downloading…", actions.pull_models)
        session.note("the downloaded models")
        return Check("Models", downloaded, True)
    except ActionError as e:
        return Check("Models", str(e), False)


def check_lines(ui, actions, session: Session, page: Page, cur) -> list[Check]:
    found = busy(ui, session, page, "Checking your lines… (up to 20 seconds)", actions.check_lines) if cur.lines else []
    if not found:
        return [Check("Lines", "none yet · run setup again to add one", None)]
    ok = {"registered": True, "ready": True, "registering": None, "skipped": None}
    return [Check("Lines" if line_id == "lines" else f"Line {line_id}", state + (f" · {detail}" if detail else ""), ok.get(state, False))
            for line_id, state, detail in found]


# ---- Done --------------------------------------------------------------------------------------

def done(ui, actions, session: Session) -> None:
    session.summary = done_text(actions, session)
    try:
        ui.choose(Page(DONE, "You're set up", "Everything is saved. Here's how to use it.", body=session.summary),
                  [Option("Finish")], back=False)
    except Cancel:
        pass  # everything is saved already: leaving here is finishing


WIDTH = 88  # the Done page shows this many columns without scrolling
AI_COMMANDS = (('ai-phone call +15550100000 --prompt "…" --json', "ai-phone's agent talks"),
               ("ai-phone call +15550100000 --relay --json", "your AI talks (say, listen)"),
               ("ai-phone listen --json", "wait for a call"),
               ("ai-phone hangup", "end the call"))


def saved_files() -> str:
    """What is actually in there: secrets.env is named only when there is one (nothing was typed in that needs it)."""
    return ("config.yaml and secrets.env (keys and passwords, private to you)"
            if paths.secrets_path().exists() else "config.yaml")


def done_text(actions, session: Session) -> str:
    cur = actions.current()
    out = [f"Saved in {home_dir()}/   {saved_files()}", ""]
    rows = [(line_id, line_address(ln),
             line_kind(ln) + (calls_out(ln) if cur.allow_outbound else "")) for line_id, ln in cur.lines.items()]
    if rows:
        a, b = (max(len(r[i]) for r in rows) for i in (0, 1))
        out += [f"{'Lines' if k == 0 else '':9}{i.ljust(a)}   {where.ljust(b)}   {kind}" for k, (i, where, kind) in enumerate(rows)]
    else:
        out.append("Lines    none yet · run setup again to add one")
    if cur.brain_kind == "local":
        brain_row = f"on this computer · {tier_name(cur.tier)}"
    elif cur.brain_kind in PLANS:
        brain_row = f"your {PLANS[cur.brain_kind].name} subscription · {model_name(cur.brain_kind, cur.model)}"
    else:
        brain_row = f"{cur.base_url} · {cur.model}"
    voice_row = (f"Phonon · {voices.describe(cur.voice, engine='phonon')}" if cur.tts_engine == "phonon" else
                 f"Kokoro · {voices.title(cur.voice)} · {voices.describe(cur.voice)} · {speed_text(cur.speed)}×")
    mode = next(label for incoming, outgoing, label in CALL_MODES
                if (incoming, outgoing) == (cur.allow_inbound, cur.allow_outbound))
    lifecycle = ("Leave ai-phone running while you want to receive calls." if cur.allow_inbound else
                 "Start ai-phone when you want to place a call; stop it with Ctrl-C afterward.")
    out += [f"Calls    {mode}", f"Brain    {brain_row}", f"Voice    {voice_row}",
            "", "Start ai-phone", "  ai-phone run", f"  {lifecycle}",
            "  If already running, stop and restart to apply these settings."]
    if cur.allow_inbound:
        out += ["", "Try it", *try_it(cur.lines)]
    commands = list(AI_COMMANDS[:2]) if cur.allow_outbound else []
    if cur.allow_inbound:
        commands.append(AI_COMMANDS[2])
    elif cur.allow_outbound:
        commands.append(("ai-phone listen --call <id> --json", "listen to your relay call"))
    commands.append(AI_COMMANDS[3])
    width = max(len(command) for command, _ in commands)
    out += ["", "Give your AI the phone: it can run these from any folder"]
    out += [f"  {command.ljust(width)}   {what}" for command, what in commands]
    return "\n".join(out)


def try_it(lines: dict[str, dict]) -> list[str]:
    """One call to a ready number and one free call to a SIP address, when there are such lines."""
    numbers = [ln["number"] for ln in lines.values() if ln.get("kind", "trunk") == "trunk" and ln.get("number")
               and provider_of(ln).get("state", "ready") == "ready"]
    users = [ln.get("user") or str(ln.get("aor") or "").removeprefix("sip:").split("@", 1)[0]
             for ln in lines.values() if ln.get("kind") == "sip"]
    out = [f"  call {numbers[0]} from any phone (your normal call rates)"] if numbers else []
    if users:
        command, what = f"  pjsua --local-port 5070 sip:{users[0]}@127.0.0.1", "talk through this computer's mic, free"
        out += [f"{command}   {what}"] if len(command) + 3 + len(what) <= WIDTH else [command, f"    {what}"]
    if out:
        return out
    if any(ln.get("kind", "trunk") == "trunk" and ln.get("from_user") == "account"
           and provider_of(ln).get("state", "ready") == "ready" for ln in lines.values()):
        return ["  call the number routed to your SIP account; check that route with your provider"]
    return ["  nothing to call yet: the number is still being set up"] if lines else ["  add a line first: ai-phone setup"]


def line_address(ln: dict) -> str:
    if ln.get("number") or ln.get("aor"):
        return ln.get("number") or ln["aor"]
    if ln.get("kind", "trunk") == "trunk" and ln.get("user"):
        host = str(ln.get("registrar") or "").removeprefix("sip:").split(";", 1)[0]
        return f"sip:{ln['user']}@{host}" if host else ln["user"]
    return ""


def provider_of(ln: dict) -> dict:
    provider = ln.get("provider")
    return provider if isinstance(provider, dict) else {}  # a hand-edited config may hold anything here


def calls_out(ln: dict) -> str:
    """A verified caller id the agent uses instead of the line's own number. A sip line's caller id is its own
    address, which is no news to anyone."""
    if ln.get("kind", "trunk") == "trunk" and ln.get("from_user") == "account" and not managed_caller_id(ln):
        return " · caller ID set by your provider"
    caller = ln.get("caller_id")
    return f" · calls out as {caller}" if caller and caller not in (ln.get("number"), ln.get("aor")) else ""


def line_kind(ln: dict) -> str:
    provider = provider_of(ln)
    if ln.get("kind") == "sip":
        return "SIP address"
    if not provider.get("name"):
        return "SIP trunk"
    carrier = CARRIER_INFO.get(provider["name"])
    kind = f"{carrier.label if carrier else str(provider['name']).capitalize()} number"
    return kind if provider.get("state", "ready") == "ready" else f"{kind} · {provider['state']}"


STEP = {MODE: call_mode, BRAIN: brain, LINE: line, VOICE: voice, CALLERS: callers, CHECK: check, DONE: done}

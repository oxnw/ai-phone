# Using ai-phone

[Back to the README](../README.md) · [Example configuration](../config.example.yaml)

Setup, carrier commands, brain configuration and the full agent command contract.

## Try it (macOS, Apple Silicon)

```bash
git clone https://github.com/oxnw/ai-phone.git && cd ai-phone
./scripts/install.sh        # the SIP library and the Python environment (needs Homebrew and uv)
uv run ai-phone setup       # choose call directions, a brain, a line and a voice; downloads and checks models
uv run ai-phone run         # start ai-phone
```

`setup` saves everything under `~/.ai-phone/` (`config.yaml`, and keys and passwords in `secrets.env`, both
readable only by you) and changes only what you pick when you run it again. `uv run ai-phone setup --plain` asks the
same questions as plain prompts. The models go to the Hugging Face cache; `uv run ai-phone models pull` fetches them
without the wizard. The commands below are written as `ai-phone …`; from the repo folder that is `uv run ai-phone …`.

Setup first asks whether to **receive and place calls**, **place calls only**, or **receive calls only**.
Receiving calls needs `ai-phone run` to stay running while you want to be reachable. For outgoing calls only,
start it when you need to call, use the call commands from another terminal, then stop it with Ctrl-C afterward.
Setup does not install a background service. These choices set independent `calls.allow_inbound` and
`calls.allow_outbound` booleans; both default to true for existing configs. Incoming calls are rejected when
receiving is off, and outgoing commands are refused when placing calls is off. Change them by running setup again
and choosing **Change call directions**, then restart a running daemon to apply the change.

To change the number people see when your agent calls, run setup again and choose **Change caller ID**.
Pick a line, then its purchased number or one of your verified numbers. You can also verify another number you
own through Twilio or Telnyx. Switching back to the line's number keeps your verified numbers for later;
restart a running daemon to apply the selection.

The Voice step offers Phonon 31M English (streaming, one bundled voice at normal speed) and Kokoro
(American and British voices, adjustable speed). Both run directly through MLX in the daemon's TTS worker.
Existing configurations keep Kokoro. To select Phonon by hand, set
`tts: {engine: phonon, voice: default, speed: 1.0}`, run `uv run ai-phone models pull`, then restart the daemon.
Only the selected speech model is required; voices and weights are loaded from disk before calls begin.

No carrier yet? Enable receiving calls, pick "A SIP address" in setup and keep the default name `agent`. Talk to it locally with
`pjsua --local-port 5070 sip:agent@127.0.0.1` (free). Substitute your SIP name if you changed it.

The carrier and SIP commands are below.

## Numbers, forwarding and SIP

```bash
# 1. Its own number (a Telnyx or Twilio account and API key; carrier charges apply)
export TELNYX_API_KEY=…
ai-phone lines add telnyx --key-env TELNYX_API_KEY --country GB --area 020
ai-phone lines add twilio --key-env TWILIO_ACCOUNT_SID --secret-env TWILIO_AUTH_TOKEN --existing +15555550100   # a number you already have: nothing is bought, and removing the line keeps it
ai-phone run

# 2. Your own mobile number. It stays on your SIM, so this is two halves:
#    calls OUT show your number once you have verified it (one SMS code) …
ai-phone lines verify-caller-id telnyx +447700900123 --key-env TELNYX_API_KEY
#    … and calls IN reach the agent through your carrier's divert (one dial code, printed here)
ai-phone lines forward telnyx --when no-answer --from +447700900123
#   dial the printed **61* code on your phone; ##002# cancels every divert

# 3. A SIP address, no carrier (agents calling agents)
ai-phone lines add sip --id agents --user agents --host home.example.org --trusted 203.0.113.7
ai-phone call sip:agents@friend.example.org --line agents

# Outbound from the agent's number (or from your verified mobile number). With no --line, a number goes out on the
# first registered trunk and a sip: address on the first sip line; with two trunks configured, name the one you mean.
ai-phone call +447700900123
ai-phone status

# The brain on this Mac: one of t4, t9, qwen2b, llama3b, gemma4b, mistral7b, gemma12b (the catalog in models.py),
# auto to pick by memory, or any mlx-lm 4-bit repo id
#   models: {tier: gemma4b}
# Or a brain that is not on this Mac (see "Brains and privacy" below)
#   brain: {kind: claude, model: haiku}                 # your Claude plan, through the Claude Code login
#   brain: {kind: chatgpt, model: gpt-5.6-luna}         # your ChatGPT plan, through the Codex login
#   brain: {kind: openai_compatible, base_url: "http://127.0.0.1:1234/v1", model: "…"}
# And the voice
#   tts: {engine: phonon, voice: default, speed: 1.0}
#   tts: {engine: kokoro, voice: bf_emma, speed: 1.0}   (af_/am_ American, bf_/bm_ British)
```

`--key-env` and `--secret-env` name variables in your environment or in `~/.ai-phone/secrets.env` (where setup saves them).

During a call the daemon opens no network connection except SIP/RTP to the peer and, only if you configured one, HTTP(S) to that brain endpoint. Buying or connecting a number talks to the carrier once, from the `lines` command or `setup`, and the daemon never loads the carrier code. Provisioning is journaled: `ai-phone lines resume` picks up where a crash left off, `ai-phone lines remove` deletes what ai-phone created at the carrier (a number you brought is disconnected and stays on your account) and lists exactly what is left for you to remove in the portal if a deletion fails — or, if the journal recording what was created has gone missing, refuses rather than delete the local record of a number that is still billing. Configuration applies at restart. The NAT and transport checks run before calling it done are in [nat-checklist.md](nat-checklist.md); the 24 h registration soak is `scripts/soak.sh`.

What an incoming call hears is set by one `inbound:` section of the config: a greeting that plays on every incoming
call, and the instructions ai-phone's own model follows when no AI is waiting in `ai-phone listen` (the example config
takes a message).

## Brains and privacy

### Downloaded models and setup recommendations

Setup's **Models** check lists complete snapshots ai-phone recognizes in the configured Hugging Face cache: its speech models, catalog brains, and the model named in the current local configuration. It does not scan every model installed on the computer or inspect Ollama's library. A model appearing as downloaded only reports cache availability; it does not select that model or change the recommendation.

For a new setup, the recommendation uses detected RAM: Qwen3.5 2B for 8–15 GB, Qwen3.5 4B for 16–31 GB, and Qwen3.5 9B for 32 GB or more. Below 8 GB or when RAM cannot be detected, setup has no recommendation. You choose or confirm the brain separately. Existing catalog selections are retained when changing settings, and setup offers to download any required models that are missing. A catalog model above the machine's detected RAM can still be selected after a warning; a custom model's memory requirement is unknown to ai-phone.

For example, on a 24 GB computer with only the 9B model cached, setup lists 9B as downloaded but recommends 4B. If 4B is selected, setup offers to download it. The runtime setting `models.tier: auto` follows its own rule: it selects 9B at 32 GB or more and 4B otherwise. The 8 GB setup recommendation for 2B does not change that runtime rule.

The agent's words can come from a model on this Mac, from a plan you already pay for, or from any
OpenAI-compatible server. Speech-to-text and the voice always run here, whichever you pick. `ai-phone setup`
asks, and the daemon says at start where the words of a call go.

| Brain | Observed response time | Where model input goes | What it needs |
|---|---|---|---|
| **On this Mac** (default) | 1500 ms p50 first audible response in the [2026-10-27 local call run](phonon-2026-10-27.md) | stays local; call audio still travels over SIP/RTP | downloaded local speech and language models |
| **Your Claude subscription** | 2.4–5.5 s on a call's first turn, about 1 s after | what callers say, the call's instructions and the conversation go to Anthropic (not audio) | `claude` installed and logged in; your plan's own limits apply |
| **Your ChatGPT subscription** | 3.9–4.4 s on a call's first turn, 1.1–1.9 s after | the same, to OpenAI | `codex` installed and logged in; your plan's own limits apply |
| **Your own endpoint** | whatever that server does | the same, to that server | an OpenAI-compatible URL, `https` unless it is on this Mac; a key if it wants one |

A subscription brain uses no API key: ai-phone drives the CLI you are already logged into (`claude -p`, or
`codex app-server`) in an empty private directory, configured to disable tools and exclude your project settings, and
streams its words to the caller. It asks each CLI what its login is before it uses it, and runs only on a
subscription login: a key is billed per request, and on the Claude side `ANTHROPIC_API_KEY` beats the login
you have, so neither it nor anything else that would re-route a CLI (`OPENAI_*`, a `*_BASE_URL`) is passed on
to one, and a login ai-phone does not recognise is refused by name rather than used.
What callers say is model input, so tool access is disabled and monitored: Claude Code is run with none and asked what it
came up with, and Codex is given a `CODEX_HOME` of ai-phone's own, so none of your MCP servers, hooks, plugins
or `AGENTS.md` is in a call, with every tool feature turned off and a read-only sandbox that approves nothing.
If a session reaches for a tool anyway, the turn ends there and the session goes, so nothing a tool produced
can reach the caller.
Install them with `npm install -g @anthropic-ai/claude-code` (then `claude`
and `/login`) or `npm install -g @openai/codex` (then `codex login`). These options send call text to the
chosen provider; the observed timings above are specific to the models and runs tested. The session belongs
to one call: it is started while the greeting plays
and let go when the call ends. Proving the login costs one tiny turn of your plan every time ai-phone starts
(and again if the brain worker has to be restarted mid-run), on top of the calls themselves.

## Agent commands

Tell your agent it has the `ai-phone` command. It has two ways to make a call.

**Phone agent mode**: ai-phone's own model holds the conversation; your agent gets the transcript at the end:

```bash
ai-phone call +15555550100 --json \
  --prompt "Ask whether they have a table for two at 8 tonight, under the name Sam."
```

A phone agent `call` blocks until the call ends, which can take up to `conversation.max_call_sec` (600 seconds). With
`--json` it also writes each progress event to stderr as one JSON line, starting with
`{"event": "created", "call_id": "c7f3a2"}`. If your agent's shell kills the command before the call ends, the call
carries on. Take the ID from that `created` line and run `ai-phone listen --call c7f3a2 --json` once the call is over;
it prints the result with the transcript. While the call is still going, that `listen` exits 11 at once, so wait and try
again (`ai-phone status` shows the call while it is up). The daemon keeps the last 20 calls.

**Relay mode**: your agent holds the conversation itself, one turn at a time:

```bash
ai-phone call +15555550100 --relay --json     # returns once answered
ai-phone say "Hi, I'd like to book a table for two at eight tonight."
ai-phone listen --json                         # waits until they finish speaking
ai-phone say "Sam."
ai-phone hangup --say "That's all, thank you. Goodbye." --json
```

`ai-phone listen` with no call in progress waits for someone to call one of the agent's numbers (it first reports a
relay call that ended since anyone last listened). When someone calls, ai-phone answers with `inbound.greeting` and
`listen` prints `incoming call c9b1e0 on main from sip:+15555550100@sip.carrier.example` (the caller's SIP URI as the
SIP stack reports it, sometimes with a display name, not a bare number); your agent carries on with `listen` and
`say`. While a phone agent call is in progress (an outbound one, or an incoming call that arrived while nobody was
listening), `listen` without `--call` exits 11 at once instead of waiting, so an agent that polls for incoming calls
should pause before it tries again. While your agent is thinking, ai-phone says "One moment." so the line never goes silent, and "Still with you,
one moment." after 10 seconds; after 25 seconds without a reply it apologises and hangs up. On an outbound relay call
the 25 seconds count from the answer until your agent's first `say`, with no "One moment." before it, so say the first
line straight away. Every outbound call starts with "Hi, this is an AI assistant calling." Emergency numbers are never
dialled (a target whose digits look like one, such as extension 100 or 1000, is refused too; call such extensions by
name), and `calls.daily_limit` caps outbound calls per day.

With `--json`, each of these four commands (`call`, `listen`, `say`, `hangup`) prints exactly one JSON object on stdout
(`call` also writes its progress events to stderr, one JSON object per line); a
malformed command line (a missing or unknown option) is reported by the argument parser on stderr with exit code 2,
and Ctrl-C while `listen`, `say` or `hangup` is waiting exits 130 (Ctrl-C during `call` hangs the call up and prints
its result).

| Exit code | Meaning |
|---|---|
| 0 | answered (relay), or the call ran and ended; `listen` got an event; `say` was spoken or interrupted; `hangup` ended the call |
| 2 | bad usage (a missing or unknown option, `--prompt` with `--relay`, text too long) |
| 5 | busy (the other end, or a call is already in progress here) |
| 6 | no answer |
| 7 | rejected, or cancelled by `hangup` while still ringing |
| 8 | redirected |
| 9 | ai-phone is not running, went away, or sent a reply this command does not understand; from `hangup`, also: the call had still not ended 40 seconds after the hangup (ai-phone waits 30 seconds for the call to end, including any `--say` text, hangs up again, then waits 10 more), so it may still be up |
| 10 | `listen` timed out (the call, if any, goes on) |
| 11 | no such call, no call in progress, call already ended, or `listen`/`say` on a phone agent call |
| 12 | another `listen` is already waiting for calls |
| 13 | emergency number |
| 14 | daily outbound call limit reached |
| 130 | interrupted with Ctrl-C |

Transcripts leave ai-phone only through the command that asked for them, and, if you chose a brain that is not on
this Mac (`brain.kind: openai_compatible`, `claude` or `chatgpt`), through that endpoint — or through your own
logged-in CLI, to Anthropic or OpenAI — during a phone agent call: what callers say and the conversation so far go
there on every turn, with the call's instructions. If your agent runs on a cloud model, that service sees the
transcript.

## Latency history

### Original local baseline (2026-09-03, Apple M5 Pro, 24 GB, 4B tier)

| Measure | p50 | p95 |
|---|---|---|
| First audible response | **874 ms** | 1438 ms |
| Barge-in stop after the caller starts talking (5 interruptions) | 219 ms | 259 ms |
| Endpoint wait / STT / LLM to first clause / TTS first chunk / egress | 297 / 96 / 312 / 164 / 9 ms | |

"First audible" is measured from the last voice-active chunk of the caller to the first
non-silent RTP frame handed to the SIP stack; it includes the 300 ms endpoint wait. Add the phone's
own jitter buffer (60–100 ms) for mouth-to-ear. Measured while the host was also swapping about
10 GB of other applications, over 20 LAN calls using 8 kHz PCMU. Callers who talk over the greeting
are heard in full.

A Kokoro rerun on **2026-10-26**, using the same Mac and model tier over 20 loopback PCMU
calls, measured **1074 ms p50 / 1719 ms p95**. All 20 turns completed, but this run missed the
1000 ms p50 gate. The host still held about 20 GB in swap and paged during the run; current
default phone instructions also differ from M1. The historical result is retained separately.
See the [rerun measurements and conditions](latency-2026-10-26.md).

With Phonon integrated on **2026-10-27**, all 20 local calls completed at **1500 ms p50 / 1939 ms p95**;
the TTS stage measured **52 / 84 ms**. All five interruption calls cancelled and resumed speech, but one
stopped in 396 ms, above the 300 ms limit. Both timing gates remain unmet in this run. The host paged
heavily, and runtime and caller-duration conditions changed. See the [Phonon integration results](phonon-2026-10-27.md).

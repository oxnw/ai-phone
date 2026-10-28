# ai-phone — Architecture essentials

Read this first. If it does not answer your question, the table at the bottom says which section of `architecture.md` does. Decisions here are settled; do not reopen them without the owner. Baseline 2026-09-02.

## What it is

One daemon, one or more lines, one phone call at a time across them. It registers to SIP trunks or PBXs (or answers at a plain SIP address), answers and places calls, and runs speech-to-text → language model → text-to-speech on the host by default. During a call, no network connections other than SIP signalling and RTP media to the peer, plus — only if you chose a brain that is not on this Mac — the one endpoint you configured, or the CLI of your own Claude or ChatGPT plan. Numbers come from lines (see below): a carrier key is used once, by `ai-phone lines` or `ai-phone setup`, never by the daemon.

The current runtime and installer support macOS on Apple Silicon. Linux backends and the M5 HTTP API/dashboard
are plans, not shipped features. A configured remote brain still uses local MLX speech models.

## The six invariants (never break these)

1. **During a call, no network except SIP/RTP to the peer, and, only if configured, one brain: an endpoint, or the Claude Code / Codex CLI, which talk HTTPS to their provider.** The default brain is the local model and uses no socket; an endpoint that is not on this machine must be `https`, a subscription brain runs the CLI you are logged into as a child of the LLM worker, and the daemon says at start that transcripts, the system prompt and the history go there. No model downloads at runtime (`HF_HUB_OFFLINE=1`), no telemetry, no update checks. STUN and trusted-peer DNS happen at start only. Buying or connecting a number uses a carrier API from `ai-phone lines …` or `ai-phone setup`, never from the daemon, which does not even import the carrier code (see `architecture.md` §14a).
2. **The PJSIP thread never blocks and never touches models.** Media callbacks only copy 640-byte frames into ring buffers. Model work lives in three child processes.
3. **Everything internal is 16 kHz int16, 20 ms frames (320 samples).** The wire is 8 kHz G.711 (PCMU/PCMA only; all other codecs disabled). PJSIP resamples 8↔16; we resample TTS 24→16.
4. **Every generation has a `gen_id`.** Cancel means: all workers drop that id, the outbound ring is flushed, late results with a stale id are discarded, the partial reply stays in history marked `[interrupted]`.
5. **Latency is measured, not claimed.** t0 = last voice-active chunk; t5 = first non-silent RTP frame handed to PJSIP (daemon-side, on the PSTN leg as on the LAN). Budget 900 ms p50, gate 1000 ms. Every turn writes a JSON record with per-stage times.
6. **One daemon, one or more lines, one call at a time across all of them.** Each line is its own pjsua2 account; every telephony event carries `line_id`; the one-call claim is a lock shared by the PJSIP thread and the asyncio thread.

## Process and layer map

```
 SIP peer ◀── SIP/RTP ──▶ ┌─ ai-phoned (main process) ─────────────────────────────┐
                          │ L1 Telephony: PJSIP endpoint, account, call, media port │  PJSIP thread
                          │        │ rx_ring / tx_ring (shared memory, 2 s)          │
                          │ L2 Pipeline: asyncio TurnController, VAD, endpointing,   │  main thread
                          │        barge-in, clause chunker, resampler, timers       │
                          │ L4 Control: CLI over Unix socket, config,                │
                          │        HTTP API (M5), TUI (M5)                           │
                          └───────┬────────────┬────────────┬────────────────────────┘
                     JSON pipe + shm ▼          ▼            ▼
                              stt-worker   llm-worker   tts-worker      (L3, child processes, one model each)
```

L1 knows nothing about models. L3 knows nothing about SIP. L2 owns the conversation.

## One turn, in order

caller audio → `rx_ring` → VAD (Silero, 32 ms chunks) → **speech_end after 300 ms silence (t0)** → STT final (100 ms) → LLM streams tokens (prompt cache hit; 250 ms to first clause, zero slack) → clause chunker (punctuation, or 12 tokens) → TTS first chunk (150 ms) → 24→16 kHz → `tx_ring` → RTP (**t5**). Barge-in: ≥ 250 ms of caller speech above a playback-raised energy floor → cancel, flush, next turn; stop ≤ 300 ms from onset.

## The stack (decided)

| Layer | Choice | Not this |
|---|---|---|
| SIP/RTP | PJSIP 2.17 via pjsua2, extension built on your Mac by `scripts/install.sh`, null audio device, media port at 16 kHz | pyVoIP, Asterisk, LiveKit |
| Pipeline | hand-rolled asyncio | Pipecat (only if it ever deletes more code than it adds) |
| STT (Mac) | mlx-audio parakeet `mlx-community/parakeet-tdt-0.6b-v2` | parakeet-mlx (fallback only) |
| LLM (Mac) | mlx-lm, `mlx-community/Qwen3.5-9B-MLX-4bit` (32 GB+) or `-4B-` (16 GB; the default below 32 GB), `enable_thinking=False`, in the daemon's worker process | Ollama or any model server |
| TTS (Mac) | Direct mlx-audio, 24 kHz: `tts.engine: phonon` uses `gradium/phonon-31m-en` with `voice: default`, speed 1.0; `kokoro` uses `mlx-community/Kokoro-82M-bf16` with configurable English voice/speed (existing config default `af_heart`) | — |
| VAD | Silero, 16 kHz, in the main process | — |
| Linux (planned; no backend modules yet) | sherpa-onnx or faster-whisper · llama-cpp-python CUDA · kokoro-onnx · silero-vad | — |
| Install | clone the repo, ./scripts/install.sh, uv run ai-phone setup | Homebrew tap, Docker, PyPI (dropped 2026-09-17) |
| License | AGPLv3 (PJSIP is GPLv2-or-later; §13 permits) | MIT/Apache |

## Call rules

Answer with the cached `inbound.greeting`; every outbound call opens with the cached disclosure "Hi, this is an AI assistant calling.", played in full. Second INVITE while busy → 486. INVITE from a disallowed source → 403. Acceptance modes: `registered_trunk` (default), `lan` (RFC 1918 + loopback + link-local; used by `ai-phone run --lan`, the test peer, and CI), `any` (warned). Silence 8 s → "Still there?", 8 s more → BYE. Max call 10 min. RTP silent 10 s → BYE. Single call through launch.

## Where things live

```
src/ai_phone/  telephony/ (endpoint, call, media_port, policy, events, state, rings)
               pipeline/  (turn, replies, prompts, vad, bargein, chunker, audio_out, timing)
               services/  (base, ipc, worker, client, errors, backends/ incl. openai_compat)
               lines/     (journal, config_writer, forward, credentials)     — provisioning bookkeeping (CLI only)
               providers/ (base, telnyx, twilio, fake)                       — carrier REST (CLI only; the daemon never imports it)
               control/   (protocol, server, client)                        — the unix control socket
               setup/     (ui, flow, actions, plain, tui, look)              — `ai-phone setup` (CLI only)
               calls.py (call records), call_guard.py (emergency block, daily limit)
               models.py (model ids, download), voices.py (English voices by speech engine), vendor/ (the built SIP library, git-ignored)
               cli.py, cli_lines.py, cli_status.py, cli_call.py (call, listen, say, hangup), cli_models.py, daemon.py, config.py, secrets.py, paths.py
native/pjsua2/ SWIG interface + build script (the extension's GPL source)
scripts/       install.sh, latency_report.py, pjsua_peer.sh, utterances.txt, soak.sh, soak_report.py
config:        ~/.ai-phone/config.yaml (0600) · secrets.env (0600) · lines/ (carrier journals) · logs: ~/Library/Logs/ai-phone
models:        the Hugging Face cache (~/.cache/huggingface), fetched by `ai-phone setup` or `ai-phone models pull`
state:         ~/.ai-phone/state.json · control.sock · calls-today.json (daily outbound counter)
```

## Lines (Milestone 2)

An agent is reachable through lines: a number bought with your own Telnyx or Twilio key (`ai-phone lines add telnyx --key-env TELNYX_API_KEY`) or one already on that account (`--existing +1555…`), your own phone diverted to it (`ai-phone lines forward main --when no-answer --from +44…` prints the dial codes), a free SIP address for agent-to-agent calls (`ai-phone lines add sip …`), or a trunk you bring. The daemon sees only SIP either way; `ai-phone call <number|sip:…>` dials out with the agent's number (or your verified mobile) as caller ID and opens with an AI disclosure line; `ai-phone status` shows registration per line. Provisioning is journaled and resumable (`lines resume`), and `lines remove` undoes everything ai-phone created at the carrier; a number you brought is disconnected, never released. Configuration applies at restart. The brain is pluggable: `brain: {kind: openai_compatible, base_url: …}` makes any OpenAI-compatible server the agent, and `{kind: claude, model: haiku}` or `{kind: chatgpt, model: gpt-5.6-luna}` makes it the Claude or ChatGPT plan you already pay for, through the CLI you are logged into — no API key, the plan's own limits, and a second or two slower to the first word than the model on this Mac. Full detail in `architecture.md` §14a.

## Agent tool (Milestone 3)

Any AI that can run shell commands uses ai-phone as its phone, in one of two modes chosen per call. **Phone agent mode** (`ai-phone call <target> --prompt "…"`): ai-phone's own model holds the whole conversation, following fixed phone instructions (the cached prefix) plus the call's prompt; it requests the end with an explicit spoken farewell followed by `[END]`, and the command returns the transcript after playback finishes. **Relay mode** (`ai-phone call <target> --relay`, then `ai-phone listen`, `say` and `hangup`): the calling AI holds the conversation; ai-phone turns the other person's speech into text and speaks the AI's replies, says "One moment." when a reply is slow and hangs up politely after `calls.relay_reply_timeout_sec` (25 s). Every incoming call hears `inbound.greeting`; `ai-phone listen` with no call in progress waits for an incoming call and takes it in relay mode, and with nobody listening an incoming call runs in phone agent mode with `inbound.instructions`. Every outbound call opens with the fixed disclosure "Hi, this is an AI assistant calling."; the daemon refuses emergency numbers and enforces `calls.daily_limit`. Each call gets a 6-character hex ID and an in-memory record (the call in progress and the last 20 that ended). `call`, `listen`, `say` and `hangup` take `--json` and then print exactly one JSON object. The calling AI reaches the daemon over the local control socket, so the six invariants are unchanged. Full detail in `architecture.md` §14b.

## Get started (Milestone 4)

No packages: people clone the repo, run `./scripts/install.sh` (Homebrew's swig, pkgconf and pjproject, `uv sync`, then the SIP library build into `src/ai_phone/vendor/`), then `uv run ai-phone setup`. Setup is a full-screen wizard (numbered prompts with `--plain` or without a terminal) that asks only what applies: where the agent thinks (this Mac, or your own endpoint), a line (a new or existing Twilio/Telnyx number, a free SIP address, or your own trunk), a voice, and what callers hear; it then checks the models, the config, the lines and the voice, and ends with how to start ai-phone and what to tell your AI. Everything it saves goes under `~/.ai-phone/` (`config.yaml`, `secrets.env`, the carrier journals), and a step's answers are written only when the step completes. `ai-phone models pull` is the same download without the wizard. Carrier and endpoint requests happen only in `lines`, `models pull` and `setup`, never in the daemon. Full detail in `architecture.md` §10.

## Milestone gates

M1 latency proof: p50 < 1.0 s at 8 kHz over the 20-utterance script, else the 4B tier becomes default (passed: p50 874 ms). M2 lines: a Telnyx number bought by `lines add` registers from behind the home router and stays registered 24 h (≥ 99 %, induced drops recovered within 60 s); an iPhone call answered under 1 s daemon-side; the iPhone diverted with `**61*` reaches the agent; `ai-phone call` rings it with the agent's number; two daemons call each other over `sip:`; Twilio passes the same inbound/outbound flow; a killed brain yields the apology, not silence; `lines remove` leaves nothing in the carrier account; no worker ever holds a line password. M3 agent tool: an AI places a phone agent call and a relay call through the CLI, both end to end with transcripts; emergency numbers refused. M4 get started: clone to a free first call on this Mac in under 10 minutes with the install script and `ai-phone setup`. M5 local dashboard and HTTP API: the API drives a call. M6 launch. Nothing is built before its gate.

## Test peer (same box)

For an isolated test daemon configured on port 5080 (the integration suite uses `AI_PHONE_TEST_SIP_PORT=5080`):

```
pjsua --local-port 5082 --null-audio --play-file utt.wav --auto-play --auto-play-hangup sip:ai@127.0.0.1:5080
```

## When to open `architecture.md`

| Question | Section |
|---|---|
| Exact PJSIP config, transports, codecs, registration backoff | §3.2, §3.3 |
| What `onIncomingCall` does, policy evaluation, lockout | §3.4.1 |
| Call and registration state machines | §3.3.1, §3.4.3 |
| Media port callbacks, ring buffer layout, overflow/underflow | §3.5 |
| NAT, STUN, RTP ports | §3.6 |
| Building the SWIG extension on Mac and in CI | §3.7, §10.5 |
| TurnController states, VAD and endpointing parameters | §4.2, §4.3 |
| Clause chunking rules, token caps, thinking off | §4.5 |
| Cancellation semantics, underrun handling | §4.6 |
| Barge-in algorithm and the echo fallback | §4.7 |
| Timers and every edge behaviour | §4.8 |
| Worker interfaces (`STTBackend` etc.) and IPC messages | §5.1, §5.2 |
| Model ids, packages, per-platform backends, memory budget | §5.3–§5.6 |
| Sample rates at every hop | §6 |
| CLI commands, control socket, a later check command, HTTP API | §7 |
| Full config schema | §8 |
| Turn record fields and latency definitions | §9 |
| Getting started (install script, `ai-phone setup`, `models pull`), repo layout, CI | §10 |
| Tests at each level | §11 |
| Security model | §12 |
| Failure modes and recovery | §13 |
| Post-launch extension points | §14 |
| Lines, providers, outbound calls, pluggable brain | §14a |
| Agent tool: call records, relay mode, prompts, emergency block, daily limit | §14b |
| Which facts were verified against source | §16 |

# ai-phone — Architecture (full)

| | |
|---|---|
| **Status** | v1.0 baseline, 2026-09-02; Milestone 2 as built in §14a (2026-09-03), Milestone 3 as built in §14b (2026-09-16), Milestone 4 (getting started) in §10 (2026-09-17). Companion to `architecture-essentials.md`. |
| **Reading order** | `architecture-essentials.md` first. Come here when it does not answer the question. |
| **Precedence** | Owner's decisions → this doc. Where the text and the decision log in §15 disagree on a *decision*, the decision log wins. |
| **Notation** | **[D]** decided; do not change without the owner. **[P]** proposed default; change if Milestone 1 measurement says so, and record the change in §15. **[V]** verified against library source, PyPI, Hugging Face, or Homebrew on 2026-09-02 by an independent reviewer. |

**Implementation status:** this document contains the original design as well as later implementation notes.
`[D]` records a decision, not proof that a feature is built. The supported runtime and installer currently require
macOS on Apple Silicon. Linux backends (§5.4), the HTTP API and dashboard (M5), and the later check command (§7.3)
are not implemented. The repository map (§10.1) and CI description (§10.5) describe the current files;
`config.example.yaml` and `docs/usage.md` describe the current configuration and commands.

---

## 1. Purpose and shape

ai-phone is a single daemon that owns one or more SIP lines (each its own registration or SIP address) and one phone conversation at a time across them. It registers to SIP trunks or PBXs, answers and places calls, and runs a streaming speech-to-text → language model → text-to-speech pipeline on the host machine by default. During a call it makes no network connections other than SIP signalling and RTP media to the peer, plus — only if the user chose a brain that is not on this machine — HTTP(S) to the one endpoint they configured, or the CLI of their own Claude or ChatGPT plan, which talks HTTPS to its provider (Milestone 2 amendment, §14a; Milestone 4, §14a.7). **[D]**

Shape in one line: **a telephony layer that knows nothing about models, a model layer that knows nothing about telephony, and an asyncio pipeline between them that owns the conversation.**

Non-goals that shape the architecture: single-call, English-only, no web UI, no cloud fallback, no external model server required (a user may point the brain at an OpenAI-compatible endpoint, or at the Claude or ChatGPT plan they already pay for, §14a.7), no SMS through launch.

### 1.1 Context diagram

```
                    PSTN / other SIP endpoints
                              │
                       SIP trunk / PBX / LAN softphone / pjsua test peer
                              │  SIP over UDP/TCP/TLS (signalling)
                              │  RTP G.711 PCMU/PCMA, 20 ms frames (media), SRTP when offered
                              ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │  ai-phone daemon (one host, one or more lines, one call at a time)        │
 │                                                                           │
 │   Layer 1 Telephony ── Layer 2 Pipeline ── Layer 3 Model services         │
 │        (PJSIP)         (asyncio, main)     (3 child processes)            │
 │                              │                                            │
 │                       Layer 4 Control plane                               │
 │                  CLI · YAML config · HTTP API (M5) · TUI (M5)             │
 └──────────────────────────────────────────────────────────────────────────┘
                              ▲
                              │  localhost only
                  operator (terminal) · Home Assistant integration (post-launch)
```

## 2. Runtime topology

### 2.1 Processes **[D]**

| Process | Role | Why a separate process |
|---|---|---|
| `ai-phoned` (main) | PJSIP endpoint, asyncio event loop, pipeline, control plane, supervisor of workers | Owns the 20 ms media clock; must never block on model compute |
| `stt-worker` | Speech-to-text backend (parakeet) | GIL isolation; a slow decode cannot stall the media thread |
| `llm-worker` | Language model backend (mlx-lm, configured endpoint or subscription CLI) | Largest local model; isolated crash domain |
| `tts-worker` | Text-to-speech backend (Phonon or Kokoro) | Same |

Workers are children of the main process (`multiprocessing`, spawn start method). There is no external model server in the default install: Ollama, vLLM and similar are excluded as a dependency **[D]** (they conflict with one-command install and would appear in the isolation firewall log). Since Milestone 2 the user may *choose* an OpenAI-compatible endpoint as the brain (`brain.kind: openai_compatible`, §14a.7), and since Milestone 4 the Claude or ChatGPT plan they already pay for, through the CLI they are logged into (`claude` / `chatgpt`, a child process of the LLM worker); that is their configuration, announced at start, not something ai-phone installs or needs.

VAD runs **in the main process** **[P]**: Silero is small (~2 MB) and its per-chunk cost (~1 ms for 32 ms of audio on Apple Silicon) is far below the frame budget; keeping it in-process removes an IPC hop from the endpointing path, which is 300 of the 900 ms budget.

### 2.2 Threads in the main process

| Thread | Owner | Does | Must not |
|---|---|---|---|
| PJSIP worker thread(s) | PJSIP (`uaConfig.threadCnt`, default 1) | SIP signalling, RTP send/receive, conference bridge ticks, calls our `AudioMediaPort` callbacks every 20 ms | Block; touch Python objects outside the ring buffers and an event queue; touch models |
| asyncio loop thread (main thread) | us | Everything else: pipeline, IPC with workers, timers, control plane | Call into pjsua2 without `libRegisterThread` having been called for that thread (main thread is registered by `libCreate`) |

PJSIP → Python callbacks are SWIG *director* calls; the extension **must be built with `swig -threads`** so the callback re-acquires the GIL correctly **[V]** (pjproject's own `swig/python/Makefile` sets `-threads`). Any *other* Python thread that calls into pjsua2 must call `Endpoint.libRegisterThread(name)` first **[V]**; in this design no thread other than the main thread does.

### 2.3 Startup sequence **[D]**

1. Parse and validate config (§8). Refuse to start on any error; print the field and the fix. A trunk whose password variable is not set is not a config error but a secrets one: that line is skipped with an error in the log and in `status`, and the other lines start.
2. Verify model weights and the spaCy G2P model are present on disk (registry manifest, §7.5). Refuse to start if anything is missing; `ai-phone models pull` fetches it (§10.4). No network here or later, other than SIP/RTP to the peer and, if configured, the brain endpoint (§14a) — plus, at start only, the STUN query and the DNS lookups for registrars and trusted peers (§14a.5), which a firewall rule set must allow.
3. Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` in the environment of every process.
4. Spawn the three workers. Each loads its model and replies `ready` with warm-up timings. Warm-up = one real inference each (a 1 s silent-plus-tone buffer for STT, a 16-token generation for the LLM with the system prompt as prefix so the prompt cache is primed, one short sentence for TTS).
5. Create the PJSIP endpoint: `libCreate` → `libInit(EpConfig)` → transports → `audDevManager().setNullDev()` → codec priorities → `libStart`.
6. Create one account per configured line and register the trunk lines (sip lines run registrar-less; with no lines configured an implicit `local` sip line is created, §14a.2).
7. Start the control plane (the control socket, §7.2; the HTTP API in M5 binds `127.0.0.1` only by default).
8. Log `ready` with: SIP registration state, resident memory per process, warm-up latencies, invite-acceptance mode.

Total cold start target: under 15 s on the 9B tier on an M5 Pro **[P]**.

### 2.4 Shutdown sequence

`SIGTERM` or `ai-phone stop` → if a call is active, play nothing, send BYE, wait ≤ 2 s for the call to release → unregister (REGISTER with Expires 0) → `libDestroy` → send `shutdown` to workers, join with 5 s timeout, then kill → exit 0. Crash of any worker while idle → restart it (max 3 restarts in 10 minutes, then exit non-zero so the service manager notices). Crash mid-call → drop the current turn, say the fixed apology sentence from a cached TTS buffer, restart the worker, continue the call.

## 3. Layer 1: Telephony (PJSIP via pjsua2)

### 3.1 Why PJSIP **[D]**

The only mature, maintained, self-contained SIP+RTP stack with a Python binding. Alternatives rejected: pyVoIP (stale since 2025-03, LAN-toy quality), Asterisk/FreeSWITCH (a second server to install), LiveKit (framework, WebRTC-first). pjproject 2.17, released 2026-04-22 **[V]**.

### 3.2 Endpoint configuration

```python
ep = pj.Endpoint(); ep.libCreate()
cfg = pj.EpConfig()
cfg.uaConfig.threadCnt = 1                      # [P] one PJSIP worker thread
cfg.uaConfig.mainThreadOnly = False
cfg.uaConfig.stunServer = [...]                 # from config.nat.stun_servers, may be empty
cfg.medConfig.clockRate = 16000                 # [V] PJSUA default is already 16000
cfg.medConfig.sndClockRate = 16000
cfg.medConfig.audioFramePtime = 20              # [V] default 20 ms
cfg.medConfig.ecTailLen = 0                     # no echo canceller: null device, no acoustic path
cfg.medConfig.noVad = True                      # PJSIP's own silence suppression off; we do VAD
cfg.medConfig.jbInit / jbMinPre / jbMaxPre      # [P] defaults; revisit if RTP jitter shows in logs
cfg.logConfig.level = config.log.pjsip_level    # default 3
ep.libInit(cfg)
```

Transports **[D]**: UDP on `config.sip.bind.port` (default 5060) bound to `config.sip.bind.addr` (default `0.0.0.0`), plus TCP on the same port, plus TLS on `port+1` if `config.sip.tls` is set. `TransportConfig.publicAddress` = `config.sip.public_addr` when set (behind NAT without STUN).

Null audio device **[D][V]**: `ep.audDevManager().setNullDev()` immediately after `libInit`. Without it a headless process has no conference-bridge clock and the media port never ticks; on macOS 26 a default CoreAudio device would also trigger a microphone permission prompt. `setNullDev2(avoidClockGap=True)` **[V]** is the knob if the null device shows clock gaps under load.

Codecs **[D]**: after `libStart`, enumerate with `ep.codecEnum2()` and set priority 255 for `PCMU/8000` and `PCMA/8000`, 0 for everything else (Opus, G.722, speex, iLBC, GSM). Both directions of the SDP offer then carry only G.711, so every measured number is over 8 kHz phone audio (premise 3).

### 3.3 Account and registration

```python
acc_cfg = pj.AccountConfig()
acc_cfg.idUri = f"sip:{user}@{domain}"
acc_cfg.regConfig.registrarUri = f"sip:{domain}"          # absent in registrar-less LAN mode
acc_cfg.regConfig.timeoutSec = config.sip.register_expiry  # [P] 300
acc_cfg.regConfig.retryIntervalSec = 30                    # [P]; see 3.3.1 for backoff
acc_cfg.sipConfig.authCreds.append(pj.AuthCredInfo("digest", "*", user, 0, password))
acc_cfg.sipConfig.proxies = [outbound_proxy] if configured
acc_cfg.natConfig.sipStunUse = acc_cfg.natConfig.mediaStunUse = PJSUA_STUN_USE_DEFAULT if stun else DISABLE
acc_cfg.natConfig.contactRewriteUse = 1                    # learn public contact from REGISTER response
acc_cfg.natConfig.udpKaIntervalSec = config.nat.keepalive_sec   # [P] 25
acc_cfg.mediaConfig.transportConfig.port = config.rtp.port_min
acc_cfg.mediaConfig.transportConfig.portRange = config.rtp.port_max - config.rtp.port_min
acc_cfg.mediaConfig.srtpUse = PJMEDIA_SRTP_OPTIONAL          # accept SRTP when offered [D]
acc_cfg.mediaConfig.srtpSecureSignaling = 0
```

#### 3.3.1 Registration state machine **[D]**

```
 UNREGISTERED ──REGISTER──▶ REGISTERING ──200──▶ REGISTERED ──expiry/2──▶ (refresh) ──▶ REGISTERED
       ▲                        │ 401/407 (challenge, automatic)                │ timeout / 5xx / transport error
       │                        │ 403 / 404 (fatal: log, stop retrying)         ▼
       └────────── backoff ◀────┴────────────────────────────────── DEGRADED ──retry──▶ REGISTERING
```

Backoff on non-fatal failure: 5 s, 10 s, 20 s, 40 s, 60 s, then every 60 s, with ±20 % jitter **[P]**. The `onRegState` callback posts `RegEvent(state, code, reason)` to the asyncio queue; the pipeline never reads PJSIP state directly. `ai-phone status` shows the current state; a later check command and the M5 API will too. Success criterion: registered 24 h unattended, which this machine satisfies by construction if refresh runs at `expiry/2`.

### 3.4 Call handling

#### 3.4.1 Inbound policy **[D]**

`onIncomingCall(prm)` runs on the PJSIP thread. It must decide fast and without models:

1. Read the source address from `prm.rdata.srcAddress` (`"ip:port"`).
2. Evaluate `accept_invites_from`:
   - `registered_trunk`: accept only if the request arrived via the registered account's transport **and** the source IP matches the registrar/proxy address learned at registration (or an explicit `sip.trusted_peers` list). Otherwise reply **403 Forbidden**.
   - `lan`: accept if source IP ∈ RFC 1918 ∪ `127.0.0.0/8` ∪ `::1` ∪ `fe80::/10` **[D]** (loopback is included so the `pjsua` test peer and CI work). Otherwise 403.
   - `any`: accept from anywhere (documented as unsafe; a later check command will warn, §7.3).
3. If a call is already active or ending: reply **486 Busy Here**.
4. Optional caller allowlist (M5): if configured and the From user is not listed, 403.
5. Otherwise: create the `Call` object, answer **180 Ringing** immediately, then post `IncomingCall(call_id, from, source)` to the asyncio queue. The pipeline answers **200 OK** when it is ready to speak the greeting (normally within one loop iteration; the 180 covers the gap).

The global `calls.allow_inbound` flag is checked before line policy or the call claim; false means 403. Final
403/486 rejections are queued to the registered asyncio thread, retaining the unclaimed Call object until sent.
PJSIP's call lock keeps the rejection behind the native incoming callback's cleanup. Rejecting synchronously
inside that callback can abort PJSIP 2.17 on an offerless INVITE when cleanup tries to free an already-cleared
cloned request. The callback never waits for the queued rejection, and a rejected call never rings or claims the
active-call slot.

Per-IP auth-failure lockout (M2): count 401/403 responses per source IP over a sliding 10 min window; after 5, drop SIP messages from that IP for 15 min **[P]**. Implemented in the same callback using a small in-memory table; persisted nowhere.

#### 3.4.2 Outbound

`Call.makeCall(uri, CallOpParam)` from the asyncio thread (the main thread is registered). Triggered by the CLI (`ai-phone call <number>`, which talks to the daemon over the control socket, §7.2) and by the M5 HTTP API. Dial string → `sip:<number>@<domain>`; E.164 normalisation is the caller's job, ai-phone only strips whitespace. Outbound calls use the same pipeline; the greeting is replaced by the fixed disclosure "Hi, this is an AI assistant calling." (§14b.5).

#### 3.4.3 Call state machine **[D]**

```
 IDLE ──INVITE accepted / makeCall──▶ RINGING ──200 OK + media active──▶ ACTIVE ──BYE (either side) / timer──▶ ENDING ──released──▶ IDLE
                                        │ CANCEL / 4xx-6xx                                                        ▲
                                        └───────────────────────────────────────────────────────────────────────────┘
```

`onCallState` and `onCallMediaState` post `CallEvent`s; the pipeline's `TurnController` (§4.2) is created on `ACTIVE` and destroyed on `IDLE`. State lives in exactly one place: `telephony/call.py`. Timers owned by the pipeline, not PJSIP: silence prompt (8 s), silence hangup (8 s more), max duration warning (9:30), max duration hangup (10:00), RTP timeout (10 s without a received frame → BYE).

Second INVITE while not `IDLE` → 486 (3.4.1 step 3). BYE mid-TTS → `ENDING`, pipeline cancels every in-flight generation (§4.6), flushes the outbound ring, logs the partial turn.

### 3.5 Media port **[D]**

`onCallMediaState`: when the call's audio media becomes active, get it with `call.getAudioMedia(idx)` and connect both directions to our port: `call_media.startTransmit(port)` (caller → us) and `port.startTransmit(call_media)` (us → caller).

Our port subclasses `pj.AudioMediaPort` **[V]** and is created with:

```python
fmt = pj.MediaFormatAudio(); fmt.type = PJMEDIA_TYPE_AUDIO
fmt.clockRate = 16000; fmt.channelCount = 1; fmt.bitsPerSample = 16; fmt.frameTimeUsec = 20000
port.createPort("ai-phone", fmt)
```

Declaring 16 kHz costs nothing: the PJSUA conference bridge already runs at 16 kHz **[V]**, so the only resampler PJSIP inserts is on the 8 kHz G.711 stream port, which exists regardless. Every frame we see is **320 samples of int16 at 16 kHz = 640 bytes per 20 ms**.

Callbacks (PJSIP thread; keep them trivial):

```python
def onFrameReceived(self, frame):   # caller audio, 640 bytes
    self.rx_ring.push(bytes(frame.buf))       # never blocks; overwrites oldest on overflow and increments a counter
    self.last_rx_ns = monotonic_ns()          # RTP-timeout watchdog reads this
def onFrameRequested(self, frame):  # audio to caller, must fill 640 bytes
    buf = self.tx_ring.pop() or SILENCE_640   # underrun → silence (comfort noise later, [P])
    frame.buf = pj.ByteVector(buf); frame.size = 640; frame.type = PJMEDIA_FRAME_TYPE_AUDIO
```

Ring buffers: two single-producer/single-consumer rings of 100 frames (2 s) backed by `multiprocessing.shared_memory` **[D]** so the same memory can later be handed to a worker without a copy; head/tail indices are 64-bit atomics via `ctypes` in the same block. Overwrite-on-overflow on RX (a stalled consumer must not stall PJSIP); silence-on-underflow on TX. Both counters are exported to the latency record (§9).

DTMF: `onDtmfDigit` → `DtmfEvent(digit)` to the queue. Unused through launch (a later check command may use it to check codecs, §7.3).

### 3.6 Address and NAT

| Config | PJSIP field | Notes |
|---|---|---|
| `sip.bind.addr` / `.port` | `TransportConfig.boundAddress` / `.port` | default `0.0.0.0:5060` |
| `sip.public_addr` | `TransportConfig.publicAddress` | when behind NAT without STUN |
| `nat.stun_servers[]` | `uaConfig.stunServer` | any STUN server the user names; none by default (a STUN server *is* a network connection outside SIP/RTP, so a later check command will flag it against the isolation criterion, §7.3) |
| `nat.keepalive_sec` | `natConfig.udpKaIntervalSec` | 25 default |
| `rtp.port_min` / `port_max` | `mediaConfig.transportConfig.port/portRange` | default 10000–10100; must be reachable |

### 3.7 Building the extension

- macOS **[V]**: `brew install swig pkgconf pjproject` gives the tools and the 2.17 bottle (static `.a` libs, headers, `libpjproject.pc`, `bin/pjsua`; **no `.i` files**). The interface files are committed in `native/pjsua2/` (`pjsua2.i` and `symbols.i` from the 2.17 source tarball, whose SHA-256 is recorded beside them), so the build downloads nothing. `native/pjsua2/build.sh` runs `swig -c++ -python -threads -w312 -I$(brew --prefix pjproject)/include pjsua2.i`, then compiles and links with `pkg-config --static --cflags --libs libpjproject`. It reads the installed pjproject keg's dependencies with `brew deps pjproject` and adds their library and pkg-config paths, so the OpenSSL major comes from that keg rather than a hard-coded version. `scripts/install.sh` runs it (§10.2).
- Future Linux build requirement **[V]**: build pjproject 2.17 from source with `--enable-shared` or `CFLAGS=-fPIC`; the static default is not position-independent and the extension will not link. No Linux build workflow is currently shipped.
- The compiled module is built on each person's own Mac, for the project environment's Python, and copied with the generated `pjsua2.py` into `src/ai_phone/vendor/` (git-ignored), where ai-phone imports it as `ai_phone.vendor.pjsua2` **[D]** (2026-09-17: no wheel is published, §15). `requires-python >=3.11,<3.14`.
- Licensing **[D][V]**: PJSIP is GPLv2-or-later; ai-phone is AGPLv3 (GPLv3 §13). The extension's corresponding source (interface files, build script, pjproject version and checksum) is `native/pjsua2/` in the repository, beside the code that uses it.

## 4. Layer 2: Pipeline (asyncio, main process)

### 4.1 Overview of one turn

```
  caller RTP ──▶ rx_ring ──▶ [drain task, 20 ms] ──▶ VAD (32 ms chunks) ──▶ endpointer
                                     │                                        │ speech_end (t0)
                                     └──▶ STT partials (display only)         ▼
                                                                   STT final (t1) ──▶ LLM stream (t2 first token)
                                                                                          │
                                                                        clause chunker (t3 first clause) ──▶ TTS stream (t4 first chunk)
                                                                                          │
                                                       24→16 kHz resample ──▶ tx_ring ──▶ PJSIP ──▶ RTP (t5 first non-silent packet)
```

Timestamps t0…t5 are the latency record (§9). Budget: t5 − t0 ≤ 900 ms p50, gate 1000 ms.

### 4.2 TurnController state machine **[D]**

```
            greeting done
 GREETING ───────────────▶ LISTENING ──speech_end──▶ THINKING ──first TTS chunk──▶ SPEAKING ──tts done──▶ LISTENING
                               ▲                        │                              │
                               │                        │ barge-in (rare: caller talks  │ barge-in
                               │                        │ during the 250 ms after end)  ▼
                               └──────────────── INTERRUPTED (cancel gen, flush tx, keep partial) ──┘
 any state ──BYE / timers──▶ ENDING
```

One `TurnController` per call. Every generation carries a monotonically increasing `gen_id`; any worker result whose `gen_id` is not current is dropped on receipt (§4.6).

### 4.3 VAD and endpointing

- VAD **[D]**: Silero, 16 kHz, 512-sample chunks (32 ms). The drain task accumulates 20 ms frames into 32 ms chunks (LCM 160 ms; a small carry buffer). Speech probability threshold 0.5 **[P]**; `speech_start` after 2 consecutive positive chunks (64 ms); `speech_end` after `endpoint_silence_ms` (300 default, range 250–350 **[D]**) of consecutive negative chunks.
- **t0 is the timestamp of the last VAD-positive chunk**, not the moment the endpointer fires **[D]**. The 300 ms of silence is inside the budget, not excluded from it.
- Utterance buffer: from `speech_start − 200 ms` (pre-roll, ring already has it) to `speech_end`. Sent to the STT worker as it accumulates (every 320 ms **[P]**) so the final decode on `speech_end` only covers the tail.
- Noise floor: exponential average of chunk RMS during non-speech, used by barge-in.

### 4.4 STT

- Partials: the worker returns partial text for each pushed chunk; the pipeline shows them in the terminal only. **Partials never reach the LLM** **[D]**.
- Final: on `speech_end`, send `Finalize(gen_id)`; the worker decodes the remaining audio and returns `Final(gen_id, text, timings)`. Budget 100 ms.
- Empty final (noise, cough): return to LISTENING silently; do not count as a turn.

### 4.5 LLM and clause chunking

- Messages: `[system, ...history(≤ 6 turns), user(final_text)]` **[D]**. The system prompt includes hostname and chip (so "which machine are you running on" is answerable) and starts with a **stable prefix** so the worker's prompt cache hits every turn. Since Milestone 3 it is the fixed phone instructions (about 80 words, plus the host line) followed by a section for each call (a `--prompt` of up to 4000 characters, or `inbound.instructions`); the cached prefix is the fixed part, and the call's section is evaluated with the history (§14b.4). The 2026-09-02 budget of 60 tokens for the whole system prompt no longer applies.
- Worker streams tokens. The clause chunker emits a TTS unit at the first of: sentence punctuation (`. ! ?`), clause punctuation (`, ; :`) once ≥ 6 tokens are pending, or 12 tokens with no punctuation **[D]**. Subsequent units are sentence-sized. Each unit is sent to TTS immediately; the LLM keeps streaming.
- Caps: `max_tokens` 120 per turn **[P]** (phone answers are short); stop sequences from the chat template.
- Thinking is disabled through the chat template: `tokenizer.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)` **[V]**.

### 4.6 TTS, output, and cancellation

- Both TTS models generate 24 kHz float32 audio. Their backends resample inside the worker to 16 kHz int16 with `soxr` (VHQ) and emit 640-byte frames. Phonon streams at an explicit 80 ms audio interval; Kokoro emits sentence chunks. Leading silent frames are trimmed. First frame pushed = t5 candidate; the actual t5 is stamped by the media port when it hands the first non-silent frame to PJSIP (a flag on the ring entry).
- Underrun: if `tx_ring` empties mid-utterance (TTS slower than real time), the port sends silence and increments `tx_underruns`; the latency record shows it. Persistent underrun → switch to the lighter tier (a later check command will recommend it, §7.3).
- When speech is generated faster than playback, the TTS worker waits for space in its output ring and checks cancellation/shutdown while waiting. The pipeline transfers only what fits in `tx_ring`, preserving order and one pending tagged frame; cached greetings and other phrases also wait for playback space. Startup caching drains on chunk events, and a turn completes only after all its audio has drained. The PJSIP callback never waits for a producer.
- **Cancellation semantics [D]**: `cancel(gen_id)` is sent to all three workers; they abandon work for that id at their next check (≤ 1 chunk). The pipeline flushes `tx_ring` immediately (the caller stops hearing us within one frame plus the phone's jitter buffer), drops any in-flight chunks whose `gen_id` is stale, and keeps the partial assistant text in history marked `[interrupted]` so the model knows what the caller heard.

### 4.7 Barge-in **[D]**

While `SPEAKING`, the drain task keeps running VAD on caller audio. Trigger = **≥ 250 ms of continuous VAD-positive chunks whose RMS exceeds `noise_floor × k_playback`** (k = 3.0 **[P]**; the floor is raised only while the tx ring is non-empty, which is the speakerphone-leak guard). On trigger: transition to `INTERRUPTED`, cancel, flush, then treat the audio since the first positive chunk as the start of the next utterance. Stop target: ≤ 300 ms from caller speech onset measured at RTP egress (250 detect + ≤ 50 flush). If PSTN hybrid echo causes false triggers on a real trunk (M2), add cross-correlation of the inbound chunk against the last 200 ms of our own output and require correlation < 0.3 **[P]**.

### 4.8 Timers and edge behaviour (owned by the pipeline)

| Event | Behaviour |
|---|---|
| Greeting | `inbound.greeting` on an incoming call, the fixed disclosure on an outbound one (§14b.5), pre-synthesised at startup and cached as 16 kHz frames; plays within one loop iteration of `ACTIVE` |
| Silence ≥ 8 s after our turn | Say the cached "Still there?" once; silence ≥ 8 s more → BYE |
| Max duration | Cached warning at 9:30; BYE at 10:00 |
| RTP timeout | No `onFrameReceived` for 10 s → BYE |
| BYE mid-turn | Cancel all, flush, log turn as partial |
| Empty STT final | Ignore, stay LISTENING |
| Worker crash mid-turn | Cached apology sentence, restart worker, continue |
| Relay mode: no `say` from the calling AI | "One moment.", "Still with you, one moment.", then "Sorry, I have to go. Goodbye." and BYE (§14b.6); the silence prompt is suppressed while waiting |

All cached sentences are synthesised at startup by the TTS worker so these paths never wait on a model.

## 5. Layer 3: Model services

### 5.1 Service interface **[D]**

Engine-agnostic so a later Pipecat adoption (or any other engine) consumes the same workers.

```python
class STTBackend(Protocol):
    def warmup(self) -> Timings
    def push(self, gen_id: int, pcm16k: memoryview) -> str | None   # returns partial text or None
    def finalize(self, gen_id: int) -> tuple[str, Timings]
    def cancel(self, gen_id: int) -> None

class LLMBackend(Protocol):
    def warmup(self, system_prompt: str) -> Timings
    def stream(self, gen_id: int, messages: list[Message], max_tokens: int) -> Iterator[str]  # tokens
    def cancel(self, gen_id: int) -> None

class TTSBackend(Protocol):
    def warmup(self) -> Timings
    def stream(self, gen_id: int, seq: int, text: str) -> Iterator[bytes]   # 16 kHz PCM16, 640-byte / 20 ms frames
    def cancel(self, gen_id: int) -> None
    native_rate: int
```

### 5.2 IPC **[D]**

- **Audio** travels through `multiprocessing.shared_memory` rings (same layout as §3.5): main → stt-worker (utterance audio), tts-worker → main (16 kHz PCM16 / 20 ms frames tagged with the generation ID). No pickling of audio, ever.
- **Control** travels over a `multiprocessing.Pipe` per worker carrying newline-delimited JSON messages (small, human-readable in logs):

| Direction | Message | Fields |
|---|---|---|
| main → stt | `push` | gen_id, ring offset, nbytes |
| main → stt | `finalize` | gen_id |
| stt → main | `partial` / `final` | gen_id, text, timings |
| main → llm | `generate` | gen_id, messages, max_tokens |
| llm → main | `token` / `done` | gen_id, text / usage + timings (prompt_eval_ms, gen_ms, tokens) |
| main → tts | `synth` | gen_id, seq, text |
| tts → main | `chunk` / `done` | gen_id, seq, ring offset, nframes |
| main → any | `cancel` | gen_id |
| any → main | `ready` / `error` / `metrics` | timings / message / resident_mb |

Workers check for `cancel` between chunks/tokens; a cancelled `gen_id` is remembered so late results are discarded on both ends.

### 5.3 macOS backends **[D][V]**

| Role | Package | Model | Notes |
|---|---|---|---|
| STT | `mlx-audio[stt]` parakeet (has streaming) | `mlx-community/parakeet-tdt-0.6b-v2` (English) | v3 is multilingual, post-launch. Standalone `parakeet-mlx` is the fallback only if measurably faster. |
| LLM | `mlx-lm` | `mlx-community/Qwen3.5-9B-MLX-4bit` (32 GB+); `mlx-community/Qwen3.5-4B-MLX-4bit` (16 GB tier; the default below 32 GB, §5.5) | Both checkpoints include a vision tower; mlx-lm discards it at load (`sanitize()`), so only the text model is resident. No text-only checkpoint exists. |
| TTS | `mlx-audio[tts]` Kokoro | `mlx-community/Kokoro-82M-bf16`, voice from `tts.voice` (default `af_heart`), 24 kHz | Needs `misaki[en]` (G2P), which pulls spaCy and PyTorch (~80 packages) and **downloads `en_core_web_sm` from GitHub at first use if absent**; `pyproject.toml` pins that model as a dependency, so `uv sync` installs it. The voice is loaded by file path from the model's own snapshot (`voices/<name>.safetensors`): given a bare name, mlx-audio looks in `prince-canuma/Kokoro-82M`, which holds only two voices. `a…` voices use American pronunciation and `b…` voices British (Kokoro's `lang_code`). |
| TTS | Direct `mlx-audio` Phonon | `gradium/phonon-31m-en`, `tts.engine: phonon`, `voice: default`, speed 1.0, 24 kHz | Apache-2.0 model, pinned revision `92741c5b07d085ff840e26fea8ce43110b87852f`. Bundled tokenizer and default voice are required locally before loading. `stream=True`, `streaming_interval=0.08`, `temperature=0.3`; no server or voice-cloning input. Empty final chunks are ignored and cancellation is checked between chunks and frames. |
| VAD | `mlx-audio` Silero (or `silero-vad`) | — | 16 kHz, 512-sample chunks |

The Phonon integration pins MLX-Audio to commit `f6d5da9626bcdc5be6eccb3e7b79648548dd1b22`
(0.6.2), with MLX 0.32.4. `tts.engine` defaults to `kokoro` for existing configurations; setup offers
both engines. Each backend receives a local voice file, and Phonon also loads its pinned snapshot by `Path`.
Phonon has no adjustable speed in this path: settings other than 1.0 are refused. Raw numbers and abbreviations
use the model's text handling; no additional number-normalization pass is applied.

LLM worker specifics: `mlx_lm.load(model_id)`; `make_prompt_cache(model)` primed with the system prompt at warm-up (since Milestone 3, with the fixed phone instructions and the "This call:" header, §14b.4); `stream_generate(model, tokenizer, prompt, max_tokens=..., prompt_cache=cache)`. Because history is a suffix of a stable prefix, the cache is reused every turn; the worker reports `prompt_eval_ms` and `gen_ms` separately so the LLM budget row (250 ms, zero slack) is diagnosable.

### 5.4 Planned Linux backends **[D]**

These are recorded backend choices for a future port, not modules available in the repository. The installer
currently refuses Linux and Windows; choosing a remote brain still leaves MLX speech recognition and synthesis
on the local machine. There is no Linux runtime or Docker image to install today.

| Role | Package | Notes |
|---|---|---|
| STT | `sherpa-onnx` (parakeet or Nemotron via ONNX) or `faster-whisper` | pick by measured latency when the Linux backends are built |
| LLM | `llama-cpp-python` with CUDA | The baseline proposed the maintainer's `cu12x` wheel index and GGUF 4-bit Qwen3.5 of the same tier; packaging is not implemented. |
| TTS | `kokoro-onnx` | requires Python < 3.14 |
| VAD | `silero-vad` | — |

### 5.5 Model registry, weights, and verification **[D]**

- Weights live in the Hugging Face cache (`~/.cache/huggingface/hub`, or wherever `HF_HOME` points), where the backends load them by repo id (amended 2026-09-17; `models.home` is not used). They are fetched only by `ai-phone models pull` and `ai-phone setup`, both through `ai_phone.models.pull`; never at daemon runtime.
- What is needed: the STT and selected TTS model always, and the tier's LLM only for the local brain. Phonon additionally requires its pinned tokenizer files and bundled default voice; a partial snapshot is missing until these exist. A model counts as present when its cached snapshot has a `config.json` and weights. `pull` fetches only what is missing, first checks that the download plus 1 GB fits on the disk, reports progress, and picks up an interrupted download where it stopped. Checking every file against a per-tier manifest (sizes and SHA-256) is left to a later check command (§7.3).
- Models: the curated catalog lives in `models.py` (`CATALOG`), one entry per local brain with its repo, name, weights size, the memory it wants and whether ai-phone has made calls with it: `t4`, `t9`, `qwen2b`, `llama3b`, `gemma4b`, `mistral7b`, `gemma12b`. Selection: `ai-phone run --tier`, else `models.tier` in the config (setup writes it); `auto` picks by total memory: 32 GB or more → t9, else t4 (the 9B model does not fit beside normal apps on 24 GB, §15). A `models.tier` containing a `/` is taken as a Hub repo id, so a hand-edited config can run any mlx-lm model; anything else is refused with the list of keys.
- Environment in every process: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `TOKENIZERS_PARALLELISM=false`.

### 5.6 Memory budget (macOS, t9) **[P]**

| Component | Resident |
|---|---|
| Qwen3.5-9B 4-bit weights + KV for ≤ 2k tokens | ~5.5 GB |
| parakeet-tdt-0.6b-v2 | ~1.2 GB |
| Kokoro-82M + misaki/spaCy/torch | ~0.6 GB |
| PJSIP, rings, Python | ~0.2 GB |
| **Total** | **~7.5 GB** |

A later check command will report the actual per-process RSS after warm-up (§7.3); the README states the number.

Resident means resident. Three things keep an idle worker's weights in RAM on a host that is short of memory (the Telnyx spike ran on a 24 GB Mac with 15 GB in swap: the first transcript came 9–10 s after the caller stopped, the model workers had been compressed to under 50 MB resident, and the caller hung up):
- every MLX worker opens a wired limit before loading (`mx.set_wired_limit` to the GPU's recommended working set, `models.wire_memory`, `AI_PHONE_WIRE_MEMORY=0` to disable) — a hint the kernel does not always honour under pressure, so not the whole answer;
- after warm-up and after every request the worker caps MLX's buffer cache at 64 MB and clears it (`mx.clear_cache`): the STT worker no longer keeps the float32 copy of its checkpoint (2.8 GB → 1.5 GB footprint) and the LLM worker drops its warm-up scratch (3.9 GB → 2.8 GB);
- an idle worker runs one tiny inference every `models.keep_warm_sec` (30 s; half a second of silence through the STT encoder, one token from the LLM's cached prefix, one short word from TTS) so the weights are never the least-recently-used pages the compressor picks first.
`conversation.reply_timeout_sec` is the backstop when all three lose: the apology, not silence.

## 6. Audio path, end to end

| Hop | Rate / format | Who converts | Notes |
|---|---|---|---|
| Wire → PJSIP | G.711 μ/A-law, 8 kHz, 20 ms | PJSIP stream port | jitter buffer here |
| PJSIP bridge | 16 kHz int16 | PJSIP resampler (8→16) | bridge default rate **[V]** |
| Bridge → our port | 16 kHz int16, 320 samples/frame | none | `rx_ring` |
| Port → VAD/STT | 16 kHz | none | VAD chunks of 512 samples |
| TTS model → worker framing | 24 kHz float32 | none | Phonon / Kokoro native |
| Worker → pipeline → port | 16 kHz int16 | `soxr` 24→16 in worker | `tx_ring` |
| Port → PJSIP → wire | 8 kHz G.711 | PJSIP resampler (16→8) + codec | |

Two resamples on the outbound path (24→16 in Python, 16→8 in PJSIP) and one inbound (8→16 in PJSIP). The 100 ms "transport" budget row covers all of them plus one 20 ms frame of jitter.

## 7. Layer 4: Control plane

### 7.1 CLI (as built through Milestone 4 unless a milestone is noted)

| Command | Does |
|---|---|
| `ai-phone run [--lan] [--tier <model>]` | Start the daemon in the foreground. `--tier` takes a catalog key or a Hub repo id and defaults to `models.tier` (§5.5). `--lan` overrides `accept_invites_from` to `lan` and skips registration if no trunk is configured (dev and the timed install test) |
| `ai-phone call <target> [--prompt TEXT \| --prompt-file PATH] [--relay] [--line ID] [--timeout-sec N] [--json] [--yes]` | Place a call through the running daemon. Phone agent mode (no `--relay`): ai-phone's model holds the conversation with the prompt, and the command blocks until the call ends and prints the result with the transcript. `--relay`: returns once the call is answered; the calling AI goes on with `listen`, `say` and `hangup` (§14b.1) |
| `ai-phone listen [--call ID] [--timeout-sec N] [--json]` | Wait (default 60 s) for the next event of the relay call in progress (`heard`, `ended`) or, with no call in progress, for an incoming call (`incoming`) |
| `ai-phone say [--call ID] [--json] TEXT…` | Speak on the relay call in progress; returns `said` or `interrupted` |
| `ai-phone hangup [--call ID] [--say TEXT] [--json]` | End the call in progress (either mode), speaking TEXT first if given; prints the call's result |
| `ai-phone status [--last-call]` | Registration state per line, the call in progress, the brain in use; `--last-call` summarises the last call's turns |
| `ai-phone lines …` | Buy or connect numbers; add, divert, verify and remove lines (§14a.4) |
| `ai-phone setup [--plain]` | The setup wizard (§10.3): call directions, a brain, a line, a voice and call settings, then a check. Full-screen in a terminal; `--plain`, or no terminal, asks the same questions as numbered prompts |
| `ai-phone models pull [--tier <model>]` | Download the models this config needs (setup time; with `lines` and `setup`, the only networked commands) |
| `ai-phone latency` (M5) | The TUI latency view |

### 7.2 Control socket **[D]**

A Unix domain socket at `$AI_PHONE_HOME/control.sock` (`~/.ai-phone/control.sock`, mode 0600 from the moment it exists; a home directory too deep for `AF_UNIX` gets a short socket in a private `ai-phone-<uid>` directory under `/tmp`) carrying newline-delimited JSON: one request per connection, then the daemon's replies on that connection. The CLI is a thin client. The M5 HTTP API is a second client of the same internal commands, so there is exactly one command implementation. Commands: `status`, `call`, `listen`, `say`, `hangup`.

| Request | Replies |
|---|---|
| `{"cmd": "status"}` | `{"ok": true, "lines": {<id>: {…}}, "active_call": {…} \| null, "call": <call record without transcript> \| null, "relay_listener": true \| false}` |
| `{"cmd": "call", "to", "line", "timeout_sec", "prompt", "relay"}` | Progress events: `{"event": "created", "call_id"}`, `{"event": "dialling" \| "ringing" \| "answered" \| "busy" \| "no_answer" \| "rejected" \| "redirected" \| "cancelled", "code", "uri", "reason"}` (`cancelled` when `hangup` ended the call while it was ringing), and in phone agent mode `{"event": "turn", "turn", "first_audible_ms", "bargein"}` per turn. Then one final reply: for `relay`, `{"ok": true, "event": "answered", "call_id", "mode": "relay"}` once the call's audio is up; otherwise `{"event": "ended", "reason", "detail", "exit_code", "call_id", "result": <call record> \| null}`; or a refusal |
| `{"cmd": "listen", "call_id"?, "timeout_sec"}` | `{"ok": true, "kind": "heard", "text"}` \| `{"ok": true, "kind": "incoming", "call_id", "line", "remote"}` (`remote` is the caller's SIP URI as PJSIP reports it) \| `{"ok": true, "kind": "ended", "result": <call record>}` \| `{"ok": true, "kind": "timeout"}` \| a refusal |
| `{"cmd": "say", "call_id"?, "text"}` | `{"ok": true, "result": "said" \| "interrupted", "call_id"}` \| a refusal |
| `{"cmd": "hangup", "call_id"?, "say"?}` | `{"ok": true, "result": <call record>}` \| a refusal |

A refusal is `{"ok": false, "error": "<one line>", "code": <exit code>}` with the codes of §14b.1. A call record is the result object of §14b.1. `call_id` defaults to the call in progress.

The server checks every request before the daemon sees it and refuses a bad one with code 2: not a JSON object; an unknown `cmd`; a `call_id` that is not 6 lowercase hex characters; a `call` without a non-empty `to`, with a non-string `line`, a non-boolean `relay`, a `prompt` that is not text of at most 4000 characters, or a `prompt` together with `relay`; a `timeout_sec` that is not a number in (0, 3600] (default 30 for `call`, 60 for `listen`); `say` text, or a `hangup` `say`, that is blank or longer than 2000 characters. A connection that sends nothing within 5 s, or something that is not JSON, is closed without a reply.

One call in progress in the daemon: a `call` while any call, inbound or outbound, is up is refused with `{"ok": false, "error": "busy", "code": 5}` (the telephony claim decides, not the server). `status`, `listen`, `say` and `hangup` run alongside a call, each on its own connection. A client that goes away never unwinds a call: its later sends become no-ops, and a relay call carries on after its `call` has returned. `listen` is the exception: the server watches its connection and cancels it the moment the client disconnects, which frees the relay-listener slot (§14b.7). On shutdown the server cancels every waiting handler.

### 7.3 A later check command

Not built; later work (§15, 2026-09-17). A later check command would report: config valid · weights present and matching a per-tier manifest · spaCy `en_core_web_sm` importable (`spacy.util.is_package`) · `import ai_phone.vendor.pjsua2` works · null audio device set · GPU in use (MLX device) · resident memory after warm-up vs tier expectation · warm-start latency of each worker · SIP registration state and last error · codec list offered (must be PCMU/PCMA only) · invite-acceptance mode with a warning for `any` · STUN configured (warning: outside the isolation criterion) · outbound connectivity check *disabled* (the check itself must not phone home). Exit code non-zero on any hard failure. Until then, the Check step of `ai-phone setup` (§10.3) covers the models, the config, the lines and the voice, and `ai-phone status` shows registration.

### 7.4 HTTP API (M5) **[D]**

Bound to `127.0.0.1:8321` by default; a bearer token from config is required if `api.bind` is not loopback. Endpoints: `POST /calls {to, prompt?, relay?}` → `{call_id}`; `GET /calls/current`; `DELETE /calls/current`; `GET /status`; `GET /turns?last=n` (latency records). No streaming, no webhooks through launch (webhooks are post-launch). The Home Assistant integration (post-launch) is a client of this API.

### 7.5 Configuration file

See §8.

## 8. Configuration schema **[D]**

`$AI_PHONE_HOME/config.yaml`, mode 0600. Validated with a typed schema at startup; unknown keys are errors.

This is the 2026-09-02 baseline, kept for reference. Milestone 2 replaced the `sip:` block with `sip:` (transports) and `lines:`, with passwords in `secrets.env` (§14a.1). Milestone 3 moved `conversation.greeting` and `conversation.system_prompt` to `inbound:` and added `calls:` (§14b.9); there is no `outbound_opening_line`, because every outbound call opens with the fixed disclosure. `config.example.yaml` is the current excerpt.

```yaml
sip:
  user: "1001"
  domain: "sip.example.net"          # registrar/trunk domain; omit for registrar-less LAN mode
  password: "…"                      # or password_env: AI_PHONE_SIP_PASSWORD
  outbound_proxy: null               # "sip:proxy.example.net:5061;transport=tls"
  register_expiry_sec: 300
  bind: { addr: "0.0.0.0", port: 5060 }
  public_addr: null
  tls: { enabled: false, ca_file: null, verify_server: true }
  trusted_peers: []                  # extra IPs allowed under registered_trunk
  accept_invites_from: registered_trunk   # registered_trunk | lan | any
  allowlist: []                      # M5: From-user allowlist; empty = everyone the mode permits
nat:
  stun_servers: []                   # empty by default; each one is a network peer outside SIP/RTP
  keepalive_sec: 25
rtp:
  port_min: 10000
  port_max: 10100
codecs: [PCMU, PCMA]                 # order = priority; only these are enabled
models:
  tier: auto                         # auto, a catalog key (§5.5), or a Hub repo id
  home: "~/.ai-phone/models"
conversation:
  greeting: "Hi, this is the house. What can I do for you?"
  outbound_opening_line: "Hi, this is an automated call from …"
  system_prompt: "…"                 # baseline: ≤ 60 tokens; hostname/chip appended automatically (now §14b.4)
  endpoint_silence_ms: 300
  bargein_min_speech_ms: 250
  bargein_playback_gain: 3.0
  history_turns: 6
  max_tokens: 120
  silence_prompt_sec: 8
  silence_hangup_sec: 8
  max_call_sec: 600
  rtp_timeout_sec: 10
api:                                 # M5
  enabled: false
  bind: "127.0.0.1:8321"
  token_env: AI_PHONE_API_TOKEN
log:
  level: info
  pjsip_level: 3
  dir: "~/Library/Logs/ai-phone"     # /var/log/ai-phone in the container
```

## 9. Latency instrumentation and the turn record **[D]**

Every turn writes one JSON line to `turns.jsonl`:

```json
{"call_id":"…","turn":3,"gen_id":17,
 "t0_speech_end_ns":…, "t1_stt_final_ns":…, "t2_llm_first_token_ns":…, "t3_first_clause_ns":…,
 "t4_tts_first_chunk_ns":…, "t5_rtp_first_nonsilent_ns":…,
 "stt_ms":…, "llm_prompt_eval_ms":…, "llm_first_clause_ms":…, "tts_first_chunk_ms":…, "egress_ms":…,
 "first_audible_ms":…, "bargein":false, "bargein_stop_ms":null,
 "rx_overruns":0, "tx_underruns":0, "tokens":…, "stt_text":"…", "reply_text":"…"}
```

Definitions **[D]**: `first_audible_ms = (t5 − t0)/1e6`. `t0` is the last VAD-positive chunk's timestamp; `t5` is stamped in `onFrameRequested` when the first non-silent frame of a generation is handed to PJSIP. `bargein_stop_ms` is from the first VAD-positive chunk of the interruption to the flush. The mouth-to-ear delta (phone speaker → Mac mic recording) is measured once per platform and stated in the README; it is *not* in the counter.

The TUI (M5) tails this file. The gate check for M1 is a script over 20 turns: `scripts/latency_report.py turns.jsonl --last 20` printing p50/p95 per stage.

## 10. Getting started

Decided 2026-09-17 (§15): nothing is packaged, published or installed as a background service. People clone the repository, run `scripts/install.sh`, then `ai-phone setup`, and start ai-phone in the foreground with `uv run ai-phone run`. macOS on Apple Silicon only; the Linux backends (§5.4) are later work.

### 10.1 Repository layout

Current source files, with generated files marked. Planned Linux backends and M5 control interfaces are not
included in this map.

```
ai-phone/
  architecture-essentials.md · architecture.md · README.md · LICENSE (AGPLv3)
  pyproject.toml                    # ai-phone; requires-python >=3.11,<3.14
  src/ai_phone/
    cli.py  cli_call.py  cli_lines.py  cli_models.py  cli_status.py
    config.py  daemon.py  models.py  voices.py  calls.py  call_guard.py  paths.py  secrets.py
    telephony/   endpoint.py call.py media_port.py policy.py rings.py events.py state.py pj.py
    pipeline/    turn.py replies.py prompts.py vad.py bargein.py chunker.py audio_out.py timing.py
    services/    base.py ipc.py worker.py client.py errors.py tts.py
      backends/  mlx_parakeet.py mlx_lm_backend.py mlx_kokoro.py mlx_phonon.py mlx_memory.py
                 openai_compat.py cli_brain.py fake.py
    control/     protocol.py server.py client.py
    lines/       journal.py config_writer.py forward.py credentials.py
    providers/   base.py telnyx.py twilio.py fake.py
    setup/       ui.py flow.py actions.py plain.py tui.py look.py     # ai-phone setup (§10.3)
    vendor/      pjsua2.py _pjsua2*.so           # built by scripts/install.sh; git-ignored
  native/pjsua2/                    # pjsua2.i, symbols.i (from pjproject 2.17), build.sh, GPL notice; the extension's source
  scripts/     install.sh  latency_report.py  pjsua_peer.sh  utterances.txt
               gate_run.sh  quiet_gate.sh  soak.sh  soak_report.py  spike_trio_latency.py
  tests/       unit/ integration/
  .github/workflows/  tests.yml
```

### 10.2 The install script (`scripts/install.sh`)

It gets a clone ready to run and is safe to run again. In order, it:

1. checks for macOS on Apple Silicon, Homebrew and uv, and stops with a one-line error that says how to get what is missing;
2. installs `swig`, `pkgconf` and `pjproject` with Homebrew, skipping any already installed (the pjproject formula also brings `pjsua`, the softphone for free test calls);
3. runs `uv sync`;
4. unless `ai_phone.vendor.pjsua2` already imports, builds the SIP library with `native/pjsua2/build.sh` for the project environment's Python (§3.7), copies it into `src/ai_phone/vendor/` and checks that it imports;
5. prints the next command, `uv run ai-phone setup`.

Each step is printed as it starts, and the first failure stops the script. It downloads no models and never touches `~/.ai-phone/`.

### 10.3 The setup wizard (`ai-phone setup [--plain]`)

A full-screen Textual app when stdin and stdout are a terminal; otherwise, or with `--plain`, the same questions as numbered prompts. The conversation is one function (`setup/flow.py`) that runs in its own thread and talks to the person only through a small UI protocol (`setup/ui.py`). Two front-ends implement it: `setup/tui.py` with `setup/look.py`, and `setup/plain.py`. Everything the conversation changes goes through `setup/actions.py`, which calls the existing config writer, secrets and provisioning code.

Each step asks only what applies to the earlier answers:

| Step | Asks | Saves |
|---|---|---|
| Welcome | Shows checks of this computer: SIP library, memory and the recommended model, downloaded models, config. With a config already there: add a line, change the brain, voice, call directions, call settings or caller ID, or run every step | — |
| Mode | Receive and place calls, place calls only, or receive calls only. Explains that receiving needs the daemon to stay running; outgoing-only users start and stop it when needed | `calls.allow_inbound`, `calls.allow_outbound` |
| Brain | On this computer (one model from the catalog in `models.py`; the list shows memory requirements and a RAM-based recommendation, and a model exceeding detected RAM is confirmed first), or your own OpenAI-compatible endpoint: URL, model and an optional key, then one short test request | `brain:`, `models.tier`; the key in `secrets.env` |
| Line | A phone number, a free SIP address, your own SIP trunk, or skip. For a phone number: Twilio or Telnyx; the key is checked with a free read-only request that lists the account's numbers; then a new number, bought only after a confirmation that shows the number and its monthly price, or one already on the account, connected for free (§14a.3); then, if wanted, the codes to divert your own mobile to it | the line; carrier keys and SIP passwords in `secrets.env`; the carrier journal in `lines/` |
| Voice | Phonon or Kokoro. Phonon has one bundled English voice at normal speed; Kokoro voices are grouped American/British and female/male with adjustable speed. Either can be previewed on this computer's speakers | `tts:` |
| Settings | When receiving: the greeting and instructions for when no AI is listening. When placing calls: the daily limit (20 unless the config has one). Values for a disabled direction are preserved | `inbound:`, `calls.daily_limit` |
| Check | Offers to download missing models, with the size (§10.4). Then checks that the config loads, each line (the running daemon's status, or a background start with stand-in models for up to 20 s that reports registration, then stops) and the voice (plays the outbound disclosure) | — |
| Done | A summary: call directions, what was saved and where, how to start and keep ai-phone available, and applicable AI commands. Incoming test examples appear only when receiving is enabled. For a SIP address the test call is `pjsua --local-port 5070 sip:<user>@127.0.0.1`, since the daemon cannot call itself | — |

The Line step offers mobile forwarding only when receiving is enabled, and caller-ID verification only when
placing calls is enabled. Connecting an existing carrier number still changes its routing; setup explains
that incoming calls will be rejected in outbound-only mode.

Returning users can choose **Change caller ID** in Welcome to open the existing-line chooser in Settings.
It shows each ready trunk's current outgoing number, then offers the line's own number, numbers already verified
for that line and provider, or fresh Twilio/Telnyx verification. Selecting an existing number writes only that
line's `caller_id`, keeps verification history and needs no carrier request or credentials. Fresh verification
uses saved keys or credentials entered for that verification only, first checking that the account lists the
selected line's number; failure or cancellation keeps the current caller ID. The shortcut explains when outgoing
calls are disabled or no phone-number line is ready. Changes
take effect on daemon restart.

The chooser recommends Qwen3.5 2B for 8–15 GB of detected RAM, 4B for 16–31 GB, and 9B for 32 GB or more.
It shows no recommendation when memory cannot be detected or is below 8 GB. A fresh setup selects the recommendation;
an existing choice stays selected. Setup saves the chosen model explicitly; the runtime `auto` rule in §5.5 is unchanged.

Everything is saved under `~/.ai-phone/` (or `AI_PHONE_HOME`): `config.yaml` (0600) and `secrets.env` (0600, written only with `write_secret`). A step's answers are kept in memory and written when the step completes, so cancelling never leaves half a step behind. The exception is a number: once it is bought or connected, its line is written at once, because the number is set up at the carrier. Running setup again reads what is there and changes only what is picked. Keys and passwords are typed into hidden fields and never shown, logged or put in an error. Setup never places an agent call; the voice sample plays locally. Choosing caller-ID verification asks the carrier to text or call the owner's number.

### 10.4 Models (`ai-phone models pull [--tier <model>]`)

Downloads what the config needs and is not on this Mac yet (§5.5), for the tier given, else the config's, else the one memory suggests. It lists the models and the download size, shows progress in a terminal, and exits 0 when done (or when nothing is missing) and 3 on an error such as too little disk space. Setup's Check step uses the same function. With `lines` and `setup`, it is the only networked command; the daemon never downloads.

### 10.5 CI **[D]**

`.github/workflows/tests.yml` defines two checks on pull requests, pushes to `main`, and manual dispatch:

- **Unit tests:** installs dependencies and builds the native binding through `scripts/install.sh`, then runs
  `env -u NO_COLOR uv run pytest tests/unit -q`.
- **Local SIP smoke:** builds the same binding and uses Homebrew's `pjsua` for the echo, fake-model daemon and
  incoming-cancellation and call-ending tests. Test daemons use temporary configurations and loopback SIP on port 5080.

Both jobs use `macos-15` and Python 3.13. Test steps set Hugging Face offline mode; the workflow does not download
model weights, use subscription brains, register a carrier or place a real call. These checks cover one Python
version on macOS; a Python ABI matrix and Linux CI remain future work. Workflow configuration alone is not
evidence of a successful run or live carrier compatibility.

## 11. Testing strategy **[D]**

| Level | What | How |
|---|---|---|
| Unit | clause chunker; barge-in detector on synthetic audio (tone under playback, speech over playback, speakerphone leak simulation); endpointer timing; policy evaluation (RFC 1918 / loopback / trusted peers); config validation; ring buffer overflow/underflow | pytest, no models, no network |
| Integration | isolated loopback SIP with `AI_PHONE_TEST_SIP_PORT=5080`; echo mode proves SIP+RTP+resampling, fake-model daemon tests exercise call handling, and separate cached-model tests exercise real backends | pytest + subprocess; real-model tests require cached weights; subscription tests require explicit opt-in |
| Latency | `scripts/utterances.txt` (20 lines) played by the peer; `latency_report.py` computes p50/p95 per stage; the M1 gate | manual on the dev machine, CI-optional |
| Soak | 24 h registration against a real trunk with a call every hour | manual, M2 |
| Isolation (planned) | firewall-observed call allowing only SIP/RTP and the configured brain; no pf ruleset is currently shipped | manual launch check, not a CI result |

## 12. Security model **[D]**

- Threat: SIP scanners on the public internet. Defaults: `registered_trunk` acceptance (unsolicited INVITEs get 403), per-IP lockout, TLS+SRTP accepted when offered, RTP port range narrow and documented.
- Secrets: config file mode 0600; SIP passwords, the brain key and the carrier keys saved by `ai-phone setup` live in `secrets.env` (0600), named by `password_env`/`api_key_env`; never logged or shown.
- Planned HTTP API (M5): loopback by default; bearer token required otherwise. This API is not implemented.
- Control socket: mode 0600, so any process of the same user can place calls and read transcripts, and nobody else can. The daemon itself refuses emergency numbers and enforces `calls.daily_limit` for every client (§14b.8).
- Isolation: no outbound connections during a call except SIP/RTP to the peer and the configured brain (§14a.7). STUN and trusted-peer DNS happen at startup. No telemetry or update checks. A firewall-observed launch check remains planned; no pf ruleset is currently shipped.
- Supply chain: pinned versions; the vendored PJSIP build script records the exact pjproject tarball SHA.

## 13. Failure modes and recovery

| Failure | Detection | Recovery |
|---|---|---|
| Worker crash | pipe EOF / process exit | restart (≤ 3 per 10 min); mid-call: cached apology, continue |
| Model missing or corrupt | its worker cannot load it at start (the daemon runs offline) | refuse to start; `ai-phone models pull` fetches what is missing |
| Trunk drop / registrar unreachable | `onRegState` error | backoff §3.3.1; calls impossible until re-registered; status shows DEGRADED |
| RTP stops mid-call | watchdog on `last_rx_ns` | BYE after 10 s |
| TTS slower than real time | `tx_underruns` > 0 | silence fill; switch to the lighter tier (a later check command will recommend it) |
| LLM over budget | `llm_first_clause_ms` p50 > 250 | switch tier to t4 (config), re-measure |
| Memory pressure | worker RSS check at warm-up | refuse tier, suggest t4 |
| Model paged out while the line is idle (busy host; seen on the Telnyx spike: 10 s to the first transcript, caller hangs up) | warm-up wires each worker's model memory (`models.wire_memory`, macOS 15+, `warm` reports `wired_limit_bytes`) | wiring reduces paging — it is a hint the kernel need not honour under pressure (§5.6), so it is not the whole answer; `conversation.reply_timeout_sec` (12 s) is the backstop that turns any stalled stage into the brain-error path: apology, listen again, goodbye on the second in a row |
| Incoming request assigned to the wrong line (PJSIP scores accounts by transport/domain/user; a registered trunk can win a LAN call meant for a `sip` line) | `AiEndpoint.onSelectAccount` re-selects with `line_for_request` (target user/host, then source proxy) | the call runs under the line it was addressed to; the override is logged |
| Runaway generation | `max_tokens` cap; `max_call_sec` | hard stop |
| Duplicate INVITE / retransmits | PJSIP transaction layer | handled by PJSIP |
| Calling AI never replies (relay mode) | no `say` since the other person finished speaking | "One moment." after `calls.relay_filler_after_ms` (1.5 s), "Still with you, one moment." at `relay_second_filler_sec` (10 s), then "Sorry, I have to go. Goodbye." and BYE at `relay_reply_timeout_sec` (25 s), `ended_by: timeout`; on an outbound relay call before the first `say`, no fillers and the goodbye 25 s after the answer (§14b.6) |
| Relay listener disappears (the `listen` client exits or is killed) | the server sees end-of-file on the `listen` connection | the `listen` is cancelled at once and the listener slot is free, so the next incoming call runs in phone agent mode with `inbound:`. If the client leaves at the very moment an incoming call is being handed to it, that call stays in relay mode with nobody listening: fillers, then the relay goodbye at the timeout (§14b.7) |
| The call does not end after `hangup` (the `--say` text never plays, or PJSIP never reports the end) | the `hangup` handler waits 30 s for the end | BYE (again, for a plain `hangup`), then 10 s more; if the call has still not ended, `hangup` fails with exit 9 ("call … did not end after the hangup"), and the call may still be up |

## 14. Extension points (post-launch, do not build early)

- **Home Assistant**: HACS integration as a client of the HTTP API (calls out, status in). Not an add-on (HA hosts have no GPU).
- **Tool use**: LLM backends already stream tokens; adding tool calls means a tool schema in the system prompt and a dispatcher in `pipeline/turn.py`. Latency cost is a second LLM round trip, so it stays behind a flag.
- **Multi-call**: Linux/GPU only; one pipeline per call, workers batched or forked; PJSIP handles multiple calls natively.
- **Multilingual**: swap to `parakeet-tdt-0.6b-v3` and other Kokoro voices; the G2P dependency changes per language.
- **Full-duplex model** (Moshi-style): replaces Layers 2–3's STT→LLM→TTS with one worker; the media port and control plane stay.
- **SMS**: SIP MESSAGE method only, where the trunk supports it.
- **MCP server**: a thin wrapper over the control-socket commands (§7.2), so an MCP client can use the agent tool without a shell.

## 14a. Lines (Milestone 2, as built 2026-09-03)

The daemon is carrier-free: it registers one pjsua2 `Account` per configured line over shared UDP/TCP(/TLS) transports and routes calls and caller IDs by line. Provisioning is a CLI concern (`ai-phone lines …`) behind a `Provider` protocol with a resumable journal; the daemon never imports `ai_phone.providers` (a test pins this). Outbound calls run through a minimal control socket (extended in Milestone 3, §7.2). The language-model worker has a second backend that streams from any OpenAI-compatible endpoint and reports recoverable `brain_error` events instead of dying. Configuration applies at daemon restart; live reload and IP-change handling are later work.

**Amended invariants.** During a call, ai-phone opens no network connection except SIP/RTP to the peer and HTTP(S), with its DNS lookup, to the one configured brain: an endpoint, or (Milestone 4) the Claude Code / Codex CLI the LLM worker drives, which talks HTTPS to its own provider (the default brain is the local model and uses no socket). STUN and `trusted_peers` DNS resolution happen at daemon start only. One daemon, one or more lines, one call at a time across all of them; every telephony event carries `line_id`. Registration retry is PJSIP's own scheme (nominally 5 s, then every 30 s, each randomised by up to 10 s); the gate "re-registered within 60 s" is for induced transport loss with the carrier reachable. Latency is daemon-side (t0 → t5), on the PSTN leg as on the LAN.

**Grammar** (validated with `fullmatch` in `config.py`, re-checked by `LineSpec.from_config` for anything that ends up in a SIP header): line id `^[a-z0-9-]{1,32}$`; trunk SIP user `^[A-Za-z0-9]{4,32}$` (the carriers' credential rule), sip-line user `^[A-Za-z0-9._-]{1,64}$`; E.164 `^\+[1-9][0-9]{6,14}$`; env var `^[A-Z_][A-Z0-9_]*$`; host `host[:port]`; registrar/proxy `sip:host[:port][;transport=udp|tcp|tls]` (never `sips:`); AOR `sip:user@host[:port]`; display name printable ASCII 1–64 without `"` or `\`. Name-addr strings are built only from validated parts, never escaped.

**Secrets.** Carrier keys come from variables named on the command line, looked up in the environment and then in `secrets.env`; `lines` never writes them, `ai-phone setup` saves the ones typed into it there (Milestone 4), and the daemon never uses them. Generated SIP passwords live in `~/.ai-phone/secrets.env` (dir 0700, file 0600, atomic, opened component-by-component with `O_NOFOLLOW`) as `AI_PHONE_LINE_<ID>_PASSWORD`. The daemon reads that file itself and moves any exported secret variables out of its environment before spawning workers; each password reaches only its own account, and workers get no secrets except the brain API key, passed as a spawn argument. Provider errors redact keys, SIDs and tokens; HTTP bodies are never logged.

**Paths.** Everything under `~/.ai-phone` comes from `ai_phone.paths` (`AI_PHONE_HOME` relocates it). The control socket falls back to a short deterministic `/tmp` path when the home is too deep for `AF_UNIX`.

### 14a.1 Configuration (`sip:` is replaced; `conversation:` keeps its tuning keys; its greeting and system prompt live in `inbound:` since Milestone 3, §14b.9)

```yaml
sip:
  bind:
    address: 0.0.0.0
    udp_port: 5060
    tcp_port: 5060          # UDP and TCP may share a port number; TLS may not share with TCP
    tls_port: 5061
  tls:                      # only needed when some line uses transport tls
    cert_file: null         # server certificate for inbound TLS (sip lines); user-supplied
    key_file: null
    ca_file: null           # CA bundle for verifying registrars; null = the certifi bundle
  public_addr: null         # if set, STUN is never used; Contact and Via carry it, and AccountMediaConfig.transportConfig.publicAddress makes SDP advertise it for RTP (tested: the c= line)
  stun_server: null         # host:port; contacted at start only; used only when public_addr is null
  nameservers: []           # for SIP SRV/NAPTR resolution (UaConfig.nameserver); empty = read /etc/resolv.conf at start; none found = SRV disabled and the registrar's A record used (logged)
  tcp_keepalive_sec: 30     # CRLF keepalive for TCP and TLS transports (pjsip_cfg()->tcp/tls keep-alive interval, exposed through our own SWIG interface since PJSIP's default is 90 s); spike 0a, still pending, verifies the field names

lines:
  main:                     # line id: [a-z0-9-]{1,32}, unique
    kind: trunk             # trunk | sip
    number: "+442079460000" # trunk: E.164
    display_name: "Sam's assistant"
    registrar: "sip:sip.telnyx.com"                 # trunk: SIP URI; REGISTER target
    register_via_proxy: true                        # trunk: false for Twilio, whose REGISTER must go straight to the SIP Domain, not through the termination trunk (AccountRegConfig.proxyUse = 0)
    outbound_proxy: null                            # trunk: SIP URI for PJSIP proxies (Twilio: "sip:<trunk>.pstn.twilio.com")
    dial_domain: "sip.telnyx.com"                   # trunk: hostname[:port] (not a URI) used in the Request-URI for PSTN targets: sip:+E164@<dial_domain>
    transport: udp                                  # udp | tcp | tls
    user: "aimain7f3a"                              # SIP username, [A-Za-z0-9]{4,32} (Telnyx's rule; fine for Twilio); unique across trunk lines
    password_env: AI_PHONE_LINE_MAIN_PASSWORD       # name of the variable holding the password
    register_expiry_sec: 300
    srtp: off               # off | optional | required; optional/required require transport tls
    accept_invites_from: registered_trunk           # registered_trunk | trusted | lan | any
    trusted_cidrs: ["192.76.120.0/24", "…"]         # registered_trunk: the carrier's published signalling ranges, written by the provider at provisioning; bring-your-own trunks fill it by hand
    trusted_peers: []                               # trusted: IPs, CIDRs or hostnames (resolved at start; all A/AAAA records trusted; unresolvable → warning, not trusted)
    caller_id: "+442079460000"                      # outbound From user part; this line's number or a verified number
    diverted_from: []                               # E.164 numbers the user has diverted to this line (recorded by `lines forward`; used for the outbound warning)
    verified_caller_ids: []                         # [{number: "+447700900123", provider: telnyx, verified_at: "2026-09-10T12:00:00Z"}], written atomically by `verify-caller-id`
    provider:                                       # only when a provider created the line
      name: telnyx
      state: ready                                  # pending | ready | failed (order rejected or cancelled by the carrier) | rollback_failed
      resources: {outbound_voice_profile_id: "…", connection_id: "…", order_id: "…", number_id: "…"}
  agents:
    kind: sip
    user: "ai"
    aor: "sip:ai@home.example.org"                  # what others dial; host is public_addr or a name that resolves to it
    display_name: "ai"
    transport: udp
    accept_invites_from: trusted
    trusted_peers: ["203.0.113.7", "friend.example.org"]
    caller_id: "sip:ai@home.example.org"

inbound:                                            # Milestone 3 (§14b.9): the greeting every incoming call hears; the instructions when no AI listens
  greeting: "Hi, you've reached an AI assistant. How can I help?"
  instructions: "Take a message: ask for the caller's name, number and what it is about."

calls:                                              # Milestone 3 (§14b.8, §14b.6)
  daily_limit: 0                                    # outbound calls per local day; 0 = no limit
  relay_filler_after_ms: 1500
  relay_second_filler_sec: 10
  relay_reply_timeout_sec: 25

brain:
  kind: local                                       # local | openai_compatible | claude | chatgpt
  base_url: "http://127.0.0.1:1234/v1"              # openai_compatible; https required unless loopback
  model: "qwen3.5-4b"                               # claude: haiku|sonnet|opus (default haiku); chatgpt: a model the
                                                    # plan lists (default gpt-5.6-luna, the fastest measured)
  api_key_env: null                                 # openai_compatible only; a subscription brain needs no key
  cli: null                                         # claude/chatgpt: path to the CLI; null = `claude`/`codex` on PATH
  connect_timeout_ms: 2000
  first_token_timeout_ms: 4000                      # the daemon passes 12000 for claude/chatgpt while this is 4000
  stream_idle_timeout_ms: 4000
  max_response_chars: 2000

tts:                                                # Milestone 4, extended 2026-10-27
  engine: kokoro                                    # kokoro (default) | phonon
  voice: af_heart                                   # Kokoro: af_/am_ American, bf_/bm_ British; Phonon: default
  speed: 1.0                                        # Kokoro: 0.5 to 2.0; Phonon: 1.0 only

conversation: {endpoint_silence_ms, bargein_*, timers …unchanged…}
```

Validation (extending `config.py`): grammar first: `user` `[A-Za-z0-9]{4,32}`; `display_name` printable ASCII without `"`, `\\`, CR or LF (name-addr strings are built only from validated parts, never escaped); registrar and proxy `sip:host[:port][;transport=x]` with host a DNS name or IP; `dial_domain` `host[:port]`; `aor` `sip:user@host[:port]`; environment-variable names `[A-Z_][A-Z0-9_]*`; E.164 `\\+[1-9][0-9]{6,14}`. Then: line ids unique; trunk `user` unique; `number` and `caller_id` E.164 (or a `sip:` URI on sip lines); `registrar`/`outbound_proxy` are `sip:` URIs and `dial_domain` a hostname; `srtp != off` requires `transport: tls`; `trusted` requires non-empty `trusted_peers`; `registered_trunk` requires non-empty `trusted_cidrs` or a resolvable registrar (a warning if only the registrar); `tls` transport on a sip line requires `sip.tls.cert_file` and `key_file`; `tcp_port != tls_port`; `password_env` must be set in the environment at start, otherwise that line is skipped with an error and the others start; `brain.base_url` non-loopback must be `https`; for Kokoro, `tts.voice` must be an English voice name (`[ab][fm]_[a-z]+`) and, unless it is the default `af_heart`, one of the voices installed with the speech model (only a warning while the model is not downloaded); `tts.speed` is 0.5 to 2.0 for Kokoro. With `tts.engine: phonon`, `tts.voice` must be `default` and speed must be 1.0. The M1 `sip:` block (`user`, `domain`, `password_env`, `bind`, `public_addr`, `trusted_peers`, `accept_invites_from`) is read for one release and mapped to `sip.bind`/`sip.public_addr` plus `lines.main`, with a deprecation warning.

### 14a.2 Line lifecycle in the daemon (as built: registration state, state file, control socket)

- Transports are created once at start from `sip.bind`: UDP, TCP, and TLS only if a line uses it. Each line is an `Account` bound with `transportId` to its transport. `idUri` = `"<display_name>" <sip:<user>@<registrar host>>` for trunks, `<sip:<user>@<aor host>>` for sip lines. `AuthCredInfo("digest", "*", user, 0, password)` for trunks. `proxies` = `[outbound_proxy]` when set.
- Per-line registration state: `unregistered → registering → registered → degraded (last REGISTER failed; PJSIP retry scheduled) → registered`. `RegState` events carry `line_id`, code, reason.
- Registration loss during an established call: `dropCallsOnFail=false`, so the call continues; the line is `degraded`; `make_call` on a trunk line whose account is not registered raises `DialError` ("line … is not registered (degraded)") before dialling; inbound on that line is unavailable until re-registered. The RTP 10 s watchdog is unchanged and is the only thing that ends a call for network reasons. Tested separately from total network loss.
- IP change (Wi-Fi to Ethernet, router reboot): not handled automatically in M2. `ai-phone status` shows the line `degraded` with the last error; the fix is a restart. (`Endpoint.handleIpChange` with `AccountIpChangeConfig` is later work; something has to drive it while the line is idle.)
- Acceptance policy per line: `registered_trunk` = the line's `trusted_cidrs` plus the registrar's addresses resolved at start; `trusted` = the line's `trusted_peers`; `lan`, `any` as before. INVITE on a line whose policy rejects the source → 403, logged with `line_id`. INVITE while any call is active, on any line → 486.
- State file: the daemon writes `~/.ai-phone/state.json` (atomic, 0600) on every registration state change and call start/end: per line `{state, registrar, expires_at, last_error, since}`, plus `{active_call: {line_id, remote, started_at} | null, brain: {kind, base_url}}`. `ai-phone status` reads it and only probes the control socket for liveness (connect and close; connection refused or a missing socket → "daemon not running"); it never holds the socket, so it works during an outbound call.
- Control socket: `~/.ai-phone/control.sock` (0600), newline-delimited JSON, one request per connection. Milestone 2 built `status`, `call` and `hangup`; Milestone 3 added `listen` and `say`, the `prompt` and `relay` fields of `call`, `call_id` and `say` on `hangup`, call records in the replies and a check of every request. The commands, their replies and the concurrency rules are in §7.2: one call in progress in the daemon, the other commands alongside it on their own connections. `ai-phone status` reads `state.json` and only probes the socket for liveness. If the client of a `call` disconnects, the call continues; the CLI's Ctrl-C handler sends `hangup` and waits up to about 7 s for the `ended` reply (a 2 s `hangup` request, then a 5 s cut-off); if the socket closes unexpectedly the CLI exits 9 ("daemon went away"). A client that goes away never unwinds the call handler: its later sends become no-ops.
- Two lines on one carrier: distinct `user`, distinct accounts, shared transport. The gate registers **two** trunk lines to Telnyx (two credential connections; the second needs no number) over one transport and asserts both reach `registered` and that an inbound call on the first is attributed to `line_id: main`. The integration suite covers the routing logic with two `sip` lines on two ports and a unit test of account selection by `line_id`.

### 14a.3 Provider protocol (`ai_phone/providers/base.py`) and the two carriers

```python
class Provider(Protocol):
    name: str
    signalling_cidrs: list[str]                       # published carrier ranges; written into the line's trusted_cidrs
    def search_numbers(self, country: str, area: str | None, limit: int) -> list[Number]: ...   # Number: e164, monthly_cost, currency, requires_regulatory: bool, source_id
    def provision(self, journal: Journal, number: Number, line_id: str, sip_user: str, sip_password: str) -> ProvisionResult: ...
    def owned_numbers(self) -> list[Number]: ...                      # the numbers already on the account; read-only, and the free key check
    def attach(self, journal: Journal, number: Number, line_id: str, sip_user: str, sip_password: str) -> ProvisionResult: ...  # provision without buying
    def poll(self, journal: Journal) -> ProvisionResult: ...          # advance a pending order
    def release(self, journal: Journal) -> ReleaseResult: ...         # undo every journaled resource, in reverse
    def start_caller_id_verification(self, e164: str) -> VerificationStep: ...  # kind: enter_code_in_cli | enter_code_on_phone; instructions; handle
    def complete_caller_id_verification(self, step: VerificationStep, code: str | None) -> bool: ...
```

- **Journal** (`~/.ai-phone/lines/<line-id>.journal.json`, dir 0700, file 0600, symlinks refused): each step is written as `started` *before* its request (including every name the request will use, such as Twilio's full SIP domain name, so reconciliation looks for exactly that name) and `done` with the resource id *after*, then flushed. Two further steps close a provisioning: `carrier_ready` (all carrier resources exist and the number is delivered) and `local_committed` (config and `secrets.env` written); `resume` finishes `local_committed` if the CLI died between the two. Each config edit (`add_line`/`update_line`/`remove_line`) runs under an exclusive `flock` on `~/.ai-phone/lines/.lock` (opened `O_NOFOLLOW`, 0600); `secrets.env` writes are atomic inside a descriptor-walked directory (`O_NOFOLLOW` at every component); journal writes are atomic too (temp file + rename) with path-based symlink checks on the directory and the file. Two `lines` commands on the same line at once are not supported and are not detected beyond that. On resume, a step that is `started` but not `done` is reconciled by **searching the carrier by the stable name** before creating anything (`ai-phone:<line-id>` as connection name, credential list name, SIP domain name; the number by E.164), so a crash between the API response and the journal write never duplicates. Idempotency keys are not relied on. `release` deletes in reverse and marks `rollback_failed` with the leftover ids if a delete fails, so the user can finish in the portal. The journal is what makes a release safe, so `lines remove` will not run one without it: a missing journal, or one that no longer accounts for an id the config recorded, refuses before the first carrier call (`--local-only` is the way out, and `_commit` copies the carrier ids into the config's `provider.resources` precisely so they outlive losing the journal).
- **Provisioning states** (carrier side only): `pending` (order not delivered, or regulatory documents required: the CLI prints the requirement and the portal link; `lines resume` continues) → `ready` (all resources exist and the number is delivered) → or `failed` (the carrier rejected or cancelled the order: the CLI says so, keeps the journal, and `lines remove` releases what was created) → or `rollback_failed`. Registration is runtime state, shown by `ai-phone status`, never stored in the provider block.
- **Secrets**: carrier keys are read from the variables named on the command line, in the environment or else in `secrets.env` (where `ai-phone setup` saves them; `lines` never writes them); `lines remove`, `lines resume` and `verify-caller-id` need the same variables. SIP passwords generated by `provision` (24 random characters guaranteed to contain a digit, an upper-case and a lower-case letter, which satisfies Twilio's rule of 12+ characters with a digit and mixed case; usernames are `ai<id-without-hyphens><4 hex>`, alphanumeric, 4–32) are written to `~/.ai-phone/secrets.env` (dir 0700, file 0600, atomic, symlinks refused) as `AI_PHONE_LINE_<ID>_PASSWORD=…`; the daemon reads that file itself at start and hands each line's password to its account only; child worker processes receive no secrets except the brain API key, which the LLM worker gets as a spawn argument, not via the environment. Logs and exceptions redact keys, passwords and `Authorization` headers; HTTP bodies are never logged. Provider HTTP: HTTPS with certificate verification, 20 s timeouts, no automatic retries of creating calls (reconciliation handles re-runs).
- **Telnyx**: `search_numbers` → available numbers with `cost_information`; `--number` performs an exact `available_phone_numbers` search for that number immediately before ordering, because Telnyx only orders numbers returned by a recent search. `provision`: outbound voice profile (`ai-phone:<line-id>`) → credential connection (name `ai-phone:<line-id>`, registration allowed, NAT contact encoding on, profile attached, `user_name`, `password`) → number order with `connection_id` → poll until `success` (or `pending` with requirements). Caller ID: Verified Numbers (`enter_code_in_cli`). `signalling_cidrs`: Telnyx's published SIP signalling ranges (pinned in the module from the carrier's published page, dated; spike 0a, still pending, re-verifies them live).
- **Twilio**: `--key-env` = Account SID, `--secret-env` = Auth Token. `search_numbers` → AvailablePhoneNumbers plus the Pricing API for the monthly price. `provision`: credential list (`ai-phone:<line-id>`) with the SIP user → SIP Domain `ai-phone-<line-id>-<6 hex>.sip.twilio.com` (a name collision reported by Twilio gets one fresh suffix, journaled before the retry) with SIP registration enabled and the credential list mapped for registration → IncomingPhoneNumber purchase with `VoiceUrl` = twimlets echo carrying `<Response><Dial><Sip>sip:<user>@<domain></Sip></Dial></Response>` (Serverless Function if spike 0b shows twimlets cannot be relied on) → Elastic SIP trunk (`ai-phone:<line-id>`) with the same credential list on termination; the line gets `outbound_proxy: "sip:<trunk>.pstn.twilio.com"`, `dial_domain: "<trunk>.pstn.twilio.com"` and `register_via_proxy: false`, so PSTN INVITEs have Request-URI `sip:+E164@<trunk>.pstn.twilio.com` and go through that proxy while REGISTER goes straight to the SIP Domain (`AccountRegConfig.proxyUse = 0`); spike 0b, still pending, will prove both on one account. Caller ID: Outgoing Caller IDs (`enter_code_on_phone`: the CLI shows the 6-digit code, Twilio calls the number, the user keys it in, the CLI polls up to 2 min). `signalling_cidrs`: Twilio's published signalling ranges for the region. Inbound via trunk origination (public address) is documented, not automated.
- **Numbers you bring** (Milestone 4, 2026-09-17): `attach` sets up everything `provision` does except the purchase. It journals a `claim` step first: a read-only lookup, before anything is created, that records the carrier's number id, `number_brought: true` and where the number was routed before (Twilio: `previous_voice_url` and `previous_voice_method`; Telnyx: `previous_connection_id`). Twilio refuses there a number that a SIP trunk or a TwiML app already routes. Then come the same SIP steps as `provision`, then a `route` step that points the number at them (Twilio: the number's voice URL, the same twimlets URL a bought number gets; Telnyx: the number's `connection_id`). The `target` step records `brought`, so `lines resume` calls `attach` again, not `provision`. **`release` never deletes a brought number**: it puts the previous routing back (on Telnyx the previous connection, or none), then deletes what ai-phone created. If the routing cannot be put back, the number stays on the account and is listed as left for the user to fix in the portal.
- **Registry**: `PROVIDERS = {"telnyx": TelnyxProvider, "twilio": TwilioProvider}`; `lines add <provider>` looks it up. Providers are httpx clients without vendor SDKs.

### 14a.4 CLI surface

```
ai-phone lines add <telnyx|twilio> --key-env NAME [--secret-env NAME] [--id main] [--country GB] [--area 020] [--number +44…] [--existing +44…] [--yes]
      # --id defaults to the provider name, or <provider>-2, -3 … if taken
      # --key-env/--secret-env name variables looked up in the environment, then in ~/.ai-phone/secrets.env
      # --existing: a number already on the account (not with --number or --area); nothing is bought: after a confirmation it is
      #   connected with attach (§14a.3); `lines remove` later disconnects it and keeps it
      # search → up to 10 numbers with monthly price and any regulatory requirement → pick (or --number; --yes takes the first)
      # → provision with a journal → if the order is still pending, the line is written as pending and the command returns 2 at once (`lines resume` polls) → otherwise:
      #   daemon running: "Restart ai-phone to bring +44 20 7946 0000 live."   daemon not running: "Run `ai-phone run`; +44 20 7946 0000 will register."
      # exit codes: 0 provisioned, 2 pending (resume later), 3 provider error (journal kept; `lines remove` cleans up)
ai-phone lines resume <id> --key-env NAME [--secret-env NAME]        # journal found by line id under ~/.ai-phone/lines/
ai-phone lines remove <id> [--key-env NAME] [--secret-env NAME] [--local-only] [--yes]
      # with keys: releases every journaled carrier resource ("Release +44 20 7946 0000 from your Telnyx account? This cannot be undone [y/N]"), then removes the line and its password
      #   a brought number: "Disconnect +44 20 7946 0000 from ai-phone and delete what ai-phone created on your Telnyx account? The number stays on your account."
      # without keys: refuses unless --local-only, which removes the line locally and prints the resources left in the carrier account
      #   (from the journal, or from the ids the config recorded when the journal is gone)
      # journal missing, or missing an id the config recorded: refuses before any carrier call, names the stranded ids and points at --local-only; exit 3
      #   (a release with no ids reports success without deleting anything, which would drop the local record while the carrier kept billing)
      # partial failure: provider state rollback_failed, line kept, leftover ids printed; exit 3
ai-phone lines add trunk --id … --number +… --registrar sip:… --user … --password-env … [--outbound-proxy sip:…] [--dial-domain …] [--transport udp|tcp|tls] [--caller-id +…] [--trusted-cidr …]
ai-phone lines add sip --id … --user … --host home.example.org [--transport udp|tcp|tls] [--trusted 203.0.113.0/24 …]
ai-phone lines list
ai-phone lines forward <id> [--when all|busy|no-answer|unreachable] --from +447700900123
      # prints: "On your iPhone, dial **61*+442079460000# and press call. Check with *#61#. Cancel every divert with ##002#." and records +447700900123 in diverted_from
ai-phone lines verify-caller-id <id> +447700900123 --key-env NAME [--secret-env NAME]
      # Telnyx: prompts for the code from the SMS or call; Twilio: shows the code to key in when Twilio calls, then polls; on success the number is allowed as caller_id
ai-phone call <+E.164 | sip:…> [--prompt TEXT | --prompt-file PATH] [--relay] [--line <id>] [--timeout-sec 30] [--json] [--yes]
      # --yes required when the target is in the line's diverted_from (calling a phone that diverts back to this line)
      # exit codes: 0 answered and the conversation ran (relay: answered); 5 busy; 6 no answer (CANCEL after timeout); 7 rejected (reason printed); 8 redirected (refused); 9 daemon went away; Ctrl-C hangs up
      # the Milestone 3 options, the listen/say/hangup commands and the other exit codes: §14b.1
ai-phone status                                    # per-line registration rows from state.json; active call; the brain in use and whether it is local
```

### 14a.5 Registration, NAT, keepalive (pjsua2 values)

- `AccountRegConfig`: `registrarUri` = `registrar`, `timeoutSec` = `register_expiry_sec`, `retryIntervalSec` 30, `firstRetryIntervalSec` 5, `randomRetryIntervalSec` 10, `dropCallsOnFail` false, `unregWaitMsec` 1000, `registerOnAdd` true, `proxyUse` 0 when `register_via_proxy` is false.
- Keepalive: UDP via `udpKaIntervalSec` 15; TCP and TLS via the CRLF keepalive interval `sip.tcp_keepalive_sec` (30 s) set through `pjsip_cfg()`, exposed by a two-line addition to our SWIG interface (we build the extension ourselves). The soak on each transport is the gate.
- `UaConfig.nameserver` = `sip.nameservers` or the resolvers read from `/etc/resolv.conf`, so SRV records for `sip.telnyx.com` resolve; without any, SRV is off and the registrar's A record is used.
- `AccountNatConfig`: `contactRewriteUse` 1, `viaRewriteUse` 1, `sdpNatRewriteUse` 1, `udpKaIntervalSec` 15; ICE and TURN off. `UaConfig.stunServer` = `sip.stun_server` only when `public_addr` is null and the user set it (spike 0a, still pending, checks whether Telnyx or Twilio document one; until then it stays null).
- TLS: `verifyServer` on; `CaListFile` = `sip.tls.ca_file` or the certifi bundle; hostname verification against a real registrar is part of the Task 16 gate (TLS held 1 h on the NAT checklist); there is no local TLS peer in the integration suite. Inbound TLS on a sip line uses the user-supplied `cert_file`/`key_file`; no self-signed generation.
- NAT checklist run during the soak on the user's network, recorded in the gate report: UDP, TCP, TLS registration each held 1 h with keepalive; SIP ALG (symptom: rewritten Contact; documented fix: TCP or TLS); IP change (documented: restart); DNS SRV resolution of the registrar with the configured nameservers; one-way audio judged with `ai-phone status --last-call` and the call's turn records (transcripts prove inbound frames, first-audible times prove outbound).
- SRTP: `srtp: optional|required` sets `srtpUse` and `srtpSecureSignaling=1` (SDES keys only over TLS). Plain UDP lines run `srtp: off`. The M1 `srtpSecureSignaling=0` setting is retired.

### 14a.6 Outbound (`ai-phone call`)

- `Call::makeCall` on the line's account. Request-URI: `sip:+E164@<dial_domain>` for PSTN targets on a trunk line; the URI itself for `sip:` targets (on a sip line by default, or a trunk line if `--line` says so). `CallOpParam.txOption.localUri` = `"<display_name>" <sip:<caller_id>@<registrar host>>` on trunk lines, and the line's `caller_id` URI verbatim on sip lines; `caller_id` is validated before dialling: this line's number, or a number in the line's `verified_caller_ids`, otherwise refused. Carrier rejections with a caller-ID reason (403, 603) are mapped to a plain message. Verification does not guarantee what the callee's handset displays (CNAM and STIR/SHAKEN attestation are the carrier's), and the CLI says so once.
- Outbound states: `dialling → ringing (180/183; early media ignored: audio that becomes active before CONFIRMED is not connected to the media port until the 200 OK) → answered (CONFIRMED) | busy (486/600) | no-answer (timeout → CANCEL) | rejected (other 4xx/5xx/6xx) | redirected (3xx → onCallRedirected returns PJSIP_REDIRECT_STOP)`. 401/407 are handled by PJSIP with the account's credentials. Voicemail is not detected; the max-call and silence timers cover it.
- On `answered`: the controller enters `OUTBOUND_OPENING` and plays the cached disclosure "Hi, this is an AI assistant calling." in full (Milestone 3, §14b.5); then normal `LISTENING`, or in phone agent mode the model's opening sentence. Everything else (gen_ids, timers, RTP baseline) is unchanged.
- Loop prevention is best-effort by design: the `diverted_from` warning before dialling, and the one-call-at-a-time 486 on any INVITE that arrives while the outbound call is up. No `Diversion`/`History-Info` parsing.

### 14a.7 Pluggable brain

- `services/backends/openai_compat.py`: an `LLMBackend` that POSTs `<base_url>/chat/completions` with `stream: true`, the call's system prompt (§14b.4; without the hostname/chip line the local brain gets) and the capped history, and yields `delta.content` tokens to the same clause chunker. Cancel closes the HTTP response. Redirects are not followed. TLS certificate verification on; `api_key_env` is sent as `Authorization: Bearer` and never logged.
- Failure rules: connect timeout, first-token timeout, idle timeout, non-2xx, malformed SSE, or `max_response_chars` exceeded → the backend raises `BrainError(kind)`; the LLM worker catches it and emits a typed IPC event `{"ev": "brain_error", "gen": <gen_id>, "kind": <kind>}` instead of dying (today any backend exception is a fatal worker error, so `run_worker` gains this catch for the generate path only). The controller, on `brain_error` for the current gen: first cancels that generation exactly like a barge-in (workers drop the gen, the outbound ring is flushed, a new `gen_id` is issued) so no partial model output is spoken over the apology, then plays the cached phrase "Sorry, I lost my train of thought. Could you say that again?", records `brain_error=<kind>` in the turn, returns to `LISTENING`; on the second consecutive `brain_error` in a call: cached goodbye and BYE. The counter resets on a turn that completes. No automatic retry (the caller is waiting). A stale-gen `brain_error` is discarded like any stale result.
- Latency is recorded per turn exactly as for the local brain (`llm_first_clause_ms`), not gated.

**A subscription as the brain** (Milestone 4, 2026-10-01). `brain.kind: claude` or `chatgpt` makes the plan the person
already pays for the agent, with no API key: `services/backends/cli_brain.py` drives the CLI they are logged into.
`ClaudeCodeLLM` runs `claude -p --verbose --no-session-persistence --setting-sources "" --strict-mcp-config --settings
'{"alwaysThinkingEnabled": false}' --input-format stream-json --output-format stream-json --include-partial-messages
--system-prompt <prompt> --model <model> --tools ""` and writes one `{"type":"user",…}` line per turn; text arrives as
`stream_event` deltas and `result` ends the turn (`--bare` must not be used: it skips the keychain login, and every
CLAUDE* variable is removed from the child's environment or the CLI takes itself for a nested session). `CodexLLM`
speaks `codex app-server`'s JSON-RPC 2.0 over stdio: `initialize` → `initialized` → `account/read` → `thread/start`
(ephemeral, `sandbox: read-only`, `approvalPolicy: never`, `baseInstructions`, the model always named explicitly
because the person's own default may be one the plan forbids) → `turn/start` per turn, `item/agentMessage/delta` until
`turn/completed`, `turn/interrupt` to cancel. Both run in `~/.ai-phone/brain/` (0700, empty), so no CLAUDE.md,
AGENTS.md or project settings of the person's own are read. One session per call: the session remembers the
instructions and the conversation it was given, so a turn sends only the newest thing the caller said; a different
system prompt, or a history that is not this session's continuation, starts a new session. A reply the caller talked
over comes back marked `[interrupted]`, so an assistant turn is matched by its place in the conversation, not its
words. The worker's new `llm.prepare` op (sent by the daemon as soon as a call's instructions are composed, before the
greeting plays, and ignored by backends without `prepare`) starts the session so the first reply does not pay the
startup. `warmup` reads the login first without inference (`claude auth status --json`, Codex's `account/read`) and
refuses an API-key login, then spends one tiny turn ("Say OK.") to prove the plan answers and throws that session away;
failures are the same `BrainError` kinds as the endpoint brain plus `cli_missing`, `not_logged_in`, `api_key_login`,
`login_unknown`, `rate_limited`, `model_not_allowed`, `tool_use`, `tools_unknown` and `tools_renamed`, and they carry a
kind only (plus, for the three kinds whose detail is a value ai-phone itself produced — a login method, a flag name —
that value, so a refusal is something the person can act on) — a provider's own
words are classified and then go to the debug log, never into the exception the worker, the daemon's log or a wizard
page sees.

**The tool boundary** (2026-10-01). What a caller says is model input, so a session that can reach a tool is a way
into this Mac from the phone. Claude Code takes `--tools ""` and reports its tool list on the `system`/`init` line,
which must be empty or the turn is `BrainError("tool_use")`. Codex has no equivalent: its config `tools` table holds
only web search, its `default_tools_enabled`/`omit_tools_from` settings belong to app connectors, its sandbox cannot
be narrower than read-only (which still reads the whole disk), and the protocol has no request that lists the tools a
thread has. Measured against codex-cli 0.157.1: `-c mcp_servers={}` is silently ignored and ten of the owner's MCP
servers started inside an ai-phone thread. So a Codex session is confined three ways and then watched. (1) Every tool
feature is disabled by name — `--disable shell_tool --disable unified_exec --disable view_image --disable apps …`,
the flags behind `codex features list`. (2) `CODEX_HOME` is `~/.ai-phone/brain/codex/` (0700), holding only a
config.toml ai-phone writes and a symlink to the person's own `auth.json`, so the login is theirs and no copy of their
tokens is made, while their MCP servers, hooks, plugins, skills and AGENTS.md are all absent. (3) `thread/start` asks
for `sandbox: read-only` with no network and `approvalPolicy: never`, and the thread's own answer is checked: anything
else (a requirements file, an MDM policy, a newer CLI) is `BrainError("tools_unknown")` before a turn is spent — the
mirror of the Claude side's assertion. Then, during a turn, an `item/started` **or** `item/completed` whose item is not
a user message, an agent message, reasoning or a compaction (an item whose type cannot be read counts as not), an
`mcpServer/startupStatus/updated`, a `configWarning`, a notification that means something ran (`hook/started`,
`command/exec/outputDelta`, `item/plan/delta`, …), or any approval or tool request (which is answered, so the turn
cannot hang) is `BrainError("tool_use")`: the turn ends there and the session is marked spent, so the rest of the call
gets a new one. The guarantee is **not** that the caller hears nothing — deltas are streamed to TTS as they arrive, so
words the model said *before* it reached for a tool have already been spoken, and buffering every reply to completion
would cost the whole latency budget this backend exists for. What is guaranteed is that **no tool output can reach the
caller**: the event queue is strictly ordered, so the abort fires on the tool's own notification, before any delta that
could carry its result is read. `_spent` is read *after* `_settle()`, because the drain is where the watch runs for a
turn the caller talked over. Verified against the real CLI with one probe turn: asked to read a file outside its
directory and `/etc/hosts`, the session produced only `userMessage`, `reasoning`, `agentMessage` and said it could not
reach them.

A tool feature ai-phone names that this Codex no longer has is its own failure, not a mystery: `codex app-server` exits
1 with `Error: Unknown feature flag: X` on stderr and nothing on stdout, which would otherwise surface as kind `failed`
with the reason visible only at debug. `warmup()` therefore asks `codex features list` (local, no turn) and raises
`BrainError("tools_renamed", "<flags>")` naming them, and the stderr line is promoted to the same kind for a CLI that
was upgraded under a running daemon. Fail-closed either way, and the wizard says which flag to blame.
`max_tokens` cannot be imposed on a CLI, so a reply past `max_response_chars` ends the turn, and the reader stops a
session that prints more than one call's worth of output (`BrainError("too_long")`). Nothing of the conversation is
logged: timings and the model only, with the CLI's own stderr at debug, truncated. The session belongs to one call:
the daemon sends `llm.release` as the call ends (the mirror of `llm.prepare`; every other brain ignores both), so no
CLI process idles between calls, and `close()` kills the whole process group, since a Node runtime starts children of
its own. Claude Code cannot be interrupted, so a turn the caller talked over has to finish before the next one is
sent; that wait is bounded by a third of `conversation.reply_timeout_sec` (the daemon passes it as
`drain_timeout_ms`), because whatever is spent on the old turn comes out of the time the new one has before the
controller apologises — past it the session is dropped and a fresh one started. The working directory is created with
`private_mkdir`, since probing a plan in the wizard can be the first thing to make `~/.ai-phone` and it must not be
left at the umask's 0755. Measured on this
Mac (first word, a call's first turn): Claude Code 2.1.286 + Haiku 2.4–5.5 s, then about 1 s; codex-cli 0.157.1 +
gpt-5.6-luna 3.9–4.4 s, then 1.1–1.9 s — so the daemon raises `first_token_timeout_ms` to 12000 for these kinds while
the config still holds the 4000 a model on this Mac gets.


**Differences between the approved design and what was built** (each came out of a review or a test): Telnyx orders carry `customer_reference "ai-phone:<line-id>"` and an unfinished order is reconciled by that reference; a delivered order whose number is not yet listed stays `pending`; Twilio provisioning is one journal step per carrier request (credential list, credential, domain, domain mapping, number, trunk, trunk credential list), each reconciled by name; `lines resume` re-runs `provision()` (idempotent through the journal) rather than `poll()`, using a `target` step written first; pending results are written only if the line loads; `carrier_ready` is journaled only for ready results; the brain backend streams in a helper thread so the first-token and idle deadlines are enforced by the consumer (headers and SSE keepalives included) and `[DONE]` ends the stream at once; `brain_error` cancels the generation like a barge-in, keeps the partial reply in history as `[interrupted]`, holds the apology until the PJSIP-side flush has happened, and the goodbye drains before the BYE; `MediaActive` and `Dtmf` carry `line_id`; a single `threading.Lock` makes the one-call claim atomic between the PJSIP thread and the asyncio thread; the outbound answer timer is owned by its call; `dialling` is queued before `makeCall` and a synchronous failure closes the sequence with `rejected`. Two more came out of the Telnyx spike (2026-09-03): incoming requests are assigned to a line by `line_for_request` (request-URI/To user and host, then the trunk's registrar/trusted proxies) through `AiEndpoint.onSelectAccount`, because PJSIP's own scoring handed a LAN INVITE for the `sip` line to the registered trunk; and the MLX workers wire their model memory at warm-up with `conversation.reply_timeout_sec` as the backstop, because a busy host paged the idle models out and the first transcript took 10 s.

## 14b. Agent tool (Milestone 3, as built 2026-09-16)

Any AI that can run shell commands uses ai-phone as its phone. It places calls from the agent's number and answers calls to it, in one of two modes chosen per call:

- **Phone agent mode.** The calling AI writes the call's instructions (`--prompt`). ai-phone's own model holds the whole conversation (listening, replying, interruptions, ending the call), and the calling AI gets the transcript when the call ends. The calling AI takes no part in between.
- **Relay mode.** The calling AI holds the conversation itself. ai-phone turns the other person's speech into text (`ai-phone listen`), speaks the AI's replies (`ai-phone say`) and fills short gaps while the AI thinks.

Nothing about the calling agent is saved: each call's instructions are written for that call, and incoming calls use one `inbound:` section. Not in this milestone: an MCP server (§14), the dashboard and HTTP API (M5), SMS, keypad tones, hold and voicemail detection, more than one call at a time. The six invariants are unchanged: the calling AI reaches the daemon over the local Unix control socket, and during a call the daemon still talks only SIP/RTP to the peer and HTTP(S) to the one configured brain endpoint.

### 14b.1 Commands

```
ai-phone call <+E.164 | sip:…> [--prompt TEXT | --prompt-file PATH] [--relay] [--line ID] [--timeout-sec 30] [--json] [--yes]
      # phone agent mode (no --relay): prints progress (call <id>, dialling, ringing, answered, one line per turn),
      #   blocks until the call ends, then prints the result:
      #     call c7f3a2 ended: completed, hung up by agent, 1m42s
      #       agent: Hi, this is an AI assistant calling.
      #       agent: I'd like to book a table for two at eight tonight.
      #       them:  Sure, what name?
      # --prompt / --prompt-file: UTF-8, at most 4000 characters (the file is read and trimmed); optional: without
      #   it the model says it is calling on behalf of the owner of the number
      # --relay: blocks until the call is answered or fails, prints "call c7f3a2 answered" and exits 0; the call carries on
      # --prompt with --relay, or --prompt with --prompt-file: usage error, exit 2; --timeout-sec is how long it may ring
ai-phone listen [--call ID] [--timeout-sec 60] [--json]
      # a relay call in progress (or --call ID): its next unread event
      #   heard: Sure, what name is it under?     everything said since the last listen, in order, as one text
      #   call c7f3a2 ended: …                     the result, as for call; with --call ID it is returned again on
      #                                            every later listen, without --call it is reported once
      # a phone agent call in progress, or --call naming one that has not ended: exit 11 at once
      # no call in progress: first a relay call that ended since anyone last listened, otherwise wait for an incoming call
      #   incoming call c9b1e0 on main from sip:+15551234567@sip.carrier.example
      #       already answered; inbound.greeting plays; the next listen returns the caller's first words; the text
      #       after "from" is the caller's SIP URI as PJSIP reports it (it may carry a display name), not a bare number
      # nothing within the timeout: prints "timeout", exits 10; a timeout never ends a call
ai-phone say [--call ID] [--json] TEXT…
      # speaks on the relay call in progress; prints "said" once it has played, or "interrupted" when the other person
      #   talked over it (the next listen returns what they said); 1 to 2000 characters; waits up to 10 s for a call
      #   whose audio is not up yet
ai-phone hangup [--call ID] [--say TEXT] [--json]
      # ends the call in either mode, speaking TEXT first if given (§14b.6); prints the call's result
```

`--call` defaults to the call in progress; a call that has ended can be named for as long as its record is kept (§14b.2). Every error ai-phone reports itself is one line, `error: …`, or with `--json` the object `{"ok": false, "error": "…", "code": N}`. With `--json` (accepted by these four commands, not by `run`, `status` or `lines`) each prints exactly one JSON object on stdout and nothing else there: the call's result with `"ok"` added; `{"ok": true, "call_id", "mode": "relay", "outcome": "answered"}` for an answered relay call; the `listen` and `say` replies of §7.2 (for `ended`, the result's fields sit beside `"kind": "ended"`); for any other ending, the result (when there is one) with `ok`, `outcome`, `detail` and `code` added. A malformed command line (a missing or unknown option) is reported by the argument parser on stderr with exit code 2, not as JSON. `call --json` also writes every progress event (`created`, `dialling`, `ringing`, `answered`, `turn`, …) to stderr, one JSON object per line. `created` carries the call ID, so a calling AI whose `call` command is killed can still fetch the result with `listen --call ID` once the call has ended.

The result of a call:

```json
{"call_id": "c7f3a2", "mode": "agent", "direction": "out", "line": "main", "remote": "+15551234567",
 "outcome": "completed", "detail": "", "ended_by": "agent",
 "created_at": 1789600000.0, "answered_at": 1789600004.2, "ended_at": 1789600106.2, "duration_sec": 102.0,
 "transcript": [{"who": "agent", "text": "Hi, this is an AI assistant calling.", "at": 1789600004.3, "interrupted": false},
                {"who": "agent", "text": "I'd like to book a table for two at eight tonight.", "at": 1789600007.1, "interrupted": false},
                {"who": "them", "text": "Sure, what name?", "at": 1789600009.8, "interrupted": false}]}
```

Times are Unix seconds; `duration_sec` runs from answer to end (null if the call was never answered). `outcome` is `completed`, `busy`, `no_answer`, `rejected`, `redirected` or `cancelled`. `ended_by` is `agent` (the model's closing reply passed end-marker validation), `them`, `timeout` (the relay reply timeout), `silence`, `max_duration`, `hangup_command`, or `error` (RTP timeout, a second brain error in a row, a model worker that died, ai-phone stopping, the `call` command failing; `detail` names it). An outbound call that was never answered has `ended_by` null, with these exceptions: `hangup_command` when `hangup` cancelled it (outcome `cancelled`); `error` when ai-phone stopped (outcome `cancelled`) or the `call` handler failed (outcome `rejected`). An incoming call that ends before its audio starts has `ended_by: them` and `answered_at` null. The transcript holds one entry per line spoken: the disclosure and the model's opening sentence are two `agent` entries.

| Exit code | Meaning |
|---|---|
| 0 | relay call answered; phone agent call answered and ended; `listen`, `say` or `hangup` did its job |
| 2 | usage: an argument-parser error, a request the CLI or the control server refuses as malformed, or (for `call`) a config file that does not load |
| 5 | busy: the other end (486/600), or a call already in progress in the daemon |
| 6 | no answer within `--timeout-sec` (CANCEL sent) |
| 7 | rejected (other 4xx–6xx; no line can dial the target; the target diverts back to this agent and `--yes` was not given), or cancelled by `hangup` while still ringing |
| 8 | redirected (3xx, not followed) |
| 9 | daemon not running, gone, or replying in a way the command does not understand; also `call` after Ctrl-C when the daemon does not confirm the end in time, and `hangup` when the call has still not ended 40 s after the hangup (30 s, then a second hangup and 10 s more): the call may still be up |
| 10 | `listen` timed out |
| 11 | no such call (only the last 20 are kept), no call in progress, the call has ended or is not connected, or `listen`/`say` on a phone agent call |
| 12 | another `listen` is already waiting for incoming calls |
| 13 | the target matches an emergency number (§14b.8) |
| 14 | the daily outbound limit is reached |
| 130 | Ctrl-C while `listen`, `say` or `hangup` waits; a `listen` is cancelled, a `say` or `hangup` already sent carries on in the daemon |

Ctrl-C during `call` sends `hangup` for that call (by its ID, once `created` has arrived) and waits up to about 7 s for the call's end (a 2 s `hangup` request, then a 5 s cut-off); the command then exits with the call's code (0 if it had been answered, 7 if it was still ringing), or 9 if the daemon never confirmed.

### 14b.2 Call records (`calls.py`)

The daemon creates a `CallRecord` when placing an outgoing call or after successfully answering an incoming call, in either mode: an ID of 6 random lowercase hex characters, unique among the records held; `mode` (`agent` | `relay`); `direction` (`in` | `out`); the line; the remote party (the target as given for an outbound call; for an incoming one, the caller's SIP URI as PJSIP reports it, which may carry a display name); the prompt; created, answered and ended times; `outcome`, `detail` and `ended_by`; the transcript; and the relay event queue. The `CallRegistry` keeps the call in progress and the last 20 that ended, in memory only, so a restart forgets them. Nothing new is written to disk for a call: each turn record in `turns.jsonl`, which already held the transcribed and the spoken text, now also carries `call` and `mode`, and `active_call` in `state.json` carries `call`, `mode` and `direction`.

An incoming caller who cancels before answering succeeds gets no record and does not consume a waiting relay listener. The telephony layer treats an already-ended call or PJSIP's session-terminated answer error as a normal cancellation; unrelated answer errors still propagate. The queued native end event releases the single-call claim, so cancellation does not bypass teardown or leave a phantom relay result.

The record, not the control connection, owns the call: a relay `call` returns at answer and the call goes on; a phone agent `call` whose client disappears runs to its natural end. The controller writes the transcript: the opening line (disclosure or greeting), each final transcript of the other person, each reply as it was spoken (without the end marker; a cut reply is marked `interrupted`), and the relay fillers and goodbye.

Relay events: when the other person finishes speaking, the text is queued as `heard`, and unread `heard` texts are merged in order, so one `listen` returns everything said since the last one. Once the call has ended and its unread speech has been read, `listen` returns `ended` with the result; `listen --call ID` keeps returning it. A `listen` cancelled before its reply went out (its client left) puts the event back for the next one. The registry remembers whether anyone has been handed a relay call's end (a `listen` or a `hangup` reply); `listen` with no call in progress reports the most recent unreported end first, and `say` or `hangup` with no call in progress names that call in its error.

### 14b.3 Who writes the replies (`pipeline/replies.py`)

`TurnController` takes a reply source instead of calling the model itself. `ModelReplies` (phone agent mode) asks the LLM worker to answer each finished utterance with the call's system prompt and history, as before. `RelayReplies` (relay mode) hands the transcript to the call record as `heard`, and the controller goes back to `LISTENING` at once. Everything else is shared by both modes: endpointing, barge-in, the timers, the brain-error path, ending.

`say` text takes the path the model's tokens take: the clause chunker, TTS, playback in `SPEAKING`, barge-in. Texts are queued, and each is spoken once the line is free: the controller is listening, the other person is not mid-utterance, and nothing else is being said. The text counts as `said` when it has played and `interrupted` when the other person talked over it; the reply is kept in the history and the transcript either way, marked when cut. If the call ends first, `say` fails with exit 11. An end marker inside `say` text is removed and not obeyed.

Incoming calls: when a relay listener is waiting and no call is in progress, the call runs in relay mode and the listener receives `incoming`; otherwise it runs in phone agent mode with `inbound.instructions`. Either way ai-phone answers and plays `inbound.greeting` first.

### 14b.4 Instructions and the prompt cache (`pipeline/prompts.py`)

The model's system prompt is, in order:

1. `PHONE_INSTRUCTIONS`, fixed text in code: an AI assistant speaking on a live phone call; short, natural sentences, one or two at a time; no lists, markdown, emoji or stage directions; no invented facts, and say so plainly when something is not known; when the purpose of the call is done or the other person says goodbye, a final "Goodbye." followed by one `[END]`; ordinary replies and questions must not carry the marker;
2. for the local brain only, the host line (`You are running on <hostname> (<machine>).`);
3. the header "This call:";
4. the call's section: on an outbound call the `--prompt` text, or, without one, "You are calling on behalf of the owner of this phone number. Be polite and brief."; on an incoming call `inbound.instructions`, or, when that is blank, the default instructions (find out who is calling and what they need, and offer to take a message). An incoming call never gets the outbound default.

The LLM worker caches parts 1 to 3. At warm-up it tokenises the prompt with two probe sections that differ from their first character, evaluates their common token prefix once and keeps that snapshot; each turn starts from a copy of it and evaluates only the call's section, the history and the new turn. Each turn therefore pays for the call's own section and the history, never for the fixed instructions. A remote brain receives the same prompt without the host line.

The end marker: the clause chunker removes [END], in any letter case, from what is spoken, and never speaks a fragment of it. A marker alone does not authorize a hangup. The controller requires exactly one terminal marker, no question in the reply, and a complete final closing sentence such as "Goodbye.", "Thanks, bye." or "Goodbye, thanks." before ending with ended_by: agent after all audio drains. Ordinary replies, empty replies, quoted or negated farewell phrases, and markers followed by more text keep the line open. Explicit hangup commands and existing timeout/error handling retain their own end reasons.

This is a conservative check of reply structure, not a determination that the call's goal has actually been completed. A model that incorrectly says a valid goodbye can still end the call. Unknown closing phrases leave the call connected; the prompt asks for an explicit final "Goodbye." to avoid relying on a broad keyword match. A caller farewell is not required, so an agent can still close a completed outbound task.

### 14b.5 Openings

- **Outbound, both modes.** The disclosure "Hi, this is an AI assistant calling." (synthesised at daemon start; not configurable) plays as soon as the call's audio is up, in `OUTBOUND_OPENING`. It always plays in full: speech during it does not cut it (the speech is kept and handled right after), and neither does a brain error or a model worker that dies (the apology or the goodbye follows it).
- **Phone agent mode.** As the disclosure starts, the model is asked for its first sentence: a cue in the history says the other person has just answered and the disclosure has been said, and asks for one sentence on why the call is being made. The model writes it while the disclosure plays; its audio is held until the disclosure has finished and then follows without a gap. What the other person said during the disclosure decides what happens next:
  - speech that ended before the disclosure did, with less than 1.5 s of held audio (the "Hello?" at pickup), is treated as a greeting: the opening plays as planned, and the held audio is dropped, with no transcript line and one INFO log line ("short speech during the disclosure treated as a greeting");
  - longer speech, or speech still going when the disclosure ends, drops the opening, and the model answers what they said instead;
  - a brain error while the opening is held drops it without a trace: the history records `[interrupted]` and the transcript gets no line, since nothing of it was heard.
- **Relay mode.** The disclosure plays, then the calling AI's first `say`.
- **Incoming, both modes.** `inbound.greeting` (synthesised at start) plays on answer. It can still be interrupted: speech over it stops it and becomes the caller's first turn.

### 14b.6 Relay wait and `hangup --say`

When a `heard` has been handed over, the controller listens again and waits for the calling AI:

- after `calls.relay_filler_after_ms` (1500) without a `say` it plays "One moment.", and at `calls.relay_second_filler_sec` (10 s) "Still with you, one moment."; each plays at most once between two `say`s;
- at `calls.relay_reply_timeout_sec` (25 s) it plays "Sorry, I have to go. Goodbye." and hangs up once that has played (after 5 s at most), with `ended_by: timeout`;
- the wait restarts whenever the other person finishes speaking again, so a caller who keeps talking does not run down the calling AI's time; a `say` ends the wait;
- while it waits, the silence prompt ("Still there?") and the silence hangup are off;
- speech while waiting never cancels anything: it becomes more `heard` text, merged with any text not yet read;
- on an outbound relay call, before the calling AI's first `say`, there are no fillers and the timeout counts from the answer, even if the other person speaks first;
- speech that starts while the previous utterance is still being transcribed is held, not treated as barge-in, and is transcribed as soon as the pending transcript is in;
- barge-in during a `say` works as it does for the model's replies: playback stops within the barge-in budget and `say` returns `interrupted`.

The three phrases are synthesised at daemon start, like the apology and the goodbye.

`hangup --say TEXT` queues the text like a `say` whose end reason is `hangup_command`: the call ends once the text has played, and also if the other person interrupts it. If the call has not ended within 30 s, the daemon hangs up (a second time for a plain `hangup`) and waits 10 s more; if it still has not ended, `hangup` fails with exit 9 ("call … did not end after the hangup"), and the call may still be up. `hangup` without text hangs up at once, with the same waits. Hanging up sends a BYE on an answered call and a CANCEL on a call that is still ringing. `hangup` on a call that is still ringing cancels it without speaking any `--say` text: the call ends with outcome `cancelled`, `call` sends a `cancelled` progress event, and the blocked `call` command exits 7.

### 14b.7 The relay listener

`listen` with no call in progress and no `--call` takes the registry's single listener slot and waits; a second one is refused with exit 12. The slot is freed when the wait ends (an incoming call, the timeout), and at once when the client disconnects: the control server watches the `listen` connection for end-of-file and cancels the handler. An incoming call that arrives just as the wait times out is still reported, since it is already in relay mode. If the client disconnects at the very moment an incoming call is being handed to it, that call runs in relay mode with nobody listening: the caller hears the greeting and, once they speak, the fillers and then the relay goodbye at the timeout (or, if they say nothing, the silence prompt and the silence hangup).

### 14b.8 Safety

- **Emergency numbers are never dialled** (`call_guard.is_emergency`, checked after the outbound enable flag and before any dial work, so every client is covered; exit 13). A target is refused when its digits equal 000, 100, 101, 102, 103, 108, 110, 112, 118, 119, 190, 911 or 999: alone, behind a 1–3 digit country code, or behind an international access prefix (00 or 011) and such a country code. A `sip:` target is checked by its user part when that part has no letters (only digits, spaces and `+ . _ - ( )`). Extensions that look like an emergency number (100, 1000, 4112) are refused too; call such extensions by a name with letters.
- **Daily limit** (`calls.daily_limit`; 0 means no limit; exit 14). It is checked before dialling. Every dial attempt the SIP stack accepted counts, busy and unanswered calls included, per local day. The count lives in `~/.ai-phone/calls-today.json` (`{"date", "count"}`, written to a temporary file, synced and renamed), so a restart does not reset it. It is a file of its own because the daemon rewrites `state.json` whole on every registration change. A missing or malformed file, or one from another day, reads as 0. If the count cannot be written after the dial (a full disk, an unwritable `~/.ai-phone`), the call is hung up at once and refused with exit 7; its record ends `rejected` with `ended_by: error` and the detail "could not update the daily call counter: …", so no attempt escapes the limit. A `call` handler that fails for any other reason also hangs up a call that has not been answered, so the phone never goes on ringing with nobody behind it.
- **Call length** stays capped by `conversation.max_call_sec` (600 s) in both modes; the silence, RTP and brain-error rules are unchanged.
- **Where transcripts go.** Transcripts leave ai-phone only through the command that asked for them (each turn also stays in `turns.jsonl`, as before), and, with a brain that is not on this machine (`brain.kind: openai_compatible`, `claude` or `chatgpt`, §14a.7), through that endpoint — or through the person's own logged-in CLI, to Anthropic or OpenAI — during the call: what callers say, the call's system prompt and the conversation so far go there on every turn in phone agent mode. If the calling AI runs on a cloud model, that service sees them. The README says so next to the examples.

### 14b.9 Configuration

`calls.allow_inbound` and `calls.allow_outbound` independently enable receiving and placing calls. Both default to
true for existing configurations, and at least one must be true. Disabled incoming calls receive SIP 403 before
ringing or claiming the call. Disabled outgoing commands return exit 7 before dialing, creating a call record or
using the daily counter. Idle `listen` is also refused when receiving is disabled, but listening, speaking and
hanging up on an existing outbound relay call (including reading its ended result) still work.

The daemon always starts explicitly with `ai-phone run`. It must stay running while receiving calls is wanted;
outbound-only users may start it when needed and stop it after their calls. No background service or automatic
per-call startup is installed. Trunk registration still runs in outbound-only mode. Configuration changes take
effect after restarting the daemon.

```yaml
inbound:                       # incoming calls: every caller hears the greeting; the instructions apply when no AI listens
  greeting: "Hi, you've reached an AI assistant. How can I help?"   # 1 to 500 characters; synthesised at start
  instructions: "…"            # at most 4000 characters; the default asks who is calling and offers to take a message
calls:
  allow_inbound: true          # receive calls while the daemon is running
  allow_outbound: true         # place calls; at least one direction must be enabled
  daily_limit: 0               # outbound calls per local day; 0 = no limit
  relay_filler_after_ms: 1500  # more than 0
  relay_second_filler_sec: 10  # must come after relay_filler_after_ms
  relay_reply_timeout_sec: 25  # must be longer than relay_second_filler_sec
```

An explicit `inbound:` section wins over anything converted from an older config. A Milestone 1 `conversation.greeting` and `conversation.system_prompt` become `inbound.greeting` and `inbound.instructions`, with a warning. How a Milestone 2 config is converted is in the decision log (§15, 2026-09-16). `lines add` writes nothing about what the agent says.

### 14b.10 Decisions made while building

Each came out of planning or a review:

- The daily counter lives in `calls-today.json`, not `state.json` (§14b.8).
- `listen` replies use the key `kind`, not `event`, so the CLI's progress handling never mistakes a result for a progress event.
- The end reasons `rtp_timeout`, `brain_error` and model-worker errors map to `ended_by: error`, with the reason in `detail`.
- In relay mode the controller returns to `LISTENING` as soon as it hands a `heard` over, with a "waiting for the calling AI" flag that turns off the silence prompt. That is how speech while waiting never cancels anything: it simply becomes more `heard` text.
- The relay wait restarts at every new `heard`; each filler still plays at most once between two `say`s.
- On an outbound relay call, before the first `say`, there are no fillers and the timeout counts from the answer.
- Relay speech that arrives while the previous utterance is still being transcribed is held, not treated as barge-in, and replayed (and finalised, if it has already ended) once the pending transcript arrives. Otherwise barge-in would cancel the transcript and lose the caller's words.
- The model's opening sentence is generated while the disclosure plays; its audio is held until the disclosure has finished.
- `hangup --say` reaches the controller as a `say` with the end reason `hangup_command`, so an interruption still ends the call; the daemon hangs up anyway after 30 s, and `hangup` reports exit 9 if the call has still not ended 10 s after that.
- A `call` cancelled by `hangup` before it was answered ends with outcome `cancelled` (exit 7), not `completed`.
- The daily limit counts every dial attempt the SIP stack accepted: every attempt can cost money.
- `listen` with no call in progress first reports a relay call that ended since anyone last listened; a `listen` whose client disconnects is cancelled at once, so it never holds the listener slot.
- "Busy" is decided by the daemon's one-call claim; the control server only checks the shape of each request.
- Argument-parser errors stay plain text on stderr with exit 2, even under `--json`; the JSON contract covers everything a command reports itself.
- Extensions such as 100 and 1000 stay blocked by the emergency rule; such extensions are dialled by name.
- A `state.json` write that fails (a full disk) is logged as a warning and never stops a call; at the end of a call the line is released and the record ended before the state is written.

## 15. Decision log

| Date | Decision | Status |
|---|---|---|
| 2026-10-27 | Add Phonon 31M English through direct MLX, with the bundled local voice and 80 ms streaming chunks. Setup selects the engine; existing configs keep Kokoro. Same framing, generation cancellation and offline worker boundary. Call latency and interruption are measured separately from the standalone TTS benchmark | [D] |
| 2026-09-02 | Complete appliance (Approach B) with the weekend prototype as Milestone 1 inside it | [D] |
| 2026-09-02 | Python; Go rejected by owner; TS/Swift/Rust evaluated and rejected | [D] |
| 2026-09-02 | PJSIP via pjsua2, vendored into the platform wheel; AGPLv3 | [D] |
| 2026-09-02 | Hand-rolled asyncio pipeline for M1; Pipecat only if it deletes more code than it adds | [D] |
| 2026-10-01 | **A subscription can be the brain** (`brain.kind: claude` / `chatgpt`, §14a.7): ai-phone drives the CLI the person is already logged into (Claude Code's stream-json session, `codex app-server`'s JSON-RPC), because that is the only way to use a paid plan without an API key. One session per call in an empty private directory, with no tools and none of their own settings; `llm.prepare` starts it while the greeting plays. Cost if wrong: a brain slower to the first word than the model on this Mac (measured 2.4–5.5 s for Haiku, 3.9–4.4 s for gpt-5.6-luna, against 874 ms p50 locally), and the conversation leaves the Mac — both documented, and the daemon says so at start | [D] |
| 2026-09-02 (amended 2026-09-03) | LLM in the daemon's own worker process; no Ollama or any external model server is required. Since Milestone 2 the user may configure an OpenAI-compatible endpoint as the brain (§14a.7); that is their choice, announced at start, not a dependency | [D] |
| 2026-09-02 | mlx-audio for STT/TTS/VAD on Mac; parakeet-mlx fallback only | [D] |
| 2026-09-02 | Media port at 16 kHz; VAD in main process; three workers; shared-memory audio, JSON control | [D]/[P] as marked |
| 2026-09-02 | `lan` mode includes loopback and link-local | [D] |
| 2026-09-02 | English-only, single-call, no web UI, no SMS through launch | [D] |
| 2026-09-03 | Premise 2 refined: **the conversation never touches the cloud; getting a number (and opt-in SMS) may, at setup or message time.** Four line kinds (provisioned via own carrier key, forwarded existing number, SIP address for agent-to-agent, bring-your-own trunk); a provisioned number belongs to the agent. Milestone 2 becomes "lines" | [D] |
| 2026-09-03 | **M2 code and docs complete** on branch m2-lines-design (Tasks 1–15 of the M2 plan, each independently reviewed with 1–4 fix rounds); the two carrier spikes and the gate (Tasks 0a, 0b, 16) wait for the owner's Telnyx and Twilio accounts, so no M2 numbers are recorded yet | [D] |
| 2026-09-03 | **M2 design approved** (builder mode, approach C: carrier-free daemon + provider protocol with Telnyx then Twilio, transactional provisioning, outbound with caller ID and AI disclosure, pluggable OpenAI-compatible brain, minimal control socket for `ai-phone call`). Receptionist demoted to a persona preset; no flagship job. Invariants amended: HTTP(S) to one configured brain endpoint allowed during a call; one daemon, several lines, one call. Registration retry = PJSIP's own scheme (5 s, then 30 s, ±10 s jitter), not exponential. Config applies at restart in M2; live reload and IP-change handling move to M4 | [D] |
| 2026-09-02 | Home Assistant is an integration, not an add-on | [D] |
| 2026-09-02 | Spike B (9B, quiet machine, no telephony): stt 26.9 ms, llm_ttft 221, llm_first_clause 408, tts_first 161, trio 600 ms p50 → projected first-audible ≈ 1000 ms, no margin | [D] |
| 2026-09-03 | **M1 gate PASSED, 4B tier, 20/20 LAN PCMU calls: first_audible p50 874 ms / p95 1438** (endpoint 297, STT 96, LLM prompt-eval 229, LLM first clause 312, TTS first chunk 164, egress 9; 0 underruns) with the host swapping ~10 GB. Barge-in, 5 interrupting callers: stop 218–259 ms after onset, all under 300; interrupting sentences transcribed whole. **Default tier is t4.** An earlier 977 ms figure was an artifact (ring geometry bug stamped t5 early) and is withdrawn; the 9B tier does not fit beside normal apps on 24 GB | [D] |
| 2026-09-03 | Endpointer noise floor is an asymmetric tracker (falls fast, rises slowly) so a speech onset the VAD flags late cannot seed it; speech heard during the greeting is kept whether or not the barge-in detector fires (it is replayed when the greeting ends or is cut). Before this fix, callers who spoke over the greeting lost everything said before it ended | [D] |
| 2026-09-03 | Shared-memory rings: every counter single-writer; geometry stored in the header (macOS page-rounds segments, so attach() must not infer the frame count); publish/consume fence via a lock round-trip; the tx ring is flushed only on the PJSIP thread, by request. Audio frames crossing a generation boundary carry the gen_id (STT input, TTS output). STT worker drains its whole ring per push. Call end cancels the workers' generation and filters events by call_id | [D] |
| 2026-09-02 | pjsua2 build: 18 s, not 15–30 min; needs `and_string.i` too and `-L` for keg-only openssl@3/opus; codec ids carry a channel suffix (`PCMU/8000/1`), match by prefix; `setNullDev()` required; release Account/port before `libDestroy` or PJSIP aborts at exit | [V] |
| 2026-09-02 | parakeet checkpoint loads as float32 (2.47 GB); cast to float16 (1.24 GB), identical transcripts, same speed; `generate()` takes an `mx.array` | [D] |
| 2026-09-02 | mlx-audio Kokoro fetches voices from `prince-canuma/Kokoro-82M` (hard-coded), not the model repo; `doctor` must pre-fetch `voices/<voice>.safetensors` from it | [V] |
| 2026-09-03 | Qwen3.5's prompt cache is hybrid (24 ArraysCache + 8 KVCache) and cannot be trimmed; the system prefix is evaluated once and each turn starts from a deep copy (0.2 ms); history is re-evaluated each turn | [D] |
| 2026-09-03 | Speech during the greeting stops it (barge-in on greeting); STT worker owns draining its ring and the daemon cancels the workers' generation at call end (otherwise consecutive calls' audio blended); TTS resamples in its worker; STT partials not streamed in M1; first TTS unit is 4–8 tokens, later units 6–12 | [D] |
| 2026-09-03 | Dev-machine finding: the 9B tier (≈8.5 GB with fp16 parakeet) swaps beside a normal app load on 24 GB; minimum-hardware tiers stand (16 GB → t4) | [D] |
| 2026-09-16 | **Personas replaced by per-call prompts; relay mode added** (Milestone 3, the agent tool, §14b). Any AI with a shell drives ai-phone: phone agent mode (`ai-phone call --prompt`, ai-phone's model holds the conversation, the transcript is returned) or relay mode (`call --relay`, `listen`, `say`, `hangup`; the calling AI holds the conversation). The `personas:` section, each line's `persona` key and `outbound_opening` are removed: an outbound call carries its own prompt and opens with the fixed disclosure "Hi, this is an AI assistant calling.", and incoming calls use one `inbound:` section. A config with `personas:` still loads: the first line's persona (or `default`) supplies `inbound.greeting` and `inbound.instructions` (from its `system_prompt`), the other personas and `outbound_opening` are dropped, line `persona` keys are ignored, and one warning says so. `lines add` no longer writes `persona`; `call --persona` and a control-socket `persona` field are refused with exit 2. Emergency numbers are refused in the daemon; `calls.daily_limit` is counted in `calls-today.json` | [D] |
| 2026-09-16 | **Roadmap renumbered:** M3 agent tool, M4 packaging, M5 local dashboard (calls, transcripts, line status, logs; reachable only from the machine) and HTTP API, M6 launch, then post-launch work with no milestone number. Rows above this one use the old numbers (M3 packaging, M4 control plane, M5 launch, M6 post-launch). Not in M3: an MCP server (a thin wrapper, later), SMS, keypad tones, more than one call | [D] |
| 2026-09-17 | **M4 is "get started", not packaging.** No Homebrew tap, Docker image, LaunchAgent or PyPI release: people clone the repo, run scripts/install.sh, then ai-phone setup (full-screen wizard, plain prompts without a terminal). Carriers can connect a number the person already has; ai-phone never releases a number it did not buy. The voice is configurable (tts:). Models download into the Hugging Face cache with ai-phone models pull. Doctor, install-service, live reload and IP-change handling move to later work | [D] |

## 16. Verified facts (2026-09-02; Milestone 2 rows added 2026-09-03)

pjproject 2.17 (2026-04-22); Homebrew bottle contents (static libs, headers, `.pc`, `pjsua`, no `.i`); `pj::AudioMediaPort` with `createPort`/`onFrameRequested`/`onFrameReceived`; PJSUA conference bridge default 16 kHz, resamples mismatched ports; `Endpoint::audDevManager().setNullDev()` and `setNullDev2(bool)`; `libRegisterThread`; SWIG `-threads` in pjproject's Python Makefile; `pjsua` flags `--null-audio --play-file --auto-play --auto-play-hangup --local-port`; PyPI `pjsua2` is a stale 2.12 sdist (2023); GPLv2-or-later → AGPLv3 via GPLv3 §13; `mlx-community/parakeet-tdt-0.6b-v2` and `-v3`; `mlx-community/nemotron-3.5-asr-streaming-0.6b`; `mlx-community/Kokoro-82M-bf16` with `voices/af_heart.*`, 24 kHz; `mlx-community/Qwen3.5-9B-MLX-4bit` and `Qwen3.5-4B-MLX-4bit` (both with vision tower, dropped by mlx-lm at load); Qwen3.5 chat template honours `enable_thinking=False`; mlx-audio 0.5.1 ships parakeet (streaming), nemotron_asr, kokoro, silero VAD; misaki 0.9.4 `Requires-Python <3.13` (uv ignores, pip enforces) and runtime `spacy.cli.download` of `en_core_web_sm`; llama-cpp-python 0.3.35 sdist-only on PyPI; kokoro-onnx requires Python < 3.14; Pipecat 1.8.1 has no SIP transport; Docker bridge networking advertises the container IP in SDP.

Milestone 2 (2026-09-03, verified against the carriers' published pages and the installed pjsua2 headers; the spikes re-verify the carrier rows live): Telnyx credential SIP connections support SIP registration with inbound delivered to the registered device, STUN `stun.telnyx.com:3478`, regional proxies `sip.telnyx.com` 192.76.120.10/64.16.250.10 (US), `sip.telnyx.eu` 185.246.41.140/141 (EU) and the other regions in `providers/telnyx.py`; Telnyx `POST /v2/number_orders` accepts `connection_id` and `customer_reference`, orders may stay pending and some countries need regulatory data, and only recently searched numbers can be ordered; Telnyx outbound caller ID must be a Telnyx number or a Verified Number; Twilio SIP Domains support SIP registration, inbound number → TwiML `<Dial><Sip>` (twimlets echo hosts static TwiML), Elastic SIP Trunking termination at `<trunk>.pstn.twilio.com` with credential-list auth, origination needs a public SIP URI, outbound caller ID must be a Twilio number or a Verified Caller ID keyed on the phone, signalling ranges 54.172.60.0/30, 54.244.51.0/30, 54.171.127.192/30, 35.156.191.128/30, 54.65.63.192/30, 54.169.127.128/30, 54.252.254.64/30, 177.71.206.192/30, number prices from the Pricing API; iPhone diverts use `**21*`/`**61*`/`**62*`/`**67*<number>#`, `##002#` cancels all; pjsua2 2.17 exposes `AccountRegConfig` (`registrarUri`, `timeoutSec`, `retryIntervalSec`, `firstRetryIntervalSec`, `randomRetryIntervalSec`, `dropCallsOnFail`, `unregWaitMsec`, `registerOnAdd`, `proxyUse`), `AccountNatConfig` (`contactRewriteUse`, `viaRewriteUse`, `sdpNatRewriteUse`, `udpKaIntervalSec`), `AccountSipConfig.transportId`/`proxies`, `UaConfig.stunServer`/`nameserver`, `TransportConfig.tlsConfig` (`certFile`, `privKeyFile`, `CaListFile`, `verifyServer`), `AccountMediaConfig.transportConfig.publicAddress`, `srtpUse`, `srtpSecureSignaling`, `Call::makeCall`, `SipTxOption.localUri`, `onCallRedirected` with `PJSIP_REDIRECT_STOP`; `PJSUA_REG_USE_OUTBOUND_PROXY`/`PJSUA_REG_USE_ACC_PROXY` are `#define`s (exposed as constants by our SWIG interface); `pjsip_cfg()->tcp.keep_alive_interval` and `->tls.keep_alive_interval` are `long` (default `PJSIP_TCP_KEEP_ALIVE_INTERVAL`, 90 s).

## 17. Glossary

**SIP** Session Initiation Protocol, the signalling standard for calls. **RTP** Real-time Transport Protocol, the media stream. **G.711 / PCMU / PCMA** the 8 kHz telephone codecs (μ-law, A-law). **INVITE / BYE / REGISTER** SIP requests to start a call, end a call, and announce reachability. **403 / 486** SIP responses Forbidden and Busy Here. **DID** a phone number from a trunk provider. **PBX** a local phone switch. **Trunk** the SIP service that connects to the phone network. **STUN** a way to learn your public address behind NAT. **VAD** voice activity detection. **Endpointing** deciding the caller has finished a sentence. **Barge-in** the caller talking over the agent. **G2P** grapheme-to-phoneme, the text front end of a TTS model. **Prompt cache** reusing the model's computation for an unchanged prompt prefix. **[D]/[P]/[V]** see the notation at the top.

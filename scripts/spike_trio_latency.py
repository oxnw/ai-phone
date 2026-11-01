# scripts/spike_trio_latency.py — measure STT / LLM / TTS stage latencies with no telephony.
# Confirmed API (mlx-audio 0.5.x, mlx-lm 0.31.3): mlx_audio.tts.utils.load_model / .stt.utils.load_model;
# tts.generate(text=, voice=, speed=, lang_code=) yields results with .audio (24 kHz float32);
# stt.generate(mx.array 16 kHz float32) -> result with .text (numpy input fails: it calls .astype(mx.bfloat16)); mlx_lm.load / stream_generate(... prompt_cache=) yields .text.
import os, sys, time, json, statistics
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np, soxr
import mlx.core as mx

LLM_ID = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen3.5-9B-MLX-4bit"
SYSTEM = "You are ai-phone, a phone assistant running locally. Answer in one short sentence."
CLAUSE_PUNCT = set(".!?,;:")
def ms(a, b): return (b - a) / 1e6
def to_np(x): return np.array(mx.array(x).astype(mx.float32)).reshape(-1)   # bf16-safe

from mlx_audio.tts.utils import load_model as load_tts
tts = load_tts("mlx-community/Kokoro-82M-bf16")
def tts_run(text, voice="af_heart"):
    t = time.monotonic_ns(); first = None; chunks = []
    for res in tts.generate(text=text, voice=voice, speed=1.0, lang_code="a"):
        chunks.append(to_np(res.audio))
        if first is None: first = time.monotonic_ns()
    return ms(t, first), np.concatenate(chunks) if chunks else np.zeros(0, np.float32)

from mlx_audio.stt.utils import load_model as load_stt
stt = load_stt("mlx-community/parakeet-tdt-0.6b-v2")
def stt_final_ms(pcm16k):
    t = time.monotonic_ns(); out = stt.generate(mx.array(np.asarray(pcm16k, np.float32)))   # parakeet expects mx.array
    return ms(t, time.monotonic_ns()), (getattr(out, "text", None) or str(out)).strip()

from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache
model, tok = load(LLM_ID)
def llm_first_clause_ms(user_text):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_text}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
    cache = make_prompt_cache(model)
    t = time.monotonic_ns(); t_first = None; t_clause = None; pending = 0; text = ""
    for r in stream_generate(model, tok, prompt, max_tokens=120, prompt_cache=cache):
        if t_first is None: t_first = time.monotonic_ns()
        pending += 1; text += r.text
        if t_clause is None and ((any(c in r.text for c in CLAUSE_PUNCT) and pending >= 6) or pending >= 12):
            t_clause = time.monotonic_ns()
    return ms(t, t_first), ms(t, t_clause or t_first), text.strip()

print("warming up", flush=True)
tts_run("Warm up."); llm_first_clause_ms("Hello?"); stt_final_ms(np.zeros(16000, dtype=np.float32))
print("warm; running utterances", flush=True)
rows = []
for line in open("scripts/utterances.txt"):
    line = line.strip()
    if not line: continue
    _, audio = tts_run(line, voice="am_adam")                       # a different voice plays the caller
    pcm16k = soxr.resample(audio, 24000, 16000)
    pcm8k = soxr.resample(soxr.resample(pcm16k, 16000, 8000), 8000, 16000)   # simulate G.711 bandwidth
    stt_ms, heard = stt_final_ms(pcm8k)
    ttft, clause, reply = llm_first_clause_ms(heard)
    tts_ms, _ = tts_run(reply.split(".")[0] + ".")
    rows.append(dict(utt=line, heard=heard, stt_ms=round(stt_ms, 1), llm_ttft_ms=round(ttft, 1), llm_clause_ms=round(clause, 1),
                     tts_ms=round(tts_ms, 1), reply=reply, sum_ms=round(stt_ms + clause + tts_ms, 1)))
    print(json.dumps(rows[-1]), flush=True)

def p(v, q): return statistics.quantiles(v, n=100)[q - 1] if len(v) > 1 else v[0]
print("SUMMARY", LLM_ID)
for k in ("stt_ms", "llm_ttft_ms", "llm_clause_ms", "tts_ms", "sum_ms"):
    v = [r[k] for r in rows]; print(f"{k:14s} p50={p(v,50):7.1f}  p95={p(v,95):7.1f}")
print("NOTE: sum_ms excludes the 300 ms endpoint and ~100 ms transport rows; gate for the trio alone is 600 ms p50.")

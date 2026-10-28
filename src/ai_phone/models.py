"""The models ai-phone runs, whether they are on this Mac, and the setup-time download (`ai-phone models pull`).
Weights live in the Hugging Face cache, where the backends load them from. The daemon never downloads."""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

STT = "mlx-community/parakeet-tdt-0.6b-v2"
TTS = "mlx-community/Kokoro-82M-bf16"
PHONON = "gradium/phonon-31m-en"
PHONON_REVISION = "92741c5b07d085ff840e26fea8ce43110b87852f"
PHONON_FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer.model", "default-voice.safetensors")
BIG_MEMORY_GB = 32
DOWNLOADING = ".ai-phone-downloading"  # in a repo's cache folder while `pull` has not finished it
SHARD_RE = re.compile(r"-of-(\d+)\.safetensors$")

log = logging.getLogger("ai-phone.models")


@dataclass(frozen=True)
class LocalModel:
    """One entry of the curated list of brains that run on this Mac."""
    key: str           # what `models.tier` holds in the config
    repo: str          # the Hugging Face repo the weights come from
    name: str          # what the wizard shows
    weights_gb: float  # what the download costs
    needs_gb: int      # the memory it wants; less than this and macOS swaps, which makes it answer slowly
    tested: bool       # ai-phone has made calls with it
    note: str


CATALOG: dict[str, LocalModel] = {m.key: m for m in (
    LocalModel("t4", "mlx-community/Qwen3.5-4B-MLX-4bit", "Qwen3.5 4B", 3.1, 16, True, "tested · first word in 0.9 s"),
    LocalModel("t9", "mlx-community/Qwen3.5-9B-MLX-4bit", "Qwen3.5 9B", 6.0, 32, True, "tested · smarter, swaps below 32 GB"),
    LocalModel("qwen2b", "mlx-community/Qwen3.5-2B-MLX-4bit", "Qwen3.5 2B", 1.8, 8, False, "untested for calls · for 8 GB Macs"),
    LocalModel("llama3b", "mlx-community/Llama-3.2-3B-Instruct-4bit", "Llama 3.2 3B", 1.8, 16, False, "untested for calls"),
    LocalModel("gemma4b", "mlx-community/gemma-3-4b-it-4bit", "Gemma 3 4B", 3.4, 16, False, "untested for calls"),
    LocalModel("mistral7b", "mlx-community/Mistral-7B-Instruct-v0.3-4bit", "Mistral 7B", 4.1, 24, False, "untested for calls"),
    LocalModel("gemma12b", "mlx-community/gemma-3-12b-it-4bit", "Gemma 3 12B", 8.1, 32, False, "untested for calls"),
)}
LLM = {key: m.repo for key, m in CATALOG.items()}  # the tier → repo view the backends and older code use


class ModelsError(RuntimeError):
    pass


def memory_gb() -> int:
    try:
        return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], timeout=5)) // 2**30
    except (OSError, subprocess.SubprocessError, ValueError):
        return 0


def recommended_tier(mem_gb: int) -> str:
    return "t9" if mem_gb >= BIG_MEMORY_GB else "t4"


def known_tier(tier: str) -> bool:
    """Whether `models.tier` names something runnable, without asking this Mac how much memory it has."""
    return tier == "auto" or tier in CATALOG or "/" in tier


def tier_help() -> str:
    return f"auto, one of {', '.join(CATALOG)}, or a Hub repo id like org/name"


def resolve_tier(tier: str) -> str:
    """The catalog key (or raw repo id) to run: "auto" picks by memory, anything with a "/" is a repo someone
    named by hand in the config."""
    if tier == "auto":
        return recommended_tier(memory_gb())
    if known_tier(tier):
        return tier
    raise ModelsError(f"unknown model {tier!r}\nUse {tier_help()}.")


def repo_for(tier: str) -> str:
    key = resolve_tier(tier)
    entry = CATALOG.get(key)
    return entry.repo if entry else key


def fits(key: str, mem_gb: int) -> bool:
    """Whether this Mac has the memory the model wants. A repo named by hand is unknown, so it is not refused."""
    entry = CATALOG.get(key)
    return True if entry is None else entry.needs_gb <= mem_gb


def tts_repo(engine: str = "kokoro") -> str:
    try:
        return {"kokoro": TTS, "phonon": PHONON}[engine]
    except KeyError:
        raise ModelsError(f"unknown speech engine {engine!r}; use kokoro or phonon") from None


def _revision(repo: str) -> dict[str, str]:
    return {"revision": PHONON_REVISION} if repo == PHONON else {}


def _snapshot_options(repo: str) -> dict:
    # The Hub checks completeness against this selection even offline. GGUF and chat-template files
    # belong to other runtimes; requiring them would reject a complete direct-MLX snapshot.
    return {**_revision(repo), "allow_patterns": list(PHONON_FILES)} if repo == PHONON else {}


def needed(tier: str, brain_kind: str, tts_engine: str = "kokoro") -> list[str]:
    return [STT, tts_repo(tts_engine)] + ([repo_for(tier)] if brain_kind == "local" else [])


def local_path(repo: str) -> Path | None:
    """The cached snapshot, if it is complete: no `pull` of it was left unfinished, it holds a config and weights,
    and every shard its index names is there. (The cache links each file in as it arrives, so a download stopped
    part-way leaves a snapshot folder behind.)"""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    if (_repo_dir(repo) / DOWNLOADING).exists():
        return None
    try:
        path = Path(snapshot_download(repo, local_files_only=True, **_snapshot_options(repo)))
    except LocalEntryNotFoundError:
        return None  # not downloaded: the ordinary answer, and the one `missing` asks for on every run
    except Exception as e:
        # unusable all the same, but a cache the process may not read looks exactly like one that holds nothing,
        # and that is the difference somebody staring at "not downloaded" needs
        log.debug("%s: the Hugging Face cache would not answer: %s: %s", repo, type(e).__name__, e)
        return None
    if repo == PHONON:
        # A voice alone is also a safetensors file. Check the actual weights and tokenizer, so a partial
        # snapshot cannot trigger Pocket TTS's fallback downloads when the worker starts offline.
        return path if all((path / name).is_file() and (path / name).stat().st_size for name in PHONON_FILES) else None
    if not (path / "config.json").is_file() or not any(path.glob("*.safetensors")):
        return None
    return path if _weights_complete(path) else None


def _weights_complete(path: Path) -> bool:
    """Whether the snapshot holds a whole set of weights, judged on the shard filenames: one of them names its own
    total (`model-00001-of-00002.safetensors` says there are two). That is the only description of the weights
    that cannot go stale — several mlx-community repos ship a `model.safetensors.index.json` left over from a
    different layout (gemma-3-4b-it-4bit names two shards beside its one consolidated file; gemma-3-12b-it-4bit
    names five beside its two real ones), and both load fine. A download stopped part-way leaves its files named
    `.incomplete`, so the glob does not count them."""
    if (path / "model.safetensors").is_file():
        return True  # one consolidated file is the whole model
    present: dict[int, set[str]] = {}
    for f in path.glob("*.safetensors"):
        found = SHARD_RE.search(f.name)
        if found:
            present.setdefault(int(found.group(1)), set()).add(f.name)
    if not present:
        return any(f.is_file() for f in path.glob("*.safetensors"))  # some other naming: check its target exists
    total = max(present)
    return len(present[total]) == total


def missing(tier: str, brain_kind: str, tts_engine: str = "kokoro") -> list[str]:
    return [r for r in needed(tier, brain_kind, tts_engine) if local_path(r) is None]


def _hub_root() -> Path:
    from huggingface_hub import constants
    return Path(constants.HF_HUB_CACHE)


def _repo_dir(repo: str) -> Path:
    return _hub_root() / ("models--" + repo.replace("/", "--"))


def _bytes_on_disk(repo: str) -> int:
    blobs = _repo_dir(repo) / "blobs"
    total = 0
    try:
        with os.scandir(blobs) as entries:
            for e in entries:
                try:
                    total += e.stat().st_size if e.is_file() else 0
                except OSError:  # a part-file renamed (finished) between the listing and the stat
                    pass
    except OSError:  # no blobs folder yet
        pass
    return total


def download_size(repos: list[str]) -> int | None:
    """Bytes still to fetch: the repos' total size less what the cache already holds. None when the Hub cannot be asked."""
    from huggingface_hub import HfApi
    try:
        total = sum(sum(s.size or 0 for s in HfApi().model_info(r, files_metadata=True, **_revision(r)).siblings or []
                        if r != PHONON or s.rfilename in PHONON_FILES) for r in repos)
    except Exception:
        return None
    return max(total - sum(_bytes_on_disk(r) for r in repos), 0)


def pull(repos: list[str], report: Callable[[int, int], None] | None = None, download=None, poll_sec: float = 0.5) -> None:
    """Download each repo into the Hugging Face cache, calling report(bytes done, bytes total) while it runs.
    report is called from a background thread (a UI must hand the update to its own thread); total is 0 when unknown."""
    from huggingface_hub.utils import are_progress_bars_disabled, disable_progress_bars, enable_progress_bars
    if download is None:
        from huggingface_hub import snapshot_download as download
    total = download_size(repos)
    root = _hub_root()
    probe = next((p for p in (root, *root.parents) if p.exists()), Path.home())
    free = shutil.disk_usage(probe).free
    if total is not None and total + 2**30 > free:
        raise ModelsError(f"not enough disk space: {total / 1e9:.1f} GB to download and {free / 1e9:.1f} GB free "
                          "(keep 1 GB spare)")
    start = sum(_bytes_on_disk(r) for r in repos)
    stop = threading.Event()

    def watch() -> None:
        while not stop.wait(poll_sec):
            try:
                done = max(sum(_bytes_on_disk(r) for r in repos) - start, 0)
                report(done if total is None else min(done, total), total or 0)  # old blobs can push done past total
            except Exception:  # progress is best effort: one bad tick must not end the watcher
                pass

    bars_were_off = are_progress_bars_disabled()
    disable_progress_bars()  # Hugging Face's own bars would draw over our progress line and the wizard
    watcher = threading.Thread(target=watch, daemon=True, name="models-progress")
    if report:
        watcher.start()
    try:
        for repo in repos:
            marker = _repo_dir(repo) / DOWNLOADING  # removed only once this repo is complete: a stop leaves it
            try:
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.touch()
                download(repo, **_snapshot_options(repo))
            except Exception as e:
                raise ModelsError(f"could not download {repo}: {type(e).__name__}: {str(e)[:160]}") from None
            marker.unlink(missing_ok=True)
    finally:
        stop.set()
        if report:
            watcher.join()
        if not bars_were_off:
            enable_progress_bars()
    if report:
        report(total or 0, total or 0)

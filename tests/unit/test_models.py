import contextlib
import inspect
import logging
import os
import socket
import subprocess
import threading
import time
import types
from pathlib import Path

import huggingface_hub
import pytest
from huggingface_hub.errors import LocalEntryNotFoundError

from ai_phone import models
from ai_phone.models import LLM, STT, TTS, ModelsError


def plenty_of_disk(monkeypatch, free=100 * 10**9):
    monkeypatch.setattr(models.shutil, "disk_usage", lambda p: types.SimpleNamespace(total=free, used=0, free=free))


TABLE = [  # the curated list, in the order the wizard offers it
    ("t4", "mlx-community/Qwen3.5-4B-MLX-4bit", "Qwen3.5 4B", 3.1, 16, True, "tested · first word in 0.9 s"),
    ("t9", "mlx-community/Qwen3.5-9B-MLX-4bit", "Qwen3.5 9B", 6.0, 32, True, "tested · smarter, swaps below 32 GB"),
    ("qwen2b", "mlx-community/Qwen3.5-2B-MLX-4bit", "Qwen3.5 2B", 1.8, 8, False, "untested for calls · for 8 GB Macs"),
    ("llama3b", "mlx-community/Llama-3.2-3B-Instruct-4bit", "Llama 3.2 3B", 1.8, 16, False, "untested for calls"),
    ("gemma4b", "mlx-community/gemma-3-4b-it-4bit", "Gemma 3 4B", 3.4, 16, False, "untested for calls"),
    ("mistral7b", "mlx-community/Mistral-7B-Instruct-v0.3-4bit", "Mistral 7B", 4.1, 24, False, "untested for calls"),
    ("gemma12b", "mlx-community/gemma-3-12b-it-4bit", "Gemma 3 12B", 8.1, 32, False, "untested for calls"),
]


def test_the_catalog_holds_the_curated_list_in_order():
    assert [(m.key, m.repo, m.name, m.weights_gb, m.needs_gb, m.tested, m.note) for m in models.CATALOG.values()] == TABLE
    assert all(key == m.key for key, m in models.CATALOG.items())
    assert LLM == {key: repo for key, repo, *_ in TABLE}  # the old tier → repo view still works


def test_resolve_tier_takes_auto_a_catalog_key_or_a_hub_repo(monkeypatch):
    monkeypatch.setattr(models, "memory_gb", lambda: 32)
    assert models.resolve_tier("auto") == "t9"
    assert models.resolve_tier("gemma12b") == "gemma12b"
    assert models.resolve_tier("org/my-own-4bit") == "org/my-own-4bit"  # hand-edited configs may name any repo
    with pytest.raises(ModelsError) as bad:
        models.resolve_tier("t5")
    assert "t5" in str(bad.value) and "gemma12b" in str(bad.value) and "org/name" in str(bad.value)
    assert not models.known_tier("t5") and models.known_tier("auto") and models.known_tier("org/x")


def test_repo_for_follows_the_catalog_and_passes_a_repo_through(monkeypatch):
    monkeypatch.setattr(models, "memory_gb", lambda: 8)
    assert models.repo_for("qwen2b") == "mlx-community/Qwen3.5-2B-MLX-4bit"
    assert models.repo_for("auto") == LLM["t4"]
    assert models.repo_for("org/my-own-4bit") == "org/my-own-4bit"


def test_fits_compares_what_a_model_needs_with_this_Mac():
    assert models.fits("qwen2b", 8) and not models.fits("t4", 8)
    assert models.fits("t9", 32) and not models.fits("gemma12b", 24)
    assert models.fits("org/unknown", 8)  # nothing is known about a hand-picked repo, so it is not refused


def test_needed_depends_on_the_brain():
    assert models.needed("t4", "local") == [STT, TTS, LLM["t4"]]
    assert models.needed("t9", "openai_compatible") == [STT, TTS]
    assert models.needed("gemma4b", "local") == [STT, TTS, "mlx-community/gemma-3-4b-it-4bit"]
    assert models.needed("org/mine", "local") == [STT, TTS, "org/mine"]


def test_recommended_tier(monkeypatch):
    assert models.recommended_tier(24) == "t4"
    assert models.recommended_tier(32) == "t9"
    assert models.recommended_tier(0) == "t4"
    assert models.resolve_tier("t9") == "t9"
    monkeypatch.setattr(models, "memory_gb", lambda: 64)
    assert models.resolve_tier("auto") == "t9"


def test_memory_gb_reads_sysctl_and_falls_back_to_zero(monkeypatch):
    monkeypatch.setattr(models.subprocess, "check_output", lambda *a, **k: b"25769803776\n")
    assert models.memory_gb() == 24

    def fail(*a, **k):
        raise subprocess.TimeoutExpired("sysctl", 5)
    monkeypatch.setattr(models.subprocess, "check_output", fail)
    assert models.memory_gb() == 0


def test_local_path_needs_config_and_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo, **k: str(tmp_path))
    assert models.local_path(TTS) is None
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"x")
    assert models.local_path(TTS) == tmp_path

    def not_cached(repo, **k):
        raise LocalEntryNotFoundError("x")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", not_cached)
    assert models.local_path(TTS) is None


def snapshot(tmp_path, monkeypatch):
    """A cache whose Kokoro snapshot holds a config and one consolidated weights file."""
    hub = tmp_path / "hub"
    snap = hub / "models--mlx-community--Kokoro-82M-bf16" / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "model.safetensors").write_bytes(b"x")
    monkeypatch.setattr(models, "_hub_root", lambda: hub)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo, **k: str(snap))
    return snap


@pytest.fixture
def offline_hub_cache(tmp_path, monkeypatch):
    """An isolated Hub cache and a socket guard for real offline cache discovery tests."""
    from huggingface_hub import constants

    home = tmp_path / "hf-home"
    hub = home / "hub"
    monkeypatch.setenv("HF_HOME", str(home))
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(hub))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)

    def no_network(*args, **kwargs):
        raise AssertionError("offline Hub cache test attempted network access")

    real_socket = socket.socket

    class OfflineSocket(real_socket):
        def connect(self, *args, **kwargs):
            no_network(*args, **kwargs)

        def connect_ex(self, *args, **kwargs):
            no_network(*args, **kwargs)

    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket, "socket", OfflineSocket)
    return types.SimpleNamespace(hub=hub, repo="example/offline-fixture", revision="offline-revision")


def hub_snapshot(cache):
    """Write a tiny cache in the layout Hub resolves for ``repo@main``."""
    repo_dir = cache.hub / "models--example--offline-fixture"
    blobs = repo_dir / "blobs"
    snapshot_dir = repo_dir / "snapshots" / cache.revision
    (repo_dir / "refs").mkdir(parents=True)
    blobs.mkdir()
    snapshot_dir.mkdir(parents=True)
    (repo_dir / "refs" / "main").write_text(cache.revision)
    (blobs / "config-blob").write_bytes(b"{}")
    (blobs / "weight-blob").write_bytes(b"not-real-weights")
    (snapshot_dir / "config.json").symlink_to(Path("../../blobs/config-blob"))
    (snapshot_dir / "model.safetensors").symlink_to(Path("../../blobs/weight-blob"))
    return types.SimpleNamespace(repo_dir=repo_dir, blobs=blobs, snapshot_dir=snapshot_dir)


def test_local_path_discovers_a_complete_hub_snapshot_offline(offline_hub_cache):
    snap = hub_snapshot(offline_hub_cache)
    assert models._hub_root() == offline_hub_cache.hub
    assert models.local_path(offline_hub_cache.repo) == snap.snapshot_dir


def test_local_path_reports_a_missing_hub_repository_offline(offline_hub_cache):
    assert models._hub_root() == offline_hub_cache.hub
    assert models.local_path(offline_hub_cache.repo) is None


def test_local_path_reports_a_snapshot_without_a_revision_reference_offline(offline_hub_cache):
    snap = hub_snapshot(offline_hub_cache)
    (snap.repo_dir / "refs" / "main").unlink()
    assert models.local_path(offline_hub_cache.repo) is None


def test_local_path_rejects_a_snapshot_with_a_missing_weight_blob_offline(offline_hub_cache):
    snap = hub_snapshot(offline_hub_cache)
    (snap.blobs / "weight-blob").unlink()
    assert models.local_path(offline_hub_cache.repo) is None


def test_local_path_says_at_debug_why_the_cache_would_not_answer(tmp_path, monkeypatch, caplog):
    """"Not downloaded" and "downloaded but the cache cannot be read" look the same from the outside, which is
    exactly the problem somebody would be trying to work out; the reason goes to the log, not to the caller."""
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        lambda repo, **k: (_ for _ in ()).throw(PermissionError(13, "Permission denied")))
    with caplog.at_level(logging.DEBUG, logger="ai-phone.models"):
        assert models.local_path(TTS) is None
    assert len(caplog.records) == 1 and caplog.records[0].levelno == logging.DEBUG
    assert TTS in caplog.records[0].getMessage() and "PermissionError" in caplog.records[0].getMessage()


def test_local_path_logs_nothing_when_the_model_is_simply_not_downloaded(tmp_path, monkeypatch, caplog):
    """A missing model is the ordinary case `missing` asks about on every run: it is not worth a line."""
    def not_cached(repo, **k):
        raise LocalEntryNotFoundError("x")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", not_cached)
    with caplog.at_level(logging.DEBUG, logger="ai-phone.models"):
        assert models.local_path(TTS) is None
    assert caplog.records == []


def test_local_path_needs_every_shard_the_filenames_declare(tmp_path, monkeypatch):
    """A shard names its own total: `model-00001-of-00002.safetensors` says there are two. That is the one
    description of the weights that cannot go stale, so it is what completeness is judged on."""
    snap = snapshot(tmp_path, monkeypatch)
    (snap / "model.safetensors").unlink()
    (snap / "model-00001-of-00002.safetensors").write_bytes(b"x")
    assert models.local_path(TTS) is None  # one of the two: a download that stopped part-way
    (snap / "model-00002-of-00002.safetensors").write_bytes(b"x")
    assert models.local_path(TTS) == snap


def test_local_path_ignores_an_index_that_describes_another_layout(tmp_path, monkeypatch):
    """Several mlx-community repos ship a leftover index: gemma-3-4b-it-4bit names two shards beside its one
    consolidated file, gemma-3-12b-it-4bit names five beside its two real ones. Believing it would make `pull`
    fetch those repos again on every run."""
    snap = snapshot(tmp_path, monkeypatch)
    (snap / "model.safetensors").unlink()
    for i in (1, 2):
        (snap / f"model-0000{i}-of-00002.safetensors").write_bytes(b"x")
    index = snap / "model.safetensors.index.json"
    index.write_text('{"metadata": {"total_size": 1}, "weight_map": '
                     + str({f"w{i}": f"model-0000{i}-of-00005.safetensors" for i in range(1, 6)}).replace("'", '"')
                     + "}")
    assert models.local_path(TTS) == snap
    index.write_text("{not json")
    assert models.local_path(TTS) == snap  # unreadable too: the filenames already answered the question


def test_an_interrupted_pull_is_not_downloaded(tmp_path, monkeypatch):
    """Quitting a download leaves the files so far; the next check must still count the model as missing."""
    snap = snapshot(tmp_path, monkeypatch)
    plenty_of_disk(monkeypatch)
    monkeypatch.setattr(models, "download_size", lambda repos: 100)
    marker = tmp_path / "hub" / "models--mlx-community--Kokoro-82M-bf16" / ".ai-phone-downloading"

    def stopped(repo):
        assert marker.exists(), "the marker is written before the download starts"
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        models.pull([TTS], download=stopped)
    assert marker.exists() and models.local_path(TTS) is None and models.missing("t4", "openai_compatible") == [TTS]  # (the fake cache serves STT too)
    with pytest.raises(ModelsError):
        models.pull([TTS], download=lambda repo: (_ for _ in ()).throw(OSError("connection reset")))
    assert models.local_path(TTS) is None
    models.pull([TTS], download=lambda repo: None)
    assert not marker.exists() and models.local_path(TTS) == snap


def test_local_path_trusts_a_consolidated_weights_file_over_a_stale_index(tmp_path, monkeypatch):
    """Some repos ship one `model.safetensors` and leave behind an index naming shards they do not have
    (mlx-community/gemma-3-4b-it-4bit does, and loads fine). The model is whole: the index is not the authority
    when the consolidated file is there, or `pull` would fetch that repo again on every run."""
    snap = snapshot(tmp_path, monkeypatch)  # one consolidated model.safetensors
    (snap / "model.safetensors.index.json").write_text(
        '{"metadata": {"total_size": 8600158944}, "weight_map": {"a.weight": "model-00001-of-00002.safetensors", '
        '"b.weight": "model-00002-of-00002.safetensors"}}')
    assert models.local_path(TTS) == snap
    assert models.missing("t4", "openai_compatible") == []  # so `pull` says it is already downloaded


def test_missing_lists_absent_repos(monkeypatch, tmp_path):
    monkeypatch.setattr(models, "local_path", lambda r: tmp_path if r == STT else None)
    assert models.missing("t4", "local") == [TTS, LLM["t4"]]


def test_download_size_is_what_the_cache_does_not_hold(monkeypatch):
    class Api:
        def model_info(self, repo, files_metadata):
            assert files_metadata
            return types.SimpleNamespace(siblings=[types.SimpleNamespace(size=300), types.SimpleNamespace(size=None)])
    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(models, "_bytes_on_disk", lambda r: 100 if r == STT else 0)
    assert models.download_size([STT, TTS]) == 500

    class Offline:
        def model_info(self, repo, files_metadata):
            raise OSError("no network")
    monkeypatch.setattr(huggingface_hub, "HfApi", Offline)
    assert models.download_size([STT]) is None


def test_bytes_on_disk_counts_the_repo_blobs(monkeypatch, tmp_path):
    monkeypatch.setattr(models, "_hub_root", lambda: tmp_path)
    assert models._bytes_on_disk(TTS) == 0
    blobs = tmp_path / "models--mlx-community--Kokoro-82M-bf16" / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "a").write_bytes(b"x" * 7)
    (blobs / "b.incomplete").write_bytes(b"x" * 3)
    assert models._bytes_on_disk(TTS) == 10


def wait_for(cond, timeout=5.0) -> bool:
    deadline = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.001)
    return True


def test_pull_downloads_each_repo_and_reports(monkeypatch, tmp_path):
    on_disk = {STT: 0, TTS: 0}
    calls, seen = [], []

    def fake(repo):
        calls.append(repo)
        on_disk[repo] += 50
        # stay "downloading" until the watcher has reported the bytes so far
        assert wait_for(lambda: (sum(on_disk.values()), 100) in seen)

    monkeypatch.setattr(models, "download_size", lambda repos: 100)
    monkeypatch.setattr(models, "_bytes_on_disk", lambda r: on_disk[r])
    monkeypatch.setattr(models, "_hub_root", lambda: tmp_path)
    plenty_of_disk(monkeypatch)
    models.pull([STT, TTS], report=lambda done, total: seen.append((done, total)), download=fake, poll_sec=0.005)
    assert calls == [STT, TTS]
    assert (50, 100) in seen and seen[-1] == (100, 100)
    assert all(0 <= done <= total == 100 for done, total in seen)


def test_progress_survives_a_failing_report_and_stays_within_the_total(monkeypatch, tmp_path):
    on_disk = {STT: 0, TTS: 0}
    seen, threads = [], set()

    def report(done, total):
        threads.add(threading.current_thread().name)
        seen.append((done, total))
        if len(seen) == 1:
            raise RuntimeError("the screen went away for a moment")

    def fake(repo):
        on_disk[repo] += 80  # the cache also holds old blobs, so the bytes on disk pass the total
        assert wait_for(lambda: len(seen) >= 2 and seen[-1] == (min(sum(on_disk.values()), 100), 100))

    monkeypatch.setattr(models, "download_size", lambda repos: 100)
    monkeypatch.setattr(models, "_bytes_on_disk", lambda r: on_disk[r])
    monkeypatch.setattr(models, "_hub_root", lambda: tmp_path)
    plenty_of_disk(monkeypatch)
    models.pull([STT, TTS], report=report, download=fake, poll_sec=0.005)
    assert (80, 100) in seen and seen[-1] == (100, 100)
    assert all(done <= 100 for done, _ in seen)
    assert "models-progress" in threads  # report runs on the watcher thread


def test_bytes_on_disk_skips_a_blob_renamed_while_counting(monkeypatch, tmp_path):
    monkeypatch.setattr(models, "_hub_root", lambda: tmp_path)
    blobs = tmp_path / "models--mlx-community--Kokoro-82M-bf16" / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "a").write_bytes(b"x" * 7)

    class Renamed:
        def is_file(self):
            return True

        def stat(self):
            raise FileNotFoundError("b.incomplete")

    real = os.scandir

    @contextlib.contextmanager
    def scandir(path):
        with real(path) as entries:
            yield [Renamed(), *entries]

    monkeypatch.setattr(models.os, "scandir", scandir)
    assert models._bytes_on_disk(TTS) == 7


def test_pull_refuses_when_the_disk_is_too_small(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(models, "download_size", lambda repos: 5 * 10**9)
    monkeypatch.setattr(models, "_hub_root", lambda: tmp_path / "hub")  # not created yet: the check looks at its parent
    plenty_of_disk(monkeypatch, free=10**9)
    with pytest.raises(ModelsError, match="not enough disk space"):
        models.pull([STT], download=calls.append)
    assert calls == []


def test_pull_wraps_download_errors(monkeypatch, tmp_path):
    def fake(repo):
        raise OSError("boom")

    monkeypatch.setattr(models, "download_size", lambda repos: None)
    monkeypatch.setattr(models, "_hub_root", lambda: tmp_path)
    plenty_of_disk(monkeypatch)
    with pytest.raises(ModelsError, match=f"could not download {TTS}: OSError: boom"):
        models.pull([TTS], download=fake)


def test_pull_uses_the_hub_by_default_without_its_progress_bars(monkeypatch, tmp_path):
    from huggingface_hub.utils import are_progress_bars_disabled

    before = are_progress_bars_disabled()
    got = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo, **k: got.append((repo, are_progress_bars_disabled())))
    monkeypatch.setattr(models, "download_size", lambda repos: 0)
    monkeypatch.setattr(models, "_hub_root", lambda: tmp_path)
    plenty_of_disk(monkeypatch)
    models.pull([STT, TTS], poll_sec=0.01)
    assert got == [(STT, True), (TTS, True)]  # its own bars would draw over our progress line and the wizard
    assert are_progress_bars_disabled() == before

    def fail(repo, **k):
        raise OSError("boom")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fail)
    with pytest.raises(ModelsError):
        models.pull([STT])
    assert are_progress_bars_disabled() == before


def test_constants_match_the_backends():
    from ai_phone.services.backends import mlx_kokoro, mlx_lm_backend, mlx_parakeet

    assert LLM["t4"] == mlx_lm_backend.T4
    assert LLM["t9"] == mlx_lm_backend.T9
    assert mlx_lm_backend.make_t4().model_id == LLM["t4"] and mlx_lm_backend.make_t9().model_id == LLM["t9"]
    assert mlx_lm_backend.make_llm("org/anything").model_id == "org/anything"
    assert STT == inspect.signature(mlx_parakeet.ParakeetSTT).parameters["model_id"].default
    assert TTS == inspect.signature(mlx_kokoro.KokoroTTS).parameters["model_id"].default


def test_phonon_selection_keeps_kokoro_out_of_required_models(monkeypatch):
    monkeypatch.setattr(models, "local_path", lambda repo: None)
    assert models.needed("t4", "local", "phonon") == [STT, models.PHONON, LLM["t4"]]
    assert models.missing("t4", "chatgpt", "phonon") == [STT, models.PHONON]
    with pytest.raises(ModelsError, match="unknown speech engine"):
        models.tts_repo("other")


def test_phonon_snapshot_requires_weights_tokenizers_and_voice_at_pinned_revision(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(models, "_hub_root", lambda: tmp_path)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo, **kw: calls.append((repo, kw)) or str(tmp_path))
    for name in models.PHONON_FILES:
        assert models.local_path(models.PHONON) is None
        (tmp_path / name).write_bytes(b"data")
    assert models.local_path(models.PHONON) == tmp_path
    assert all(repo == models.PHONON and kw == {"local_files_only": True, "revision": models.PHONON_REVISION,
                                               "allow_patterns": list(models.PHONON_FILES)}
               for repo, kw in calls)
    for name in models.PHONON_FILES:
        (tmp_path / name).write_bytes(b"")
        assert models.local_path(models.PHONON) is None
        (tmp_path / name).write_bytes(b"data")


def test_phonon_download_and_size_query_use_the_same_pinned_revision(tmp_path, monkeypatch):
    calls = []

    class Api:
        def model_info(self, repo, **kw):
            assert repo == models.PHONON
            assert kw == {"files_metadata": True, "revision": models.PHONON_REVISION}
            return types.SimpleNamespace(siblings=[types.SimpleNamespace(size=100, rfilename="model.safetensors"),
                                                  types.SimpleNamespace(size=1000, rfilename="model.gguf")])

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(models, "_hub_root", lambda: tmp_path)
    plenty_of_disk(monkeypatch)
    assert models.download_size([models.PHONON]) == 100
    models.pull([models.PHONON], download=lambda repo, **kw: calls.append((repo, kw)))
    assert calls == [(models.PHONON, {"revision": models.PHONON_REVISION,
                                    "allow_patterns": list(models.PHONON_FILES)})]

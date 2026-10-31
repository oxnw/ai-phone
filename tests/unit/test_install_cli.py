"""The installed command must select its checkout without changing the caller's working directory."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/install_cli.py"
spec = importlib.util.spec_from_file_location("install_cli", SCRIPT)
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


@pytest.fixture
def installation(tmp_path):
    # Every component is synthetic: no real settings, environment installation, or ai-phone daemon is used.
    source = tmp_path / "checkout ' $(touch INJECTED) $HOME"
    package = source / "src/ai_phone"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "cli.py").write_text('''import json, os, sys
from pathlib import Path

def main():
    data = {"args": sys.argv[1:], "cwd": os.getcwd(), "home": os.environ.get("AI_PHONE_HOME"),
            "python": sys.executable, "source": __file__}
    if "--read" in sys.argv:
        data["text"] = Path(sys.argv[sys.argv.index("--read") + 1]).read_text()
    print(json.dumps(data))
    return int(os.environ.get("LAUNCHER_TEST_EXIT", "0"))
''')
    python = source / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    bin_dir = tmp_path / "local bin ' $HOME"
    return source, python, bin_dir


def test_launcher_selects_source_from_any_cwd_and_preserves_args_paths_and_exit(installation, tmp_path):
    source, python, bin_dir = installation
    launcher = installer.install_launcher(source, python, bin_dir)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "prompt with spaces.txt").write_text("A relative prompt stays relative to this folder.")
    # A local package with the same name must not shadow the installed checkout.
    shadow = elsewhere / "ai_phone"
    shadow.mkdir()
    (shadow / "__init__.py").write_text("raise RuntimeError('wrong checkout')")
    env = {**os.environ, "PATH": "", "AI_PHONE_HOME": str(tmp_path / "settings"),
           "PYTHONPATH": str(elsewhere), "LAUNCHER_TEST_EXIT": "17"}
    args = ["call", "--read", "prompt with spaces.txt", "--prompt", "literal $HOME; $(touch INJECTED)", ""]
    result = subprocess.run([str(launcher), *args], cwd=elsewhere, env=env, capture_output=True, text=True)
    assert result.returncode == 17, result.stderr
    data = json.loads(result.stdout)
    assert data["args"] == args
    assert Path(data["cwd"]) == elsewhere
    assert data["home"] == str(tmp_path / "settings")
    assert data["python"] == str(python)  # do not resolve a venv's Python symlink to its base interpreter
    assert Path(data["source"]) == source / "src/ai_phone/cli.py"
    assert data["text"] == "A relative prompt stays relative to this folder."
    assert not (elsewhere / "INJECTED").exists()
    assert launcher.stat().st_mode & 0o777 == 0o755


def test_install_can_repeat_for_same_checkout_and_refresh_its_environment(installation, tmp_path):
    source, python, bin_dir = installation
    launcher = installer.install_launcher(source, python, bin_dir)
    content = launcher.read_bytes()
    assert installer.install_launcher(source, python, bin_dir) == launcher
    assert launcher.read_bytes() == content
    another_python = tmp_path / "another-python"
    another_python.symlink_to(sys.executable)
    installer.install_launcher(source, another_python, bin_dir)
    result = subprocess.run([str(launcher), "status"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["python"] == str(another_python)
    assert not list(bin_dir.glob(".ai-phone-*"))


@pytest.mark.parametrize("symlink", [False, True])
def test_unrelated_command_is_not_replaced_without_explicit_option(installation, tmp_path, symlink):
    source, python, bin_dir = installation
    bin_dir.mkdir()
    target = bin_dir / "ai-phone"
    old = tmp_path / "old-command"
    old.write_text("unrelated contents")
    if symlink:
        target.symlink_to(old)
    else:
        target.write_bytes(old.read_bytes())
    with pytest.raises(ValueError, match="refusing to replace"):
        installer.install_launcher(source, python, bin_dir)
    assert target.read_bytes() == old.read_bytes()
    installer.install_launcher(source, python, bin_dir, replace=True)
    assert not target.is_symlink() and target.read_text().startswith("#!/bin/sh\n")
    assert old.read_text() == "unrelated contents"


def test_other_checkout_requires_explicit_replacement(installation, tmp_path):
    source, python, bin_dir = installation
    target = installer.install_launcher(source, python, bin_dir)
    original = target.read_bytes()
    other = tmp_path / "another checkout"
    (other / "src/ai_phone").mkdir(parents=True)
    (other / "src/ai_phone/cli.py").write_text("def main(): return 0\n")
    with pytest.raises(ValueError, match="--replace"):
        installer.install_launcher(other, python, bin_dir)
    assert target.read_bytes() == original
    installer.install_launcher(other, python, bin_dir, replace=True)
    assert installer._managed_target(target)["source"] == str(other)


@pytest.mark.parametrize("missing", ["source", "python"])
def test_moved_install_fails_clearly_without_falling_back(installation, tmp_path, missing):
    source, python, bin_dir = installation
    target = installer.install_launcher(source, python, bin_dir)
    if missing == "source":
        (source / "src/ai_phone/cli.py").unlink()
    else:
        python.unlink()
    other = tmp_path / "fallback"
    other.mkdir()
    (other / "ai-phone").write_text("#!/bin/sh\nprintf 'wrong installation'\n")
    (other / "ai-phone").chmod(0o755)
    result = subprocess.run([str(target), "status"], env={**os.environ, "PATH": str(other)},
                            capture_output=True, text=True)
    assert result.returncode == 127
    assert "scripts/install.sh" in result.stderr and "missing" in result.stderr
    assert not result.stdout


def test_invalid_install_does_not_create_launcher(installation):
    source, python, bin_dir = installation
    python.unlink()
    with pytest.raises(ValueError, match="Python environment was not found"):
        installer.install_launcher(source, python, bin_dir)
    assert not bin_dir.exists()


def test_installer_reports_exact_path_step_until_command_is_reachable(installation, tmp_path):
    source, python, bin_dir = installation
    args = [sys.executable, str(SCRIPT), "--source", str(source), "--python", str(python), "--bin-dir", str(bin_dir)]
    env = {**os.environ, "HOME": str(tmp_path), "PATH": "/usr/bin:/bin"}
    missing = subprocess.run(args, env=env, capture_output=True, text=True)
    assert missing.returncode == 0, missing.stderr
    assert "export PATH=" in missing.stdout and "shell's startup file" in missing.stdout
    assert "from any folder" not in missing.stdout
    ready = subprocess.run(args, env={**env, "PATH": str(bin_dir) + ":/usr/bin:/bin"}, capture_output=True, text=True)
    assert ready.returncode == 0, ready.stderr
    assert "Next: ai-phone setup (from any folder)" in ready.stdout
    assert "export PATH=" not in ready.stdout


def test_installer_detects_shadowing_by_an_earlier_command(installation, tmp_path):
    source, python, bin_dir = installation
    earlier = tmp_path / "earlier"
    earlier.mkdir()
    shadow = earlier / "ai-phone"
    shadow.write_text("#!/bin/sh\nexit 0\n")
    shadow.chmod(0o755)
    result = subprocess.run([sys.executable, str(SCRIPT), "--source", str(source), "--python", str(python),
                             "--bin-dir", str(bin_dir)], env={**os.environ, "PATH": f"{earlier}:{bin_dir}"},
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "export PATH=" in result.stdout and "from any folder" not in result.stdout

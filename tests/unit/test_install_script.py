import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "install.sh"


def test_script_parses():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0
    assert os.access(SCRIPT, os.X_OK)


def test_missing_homebrew_is_a_clear_error(tmp_path):
    # a PATH with the basic tools only: no brew, no uv
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    for tool in ("uname", "dirname", "printf", "cat"):
        src = next(p for d in ("/usr/bin", "/bin") if (p := Path(d) / tool).exists())
        (bin_ / tool).symlink_to(src)
    r = subprocess.run(["/bin/bash", str(SCRIPT)], env={"PATH": str(bin_), "HOME": str(tmp_path)}, capture_output=True, text=True)
    assert r.returncode == 1
    assert "Homebrew is required" in r.stderr and "https://brew.sh" in r.stderr


def _mock_install(tmp_path, *, installed=None, build_result="current"):
    """Run only the copied installer: fake brew/uv/build, real Python for the capability probe."""
    import sys

    repo = tmp_path / "checkout"
    scripts = repo / "scripts"
    vendor = repo / "src/ai_phone/vendor"
    native = repo / "native/pjsua2"
    bin_ = tmp_path / "bin"
    for directory in (scripts, vendor, native, bin_):
        directory.mkdir(parents=True)
    (scripts / "install.sh").write_text(SCRIPT.read_text())
    (vendor.parent / "__init__.py").write_text("")
    (vendor / "__init__.py").write_text("")
    if installed is not None:
        source = ("# old importable binding without the verifier\n" if installed == "old" else
                  f"def ai_phone_tls_verifier_version():\n    return {installed}\n")
        (vendor / "pjsua2.py").write_text(source)

    def executable(path, text):
        path.write_text(text)
        path.chmod(0o755)

    executable(bin_ / "uname", '''#!/bin/sh
case "$1" in -s) echo Darwin;; -m) echo arm64;; *) exit 90;; esac
''')
    executable(bin_ / "brew", '''#!/bin/sh
printf 'brew %s\n' "$*" >> "$INSTALL_TEST_LOG"
[ "$1" = list ] || exit 91
''')
    executable(bin_ / "uv", '''#!/bin/sh
printf 'uv %s\n' "$*" >> "$INSTALL_TEST_LOG"
case "$1" in
  sync) exit 0;;
  run)
    shift
    [ "$1" = python ] || exit 92
    shift
    exec "$INSTALL_TEST_PYTHON" -S "$@";;
  *) exit 93;;
esac
''')
    executable(native / "build.sh", '''#!/bin/sh
printf 'native build\n' >> "$INSTALL_TEST_LOG"
[ "$INSTALL_TEST_BUILD_RESULT" != fail ] || exit 94
mkdir -p native/pjsua2/build
if [ "$INSTALL_TEST_BUILD_RESULT" = current ]; then
  printf 'def ai_phone_tls_verifier_version():\n    return 1\n' > native/pjsua2/build/pjsua2.py
else
  printf '# importable binding still missing the capability\n' > native/pjsua2/build/pjsua2.py
fi
printf 'placeholder native artifact\n' > native/pjsua2/build/_pjsua2_test.so
''')
    calls = tmp_path / "calls.log"
    env = {
        "PATH": f"{bin_}:/usr/bin:/bin", "HOME": str(tmp_path),
        "PYTHONPATH": str(repo / "src"), "PYTHONDONTWRITEBYTECODE": "1",
        "INSTALL_TEST_LOG": str(calls), "INSTALL_TEST_PYTHON": sys.executable,
        "INSTALL_TEST_BUILD_RESULT": build_result,
    }
    result = subprocess.run(["/bin/bash", str(scripts / "install.sh")], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=20)
    return result, repo, calls.read_text()


def test_current_native_capability_skips_the_build(tmp_path):
    result, repo, calls = _mock_install(tmp_path, installed=1)
    assert result.returncode == 0, result.stderr
    assert "already built (TLS verifier version 1)" in result.stdout
    assert "native build" not in calls
    assert not (repo / "native/pjsua2/build").exists()
    assert "==> Ready" in result.stdout


@pytest.mark.parametrize("installed", [None, "old", 0, 2])
def test_missing_or_incompatible_native_capability_rebuilds(tmp_path, installed):
    result, repo, calls = _mock_install(tmp_path, installed=installed)
    assert result.returncode == 0, result.stderr
    assert calls.count("native build\n") == 1
    assert (repo / "src/ai_phone/vendor/_pjsua2_test.so").is_file()
    assert "return 1" in (repo / "src/ai_phone/vendor/pjsua2.py").read_text()
    assert "==> Ready" in result.stdout


def test_native_build_failure_stops_before_installing_or_announcing_ready(tmp_path):
    result, repo, calls = _mock_install(tmp_path, installed="old", build_result="fail")
    assert result.returncode == 1
    assert "the SIP library build failed" in result.stderr
    assert "native/pjsua2/README.md" in result.stderr
    assert calls.count("native build\n") == 1
    assert "old importable binding" in (repo / "src/ai_phone/vendor/pjsua2.py").read_text()
    assert not (repo / "src/ai_phone/vendor/_pjsua2_test.so").exists()
    assert "==> Ready" not in result.stdout


def test_rebuilt_library_must_provide_the_required_capability(tmp_path):
    result, repo, calls = _mock_install(tmp_path, installed="old", build_result="stale")
    assert result.returncode == 1
    assert "cannot load TLS verifier version 1" in result.stderr
    assert "native/pjsua2/README.md" in result.stderr
    assert calls.count("native build\n") == 1
    assert (repo / "src/ai_phone/vendor/_pjsua2_test.so").is_file()
    assert "==> Ready" not in result.stdout

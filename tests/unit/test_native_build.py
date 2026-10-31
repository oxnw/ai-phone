"""Exercise linker selection with fake Homebrew tools; no native build is run."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "native" / "pjsua2" / "build.sh"


@pytest.fixture
def build(tmp_path):
    project = tmp_path / "native"
    project.mkdir()
    script = project / "build.sh"
    shutil.copyfile(SCRIPT, script)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    kegs = tmp_path / "kegs"
    for formula in ("pjproject", "openssl@3", "openssl@4", "opus"):
        (kegs / formula / "lib" / "pkgconfig").mkdir(parents=True)
        (kegs / formula / "include").mkdir()
    # Data-only dependencies do not need a library search path.
    (kegs / "ca-certificates").mkdir()
    log = tmp_path / "tools.jsonl"
    stub = f"#!{sys.executable}\n" + '''\
import json
import os
from pathlib import Path
import sys

name = Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ["TOOL_LOG"], "a") as output:
    output.write(json.dumps({"tool": name, "args": args,
                            "pkg_config_path": os.environ.get("PKG_CONFIG_PATH", "")}) + "\\n")
if name == "brew":
    if args == ["deps", "pjproject"]:
        if os.environ.get("DEPS_FAIL"):
            sys.exit(1)
        print("ca-certificates\\n" + os.environ["RECORDED_SSL"] + "\\nopus")
    elif args[:2] == ["--prefix", "--installed"]:
        path = Path(os.environ["KEGS"]) / args[2]
        if not path.is_dir():
            sys.exit(1)
        print(path)
    else:
        raise AssertionError(args)
elif name == "pkg-config":
    if args == ["--static", "--cflags", "libpjproject"]:
        print("-DPJ_TEST")
    elif args == ["--static", "--libs", "libpjproject"]:
        print("-lpjsua2 -lssl -lcrypto -lopus")
    else:
        raise AssertionError(args)
elif name not in {"swig", "clang++"}:
    raise AssertionError(name)
'''
    for name in ("brew", "pkg-config", "swig", "clang++"):
        path = bin_dir / name
        path.write_text(stub)
        path.chmod(0o755)

    def run(ssl="openssl@4", **extra):
        env = {**os.environ, "PATH": f"{bin_dir}:/usr/bin:/bin", "PYTHON": sys.executable,
               "KEGS": str(kegs), "TOOL_LOG": str(log), "RECORDED_SSL": ssl,
               "PKG_CONFIG_PATH": "/unrelated/pkgconfig", **extra}
        result = subprocess.run(["/bin/bash", str(script)], env=env, capture_output=True, text=True)
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        return result, calls

    return run, kegs


@pytest.mark.parametrize("major", [3, 4])
def test_uses_the_installed_pjproject_openssl_when_both_majors_exist(build, major):
    run, kegs = build
    result, calls = run(f"openssl@{major}")
    assert result.returncode == 0, result.stderr
    compiler = next(call for call in calls if call["tool"] == "clang++")
    selected = f"-L{kegs}/openssl@{major}/lib"
    other = f"-L{kegs}/openssl@{7 - major}/lib"
    assert selected in compiler["args"] and other not in compiler["args"]
    assert f"-I{kegs}/openssl@{major}/include" in compiler["args"]
    assert f"-I{kegs}/openssl@{7 - major}/include" not in compiler["args"]
    assert "tls_verify.cpp" in compiler["args"]
    assert "-I." in compiler["args"]
    assert compiler["args"].index(selected) < compiler["args"].index("-lssl")
    assert f"-L{kegs}/opus/lib" in compiler["args"]
    assert f"-L{kegs}/ca-certificates/lib" not in compiler["args"]
    config_path = compiler["pkg_config_path"].split(":")
    assert config_path[0] == str(kegs / "pjproject" / "lib" / "pkgconfig")
    assert str(kegs / f"openssl@{major}" / "lib" / "pkgconfig") in config_path
    assert str(kegs / f"openssl@{7 - major}" / "lib" / "pkgconfig") not in config_path


def test_missing_recorded_dependency_stops_before_building_instead_of_using_other_ssl(build):
    run, kegs = build
    shutil.rmtree(kegs / "openssl@4")
    result, calls = run()
    assert result.returncode != 0
    assert "pjproject dependency openssl@4 is missing" in result.stderr
    assert not any(call["tool"] in {"swig", "clang++"} for call in calls)


def test_dependency_lookup_failure_stops_before_building(build):
    run, _ = build
    result, calls = run(DEPS_FAIL="1")
    assert result.returncode != 0
    assert "could not read pjproject's installed dependencies" in result.stderr
    assert not any(call["tool"] in {"swig", "clang++"} for call in calls)

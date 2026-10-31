"""Expose this checkout's existing Python environment as `ai-phone` on the user's PATH."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
import tempfile

MARKER = "# ai-phone-launcher: "
ROOT = Path(__file__).resolve().parents[1]


def launcher_text(source: Path, python: Path) -> str:
    metadata = json.dumps({"version": 1, "source": str(source), "python": str(python)})
    return f'''#!/bin/sh
{MARKER}{metadata}
source_dir={shlex.quote(str(source))}
python={shlex.quote(str(python))}
if [ ! -f "$source_dir/src/ai_phone/cli.py" ]; then
    printf '%s\\n' "ai-phone: the installed checkout is missing: $source_dir" "Run scripts/install.sh from your ai-phone checkout to reinstall the command." >&2
    exit 127
fi
if [ ! -x "$python" ]; then
    printf '%s\\n' "ai-phone: the installed Python environment is missing: $python" "Run scripts/install.sh from your ai-phone checkout to restore it." >&2
    exit 127
fi
export PYTHONPATH="$source_dir/src${{PYTHONPATH:+:$PYTHONPATH}}"
exec "$python" -P -c 'import sys; from ai_phone.cli import main; sys.exit(main())' "$@"
'''


def _managed_target(path: Path) -> dict | None:
    if path.is_symlink() or not path.is_file():
        return None
    with path.open(encoding="utf-8", errors="replace") as stream:
        first = stream.readline(8192)
        second = stream.readline(8192)
    if first != "#!/bin/sh\n" or not second.startswith(MARKER):
        return None
    try:
        metadata = json.loads(second[len(MARKER):])
    except ValueError:
        return None
    return metadata if isinstance(metadata, dict) and metadata.get("version") == 1 else None


def install_launcher(source: Path, python: Path, bin_dir: Path, *, replace: bool = False) -> Path:
    source = source.expanduser().resolve()
    # Resolving a venv's Python symlink would select the base interpreter and lose its packages.
    python = python.expanduser().absolute()
    bin_dir = bin_dir.expanduser().absolute()
    if not (source / "src/ai_phone/cli.py").is_file():
        raise ValueError(f"ai-phone source was not found in {source}")
    if ":" in str(source):
        raise ValueError("the checkout path cannot contain ':' because it is used in PYTHONPATH")
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError(f"Python environment was not found at {python}; run scripts/install.sh first")
    target = bin_dir / "ai-phone"
    text = launcher_text(source, python)
    if target.exists() or target.is_symlink():
        existing = _managed_target(target)
        if not replace and (existing is None or existing.get("source") != str(source)):
            raise ValueError(f"refusing to replace {target}; it is not a launcher for this checkout. "
                             "Use --replace only if you want this installation to take its place")
        if target.is_dir():
            raise ValueError(f"refusing to replace directory {target}")
    bin_dir.mkdir(parents=True, exist_ok=True)
    pending = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=".ai-phone-", dir=bin_dir,
                                         delete=False) as stream:
            pending = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fchmod(stream.fileno(), 0o755)
            os.fsync(stream.fileno())
        os.replace(pending, target)
        pending = None
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT, help="checkout to run (default: this checkout)")
    parser.add_argument("--python", type=Path, default=Path(sys.executable),
                        help="Python environment to use (default: the interpreter running this installer)")
    parser.add_argument("--bin-dir", type=Path, default=Path.home() / ".local/bin",
                        help="launcher directory (default: ~/.local/bin)")
    parser.add_argument("--replace", action="store_true", help="replace an existing launcher from another checkout")
    args = parser.parse_args(argv)
    try:
        target = install_launcher(args.source, args.python, args.bin_dir, replace=args.replace)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Installed {target}")
    found = shutil.which("ai-phone")
    if found and Path(found).resolve() == target.resolve():
        print("Next: ai-phone setup (from any folder)")
    else:
        print("To use this command in this terminal, run:")
        print(f"  export PATH={shlex.quote(str(target.parent))}:\"$PATH\"")
        print("Add that line to your shell's startup file for future terminals.")
        print("Then run: ai-phone setup")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

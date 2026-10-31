#!/usr/bin/env bash
# native/pjsua2/build.sh — build the pjsua2 Python extension against the Homebrew pjproject bottle.
# Inputs: pjsua2.i and symbols.i (from the pjproject 2.17 source tarball), brew swig + pjproject.
# Output: build/pjsua2.py and build/_pjsua2<EXT_SUFFIX>
set -euo pipefail
cd "$(dirname "$0")"
fail() { echo "error: $*" >&2; exit 1; }
PJ_PREFIX="$(brew --prefix --installed pjproject)" || fail "pjproject is not installed; run brew install pjproject"
# With no options, brew deps reads the installed keg's runtime dependencies.
# The current formula can name a different OpenSSL major from that keg.
DEPENDENCIES="$(brew deps pjproject)" || fail "could not read pjproject's installed dependencies"
DEPENDENCY_LIBS=()
DEPENDENCY_INCLUDES=()
while IFS= read -r dependency; do
  [ -n "$dependency" ] || continue
  prefix="$(brew --prefix --installed "$dependency")" || fail "pjproject dependency $dependency is missing; run brew reinstall pjproject"
  if [ -d "$prefix/include" ]; then
    DEPENDENCY_INCLUDES+=("-I$prefix/include")
  fi
  if [ -d "$prefix/lib" ]; then
    DEPENDENCY_LIBS+=("-L$prefix/lib")
    PKG_CONFIG_PATH="$prefix/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
  fi
done <<< "$DEPENDENCIES"
export PKG_CONFIG_PATH="$PJ_PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
# build for the interpreter that will import the extension: the project environment (or $PYTHON)
PY="${PYTHON:-$(uv python find 2>/dev/null || command -v python3)}"
"$PY" -c 'import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] < (3, 14) else 1)' || { echo "need Python >=3.11,<3.14 (got $("$PY" -V))" >&2; exit 1; }
PY_INC="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
EXT_SUFFIX="$("$PY" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
OUT="${OUT:-build}"
mkdir -p "$OUT"
echo "python: $PY"
swig -c++ -python -threads -w312 -I"$PJ_PREFIX/include" -o "$OUT/pjsua2_wrap.cpp" -outdir "$OUT" pjsua2.i
CXXFLAGS="$(pkg-config --static --cflags libpjproject)"
LIBS="$(pkg-config --static --libs libpjproject)"
# Some keg-only dependencies are named by libpjproject.pc without a search path.
clang++ -std=c++17 -O2 -fPIC -shared -undefined dynamic_lookup \
  $CXXFLAGS -I. -I"$PY_INC" "${DEPENDENCY_INCLUDES[@]}" "$OUT/pjsua2_wrap.cpp" tls_verify.cpp "${DEPENDENCY_LIBS[@]}" $LIBS \
  -o "$OUT/_pjsua2$EXT_SUFFIX"
echo "built: $OUT/_pjsua2$EXT_SUFFIX and $OUT/pjsua2.py"

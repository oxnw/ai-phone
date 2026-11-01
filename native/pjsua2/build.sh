#!/usr/bin/env bash
# native/pjsua2/build.sh — build the pjsua2 Python extension against the Homebrew pjproject bottle.
# Inputs: pinned PJSIP interfaces/Opus adapter, brew swig + pjproject + opus.
# Output: build/pjsua2.py and build/_pjsua2<EXT_SUFFIX>
set -euo pipefail
cd "$(dirname "$0")"
fail() { echo "error: $*" >&2; exit 1; }
PJ_PREFIX="$(brew --prefix --installed pjproject)" || fail "pjproject is not installed; run brew install pjproject"
OPUS_PREFIX="$(brew --prefix --installed opus)" || fail "opus is not installed; run brew install opus"
# With no options, brew deps reads the installed keg's runtime dependencies.
# The current formula can name a different OpenSSL major from that keg.
DEPENDENCIES="$(brew deps pjproject)" || fail "could not read pjproject's installed dependencies"
DEPENDENCY_LIBS=("-L$OPUS_PREFIX/lib")
DEPENDENCY_INCLUDES=("-I$OPUS_PREFIX/include")
PKG_CONFIG_PATH="$OPUS_PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
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
shasum -a 256 -c pjproject-opus.c.sha256 || fail "the bundled Opus adapter checksum does not match"
echo "python: $PY"
swig -c++ -python -threads -w312 -I"$PJ_PREFIX/include" -o "$OUT/pjsua2_wrap.cpp" -outdir "$OUT" pjsua2.i
CXXFLAGS="$(pkg-config --static --cflags libpjproject)"
LIBS="$(pkg-config --static --libs libpjproject)"
OPUS_CFLAGS="$(pkg-config --cflags opus)"
OPUS_LIBS="$(pkg-config --libs opus)"
# The C wrapper emits the upstream factory only if this pjproject lacks Opus.
# Compile as C, preserving the C ABI used by the installed PJMEDIA libraries.
clang -std=c11 -O2 -fPIC $CXXFLAGS $OPUS_CFLAGS "${DEPENDENCY_INCLUDES[@]}" \
  -c opus_factory.c -o "$OUT/opus_factory.o"
# Some keg-only dependencies are named by libpjproject.pc without a search path.
clang++ -std=c++17 -O2 -fPIC -shared -undefined dynamic_lookup \
  $CXXFLAGS -I. -I"$PY_INC" "${DEPENDENCY_INCLUDES[@]}" "$OUT/pjsua2_wrap.cpp" tls_verify.cpp \
  opus_support.cpp "$OUT/opus_factory.o" "${DEPENDENCY_LIBS[@]}" $LIBS $OPUS_LIBS \
  -o "$OUT/_pjsua2$EXT_SUFFIX"
echo "built: $OUT/_pjsua2$EXT_SUFFIX and $OUT/pjsua2.py"

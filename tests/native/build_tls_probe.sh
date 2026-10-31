#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
fail() { echo "error: $*" >&2; exit 1; }
OUTPUT="${1:?usage: build_tls_probe.sh /absolute/output}"
[[ "$OUTPUT" = /* ]] || fail "output must be absolute"
PJ_PREFIX="$(brew --prefix --installed pjproject)" || fail "pjproject is not installed"
DEPENDENCIES="$(brew deps pjproject)" || fail "cannot resolve installed pjproject dependencies"
INCLUDES=(); LIB_DIRS=(); OPENSSL_BIN=""
while IFS= read -r dependency; do
  [ -n "$dependency" ] || continue
  prefix="$(brew --prefix --installed "$dependency")" || fail "missing dependency $dependency"
  [ ! -d "$prefix/include" ] || INCLUDES+=("-I$prefix/include")
  if [ -d "$prefix/lib" ]; then
    LIB_DIRS+=("-L$prefix/lib")
    PKG_CONFIG_PATH="$prefix/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
  fi
  case "$dependency" in openssl|openssl@*) OPENSSL_BIN="$prefix/bin/openssl";; esac
done <<< "$DEPENDENCIES"
[ -n "$OPENSSL_BIN" ] || fail "installed pjproject has no OpenSSL dependency"
export PKG_CONFIG_PATH="$PJ_PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
mkdir -p "$(dirname "$OUTPUT")"
clang++ -std=c++17 -Wall -Wextra -Werror -DAI_PHONE_TLS_VERIFY_TESTS \
  -Inative/pjsua2 "${INCLUDES[@]}" $(pkg-config --static --cflags libpjproject) \
  tests/native/tls_probe.cpp native/pjsua2/tls_verify.cpp \
  "${LIB_DIRS[@]}" $(pkg-config --static --libs libpjproject) -o "$OUTPUT"
# No execution of the newly built binary. Metadata names only public toolchain paths.
PYTHON="${PYTHON:-python3}"
"$PYTHON" - "$OUTPUT.json" "$OPENSSL_BIN" "$(pkg-config --modversion libpjproject)" <<'JSON'
import json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"openssl":sys.argv[2],"pjproject_version":sys.argv[3]},indent=2)+"\n")
JSON

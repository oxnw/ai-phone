#!/usr/bin/env bash
# scripts/install.sh — get a clone of ai-phone ready to run: the SIP library and the Python environment.
# Safe to run again. It never touches ~/.ai-phone and downloads no models (`uv run ai-phone setup` does that).
set -euo pipefail
cd "$(dirname "$0")/.."

step() { printf '\n==> %s\n' "$*"; }
fail() { printf '\nerror: %s\n' "$*" >&2; exit 1; }

[ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ] || fail "ai-phone runs on macOS on Apple Silicon"
command -v brew >/dev/null 2>&1 || fail "Homebrew is required: https://brew.sh"
command -v uv >/dev/null 2>&1 || fail "uv is required: brew install uv (or see https://docs.astral.sh/uv/)"

# the SIP library build (native/pjsua2/build.sh) needs swig, pkg-config (pkgconf) and the pjproject headers and libraries;
# the pjproject formula also brings pjsua, the softphone used for free test calls
step "Installing swig, pkgconf and pjproject with Homebrew"
for formula in swig pkgconf pjproject; do
  if brew list --formula "$formula" >/dev/null 2>&1; then echo "$formula: already installed"; else brew install "$formula"; fi
done

step "Setting up the Python environment"
uv sync

sip_binding_ready() {
  uv run python -c 'from ai_phone.vendor import pjsua2; raise SystemExit(0 if pjsua2.ai_phone_tls_verifier_version() == 1 else 1)' >/dev/null 2>&1
}

step "Building the SIP library"
if sip_binding_ready; then
  echo "already built (TLS verifier version 1)"
else
  PYTHON="$(uv run python -c 'import sys; print(sys.executable)')" ./native/pjsua2/build.sh || fail "the SIP library build failed; see native/pjsua2/README.md"
  mkdir -p src/ai_phone/vendor
  cp native/pjsua2/build/pjsua2.py native/pjsua2/build/_pjsua2*.so src/ai_phone/vendor/
  sip_binding_ready || fail "the SIP library was built but cannot load TLS verifier version 1; see native/pjsua2/README.md"
fi

step "Ready"
echo "Next: uv run ai-phone setup"

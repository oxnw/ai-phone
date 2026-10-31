# native/pjsua2

`pjsua2.i` and `symbols.i` are from pjproject 2.17 (GPLv2-or-later; the tarball's
SHA-256 is recorded in `pjproject-2.17.tar.gz.sha256`). ai-phone is AGPLv3, which
GPLv3 §13 permits for GPLv2-or-later code. The compiled extension ai-phone uses is
built exactly by `build.sh` (`scripts/install.sh` runs it and copies the result into
`src/ai_phone/vendor/`); this directory is the corresponding source (GPLv3 §6(d)).
PJSIP's license text is included in [COPYING](COPYING).

Build on macOS: `brew install swig pkgconf pjproject`, then `./build.sh`. Outputs land in
`build/` (git-ignored). `smoke.py` starts PJSIP headless with the null audio device.

`pjsua2.i` also defines two ai-phone helpers, `ai_phone_set_keepalive(tcp_sec, tls_sec)` and
`ai_phone_get_keepalive()`, exposing PJSIP's process-wide TCP/TLS CRLF keepalive interval (`pjsip_cfg()`),
which pjsua2 does not expose.

`tls_verify.cpp` adds a dedicated TLS transport for the explicit per-line `verified_wildcard` policy. It uses
PJSIP's native `on_verify_cb` during the handshake, before SIP traffic, to check the configured next-hop DNS name,
certificate chain, validity dates and exact or single-label wildcard DNS SAN. There is no Common Name fallback,
partial/multi-label wildcard matching or Python callback. OpenSSL headers and libraries come from pjproject's
actual dependency; no OpenSSL objects cross into Python's `ssl` library.

The binding exposes:

- `ai_phone_tls_verifier_version()`: `1` for the supported PJSIP 2.17/OpenSSL build, otherwise `0`.
- `ai_phone_create_verified_tls_transport(config, expected_host)`: a dedicated ephemeral IPv4 factory and PJSUA
  transport ID, or a negative PJ status. The mandatory verifier is installed before the listener exists.
- `ai_phone_deactivate_tls_verifier()`: rejects new handshakes during shutdown without freeing policy state or
  closing established connections; no new guarded factory can be created until destruction and reset.
- `ai_phone_reset_tls_verifier()`: clears policy state only after `libDestroy()`; it refuses a live endpoint.

Each factory retains one immutable expected hostname through connection retries and destruction. Independently
accepted incoming TLS connections are rejected, including clients without certificates; incoming SIP on a verified
outgoing connection still uses the usual SIP checks. Ordinary strict transports retain PJSIP's verification.

`./scripts/install.sh` checks the verifier version and rebuilds an old importable binding. An opted-in line fails
with rebuild guidance if this capability is unavailable; it never falls back to an unverified transport.

The mandatory loopback cases live in `tests/integration/test_tls_verification.py`. Their isolated native probe
links this same source with test-only observation/fault hooks; those hooks are absent from the installed binding.
The tests generate temporary certificates and use a local DNS fixture without models, accounts or public network
access. All 41 maintained native TLS cases passed, including shutdown rejection. Two tests in
`tests/integration/test_tls_endpoint.py` also pass through the actual Python/SWIG telephony path. The combined
build passed 1,773 unit tests and 10 local SIP call tests. The earlier private proof passed 44 checks.

'''Exercise the real Python binding and Telephony lifecycle against loopback TLS.

The native adapter matrix separately covers wildcard and other certificate cases.
These two cases ensure the Python entry points actually wire a verified factory
into production account registration. No daemon, user settings, models or carrier
account are used; the native child has an isolated home and a bounded lifetime.
'''
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys

import pytest

from tests.integration.ports import SIP_PORT
from tests.integration.tls_certificates import CertificateSet
from tests.integration.tls_peer import TLSSIPPeer

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]

CHILD = r'''
import resource
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import sys

root, result_path, ca_path, peer_port, sip_port, expected = sys.argv[1:]
sys.path[:0] = [str(Path(root) / "src"), root]

from ai_phone.config import LineConfig
from ai_phone.telephony import endpoint, events as ev
from ai_phone.telephony.pj import pj
from ai_phone.telephony.rings import FrameRing
from tests.integration.ports import config

# With no asynchronous nameservers PJSIP uses the system localhost mapping.
# Nothing in this child resolves a non-local name or accesses real configuration.
endpoint._system_resolvers = lambda: []
cfg = config()
assert cfg.sip.bind.udp_port == int(sip_port)
cfg.sip.bind.address = "127.0.0.1"
cfg.sip.tls.ca_file = ca_path
cfg.sip.stun_server = None
cfg.sip.nameservers = []
cfg.log.pjsip_level = 0
cfg.calls.allow_inbound = False
uri = f"sip:localhost:{int(peer_port)};transport=tls"
line = LineConfig(kind="trunk", number="+15555550100", user="alice",
                  registrar=uri, outbound_proxy=uri, dial_domain="localhost",
                  from_user="account", transport="tls", register_via_proxy=True,
                  tls_hostname_policy="verified_wildcard", srtp="off")
cfg.lines = {"tls-test": line}
spec = endpoint.LineSpec.from_config("tls-test", line, "synthetic-secret")
loop = asyncio.new_event_loop()
queue = asyncio.Queue()
rx, tx = FrameRing(None, 20), FrameRing(None, 20)
tel = endpoint.Telephony(cfg, [spec], loop, queue, rx, tx)
report = {"events": [], "expected_success": expected == "success"}

async def run():
    try:
        assert pj.ai_phone_tls_verifier_version() == 1
        tel.start()
        # This is the real bound adapter, not the test observer/fault build.
        assert not hasattr(pj, "ai_phone_tls_test")
        report["reset_while_running"] = pj.ai_phone_reset_tls_verifier()
        assert report["reset_while_running"] is False
        transport_id = tel.line_transport_ids["tls-test"]
        assert transport_id >= 0
        address = str(tel.ep.transportGetInfo(transport_id).localAddress)
        report["guarded_port"] = int(address.rsplit(":", 1)[-1])
        while True:
            item = await asyncio.wait_for(queue.get(), timeout=15)
            report["events"].append({"type": type(item).__name__, **asdict(item)})
            assert not isinstance(item, ev.IncomingCall), "unexpected application call delivery"
            if isinstance(item, ev.RegState) and item.code >= 200:
                report["registration"] = asdict(item)
                if report["expected_success"]:
                    assert item.active and item.code == 200, item
                else:
                    assert not item.active and item.code >= 400, item
                break
    finally:
        tel.stop()
        await asyncio.sleep(0)  # callbacks queued before libDestroy must remain safe
        report["stopped"] = tel.ep is None and not tel.accounts and not tel.line_transport_ids
        report["reset_after_stop"] = pj.ai_phone_reset_tls_verifier()
        assert report["stopped"] and report["reset_after_stop"]

try:
    loop.run_until_complete(asyncio.wait_for(run(), timeout=20))
finally:
    loop.close()
    for ring in (rx, tx):
        ring.close()
        ring.unlink()
Path(result_path).write_text(json.dumps(report, indent=2) + "\n")
'''


def _environment(home: Path) -> dict[str, str]:
    home.mkdir(mode=0o700)
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(home), "TMPDIR": str(home), "AI_PHONE_HOME": str(home / "settings"),
        "AI_PHONE_TEST_SIP_PORT": str(SIP_PORT), "PYTHONUNBUFFERED": "1",
        "HOMEBREW_NO_AUTO_UPDATE": "1", "HOMEBREW_NO_ANALYTICS": "1",
        "HOMEBREW_NO_ENV_HINTS": "1",
    }


@pytest.fixture(scope="module")
def endpoint_certificates(tmp_path_factory):
    temporary = tmp_path_factory.mktemp("python-tls-certificates")
    env = _environment(temporary / "home")

    def brew(*args):
        done = subprocess.run(["brew", *args], env=env, capture_output=True, text=True, timeout=30)
        assert done.returncode == 0, "installed pjproject/OpenSSL toolchain is required"
        return done.stdout.strip()

    dependencies = [name for name in brew("deps", "pjproject").splitlines()
                    if name == "openssl" or name.startswith("openssl@")]
    assert len(dependencies) == 1, "cannot identify pjproject's installed OpenSSL dependency"
    openssl = Path(brew("--prefix", "--installed", dependencies[0])) / "bin" / "openssl"
    certificates = CertificateSet(temporary / "certs", openssl=str(openssl))
    certificates.cases["localhost"] = certificates._issue(
        "localhost", "trusted", ["localhost"], "localhost")
    return certificates


def _kill_group(proc):
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    proc.wait(timeout=5)


def _assert_port_released(port: int, kind=socket.SOCK_STREAM):
    with socket.socket(socket.AF_INET, kind) as sock:
        if kind == socket.SOCK_STREAM:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))


@pytest.mark.parametrize("certificate,success", [("localhost", True), ("wrong_host", False)])
def test_python_telephony_uses_verified_tls_factory(tmp_path, endpoint_certificates, certificate, success):
    assert 5060 not in (SIP_PORT, SIP_PORT + 1), "use isolated integration ports, normally 5080/5081"
    _assert_port_released(SIP_PORT, socket.SOCK_DGRAM)
    _assert_port_released(SIP_PORT)
    result_path, log_path = tmp_path / "result.json", tmp_path / "native.log"
    env = _environment(tmp_path / "home")
    cert = endpoint_certificates.cases[certificate]
    with TLSSIPPeer(cert.cert, cert.key) as peer, log_path.open("w") as output:
        args = [sys.executable, "-I", "-c", CHILD, str(ROOT), str(result_path),
                str(endpoint_certificates.ca), str(peer.port), str(SIP_PORT),
                "success" if success else "failure"]
        proc = subprocess.Popen(args, cwd=tmp_path, env=env, stdout=output,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                _kill_group(proc)
                pytest.fail("Python TLS endpoint child did not complete within 30 seconds")
            assert proc.returncode == 0, log_path.read_text()[-8000:]
            report = json.loads(result_path.read_text())
            assert report["stopped"] and report["reset_after_stop"]
            assert report["reset_while_running"] is False
            assert report["registration"]["active"] is success
            # Require an observed, closed peer connection, never an empty observation.
            peer.wait_for(lambda connections: connections and all(c.closed for c in connections), timeout=5)
            if success:
                assert all(c.server_name == "localhost" for c in peer.connections), peer.summary()
                assert report["registration"]["code"] == 200
                assert any(c.authenticated for c in peer.connections), peer.summary()
                assert any(c.app_bytes > 0 for c in peer.connections)
            else:
                assert report["registration"]["code"] >= 400
                assert not any(c.authenticated for c in peer.connections), peer.summary()
                assert all(c.app_bytes == 0 for c in peer.connections), peer.summary()
                assert not any(e["type"] == "IncomingCall" for e in report["events"])
            assert not peer.errors, peer.errors
            _assert_port_released(report["guarded_port"])
        finally:
            _kill_group(proc)
    assert not peer.errors, peer.errors
    _assert_port_released(SIP_PORT, socket.SOCK_DGRAM)
    _assert_port_released(SIP_PORT)

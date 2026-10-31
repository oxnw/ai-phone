"""Authenticate a real PJSIP trunk, then enforce each possible From policy.

The loopback server validates digest authentication separately from identity.
Its explicit 403 for the wrong From isolates that policy; it is not a simulation
of a particular carrier's authentication retries. All credentials are synthetic.
"""
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

import pytest

from tests.integration.ports import PEER_PORT, SIP_PORT, rtp_range

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
ACCOUNT = "trunktestuser"
PASSWORD = "loopback-only-not-a-secret"
NUMBER = "+15555550100"
TARGET = "+15555550199"
REALM = "ai-phone-loopback"

# Native failures must not abort pytest. No model workers or live configuration
# are involved: this child only drives the production telephony layer.
CHILD = '''\
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import resource
import sys

from ai_phone.config import load_config
from ai_phone.telephony import events as ev
from ai_phone.telephony.endpoint import LineSpec, Telephony
from ai_phone.telephony.rings import FrameRing


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    password = "loopback-only-not-a-secret"
    cfg = load_config(Path(sys.argv[1]), env={"AI_PHONE_TEST_TRUNK_PASSWORD": password})
    spec = LineSpec.from_config("trunk", cfg.lines["trunk"], password)
    loop = asyncio.new_event_loop()
    events = asyncio.Queue()
    rx, tx = FrameRing(None, 100), FrameRing(None, 100)
    tel = Telephony(cfg, [spec], loop, events, rx, tx)
    seen = []

    async def next_event():
        event = await asyncio.wait_for(events.get(), timeout=10)
        seen.append({"type": type(event).__name__, **asdict(event)})
        return event

    async def run():
        try:
            tel.start()
            while True:
                event = await next_event()
                if isinstance(event, ev.RegState):
                    assert event.active and event.code == 200, event
                    break
            call_id = tel.make_call("trunk", "+15555550199", timeout_sec=10)
            while True:
                event = await next_event()
                if isinstance(event, ev.MediaActive):
                    tel.hangup(call_id)
                if isinstance(event, ev.CallStateChanged) and event.state == "ended":
                    assert tel.release_call(call_id)
                    break
        finally:
            tel.stop()
            await asyncio.sleep(0)  # queued cleanup must remain safe after libDestroy

    try:
        loop.run_until_complete(asyncio.wait_for(run(), timeout=25))
    finally:
        loop.close()
        for ring in (rx, tx):
            ring.close()
            ring.unlink()
    Path(sys.argv[2]).write_text(json.dumps(seen))


if __name__ == "__main__":
    main()
'''


def _headers(packet):
    lines = packet.partition("\r\n\r\n")[0].split("\r\n")
    return lines[0], {key.lower(): value.strip() for key, value in
                      (line.split(":", 1) for line in lines[1:] if ":" in line)}


def _from_user(headers):
    match = re.search(r"sip:([^@>]+)@", headers["from"])
    assert match, headers["from"]
    return match[1]


def _digest(header, method, uri, nonce):
    assert header.startswith("Digest "), header
    fields = {key: quoted or plain for key, quoted, plain in re.findall(
        r'(\w+)\s*=\s*(?:"([^"\\]*)"|([^,\s]+))', header[7:])}
    assert fields["username"] == ACCOUNT and fields["realm"] == REALM, fields
    assert fields["uri"] == uri and fields["nonce"] == nonce, fields
    assert fields.get("algorithm", "MD5").upper() == "MD5", fields
    assert fields["qop"] == "auth" and fields["cnonce"] and len(fields["nc"]) == 8, fields

    def md5(text):
        return hashlib.md5(text.encode()).hexdigest()

    expected = md5(":".join((md5(f"{ACCOUNT}:{REALM}:{PASSWORD}"), nonce,
                             fields["nc"], fields["cnonce"], "auth", md5(f"{method}:{uri}"))))
    assert hmac.compare_digest(fields["response"], expected), "invalid SIP digest response"


class TrunkPeer:
    def __init__(self, sock, required_user, wire_log):
        self.sock, self.required_user, self.wire_log = sock, required_user, wire_log
        self.registered = False
        self.unregistered = False
        self.invites = []
        self.accepted_cseq = None
        self.acknowledged = set()
        self.hung_up = False

    def respond(self, address, headers, code, extra=(), body=""):
        reasons = {200: "OK", 401: "Unauthorized", 403: "Forbidden", 407: "Proxy Authentication Required"}
        recipient = headers["to"]
        if ";tag=" not in recipient:
            recipient += ";tag=loopback-trunk"
        response = "\r\n".join([
            f"SIP/2.0 {code} {reasons[code]}", f"Via: {headers['via']}",
            f"From: {headers['from']}", f"To: {recipient}",
            f"Call-ID: {headers['call-id']}", f"CSeq: {headers['cseq']}",
            *extra, f"Content-Length: {len(body.encode())}", "", "",
        ]) + body
        self.wire_log.write("SENT " + response + "\n")
        self.wire_log.flush()
        self.sock.sendto(response.encode(), address)

    def handle(self, packet, address):
        self.wire_log.write("RECEIVED " + packet + "\n")
        self.wire_log.flush()
        first, headers = _headers(packet)
        method, uri, version = first.split()
        assert version == "SIP/2.0"
        if method in ("REGISTER", "INVITE"):
            registration = method == "REGISTER"
            auth_name = "authorization" if registration else "proxy-authorization"
            nonce = "registration-nonce" if registration else "invite-nonce"
            if auth_name not in headers:
                name = "WWW-Authenticate" if registration else "Proxy-Authenticate"
                self.respond(address, headers, 401 if registration else 407, [
                    f'{name}: Digest realm="{REALM}", nonce="{nonce}", algorithm=MD5, qop="auth"'])
                return
            _digest(headers[auth_name], method, uri, nonce)
            if registration:
                assert _from_user(headers) == ACCOUNT
                expiry = headers.get("expires", "300")
                self.unregistered |= expiry == "0" or ";expires=0" in headers["contact"]
                self.registered |= not self.unregistered
                self.respond(address, headers, 200, [f"Contact: {headers['contact']}", f"Expires: {expiry}"])
                return
            assert self.registered, "INVITE arrived before authenticated registration"
            assert uri == f"sip:{TARGET}@127.0.0.1:{PEER_PORT}"
            self.invites.append(_from_user(headers))
            if self.invites[-1] != self.required_user:
                self.respond(address, headers, 403)
                return
            self.accepted_cseq = headers["cseq"].split()[0]
            body = ("v=0\r\no=trunk 1 1 IN IP4 127.0.0.1\r\ns=Trunk identity test\r\n"
                    "c=IN IP4 127.0.0.1\r\nt=0 0\r\n"
                    f"m=audio {rtp_range(PEER_PORT)[0]} RTP/AVP 0\r\n"
                    "a=rtpmap:0 PCMU/8000\r\na=sendrecv\r\n")
            self.respond(address, headers, 200, [f"Contact: <sip:peer@127.0.0.1:{PEER_PORT}>",
                                                "Content-Type: application/sdp"], body)
        elif method == "ACK":
            self.acknowledged.add(headers["cseq"].split()[0])
        elif method == "BYE":
            self.hung_up = True
            self.respond(address, headers, 200)
        else:
            raise AssertionError(f"unexpected SIP method: {method}")


def _cleanup(proc):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            break
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            continue
    proc.wait(timeout=5)
    try:
        os.killpg(proc.pid, 0)
    except ProcessLookupError:
        return
    raise AssertionError(f"test process group {proc.pid} survived cleanup")


@pytest.mark.parametrize("from_user", ["number", "account"])
@pytest.mark.parametrize("required_identity", ["number", "account"])
def test_registered_trunk_authentication_and_from_policy(tmp_path, from_user, required_identity):
    assert 5060 not in range(SIP_PORT, PEER_PORT + 1), "Set AI_PHONE_TEST_SIP_PORT=5080"
    for kind in (socket.SOCK_DGRAM, socket.SOCK_STREAM):
        with socket.socket(socket.AF_INET, kind) as probe:
            if kind == socket.SOCK_STREAM:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", SIP_PORT))
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    low, high = rtp_range(SIP_PORT)
    config = tmp_path / "config.yaml"
    config.write_text(
        f"sip:\n  bind: {{address: '127.0.0.1', udp_port: {SIP_PORT}, tcp_port: {SIP_PORT}, tls_port: {SIP_PORT + 1}}}\n"
        f"rtp: {{port_min: {low}, port_max: {high}}}\n"
        f"lines:\n  trunk:\n    kind: trunk\n    user: {ACCOUNT}\n    number: '{NUMBER}'\n"
        f"    registrar: sip:127.0.0.1:{PEER_PORT}\n    dial_domain: 127.0.0.1:{PEER_PORT}\n"
        f"    from_user: {from_user}\n    password_env: AI_PHONE_TEST_TRUNK_PASSWORD\n"
        f"log:\n  dir: '{tmp_path}'\n  pjsip_level: 4\n")
    config.chmod(0o600)
    runner, result, log = (tmp_path / name for name in ("run_trunk.py", "events.json", "native.log"))
    runner.write_text(CHILD)
    env = {**os.environ, "AI_PHONE_HOME": str(home), "PYTHONUNBUFFERED": "1"}
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock, log.open("w") as output, \
            (tmp_path / "sip.log").open("w") as wire_log:
        sock.bind(("127.0.0.1", PEER_PORT))
        sock.settimeout(0.1)
        peer = TrunkPeer(sock, NUMBER if required_identity == "number" else ACCOUNT, wire_log)
        proc = subprocess.Popen([sys.executable, str(runner), str(config), str(result)],
                                cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            deadline = time.monotonic() + 35
            while proc.poll() is None:
                assert time.monotonic() < deadline, "native SIP child timed out"
                try:
                    packet, address = sock.recvfrom(65535)
                except socket.timeout:
                    continue
                if packet.strip():  # UDP keepalive has no SIP request line
                    peer.handle(packet.decode(), address)
            assert proc.returncode == 0, log.read_text()
            assert peer.registered and peer.unregistered
            assert peer.invites and set(peer.invites) == {NUMBER if from_user == "number" else ACCOUNT}
            events = json.loads(result.read_text())
            progress = [event for event in events if event["type"] == "OutboundProgress"]
            assert progress[0]["state"] == "dialling", events
            matched = from_user == required_identity
            assert progress[-1]["state"] == ("answered" if matched else "rejected"), events
            assert progress[-1]["code"] == (200 if matched else 403), events
            assert any(event["type"] == "MediaActive" for event in events) == matched, events
            assert peer.acknowledged and peer.hung_up == matched
            if matched:
                assert peer.accepted_cseq in peer.acknowledged, "the accepted INVITE was never ACKed"
            assert any(event["type"] == "CallStateChanged" and event["state"] == "ended" for event in events)
        except Exception as error:
            error.add_note(f"Trunk regression logs: {tmp_path}\n" + log.read_text()[-5000:])
            raise
        finally:
            _cleanup(proc)

"""Real TLS verification against the production native adapter, without carriers or models.

Mandatory matrix: 15 certificate/identity cases; 21 coexistence/reconnect/inbound
and shutdown cases; seven certificate parser assertions; three allocation/ownership rollbacks;
and repeated endpoint creation/destruction/reset in the same process. No case skips
when a compiler, OpenSSL or native capability is missing: CI must build the support.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time

import pytest

from tests.integration.tls_certificates import CertificateSet
from tests.integration.tls_peer import LocalDNS, TLSSIPPeer, probe_inbound

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
# Apply RLIMIT_CORE before loading the native executable, without preexec_fn in a threaded parent.
EXEC_NATIVE = ("import os,resource,sys; resource.setrlimit(resource.RLIMIT_CORE,(0,0)); "
               "os.execv(sys.argv[1],sys.argv[1:])")


def isolated_environment(home: Path) -> dict[str, str]:
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    home.chmod(0o700)
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(home), "TMPDIR": str(home),
            "HOMEBREW_NO_AUTO_UPDATE": "1", "HOMEBREW_NO_ANALYTICS": "1", "HOMEBREW_NO_ENV_HINTS": "1"}


def native_command(executable: Path, args) -> list[str]:
    return [sys.executable, "-I", "-S", "-c", EXEC_NATIVE, str(executable), *map(str, args)]


def json_rows(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue  # PJSIP can print its initialization banner before console logging is configured.
        if isinstance(row, dict):
            rows.append(row)
    return rows


@dataclass(frozen=True)
class NativeBuild:
    executable: Path
    openssl: str


@pytest.fixture(scope="module")
def tls_build(tmp_path_factory) -> NativeBuild:
    home = tmp_path_factory.mktemp("tls-build")
    executable = home / "tls-probe"
    script = ROOT / "tests/native/build_tls_probe.sh"
    assert script.is_file(), "build the maintained native TLS probe; do not skip TLS verification"
    result = subprocess.run(["bash", str(script), str(executable)], cwd=ROOT, env=isolated_environment(home),
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, f"native TLS probe build failed:\n{result.stdout}\n{result.stderr}"
    assert executable.is_file(), "native TLS probe builder produced no executable"
    metadata = json.loads(Path(str(executable) + ".json").read_text())
    openssl = metadata["openssl"]
    assert Path(openssl).is_file(), "probe builder must identify the OpenSSL dependency it linked"
    return NativeBuild(executable, openssl)


@pytest.fixture(scope="module")
def tls_certificates(tmp_path_factory, tls_build) -> CertificateSet:
    return CertificateSet(tmp_path_factory.mktemp("tls-certificates") / "certs", openssl=tls_build.openssl)


class Native:
    """One bounded native child, all JSON observations retained, with mandatory clean teardown."""

    def __init__(self, executable: Path, args, home: Path):
        self._condition = threading.Condition()
        self._rows: list[dict] = []
        self._cursor = 0
        self._output: list[str] = []
        self._log = (home / "native.log").open("w")
        self.proc = subprocess.Popen(native_command(executable, args), cwd=home, env=isolated_environment(home),
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, start_new_session=True)

        def read():
            for line in self.proc.stdout:
                self._log.write(line)
                self._log.flush()
                with self._condition:
                    self._output.append(line)
                    self._rows.extend(json_rows(line))
                    self._condition.notify_all()
            with self._condition:
                self._condition.notify_all()

        self.reader = threading.Thread(target=read, name="native-tls-observer", daemon=True)
        self.reader.start()
        try:
            self.ready = self.wait(lambda row: row.get("event") == "ready")
        except BaseException:
            self._kill()
            self.reader.join(2)
            self._log.close()
            raise

    @property
    def rows(self):
        with self._condition:
            return list(self._rows)

    def wait(self, predicate, timeout=10):
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                for index in range(self._cursor, len(self._rows)):
                    row = self._rows[index]
                    if predicate(row):
                        self._cursor = index + 1
                        return row
                if self.proc.poll() is not None:
                    raise AssertionError(f"native probe exited {self.proc.returncode}:\n{''.join(self._output)[-6000:]}")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AssertionError(f"native observation missing:\n{''.join(self._output)[-6000:]}")
                self._condition.wait(min(remaining, 0.2))

    def send(self, command):
        assert self.proc.poll() is None, "native probe exited before its command"
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()

    def activate(self):
        self.send("activate")
        self.wait(lambda row: row.get("event") == "activated")

    def _kill(self):
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.proc.wait(timeout=5)

    def close(self):
        try:
            if self.proc.poll() is None:
                self.send("shutdown")
                try:
                    self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._kill()
                    raise AssertionError("native TLS cleanup timed out")
            self.reader.join(2)
            assert not self.reader.is_alive(), "native output observer did not stop"
            assert self.proc.returncode == 0, "".join(self._output)[-6000:]
            assert any(row.get("event") == "destroyed" for row in self.rows), "native destruction was not observed"
            assert not any(row.get("event") == "overflow" for row in self.rows), "native observation buffer overflowed"
        finally:
            if self.proc.poll() is None:
                self._kill()
            self.reader.join(2)
            self._log.close()
            self.proc.stdin.close()
            self.proc.stdout.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


@contextmanager
def scenario(tmp_path, tls_build, certificates, *, certificate="wildcard", server_certificate=True, extra_args=()):
    case = certificates.cases[certificate]
    with LocalDNS() as dns, TLSSIPPeer(case.cert, case.key) as peer:
        args = ["--ca", certificates.ca, "--dns-port", dns.port, "--peer-port", peer.port, "--threads", 1]
        if server_certificate:
            exact = certificates.cases["exact"]
            args += ["--cert", exact.cert, "--key", exact.key]
        with Native(tls_build.executable, [*args, *extra_args], tmp_path) as native:
            yield native, peer
        assert not dns.errors, dns.errors
        assert not peer.errors, peer.errors
    assert not peer.errors, peer.errors


def registration(native, peer, target, slot, *, success, reason=None):
    previous = len(peer.connections)
    native.send("register " + target)
    row = native.wait(lambda event: event.get("event") == "registration" and event.get("slot") == slot
                      and event.get("value") != 401 and (reason is None or event.get("reason") == reason))
    if success:
        assert row["value"] == 200, row
        peer.wait_for(lambda connections: any(c.authenticated for c in connections[previous:]))
    else:
        assert row["value"] >= 400, row
        # A failure callback or a timeout alone is insufficient: actually observe closed connections and zero bytes.
        peer.wait_for(lambda connections: bool(connections[previous:]) and all(c.closed for c in connections[previous:]))
        assert all(c.app_bytes == 0 for c in peer.connections[previous:]), peer.summary()
    return row


CERTIFICATE_CASES = [
    ("exact", "strict", True), ("exact", "guarded", True), ("wildcard", "guarded", True),
    ("wildcard", "strict", False), ("wrong_host", "guarded", False), ("multi_label", "guarded", False),
    ("partial_wildcard", "guarded", False), ("no_san", "guarded", False), ("misleading_cn", "guarded", False),
    ("expired", "guarded", False), ("future", "guarded", False), ("untrusted", "guarded", False),
    ("broken_chain", "guarded", False), ("complete_chain", "guarded", True), ("wildcard", "wrong_next_hop", False),
]


@pytest.mark.parametrize("certificate,mode,accepted", CERTIFICATE_CASES,
                         ids=[f"{cert}-{mode}" for cert, mode, _ in CERTIFICATE_CASES])
def test_certificate_and_next_hop_verification(tmp_path, tls_build, tls_certificates, certificate, mode, accepted):
    extra = ("--expected-host", "other.carrier.test") if mode == "wrong_next_hop" else ()
    with scenario(tmp_path, tls_build, tls_certificates, certificate=certificate, extra_args=extra) as (native, peer):
        native.activate()
        target = "strict" if mode == "strict" else "guarded"
        registration(native, peer, target, -1 if target == "strict" else 0, success=accepted)
        if not accepted and target != "strict":
            assert any(row.get("event") == "verify" and row.get("value") == 0 for row in native.rows)
    if not accepted:
        assert peer.connections and all(c.closed and c.app_bytes == 0 for c in peer.connections), peer.summary()
        assert not any(row.get("event") == "application_request" for row in native.rows)


@pytest.mark.parametrize("first", ["strict", "guarded"])
def test_strict_and_wildcard_lines_do_not_share_connections(tmp_path, tls_build, tls_certificates, first):
    with scenario(tmp_path, tls_build, tls_certificates) as (native, peer):
        native.activate()
        for target in (first, "guarded" if first == "strict" else "strict"):
            registration(native, peer, target, -1 if target == "strict" else 0, success=target == "guarded")
        assert len(peer.connections) >= 2
        assert any(c.authenticated for c in peer.connections)
        assert any(c.closed and c.app_bytes == 0 for c in peer.connections)


@pytest.mark.parametrize("same_host", [False, True], ids=["different-hosts", "same-host"])
def test_two_guarded_factories_keep_their_own_connections(tmp_path, tls_build, tls_certificates, same_host):
    extra = ("--host2", "tenant.carrier.test") if same_host else ()
    with scenario(tmp_path, tls_build, tls_certificates, extra_args=extra) as (native, peer):
        native.activate()
        registration(native, peer, "guarded", 0, success=True)
        registration(native, peer, "guarded2", 1, success=True)
        assert len([c for c in peer.connections if c.authenticated]) == 2
        assert {c.server_name for c in peer.connections} == ({"tenant.carrier.test"} if same_host else
                                                            {"tenant.carrier.test", "other.carrier.test"})
        for slot in (0, 1):
            assert any(row.get("event") == "verify" and row.get("slot") == slot and row.get("value") == 1
                       for row in native.rows)


def test_changed_next_hop_cannot_borrow_an_authenticated_connection(tmp_path, tls_build, tls_certificates):
    with scenario(tmp_path, tls_build, tls_certificates) as (native, peer):
        native.activate()
        registration(native, peer, "guarded", 0, success=True)
        registration(native, peer, "wronghop", 0, success=False, reason="wrong_hop")
        assert any(row.get("event") == "verify" and row.get("reason") == "next_hop" for row in native.rows)


def test_automatic_reconnect_revalidates_bad_peer_then_recovers(tmp_path, tls_build, tls_certificates):
    with scenario(tmp_path, tls_build, tls_certificates, extra_args=("--retry-sec", 1)) as (native, peer):
        native.activate()
        registration(native, peer, "guarded", 0, success=True)
        invalid_start = len(peer.connections)
        bad = tls_certificates.cases["wrong_host"]
        peer.rotate_certificate(bad.cert, bad.key)
        peer.close_connections()
        native.wait(lambda row: row.get("event") == "verify" and row.get("value") == 0
                    and row.get("reason") == "dns_san", timeout=15)
        peer.wait_for(lambda conns: bool(conns[invalid_start:]) and all(c.closed for c in conns[invalid_start:]))
        rejected = list(peer.connections[invalid_start:])
        assert rejected and all(c.app_bytes == 0 for c in rejected), peer.summary()
        accepted_before = sum(row.get("event") == "verify" and row.get("value") == 1 for row in native.rows)
        good_start = len(peer.connections)
        good = tls_certificates.cases["wildcard"]
        peer.rotate_certificate(good.cert, good.key)
        native.wait(lambda row: row.get("event") == "registration" and row.get("slot") == 0
                    and row.get("value") == 200, timeout=15)
        peer.wait_for(lambda conns: any(c.authenticated for c in conns[good_start:]))
        assert sum(row.get("event") == "verify" and row.get("value") == 1 for row in native.rows) > accepted_before
    assert all(c.closed and c.app_bytes == 0 for c in rejected)


@pytest.mark.parametrize("registered", [False, True], ids=["first-connection", "reconnect"])
def test_deactivation_rejects_new_connections_before_sip(tmp_path, tls_build, tls_certificates, registered):
    with scenario(tmp_path, tls_build, tls_certificates, extra_args=("--retry-sec", 1)) as (native, peer):
        native.activate()
        if registered:
            registration(native, peer, "guarded", 0, success=True)
        native.send("deactivate")
        native.wait(lambda row: row.get("event") == "deactivated" and row.get("value") == 1)
        previous = len(peer.connections)
        if registered:
            # PJSIP's real registration retry must also obey the shutdown boundary.
            peer.close_connections()
            native.wait(lambda row: row.get("event") == "registration" and row.get("slot") == 0
                        and row.get("value", 0) >= 400, timeout=15)
            peer.wait_for(lambda conns: bool(conns[previous:]) and all(c.closed for c in conns[previous:]))
        else:
            registration(native, peer, "guarded", 0, success=False)
        assert any(row.get("event") == "verify" and row.get("slot") == 0 and row.get("value") == 0
                   and row.get("reason") == "inactive" for row in native.rows)
        assert all(c.app_bytes == 0 for c in peer.connections[previous:]), peer.summary()
    rejected = peer.connections[previous:]
    assert rejected and all(c.closed and c.app_bytes == 0 for c in rejected), peer.summary()
    assert not any(row.get("event") == "application_request" for row in native.rows)


def test_authenticated_child_survives_listener_close_until_endpoint_destruction(tmp_path, tls_build, tls_certificates):
    with scenario(tmp_path, tls_build, tls_certificates) as (native, peer):
        native.activate()
        registration(native, peer, "guarded", 0, success=True)
        connection = next(c for c in peer.connections if c.authenticated)
        for close_listener in (False, True):
            if close_listener:
                native.send("close guarded")
                native.wait(lambda row: row.get("event") == "listener_closed")
            call_id = peer.send_options(connection.id)
            native.wait(lambda row: row.get("event") == "application_request" and row.get("slot") == 0)
            peer.wait_for(lambda conns: any(message["status"] == 200 and message["headers"].get("call-id") == [call_id]
                                           for message in conns[connection.id].messages))
    assert connection.closed


@pytest.mark.parametrize("client", ["none", "client_trusted", "client_untrusted"])
@pytest.mark.parametrize("active", [False, True], ids=["before-activation", "active"])
@pytest.mark.parametrize("server_certificate", [False, True], ids=["no-server-certificate", "server-certificate"])
def test_independent_inbound_tls_is_rejected_from_listener_creation(tmp_path, tls_build, tls_certificates,
                                                                   client, active, server_certificate):
    with scenario(tmp_path, tls_build, tls_certificates, server_certificate=server_certificate) as (native, peer):
        if active:
            native.activate()
        kwargs = {} if client == "none" else {"client_cert": tls_certificates.cases[client].cert,
                                               "client_key": tls_certificates.cases[client].key}
        observation = probe_inbound(native.ready["guarded_port"], tls_certificates.ca, **kwargs)
        rejected = native.wait(lambda row: row.get("event") == "inbound_rejected" and row.get("slot") == 0)
        assert rejected["value"] != 0, rejected
        assert observation["received_bytes"] == 0, observation
        assert not peer.connections, "an inactive account unexpectedly connected to the peer"
    assert not any(row.get("event") == "application_request" for row in native.rows)


def run_bounded(executable, args, home, *, timeout=20):
    """Short native modes still run in their own process group and cannot leak children on timeout."""
    process = subprocess.Popen(native_command(executable, args), cwd=home, env=isolated_environment(home),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        output, _ = process.communicate(timeout=timeout)
    except BaseException:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait(timeout=5)
        raise
    (home / "native.log").write_text(output)
    assert process.returncode == 0, output[-6000:]
    rows = json_rows(output)
    assert not any(row.get("event") == "overflow" for row in rows)
    return rows


def assert_listener_released(port):
    assert 1024 <= port <= 65535
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", port))
        listener.listen(1)


def test_certificate_parser_rejects_missing_and_malformed_data(tmp_path, tls_build, tls_certificates):
    rows = run_bounded(tls_build.executable, ["--selfcheck", tls_certificates.cases["exact"].cert], tmp_path)
    results = {row["case"]: row["passed"] for row in rows if row.get("event") == "selfcheck"}
    assert results == {name: True for name in ("missing", "empty", "malformed", "truncated", "missing_host", "valid", "wrong_host")}


@pytest.mark.parametrize("phase", ["before-register", "after-register", "capacity"])
def test_partial_transport_creation_rolls_back_without_leaked_listener(tmp_path, tls_build, tls_certificates, phase):
    exact = tls_certificates.cases["exact"]
    with LocalDNS() as dns, TLSSIPPeer(exact.cert, exact.key) as peer:
        args = ["--ca", tls_certificates.ca, "--cert", exact.cert, "--key", exact.key,
                "--dns-port", dns.port, "--peer-port", peer.port, "--fail-at", phase]
        rows = run_bounded(tls_build.executable, args, tmp_path)
        faults = [row for row in rows if row.get("event") == "fault_rollback"]
        assert len(faults) == 1 and faults[0]["phase"] == phase, rows
        fault = faults[0]
        assert fault["rollback_status"] == 0, fault
        assert (fault["register_status"] != 0) if phase == "capacity" else (fault["register_status"] == 0)
        assert any(row.get("event") == "destroyed" and row.get("value", 0) == 0 and row.get("reset") is True
                   for row in rows)
        assert_listener_released(fault["port"])
        assert not peer.connections and not peer.errors and not dns.errors


def test_repeated_endpoint_cycles_reset_native_policy_and_release_ports(tmp_path, tls_build, tls_certificates):
    exact = tls_certificates.cases["exact"]
    with LocalDNS() as dns, TLSSIPPeer(exact.cert, exact.key) as peer:
        args = ["--ca", tls_certificates.ca, "--cert", exact.cert, "--key", exact.key,
                "--dns-port", dns.port, "--peer-port", peer.port, "--cycles", 3]
        rows = run_bounded(tls_build.executable, args, tmp_path)
        cycles = [row for row in rows if row.get("event") == "cycle_destroyed"]
        assert len(cycles) == 3, rows
        assert {row["cycle"] for row in cycles} == {1, 2, 3}
        for row in cycles:
            assert row["reset"] is True and row["active_reset_rejected"] is True, row
            assert row["ports"], row
            for port in row["ports"]:
                assert_listener_released(port)
        assert not peer.connections and not peer.errors and not dns.errors

"""Loopback-only DNS and TLS/SIP fixtures for certificate verification; stdlib only.

No object binds a socket until start()/context entry. Every received plaintext TLS byte is
retained before parsing, so rejected-peer tests can observe the actual application boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import hmac
from pathlib import Path
import re
import secrets
import socket
import ssl
import struct
import threading
import time


class LocalDNS:
    def __init__(self, names=("tenant.carrier.test", "other.carrier.test", "deep.tenant.carrier.test")):
        self.names = {name.lower().rstrip(".") for name in names}
        if not self.names or any(not name.endswith(".test") for name in self.names):
            raise ValueError("DNS fixture is limited to explicit .test names")
        self.host, self.port = "127.0.0.1", None
        self.questions = []
        self.errors = []
        self._stop = threading.Event()
        self._thread = self._sock = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.stop()

    def start(self):
        if self._sock is not None:
            raise RuntimeError("DNS fixture already started")
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((self.host, 0))
        self._sock.settimeout(0.1)
        self.port = self._sock.getsockname()[1]
        self._thread = threading.Thread(target=self._serve, name="tls-proof-dns", daemon=True)
        self._thread.start()
        return self

    def _answer(self, packet: bytes) -> bytes:
        if len(packet) < 12:
            raise ValueError("short DNS packet")
        ident, flags, count, _, _, _ = struct.unpack("!6H", packet[:12])
        if count != 1 or flags & 0x8000:
            raise ValueError("DNS fixture expects one query")
        labels, cursor = [], 12
        while True:
            if cursor >= len(packet):
                raise ValueError("truncated DNS name")
            n = packet[cursor]; cursor += 1
            if n == 0:
                break
            if n > 63 or cursor + n > len(packet):
                raise ValueError("unsupported/compressed DNS question")
            labels.append(packet[cursor:cursor+n].decode("ascii").lower()); cursor += n
        if cursor + 4 > len(packet):
            raise ValueError("truncated DNS question")
        qtype, qclass = struct.unpack("!2H", packet[cursor:cursor+4]); cursor += 4
        name = ".".join(labels)
        self.questions.append((name, qtype))
        known = name in self.names and qclass == 1
        answer = b""
        if known and qtype == 1:
            answer = b"\xc0\x0c" + struct.pack("!HHIH", 1, 1, 1, 4) + socket.inet_aton(self.host)
        # Known AAAA/SRV queries get authoritative empty answers; unknown names get NXDOMAIN.
        status = 0x8400 | (flags & 0x0100) | (0 if known else 3)
        return struct.pack("!6H", ident, status, 1, bool(answer), 0, 0) + packet[12:cursor] + answer

    def _serve(self):
        while not self._stop.is_set():
            try:
                packet, address = self._sock.recvfrom(4096)
                self._sock.sendto(self._answer(packet), address)
            except socket.timeout:
                continue
            except OSError:
                if not self._stop.is_set():
                    self.errors.append("DNS socket error")
                break
            except (ValueError, UnicodeError) as error:
                self.errors.append(str(error))

    def stop(self):
        self._stop.set()
        if self._sock:
            self._sock.close()
        if self._thread:
            self._thread.join(2)
            if self._thread.is_alive():
                raise RuntimeError("DNS fixture did not stop")


@dataclass
class Connection:
    id: int
    peer_address: tuple
    handshake_ok: bool = False
    tls_version: str | None = None
    server_name: str | None = None
    handshake_error: str | None = None
    received: bytearray = field(default_factory=bytearray)
    messages: list[dict] = field(default_factory=list)
    authenticated: bool = False
    closed: bool = False
    contact: str = ""

    @property
    def app_bytes(self):
        return len(self.received)


class TLSSIPPeer:
    """One listener, many observed connections, certificate rotation on subsequent handshakes."""

    def __init__(self, cert: Path, key: Path, *, challenge=True, username="alice", password="synthetic-secret",
                 host_name="tenant.carrier.test", port=0, max_bytes=262144):
        if not host_name.endswith(".test"):
            raise ValueError("peer hostname must be synthetic .test")
        self.host, self.port, self.host_name = "127.0.0.1", port, host_name
        self.challenge, self.username, self.password = challenge, username, password
        self.realm, self.nonce = "tls-proof.test", secrets.token_hex(16)
        self.max_bytes = max_bytes
        self.connections: list[Connection] = []
        self.errors = []
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._listener = self._thread = None
        self._threads = []
        self._sockets = {}
        self._context = self._new_context(cert, key)

    @staticmethod
    def _new_context(cert, key):
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(str(cert), str(key))
        context.set_servername_callback(lambda sock, name, _: setattr(sock, "_proof_sni", name))
        return context

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.stop()

    def start(self):
        if self._listener is not None:
            raise RuntimeError("TLS fixture already started")
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind((self.host, self.port))
        self._listener.listen(16)
        self._listener.settimeout(0.1)
        self.port = self._listener.getsockname()[1]
        self._thread = threading.Thread(target=self._accept, name="tls-proof-listener", daemon=True)
        self._thread.start()
        return self

    def rotate_certificate(self, cert: Path, key: Path):
        context = self._new_context(cert, key)
        with self._condition:
            self._context = context

    def _accept(self):
        while not self._stop.is_set():
            try:
                raw, address = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                if not self._stop.is_set():
                    self.errors.append("TLS listener socket error")
                break
            with self._condition:
                observation = Connection(len(self.connections), address)
                self.connections.append(observation)
                context = self._context
                self._sockets[observation.id] = raw
                thread = threading.Thread(target=self._serve, args=(raw, context, observation),
                                          name=f"tls-proof-connection-{observation.id}", daemon=True)
                self._threads.append(thread)
                thread.start()
                self._condition.notify_all()

    def _serve(self, raw, context, observation):
        stream = raw
        try:
            raw.settimeout(2)
            stream = context.wrap_socket(raw, server_side=True)
            stream.settimeout(0.1)
            with self._condition:
                self._sockets[observation.id] = stream
                observation.handshake_ok = True
                observation.tls_version = stream.version()
                observation.server_name = getattr(stream, "_proof_sni", None)
                self._condition.notify_all()
            pending = bytearray()
            while not self._stop.is_set():
                try:
                    part = stream.recv(8192)
                except socket.timeout:
                    continue
                if not part:
                    break
                with self._condition:
                    # Observe bytes before parsing. Malformed or partial SIP is not lost.
                    observation.received.extend(part)
                    self._condition.notify_all()
                if len(observation.received) > self.max_bytes:
                    raise ValueError("peer application-byte limit exceeded")
                pending.extend(part)
                while True:
                    message = self._take_message(pending)
                    if message is None:
                        break
                    with self._condition:
                        observation.messages.append(message)
                        self._condition.notify_all()
                    if message["method"] == "REGISTER":
                        allowed = not self.challenge or self._valid_digest(message)
                        if allowed:
                            observation.authenticated = True
                            observation.contact = message["headers"].get("contact", [""])[0]
                            self._respond(stream, message, 200, "OK")
                        elif "authorization" in message["headers"]:
                            self._respond(stream, message, 403, "Forbidden")
                        else:
                            self._respond(stream, message, 401, "Unauthorized")
                        with self._condition:
                            self._condition.notify_all()
                    elif message["method"] == "OPTIONS":
                        self._respond(stream, message, 200, "OK")
        except (OSError, ssl.SSLError) as error:
            # TLS alert/no-certificate failures are expected observations, not harness successes.
            observation.handshake_error = type(error).__name__
        except Exception as error:
            self.errors.append(f"connection {observation.id}: {type(error).__name__}: {error}")
        finally:
            try:
                stream.close()
            finally:
                with self._condition:
                    self._sockets.pop(observation.id, None)
                    observation.closed = True
                    self._condition.notify_all()

    @staticmethod
    def _take_message(buffer):
        # CRLF keepalives count as received bytes but are not SIP messages.
        while buffer.startswith(b"\r\n"):
            del buffer[:2]
        end = buffer.find(b"\r\n\r\n")
        if end < 0:
            return None
        lines = bytes(buffer[:end]).decode("utf-8", "replace").split("\r\n")
        headers = {}
        for line in lines[1:]:
            if ":" not in line:
                raise ValueError("invalid SIP header")
            name, value = line.split(":", 1)
            headers.setdefault(name.lower().strip(), []).append(value.strip())
        size = int(headers.get("content-length", ["0"])[0])
        if size < 0 or size > 65536:
            raise ValueError("invalid SIP content length")
        total = end + 4 + size
        if len(buffer) < total:
            return None
        raw = bytes(buffer[:total]); del buffer[:total]
        first = lines[0].split(" ")
        response = first[0] == "SIP/2.0"
        return {"start_line": lines[0], "method": None if response else first[0],
                "uri": None if response else first[1], "status": int(first[1]) if response else None,
                "headers": headers, "bytes": len(raw)}

    def _valid_digest(self, message):
        value = message["headers"].get("authorization", [""])[0]
        if not value.lower().startswith("digest "):
            return False
        pairs = re.findall(r'(\w+)\s*=\s*(?:"([^"\\]*(?:\\.[^"\\]*)*)"|([^,\s]+))', value[7:])
        fields = {name.lower(): quoted or token for name, quoted, token in pairs}
        if len(fields) != len(pairs):
            return False
        if any(fields.get(k) != v for k, v in {"username": self.username, "realm": self.realm,
                                               "nonce": self.nonce, "uri": message["uri"]}.items()):
            return False
        if fields.get("algorithm", "MD5").upper() != "MD5":
            return False
        md5 = lambda value: hashlib.md5(value.encode(), usedforsecurity=False).hexdigest()
        a1 = md5(f"{self.username}:{self.realm}:{self.password}")
        a2 = md5(f"{message['method']}:{message['uri']}")
        if fields.get("qop") == "auth" and fields.get("nc") and fields.get("cnonce"):
            expected = md5(f"{a1}:{self.nonce}:{fields['nc']}:{fields['cnonce']}:auth:{a2}")
        elif "qop" not in fields:
            expected = md5(f"{a1}:{self.nonce}:{a2}")
        else:
            return False
        return hmac.compare_digest(fields.get("response", ""), expected)

    def _respond(self, stream, request, code, reason):
        headers = request["headers"]
        output = [f"SIP/2.0 {code} {reason}"]
        output.extend("Via: " + value for value in headers.get("via", []))
        output += ["From: " + headers["from"][0], "To: " + headers["to"][0] +
                   ("" if ";tag=" in headers["to"][0] else ";tag=tls-proof"),
                   "Call-ID: " + headers["call-id"][0], "CSeq: " + headers["cseq"][0]]
        if code == 401:
            output.append(f'WWW-Authenticate: Digest realm="{self.realm}", nonce="{self.nonce}", algorithm=MD5, qop="auth"')
        if code == 200 and request["method"] == "REGISTER":
            output.extend("Contact: " + value for value in headers.get("contact", []))
            output.append("Expires: " + headers.get("expires", ["60"])[0])
        output.append("Content-Length: 0")
        stream.sendall(("\r\n".join(output) + "\r\n\r\n").encode())

    def send_options(self, connection_id: int) -> str:
        """Send SIP back over one authenticated outbound connection; return response Call-ID to await."""
        with self._condition:
            observation = self.connections[connection_id]
            if not observation.authenticated:
                raise ValueError("OPTIONS requires an authenticated REGISTER first")
            stream = self._sockets[connection_id]
            contact = re.search(r"<([^>]+)>", observation.contact)
            uri = contact.group(1) if contact else f"sip:{self.username}@{self.host_name}"
            call_id = "options-" + secrets.token_hex(8) + "@peer.test"
            branch = "z9hG4bK-" + secrets.token_hex(8)
            message = (f"OPTIONS {uri} SIP/2.0\r\nVia: SIP/2.0/TLS {self.host_name}:{self.port};branch={branch}\r\n"
                       f"Max-Forwards: 70\r\nFrom: <sip:peer@{self.host_name}>;tag=proof-options\r\n"
                       f"To: <sip:{self.username}@{self.host_name}>\r\nCall-ID: {call_id}\r\nCSeq: 1 OPTIONS\r\n"
                       "Content-Length: 0\r\n\r\n")
            stream.sendall(message.encode())
            return call_id

    def wait_for(self, predicate, timeout=5.0):
        deadline = time.monotonic() + timeout
        with self._condition:
            while not predicate(self.connections):
                if self.errors:
                    raise RuntimeError("; ".join(self.errors))
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("TLS peer observation did not arrive")
                self._condition.wait(remaining)
            return self.connections

    def close_connections(self):
        with self._condition:
            streams = list(self._sockets.values())
        for stream in streams:
            try:
                stream.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            stream.close()

    def summary(self):
        with self._condition:
            return [{"id": c.id, "handshake_ok": c.handshake_ok, "tls_version": c.tls_version,
                     "server_name": c.server_name, "handshake_error": c.handshake_error,
                     "app_bytes": c.app_bytes, "authenticated": c.authenticated, "closed": c.closed,
                     "messages": [{"method": m["method"], "status": m["status"], "bytes": m["bytes"]} for m in c.messages]}
                    for c in self.connections]

    def stop(self):
        self._stop.set()
        if self._listener:
            self._listener.close()
        self.close_connections()
        threads = ([self._thread] if self._thread else []) + self._threads
        deadline = time.monotonic() + 4
        for thread in threads:
            thread.join(max(0, deadline - time.monotonic()))
        if any(thread.is_alive() for thread in threads):
            raise RuntimeError("TLS peer thread did not stop")


def probe_inbound(port: int, ca: Path, *, client_cert: Path | None = None, client_key: Path | None = None,
                  server_name="tenant.carrier.test", timeout=2.0) -> dict:
    """Attempt independent inbound TLS and OPTIONS; caller MUST also inspect native delivery counters.

    TLS 1.3 can let a client finish its side before the server's rejection alert arrives, so a
    successful client send is not proof of delivery and a later TLS alert is not sufficient proof
    of the zero-delivery property. The native probe must supply that independent observation.
    """
    if not server_name.endswith(".test"):
        raise ValueError("inbound fixture requires a synthetic .test hostname")
    if bool(client_cert) != bool(client_key):
        raise ValueError("client certificate and key must be supplied together")
    context = ssl.create_default_context(cafile=str(ca))
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    if client_cert:
        context.load_cert_chain(str(client_cert), str(client_key))
    result = {"handshake_ok": False, "sent_bytes": 0, "received_bytes": 0, "error": None}
    payload = ("OPTIONS sip:alice@tenant.carrier.test SIP/2.0\r\n"
               "Via: SIP/2.0/TLS client.carrier.test;branch=z9hG4bK-inbound-proof\r\n"
               "Max-Forwards: 70\r\nFrom: <sip:client@client.carrier.test>;tag=inbound-proof\r\n"
               "To: <sip:alice@tenant.carrier.test>\r\nCall-ID: inbound-proof@client.test\r\n"
               "CSeq: 1 OPTIONS\r\nContent-Length: 0\r\n\r\n").encode()
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout) as raw:
            with context.wrap_socket(raw, server_hostname=server_name) as stream:
                result["handshake_ok"] = True
                stream.sendall(payload)
                result["sent_bytes"] = len(payload)
                result["received_bytes"] = len(stream.recv(8192))
    except (OSError, ssl.SSLError) as error:
        result["error"] = type(error).__name__
    return result

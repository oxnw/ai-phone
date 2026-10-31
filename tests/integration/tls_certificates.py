"""Synthetic TLS certificate matrix. Call only with a newly-created private temporary directory."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import shutil
import subprocess


@dataclass(frozen=True)
class Certificate:
    cert: Path
    key: Path
    expected_host: str = "tenant.carrier.test"


class CertificateSet:
    """Generate certificates with installed OpenSSL; keys never leave caller's private temporary home."""

    def __init__(self, root: Path, *, openssl: str | None = None):
        self.root = Path(root)
        if self.root.exists() and any(self.root.iterdir()):
            raise ValueError("certificate directory must be empty")
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.chmod(0o700)
        self.openssl = openssl or shutil.which("openssl")
        if not self.openssl:
            raise RuntimeError("installed openssl executable is required")
        self.cases: dict[str, Certificate] = {}
        self.now = datetime.now(timezone.utc).replace(microsecond=0)
        self._make_ca("trusted")
        self._make_ca("untrusted")
        self.ca = self.root / "trusted" / "ca.pem"
        self.untrusted_ca = self.root / "untrusted" / "ca.pem"
        for name, sans, cn in [
            ("exact", ["tenant.carrier.test"], "tenant.carrier.test"),
            ("wildcard", ["*.carrier.test"], "unused.carrier.test"),
            ("wrong_host", ["other.carrier.test"], "other.carrier.test"),
            ("multi_label", ["*.test"], "unused.carrier.test"),
            ("partial_wildcard", ["ten*.carrier.test"], "unused.carrier.test"),
            ("no_san", [], "tenant.carrier.test"),
            ("misleading_cn", ["other.carrier.test"], "tenant.carrier.test"),
        ]:
            self.cases[name] = self._issue(name, "trusted", sans, cn)
        self.cases["expired"] = self._issue("expired", "trusted", ["tenant.carrier.test"], "tenant.carrier.test",
            start=self.now - timedelta(days=3), end=self.now - timedelta(days=1))
        self.cases["future"] = self._issue("future", "trusted", ["tenant.carrier.test"], "tenant.carrier.test",
            start=self.now + timedelta(days=1), end=self.now + timedelta(days=3))
        self.cases["untrusted"] = self._issue("untrusted-leaf", "untrusted", ["tenant.carrier.test"], "tenant.carrier.test")
        self.cases["client_trusted"] = self._issue("client-trusted", "trusted", ["client.carrier.test"], "client.carrier.test", client=True)
        self.cases["client_untrusted"] = self._issue("client-untrusted", "untrusted", ["client.carrier.test"], "client.carrier.test", client=True)
        intermediate = self._issue("intermediate", "trusted", [], "Synthetic intermediate", is_ca=True)
        self._ca_database("intermediate", intermediate.cert, intermediate.key)
        broken = self._issue("broken-chain", "intermediate", ["tenant.carrier.test"], "tenant.carrier.test")
        self.cases["broken_chain"] = broken  # missing intermediate is intentional
        full_chain = self.root / "complete-chain.pem"
        full_chain.write_bytes(broken.cert.read_bytes() + intermediate.cert.read_bytes())
        self.cases["complete_chain"] = Certificate(full_chain, broken.key)

    def _run(self, *args: str) -> None:
        result = subprocess.run([self.openssl, *map(str, args)], cwd=self.root, capture_output=True, timeout=30)
        if result.returncode:
            # Synthetic paths only; never include key data or arbitrary OpenSSL diagnostic output.
            raise RuntimeError(f"openssl {args[0]} failed with exit {result.returncode}")

    def _make_ca(self, name: str) -> None:
        directory = self.root / name
        directory.mkdir(mode=0o700)
        key, cert = directory / "ca.key", directory / "ca.pem"
        self._run("req", "-x509", "-newkey", "rsa:2048", "-nodes", "-sha256", "-days", "30",
                  "-keyout", key, "-out", cert, "-subj", f"/CN=Synthetic {name} TLS proof CA",
                  "-addext", "basicConstraints=critical,CA:TRUE", "-addext", "keyUsage=critical,keyCertSign,cRLSign")
        key.chmod(0o600)
        self._ca_database(name, cert, key)

    def _ca_database(self, name: str, cert: Path, key: Path) -> None:
        directory = self.root / name
        directory.mkdir(exist_ok=True, mode=0o700)
        (directory / "newcerts").mkdir(mode=0o700)
        (directory / "index.txt").write_text("")
        (directory / "serial").write_text("1000\n")
        # OpenSSL config quotes paths so temporary directory spaces are supported.
        def quoted(path):
            return '"' + str(path).replace('\\', '\\\\').replace('"', '\\"') + '"'
        (directory / "ca.cnf").write_text(f"""[ca]
default_ca = proof
[proof]
database = {quoted(directory / 'index.txt')}
new_certs_dir = {quoted(directory / 'newcerts')}
serial = {quoted(directory / 'serial')}
certificate = {quoted(cert)}
private_key = {quoted(key)}
default_md = sha256
default_days = 7
policy = names
unique_subject = no
copy_extensions = none
[names]
commonName = supplied
""")

    def _issue(self, name: str, issuer: str, sans: list[str], cn: str, *, start=None, end=None,
               client=False, is_ca=False) -> Certificate:
        key, csr, cert, ext = (self.root / f"{name}.{suffix}" for suffix in ("key", "csr", "pem", "ext"))
        self._run("req", "-new", "-newkey", "rsa:2048", "-nodes", "-sha256", "-keyout", key,
                  "-out", csr, "-subj", f"/CN={cn}")
        key.chmod(0o600)
        contents = "[leaf]\n"
        if is_ca:
            contents += "basicConstraints=critical,CA:TRUE,pathlen:0\nkeyUsage=critical,keyCertSign,cRLSign\n"
        else:
            contents += ("basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\n"
                         + f"extendedKeyUsage={'clientAuth' if client else 'serverAuth'}\n")
        if sans:
            contents += "subjectAltName=" + ",".join("DNS:" + name for name in sans) + "\n"
        ext.write_text(contents)
        start = start or self.now - timedelta(minutes=5)
        end = end or self.now + timedelta(days=7)
        date = lambda value: value.strftime("%Y%m%d%H%M%SZ")
        self._run("ca", "-batch", "-notext", "-config", self.root / issuer / "ca.cnf", "-in", csr,
                  "-out", cert, "-startdate", date(start), "-enddate", date(end), "-extfile", ext, "-extensions", "leaf")
        return Certificate(cert, key)

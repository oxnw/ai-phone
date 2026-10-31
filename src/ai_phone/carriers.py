"""Carrier setup choices, with no HTTP clients or provider imports."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Credential:
    env: str
    label: str
    argument: str
    placeholder: str = ""
    secret: bool = True


@dataclass(frozen=True)
class Carrier:
    label: str
    credentials: tuple[Credential, ...]
    purchase: bool = True
    managed_caller_id: bool = False


CARRIERS = {
    "twilio": Carrier("Twilio", (
        Credential("TWILIO_ACCOUNT_SID", "Account SID", "key", "AC…"),
        Credential("TWILIO_AUTH_TOKEN", "Auth Token", "secret", "32 characters"),
    )),
    "telnyx": Carrier("Telnyx", (Credential("TELNYX_API_KEY", "API key", "key", "KEY…"),)),
    "signalwire": Carrier("SignalWire", (
        Credential("SIGNALWIRE_PROJECT_ID", "Project ID", "key"),
        Credential("SIGNALWIRE_API_TOKEN", "API token", "secret"),
        Credential("SIGNALWIRE_SPACE", "Space host", "space", "example.signalwire.com", False),
    ), purchase=False, managed_caller_id=True),
    "fake": Carrier("Fake", (Credential("FAKE_API_KEY", "API key", "key"),)),
}
CARRIER_KEYS = {name: tuple(field.env for field in carrier.credentials) for name, carrier in CARRIERS.items()}


def provider_kwargs(carrier: str, credentials: dict[str, str]) -> dict:
    result = {"secret": None}
    result.update({field.argument: credentials[field.env] for field in CARRIERS[carrier].credentials})
    return result


def managed_caller_id(line: dict) -> bool:
    provider = line.get("provider")
    carrier = CARRIERS.get(provider.get("name", "")) if isinstance(provider, dict) else None
    return bool(line.get("from_user", "number") == "account" and carrier and carrier.managed_caller_id)

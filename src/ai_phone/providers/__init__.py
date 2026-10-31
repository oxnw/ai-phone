"""Carrier registry. A third carrier is a new module and one line here; the daemon never imports this package."""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_phone.providers.base import Provider

PROVIDERS = {
    "fake": "ai_phone.providers.fake:FakeProvider",
    "telnyx": "ai_phone.providers.telnyx:TelnyxProvider",
    "twilio": "ai_phone.providers.twilio:TwilioProvider",
    "signalwire": "ai_phone.providers.signalwire:SignalWireProvider",
}


def load_provider(name: str, key: str, secret: str | None = None, **kw) -> "Provider":
    try:
        target = PROVIDERS[name]
    except KeyError:
        raise ValueError(f"unknown provider {name!r}; known: {', '.join(sorted(PROVIDERS))}") from None
    mod, _, cls = target.partition(":")
    return getattr(importlib.import_module(mod), cls)(key=key, secret=secret, **kw)

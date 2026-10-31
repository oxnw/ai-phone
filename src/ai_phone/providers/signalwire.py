"""SignalWire provisioning, used by the CLI only.

Fabric resource IDs address endpoint CRUD and calling-address assignments.
Legacy number routes use nested SIP endpoint IDs.
The HTTP contract is SignalWire's published TypeSpec (see the provider tests).
"""
from __future__ import annotations

import re
import secrets
from urllib.parse import urljoin, urlsplit
from uuid import UUID

import httpx

from ai_phone.lines.journal import Journal
from ai_phone.providers.base import Number, ProviderError, ProvisionResult, ReleaseResult, VerificationStep

ENDPOINTS = "/api/fabric/resources/sip_endpoints"
NUMBERS = "/api/relay/rest/phone_numbers"
PHONE_ADDRESSES = "/api/fabric/phone_number_addresses"
VERIFIED = "/api/relay/rest/verified_caller_ids"
PROFILE = "/api/relay/rest/sip_profile"
_HOST = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)+\Z")
_E164 = re.compile(r"\+[1-9][0-9]{6,14}\Z")
_ROUTE_FIELDS = {
    "relay_sip_endpoint": ("call_sip_endpoint_id",),
    "relay_application": ("call_relay_application",),
    "relay_context": ("call_relay_context", "call_relay_context_status_callback_url"),
    "relay_topic": ("call_relay_topic", "call_relay_topic_status_callback_url"),
    "relay_script": ("call_relay_script_url",),
    "relay_connector": ("call_relay_connector_id",),
    "relay_verto_endpoint": ("call_verto_resource",),
    "laml_application": ("call_laml_application_id",),
    "laml_webhooks": ("call_request_url", "call_request_method", "call_fallback_url", "call_fallback_method", "call_status_callback_url", "call_status_callback_method"),
    "dialogflow": ("call_dialogflow_agent_id",),
    "video_room": ("call_video_room_id",),
}
_VOICE_FIELDS = tuple(dict.fromkeys(k for fields in _ROUTE_FIELDS.values() for k in fields))
_MEDIA = {"codecs": ["PCMU", "PCMA"], "ciphers": ["AES_CM_128_HMAC_SHA1_80"], "encryption": "required"}
# Only documented fixed codes are surfaced. Carrier messages can echo credentials.
_ERROR_CODES = frozenset({"missing_required_parameter", "invalid_parameter", "provided_id_is_unrecognized",
                          "invalid_resource_type", "already_assigned"})


def _request_label(method: str, path: str) -> str:
    for base, name in ((PHONE_ADDRESSES, "phone-number address"), (ENDPOINTS, "SIP endpoint"),
                       (NUMBERS, "phone number"), (VERIFIED, "caller-ID verification"), (PROFILE, "SIP profile")):
        if path == base or path.startswith(base + "/"):
            return f"{method} {name}"
    return "API request"


def normalize_space(space: str) -> str:
    """Accept a Space hostname or its HTTPS origin, never a credential destination URL."""
    if not isinstance(space, str) or space != space.strip():
        raise ProviderError("SignalWire Space must be its full hostname, such as example.signalwire.com.")
    u = urlsplit(space if "://" in space else "https://" + space)
    host = u.hostname or ""
    if (u.scheme != "https" or u.username is not None or u.password is not None
            or u.netloc.lower() != host or u.path not in ("", "/") or u.query or u.fragment
            or not _HOST.fullmatch(host) or not host.endswith(".signalwire.com")):
        raise ProviderError("SignalWire Space must be an HTTPS *.signalwire.com hostname without a path or port.")
    return host


def _id(value) -> str:
    try:
        if not isinstance(value, str) or str(UUID(value)) != value.lower():
            raise ValueError
    except (ValueError, AttributeError, TypeError):
        raise ProviderError("SignalWire returned an invalid resource ID.") from None
    return str(UUID(value))


def _number(value) -> str:
    if not isinstance(value, str) or not _E164.fullmatch(value):
        raise ProviderError("A phone number in E.164 format is required.")
    return value


def _route(record: dict) -> dict:
    """Snapshot only documented, restorable voice fields; never touch messaging."""
    handler = record.get("call_handler")
    fields = _ROUTE_FIELDS.get(handler)
    if not fields or not record.get(fields[0]) or record.get("call_receive_mode") not in ("voice", "fax"):
        raise ProviderError("This number's current voice route cannot be safely restored through the documented API. Configure a supported voice route in SignalWire first, or add the line manually with ai-phone lines add trunk --preset signalwire.")
    return {"call_handler": handler, "call_receive_mode": record["call_receive_mode"],
            **{k: record[k] for k in fields if record.get(k) is not None}}


def _empty_default_route(record: dict) -> bool:
    # Empty LaML webhooks is the normal fresh-number state. Require the complete
    # nullable fields, not just a missing URL in an incomplete API response.
    return ("call_handler" in record and record["call_handler"] in (None, "laml_webhooks")
            and record.get("call_receive_mode") == "voice"
            and "calling_handler_resource_id" in record and record["calling_handler_resource_id"] is None
            and all(k in record and record[k] is None for k in _VOICE_FIELDS))


def _matches(record: dict, route: dict) -> bool:
    return all(record.get(k) == v for k, v in route.items())


class SignalWireProvider:
    name = "signalwire"
    # No authoritative fixed signaling CIDRs are assumed. registered_trunk resolves
    # the actual account SIP domain at daemon startup and still checks source IPs.
    signalling_cidrs: list[str] = []

    def __init__(self, key: str, secret: str | None = None, *, space: str, http: httpx.Client | None = None):
        self.space = normalize_space(space)
        self.project = _id(key)
        if not secret:
            raise ProviderError("SignalWire API token is required.")
        self.origin = "https://" + self.space
        self._auth = httpx.BasicAuth(self.project, secret)
        self.http = http if http is not None else httpx.Client(timeout=20.0, follow_redirects=False)

    def _request(self, method: str, path: str, *, json=None, params=None) -> dict:
        url = urljoin(self.origin, path)
        self._safe_url(url)
        try:
            response = self.http.request(method, url, json=json, params=params, auth=self._auth,
                                         timeout=20.0, follow_redirects=False)
        except (httpx.HTTPError, httpx.InvalidURL):
            raise ProviderError("SignalWire request failed; its outcome may be unknown. Resume to reconcile it.", retryable=True) from None
        if not 200 <= response.status_code < 300:
            # Do not expose arbitrary error text, IDs, URLs, or response fields.
            try:
                error_body = response.json()
            except ValueError:
                error_body = None
            code = error_body.get("code") if isinstance(error_body, dict) else None
            detail = f"; {code}" if isinstance(code, str) and code in _ERROR_CODES else ""
            raise ProviderError(f"SignalWire {_request_label(method, path)} failed (HTTP {response.status_code}{detail}).",
                                retryable=response.status_code in (408, 429) or response.status_code >= 500,
                                status=response.status_code)
        if response.status_code == 204:
            return {}
        try:
            result = response.json()
        except ValueError:
            raise ProviderError("SignalWire returned an invalid JSON response.") from None
        if not isinstance(result, dict):
            raise ProviderError("SignalWire returned an invalid response shape.")
        return result

    def _safe_url(self, url: str, path: str | None = None) -> None:
        u = urlsplit(url)
        if (u.scheme != "https" or u.netloc != self.space or u.username or u.password or u.fragment
                or (path is not None and u.path != path)):
            raise ProviderError("SignalWire returned an unsafe pagination URL.")

    def _list(self, path: str, **params) -> list[dict]:
        url, seen, records = self.origin + path, set(), []
        for _ in range(1000):
            if url in seen:
                raise ProviderError("SignalWire pagination repeated a page.")
            seen.add(url)
            result = self._request("GET", url, params=params or None)
            params = {}
            data, links = result.get("data"), result.get("links", {})
            if not isinstance(data, list) or not all(isinstance(r, dict) for r in data) or not isinstance(links, dict):
                raise ProviderError("SignalWire returned an invalid list response.")
            records.extend(data)
            next_url = links.get("next")
            if not next_url:
                return records
            if not isinstance(next_url, str):
                raise ProviderError("SignalWire returned an invalid pagination link.")
            url = urljoin(self.origin, next_url)
            self._safe_url(url, path)
        raise ProviderError("SignalWire pagination exceeded the safety limit.")

    def validate_context(self, journal: Journal) -> None:
        context = journal.doc.get("signalwire", {})
        if (journal.doc.get("provider") != self.name or not isinstance(context, dict)
                or context.get("space") != self.space or context.get("project_id") != self.project):
            raise ProviderError("This SignalWire journal belongs to a different Space or Project. Use its original credentials.")

    def _bind_context(self, journal: Journal) -> None:
        if "signalwire" not in journal.doc and journal.doc.get("provider") == self.name:
            journal.doc["signalwire"] = {"space": self.space, "project_id": self.project}
            journal._save()
        self.validate_context(journal)

    def search_numbers(self, country: str, area: str | None, limit: int) -> list[Number]:
        raise ProviderError("Buy a number in your SignalWire dashboard, then attach it with --existing. Automatic purchase is not supported.")

    def provision(self, journal: Journal, number: Number, line_id: str, sip_user: str, sip_password: str) -> ProvisionResult:
        raise ProviderError("Buy a number in your SignalWire dashboard, then attach it with --existing. Automatic purchase is not supported.")

    def owned_numbers(self) -> list[Number]:
        return [Number(_number(r.get("number")), None, None, False, _id(r.get("id")), r.get("country_code") or "")
                for r in self._list(NUMBERS) if "voice" in r.get("capabilities", [])]

    def _owned_number(self, rid: str, e164: str) -> dict:
        record = self._request("GET", NUMBERS + "/" + _id(rid))
        if record.get("id") != rid or record.get("number") != e164 or "voice" not in record.get("capabilities", []):
            raise ProviderError("SignalWire number ownership or voice capability no longer matches this line.")
        return record

    def _address_shape(self, record: dict, number_id: str, e164: str, address_id: str | None = None) -> dict:
        if (record.get("type") != "phone" or record.get("handler_type") != "calling"
                or record.get("phone_number_id") != number_id or record.get("phone_number") != e164
                or "resource_id" not in record or (address_id is not None and record.get("id") != address_id)):
            raise ProviderError("SignalWire calling-address ownership does not match this line.")
        _id(record.get("id"))
        if record["resource_id"] is not None:
            _id(record["resource_id"])
        return record

    def _number_address(self, number_id: str, e164: str) -> dict:
        # This collection has no filters. Include partial identity matches so a
        # malformed or duplicate address cannot be silently ignored.
        matches = [r for r in self._list(PHONE_ADDRESSES)
                   if r.get("handler_type") == "calling"
                   and (r.get("phone_number_id") == number_id or r.get("phone_number") == e164)]
        if len(matches) != 1:
            raise ProviderError("SignalWire did not return exactly one calling address for this number.")
        address = self._address_shape(matches[0], number_id, e164)
        address = self._address_shape(self._request("GET", PHONE_ADDRESSES + "/" + address["id"]),
                                      number_id, e164, address["id"])
        return address

    def _unassigned_address(self, number_id: str, e164: str) -> dict:
        address = self._number_address(number_id, e164)
        if address["resource_id"] is not None:
            raise ProviderError("This number's calling address already has a resource; it has not been overwritten.")
        return address

    def _assigned_sip_route(self, number: dict, number_id: str, e164: str) -> dict:
        """Verify both identifiers of the existing Fabric route before replacing it."""
        if number.get("call_receive_mode") != "voice":
            raise ProviderError("This SignalWire number is not set to receive voice calls. Select voice instead of fax in SignalWire, then try again. No carrier settings were changed.")
        resource_id = _id(number.get("calling_handler_resource_id"))
        endpoint_id = _id(number.get("call_sip_endpoint_id"))
        old = self._request("GET", ENDPOINTS + "/" + resource_id)
        endpoint = old.get("sip_endpoint")
        if (old.get("id") != resource_id or old.get("project_id") != self.project
                or not isinstance(endpoint, dict) or endpoint.get("id") != endpoint_id):
            raise ProviderError("SignalWire's existing SIP route does not match its Fabric resource; it has not been changed.")
        address = self._number_address(number_id, e164)
        if address["resource_id"] != resource_id:
            raise ProviderError("SignalWire's calling address does not match the number's existing route; it has not been changed.")
        # Keep the previous assignment and voice metadata before any routing write.
        previous = {k: number[k] for k in (*_VOICE_FIELDS, "call_handler", "call_receive_mode", "calling_handler_resource_id")
                    if k in number}
        return {"route_mode": "calling_address", "calling_address_id": address["id"],
                "previous_calling_resource_id": resource_id, "previous_voice_route": previous}

    def _calling_address(self, journal: Journal) -> dict:
        self.validate_context(journal)
        r = journal.resources()
        if r.get("route_mode") != "calling_address" or "previous_calling_resource_id" not in r:
            raise ProviderError("SignalWire calling-address journal is incomplete.")
        if r["previous_calling_resource_id"] is not None:
            _id(r["previous_calling_resource_id"])
        rid = _id(r.get("calling_address_id"))
        record = self._request("GET", PHONE_ADDRESSES + "/" + rid)
        return self._address_shape(record, r["brought_number_id"], r["phone_number"], rid)

    def _assign_calling_address(self, journal: Journal) -> None:
        r = journal.resources()
        number = self._owned_number(r["brought_number_id"], r["phone_number"])
        address = self._calling_address(journal)
        if number.get("call_receive_mode") != "voice":
            raise ProviderError("SignalWire number receive mode changed; the calling address was not modified.")
        if address["resource_id"] == r["sip_resource_id"]:
            # The server applied a previous assignment whose response was lost.
            if not _matches(number, self._assigned_number_route(r)):
                raise ProviderError("SignalWire has not confirmed the number's calling-address route.", retryable=True)
            journal.start("number_routed")
            journal.finish("number_routed")
            return
        previous_resource = r["previous_calling_resource_id"]
        if address["resource_id"] != previous_resource or not _matches(number, r["previous_voice_route"]):
            raise ProviderError("The number's voice route changed outside ai-phone; it has not been overwritten.")
        status, _ = journal.start("number_routed")
        if status == "done":
            raise ProviderError("The number's completed voice route changed outside ai-phone; it has not been overwritten.")
        if previous_resource is not None:
            # An assigned channel uses PATCH, not POST or a legacy nested SIP ID.
            # Repeating this update is safe only after confirming the saved old route.
            response = self._request("PATCH", PHONE_ADDRESSES + "/" + r["calling_address_id"],
                                     json={"resource_id": r["sip_resource_id"]})
        else:
            if status != "new":
                raise ProviderError("SignalWire calling-address assignment has an unknown or changed outcome. No second assignment was sent; check the dashboard before resuming.", retryable=True)
            # POST links an unassigned channel; it never purchases a number.
            response = self._request("POST", PHONE_ADDRESSES, json={"phone_number_id": r["brought_number_id"],
                                     "handler_type": "calling", "resource_id": r["sip_resource_id"]})
        self._address_shape(response, r["brought_number_id"], r["phone_number"], r["calling_address_id"])
        if (self._calling_address(journal)["resource_id"] != r["sip_resource_id"]
                or not _matches(self._owned_number(r["brought_number_id"], r["phone_number"]), self._assigned_number_route(r))):
            raise ProviderError("SignalWire has not confirmed the calling-address assignment.", retryable=True)
        journal.finish("number_routed")

    @staticmethod
    def _assigned_number_route(resources: dict) -> dict:
        return {"call_handler": "relay_sip_endpoint", "call_receive_mode": "voice",
                "calling_handler_resource_id": resources["sip_resource_id"],
                "call_sip_endpoint_id": resources["sip_endpoint_id"]}

    def _clear_calling_address(self, journal: Journal) -> None:
        r = journal.resources()
        number = self._owned_number(r["brought_number_id"], r["phone_number"])
        address = self._calling_address(journal)
        previous_resource = r["previous_calling_resource_id"]
        if address["resource_id"] not in (previous_resource, r["sip_resource_id"]):
            raise ProviderError("The calling address changed outside ai-phone; the endpoint was retained.")
        if address["resource_id"] == r["sip_resource_id"]:
            if not _matches(number, self._assigned_number_route(r)):
                raise ProviderError("The number's voice route changed outside ai-phone; the endpoint was retained.")
            if previous_resource is None:
                self._request("DELETE", PHONE_ADDRESSES + "/" + r["calling_address_id"])
            else:
                self._request("PATCH", PHONE_ADDRESSES + "/" + r["calling_address_id"],
                              json={"resource_id": previous_resource})
            address = self._calling_address(journal)
            number = self._owned_number(r["brought_number_id"], r["phone_number"])
        if address["resource_id"] != previous_resource:
            raise ProviderError("SignalWire did not confirm the previous calling-address assignment; the endpoint was retained.")
        if previous_resource is not None and not _matches(number, r["previous_voice_route"]):
            raise ProviderError("SignalWire did not restore the previous voice route; the endpoint was retained.")

    def _endpoint_shape(self, record: dict, username: str, resource_id: str | None = None, endpoint_id: str | None = None) -> dict:
        endpoint = record.get("sip_endpoint")
        if (not isinstance(endpoint, dict) or record.get("project_id") != self.project
                or endpoint.get("username") != username
                or (resource_id is not None and record.get("id") != resource_id)
                or (endpoint_id is not None and endpoint.get("id") != endpoint_id)):
            raise ProviderError("SignalWire SIP endpoint ownership does not match this line.")
        _id(record.get("id")); _id(endpoint.get("id"))
        return endpoint

    def _endpoint(self, journal: Journal) -> dict:
        r = journal.resources()
        step = journal._step("sip_endpoint")
        if not step or step["status"] != "done":
            raise ProviderError("SignalWire SIP endpoint creation is not yet reconciled.")
        record = self._request("GET", ENDPOINTS + "/" + _id(r.get("sip_resource_id")))
        return self._endpoint_shape(record, step["planned"]["username"], r["sip_resource_id"], r.get("sip_endpoint_id"))

    def _reconcile_endpoint(self, journal: Journal) -> None:
        step = journal._step("sip_endpoint")
        if not step or step["status"] == "done":
            return
        planned = step["planned"]
        matches = [r for r in self._list(ENDPOINTS) if isinstance(r.get("sip_endpoint"), dict)
                   and r["sip_endpoint"].get("username") == planned["username"]]
        if len(matches) != 1:
            raise ProviderError("SignalWire endpoint creation has an unknown outcome. No second endpoint was created; retry resume after checking the dashboard.", retryable=True)
        record = matches[0]
        endpoint = self._endpoint_shape(record, planned["username"])
        if (endpoint.get("caller_id") != planned["owner_marker"] or endpoint.get("send_as") != planned["number"]
                or endpoint.get("call_handler") != "block-pstn" or any(endpoint.get(k) != v for k, v in _MEDIA.items())):
            raise ProviderError("An existing SignalWire endpoint has this username but its creation marker does not match. It has not been adopted.")
        journal.finish("sip_endpoint", sip_resource_id=record["id"], sip_endpoint_id=endpoint["id"])

    def attach(self, journal: Journal, number: Number, line_id: str, sip_user: str, sip_password: str) -> ProvisionResult:
        self._bind_context(journal)
        if line_id != journal.doc.get("line_id") or not re.fullmatch(r"[A-Za-z0-9]{4,32}", sip_user):
            raise ProviderError("SignalWire line identity or SIP username is invalid.")
        e164 = _number(number.e164)
        prior = journal._step("number_claim")
        source_id = number.source_id
        if source_id == e164 or not source_id:  # older generic resume targets contain only E.164
            if prior:
                source_id = prior["planned"].get("number_id")
            else:
                matches = [n for n in self.owned_numbers() if n.e164 == e164]
                if len(matches) != 1:
                    raise ProviderError("SignalWire did not return one owned voice number matching this line.")
                source_id = matches[0].source_id
        number_id = _id(source_id)
        if prior and (prior["planned"].get("number") != e164 or prior["planned"].get("number_id") != number_id):
            raise ProviderError("The selected number differs from the existing SignalWire journal.")
        number_record = self._owned_number(number_id, e164)
        if not prior or prior["status"] != "done":
            if journal._step("sip_endpoint"):
                raise ProviderError("SignalWire number ownership snapshot is incomplete; review the journal before resuming.")
            route_state = {}
            if _empty_default_route(number_record):
                address = self._unassigned_address(number_id, e164)
                previous = {k: number_record[k] for k in (*_VOICE_FIELDS, "call_handler", "call_receive_mode", "calling_handler_resource_id")}
                route_state = {"route_mode": "calling_address", "calling_address_id": address["id"],
                               "previous_calling_resource_id": None}
            else:
                previous = _route(number_record)  # refuse an unrestorable route before mutations
                if (number_record.get("call_handler") == "relay_sip_endpoint"
                        and number_record.get("calling_handler_resource_id") is not None):
                    route_state = self._assigned_sip_route(number_record, number_id, e164)
                    previous = route_state.pop("previous_voice_route")
            journal.start("number_claim", number=e164, number_id=number_id)
            journal.finish("number_claim", brought_number_id=number_id, previous_voice_route=previous, phone_number=e164, **route_state)
        else:
            # Older pending journals attempted legacy PUT for an existing Fabric
            # SIP route. Migrate only while the exact saved route is still present;
            # an already-applied route or a completed setup keeps its old contract.
            saved = journal.resources()
            routed = journal._step("number_routed")
            previous = saved.get("previous_voice_route", {})
            if (not saved.get("route_mode") and (not routed or routed["status"] != "done")
                    and previous.get("call_handler") == "relay_sip_endpoint"
                    and previous.get("call_receive_mode") in ("voice", "fax")
                    and isinstance(previous.get("call_sip_endpoint_id"), str)
                    and _matches(number_record, previous)
                    and number_record.get("calling_handler_resource_id") is not None):
                route_state = self._assigned_sip_route(number_record, number_id, e164)
                claim = journal._step("number_claim")
                journal.finish("number_claim", **{**claim["result"], **route_state})
        profile = self._request("GET", PROFILE)
        domain = profile.get("domain")
        if not isinstance(domain, str) or not _HOST.fullmatch(domain) or not domain.endswith(".signalwire.com"):
            raise ProviderError("SignalWire returned an invalid SIP profile domain.")
        context = journal.doc["signalwire"]
        if context.get("sip_domain", domain) != domain:
            raise ProviderError("The SignalWire SIP profile domain changed; review the line before resuming.")
        context["sip_domain"] = domain
        journal._save()
        step = journal._step("sip_endpoint")
        if not step:
            if any(isinstance(r.get("sip_endpoint"), dict) and r["sip_endpoint"].get("username") == sip_user for r in self._list(ENDPOINTS)):
                raise ProviderError("A SignalWire SIP endpoint already uses this username; it has not been adopted.")
            marker = "ai" + secrets.token_hex(6)  # <=15 characters, safe caller-ID name until activation
            journal.start("sip_endpoint", username=sip_user, owner_marker=marker, number=e164)
            record = self._request("POST", ENDPOINTS, json={"username": sip_user, "password": sip_password,
                "caller_id": marker, "send_as": e164, "call_handler": "block-pstn", **_MEDIA})
            endpoint = self._endpoint_shape(record, sip_user)
            journal.finish("sip_endpoint", sip_resource_id=record["id"], sip_endpoint_id=endpoint["id"])
        else:
            if step["planned"].get("username") != sip_user:
                raise ProviderError("The SIP username differs from this SignalWire journal.")
            self._reconcile_endpoint(journal)
        r = journal.resources()
        endpoint = self._endpoint(journal)
        status, _ = journal.start("endpoint_configured")
        desired = {"caller_id": "ai-phone", "call_handler": "passthrough", **_MEDIA}
        if status == "done":
            # Resume must check the same media/outbound invariants as poll before
            # changing a number's route. A completed step is not live evidence.
            if endpoint.get("call_handler") != "passthrough" or any(endpoint.get(k) != v for k, v in _MEDIA.items()):
                raise ProviderError("The SignalWire endpoint settings changed; review the line before resuming.")
            _number(endpoint.get("send_as"))  # a verified caller-ID change may be intentional
        else:
            if not _matches(endpoint, desired):
                self._request("PUT", ENDPOINTS + "/" + r["sip_resource_id"], json=desired)
                endpoint = self._endpoint(journal)
            if not _matches(endpoint, desired) or endpoint.get("send_as") != e164:
                raise ProviderError("SignalWire did not confirm the SIP endpoint settings.")
            journal.finish("endpoint_configured")
        if r.get("route_mode") == "calling_address":
            self._assign_calling_address(journal)
            return self.poll(journal)
        ours = {"call_handler": "relay_sip_endpoint", "call_sip_endpoint_id": r["sip_endpoint_id"], "call_receive_mode": "voice"}
        current = self._owned_number(number_id, e164)
        if not _matches(current, ours) and not _matches(current, r["previous_voice_route"]):
            raise ProviderError("The number's voice route changed outside ai-phone; it has not been overwritten.")
        status, _ = journal.start("number_routed")
        if not _matches(current, ours):
            if status == "done":
                raise ProviderError("The number's completed voice route changed outside ai-phone; it has not been overwritten.")
            self._request("PUT", NUMBERS + "/" + number_id, json=ours)
        if not _matches(self._owned_number(number_id, e164), ours):
            raise ProviderError("SignalWire has not confirmed the number's SIP route.", retryable=True)
        journal.finish("number_routed")
        return self.poll(journal)

    def poll(self, journal: Journal) -> ProvisionResult:
        self.validate_context(journal)
        if "number_routed" not in journal.steps_done():
            return ProvisionResult("pending", "Resume provisioning to reconcile SignalWire resources.")
        r = journal.resources()
        endpoint = self._endpoint(journal)
        number = self._owned_number(r["brought_number_id"], r["phone_number"])
        if r.get("route_mode") == "calling_address":
            routed = (self._calling_address(journal)["resource_id"] == r["sip_resource_id"]
                      and _matches(number, self._assigned_number_route(r)))
        else:
            ours = {"call_handler": "relay_sip_endpoint", "call_sip_endpoint_id": r["sip_endpoint_id"], "call_receive_mode": "voice"}
            routed = _matches(number, ours)
        if not routed:
            raise ProviderError("The SignalWire number is no longer routed to this line.")
        if endpoint.get("call_handler") != "passthrough" or any(endpoint.get(k) != v for k, v in _MEDIA.items()):
            raise ProviderError("The SignalWire endpoint settings changed; review the line before resuming.")
        domain = journal.doc["signalwire"]["sip_domain"]
        journal.set_state("ready")
        return ProvisionResult("ready", "SignalWire number attached.", {
            "kind": "trunk", "number": r["phone_number"],
            "caller_id": _number(endpoint.get("send_as")), "user": endpoint["username"], "from_user": "account",
            "registrar": f"sip:{domain}:5061;transport=tls", "outbound_proxy": f"sip:{domain}:5061;transport=tls",
            "dial_domain": f"{domain}:5061", "register_via_proxy": True, "transport": "tls", "srtp": "required",
            "tls_hostname_policy": "verified_wildcard",
            "accept_invites_from": "registered_trunk", "trusted_cidrs": [],
            "provider": {"name": self.name, "state": "ready", "resources": {k: r[k] for k in ("sip_resource_id", "sip_endpoint_id", "brought_number_id", "calling_address_id") if k in r}},
        })

    def release(self, journal: Journal) -> ReleaseResult:
        self.validate_context(journal)
        try:
            self._reconcile_endpoint(journal)
            r = journal.resources()
            if "sip_resource_id" not in r:
                return ReleaseResult(True)
            # Validate endpoint ownership before restoring or deleting anything.
            try:
                self._endpoint(journal)
            except ProviderError as error:
                if error.status != 404:
                    raise
            ours = {"call_handler": "relay_sip_endpoint", "call_sip_endpoint_id": r["sip_endpoint_id"], "call_receive_mode": "voice"}
            if journal._step("number_routed") and r.get("route_mode") == "calling_address":
                self._clear_calling_address(journal)
            elif journal._step("number_routed"):
                current = self._owned_number(r["brought_number_id"], r["phone_number"])
                previous = r["previous_voice_route"]
                if _matches(current, ours):
                    self._request("PUT", NUMBERS + "/" + r["brought_number_id"], json=previous)
                    current = self._owned_number(r["brought_number_id"], r["phone_number"])
                if not _matches(current, previous):
                    raise ProviderError("The number's voice route changed or could not be restored; the endpoint was retained.")
            try:
                self._request("DELETE", ENDPOINTS + "/" + r["sip_resource_id"])
            except ProviderError as error:
                if error.status != 404:
                    raise
            return ReleaseResult(True)
        except ProviderError:
            journal.set_state("rollback_failed")
            return ReleaseResult(False, {k: v for k, v in journal.resources().items() if k in ("sip_resource_id", "sip_endpoint_id", "brought_number_id", "calling_address_id")} or {"endpoint_creation": "unresolved"})

    def _allowed_caller_id(self, e164: str) -> bool:
        if any(n.e164 == e164 for n in self.owned_numbers()):
            return True
        return any(r.get("number") == e164 and r.get("verified") is True for r in self._list(VERIFIED, filter_number=e164))

    def apply_caller_id(self, journal: Journal, number: str) -> str:
        self.validate_context(journal)
        number = _number(number)
        endpoint = self._endpoint(journal)
        if not self._allowed_caller_id(number):
            raise ProviderError("SignalWire caller ID must be an owned or verified number on this Project.")
        change = journal.doc.setdefault("caller_id_change", {})
        if change.get("desired") and not change.get("local_committed"):
            if change["desired"] != number:
                raise ProviderError("Resume the pending SignalWire caller-ID change first.")
        else:
            change.update(desired=number, previous=endpoint.get("send_as"), remote_applied=False, local_committed=False)
            journal._save()
        if endpoint.get("send_as") not in (change["previous"], number):
            raise ProviderError("The SignalWire caller ID changed outside ai-phone; it has not been overwritten.")
        if endpoint.get("send_as") != number:
            self._request("PUT", ENDPOINTS + "/" + journal.resources()["sip_resource_id"], json={"send_as": number})
        if self._endpoint(journal).get("send_as") != number:
            raise ProviderError("SignalWire did not confirm the selected caller ID.", retryable=True)
        change["remote_applied"] = True
        journal._save()
        return number

    def pending_caller_id(self, journal: Journal) -> str | None:
        self.validate_context(journal)
        change = journal.doc.get("caller_id_change", {})
        return change.get("desired") if not change.get("local_committed") else None

    def finish_caller_id(self, journal: Journal) -> None:
        self.validate_context(journal)
        change = journal.doc.get("caller_id_change", {})
        if not change.get("remote_applied"):
            raise ProviderError("SignalWire caller ID has not been confirmed remotely.")
        change["local_committed"] = True
        journal._save()

    def start_caller_id_verification(self, e164: str) -> VerificationStep:
        e164 = _number(e164)
        owned = [n for n in self.owned_numbers() if n.e164 == e164]
        if len(owned) > 1:
            raise ProviderError("SignalWire returned ambiguous owned numbers.")
        if owned:
            return VerificationStep("enter_code_on_phone", "This Project already owns the number; no verification call is needed.",
                                    {"owned_number_id": owned[0].source_id, "number": e164, "space": self.space, "project_id": self.project})
        existing = [r for r in self._list(VERIFIED, filter_number=e164) if r.get("number") == e164]
        if len(existing) > 1:
            raise ProviderError("SignalWire returned ambiguous caller-ID verification records.")
        if existing:
            record = existing[0]
            if record.get("verified") is not True:
                self._request("POST", VERIFIED + "/" + _id(record.get("id")) + "/verification")
        else:
            record = self._request("POST", VERIFIED, json={"number": e164})
        if record.get("number") != e164:
            raise ProviderError("SignalWire returned a different caller-ID verification number.")
        verified = record.get("verified") is True
        return VerificationStep("enter_code_on_phone" if verified else "enter_code_in_cli",
                                "Already verified; no verification call is needed." if verified else "SignalWire calls this number and reads a six-digit code. Enter it here.",
                                {"id": _id(record.get("id")), "number": e164, "space": self.space, "project_id": self.project})

    def complete_caller_id_verification(self, step: VerificationStep, code: str | None) -> bool:
        handle = step.handle
        if handle.get("space") != self.space or handle.get("project_id") != self.project:
            raise ProviderError("Caller-ID verification belongs to a different SignalWire Space or Project.")
        number = _number(handle.get("number"))
        if handle.get("owned_number_id"):
            self._owned_number(_id(handle["owned_number_id"]), number)
            return True
        rid = _id(handle.get("id"))
        record = self._request("GET", VERIFIED + "/" + rid)
        if record.get("id") != rid or record.get("number") != number:
            raise ProviderError("SignalWire caller-ID verification ownership changed.")
        if record.get("verified") is True:
            return True
        if not isinstance(code, str) or not re.fullmatch(r"[0-9]{6}", code):
            return False
        try:
            self._request("PUT", VERIFIED + "/" + rid + "/verification", json={"verification_code": code})
        except ProviderError as error:
            if error.status in (400, 422):
                return False
            raise
        record = self._request("GET", VERIFIED + "/" + rid)
        return record.get("id") == rid and record.get("number") == number and record.get("verified") is True

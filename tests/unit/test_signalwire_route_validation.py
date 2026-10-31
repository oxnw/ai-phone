"""Assigned Fabric routes use the documented address PATCH, with reversible recovery.

Contract: https://signalwire.com/docs/apis/rest/phone-number-addresses/update-phone-number-address
These are offline API-boundary regressions, not live carrier compatibility claims.
"""
from copy import deepcopy
import json

import httpx
import pytest

from ai_phone.lines.journal import Journal
from ai_phone.providers.base import ProviderError
from ai_phone.providers.signalwire import ENDPOINTS, NUMBERS, PHONE_ADDRESSES
from tests.unit.test_signalwire import (
    CALLING_ADDRESS, ENDPOINT, NUMBER, OTHER_PROJECT, PASSWORD, PHONE, PREVIOUS_ENDPOINT,
    PROJECT, RESOURCE, SPACE, TOKEN, USER, Carrier, attach, provider, response_fixture,
)

PREVIOUS_RESOURCE = "77777777-7777-4777-8777-777777777777"
FOREIGN_RESOURCE = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"


class AssignedCarrier(Carrier):
    """Both endpoints are Fabric resources; calling and messaging are separate."""

    def __init__(self):
        super().__init__()
        self.number = response_fixture("phone_number")
        self.previous_number = deepcopy(self.number)
        self.old_endpoint = response_fixture("sip_endpoint")
        self.old_endpoint.update(id=PREVIOUS_RESOURCE)
        self.old_endpoint["sip_endpoint"].update(id=PREVIOUS_ENDPOINT, username="prioruser")
        self.addresses = response_fixture("phone_number_addresses")
        self.calling["resource_id"] = PREVIOUS_RESOURCE
        self.ignore_repoint = False
        self.before_repoint = None

    @property
    def calling(self):
        return next(a for a in self.addresses if a["handler_type"] == "calling")

    def __call__(self, request):
        method, path = request.method, request.url.path
        body = json.loads(request.content) if request.content else {}
        if path == ENDPOINTS + "/" + PREVIOUS_RESOURCE and method == "GET":
            self.calls.append((method, path, body))
            return httpx.Response(200, json=self.old_endpoint)
        if path == NUMBERS + "/" + NUMBER and method == "PUT":
            self.calls.append((method, path, body))
            # Real-world failure under investigation. The address contract, not
            # a guessed permissive legacy PUT response, defines our new behavior.
            return httpx.Response(422, json={"code": "invalid_parameter", "message": TOKEN})
        if not path.startswith(PHONE_ADDRESSES):
            return super().__call__(request)
        self.calls.append((method, path, body))
        assert request.url.host == SPACE
        assert request.headers["authorization"] == httpx.BasicAuth(PROJECT, TOKEN)._auth_header
        if path == PHONE_ADDRESSES and method == "GET":
            return self.listing(PHONE_ADDRESSES, self.addresses)
        if path == PHONE_ADDRESSES + "/" + CALLING_ADDRESS:
            if method == "GET":
                return httpx.Response(200, json=self.calling)
            if method == "PATCH":
                assert set(body) == {"resource_id"}
                target = body["resource_id"]
                assert target in (RESOURCE, PREVIOUS_RESOURCE)
                if self.before_repoint:
                    self.before_repoint()
                if self.failure == "before_repoint":
                    raise httpx.ReadTimeout("secret " + TOKEN, request=request)
                if self.failure == "restore" and target == PREVIOUS_RESOURCE:
                    return httpx.Response(503, json={"error": TOKEN})
                if not self.ignore_repoint:
                    self.calling["resource_id"] = target
                    if target == RESOURCE:
                        self.number.update(call_handler="relay_sip_endpoint", call_receive_mode="voice",
                                           calling_handler_resource_id=RESOURCE, call_sip_endpoint_id=ENDPOINT)
                    else:
                        self.number.update({k: v for k, v in self.previous_number.items()
                                            if k.startswith("call_") or k == "calling_handler_resource_id"})
                if self.failure == "repoint" or (self.failure == "restore_lost" and target == PREVIOUS_RESOURCE):
                    raise httpx.ReadTimeout("secret " + TOKEN, request=request)
                return httpx.Response(200, json=self.calling)
        raise AssertionError(f"Unexpected address operation: {method} {path}")


def ready(tmp_path):
    carrier = AssignedCarrier()
    p = provider(carrier)
    journal = Journal.open(tmp_path, "main", "signalwire")
    result = attach(p, journal)
    return carrier, p, journal, result


def writes(carrier):
    return [(m, p, b) for m, p, b in carrier.calls if m != "GET"]


def test_existing_fabric_route_uses_patch_and_preserves_messaging_and_old_endpoint(tmp_path):
    carrier = AssignedCarrier()
    original_addresses = deepcopy(carrier.addresses)
    original_number = deepcopy(carrier.number)
    p = provider(carrier)
    journal = Journal.open(tmp_path, "main", "signalwire")

    def check_durable_rollback_before_patch():
        saved = json.loads(journal.path.read_text())
        claim = next(s for s in saved["steps"] if s["name"] == "number_claim")["result"]
        assert claim["previous_calling_resource_id"] == PREVIOUS_RESOURCE
        assert claim["calling_address_id"] == CALLING_ADDRESS
        assert claim["previous_voice_route"]["call_sip_endpoint_id"] == PREVIOUS_ENDPOINT
        assert next(s for s in saved["steps"] if s["name"] == "number_routed")["status"] == "started"

    carrier.before_repoint = check_durable_rollback_before_patch
    result = attach(p, journal)
    assert result.state == "ready"
    assert journal.resources()["route_mode"] == "calling_address"
    assert result.line["provider"]["resources"]["calling_address_id"] == CALLING_ADDRESS
    assert ("PATCH", PHONE_ADDRESSES + "/" + CALLING_ADDRESS, {"resource_id": RESOURCE}) in writes(carrier)
    assert not any(m == "POST" and path == PHONE_ADDRESSES or m == "PUT" and path.startswith(NUMBERS)
                   for m, path, _ in writes(carrier))
    assert [a for a in carrier.addresses if a["handler_type"] == "messaging"] == [
        a for a in original_addresses if a["handler_type"] == "messaging"]
    assert {k: v for k, v in carrier.number.items() if k.startswith("message")} == {
        k: v for k, v in original_number.items() if k.startswith("message")}
    assert carrier.old_endpoint["id"] == PREVIOUS_RESOURCE


@pytest.mark.parametrize("failure", ["before_repoint", "repoint"])
def test_lost_patch_response_reconciles_without_duplicate_endpoint(tmp_path, failure):
    carrier = AssignedCarrier()
    carrier.failure = failure
    p = provider(carrier)
    journal = Journal.open(tmp_path, "main", "signalwire")
    with pytest.raises(ProviderError):
        attach(p, journal)
    assert journal._step("number_routed")["status"] == "started"
    carrier.failure = None
    assert attach(p, journal).state == "ready"
    assert sum(m == "POST" and path == ENDPOINTS for m, path, _ in carrier.calls) == 1
    assert sum(m == "PATCH" for m, _, _ in carrier.calls) == (2 if failure == "before_repoint" else 1)


def test_failed_legacy_journal_migrates_only_after_exact_old_resource_checks(tmp_path):
    carrier = AssignedCarrier()
    p = provider(carrier)
    journal = Journal.open(tmp_path, "main", "signalwire")
    p._bind_context(journal)
    journal.start("number_claim", number=PHONE, number_id=NUMBER)
    journal.finish("number_claim", brought_number_id=NUMBER, phone_number=PHONE,
                   previous_voice_route={"call_handler": "relay_sip_endpoint", "call_receive_mode": "voice",
                                         "call_sip_endpoint_id": PREVIOUS_ENDPOINT})
    # Emulate the older setup's completed endpoint creation/configuration.
    carrier.endpoint = response_fixture("sip_endpoint")
    carrier.endpoint["sip_endpoint"].update(username=USER, caller_id="ai-phone", send_as=PHONE,
                                           call_handler="passthrough", codecs=["PCMU", "PCMA"],
                                           ciphers=["AES_CM_128_HMAC_SHA1_80"], encryption="required")
    journal.start("sip_endpoint", username=USER, owner_marker="aioldmarker", number=PHONE)
    journal.finish("sip_endpoint", sip_resource_id=RESOURCE, sip_endpoint_id=ENDPOINT)
    journal.start("endpoint_configured")
    journal.finish("endpoint_configured")
    journal.start("number_routed")
    assert attach(p, journal).state == "ready"
    assert journal.resources()["previous_calling_resource_id"] == PREVIOUS_RESOURCE
    assert journal.resources()["previous_voice_route"]["calling_handler_resource_id"] == PREVIOUS_RESOURCE
    assert writes(carrier) == [("PATCH", PHONE_ADDRESSES + "/" + CALLING_ADDRESS, {"resource_id": RESOURCE})]


@pytest.mark.parametrize("mismatch", ["project", "resource", "nested", "address", "duplicate", "number"])
def test_existing_route_identity_mismatch_refuses_before_creating_endpoint(tmp_path, mismatch):
    carrier = AssignedCarrier()
    if mismatch == "project":
        carrier.old_endpoint["project_id"] = OTHER_PROJECT
    elif mismatch == "resource":
        carrier.old_endpoint["id"] = FOREIGN_RESOURCE
    elif mismatch == "nested":
        carrier.old_endpoint["sip_endpoint"]["id"] = ENDPOINT
    elif mismatch == "address":
        carrier.calling["resource_id"] = FOREIGN_RESOURCE
    elif mismatch == "duplicate":
        carrier.addresses.append(deepcopy(carrier.calling))
    else:
        carrier.calling["phone_number"] = "+15555559999"
    journal = Journal.open(tmp_path, "main", "signalwire")
    with pytest.raises(ProviderError):
        attach(provider(carrier), journal)
    assert writes(carrier) == []
    assert journal._step("sip_endpoint") is None


def test_unconfirmed_patch_stays_pending_and_does_not_claim_success(tmp_path):
    carrier = AssignedCarrier()
    carrier.ignore_repoint = True
    p = provider(carrier)
    journal = Journal.open(tmp_path, "main", "signalwire")
    with pytest.raises(ProviderError, match="not confirmed"):
        attach(p, journal)
    assert journal._step("number_routed")["status"] == "started"
    assert journal.state != "ready"


def test_release_restores_original_resource_before_deleting_only_new_endpoint(tmp_path):
    carrier, p, journal, _ = ready(tmp_path)
    carrier.calls.clear()
    assert p.release(journal).ok
    assert writes(carrier) == [
        ("PATCH", PHONE_ADDRESSES + "/" + CALLING_ADDRESS, {"resource_id": PREVIOUS_RESOURCE}),
        ("DELETE", ENDPOINTS + "/" + RESOURCE, {}),
    ]
    assert carrier.number == carrier.previous_number
    assert carrier.calling["resource_id"] == PREVIOUS_RESOURCE
    assert carrier.old_endpoint["id"] == PREVIOUS_RESOURCE


@pytest.mark.parametrize("failure", ["restore", "restore_lost"])
def test_failed_restore_retains_endpoint_and_retry_reconciles(tmp_path, failure):
    carrier, p, journal, _ = ready(tmp_path)
    carrier.calls.clear()
    carrier.failure = failure
    assert not p.release(journal).ok
    assert not any(m == "DELETE" for m, _, _ in carrier.calls)
    assert carrier.endpoint is not None
    carrier.failure = None
    assert p.release(journal).ok
    assert carrier.calling["resource_id"] == PREVIOUS_RESOURCE
    assert carrier.endpoint is None


@pytest.mark.parametrize("drift", ["address", "number", "receive"])
def test_remove_refuses_external_route_drift_without_writes(tmp_path, drift):
    carrier, p, journal, _ = ready(tmp_path)
    if drift == "address":
        carrier.calling["resource_id"] = FOREIGN_RESOURCE
    elif drift == "number":
        carrier.number["calling_handler_resource_id"] = FOREIGN_RESOURCE
    else:
        carrier.number["call_receive_mode"] = "fax"
    carrier.calls.clear()
    assert not p.release(journal).ok
    assert writes(carrier) == []
    assert carrier.endpoint is not None


def test_resume_does_not_overwrite_a_changed_calling_address(tmp_path):
    carrier = AssignedCarrier()
    carrier.failure = "before_repoint"
    p = provider(carrier)
    journal = Journal.open(tmp_path, "main", "signalwire")
    with pytest.raises(ProviderError):
        attach(p, journal)
    carrier.failure = None
    carrier.calling["resource_id"] = FOREIGN_RESOURCE
    carrier.calls.clear()
    with pytest.raises(ProviderError, match="changed outside"):
        attach(p, journal)
    assert writes(carrier) == []


@pytest.mark.parametrize("assigned", [False, True], ids=["legacy", "fabric"])
def test_completed_route_resume_preserves_external_restoration(tmp_path, assigned):
    carrier = AssignedCarrier() if assigned else Carrier()
    previous_number = deepcopy(carrier.number)
    p = provider(carrier)
    journal = Journal.open(tmp_path, "main", "signalwire")
    assert attach(p, journal).state == "ready"
    assert journal._step("number_routed")["status"] == "done"
    endpoint = deepcopy(carrier.endpoint)
    claim = deepcopy(journal._step("number_claim"))

    # The owner intentionally moved the number back to its original route in
    # the portal. A completed setup must not treat this as a request to retry.
    carrier.number = deepcopy(previous_number)
    if assigned:
        carrier.calling["resource_id"] = PREVIOUS_RESOURCE
    carrier.calls.clear()

    with pytest.raises(ProviderError, match="completed voice route changed outside"):
        attach(p, journal)
    assert writes(carrier) == []
    assert carrier.number == previous_number
    assert carrier.endpoint == endpoint
    assert journal._step("number_claim") == claim
    assert journal._step("number_routed")["status"] == "done"
    if assigned:
        assert carrier.calling["resource_id"] == PREVIOUS_RESOURCE


@pytest.mark.parametrize("assigned", [False, True], ids=["legacy", "fabric"])
def test_completed_route_resume_with_unchanged_route_needs_no_writes(tmp_path, assigned):
    carrier = AssignedCarrier() if assigned else Carrier()
    p = provider(carrier)
    journal = Journal.open(tmp_path, "main", "signalwire")
    assert attach(p, journal).state == "ready"
    carrier.calls.clear()

    assert attach(p, journal).state == "ready"
    assert writes(carrier) == []


@pytest.mark.parametrize("code", ["missing_required_parameter", "invalid_parameter", "provided_id_is_unrecognized",
                                   "invalid_resource_type", "already_assigned"])
def test_http_error_names_operation_and_only_documented_fixed_code(code):
    def handler(request):
        return httpx.Response(422, json={"code": code, "message": TOKEN + PASSWORD,
                                        "detail": request.headers["authorization"]})
    with pytest.raises(ProviderError) as caught:
        provider(handler)._request("PATCH", PHONE_ADDRESSES + "/" + CALLING_ADDRESS, json={"resource_id": RESOURCE})
    text = str(caught.value)
    assert "PATCH phone-number address" in text and code in text and "HTTP 422" in text
    assert TOKEN not in text and PASSWORD not in text and CALLING_ADDRESS not in text and RESOURCE not in text
    assert not caught.value.retryable


@pytest.mark.parametrize("body", [{"code": TOKEN, "message": PASSWORD}, {"code": [TOKEN]}, [TOKEN], None])
def test_unknown_error_payload_never_leaks(body):
    def handler(request):
        return httpx.Response(422, json=body)
    with pytest.raises(ProviderError) as caught:
        provider(handler)._request("PUT", NUMBERS + "/" + NUMBER, json={"call_sip_endpoint_id": ENDPOINT})
    text = str(caught.value)
    assert "PUT phone number" in text and "HTTP 422" in text
    assert TOKEN not in text and PASSWORD not in text and NUMBER not in text


@pytest.mark.parametrize("pending", [False, True])
def test_assigned_fabric_fax_route_refused_before_new_carrier_mutations(tmp_path, pending):
    carrier = AssignedCarrier()
    carrier.number["call_receive_mode"] = "fax"
    p = provider(carrier)
    journal = Journal.open(tmp_path, "main", "signalwire")
    if pending:
        # An older failed attempt already owns an endpoint. Preflight must refuse
        # before reconfiguring it or migrating its saved restoration contract.
        p._bind_context(journal)
        journal.start("number_claim", number=PHONE, number_id=NUMBER)
        journal.finish("number_claim", brought_number_id=NUMBER, phone_number=PHONE,
                       previous_voice_route={"call_handler": "relay_sip_endpoint", "call_receive_mode": "fax",
                                             "call_sip_endpoint_id": PREVIOUS_ENDPOINT})
        carrier.endpoint = response_fixture("sip_endpoint")
        carrier.endpoint["sip_endpoint"].update(username=USER, caller_id="aioldmarker", send_as=PHONE,
                                               call_handler="block-pstn")
        journal.start("sip_endpoint", username=USER, owner_marker="aioldmarker", number=PHONE)
        journal.finish("sip_endpoint", sip_resource_id=RESOURCE, sip_endpoint_id=ENDPOINT)
        journal.start("endpoint_configured")
        journal.start("number_routed")
    prior_claim = deepcopy(journal._step("number_claim"))
    prior_endpoint = deepcopy(carrier.endpoint)
    with pytest.raises(ProviderError, match="Select voice instead of fax in SignalWire"):
        attach(p, journal)
    assert writes(carrier) == []
    assert carrier.endpoint == prior_endpoint
    assert journal._step("number_claim") == prior_claim
    assert "route_mode" not in journal.resources()
    assert carrier.number["call_receive_mode"] == "fax"


def test_legacy_fax_route_still_uses_existing_voice_conversion_and_restoration(tmp_path):
    carrier = Carrier()
    carrier.number["call_receive_mode"] = "fax"
    p = provider(carrier)
    journal = Journal.open(tmp_path, "main", "signalwire")
    assert attach(p, journal).state == "ready"
    assert carrier.number["call_receive_mode"] == "voice"
    assert journal.resources()["previous_voice_route"]["call_receive_mode"] == "fax"
    assert "route_mode" not in journal.resources()
    assert p.release(journal).ok
    assert carrier.number["call_receive_mode"] == "fax"

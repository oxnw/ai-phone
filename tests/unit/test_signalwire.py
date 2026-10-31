"""Offline API-contract/lifecycle tests; see tests/fixtures/signalwire/README.md.

Request validation and response fixtures are independent of the provider code.
These mocks do not establish live carrier interoperability.
"""
from copy import deepcopy
import json
from pathlib import Path

import httpx
import pytest

from ai_phone.lines.journal import Journal
from ai_phone.providers import load_provider
from ai_phone.providers.base import Number, ProviderError
from ai_phone.providers.signalwire import ENDPOINTS, NUMBERS, PHONE_ADDRESSES, PROFILE, VERIFIED, SignalWireProvider, normalize_space

PROJECT = "11111111-1111-4111-8111-111111111111"
OTHER_PROJECT = "11111111-1111-4111-8111-222222222222"
RESOURCE = "22222222-2222-4222-8222-222222222222"
ENDPOINT = "33333333-3333-4333-8333-333333333333"
PREVIOUS_ENDPOINT = "44444444-4444-4444-8444-444444444444"
NUMBER = "55555555-5555-4555-8555-555555555555"
VERIFICATION = "66666666-6666-4666-8666-666666666666"
SPACE = "test.signalwire.com"
DOMAIN = "actual-profile.sip.signalwire.com"
PHONE = "+15555550100"
PERSONAL = "+15555550200"
PASSWORD = "UnitTestOnlyPassword123"
TOKEN = "UnitTestOnlyToken"
USER = "aimain1234"


def response_fixture(name):
    return json.loads((Path(__file__).parents[1] / "fixtures" / "signalwire" / f"{name}.json").read_text())


# Published writable fields, deliberately independent of the adapter's constants.
ENDPOINT_WRITABLE = ("username", "password", "caller_id", "send_as", "ciphers", "codecs",
                     "encryption", "call_handler", "calling_handler_resource_id")
ENDPOINT_CODECS = {"PCMU", "PCMA", "G722", "G729", "OPUS", "OPUS@48000H@20I", "OPUS@24000H@20I",
                   "OPUS@16000H@20I", "OPUS@8000H@20I", "VP8", "H264"}
ENDPOINT_CIPHERS = {"AEAD_AES_256_GCM_8", "AES_256_CM_HMAC_SHA1_80", "AES_CM_128_HMAC_SHA1_80",
                    "AES_256_CM_HMAC_SHA1_32", "AES_CM_128_HMAC_SHA1_32"}


def valid_endpoint_request(body, *, create=False):
    if not isinstance(body, dict) or set(body) - set(ENDPOINT_WRITABLE):
        return False
    if create and not {"username", "password"} <= set(body):
        return False
    for key in ("username", "password", "caller_id", "send_as"):
        if key in body and not isinstance(body[key], str):
            return False
    for key, choices in (("codecs", ENDPOINT_CODECS), ("ciphers", ENDPOINT_CIPHERS)):
        if key in body and (not isinstance(body[key], list) or any(not isinstance(v, str) or v not in choices for v in body[key])):
            return False
    if "encryption" in body and body["encryption"] not in {"default", "optional", "required"}:
        return False
    if "call_handler" in body and body["call_handler"] not in {"default", "passthrough", "block-pstn", "resource"}:
        return False
    if "calling_handler_resource_id" in body and body["calling_handler_resource_id"] is not None:
        return isinstance(body["calling_handler_resource_id"], str)
    return True


class Carrier:
    def __init__(self):
        self.number = response_fixture("phone_number")
        # This fake exercises legacy number routing without a Fabric assignment.
        # Assigned Fabric SIP routes have their own calling-address contract tests.
        self.number["calling_handler_resource_id"] = None
        self.profile = response_fixture("sip_profile")
        self.endpoint = None
        self.verified = {}
        self.calls = []
        self.failure = None
        self.ignore_send_as = False
        self.ignore_verification = False

    def endpoint_update(self, body):
        # The server has independent response metadata and never reflects password.
        for field in ENDPOINT_WRITABLE:
            if field != "password" and field in body:
                if field != "send_as" or not self.ignore_send_as:
                    self.endpoint["sip_endpoint"][field] = deepcopy(body[field])

    @staticmethod
    def listing(path, records):
        return httpx.Response(200, json={"data": records, "links": {"self": "https://" + SPACE + path}})

    def __call__(self, request):
        method, path = request.method, request.url.path
        body = json.loads(request.content) if request.content else {}
        self.calls.append((method, path, body))
        assert request.url.host == SPACE
        assert request.headers["authorization"] == httpx.BasicAuth(PROJECT, TOKEN)._auth_header
        assert request.extensions["timeout"]["read"] == 20.0
        if path == PROFILE and method == "GET":
            return httpx.Response(200, json=self.profile)
        if path == NUMBERS and method == "GET":
            return self.listing(NUMBERS, [self.number])
        if path == NUMBERS + "/" + NUMBER:
            if method == "GET":
                return httpx.Response(200, json=self.number)
            if method == "PUT":
                # Narrow voice-route update contract: response-only fields must not
                # be echoed into PUT, and unrelated messaging must remain untouched.
                allowed = set(VOICE_FIELDS) | {"call_handler", "call_receive_mode"}
                if set(body) - allowed or any(not isinstance(value, str) for value in body.values()):
                    return httpx.Response(422, json={"error": "Invalid number update"})
                if self.failure == "restore" and body.get("call_sip_endpoint_id") == PREVIOUS_ENDPOINT:
                    return httpx.Response(503, json={"error": TOKEN})
                for field in allowed:
                    if field in body:
                        self.number[field] = body[field]
                if self.failure == "route" and body.get("call_sip_endpoint_id") == ENDPOINT:
                    raise httpx.ReadTimeout("secret " + TOKEN, request=request)
                return httpx.Response(200, json=self.number)
        if path == ENDPOINTS:
            if method == "GET":
                return self.listing(ENDPOINTS, [self.endpoint] if self.endpoint else [])
            if method == "POST":
                if not valid_endpoint_request(body, create=True):
                    return httpx.Response(422, json={"error": "Invalid endpoint create"})
                if self.failure == "before_create":
                    raise httpx.ConnectError("secret " + TOKEN, request=request)
                self.endpoint = response_fixture("sip_endpoint")
                self.endpoint_update(body)
                if self.failure == "create":
                    raise httpx.ReadTimeout("secret " + TOKEN, request=request)
                return httpx.Response(201, json=self.endpoint)
        if path == ENDPOINTS + "/" + RESOURCE:
            if self.endpoint is None:
                return httpx.Response(404, json={"error": "missing"})
            if method == "GET":
                return httpx.Response(200, json=self.endpoint)
            if method == "PUT":
                if not valid_endpoint_request(body):
                    return httpx.Response(422, json={"error": "Invalid endpoint update"})
                self.endpoint_update(body)
                if self.failure == "configure":
                    raise httpx.ReadTimeout("secret " + TOKEN, request=request)
                return httpx.Response(200, json=self.endpoint)
            if method == "DELETE":
                self.endpoint = None
                if self.failure == "delete":
                    raise httpx.ReadTimeout("secret " + TOKEN, request=request)
                return httpx.Response(204)
        if path == VERIFIED:
            if method == "GET":
                return self.listing(VERIFIED, list(self.verified.values()))
            if method == "POST":
                if set(body) - {"number", "name", "extension"} or not isinstance(body.get("number"), str):
                    return httpx.Response(422, json={"error": "Invalid verification create"})
                record = response_fixture("verified_caller_id")
                record["number"] = body["number"]
                record["name"] = body.get("name", body["number"])
                record["extension"] = body.get("extension")
                self.verified[VERIFICATION] = record
                return httpx.Response(201, json=record)
        if path == VERIFIED + "/" + VERIFICATION and method == "GET":
            return httpx.Response(200, json=self.verified[VERIFICATION])
        if path == VERIFIED + "/" + VERIFICATION + "/verification":
            if method == "PUT":
                if set(body) != {"verification_code"} or body["verification_code"] != "123456":
                    return httpx.Response(422, json={"error": TOKEN})
                if not self.ignore_verification:
                    self.verified[VERIFICATION].update(verified=True, verified_at="2026-01-01T00:01:00Z", status="Verified")
            return httpx.Response(200, json=self.verified[VERIFICATION])
        raise AssertionError(f"Unexpected {method} {path}")


def provider(carrier, **kw):
    return SignalWireProvider(key=kw.pop("key", PROJECT), secret=TOKEN, space=kw.pop("space", SPACE),
                              http=httpx.Client(transport=httpx.MockTransport(carrier), timeout=None), **kw)


def attach(p, journal):
    return p.attach(journal, Number(PHONE, None, None, False, NUMBER, "US"), "main", USER, PASSWORD)


def ready(tmp_path):
    carrier = Carrier()
    p = provider(carrier)
    journal = Journal.open(tmp_path, "main", "signalwire")
    result = attach(p, journal)
    return carrier, p, journal, result


def writes(carrier):
    return [(m, p, b) for m, p, b in carrier.calls if m != "GET"]


def test_factory_supports_space_without_changing_other_providers():
    p = load_provider("signalwire", PROJECT, TOKEN, space=SPACE)
    assert isinstance(p, SignalWireProvider)


@pytest.mark.parametrize("space", [SPACE, "https://" + SPACE, "https://" + SPACE + "/"])
def test_space_normalization(space):
    assert normalize_space(space) == SPACE


@pytest.mark.parametrize("space", ["http://" + SPACE, "test.signalwire.com.evil.test", "https://evil.test@" + SPACE,
    SPACE + ":443", SPACE + "/api", SPACE + "?token=x", SPACE + "#x", SPACE + " ", "localhost", "https://test.signalwire.com\\@evil.test"])
def test_untrusted_space_rejected_before_http(space):
    with pytest.raises(ProviderError):
        normalize_space(space)


def test_attach_uses_outer_id_for_crud_and_inner_id_for_number_route(tmp_path):
    carrier, p, journal, result = ready(tmp_path)
    assert RESOURCE != ENDPOINT
    assert result.state == "ready"
    assert "id" not in result.line  # line ID is the config mapping key, not a field
    assert result.line["provider"]["resources"] == {"sip_resource_id": RESOURCE, "sip_endpoint_id": ENDPOINT, "brought_number_id": NUMBER}
    assert result.line["from_user"] == "account"
    assert result.line["registrar"] == f"sip:{DOMAIN}:5061;transport=tls"
    assert result.line["dial_domain"] == DOMAIN + ":5061"
    assert result.line["tls_hostname_policy"] == "verified_wildcard"
    assert result.line["transport"] == "tls" and result.line["srtp"] == "required"
    assert result.line["trusted_cidrs"] == [] and result.line["accept_invites_from"] == "registered_trunk"
    assert carrier.number["call_sip_endpoint_id"] == ENDPOINT
    assert carrier.number["message_request_url"] == "https://example.org/messages"
    assert carrier.endpoint["sip_endpoint"]["send_as"] == PHONE
    assert carrier.endpoint["sip_endpoint"]["call_handler"] == "passthrough"
    text = journal.path.read_text()
    assert PASSWORD not in text and TOKEN not in text
    assert journal.doc["signalwire"] == {"space": SPACE, "project_id": PROJECT, "sip_domain": DOMAIN}
    assert p.poll(journal).state == "ready"


def test_automatic_purchase_never_makes_a_request(tmp_path):
    carrier, p = Carrier(), None
    p = provider(carrier)
    journal = Journal.open(tmp_path, "main", "signalwire")
    with pytest.raises(ProviderError, match="--existing"):
        p.search_numbers("GB", "20", 5)
    with pytest.raises(ProviderError, match="--existing"):
        p.provision(journal, Number(PHONE, None, None, False, ""), "main", USER, PASSWORD)
    assert carrier.calls == []


@pytest.mark.parametrize("handler", [None, "ai_agent", "fabric_subscriber", "default"])
def test_unrestorable_voice_route_fails_before_mutations(tmp_path, handler):
    carrier = Carrier()
    carrier.number["call_handler"] = handler
    journal = Journal.open(tmp_path, "main", "signalwire")
    with pytest.raises(ProviderError, match="safely restored"):
        attach(provider(carrier), journal)
    assert not writes(carrier)


@pytest.mark.parametrize("failure", ["create", "configure", "route"])
def test_resume_reconciles_unknown_outcomes_without_second_post(tmp_path, failure):
    carrier = Carrier()
    carrier.failure = failure
    p = provider(carrier)
    journal = Journal.open(tmp_path, "main", "signalwire")
    with pytest.raises(ProviderError):
        attach(p, journal)
    carrier.failure = None
    assert attach(p, Journal.open(tmp_path, "main")).state == "ready"
    assert sum(m == "POST" and path == ENDPOINTS for m, path, _ in carrier.calls) == 1


def test_unknown_creation_not_found_does_not_blindly_repeat_post(tmp_path):
    carrier = Carrier()
    carrier.failure = "before_create"
    p = provider(carrier)
    journal = Journal.open(tmp_path, "main", "signalwire")
    with pytest.raises(ProviderError):
        attach(p, journal)
    carrier.failure = None
    with pytest.raises(ProviderError, match="unknown outcome"):
        attach(p, journal)
    assert sum(m == "POST" for m, _, _ in carrier.calls) == 1
    assert not p.release(journal).ok


def test_does_not_adopt_preexisting_same_username(tmp_path):
    carrier, p, _, _ = ready(tmp_path / "first")
    second = Journal.open(tmp_path / "second", "main", "signalwire")
    before = len(writes(carrier))
    with pytest.raises(ProviderError, match="already uses"):
        attach(p, second)
    assert len(writes(carrier)) == before


def test_creation_reconciliation_requires_marker(tmp_path):
    carrier = Carrier()
    carrier.failure = "create"
    p = provider(carrier)
    journal = Journal.open(tmp_path, "main", "signalwire")
    with pytest.raises(ProviderError):
        attach(p, journal)
    carrier.endpoint["sip_endpoint"]["caller_id"] = "someone else"
    carrier.failure = None
    before = len(writes(carrier))
    with pytest.raises(ProviderError, match="marker"):
        attach(p, journal)
    assert not p.release(journal).ok
    assert len(writes(carrier)) == before


def test_context_mismatch_prevents_all_remote_work(tmp_path):
    carrier, p, journal, _ = ready(tmp_path)
    wrong = provider(carrier, key=OTHER_PROJECT)
    before = len(carrier.calls)
    for action in (lambda: attach(wrong, journal), lambda: wrong.poll(journal), lambda: wrong.release(journal),
                   lambda: wrong.apply_caller_id(journal, PHONE)):
        with pytest.raises(ProviderError, match="different Space or Project"):
            action()
    assert len(carrier.calls) == before


def test_release_restores_previous_endpoint_and_keeps_number_and_messages(tmp_path):
    carrier, p, journal, _ = ready(tmp_path)
    assert p.release(journal).ok
    assert carrier.number["call_sip_endpoint_id"] == PREVIOUS_ENDPOINT
    assert carrier.number["message_handler"] == "laml_webhooks"
    assert carrier.endpoint is None
    assert not any(m == "DELETE" and path.startswith(NUMBERS) for m, path, _ in carrier.calls)
    assert p.release(journal).ok


def test_failed_restoration_retains_endpoint_and_cleanup_is_retryable(tmp_path):
    carrier, p, journal, _ = ready(tmp_path)
    carrier.failure = "restore"
    result = p.release(journal)
    assert not result.ok and result.leftover["sip_resource_id"] == RESOURCE
    assert carrier.endpoint is not None and journal.state == "rollback_failed"
    carrier.failure = None
    assert p.release(journal).ok


def test_portal_route_edit_is_not_overwritten_during_cleanup(tmp_path):
    carrier, p, journal, _ = ready(tmp_path)
    carrier.number.update(call_handler="relay_script", call_relay_script_url="https://example.org/new")
    before = len(writes(carrier))
    assert not p.release(journal).ok
    assert len(writes(carrier)) == before and carrier.endpoint is not None


def test_endpoint_ownership_is_checked_before_restoration(tmp_path):
    carrier, p, journal, _ = ready(tmp_path)
    carrier.endpoint["project_id"] = OTHER_PROJECT
    before = len(writes(carrier))
    assert not p.release(journal).ok
    assert len(writes(carrier)) == before


def test_delete_response_loss_is_reconciled(tmp_path):
    carrier, p, journal, _ = ready(tmp_path)
    carrier.failure = "delete"
    assert not p.release(journal).ok
    carrier.failure = None
    assert p.release(journal).ok


def test_caller_id_change_survives_local_write_failure_and_preserves_metadata(tmp_path):
    carrier, p, journal, _ = ready(tmp_path)
    carrier.verified[VERIFICATION] = {"id": VERIFICATION, "number": PERSONAL, "verified": True}
    journal.doc["caller_id_change"] = {"verified_record": {"number": PERSONAL}}
    assert p.apply_caller_id(journal, PERSONAL) == PERSONAL
    assert p.pending_caller_id(journal) == PERSONAL
    restored = Journal.open(tmp_path, "main")
    before = len(writes(carrier))
    assert p.apply_caller_id(restored, PERSONAL) == PERSONAL
    assert len(writes(carrier)) == before
    p.finish_caller_id(restored)
    assert p.pending_caller_id(restored) is None
    assert restored.doc["caller_id_change"]["verified_record"] == {"number": PERSONAL}
    assert p.apply_caller_id(restored, PHONE) == PHONE
    p.finish_caller_id(restored)
    assert p.poll(restored).line["caller_id"] == PHONE


def test_caller_id_refuses_unverified_or_revoked_number(tmp_path):
    carrier, p, journal, _ = ready(tmp_path)
    before = len(writes(carrier))
    with pytest.raises(ProviderError, match="owned or verified"):
        p.apply_caller_id(journal, PERSONAL)
    assert len(writes(carrier)) == before
    carrier.verified[VERIFICATION] = {"id": VERIFICATION, "number": PERSONAL, "verified": True}
    p.apply_caller_id(journal, PERSONAL)
    carrier.verified[VERIFICATION]["verified"] = False
    with pytest.raises(ProviderError, match="owned or verified"):
        p.apply_caller_id(journal, PERSONAL)


def test_caller_id_does_not_claim_success_without_remote_readback(tmp_path):
    carrier, p, journal, _ = ready(tmp_path)
    carrier.verified[VERIFICATION] = {"id": VERIFICATION, "number": PERSONAL, "verified": True}
    carrier.ignore_send_as = True
    with pytest.raises(ProviderError, match="did not confirm"):
        p.apply_caller_id(journal, PERSONAL)
    assert not journal.doc["caller_id_change"]["remote_applied"]
    assert p.pending_caller_id(journal) == PERSONAL
    with pytest.raises(ProviderError):
        p.finish_caller_id(journal)


def test_verification_requires_matching_remote_verified_record():
    carrier, p = Carrier(), None
    p = provider(carrier)
    step = p.start_caller_id_verification(PERSONAL)
    assert step.kind == "enter_code_in_cli"
    assert not p.complete_caller_id_verification(step, "000000")
    carrier.ignore_verification = True
    assert not p.complete_caller_id_verification(step, "123456")
    carrier.ignore_verification = False
    assert p.complete_caller_id_verification(step, "123456")
    assert not any(TOKEN in str(v) or PASSWORD in str(v) for v in step.handle.values())


def test_verification_reuses_existing_record_instead_of_duplicating():
    carrier = Carrier()
    p = provider(carrier)
    p.start_caller_id_verification(PERSONAL)
    p.start_caller_id_verification(PERSONAL)
    assert sum(m == "POST" and path == VERIFIED for m, path, _ in carrier.calls) == 1
    assert sum(m == "POST" and path.endswith("/verification") for m, path, _ in carrier.calls) == 1


def test_owned_number_pagination_and_exact_number_fields():
    calls = []
    def handler(request):
        calls.append(str(request.url))
        if len(calls) == 1:
            return httpx.Response(200, json={"data": [], "links": {"next": f"https://{SPACE}{NUMBERS}?page_number=1&page_token=opaque"}})
        return httpx.Response(200, json={"data": [Carrier().number], "links": {}})
    result = provider(handler).owned_numbers()
    assert result[0].e164 == PHONE and result[0].source_id == NUMBER
    assert result[0].monthly_cost is None and result[0].currency is None
    assert "page_token=opaque" in calls[1]


@pytest.mark.parametrize("link", ["https://evil.test/api/relay/rest/phone_numbers", f"https://{SPACE}/api/other", f"http://{SPACE}{NUMBERS}", f"https://user@{SPACE}{NUMBERS}"])
def test_pagination_cannot_leak_basic_auth(link):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"data": [], "links": {"next": link}})
    with pytest.raises(ProviderError, match="unsafe"):
        provider(handler).owned_numbers()
    assert len(calls) == 1


def test_pagination_cycle_is_bounded():
    def handler(request):
        return httpx.Response(200, json={"data": [], "links": {"next": f"https://{SPACE}{NUMBERS}"}})
    with pytest.raises(ProviderError, match="repeated"):
        provider(handler).owned_numbers()


@pytest.mark.parametrize("status", [301, 401, 403, 429, 500])
def test_http_errors_never_expose_response_text_or_follow_redirects(status):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(status, json={"error": TOKEN + PASSWORD}, headers={"location": "https://evil.test"})
    with pytest.raises(ProviderError) as caught:
        provider(handler).owned_numbers()
    assert TOKEN not in str(caught.value) and PASSWORD not in str(caught.value)
    assert caught.value.status == status and len(calls) == 1


def test_resume_after_crash_between_number_claim_start_and_finish(tmp_path):
    carrier = Carrier()
    p = provider(carrier)
    journal = Journal.open(tmp_path, "main", "signalwire")
    p._bind_context(journal)
    journal.start("number_claim", number=PHONE, number_id=NUMBER)
    synthetic = Number(PHONE, None, None, False, PHONE)
    assert p.attach(journal, synthetic, "main", USER, PASSWORD).state == "ready"


def test_synthetic_resume_number_is_resolved_before_claim(tmp_path):
    carrier = Carrier()
    p = provider(carrier)
    journal = Journal.open(tmp_path, "main", "signalwire")
    synthetic = Number(PHONE, None, None, False, PHONE)
    assert p.attach(journal, synthetic, "main", USER, PASSWORD).state == "ready"


def test_already_verified_retry_requires_neither_call_nor_new_code():
    carrier = Carrier()
    carrier.verified[VERIFICATION] = {"id": VERIFICATION, "number": PERSONAL, "verified": True}
    p = provider(carrier)
    step = p.start_caller_id_verification(PERSONAL)
    assert step.kind == "enter_code_on_phone" and "Already verified" in step.instructions
    assert p.complete_caller_id_verification(step, None)
    assert writes(carrier) == []


def test_project_owned_number_never_starts_external_verification():
    carrier = Carrier()
    p = provider(carrier)
    step = p.start_caller_id_verification(PHONE)
    assert step.kind == "enter_code_on_phone"
    assert p.complete_caller_id_verification(step, None)
    assert writes(carrier) == []
    assert not any(path.startswith(VERIFIED) for _, path, _ in carrier.calls)


def test_real_adapter_cli_resume_after_lost_create_response(tmp_path, monkeypatch):
    from ai_phone.cli import main
    from ai_phone.config import load_config
    import ai_phone.cli_lines as cli_lines

    monkeypatch.setenv("AI_PHONE_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("inbound:\n  greeting: Hi\n")
    carrier = Carrier()
    carrier.failure = "create"
    p = provider(carrier)
    monkeypatch.setattr(cli_lines, "_provider", lambda a: p)
    monkeypatch.setattr(cli_lines, "generate_sip_user", lambda line_id: USER)
    monkeypatch.setattr(cli_lines, "generate_sip_password", lambda: PASSWORD)
    args = ["lines", "add", "signalwire", "--key-env", "SIGNALWIRE_PROJECT_ID", "--secret-env", "SIGNALWIRE_API_TOKEN",
            "--space", SPACE, "--existing", PHONE, "--id", "main", "--yes"]
    assert main(args) == 3
    journal = Journal.open(tmp_path / "lines", "main")
    assert journal._step("target")["planned"]["source_id"] == NUMBER
    carrier.failure = None
    assert main(["lines", "resume", "main", "--key-env", "SIGNALWIRE_PROJECT_ID", "--secret-env", "SIGNALWIRE_API_TOKEN", "--space", SPACE]) == 0
    line = load_config(tmp_path / "config.yaml").lines["main"]
    assert line.provider.resources["sip_resource_id"] == RESOURCE
    assert line.provider.resources["sip_endpoint_id"] == ENDPOINT
    assert line.from_user == "account" and line.caller_id == PHONE
    assert sum(m == "POST" and path == ENDPOINTS for m, path, _ in carrier.calls) == 1


def test_project_id_is_canonical_before_context_and_http(tmp_path):
    project = "abcdefab-cdef-4abc-8def-abcdefabcdef"
    p = SignalWireProvider(key=project.upper(), secret=TOKEN, space=SPACE)
    assert p.project == project
    journal = Journal.open(tmp_path, "main", "signalwire")
    p._bind_context(journal)
    assert journal.doc["signalwire"]["project_id"] == project
    record = {"id": RESOURCE, "project_id": project, "sip_endpoint": {"id": ENDPOINT, "username": USER}}
    assert p._endpoint_shape(record, USER)["id"] == ENDPOINT


CALLING_ADDRESS = "88888888-8888-4888-8888-888888888888"
MESSAGING_ADDRESS = "99999999-9999-4999-8999-999999999999"
# Nullable destination/method fields from the published PhoneNumber response.
VOICE_FIELDS = ("call_request_url", "call_request_method", "call_fallback_url", "call_fallback_method",
                "call_status_callback_url", "call_status_callback_method", "call_laml_application_id",
                "call_dialogflow_agent_id", "call_relay_topic", "call_relay_topic_status_callback_url",
                "call_relay_script_url", "call_relay_context", "call_relay_context_status_callback_url",
                "call_relay_application", "call_relay_connector_id", "call_sip_endpoint_id",
                "call_verto_resource", "call_video_room_id")


class FreshCarrier(Carrier):
    """The real fresh-number shape: empty LaML defaults, existing unassigned channels."""
    def __init__(self):
        super().__init__()
        self.number.update({field: None for field in VOICE_FIELDS})
        self.number.update(call_handler="laml_webhooks", calling_handler_resource_id=None)
        self.addresses = response_fixture("phone_number_addresses")
        self.ignore_assignment = False
        self.ignore_clear = False
        self.before_assignment = None

    @property
    def calling(self):
        return self.addresses[0]

    def __call__(self, request):
        if not request.url.path.startswith(PHONE_ADDRESSES):
            return super().__call__(request)
        method, path = request.method, request.url.path
        body = json.loads(request.content) if request.content else {}
        self.calls.append((method, path, body))
        assert request.url.host == SPACE
        assert request.headers["authorization"] == httpx.BasicAuth(PROJECT, TOKEN)._auth_header
        if path == PHONE_ADDRESSES and method == "GET":
            assert not request.url.params  # this endpoint has no number filters
            return self.listing(PHONE_ADDRESSES, self.addresses)
        if path == PHONE_ADDRESSES and method == "POST":
            if body != {"phone_number_id": NUMBER, "handler_type": "calling", "resource_id": RESOURCE}:
                return httpx.Response(422, json={"error": "Invalid address assignment"})
            if self.before_assignment:
                self.before_assignment()
            if self.failure == "before_assign":
                raise httpx.ReadTimeout("secret " + TOKEN, request=request)
            if self.calling["resource_id"] is not None:
                return httpx.Response(422, json={"code": "already_assigned"})
            if not self.ignore_assignment:
                self.calling["resource_id"] = RESOURCE
                self.number.update(call_handler="relay_sip_endpoint", call_sip_endpoint_id=ENDPOINT,
                                   calling_handler_resource_id=RESOURCE)
            if self.failure == "assign":
                raise httpx.ReadTimeout("secret " + TOKEN, request=request)
            return httpx.Response(201, json=self.calling)
        if path == PHONE_ADDRESSES + "/" + CALLING_ADDRESS:
            if method == "GET":
                return httpx.Response(200, json=self.calling)
            if method == "DELETE":
                if self.failure == "clear":
                    return httpx.Response(503, json={"error": TOKEN})
                if not self.ignore_clear:
                    self.calling["resource_id"] = None
                    self.number.update({field: None for field in VOICE_FIELDS})
                    self.number.update(call_handler="laml_webhooks", calling_handler_resource_id=None)
                if self.failure == "clear_lost":
                    raise httpx.ReadTimeout("secret " + TOKEN, request=request)
                return httpx.Response(204)
        raise AssertionError(f"Unexpected address operation {method} {path}")


def fresh_ready(tmp_path):
    carrier = FreshCarrier()
    p = provider(carrier)
    journal = Journal.open(tmp_path, "main", "signalwire")
    result = attach(p, journal)
    return carrier, p, journal, result


@pytest.mark.parametrize("handler", ["laml_webhooks", None])
def test_fresh_default_uses_outer_resource_and_journals_unassigned_address_before_post(tmp_path, handler):
    carrier = FreshCarrier()
    carrier.number["call_handler"] = handler
    p = provider(carrier)
    journal = Journal.open(tmp_path, "main", "signalwire")
    def before_assignment():
        saved = Journal.open(tmp_path, "main")
        assert saved.resources()["calling_address_id"] == CALLING_ADDRESS
        assert saved.resources()["route_mode"] == "calling_address"
        assert saved.resources()["previous_calling_resource_id"] is None
        assert saved._step("number_routed")["status"] == "started"
    carrier.before_assignment = before_assignment
    result = attach(p, journal)
    assert result.state == "ready"
    assert result.line["provider"]["resources"]["calling_address_id"] == CALLING_ADDRESS
    assert carrier.calling["resource_id"] == RESOURCE != ENDPOINT
    assert carrier.addresses[1]["resource_id"] is None
    assert not any(method == "PUT" and path.startswith(NUMBERS) for method, path, _ in carrier.calls)
    assert p.poll(journal).state == "ready"


def test_fresh_assignment_lost_response_resumes_without_second_post(tmp_path):
    carrier = FreshCarrier()
    carrier.failure = "assign"
    p = provider(carrier)
    journal = Journal.open(tmp_path, "main", "signalwire")
    with pytest.raises(ProviderError):
        attach(p, journal)
    carrier.failure = None
    assert attach(p, Journal.open(tmp_path, "main")).state == "ready"
    assert sum(method == "POST" and path == PHONE_ADDRESSES for method, path, _ in carrier.calls) == 1


def test_fresh_unknown_assignment_not_applied_does_not_repeat_post(tmp_path):
    carrier = FreshCarrier()
    carrier.failure = "before_assign"
    p = provider(carrier)
    journal = Journal.open(tmp_path, "main", "signalwire")
    with pytest.raises(ProviderError):
        attach(p, journal)
    carrier.failure = None
    with pytest.raises(ProviderError, match="No second assignment"):
        attach(p, journal)
    assert sum(method == "POST" and path == PHONE_ADDRESSES for method, path, _ in carrier.calls) == 1
    assert p.release(journal).ok
    assert carrier.endpoint is None and carrier.calling["resource_id"] is None


def test_fresh_assignment_requires_remote_readback(tmp_path):
    carrier = FreshCarrier()
    carrier.ignore_assignment = True
    p = provider(carrier)
    journal = Journal.open(tmp_path, "main", "signalwire")
    with pytest.raises(ProviderError, match="not confirmed"):
        attach(p, journal)
    assert journal._step("number_routed")["status"] == "started"
    assert p.poll(journal).state == "pending"


@pytest.mark.parametrize("problem", ["missing", "duplicate", "wrong_number", "wrong_number_id", "wrong_type", "missing_resource", "malformed_resource", "already_assigned", "invalid_id"])
def test_fresh_address_requires_one_valid_unassigned_owned_calling_channel(tmp_path, problem):
    carrier = FreshCarrier()
    if problem == "missing":
        carrier.addresses = carrier.addresses[1:]
    elif problem == "duplicate":
        carrier.addresses.append(dict(carrier.calling))
    elif problem == "wrong_number":
        carrier.calling["phone_number"] = PERSONAL
    elif problem == "wrong_number_id":
        carrier.calling["phone_number_id"] = OTHER_PROJECT
    elif problem == "wrong_type":
        carrier.calling["type"] = "sip"
    elif problem == "missing_resource":
        carrier.calling.pop("resource_id")
    elif problem == "malformed_resource":
        carrier.calling["resource_id"] = "not-an-id"
    elif problem == "already_assigned":
        carrier.calling["resource_id"] = PREVIOUS_ENDPOINT
    else:
        carrier.calling["id"] = "not-an-id"
    journal = Journal.open(tmp_path, "main", "signalwire")
    with pytest.raises(ProviderError):
        attach(provider(carrier), journal)
    assert writes(carrier) == [] and journal._step("number_claim") is None


def test_fresh_missing_nullable_field_is_not_treated_as_verified_empty_route(tmp_path):
    carrier = FreshCarrier()
    carrier.number.pop("call_fallback_url")
    with pytest.raises(ProviderError, match="safely restored"):
        attach(provider(carrier), Journal.open(tmp_path, "main", "signalwire"))
    assert writes(carrier) == []


def test_fresh_release_unassigns_calling_only_before_deleting_endpoint(tmp_path):
    carrier, p, journal, _ = fresh_ready(tmp_path)
    messaging = dict(carrier.addresses[1])
    assert p.release(journal).ok
    assert carrier.calling["resource_id"] is None and carrier.endpoint is None
    assert carrier.addresses[1] == messaging
    removed = [path for method, path, _ in carrier.calls if method == "DELETE"]
    assert removed == [PHONE_ADDRESSES + "/" + CALLING_ADDRESS, ENDPOINTS + "/" + RESOURCE]
    assert p.release(journal).ok
    assert sum(method == "DELETE" and path.startswith(PHONE_ADDRESSES) for method, path, _ in carrier.calls) == 1


@pytest.mark.parametrize("failure", ["clear", "clear_lost"])
def test_fresh_clear_failure_or_lost_response_retains_endpoint_and_resumes(tmp_path, failure):
    carrier, p, journal, _ = fresh_ready(tmp_path)
    carrier.failure = failure
    result = p.release(journal)
    assert not result.ok and result.leftover["calling_address_id"] == CALLING_ADDRESS
    assert carrier.endpoint is not None
    carrier.failure = None
    assert p.release(journal).ok and carrier.endpoint is None
    expected_clears = 1 if failure == "clear_lost" else 2
    assert sum(method == "DELETE" and path.startswith(PHONE_ADDRESSES) for method, path, _ in carrier.calls) == expected_clears


def test_fresh_clear_requires_readback_before_endpoint_deletion(tmp_path):
    carrier, p, journal, _ = fresh_ready(tmp_path)
    carrier.ignore_clear = True
    assert not p.release(journal).ok and carrier.endpoint is not None
    carrier.ignore_clear = False
    assert p.release(journal).ok


def test_fresh_external_assignment_is_not_overwritten_or_deleted(tmp_path):
    carrier, p, journal, _ = fresh_ready(tmp_path)
    carrier.calling["resource_id"] = PREVIOUS_ENDPOINT
    before = len(writes(carrier))
    with pytest.raises(ProviderError, match="changed outside"):
        attach(p, journal)
    assert not p.release(journal).ok
    assert len(writes(carrier)) == before and carrier.endpoint is not None


@pytest.mark.parametrize("field,value", [("handler_type", "messaging"), ("phone_number", PERSONAL),
    ("phone_number_id", OTHER_PROJECT), ("id", MESSAGING_ADDRESS), ("type", "sip")])
def test_fresh_release_revalidates_address_identity_before_clear(tmp_path, field, value):
    carrier, p, journal, _ = fresh_ready(tmp_path)
    carrier.calling[field] = value
    before = len(writes(carrier))
    assert not p.release(journal).ok
    assert len(writes(carrier)) == before and carrier.endpoint is not None


# Contract checks use the same offline carrier as lifecycle tests so a new write
# cannot gain a readback field merely by having sent it in the request.
def test_contract_response_metadata_and_shared_profile_survive_setup(tmp_path):
    carrier, p, journal, result = ready(tmp_path)
    template = response_fixture("sip_endpoint")
    assert result.state == "ready"
    for field in ("id", "project_id", "display_name", "type", "created_at", "updated_at"):
        assert carrier.endpoint[field] == template[field]
    assert carrier.endpoint["sip_endpoint"]["calling_handler_resource_id"] is None
    assert "password" not in carrier.endpoint["sip_endpoint"]
    assert carrier.profile == response_fixture("sip_profile")
    assert not any(path == PROFILE for _, path, _ in writes(carrier))
    # Password is writable, but absent from create, update and GET response models.
    updated = p._request("PUT", ENDPOINTS + "/" + RESOURCE, json={"password": "ReplacementTestOnly123"})
    assert "password" not in updated["sip_endpoint"]
    assert "password" not in p._request("GET", ENDPOINTS + "/" + RESOURCE)["sip_endpoint"]
    assert "ReplacementTestOnly123" not in journal.path.read_text()
    # Response additions are harmless and never become request or journal fields.
    carrier.endpoint["future_metadata"] = {"response_only": True}
    carrier.endpoint["sip_endpoint"]["future_setting"] = "server value"
    before = len(writes(carrier))
    assert attach(p, journal).state == "ready"
    assert len(writes(carrier)) == before
    assert "future_metadata" not in journal.path.read_text()


@pytest.mark.parametrize("method", ["POST", "PUT"])
@pytest.mark.parametrize("field", ["id", "project_id", "display_name", "created_at", "unsupported_setting"])
def test_contract_endpoint_rejects_response_only_or_unknown_request_fields(tmp_path, method, field):
    carrier, p, _, _ = ready(tmp_path)
    previous = deepcopy(carrier.endpoint)
    body = {"username": USER, "password": PASSWORD, field: "not-writable"}
    path = ENDPOINTS if method == "POST" else ENDPOINTS + "/" + RESOURCE
    with pytest.raises(ProviderError) as error:
        p._request(method, path, json=body)
    assert error.value.status == 422
    assert carrier.endpoint == previous


@pytest.mark.parametrize("field,value", [("codecs", "PCMU"), ("codecs", ["not-a-codec"]),
    ("ciphers", ["not-a-cipher"]), ("encryption", "disabled"), ("call_handler", "relay_sip_endpoint")])
def test_contract_endpoint_rejects_invalid_setting_types_and_enums(tmp_path, field, value):
    carrier, p, _, _ = ready(tmp_path)
    previous = deepcopy(carrier.endpoint)
    with pytest.raises(ProviderError) as error:
        p._request("PUT", ENDPOINTS + "/" + RESOURCE, json={field: value})
    assert error.value.status == 422
    assert carrier.endpoint == previous


@pytest.mark.parametrize("field", ["calling_handler_resource_id", "capabilities", "number", "country_code"])
def test_contract_number_rejects_response_only_fields_in_route_update(tmp_path, field):
    carrier, p, _, _ = ready(tmp_path)
    previous = deepcopy(carrier.number)
    with pytest.raises(ProviderError) as error:
        p._request("PUT", NUMBERS + "/" + NUMBER, json={field: "not-writable"})
    assert error.value.status == 422
    assert carrier.number == previous


def test_contract_address_assignment_rejects_inner_endpoint_id():
    carrier = FreshCarrier()
    p = provider(carrier)
    before = deepcopy(carrier.addresses)
    with pytest.raises(ProviderError) as error:
        p._request("POST", PHONE_ADDRESSES, json={"phone_number_id": NUMBER, "handler_type": "calling", "resource_id": ENDPOINT})
    assert error.value.status == 422
    assert carrier.addresses == before
    assert "channel" not in carrier.calling
    assert carrier.calling["created_at"] and carrier.calling["updated_at"]


@pytest.mark.parametrize("fresh", [False, True], ids=["legacy-route", "calling-address"])
@pytest.mark.parametrize("field,value", [("call_handler", "block-pstn"), ("encryption", "optional"),
    ("codecs", ["PCMA", "PCMU"]), ("ciphers", ["AES_CM_128_HMAC_SHA1_32"]), ("send_as", "not-e164")])
def test_completed_endpoint_drift_blocks_first_route_write(tmp_path, monkeypatch, fresh, field, value):
    carrier = FreshCarrier() if fresh else Carrier()
    p = provider(carrier)
    journal = Journal.open(tmp_path, "main", "signalwire")
    original_finish = journal.finish

    def interrupted_finish(step, **result):
        original_finish(step, **result)
        if step == "endpoint_configured":
            raise RuntimeError("simulated crash after endpoint confirmation")

    monkeypatch.setattr(journal, "finish", interrupted_finish)
    with pytest.raises(RuntimeError, match="simulated crash"):
        attach(p, journal)
    assert "endpoint_configured" in journal.steps_done()
    assert journal._step("number_routed") is None
    carrier.endpoint["sip_endpoint"][field] = value
    before = deepcopy(writes(carrier))
    number_before = deepcopy(carrier.number)
    # Reopen exactly as CLI resume does, retaining the completed journal stage.
    with pytest.raises(ProviderError):
        attach(p, Journal.open(tmp_path, "main"))
    assert writes(carrier) == before
    assert carrier.number == number_before
    if fresh:
        assert carrier.calling["resource_id"] is None
    else:
        assert carrier.number["call_sip_endpoint_id"] == PREVIOUS_ENDPOINT


@pytest.mark.parametrize("fresh", [False, True], ids=["legacy-route", "calling-address"])
@pytest.mark.parametrize("field", ["project_id", "resource_id", "endpoint_id", "username"])
def test_completed_endpoint_identity_drift_never_mutates_route(tmp_path, fresh, field):
    carrier, p, journal, _ = (fresh_ready if fresh else ready)(tmp_path)
    if field == "project_id":
        carrier.endpoint["project_id"] = OTHER_PROJECT
    elif field == "resource_id":
        carrier.endpoint["id"] = PREVIOUS_ENDPOINT
    elif field == "endpoint_id":
        carrier.endpoint["sip_endpoint"]["id"] = PREVIOUS_ENDPOINT
    else:
        carrier.endpoint["sip_endpoint"]["username"] = "otheruser"
    before = deepcopy(writes(carrier))
    with pytest.raises(ProviderError, match="ownership"):
        attach(p, journal)
    assert writes(carrier) == before


@pytest.mark.parametrize("fresh", [False, True], ids=["legacy-route", "calling-address"])
@pytest.mark.parametrize("local_committed", [False, True])
def test_completed_endpoint_preserves_intentional_caller_id_on_resume(tmp_path, fresh, local_committed):
    carrier, p, journal, _ = (fresh_ready if fresh else ready)(tmp_path)
    record = response_fixture("verified_caller_id")
    record.update(verified=True, verified_at="2026-01-01T00:01:00Z", status="Verified")
    carrier.verified[VERIFICATION] = record
    assert p.apply_caller_id(journal, PERSONAL) == PERSONAL
    if local_committed:
        p.finish_caller_id(journal)
    before = deepcopy(writes(carrier))
    result = attach(p, Journal.open(tmp_path, "main"))
    assert result.state == "ready" and result.line["caller_id"] == PERSONAL
    assert writes(carrier) == before

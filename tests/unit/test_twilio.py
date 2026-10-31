import copy
import json
import urllib.parse
from pathlib import Path

import httpx
import pytest

from ai_phone.lines.journal import Journal
from ai_phone.providers.base import ProviderError
from ai_phone.providers.twilio import SIGNALLING_CIDRS, TwilioProvider

SID = "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
CONTRACT = json.loads((Path(__file__).parents[1] / "fixtures/twilio/provisioning.json").read_text())


def _response(kind: str, sid: str, **values) -> dict:
    record = copy.deepcopy(CONTRACT["response_templates"][kind])
    assert values.keys() <= record.keys(), "test response fields must come from the documented contract"
    record.update(account_sid=SID, sid=sid, **values)
    return record


def _page(request: httpx.Request, key: str, records: list[dict]) -> dict:
    dialect = "trunking" if request.url.host == "trunking.twilio.com" else "classic"
    page = copy.deepcopy(CONTRACT["pagination_templates"][dialect])
    if dialect == "trunking":
        page["meta"].update(url=str(request.url), first_page_url=str(request.url), key=key)
    else:
        page.update(uri=request.url.raw_path.decode(), first_page_uri=request.url.raw_path.decode())
    page[key] = records
    return page


def _accept(kind: str, form: dict) -> None:
    assert form.keys() <= set(CONTRACT["request_fields"][kind]), "unsupported Twilio request field"


class Carrier:
    def __init__(self, trial: bool = False):
        self.credlists, self.creds, self.domains, self.mappings, self.numbers, self.trunks, self.trunk_cls, self.callerids = {}, {}, {}, {}, {}, {}, {}, {}
        self.calls = []
        self.updates = []  # (number sid, form) of every POST that changes an existing number
        self.trial = trial  # trial accounts get no Elastic SIP Trunking

    @staticmethod
    def domain_response(sid, form):
        return _response("domain", sid, domain_name=form["DomainName"], friendly_name=form.get("FriendlyName"),
                         sip_registration=form.get("SipRegistration") == "true", voice_method=form.get("VoiceMethod", "POST"))

    @staticmethod
    def number_response(sid, form):
        return _response("number", sid, phone_number=form["PhoneNumber"], friendly_name=form.get("FriendlyName"),
                         voice_url=form.get("VoiceUrl"), voice_method=form.get("VoiceMethod", "POST"),
                         trunk_sid=form.get("TrunkSid"), voice_application_sid=form.get("VoiceApplicationSid"),
                         capabilities=form.get("Capabilities", {"voice": True, "sms": True}))

    def __call__(self, req: httpx.Request) -> httpx.Response:
        self.calls.append((req.method, req.url.host, req.url.path))
        assert req.headers["authorization"].startswith("Basic ")
        form = dict(urllib.parse.parse_qsl(req.content.decode())) if req.content else {}
        p, m, host = req.url.path, req.method, req.url.host
        if host == "pricing.twilio.com":
            return httpx.Response(200, json={"phone_number_prices": [{"number_type": "local", "current_price": "1.15", "base_price": "1.15"}], "price_unit": "USD"})
        if m == "GET" and p.endswith("/AvailablePhoneNumbers/GB/Local.json"):
            return httpx.Response(200, json={"available_phone_numbers": [{"phone_number": "+442079460000"}]})
        if m == "POST" and p.endswith("/SIP/CredentialLists.json"):
            _accept("credential_list", form)
            sid = f"CL{len(self.credlists) + 1}"
            self.credlists[sid] = form
            return httpx.Response(201, json=_response("credential_list", sid, friendly_name=form["FriendlyName"]))
        if m == "GET" and p.endswith("/SIP/CredentialLists.json"):
            return httpx.Response(200, json=_page(req, "credential_lists", [_response("credential_list", k, friendly_name=v["FriendlyName"]) for k, v in self.credlists.items()]))
        if m == "POST" and "/Credentials.json" in p:
            _accept("credential", form)
            sid = f"CR{len(self.creds) + 1}"
            self.creds[sid] = {**form, "CredentialListSid": p.split("/")[-2]}
            return httpx.Response(201, json=_response("credential", sid, username=form["Username"], credential_list_sid=p.split("/")[-2]))
        if m == "GET" and "/Credentials.json" in p:
            return httpx.Response(200, json=_page(req, "credentials", [_response("credential", k, username=v["Username"], credential_list_sid=v["CredentialListSid"]) for k, v in self.creds.items() if v["CredentialListSid"] == p.split("/")[-2]]))
        if m == "POST" and p.endswith("/SIP/Domains.json"):
            _accept("domain", form)
            sid = f"SD{len(self.domains) + 1}"
            self.domains[sid] = form
            return httpx.Response(201, json=self.domain_response(sid, form))
        if m == "GET" and p.endswith("/SIP/Domains.json"):
            return httpx.Response(200, json=_page(req, "domains", [self.domain_response(k, v) for k, v in self.domains.items()]))
        if m == "POST" and "/Auth/Registrations/CredentialListMappings.json" in p:
            _accept("domain_mapping", form)
            self.mappings[p.split("/")[6]] = form["CredentialListSid"]
            return httpx.Response(201, json=_response("domain_mapping", form["CredentialListSid"]))
        if m == "GET" and "/Auth/Registrations/CredentialListMappings.json" in p:
            dsid = p.split("/")[6]
            return httpx.Response(200, json=_page(req, "contents", [_response("domain_mapping", self.mappings[dsid])] if dsid in self.mappings else []))
        if m == "POST" and p.endswith("/IncomingPhoneNumbers.json"):
            _accept("number", form)
            sid = f"PN{len(self.numbers) + 1}"
            self.numbers[sid] = form
            return httpx.Response(201, json=self.number_response(sid, form))
        if m == "GET" and p.endswith("/IncomingPhoneNumbers.json"):
            want = req.url.params.get("PhoneNumber")  # no filter: every number on the account
            return httpx.Response(200, json=_page(req, "incoming_phone_numbers", [self.number_response(k, v)
                for k, v in self.numbers.items() if want is None or v["PhoneNumber"] == want]))
        if m == "POST" and "/IncomingPhoneNumbers/" in p:
            sid = p.rsplit("/", 1)[1].removesuffix(".json")
            if sid not in self.numbers:
                return httpx.Response(404, json={"message": f"no number {sid}"})
            _accept("number", form)
            self.updates.append((sid, form))
            self.numbers[sid].update(form)
            return httpx.Response(200, json=self.number_response(sid, self.numbers[sid]))
        if host == "trunking.twilio.com" and m == "POST" and p == "/v1/Trunks":
            if self.trial:
                return httpx.Response(403, json={"code": 20003, "message": "SIP trunking is only available after you upgrade your account"})
            _accept("trunk", form)
            sid = f"TK{len(self.trunks) + 1}"
            self.trunks[sid] = form
            return httpx.Response(201, json=_response("trunk", sid, friendly_name=form["FriendlyName"], domain_name=form["DomainName"]))
        if host == "trunking.twilio.com" and m == "GET" and p == "/v1/Trunks":
            return httpx.Response(200, json=_page(req, "trunks", [_response("trunk", k, friendly_name=v["FriendlyName"], domain_name=v["DomainName"]) for k, v in self.trunks.items()]))
        if host == "trunking.twilio.com" and m == "POST" and p.endswith("/CredentialLists"):
            _accept("trunk_mapping", form)
            self.trunk_cls[p.split("/")[3]] = form["CredentialListSid"]
            return httpx.Response(201, json=_response("trunk_mapping", form["CredentialListSid"], trunk_sid=p.split("/")[3]))
        if host == "trunking.twilio.com" and m == "GET" and p.endswith("/CredentialLists"):
            tsid = p.split("/")[3]
            return httpx.Response(200, json=_page(req, "credential_lists", [_response("trunk_mapping", self.trunk_cls[tsid], trunk_sid=tsid)] if tsid in self.trunk_cls else []))
        if m == "DELETE":
            for store in (self.numbers, self.domains, self.credlists, self.trunks):
                store.pop(p.rstrip(".json").rsplit("/", 1)[1], None)
            return httpx.Response(204)
        if m == "POST" and p.endswith("/OutgoingCallerIds.json"):
            self.callerids[form["PhoneNumber"]] = "pending"
            return httpx.Response(201, json={"validation_code": "654321", "phone_number": form["PhoneNumber"]})
        if m == "GET" and p.endswith("/OutgoingCallerIds.json"):
            n = req.url.params.get("PhoneNumber")
            return httpx.Response(200, json={"outgoing_caller_ids": [{"phone_number": n}] if self.callerids.get(n) == "verified" else []})
        return httpx.Response(404, json={"message": f"unexpected {m} {host}{p}"})


def _provider(carrier):
    return TwilioProvider(key=SID, secret="tok", http=httpx.Client(transport=httpx.MockTransport(carrier)))


def test_cidrs_pinned():
    import ipaddress
    assert SIGNALLING_CIDRS and all(ipaddress.ip_network(c) for c in SIGNALLING_CIDRS)


def test_search_uses_pricing_api():
    n = _provider(Carrier()).search_numbers("GB", "20", 5)[0]
    assert n.e164 == "+442079460000" and n.monthly_cost == 1.15 and n.currency == "USD"


def test_provision_creates_domain_number_trunk(tmp_path: Path):
    c = Carrier()
    p = _provider(c)
    j = Journal.open(tmp_path, "main", "twilio")
    r = p.provision(j, p.search_numbers("GB", None, 1)[0], "main", "aimain1234", "Passw0rdPassw0rdPassw0rd")
    assert r.state == "ready"
    dom = next(iter(c.domains.values()))
    assert dom["DomainName"].startswith("ai-phone-main-") and dom["DomainName"].endswith(".sip.twilio.com") and dom["SipRegistration"] == "true"
    assert c.mappings["SD1"] == "CL1" and c.creds["CR1"]["Username"] == "aimain1234"
    num = next(iter(c.numbers.values()))
    assert num["PhoneNumber"] == "+442079460000" and "twimlets.com/echo" in num["VoiceUrl"] and "aimain1234%40ai-phone-main-" in num["VoiceUrl"]
    trunk = next(iter(c.trunks.values()))
    assert trunk["DomainName"].endswith(".pstn.twilio.com") and c.trunk_cls["TK1"] == "CL1"
    assert r.line["registrar"] == f"sip:{dom['DomainName']}" and r.line["outbound_proxy"] == f"sip:{trunk['DomainName']}"
    assert r.line["dial_domain"] == trunk["DomainName"] and r.line["register_via_proxy"] is False and r.line["user"] == "aimain1234"
    assert r.line["provider"]["resources"]["domain_sid"] == "SD1"
    assert [s["name"] for s in j.doc["steps"]] == ["credential_list", "credential", "domain", "domain_mapping", "number", "trunk", "trunk_credential_list"]


def test_planned_domain_name_is_journaled_before_creation_and_reused(tmp_path: Path):
    c = Carrier()
    crash = {"armed": True}

    def handler(req):  # the carrier creates the domain, then the response is lost before the journal's `done`
        r = c(req)
        if crash["armed"] and req.method == "POST" and req.url.path.endswith("/SIP/Domains.json"):
            raise httpx.ConnectError("lost")
        return r

    p = TwilioProvider(key=SID, secret="tok", http=httpx.Client(transport=httpx.MockTransport(handler)))
    j = Journal.open(tmp_path, "main", "twilio")
    n = p.search_numbers("GB", None, 1)[0]
    with pytest.raises(ProviderError):
        p.provision(j, n, "main", "aimain1234", "Passw0rdPassw0rdPassw0rd")
    planned = next(s["planned"]["domain_name"] for s in j.doc["steps"] if s["name"] == "domain")
    assert "SD1" not in c.mappings  # the crash happened before the registration mapping
    crash["armed"] = False
    r = p.provision(Journal.open(tmp_path, "main"), n, "main", "aimain1234", "Passw0rdPassw0rdPassw0rd")
    assert len(c.domains) == 1 and next(iter(c.domains.values()))["DomainName"] == planned and r.state == "ready"
    assert c.mappings["SD1"] == "CL1" and len(c.creds) == 1 and c.trunk_cls["TK1"] == "CL1"  # the steps after the crash ran once each


MUTATING = 7  # credential list, credential, domain, mapping, number, trunk, trunk credential list


@pytest.mark.parametrize("crash_at", list(range(1, MUTATING + 1)))
@pytest.mark.parametrize("later_page", [False, True])
def test_a_lost_response_after_any_step_is_reconciled_without_duplicates(tmp_path: Path, crash_at: int, later_page: bool):
    c = Carrier()
    posts = {"n": 0, "armed": True}

    def handler(req):
        r = c(req)  # the carrier always processes the request; we lose the response on the crash_at-th mutation
        if later_page and req.method == "GET":
            body = r.json()
            key = next((k for k in ("credentials", "credential_lists", "domains", "contents", "incoming_phone_numbers", "trunks") if k in body), None)
            if key and not req.url.params.get("PageToken"):
                body[key] = []  # the resource exists, but only on the second page
                next_url = req.url.copy_add_param("PageToken", "second")
                if req.url.host == "trunking.twilio.com":
                    body["meta"]["next_page_url"] = str(next_url)
                else:
                    body["next_page_uri"] = next_url.raw_path.decode()
                r = httpx.Response(200, json=body)
        if req.method == "POST" and posts["armed"]:
            posts["n"] += 1
            if posts["n"] == crash_at:
                raise httpx.ConnectError("lost")
        return r

    p = TwilioProvider(key=SID, secret="tok", http=httpx.Client(transport=httpx.MockTransport(handler)))
    n = p.search_numbers("GB", None, 1)[0]
    with pytest.raises(ProviderError):
        p.provision(Journal.open(tmp_path, "main", "twilio"), n, "main", "aimain1234", "Passw0rdPassw0rdPassw0rd")
    posts["armed"] = False
    r = p.provision(Journal.open(tmp_path, "main"), n, "main", "aimain1234", "Passw0rdPassw0rdPassw0rd")
    assert r.state == "ready"
    assert (len(c.credlists), len(c.creds), len(c.domains), len(c.mappings), len(c.numbers), len(c.trunks), len(c.trunk_cls)) == (1, 1, 1, 1, 1, 1, 1)


def test_errors_never_leak_the_sid_or_token():
    def boom(req):
        return httpx.Response(401, json={"message": f"bad auth for {SID} with tok"})
    p = TwilioProvider(key=SID, secret="tok", http=httpx.Client(transport=httpx.MockTransport(boom)))
    with pytest.raises(ProviderError) as e:
        p.search_numbers("GB", None, 1)
    assert SID not in str(e.value) and "tok" not in str(e.value).replace("token", "")


def test_injected_client_still_gets_the_request_timeout():
    seen = {}

    def handler(req):
        seen["timeout"] = req.extensions.get("timeout")
        return httpx.Response(200, json={"available_phone_numbers": [], "phone_number_prices": [], "price_unit": "USD"})

    p = TwilioProvider(key=SID, secret="tok", http=httpx.Client(transport=httpx.MockTransport(handler), timeout=None))
    p.search_numbers("GB", None, 1)
    assert seen["timeout"]["read"] == 20.0


def test_release_and_caller_id(tmp_path: Path):
    c = Carrier()
    p = _provider(c)
    j = Journal.open(tmp_path, "main", "twilio")
    p.provision(j, p.search_numbers("GB", None, 1)[0], "main", "aimain1234", "Passw0rdPassw0rdPassw0rd")
    assert p.release(j).ok and not c.numbers and not c.domains and not c.credlists and not c.trunks
    step = p.start_caller_id_verification("+447700900123")
    assert step.kind == "enter_code_on_phone" and "654321" in step.instructions
    assert p.complete_caller_id_verification(step, None) is False
    c.callerids["+447700900123"] = "verified"
    assert p.complete_caller_id_verification(step, None) is True


def test_domain_name_collision_gets_one_fresh_suffix(tmp_path: Path):
    c = Carrier()
    seen = []

    def handler(req):
        if req.method == "POST" and req.url.path.endswith("/SIP/Domains.json"):
            seen.append(dict(urllib.parse.parse_qsl(req.content.decode()))["DomainName"])
            if len(seen) == 1:
                return httpx.Response(400, json={"message": "A SIP Domain with that name already exists"})
        return c(req)

    p = TwilioProvider(key=SID, secret="tok", http=httpx.Client(transport=httpx.MockTransport(handler)))
    j = Journal.open(tmp_path, "main", "twilio")
    r = p.provision(j, p.search_numbers("GB", None, 1)[0], "main", "aimain1234", "Passw0rdPassw0rdPassw0rd")
    assert r.state == "ready" and len(seen) == 2 and seen[0] != seen[1]
    assert next(s["planned"]["domain_name"] for s in j.doc["steps"] if s["name"] == "domain") == seen[1] == r.line["registrar"][4:]


def test_server_error_mentioning_already_is_not_a_name_collision(tmp_path: Path):
    """Only a 4xx saying the name is taken earns the fresh suffix. A 5xx (or a transport error) is retryable in
    place; renaming after one would orphan the domain the failed request may have created."""
    c = Carrier()
    seen = []

    def handler(req):
        if req.method == "POST" and req.url.path.endswith("/SIP/Domains.json"):
            seen.append(dict(urllib.parse.parse_qsl(req.content.decode()))["DomainName"])
            return httpx.Response(503, json={"message": "that domain is already being provisioned; try again"})
        return c(req)

    p = TwilioProvider(key=SID, secret="tok", http=httpx.Client(transport=httpx.MockTransport(handler)))
    j = Journal.open(tmp_path, "main", "twilio")
    with pytest.raises(ProviderError) as ei:
        p.provision(j, p.search_numbers("GB", None, 1)[0], "main", "aimain1234", "Passw0rdPassw0rdPassw0rd")
    assert ei.value.status == 503 and ei.value.retryable and len(seen) == 1  # not renamed, not re-POSTed


def test_trial_account_gets_an_inbound_only_line_and_resume_adds_outbound_after_upgrade(tmp_path: Path):
    carrier = Carrier(trial=True)
    p = _provider(carrier)
    j = Journal.open(tmp_path, "tw", "twilio")
    num = p.search_numbers("GB", None, 1)[0]
    r = p.provision(j, num, "tw", "aitw1234", "Passw0rdPassw0rdPassw0rd")
    assert r.state == "ready" and "inbound calls only" in r.message and "resume" in r.message
    assert "dial_domain" not in r.line and "outbound_proxy" not in r.line and "trunk_sid" not in r.line["provider"]["resources"]
    assert r.line["registrar"].endswith(".sip.twilio.com") and carrier.numbers and carrier.domains and not carrier.trunks
    # the account is upgraded later: resume re-runs only the trunk steps
    carrier.trial = False
    r2 = p.provision(j, num, "tw", "aitw1234", "Passw0rdPassw0rdPassw0rd")
    assert r2.state == "ready" and "inbound calls only" not in r2.message
    assert r2.line["dial_domain"].endswith(".pstn.twilio.com") and len(carrier.trunks) == 1 and len(carrier.numbers) == 1
    assert list(carrier.trunk_cls.values()) == [r2.line["provider"]["resources"]["credential_list_sid"]]
    rel = p.release(j)
    assert rel.ok and not carrier.trunks and not carrier.numbers and not carrier.domains and not carrier.credlists


# ---- a number the person already has ------------------------------------------------------------

def _owned(c: Carrier, **extra) -> Carrier:
    c.numbers["PN9"] = {"PhoneNumber": "+15550100001", "VoiceUrl": "https://example.com/old", "VoiceMethod": "POST", **extra}
    return c


def _number_posts(c: Carrier, path_end: str) -> list:
    return [x for x in c.calls if x[0] == "POST" and x[2].endswith(path_end)]


def test_owned_numbers_lists_voice_numbers():
    c = _owned(Carrier())
    c.numbers["PN8"] = {"PhoneNumber": "+15550100002"}
    c.numbers["PN7"] = {"PhoneNumber": "+15550100003", "Capabilities": {"voice": False, "sms": True}}  # texts only
    p = _provider(c)
    owned = p.owned_numbers()
    assert [n.e164 for n in owned] == ["+15550100001", "+15550100002"]
    assert [n.source_id for n in owned] == ["PN9", "PN8"]
    assert not [x for x in c.calls if x[0] == "POST"]


def test_attach_routes_an_owned_number_without_buying(tmp_path: Path):
    from ai_phone.providers.base import Number
    from ai_phone.providers.twilio import _twiml_url
    c = _owned(Carrier())
    p = _provider(c)
    j = Journal.open(tmp_path, "main", "twilio")
    r = p.attach(j, Number("+15550100001", None, None, False, "PN9"), "main", "u1234", "pw")
    assert not _number_posts(c, "/IncomingPhoneNumbers.json"), "nothing may be bought"
    assert len(_number_posts(c, "/IncomingPhoneNumbers/PN9.json")) == 1
    domain = next(iter(c.domains.values()))["DomainName"]
    assert c.updates == [("PN9", {"VoiceUrl": _twiml_url("u1234", domain), "VoiceMethod": "GET"})]
    assert r.state == "ready" and r.line["number"] == "+15550100001" and r.line["caller_id"] == "+15550100001"
    resources = r.line["provider"]["resources"]
    assert resources == {"credential_list_sid": "CL1", "domain_sid": "SD1", "brought_number_sid": "PN9", "trunk_sid": "TK1"}
    assert r.line["registrar"] == f"sip:{domain}"
    res = j.resources()
    assert res["number_brought"] is True and res["brought_number_sid"] == "PN9" and "number_sid" not in res
    assert res["previous_voice_url"] == "https://example.com/old" and res["previous_voice_method"] == "POST"
    assert [s["name"] for s in j.doc["steps"]] == ["claim", "credential_list", "credential", "domain", "domain_mapping", "route",
                                                  "trunk", "trunk_credential_list"]
    assert list(c.numbers) == ["PN9"]


def test_attach_refuses_a_number_not_on_the_account(tmp_path: Path):
    from ai_phone.providers.base import Number
    c = Carrier()
    p = _provider(c)
    with pytest.raises(ProviderError, match="not on this Twilio account"):
        p.attach(Journal.open(tmp_path, "main", "twilio"), Number("+15550100001", None, None, False, "PN9"), "main", "u1234", "pw")
    assert c.credlists == {} and c.domains == {} and c.numbers == {}


def test_attach_refuses_a_number_routed_by_a_trunk(tmp_path: Path):
    from ai_phone.providers.base import Number
    c = _owned(Carrier(), TrunkSid="TK77")
    p = _provider(c)
    with pytest.raises(ProviderError, match="SIP trunk or a TwiML app") as ei:
        p.attach(Journal.open(tmp_path, "main", "twilio"), Number("+15550100001", None, None, False, "PN9"), "main", "u1234", "pw")
    assert "+15550100001" in str(ei.value) and "Twilio console" in str(ei.value)
    assert c.credlists == {} and c.domains == {} and c.trunks == {} and c.updates == []


def test_attach_refuses_a_number_routed_by_a_twiml_app(tmp_path: Path):
    from ai_phone.providers.base import Number
    c = _owned(Carrier(), VoiceApplicationSid="AP1")
    with pytest.raises(ProviderError, match="SIP trunk or a TwiML app"):
        _provider(c).attach(Journal.open(tmp_path, "main", "twilio"), Number("+15550100001", None, None, False, "PN9"), "main", "u1234", "pw")
    assert c.credlists == {} and c.updates == []


def test_release_of_a_brought_number_restores_it_and_keeps_it(tmp_path: Path):
    from ai_phone.providers.base import Number
    c = _owned(Carrier())
    p = _provider(c)
    j = Journal.open(tmp_path, "main", "twilio")
    p.attach(j, Number("+15550100001", None, None, False, "PN9"), "main", "u1234", "pw")
    c.updates.clear()
    assert p.release(j).ok
    assert not [x for x in c.calls if x[0] == "DELETE" and x[2].endswith("/IncomingPhoneNumbers/PN9.json")]
    assert c.updates == [("PN9", {"VoiceUrl": "https://example.com/old", "VoiceMethod": "POST"})]
    assert c.numbers["PN9"]["VoiceUrl"] == "https://example.com/old" and c.numbers["PN9"]["VoiceMethod"] == "POST"
    assert not c.domains and not c.credlists and not c.trunks


def _old_format(j: Journal, key: str) -> None:
    """What e4a8c67 wrote: the brought number under the bought number's key, flagged `number_brought`."""
    for s in j.doc["steps"]:
        if f"brought_{key}" in s["result"]:
            s["result"][key] = s["result"].pop(f"brought_{key}")
    j._save()


def test_release_of_an_old_format_brought_number_keeps_it(tmp_path: Path):
    from ai_phone.providers.base import Number
    c = _owned(Carrier())
    p = _provider(c)
    j = Journal.open(tmp_path, "main", "twilio")
    p.attach(j, Number("+15550100001", None, None, False, "PN9"), "main", "u1234", "pw")
    _old_format(j, "number_sid")
    assert j.resources()["number_sid"] == "PN9" and "brought_number_sid" not in j.resources()
    c.updates.clear()
    assert p.release(j).ok
    assert not [x for x in c.calls if x[0] == "DELETE" and "IncomingPhoneNumbers" in x[2]]
    assert c.updates == [("PN9", {"VoiceUrl": "https://example.com/old", "VoiceMethod": "POST"})] and "PN9" in c.numbers
    assert not c.domains and not c.credlists and not c.trunks


def test_release_of_a_brought_number_that_cannot_be_restored_keeps_it_and_says_so(tmp_path: Path):
    from ai_phone.providers.base import Number
    c = _owned(Carrier())
    p = _provider(c)
    j = Journal.open(tmp_path, "main", "twilio")
    p.attach(j, Number("+15550100001", None, None, False, "PN9"), "main", "u1234", "pw")
    del c.numbers["PN9"]  # the restore is refused (404); nothing may be deleted in its place
    res = p.release(j)
    assert not res.ok and "stays on your account" in res.leftover["brought_number_sid"] and j.state == "rollback_failed"
    assert not [x for x in c.calls if x[0] == "DELETE" and "IncomingPhoneNumbers" in x[2]]
    assert not c.domains and not c.credlists and not c.trunks


def test_attach_resumes_after_a_crash_in_route(tmp_path: Path):
    from ai_phone.providers.base import Number
    c = _owned(Carrier())
    crash = {"armed": True}

    def handler(req):
        if crash["armed"] and req.method == "POST" and req.url.path.endswith("/IncomingPhoneNumbers/PN9.json"):
            crash["armed"] = False
            raise httpx.ConnectError("lost")
        return c(req)

    p = TwilioProvider(key=SID, secret="tok", http=httpx.Client(transport=httpx.MockTransport(handler)))
    n = Number("+15550100001", None, None, False, "PN9")
    with pytest.raises(ProviderError):
        p.attach(Journal.open(tmp_path, "main", "twilio"), n, "main", "u1234", "pw")
    j = Journal.open(tmp_path, "main")
    r = p.attach(j, n, "main", "u1234", "pw")
    assert r.state == "ready" and len(c.credlists) == 1 and len(c.domains) == 1 and len(c.creds) == 1 and len(c.trunks) == 1
    assert "twimlets.com/echo" in c.numbers["PN9"]["VoiceUrl"] and c.numbers["PN9"]["VoiceMethod"] == "GET"
    assert j.resources()["previous_voice_url"] == "https://example.com/old"  # the claim was not re-read after routing
    assert p.poll(j).line["number"] == "+15550100001"


def _trunk_fails_once(c: Carrier) -> TwilioProvider:
    failed = {"done": False}

    def handler(req):
        if not failed["done"] and req.method == "POST" and req.url.path == "/v1/Trunks":
            failed["done"] = True
            return httpx.Response(500, json={"message": "try again"})
        return c(req)

    return TwilioProvider(key=SID, secret="tok", http=httpx.Client(transport=httpx.MockTransport(handler)))


def test_a_buy_then_a_bring_in_one_journal_releases_the_bought_number(tmp_path: Path):
    """A journal shared by a failed purchase and a later attach: the bought number is still deleted, the brought one kept."""
    from ai_phone.providers.base import Number
    c = _owned(Carrier())
    p = _trunk_fails_once(c)
    j = Journal.open(tmp_path, "main", "twilio")
    with pytest.raises(ProviderError):
        p.provision(j, p.search_numbers("GB", None, 1)[0], "main", "u1234", "pw")
    assert c.numbers["PN2"]["PhoneNumber"] == "+442079460000"
    assert p.attach(j, Number("+15550100001", None, None, False, "PN9"), "main", "u1234", "pw").state == "ready"
    res = j.resources()
    assert res["number_sid"] == "PN2" and res["brought_number_sid"] == "PN9"
    c.updates.clear()
    assert p.release(j).ok
    assert list(c.numbers) == ["PN9"] and c.numbers["PN9"]["VoiceUrl"] == "https://example.com/old"
    assert c.updates == [("PN9", {"VoiceUrl": "https://example.com/old", "VoiceMethod": "POST"})]
    assert not c.domains and not c.credlists and not c.trunks


def test_a_bring_then_a_buy_in_one_journal_restores_the_brought_number(tmp_path: Path):
    from ai_phone.providers.base import Number
    c = _owned(Carrier())
    p = _trunk_fails_once(c)
    j = Journal.open(tmp_path, "main", "twilio")
    with pytest.raises(ProviderError):
        p.attach(j, Number("+15550100001", None, None, False, "PN9"), "main", "u1234", "pw")
    assert "twimlets.com/echo" in c.numbers["PN9"]["VoiceUrl"]
    assert p.provision(j, p.search_numbers("GB", None, 1)[0], "main", "u1234", "pw").state == "ready"
    c.updates.clear()
    assert p.release(j).ok
    assert list(c.numbers) == ["PN9"], "the bought number is released"
    assert c.updates == [("PN9", {"VoiceUrl": "https://example.com/old", "VoiceMethod": "POST"})]
    assert c.numbers["PN9"]["VoiceUrl"] == "https://example.com/old" and not c.domains


# ---- documented response contracts and paginated recovery --------------------------------------

def test_setup_uses_response_fields_without_requiring_request_echo(tmp_path):
    carrier = Carrier()
    responses = []

    def handler(request):
        response = carrier(request)
        if request.method == "POST":
            responses.append((request.url.path, response.json()))
        return response

    provider = _provider(handler)
    number = provider.search_numbers("GB", None, 1)[0]
    result = provider.provision(Journal.open(tmp_path, "main", "twilio"), number, "main", "aimain1234", "Passw0rdPassw0rdPassw0rd")
    assert result.state == "ready"
    credential = next(body for path, body in responses if path.endswith("/Credentials.json"))
    assert credential["username"] == "aimain1234"
    assert "password" not in credential and "Password" not in credential
    domain = next(body for path, body in responses if path.endswith("/SIP/Domains.json"))
    assert domain["sip_registration"] is True and domain["voice_url"] is None
    assert all(not any(key[:1].isupper() for key in body) for _, body in responses)


def test_owned_number_listing_reads_all_pages():
    first = _response("number", "PN1", phone_number="+15550100001")
    second = _response("number", "PN2", phone_number="+15550100002")
    requests = []

    def handler(request):
        requests.append(request)
        page = _page(request, "incoming_phone_numbers", [second if request.url.params.get("PageToken") else first])
        if len(requests) == 1:
            page["next_page_uri"] = request.url.copy_add_param("PageToken", "second").raw_path.decode()
        return httpx.Response(200, json=page)

    numbers = _provider(handler).owned_numbers()
    assert [number.e164 for number in numbers] == ["+15550100001", "+15550100002"]
    assert len(requests) == 2 and all(request.method == "GET" for request in requests)


@pytest.mark.parametrize("following", [
    "https://example.com/steal", "http://api.twilio.com/2010-04-01/Accounts/ignored",
    "https://api.twilio.com/2010-04-01/Accounts/another/IncomingPhoneNumbers.json",
    "https://api.twilio.com/2010-04-01/Accounts/ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx/SIP/Domains.json",
    "https://someone@api.twilio.com/2010-04-01/Accounts/ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx/IncomingPhoneNumbers.json",
    "?Page=1#fragment", "https://[invalid", 42,
])
def test_pagination_never_sends_auth_to_another_destination(following):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"incoming_phone_numbers": [], "next_page_uri": following})

    with pytest.raises(ProviderError, match="pagination link"):
        _provider(handler).owned_numbers()
    assert len(requests) == 1


def test_paginated_lookup_rejects_redirects_even_with_redirecting_client():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(302, headers={"Location": "https://example.com/steal"})

    provider = TwilioProvider(key=SID, secret="tok", http=httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True))
    with pytest.raises(ProviderError) as error:
        provider.owned_numbers()
    assert error.value.status == 302 and len(requests) == 1


def test_repeated_pagination_link_stops_without_creating_resources(tmp_path):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"credential_lists": [], "next_page_uri": request.url.raw_path.decode()})

    journal = Journal.open(tmp_path, "main", "twilio")
    journal.start("credential_list", name="ai-phone:main")
    with pytest.raises(ProviderError, match="repeated a pagination link"):
        _provider(handler)._sip_domain(journal, "main", "aimain1234", "Passw0rdPassw0rdPassw0rd")
    assert len(requests) == 1 and requests[0].method == "GET"


def test_matching_resource_names_on_different_pages_are_ambiguous(tmp_path):
    requests = []

    def handler(request):
        requests.append(request)
        body = _page(request, "credential_lists", [_response("credential_list", f"CL{len(requests)}", friendly_name="ai-phone:main")])
        if len(requests) == 1:
            body["next_page_uri"] = request.url.copy_add_param("PageToken", "second").raw_path.decode()
        return httpx.Response(200, json=body)

    journal = Journal.open(tmp_path, "main", "twilio")
    journal.start("credential_list", name="ai-phone:main")
    with pytest.raises(ProviderError, match="multiple matching resources"):
        _provider(handler)._sip_domain(journal, "main", "aimain1234", "Passw0rdPassw0rdPassw0rd")
    assert len(requests) == 2 and all(request.method == "GET" for request in requests)


@pytest.mark.parametrize("body", [[], {}, {"incoming_phone_numbers": None}, {"incoming_phone_numbers": [None]},
                                  {"incoming_phone_numbers": [], "meta": []}])
def test_malformed_list_is_not_treated_as_an_empty_account(body):
    with pytest.raises(ProviderError, match="invalid"):
        _provider(lambda request: httpx.Response(200, json=body)).owned_numbers()



@pytest.mark.parametrize("step, pagination", [
    pytest.param("credential_list", {}, id="classic-missing-marker"),
    pytest.param("credential_list", {"next_page_uri": ""}, id="classic-empty-marker"),
    pytest.param("credential_list", {"meta": {"next_page_url": None}}, id="classic-wrong-dialect"),
    pytest.param("credential_list", {"next_page_uri": "?PageToken=second", "meta": {}}, id="classic-mixed-empty-meta"),
    pytest.param("credential_list", {"next_page_uri": None, "meta": {"next_page_url": "?PageToken=second"}}, id="classic-mixed-next"),
    pytest.param("trunk", {}, id="trunking-missing-meta"),
    pytest.param("trunk", {"meta": {}}, id="trunking-missing-marker"),
    pytest.param("trunk", {"meta": {"next_page_url": ""}}, id="trunking-empty-marker"),
    pytest.param("trunk", {"next_page_uri": None}, id="trunking-wrong-dialect"),
    pytest.param("trunk", {"meta": {"next_page_url": "?PageToken=second"}, "next_page_uri": None}, id="trunking-mixed-next"),
    pytest.param("trunk", {"meta": {"next_page_url": None}, "next_page_uri": "?PageToken=second"}, id="trunking-mixed-terminal"),
])
def test_incomplete_pagination_cannot_create_or_complete_a_resumed_step(tmp_path, step, pagination):
    carrier = Carrier()
    requests = []
    key = "credential_lists" if step == "credential_list" else "trunks"

    def handler(request):
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={key: [], **pagination})
        return carrier(request)

    journal = Journal.open(tmp_path, "main", "twilio")
    if step == "trunk":
        journal.start("credential_list", name="ai-phone:main")
        journal.finish("credential_list", credential_list_sid="CL1")
    journal.start(step, name="ai-phone:main")
    before = copy.deepcopy(journal.doc)
    provider = _provider(handler)
    with pytest.raises(ProviderError, match="pagination"):
        if step == "credential_list":
            provider._sip_domain(journal, "main", "aimain1234", "Passw0rdPassw0rdPassw0rd")
        else:
            provider._trunk(journal, "main", "+15550100001")
    assert len(requests) == 1 and requests[0].method == "GET"
    assert carrier.calls == [], "no mutating request may reach the carrier"
    assert journal.doc == before and json.loads(journal.path.read_text()) == before


def test_pagination_has_a_finite_page_limit_before_retry_can_create_anything(tmp_path):
    requests = []

    def handler(request):
        requests.append(request)
        following = request.url.copy_set_param("PageToken", str(len(requests)))
        return httpx.Response(200, json={"credential_lists": [], "next_page_uri": following.raw_path.decode()})

    journal = Journal.open(tmp_path, "main", "twilio")
    journal.start("credential_list", name="ai-phone:main")
    with pytest.raises(ProviderError, match="page limit"):
        _provider(handler)._sip_domain(journal, "main", "aimain1234", "Passw0rdPassw0rdPassw0rd")
    assert len(requests) == 100 and all(request.method == "GET" for request in requests)

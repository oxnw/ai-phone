import copy
import json
import re
from pathlib import Path

import httpx
import pytest

from ai_phone.lines.journal import Journal
from ai_phone.providers.base import Number, ProviderError
from ai_phone.providers.telnyx import SIGNALLING_CIDRS, TELNYX_API, TelnyxProvider


FIXTURES = Path(__file__).parents[1] / "fixtures" / "telnyx"
_METADATA = ("id", "record_type", "created_at", "updated_at")


def contract_fixture(name):
    return json.loads((FIXTURES / name).read_text())["data"]


def contract_fields(record, template):
    """Project explicit documented fields instead of echoing arbitrary request keys."""
    result = {}
    for key, shape in template.items():
        if key in _METADATA or key not in record:
            continue
        value = record[key]
        result[key] = (contract_fields(value, shape) if isinstance(shape, dict) and isinstance(value, dict)
                       else copy.deepcopy(value))
    return result


def contract_response(record, template, resource_id):
    return {**{key: template[key] for key in _METADATA if key in template},
            **contract_fields(record, template), "id": resource_id}


class Carrier:
    """Just enough of api.telnyx.com/v2 to provision, poll, release and verify."""

    def __init__(self, order_status="success", crash_on=None, mask_passwords=False):
        self.profiles, self.connections, self.orders, self.numbers, self.verified = {}, {}, {}, {}, {}
        self.order_status, self.crash_on = order_status, crash_on
        self.calls = []
        self.patches = []  # (number id, body) of every PATCH to a number
        self.page_size = None
        self.mask_passwords = mask_passwords
        self.connection_shape = contract_fixture("credential_connection_create.json")
        self.profile_shape = contract_fixture("outbound_voice_profile.json")

    def _connection_response(self, record, resource_id, *, read=False):
        response = contract_response(record, self.connection_shape, resource_id)
        if read and self.mask_passwords and "password" in response:
            response["password"] = contract_fixture("credential_connection_retrieve_masked.json")["password"]
        return response

    def _list(self, req, rows):
        size = self.page_size or int(req.url.params.get("page[size]", 100))
        page = int(req.url.params.get("page[number]", 1))
        return httpx.Response(200, json={"data": rows[(page - 1) * size:page * size],
                                        "meta": {"page_number": page, "total_pages": (len(rows) + size - 1) // size}})

    def __call__(self, req: httpx.Request) -> httpx.Response:
        self.calls.append((req.method, req.url.path))
        assert req.headers["authorization"] == "Bearer KEY"
        body = json.loads(req.content) if req.content else {}
        p, m = req.url.path, req.method
        if m == "GET" and p == "/v2/available_phone_numbers":
            return httpx.Response(200, json={"data": [{"phone_number": "+442079460000", "cost_information": {"monthly_cost": "1.00", "currency": "GBP"}, "requirements": []}]})
        if m == "POST" and p == "/v2/outbound_voice_profiles":
            # Contract learned from Telnyx's 422 response: tags permit letters, digits, dashes and underscores.
            # Validate requests before storing anything so a permissive echo fake cannot hide this failure.
            tags = body.get("tags", [])
            if not isinstance(tags, list) or any(not isinstance(tag, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", tag) for tag in tags):
                return httpx.Response(422, json={"errors": [{"detail": "Tags may only contain letters, digits, dashes and underscores."}]})
            stored = contract_fields(body, self.profile_shape)
            stored.setdefault("traffic_type", "conversational")
            stored.setdefault("whitelisted_destinations", ["US", "CA"])
            stored.setdefault("enabled", True)
            pid = f"p{len(self.profiles) + 1}"; self.profiles[pid] = stored
            if self.crash_on == "profile":
                return httpx.Response(500, json={"errors": [{"detail": "boom"}]})
            return httpx.Response(201, json={"data": contract_response(stored, self.profile_shape, pid)})
        if m == "GET" and p == "/v2/outbound_voice_profiles":
            return self._list(req, [contract_response(v, self.profile_shape, k) for k, v in self.profiles.items()])
        if m == "POST" and p == "/v2/credential_connections":
            # Unsupported request fields cannot become response fields, including the subdomain
            # setting that belongs to other connection types. Only documented defaults are added.
            stored = contract_fields(body, self.connection_shape)
            stored.setdefault("active", True)
            stored.setdefault("encode_contact_header_enabled", False)
            for direction in ("inbound", "outbound"):
                stored.setdefault(direction, {})
                if isinstance(stored[direction], dict):
                    stored[direction].setdefault("channel_limit", None)
            cid = f"c{len(self.connections) + 1}"; self.connections[cid] = stored
            if self.crash_on == "connection":
                return httpx.Response(500, json={"errors": [{"detail": "boom"}]})
            return httpx.Response(201, json={"data": self._connection_response(stored, cid)})
        if m == "GET" and p == "/v2/credential_connections":
            return self._list(req, [self._connection_response(v, k, read=True) for k, v in self.connections.items()])
        if m == "GET" and p.startswith("/v2/credential_connections/"):
            cid = p.rsplit("/", 1)[1]
            if cid not in self.connections:
                return httpx.Response(404, json={"errors": [{"detail": "Resource not found"}]})
            return httpx.Response(200, json={"data": self._connection_response(self.connections[cid], cid, read=True)})
        if m == "POST" and p == "/v2/number_orders":
            oid = f"o{len(self.orders) + 1}"
            self.orders[oid] = {"status": self.order_status, "phone_numbers": body["phone_numbers"], "connection_id": body.get("connection_id"),
                                "customer_reference": body.get("customer_reference")}
            if self.order_status == "success":
                self.numbers["n1"] = {"phone_number": body["phone_numbers"][0]["phone_number"], "connection_id": body.get("connection_id")}
            if self.crash_on == "order":
                return httpx.Response(500, json={"errors": [{"detail": "boom"}]})
            return httpx.Response(201, json={"data": {"id": oid, **self.orders[oid]}})
        if m == "GET" and p == "/v2/number_orders":
            ref = req.url.params.get("filter[customer_reference]")
            return httpx.Response(200, json={"data": [{"id": k, **v} for k, v in self.orders.items() if v.get("customer_reference") == ref]})
        if m == "GET" and p.startswith("/v2/number_orders/"):
            oid = p.rsplit("/", 1)[1]
            return httpx.Response(200, json={"data": {"id": oid, **self.orders[oid]}})
        if m == "GET" and p == "/v2/phone_numbers":
            want = req.url.params.get("filter[phone_number]")  # no filter: every number on the account
            return httpx.Response(200, json={"data": [{"id": k, **v} for k, v in self.numbers.items() if want is None or v["phone_number"] == want]})
        if m == "PATCH" and p.startswith("/v2/phone_numbers/"):
            nid = p.rsplit("/", 1)[1]
            if nid not in self.numbers:
                return httpx.Response(404, json={"errors": [{"detail": "Resource not found"}]})
            self.patches.append((nid, body))
            self.numbers[nid].update(body)
            return httpx.Response(200, json={"data": {"id": nid, **self.numbers[nid]}})
        if m == "DELETE":
            kind, rid = p.rsplit("/", 2)[1:]
            store = {"phone_numbers": self.numbers, "credential_connections": self.connections, "outbound_voice_profiles": self.profiles}[kind]
            store.pop(rid, None)
            return httpx.Response(200, json={"data": {"id": rid}})
        if m == "POST" and p == "/v2/verified_numbers":
            self.verified[body["phone_number"]] = "pending"
            return httpx.Response(200, json={"data": {"phone_number": body["phone_number"], "verification_method": body["verification_method"]}})
        if m == "POST" and p.endswith("/actions/verify"):
            ok = body["verification_code"] == "123456"
            return httpx.Response(200 if ok else 400, json={"data": {"phone_number": p.split("/")[3]}} if ok else {"errors": [{"detail": "bad code"}]})
        return httpx.Response(404, json={"errors": [{"detail": f"unexpected {m} {p}"}]})


def _provider(carrier):
    # the same base URL production uses, so the /v2 path composition under test is the real one
    return TelnyxProvider(key="KEY", http=httpx.Client(transport=httpx.MockTransport(carrier), base_url=TELNYX_API))


def test_default_client_targets_the_v2_api():
    p = TelnyxProvider(key="KEY")
    assert str(p.http.base_url).rstrip("/") == "https://api.telnyx.com"
    req = p.http.build_request("GET", "/v2/available_phone_numbers")
    assert str(req.url) == "https://api.telnyx.com/v2/available_phone_numbers"
    assert p.http.timeout.read == 20.0


def test_injected_client_still_gets_the_request_timeout():
    seen = {}

    def handler(req):
        seen["timeout"] = req.extensions.get("timeout")
        return httpx.Response(200, json={"data": []})

    p = TelnyxProvider(key="KEY", http=httpx.Client(transport=httpx.MockTransport(handler), base_url=TELNYX_API, timeout=None))
    p.search_numbers("GB", None, 1)
    assert seen["timeout"]["read"] == 20.0


def test_cidrs_are_pinned_and_valid():
    import ipaddress
    assert SIGNALLING_CIDRS and all(ipaddress.ip_network(c) for c in SIGNALLING_CIDRS)


def test_search_and_provision_ready(tmp_path: Path):
    carrier = Carrier()
    p = _provider(carrier)
    nums = p.search_numbers("GB", "20", 5)
    assert nums[0].e164 == "+442079460000" and nums[0].monthly_cost == 1.0 and nums[0].currency == "GBP" and not nums[0].requires_regulatory
    j = Journal.open(tmp_path, "main", "telnyx")
    r = p.provision(j, nums[0], "main", "aimain1234", "Passw0rdPassw0rdPassw0rd")
    assert r.state == "ready"
    assert r.line["registrar"] == "sip:sip.telnyx.com" and r.line["dial_domain"] == "sip.telnyx.com" and r.line["user"] == "aimain1234"
    assert r.line["number"] == "+442079460000" and r.line["caller_id"] == "+442079460000" and r.line["trusted_cidrs"] == SIGNALLING_CIDRS
    assert r.line["provider"]["resources"] == {"outbound_voice_profile_id": "p1", "connection_id": "c1", "order_id": "o1", "number_id": "n1"}
    conn = carrier.connections["c1"]
    assert conn["connection_name"] == "ai-phone:main" and conn["user_name"] == "aimain1234" and conn["outbound"]["outbound_voice_profile_id"] == "p1"
    assert carrier.profiles["p1"]["whitelisted_destinations"] == ["CA", "GB", "US"], "the number's own country must be callable"


def test_pending_order_then_poll(tmp_path: Path):
    carrier = Carrier(order_status="pending")
    p = _provider(carrier)
    j = Journal.open(tmp_path, "main", "telnyx")
    r = p.provision(j, p.search_numbers("GB", None, 1)[0], "main", "aimain1234", "Passw0rdPassw0rdPassw0rd")
    assert r.state == "pending" and j.state == "pending"
    carrier.orders["o1"]["status"] = "success"
    carrier.numbers["n1"] = {"phone_number": "+442079460000", "connection_id": "c1"}
    assert p.poll(Journal.open(tmp_path, "main")).state == "ready"


def test_reconciles_by_name_after_a_crash(tmp_path: Path):
    carrier = Carrier(crash_on="profile")
    p = _provider(carrier)
    j = Journal.open(tmp_path, "main", "telnyx")
    with pytest.raises(ProviderError):
        p.provision(j, p.search_numbers("GB", None, 1)[0], "main", "aimain1234", "Passw0rdPassw0rdPassw0rd")
    assert len(carrier.profiles) == 1   # the carrier created it, the 500 hid the id from us
    carrier.crash_on = None
    r = p.provision(Journal.open(tmp_path, "main"), p.search_numbers("GB", None, 1)[0], "main", "aimain1234", "Passw0rdPassw0rdPassw0rd")
    assert r.state == "ready" and len(carrier.profiles) == 1 and r.line["provider"]["resources"]["outbound_voice_profile_id"] == "p1"


def test_order_crash_is_reconciled_by_customer_reference(tmp_path: Path):
    carrier = Carrier(crash_on="order")
    p = _provider(carrier)
    n = p.search_numbers("GB", None, 1)[0]
    j = Journal.open(tmp_path, "main", "telnyx")
    with pytest.raises(ProviderError):
        p.provision(j, n, "main", "aimain1234", "Passw0rdPassw0rdPassw0rd")
    assert len(carrier.orders) == 1 and carrier.orders["o1"]["customer_reference"] == "ai-phone:main"
    carrier.crash_on = None
    r = p.provision(Journal.open(tmp_path, "main"), n, "main", "aimain1234", "Passw0rdPassw0rdPassw0rd")
    assert r.state == "ready" and len(carrier.orders) == 1 and r.line["provider"]["resources"]["order_id"] == "o1"


def test_success_without_a_listed_number_stays_pending(tmp_path: Path):
    carrier = Carrier()
    p = _provider(carrier)
    n = p.search_numbers("GB", None, 1)[0]
    j = Journal.open(tmp_path, "main", "telnyx")
    r = p.provision(j, n, "main", "aimain1234", "Passw0rdPassw0rdPassw0rd")
    assert r.state == "ready"
    carrier.numbers.clear()  # a fresh journal whose order succeeded but whose number is not visible yet
    j2 = Journal.open(tmp_path, "later", "telnyx")
    j2.start("profile", name="x"); j2.finish("profile", outbound_voice_profile_id="p1")
    j2.start("connection", name="x", user_name="aimain1234"); j2.finish("connection", connection_id="c1")
    j2.start("order", phone_number="+442079460000", customer_reference="ai-phone:later"); j2.finish("order", order_id="o9", order_status="success")
    assert p.poll(j2).state == "pending" and j2.state == "pending"


def test_release_deletes_in_reverse(tmp_path: Path):
    carrier = Carrier()
    p = _provider(carrier)
    j = Journal.open(tmp_path, "main", "telnyx")
    p.provision(j, p.search_numbers("GB", None, 1)[0], "main", "aimain1234", "Passw0rdPassw0rdPassw0rd")
    res = p.release(j)
    assert res.ok and not carrier.numbers and not carrier.connections and not carrier.profiles
    deletes = [c for c in carrier.calls if c[0] == "DELETE"]
    assert [d[1].split("/")[2] for d in deletes] == ["phone_numbers", "credential_connections", "outbound_voice_profiles"]


def test_caller_id_verification():
    p = _provider(Carrier())
    step = p.start_caller_id_verification("+447700900123")
    assert step.kind == "enter_code_in_cli" and "+447700900123" in step.instructions
    assert p.complete_caller_id_verification(step, "999999") is False
    assert p.complete_caller_id_verification(step, "123456") is True


def test_errors_never_leak_the_key_the_password_or_the_body(tmp_path: Path):
    def boom(req):
        if req.url.path == "/v2/available_phone_numbers":
            return httpx.Response(401, json={"errors": [{"detail": "bad key KEY"}]})
        return httpx.Response(500, text="<html>internal error dumping request: password=Passw0rdPassw0rdPassw0rd</html>")
    p = TelnyxProvider(key="KEY", http=httpx.Client(transport=httpx.MockTransport(boom), base_url=TELNYX_API))
    with pytest.raises(ProviderError) as e:
        p.search_numbers("GB", None, 1)
    assert "KEY" not in str(e.value)
    j = Journal.open(tmp_path, "main", "telnyx")
    with pytest.raises(ProviderError) as e2:
        p.provision(j, Number("+442079460000", 1.0, "GBP", False, "x"), "main", "aimain1234", "Passw0rdPassw0rdPassw0rd")
    assert "Passw0rd" not in str(e2.value) and "<html>" not in str(e2.value) and "non-JSON" in str(e2.value)


def _capped(carrier: Carrier) -> TelnyxProvider:
    """Telnyx: 'You may only have 1 Outbound Voice Profile(s) at your account level' (seen on a freshly paid account)."""
    carrier.profiles["p0"] = {"name": "default", "traffic_type": "conversational"}
    inner = carrier.__call__

    def capped(req: httpx.Request) -> httpx.Response:
        if req.method == "POST" and req.url.path == "/v2/outbound_voice_profiles":
            carrier.calls.append((req.method, req.url.path))
            return httpx.Response(403, json={"errors": [{"detail": "You may only have 1 Outbound Voice Profile(s) at your account level."}]})
        return inner(req)

    return TelnyxProvider(key="KEY", http=httpx.Client(transport=httpx.MockTransport(capped), base_url=TELNYX_API))


def test_profile_cap_shares_the_accounts_profile_and_release_keeps_it(tmp_path: Path):
    carrier = Carrier()
    p = _capped(carrier)
    j = Journal.open(tmp_path, "gate", "telnyx")
    r = p.provision(j, p.search_numbers("GB", None, 1)[0], "gate", "aigate1234", "Passw0rdPassw0rdPassw0rd")
    assert r.state == "ready"
    assert r.line["provider"]["resources"]["outbound_voice_profile_id"] == "p0" and "profile_shared" not in r.line["provider"]["resources"]
    assert j.resources()["profile_shared"] is True  # the journal keeps the flag `release` reads
    assert carrier.connections["c1"]["outbound"]["outbound_voice_profile_id"] == "p0"
    rel = p.release(j)
    assert rel.ok and "p0" in carrier.profiles, "the account's own profile must survive `lines remove`"
    assert "c1" not in carrier.connections and "n1" not in carrier.numbers


def test_profile_cap_without_any_profile_is_an_error(tmp_path: Path):
    carrier = Carrier()
    inner = carrier.__call__

    def capped(req: httpx.Request) -> httpx.Response:
        if req.method == "POST" and req.url.path == "/v2/outbound_voice_profiles":
            return httpx.Response(403, json={"errors": [{"detail": "You may only have 1 Outbound Voice Profile(s) at your account level."}]})
        return inner(req)

    p = TelnyxProvider(key="KEY", http=httpx.Client(transport=httpx.MockTransport(capped), base_url=TELNYX_API))
    j = Journal.open(tmp_path, "gate", "telnyx")
    with pytest.raises(ProviderError, match="no outbound voice profile to share"):
        p.provision(j, p.search_numbers("GB", None, 1)[0], "gate", "aigate1234", "Passw0rdPassw0rdPassw0rd")


def test_shared_profile_line_loads(tmp_path: Path, monkeypatch):
    """The config's provider.resources holds strings only: a journal flag copied into it made the line unloadable."""
    from ai_phone.config import load_config
    from ai_phone.lines.config_writer import add_line
    monkeypatch.setenv("AI_PHONE_HOME", str(tmp_path))
    cfg = tmp_path / "config.yaml"
    cfg.write_text("inbound:\n  greeting: Hi\n")
    p = _capped(Carrier())
    r = p.provision(Journal.open(tmp_path / "lines", "gate", "telnyx"), p.search_numbers("GB", None, 1)[0], "gate", "aigate1234",
                    "Passw0rdPassw0rdPassw0rd")
    assert "profile_shared" not in r.line["provider"]["resources"]
    add_line(cfg, "gate", r.line)
    ln = load_config(cfg).lines["gate"]
    assert ln.provider.resources["outbound_voice_profile_id"] == "p0" and ln.number == "+442079460000"


# ---- a number the person already has ------------------------------------------------------------

def _owned(carrier: Carrier, **number) -> Carrier:
    carrier.numbers["n9"] = {"phone_number": "+15550100001", **number}
    return carrier


def test_owned_numbers_lists_the_account_numbers():
    carrier = _owned(Carrier(), connection_id="old-conn")
    p = _provider(carrier)
    assert [(n.e164, n.source_id) for n in p.owned_numbers()] == [("+15550100001", "n9")]
    assert [c[0] for c in carrier.calls] == ["GET"]


def test_attach_connects_an_owned_number_without_ordering(tmp_path: Path):
    carrier = _owned(Carrier(), connection_id="old-conn")
    p = _provider(carrier)
    j = Journal.open(tmp_path, "main", "telnyx")
    r = p.attach(j, Number("+15550100001", None, None, False, "n9"), "main", "aimain1234", "Passw0rdPassw0rdPassw0rd")
    assert ("POST", "/v2/number_orders") not in carrier.calls and not carrier.orders
    assert carrier.numbers["n9"]["connection_id"] == "c1" and carrier.connections["c1"]["connection_name"] == "ai-phone:main"
    assert carrier.patches == [("n9", {"connection_id": "c1"})]
    assert r.state == "ready" and j.state == "ready" and r.line["number"] == "+15550100001" and r.line["user"] == "aimain1234"
    assert r.line["provider"]["resources"] == {"outbound_voice_profile_id": "p1", "connection_id": "c1", "brought_number_id": "n9"}
    res = j.resources()
    assert res["number_brought"] is True and res["brought_number_id"] == "n9" and "number_id" not in res
    assert res["previous_connection_id"] == "old-conn"
    assert carrier.profiles["p1"]["whitelisted_destinations"] == ["CA", "US"]
    assert [s["name"] for s in j.doc["steps"]] == ["claim", "profile", "connection", "route"]
    assert p.poll(j).line["number"] == "+15550100001"


def test_attach_refuses_a_number_not_on_the_account(tmp_path: Path):
    carrier = Carrier()
    p = _provider(carrier)
    with pytest.raises(ProviderError, match="not on this Telnyx account"):
        p.attach(Journal.open(tmp_path, "main", "telnyx"), Number("+15550100001", None, None, False, "n9"), "main", "aimain1234", "pw")
    assert carrier.profiles == {} and carrier.connections == {}


def test_attach_resumes_after_a_lost_route_response(tmp_path: Path):
    carrier = _owned(Carrier(), connection_id="old-conn")
    lost = {"armed": True}

    def handler(req):
        r = carrier(req)
        if lost["armed"] and req.method == "PATCH":
            lost["armed"] = False
            raise httpx.ConnectError("lost")
        return r

    p = TelnyxProvider(key="KEY", http=httpx.Client(transport=httpx.MockTransport(handler), base_url=TELNYX_API))
    n = Number("+15550100001", None, None, False, "n9")
    with pytest.raises(ProviderError):
        p.attach(Journal.open(tmp_path, "main", "telnyx"), n, "main", "aimain1234", "pw")
    j = Journal.open(tmp_path, "main")
    r = p.attach(j, n, "main", "aimain1234", "pw")
    assert r.state == "ready" and len(carrier.profiles) == 1 and len(carrier.connections) == 1
    assert j.resources()["previous_connection_id"] == "old-conn"  # the claim was not re-read after the move


def test_release_of_a_brought_number_reconnects_it_and_keeps_it(tmp_path: Path):
    carrier = _owned(Carrier(), connection_id="old-conn")
    p = _provider(carrier)
    j = Journal.open(tmp_path, "main", "telnyx")
    p.attach(j, Number("+15550100001", None, None, False, "n9"), "main", "aimain1234", "pw")
    before = len(carrier.calls)
    assert p.release(j).ok
    assert ("DELETE", "/v2/phone_numbers/n9") not in carrier.calls and "n9" in carrier.numbers
    assert carrier.numbers["n9"]["connection_id"] == "old-conn"
    assert not carrier.connections and not carrier.profiles
    order = [c for c in carrier.calls[before:] if c[0] in ("PATCH", "DELETE")]
    assert order == [("PATCH", "/v2/phone_numbers/n9"), ("DELETE", "/v2/credential_connections/c1"),
                     ("DELETE", "/v2/outbound_voice_profiles/p1")], "the number leaves the connection before it is deleted"


def test_release_of_an_old_format_brought_number_keeps_it(tmp_path: Path):
    """What e4a8c67 wrote: the brought number under `number_id`, flagged `number_brought`."""
    carrier = _owned(Carrier(), connection_id="old-conn")
    p = _provider(carrier)
    j = Journal.open(tmp_path, "main", "telnyx")
    p.attach(j, Number("+15550100001", None, None, False, "n9"), "main", "aimain1234", "pw")
    for s in j.doc["steps"]:
        if "brought_number_id" in s["result"]:
            s["result"]["number_id"] = s["result"].pop("brought_number_id")
    j._save()
    assert j.resources()["number_id"] == "n9" and j.resources()["number_brought"] is True
    assert p.release(j).ok
    assert ("DELETE", "/v2/phone_numbers/n9") not in carrier.calls and "n9" in carrier.numbers
    assert carrier.numbers["n9"]["connection_id"] == "old-conn"
    assert not carrier.connections and not carrier.profiles


def test_release_of_a_brought_number_with_no_previous_connection_sends_null(tmp_path: Path):
    carrier = _owned(Carrier())
    p = _provider(carrier)
    j = Journal.open(tmp_path, "main", "telnyx")
    p.attach(j, Number("+15550100001", None, None, False, "n9"), "main", "aimain1234", "pw")
    assert j.resources()["previous_connection_id"] == ""
    assert p.release(j).ok
    assert carrier.patches[-1] == ("n9", {"connection_id": None}) and "n9" in carrier.numbers


def test_release_of_a_brought_number_that_cannot_be_moved_still_deletes_the_rest(tmp_path: Path):
    carrier = _owned(Carrier(), connection_id="old-conn")
    p = _provider(carrier)
    j = Journal.open(tmp_path, "main", "telnyx")
    p.attach(j, Number("+15550100001", None, None, False, "n9"), "main", "aimain1234", "pw")
    del carrier.numbers["n9"]  # the move back is refused (404)
    res = p.release(j)
    assert not res.ok and "stays on your account" in res.leftover["brought_number_id"] and j.state == "rollback_failed"
    assert not [c for c in carrier.calls if c == ("DELETE", "/v2/phone_numbers/n9")]
    assert not carrier.connections and not carrier.profiles


def test_owned_numbers_carry_their_country():
    from ai_phone.providers.telnyx import _country_of
    carrier = _owned(Carrier())
    carrier.numbers["n8"] = {"phone_number": "+442079460001"}
    carrier.numbers["n7"] = {"phone_number": "+15550100002", "country_iso_alpha2": "CA"}  # the record's own country wins
    assert [(n.e164, n.country) for n in _provider(carrier).owned_numbers()] == [
        ("+15550100001", "US"), ("+442079460001", "GB"), ("+15550100002", "CA")]
    assert _country_of("+447700900123") == "GB" and _country_of("+9990000000") == ""  # +999 is not assigned to anyone


def test_attaching_a_uk_number_whitelists_gb(tmp_path: Path):
    """A brought number must be able to call its own country, as a bought one can."""
    carrier = Carrier()
    carrier.numbers["n9"] = {"phone_number": "+442079460001", "connection_id": "old-conn"}
    p = _provider(carrier)
    r = p.attach(Journal.open(tmp_path, "main", "telnyx"), Number("+442079460001", None, None, False, "n9"), "main", "aimain1234", "pw")
    assert r.state == "ready" and carrier.profiles["p1"]["whitelisted_destinations"] == ["CA", "GB", "US"]


def test_resuming_a_uk_purchase_still_whitelists_gb(tmp_path: Path):
    """`lines resume` passes a number without a country: the profile still allows the number's own country."""
    carrier = Carrier()
    p = _provider(carrier)
    r = p.provision(Journal.open(tmp_path, "main", "telnyx"), Number("+442079460000", None, None, False, "+442079460000"),
                    "main", "aimain1234", "pw")
    assert r.state == "ready" and carrier.profiles["p1"]["whitelisted_destinations"] == ["CA", "GB", "US"]


def test_a_buy_then_a_bring_in_one_journal_releases_the_bought_number(tmp_path: Path):
    """A journal shared by a purchase whose line was never written and a later attach: the bought number is still
    deleted, the brought one moved back and kept."""
    carrier = _owned(Carrier(), connection_id="old-conn")
    p = _provider(carrier)
    j = Journal.open(tmp_path, "main", "telnyx")
    assert p.provision(j, p.search_numbers("GB", None, 1)[0], "main", "aimain1234", "pw").state == "ready"
    assert p.attach(j, Number("+15550100001", None, None, False, "n9"), "main", "aimain1234", "pw").state == "ready"
    res = j.resources()
    assert res["number_id"] == "n1" and res["brought_number_id"] == "n9"
    assert p.release(j).ok
    assert list(carrier.numbers) == ["n9"] and carrier.numbers["n9"]["connection_id"] == "old-conn"
    assert not carrier.connections and not carrier.profiles


def test_a_bring_then_a_buy_in_one_journal_restores_the_brought_number(tmp_path: Path):
    carrier = _owned(Carrier(), connection_id="old-conn")
    refused = {"done": False}

    def handler(req):
        if not refused["done"] and req.method == "PATCH":
            refused["done"] = True
            return httpx.Response(500, json={"errors": [{"detail": "try again"}]})
        return carrier(req)

    p = TelnyxProvider(key="KEY", http=httpx.Client(transport=httpx.MockTransport(handler), base_url=TELNYX_API))
    j = Journal.open(tmp_path, "main", "telnyx")
    with pytest.raises(ProviderError):
        p.attach(j, Number("+15550100001", None, None, False, "n9"), "main", "aimain1234", "pw")
    r = p.provision(j, p.search_numbers("GB", None, 1)[0], "main", "aimain1234", "pw")
    assert r.state == "ready" and r.line["number"] == "+442079460000" and r.line["provider"]["resources"]["number_id"] == "n1"
    assert p.release(j).ok
    assert list(carrier.numbers) == ["n9"], "the bought number is released"
    assert carrier.numbers["n9"]["connection_id"] == "old-conn" and not carrier.connections and not carrier.profiles


# ---- reconciliation must establish identity, not just a matching name ----------------------------

SIP_USER = "aimain1234"
SIP_PASSWORD = "LocalSyntheticPassword123"


def _connection_record(**fields):
    return {"connection_name": "ai-phone:main", "user_name": SIP_USER, "password": SIP_PASSWORD, "active": True,
            "sip_uri_calling_preference": "disabled", "encode_contact_header_enabled": True,
            "inbound": {"channel_limit": 1},
            "outbound": {"outbound_voice_profile_id": "p1", "channel_limit": 1}, **fields}


def _connection_attempt(tmp_path, carrier, *, completed=False):
    carrier.profiles["p1"] = {"name": "ai-phone:main", "traffic_type": "conversational"}
    j = Journal.open(tmp_path, "main", "telnyx")
    j.start("profile", name="ai-phone:main")
    j.finish("profile", outbound_voice_profile_id="p1")
    j.start("connection", name="ai-phone:main", user_name=SIP_USER)
    if completed:
        j.finish("connection", connection_id="c1")
    return j


def _resume_connection(p, j):
    return p.provision(j, Number("+15555550100", None, None, False, "n1"), "main", SIP_USER, SIP_PASSWORD)


def test_lost_connection_response_recovers_matching_attempt_without_another_post(tmp_path):
    carrier = _owned(Carrier(crash_on="connection"))
    p = _provider(carrier)
    j = Journal.open(tmp_path, "main", "telnyx")
    n = Number("+15550100001", None, None, False, "n9")
    with pytest.raises(ProviderError):
        p.attach(j, n, "main", SIP_USER, SIP_PASSWORD)
    assert "connection_id" not in j.resources() and not carrier.patches
    carrier.crash_on = None
    assert p.attach(Journal.open(tmp_path, "main"), n, "main", SIP_USER, SIP_PASSWORD).state == "ready"
    assert carrier.calls.count(("POST", "/v2/credential_connections")) == 1
    assert len(carrier.connections) == 1


@pytest.mark.parametrize("changes", [
    {"user_name": "anotheruser"}, {"password": "OtherSyntheticPassword123"},
    {"active": False}, {"active": 1}, {"sip_uri_calling_preference": "unrestricted"},
    {"encode_contact_header_enabled": False},
    {"inbound": {"channel_limit": 2}}, {"inbound": {}},
    {"sip_uri_calling_preference": None},
    {"outbound": {"outbound_voice_profile_id": "other-profile", "channel_limit": 1}},
    {"outbound": {"outbound_voice_profile_id": "p1", "channel_limit": 2}},
])
def test_uncertain_connection_rejects_changed_credentials_or_settings_without_mutation(tmp_path, changes):
    carrier = Carrier()
    j = _connection_attempt(tmp_path, carrier)
    carrier.connections["c1"] = _connection_record(**changes)
    before = copy.deepcopy(carrier.connections)
    with pytest.raises(ProviderError, match="does not match.*saved SIP credentials or settings") as error:
        _resume_connection(_provider(carrier), j)
    assert all(method == "GET" for method, _ in carrier.calls)
    assert carrier.connections == before and "connection_id" not in j.resources()
    assert SIP_PASSWORD not in str(error.value) and "OtherSyntheticPassword123" not in str(error.value)
    assert SIP_USER not in str(error.value) and "anotheruser" not in str(error.value)
    assert SIP_PASSWORD not in j.path.read_text()


@pytest.mark.parametrize("same_credentials", [False, True])
def test_fresh_setup_never_adopts_an_existing_connection_by_name(tmp_path, same_credentials):
    carrier = Carrier()
    carrier.connections["foreign"] = _connection_record(**({} if same_credentials else {"user_name": "anotheruser"}))
    j = Journal.open(tmp_path, "main", "telnyx")
    with pytest.raises(ProviderError, match="already uses this setup name"):
        _resume_connection(_provider(carrier), j)
    assert all(method == "GET" for method, _ in carrier.calls)
    assert not carrier.profiles and not j.resources() and len(carrier.connections) == 1


@pytest.mark.parametrize("page_size", [None, 1])
def test_duplicate_connection_candidates_are_refused_even_across_pages(tmp_path, page_size):
    carrier = Carrier()
    carrier.page_size = page_size
    j = _connection_attempt(tmp_path, carrier)
    carrier.connections = {"c1": _connection_record(), "c2": _connection_record()}
    with pytest.raises(ProviderError, match="Multiple Telnyx resources"):
        _resume_connection(_provider(carrier), j)
    assert "connection_id" not in j.resources() and all(method == "GET" for method, _ in carrier.calls)
    if page_size == 1:
        assert carrier.calls.count(("GET", "/v2/credential_connections")) == 2


@pytest.mark.parametrize("matching", [True, False])
def test_duplicate_422_only_reconciles_the_matching_attempt(tmp_path, matching):
    carrier = _owned(Carrier())
    def handler(req):
        if req.method == "POST" and req.url.path == "/v2/credential_connections":
            carrier.calls.append((req.method, req.url.path))
            body = json.loads(req.content)
            carrier.connections["c1"] = body if matching else {**body, "password": "OtherSyntheticPassword123"}
            return httpx.Response(422, json={"errors": [{"detail": "duplicate connection name"}]})
        return carrier(req)
    p = _provider(handler)
    j = Journal.open(tmp_path, "main", "telnyx")
    n = Number("+15550100001", None, None, False, "n9")
    if matching:
        assert p.attach(j, n, "main", SIP_USER, SIP_PASSWORD).state == "ready"
        assert j.resources()["connection_id"] == "c1" and len(carrier.patches) == 1
    else:
        with pytest.raises(ProviderError, match="does not match"):
            p.attach(j, n, "main", SIP_USER, SIP_PASSWORD)
        assert "connection_id" not in j.resources() and not carrier.patches
    assert carrier.calls.count(("POST", "/v2/credential_connections")) == 1
    assert not [call for call in carrier.calls if call[0] in ("PATCH", "DELETE") and "credential_connections" in call[1]]


@pytest.mark.parametrize("attach", [True, False])
@pytest.mark.parametrize("changes", [{"password": "OtherSyntheticPassword123"}, {"user_name": "anotheruser"}])
def test_completed_legacy_connection_is_rechecked_before_routing_or_reporting_ready(tmp_path, attach, changes):
    carrier = _owned(Carrier())
    j = _connection_attempt(tmp_path, carrier, completed=True)
    carrier.connections["c1"] = _connection_record(**changes)
    p = _provider(carrier)
    with pytest.raises(ProviderError, match="does not match"):
        if attach:
            p.attach(j, Number("+15550100001", None, None, False, "n9"), "main", SIP_USER, SIP_PASSWORD)
        else:
            _resume_connection(p, j)
    assert ("GET", "/v2/credential_connections/c1") in carrier.calls
    assert all(method == "GET" for method, _ in carrier.calls)
    assert not carrier.patches and not carrier.orders and j.state != "ready"


def test_completed_connection_missing_at_carrier_is_not_recreated(tmp_path):
    carrier = Carrier()
    j = _connection_attempt(tmp_path, carrier, completed=True)
    with pytest.raises(ProviderError, match="no longer exists"):
        _resume_connection(_provider(carrier), j)
    assert all(method == "GET" for method, _ in carrier.calls) and not carrier.connections


def test_completed_connection_can_resume_with_unchanged_credentials(tmp_path):
    carrier = _owned(Carrier())
    j = _connection_attempt(tmp_path, carrier, completed=True)
    carrier.connections["c1"] = _connection_record()
    assert _provider(carrier).attach(j, Number("+15550100001", None, None, False, "n9"),
                                     "main", SIP_USER, SIP_PASSWORD).state == "ready"
    assert not [call for call in carrier.calls if call[0] == "POST"]


@pytest.mark.parametrize("legacy", [False, True])
def test_same_name_foreign_profile_is_not_claimed_for_cleanup(tmp_path, legacy):
    carrier = Carrier()
    carrier.profiles["foreign"] = {"name": "ai-phone:main", "traffic_type": "conversational",
                                   "whitelisted_destinations": ["CA", "US"], "tags": ["another-setup"]}
    j = Journal.open(tmp_path, "main", "telnyx")
    if legacy:
        j.start("profile", name="ai-phone:main")
    with pytest.raises(ProviderError, match="cannot be identified"):
        _resume_connection(_provider(carrier), j)
    assert "outbound_voice_profile_id" not in j.resources() and "connection_id" not in j.resources()
    assert all(method == "GET" for method, _ in carrier.calls)
    assert _provider(carrier).release(j).ok and "foreign" in carrier.profiles


def test_lost_profile_response_has_a_persisted_recovery_tag_before_the_post(tmp_path):
    carrier = Carrier(crash_on="profile")
    def handler(req):
        if req.method == "POST" and req.url.path == "/v2/outbound_voice_profiles":
            planned = next(s["planned"] for s in Journal.open(tmp_path, "main").doc["steps"] if s["name"] == "profile")
            assert json.loads(req.content)["tags"] == [planned["recovery_tag"]]
        return carrier(req)
    p = _provider(handler)
    j = Journal.open(tmp_path, "main", "telnyx")
    with pytest.raises(ProviderError):
        _resume_connection(p, j)
    tag = carrier.profiles["p1"]["tags"][0]
    carrier.crash_on = None
    carrier.numbers["n9"] = {"phone_number": "+15555550100"}
    assert p.attach(Journal.open(tmp_path, "main"), Number("+15555550100", None, None, False, "n9"),
                    "main", SIP_USER, SIP_PASSWORD).state == "ready"
    assert carrier.profiles["p1"]["tags"] == [tag]
    assert carrier.calls.count(("POST", "/v2/outbound_voice_profiles")) == 1


def test_profile_recovery_rejects_changed_request_settings(tmp_path):
    carrier = Carrier(crash_on="profile")
    p = _provider(carrier)
    j = Journal.open(tmp_path, "main", "telnyx")
    with pytest.raises(ProviderError):
        _resume_connection(p, j)
    carrier.profiles["p1"]["whitelisted_destinations"] = ["US"]
    carrier.calls.clear()
    with pytest.raises(ProviderError, match="cannot be identified"):
        _resume_connection(p, Journal.open(tmp_path, "main"))
    assert all(method == "GET" for method, _ in carrier.calls)
    assert "outbound_voice_profile_id" not in j.resources()


def test_unverifiable_search_pagination_cannot_adopt_or_create(tmp_path):
    def handler(req):
        assert req.method == "GET"
        return httpx.Response(200, json={"data": [{"id": "c1", **_connection_record()}]})
    with pytest.raises(ProviderError, match="search could not be verified"):
        _resume_connection(_provider(handler), Journal.open(tmp_path, "main", "telnyx"))


def test_lost_connection_transport_response_is_recovered_without_creating_again(tmp_path):
    carrier = _owned(Carrier())
    lost = True
    def handler(req):
        nonlocal lost
        response = carrier(req)
        if lost and req.method == "POST" and req.url.path == "/v2/credential_connections":
            lost = False
            raise httpx.ReadTimeout("lost response")
        return response
    p = _provider(handler)
    j = Journal.open(tmp_path, "main", "telnyx")
    number = Number("+15550100001", None, None, False, "n9")
    with pytest.raises(ProviderError, match="ReadTimeout"):
        p.attach(j, number, "main", SIP_USER, SIP_PASSWORD)
    assert p.attach(Journal.open(tmp_path, "main"), number, "main", SIP_USER, SIP_PASSWORD).state == "ready"
    assert carrier.calls.count(("POST", "/v2/credential_connections")) == 1


def test_duplicate_422_without_a_matching_connection_remains_an_error(tmp_path):
    carrier = Carrier()
    def handler(req):
        if req.method == "POST" and req.url.path == "/v2/credential_connections":
            carrier.calls.append((req.method, req.url.path))
            return httpx.Response(422, json={"errors": [{"detail": "invalid connection"}]})
        return carrier(req)
    j = Journal.open(tmp_path, "main", "telnyx")
    with pytest.raises(ProviderError, match="no matching setup was found"):
        _resume_connection(_provider(handler), j)
    assert "connection_id" not in j.resources() and not carrier.orders and not carrier.patches
    assert carrier.calls.count(("POST", "/v2/credential_connections")) == 1


def test_credentials_read_from_carrier_are_redacted_from_later_errors(tmp_path):
    carrier = Carrier()
    j = _connection_attempt(tmp_path, carrier)
    remote_password = "ForeignSyntheticPassword987"
    carrier.connections["c1"] = _connection_record(password=remote_password)
    def handler(req):
        if req.url.path == "/v2/credential_connections/c1":
            return httpx.Response(500, json={"errors": [{"detail": f"failed for {remote_password}"}]})
        return carrier(req)
    with pytest.raises(ProviderError) as error:
        _resume_connection(_provider(handler), j)
    assert remote_password not in str(error.value) and SIP_PASSWORD not in str(error.value)
    assert "***" in str(error.value)


def test_conflicting_journal_identity_is_refused_before_carrier_requests(tmp_path):
    carrier = Carrier()
    j = _connection_attempt(tmp_path, carrier)
    j.doc["steps"][-1]["planned"]["user_name"] = "anotheruser"
    with pytest.raises(ProviderError, match="journal disagrees"):
        _resume_connection(_provider(carrier), j)
    assert not carrier.calls


@pytest.mark.parametrize("name", ["ai-phone:other", "not-an-ai-phone-name", "ai-phone:../outside"])
def test_recorded_profile_and_connection_names_must_remain_consistent(tmp_path, name):
    carrier = Carrier()
    j = _connection_attempt(tmp_path, carrier, completed=True)
    j.doc["steps"][0]["planned"]["name"] = name
    with pytest.raises(ProviderError, match="journal disagrees"):
        _resume_connection(_provider(carrier), j)
    assert not carrier.calls


def test_lost_profile_response_accepts_carrier_normalized_destination_order(tmp_path):
    carrier = _owned(Carrier(crash_on="profile"))
    p = _provider(carrier)
    j = Journal.open(tmp_path, "main", "telnyx")
    number = Number("+15550100001", None, None, False, "n9")
    with pytest.raises(ProviderError):
        p.attach(j, number, "main", SIP_USER, SIP_PASSWORD)
    carrier.profiles["p1"]["whitelisted_destinations"].reverse()
    carrier.crash_on = None
    assert p.attach(Journal.open(tmp_path, "main"), number, "main", SIP_USER, SIP_PASSWORD).state == "ready"
    assert carrier.calls.count(("POST", "/v2/outbound_voice_profiles")) == 1


@pytest.mark.parametrize("field", ["tags", "whitelisted_destinations"])
def test_profile_set_fields_require_identical_members_but_ignore_order(field):
    expected = {field: ["first", "second"]}
    assert TelnyxProvider._matches({field: ["second", "first"]}, expected)
    for actual in (["first"], ["first", "second", "extra"], None, "first", ["first", {}]):
        assert not TelnyxProvider._matches({field: actual}, expected)


# Recovery tags are constrained by the carrier, not arbitrary customer-reference strings.

OLD_PROFILE_TAG = "ai-phone:" + "a" * 32


def _rejected_profile_attempt(tmp_path):
    j = Journal.open(tmp_path, "main", "telnyx")
    j.start("claim", phone_number="+15550100001")
    j.finish("claim", brought_number_id="n9", number_brought=True, previous_connection_id="old-conn")
    j.start("profile", name="ai-phone:main", recovery_tag=OLD_PROFILE_TAG)
    return j


def _resume_rejected_profile(provider, journal):
    return provider.attach(journal, Number("+15550100001", None, None, False, "n9"), "main", SIP_USER, SIP_PASSWORD)


@pytest.mark.parametrize("tag", [OLD_PROFILE_TAG, "tag with spaces", "tag/slash", "tag.ending", "", None])
def test_profile_request_validation_rejects_invalid_tags_before_resource_creation(tag):
    carrier = Carrier()
    with pytest.raises(ProviderError) as error:
        _provider(carrier)._call("POST", "/outbound_voice_profiles", json={"name": "ai-phone:main", "tags": [tag]})
    assert error.value.status == 422 and not carrier.profiles


def test_new_profile_tag_meets_carrier_validation_and_is_saved_before_request(tmp_path):
    carrier = _owned(Carrier())
    def handler(req):
        if req.method == "POST" and req.url.path == "/v2/outbound_voice_profiles":
            tag, = json.loads(req.content)["tags"]
            assert re.fullmatch(r"ai-phone-[0-9a-f]{32}", tag)
            assert Journal.open(tmp_path, "main")._step("profile")["planned"]["recovery_tag"] == tag
        return carrier(req)
    assert _resume_rejected_profile(_provider(handler), Journal.open(tmp_path, "main", "telnyx")).state == "ready"
    assert len(carrier.profiles) == len(carrier.connections) == 1


@pytest.mark.parametrize("lost_response", [False, True])
def test_rejected_old_tag_retries_only_after_empty_reads_and_persists_replacement(tmp_path, lost_response):
    carrier = _owned(Carrier(crash_on="profile" if lost_response else None), connection_id="old-conn")
    j = _rejected_profile_attempt(tmp_path)
    def handler(req):
        if req.method == "POST" and req.url.path == "/v2/outbound_voice_profiles":
            assert carrier.calls == [("GET", "/v2/credential_connections"), ("GET", "/v2/outbound_voice_profiles")]
            tag, = json.loads(req.content)["tags"]
            assert tag == OLD_PROFILE_TAG.replace(":", "-")
            assert Journal.open(tmp_path, "main")._step("profile")["planned"]["recovery_tag"] == tag
        return carrier(req)
    provider = _provider(handler)
    if lost_response:
        with pytest.raises(ProviderError):
            _resume_rejected_profile(provider, j)
        assert not carrier.patches
        carrier.crash_on = None
        j = Journal.open(tmp_path, "main")
    assert _resume_rejected_profile(provider, j).state == "ready"
    assert carrier.calls.count(("POST", "/v2/outbound_voice_profiles")) == 1
    assert carrier.calls.count(("POST", "/v2/credential_connections")) == 1
    assert len(carrier.profiles) == len(carrier.connections) == 1


@pytest.mark.parametrize("resource", ["profile", "connection", "duplicate_profiles"])
def test_old_tag_is_not_rewritten_when_carrier_resources_exist(tmp_path, resource):
    carrier = _owned(Carrier())
    j = _rejected_profile_attempt(tmp_path)
    if resource == "connection":
        carrier.connections["c1"] = _connection_record()
    else:
        profile = {"name": "ai-phone:main", "traffic_type": "conversational", "whitelisted_destinations": ["CA", "US"],
                   "tags": ["foreign-tag"]}
        carrier.profiles["foreign"] = profile
        if resource == "duplicate_profiles":
            carrier.profiles["other"] = dict(profile)
            carrier.page_size = 1
    before = j.path.read_bytes()
    with pytest.raises(ProviderError):
        _resume_rejected_profile(_provider(carrier), j)
    assert j.path.read_bytes() == before
    assert all(method == "GET" for method, _ in carrier.calls)
    assert not carrier.patches


def test_existing_matching_profile_keeps_old_tag_and_is_not_recreated(tmp_path):
    carrier = _owned(Carrier())
    j = _rejected_profile_attempt(tmp_path)
    carrier.profiles["p1"] = {"name": "ai-phone:main", "traffic_type": "conversational",
                              "whitelisted_destinations": ["CA", "US"], "tags": [OLD_PROFILE_TAG]}
    assert _resume_rejected_profile(_provider(carrier), j).state == "ready"
    assert j._step("profile")["planned"]["recovery_tag"] == OLD_PROFILE_TAG
    assert carrier.profiles["p1"]["tags"] == [OLD_PROFILE_TAG]
    assert ("POST", "/v2/outbound_voice_profiles") not in carrier.calls


@pytest.mark.parametrize("state", ["connection_started", "later_step", "partial_result", "unknown_tag"])
def test_old_tag_migration_refuses_evidence_of_later_work_or_unrecognized_metadata(tmp_path, state):
    carrier = _owned(Carrier())
    j = _rejected_profile_attempt(tmp_path)
    if state == "connection_started":
        j.start("connection", name="ai-phone:main", user_name=SIP_USER)
    elif state == "later_step":
        j.start("route", brought_number_id="n9", connection_id="c1")
    elif state == "partial_result":
        j._step("profile")["result"] = {"outbound_voice_profile_id": "p1"}
        j._save()
    else:
        j._step("profile")["planned"]["recovery_tag"] = "unknown:tag"
        j._save()
    before = j.path.read_bytes()
    with pytest.raises(ProviderError, match="cannot be safely retried automatically"):
        _resume_rejected_profile(_provider(carrier), j)
    assert j.path.read_bytes() == before
    assert all(method == "GET" for method, _ in carrier.calls)


def test_old_tag_migration_requires_a_verifiable_empty_profile_search(tmp_path):
    carrier = _owned(Carrier())
    j = _rejected_profile_attempt(tmp_path)
    before = j.path.read_bytes()
    def handler(req):
        if req.url.path == "/v2/outbound_voice_profiles":
            assert req.method == "GET"
            return httpx.Response(200, json={"data": []})  # missing pagination is not proof of an empty search
        return carrier(req)
    with pytest.raises(ProviderError, match="could not be verified"):
        _resume_rejected_profile(_provider(handler), j)
    assert j.path.read_bytes() == before and not carrier.profiles and not carrier.connections


# Credential connections do not expose the subdomain controls used by IP/FQDN connections.


def test_credential_api_omits_unsupported_subdomain_field_in_create_and_detail_responses():
    carrier = Carrier()
    provider = _provider(carrier)
    request = _connection_record(inbound={"channel_limit": 1, "sip_subdomain_receive_settings": "only_my_connections"})
    created = provider._call("POST", "/credential_connections", json=request)
    detail = provider._call("GET", "/credential_connections/c1")
    assert created["inbound"] == detail["inbound"] == {"channel_limit": 1}
    assert created["sip_uri_calling_preference"] == detail["sip_uri_calling_preference"] == "disabled"
    assert request["inbound"]["sip_subdomain_receive_settings"] == "only_my_connections"


def test_fresh_credential_connection_succeeds_without_subdomain_response_field(tmp_path):
    carrier = _owned(Carrier(), connection_id="previous-route")
    sent = []

    def handler(req):
        if req.method == "POST" and req.url.path == "/v2/credential_connections":
            body = json.loads(req.content)
            sent.append(body)
            assert "sip_subdomain_receive_settings" not in body["inbound"]
            assert body["sip_uri_calling_preference"] == "disabled"
        return carrier(req)

    saved = Journal.open(tmp_path, "main", "telnyx")
    result = _provider(handler).attach(saved, Number("+15550100001", None, None, False, "n9"),
                                       "main", SIP_USER, SIP_PASSWORD)
    assert result.state == saved.state == "ready"
    assert len(sent) == len(carrier.connections) == len(carrier.profiles) == 1
    assert carrier.connections["c1"]["inbound"] == {"channel_limit": 1}
    assert carrier.patches == [("n9", {"connection_id": "c1"})]
    assert not carrier.orders


@pytest.mark.parametrize("completed", [False, True])
def test_existing_uk_brought_connection_without_subdomain_field_retries_without_recreating(tmp_path, completed):
    number = Number("+442079460000", None, None, False, "owned-uk", country="GB")
    carrier = Carrier()
    carrier.numbers["owned-uk"] = {"phone_number": number.e164, "connection_id": "previous-route"}
    carrier.profiles["p1"] = {"name": "ai-phone:main", "traffic_type": "conversational",
                              "whitelisted_destinations": ["CA", "GB", "US"]}
    carrier.connections["c1"] = _connection_record()
    connection_before = copy.deepcopy(carrier.connections["c1"])
    profile_before = copy.deepcopy(carrier.profiles["p1"])
    saved = Journal.open(tmp_path, "main", "telnyx")
    saved.start("claim", phone_number=number.e164)
    saved.finish("claim", brought_number_id="owned-uk", number_brought=True, previous_connection_id="previous-route")
    saved.start("profile", name="ai-phone:main")
    saved.finish("profile", outbound_voice_profile_id="p1")
    saved.start("connection", name="ai-phone:main", user_name=SIP_USER)
    if completed:
        saved.finish("connection", connection_id="c1")

    # A started step is what the former false POST-response mismatch left behind.
    result = _provider(carrier).attach(saved, number, "main", SIP_USER, SIP_PASSWORD)
    assert result.state == saved.state == "ready"
    assert result.line["number"] == number.e164 and result.line["user"] == SIP_USER
    assert saved.resources()["connection_id"] == "c1"
    assert saved.resources()["previous_connection_id"] == "previous-route"
    assert carrier.connections == {"c1": connection_before}
    assert carrier.profiles == {"p1": profile_before}
    assert ("GET", "/v2/credential_connections/c1") in carrier.calls
    assert [(method, path) for method, path in carrier.calls if method != "GET"] == [
        ("PATCH", "/v2/phone_numbers/owned-uk")]
    assert carrier.patches == [("owned-uk", {"connection_id": "c1"})]
    assert not carrier.orders


# The fixture responses come from documented shapes, with synthetic values, rather than request echo.


def test_documented_create_response_verifies_with_extra_read_only_fields():
    response = contract_fixture("credential_connection_create.json")
    provider = _provider(Carrier())
    assert provider._connection(response, _connection_record(), read=False) == response
    assert response["record_type"] == "credential_connection"


def test_mock_uses_supported_fields_and_documented_defaults_only():
    carrier = Carrier()
    provider = _provider(carrier)
    created = provider._call("POST", "/credential_connections", json={
        "connection_name": "ai-phone:main", "user_name": SIP_USER, "password": SIP_PASSWORD,
        "unexpected_request_field": "must-not-echo", "inbound": {"sip_subdomain_receive_settings": "from_anyone"},
        "outbound": {"unexpected_nested_field": "must-not-echo"},
    })
    assert "unexpected_request_field" not in created
    assert created["inbound"] == {"channel_limit": None}
    assert created["outbound"] == {"channel_limit": None}
    assert created["active"] is True and created["encode_contact_header_enabled"] is False
    assert "sip_uri_calling_preference" not in created  # no documented default is invented
    profile = provider._call("POST", "/outbound_voice_profiles", json={"name": "ai-phone:main", "arbitrary": "ignored"})
    assert "arbitrary" not in profile
    assert profile["traffic_type"] == "conversational" and profile["enabled"] is True
    assert profile["whitelisted_destinations"] == ["US", "CA"]


@pytest.mark.parametrize("completed", [False, True])
def test_masked_account_can_create_but_cannot_safely_reconcile_an_existing_password(tmp_path, completed):
    carrier = _owned(Carrier(mask_passwords=True, crash_on=None if completed else "connection"), connection_id="previous-route")
    provider = _provider(carrier)
    saved = Journal.open(tmp_path, "main", "telnyx")
    number = Number("+15550100001", None, None, False, "n9")
    if completed:
        assert provider.attach(saved, number, "main", SIP_USER, SIP_PASSWORD).state == "ready"
    else:
        with pytest.raises(ProviderError):
            provider.attach(saved, number, "main", SIP_USER, SIP_PASSWORD)
        assert "connection_id" not in saved.resources()
    assert carrier.connections["c1"]["password"] == SIP_PASSWORD  # create returned the real synthetic password
    before = copy.deepcopy((carrier.connections, carrier.profiles, carrier.numbers, saved.doc))
    carrier.calls.clear()
    carrier.crash_on = None
    with pytest.raises(ProviderError, match="masks the SIP password") as error:
        provider.attach(saved, number, "main", SIP_USER, SIP_PASSWORD)
    assert "manual SIP trunk" in str(error.value) and "does not match" not in str(error.value)
    assert SIP_PASSWORD not in str(error.value) and SIP_USER not in str(error.value)
    assert all(method == "GET" for method, _ in carrier.calls)
    assert (carrier.connections, carrier.profiles, carrier.numbers, saved.doc) == before
    assert len(carrier.connections) == len(carrier.profiles) == 1


def test_mask_token_is_never_accepted_as_password_equality():
    response = contract_fixture("credential_connection_retrieve_masked.json")
    with pytest.raises(ProviderError, match="masks the SIP password"):
        _provider(Carrier())._connection(response, _connection_record(password="********"), read=False)


@pytest.mark.parametrize("password", [None, "", 123, "missing"])
def test_unavailable_password_is_reported_as_unverifiable_without_adoption(tmp_path, password):
    carrier = Carrier()
    saved = _connection_attempt(tmp_path, carrier)
    record = _connection_record(password=password)
    if password == "missing":
        record.pop("password")
    carrier.connections["c1"] = record
    before = copy.deepcopy((carrier.connections, saved.doc))
    with pytest.raises(ProviderError, match="did not return a verifiable SIP password") as error:
        _resume_connection(_provider(carrier), saved)
    assert "does not match" not in str(error.value)
    assert SIP_PASSWORD not in str(error.value) and SIP_USER not in str(error.value)
    assert (carrier.connections, saved.doc) == before
    assert all(method == "GET" for method, _ in carrier.calls)


def test_missing_password_in_create_response_pauses_then_resumes_from_readable_detail(tmp_path):
    carrier = _owned(Carrier(), connection_id="previous-route")

    def handler(req):
        response = carrier(req)
        if req.method == "POST" and req.url.path == "/v2/credential_connections":
            body = response.json()
            body["data"].pop("password")
            return httpx.Response(response.status_code, json=body)
        return response

    provider = _provider(handler)
    saved = Journal.open(tmp_path, "main", "telnyx")
    number = Number("+15550100001", None, None, False, "n9")
    with pytest.raises(ProviderError, match="verifiable SIP password"):
        provider.attach(saved, number, "main", SIP_USER, SIP_PASSWORD)
    assert not carrier.patches and "connection_id" not in saved.resources()
    assert provider.attach(saved, number, "main", SIP_USER, SIP_PASSWORD).state == "ready"
    assert carrier.calls.count(("POST", "/v2/credential_connections")) == 1
    assert carrier.calls.count(("POST", "/v2/outbound_voice_profiles")) == 1
    assert carrier.patches == [("n9", {"connection_id": "c1"})]


@pytest.mark.parametrize("field,value", [
    ("active", "true"), ("encode_contact_header_enabled", 1), ("sip_uri_calling_preference", None),
    ("inbound.channel_limit", "1"), ("outbound.channel_limit", None), ("outbound.outbound_voice_profile_id", 1),
])
def test_setting_diagnostics_name_fields_without_accepting_missing_or_wrong_types(tmp_path, field, value):
    carrier = Carrier()
    saved = _connection_attempt(tmp_path, carrier)
    record = _connection_record()
    target, _, leaf = field.rpartition(".")
    (record[target] if target else record)[leaf] = value
    carrier.connections["c1"] = record
    with pytest.raises(ProviderError, match="Different fields") as error:
        _resume_connection(_provider(carrier), saved)
    assert f"Different fields: {field}." in str(error.value)
    assert SIP_PASSWORD not in str(error.value) and SIP_USER not in str(error.value)
    assert all(method == "GET" for method, _ in carrier.calls)
    assert "connection_id" not in saved.resources()

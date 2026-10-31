"""Twilio: Programmable Voice SIP Domain with registration for inbound, Elastic SIP Trunking termination for outbound.
Inbound TwiML is hosted on twimlets echo (Twilio-hosted, no server of ours). REGISTER must not go through the trunk proxy."""
from __future__ import annotations

import secrets
import urllib.parse

import httpx

from ai_phone.lines.journal import Journal
from ai_phone.providers.base import Number, ProviderError, ProvisionResult, ReleaseResult, VerificationStep

API = "https://api.twilio.com/2010-04-01/Accounts"
TRUNKING = "https://trunking.twilio.com/v1"
PRICING = "https://pricing.twilio.com/v1"
# Twilio SIP signalling ranges (every edge, so a line works whichever region Twilio routes through), from
# https://www.twilio.com/docs/sip-trunking/ip-addresses as published on 2026-09-03; the Task 0b spike note re-verifies.
SIGNALLING_CIDRS: list[str] = [
    "54.172.60.0/30", "54.244.51.0/30",           # North America (Virginia, Oregon)
    "54.171.127.192/30", "35.156.191.128/30",     # Europe (Ireland, Frankfurt)
    "54.65.63.192/30", "54.169.127.128/30", "54.252.254.64/30",   # Asia-Pacific (Tokyo, Singapore, Sydney)
    "177.71.206.192/30",                          # South America (São Paulo)
]


def _twiml_url(user: str, domain: str) -> str:
    twiml = f"<Response><Dial><Sip>sip:{user}@{domain}</Sip></Dial></Response>"
    return "https://twimlets.com/echo?Twiml=" + urllib.parse.quote(twiml, safe="")


class TwilioProvider:
    name = "twilio"
    signalling_cidrs = SIGNALLING_CIDRS

    def __init__(self, key: str, secret: str | None = None, http: httpx.Client | None = None):
        if not secret:
            raise ProviderError("twilio needs --secret-env (the Auth Token) as well as --key-env (the Account SID)")
        self.sid, self._token = key, secret
        self.http = http or httpx.Client(timeout=20.0)
        self.http.auth = (key, secret)

    def _redact(self, text: str) -> str:
        for v in sorted((v for v in getattr(self, "_sensitive", []) if v), key=len, reverse=True):
            text = text.replace(v, "***")
        return text.replace(self._token, "***").replace(self.sid, "AC***")

    def _call(self, method: str, url: str, **kw) -> dict:
        shown = self._redact(url.split("?")[0])
        try:
            r = self.http.request(method, url, timeout=20.0, follow_redirects=False, **kw)  # per request, so an injected client cannot bypass it
        except httpx.HTTPError as e:
            raise ProviderError(f"twilio {method} {shown}: {type(e).__name__}", retryable=True) from None
        if r.status_code >= 300:
            try:
                msg = str(r.json().get("message", ""))[:300]  # the carrier's message, never the body
            except ValueError:
                msg = "(non-JSON error response)"
            raise ProviderError(f"twilio {method} {shown} -> {r.status_code}: {self._redact(msg)}", retryable=r.status_code >= 500,
                                status=r.status_code)
        return r.json() if r.content else {}

    def _acct(self, path: str) -> str:
        return f"{API}/{self.sid}{path}"

    def _list(self, url: str, key: str, *, params: dict | None = None) -> list[dict]:
        """Read both Twilio pagination dialects without leaving this exact collection."""
        origin = urllib.parse.urlsplit(url)
        trunking = url.startswith(f"{TRUNKING}/")
        current = str(httpx.URL(url, params={"PageSize": "100", **(params or {})}))
        seen, records = set(), []
        for _ in range(100):
            try:
                parsed = urllib.parse.urlsplit(current)
            except ValueError:
                raise ProviderError("Twilio returned an invalid pagination link.") from None
            if (parsed.scheme != "https" or parsed.netloc != origin.netloc or parsed.path != origin.path
                    or parsed.username is not None or parsed.password is not None or parsed.fragment):
                raise ProviderError("Twilio returned an unsafe pagination link; no further request was sent.")
            if current in seen:
                raise ProviderError("Twilio repeated a pagination link; the resource search is incomplete.")
            seen.add(current)
            page = self._call("GET", current)
            rows = page.get(key) if isinstance(page, dict) else None
            if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
                raise ProviderError("Twilio returned an invalid resource list; the resource search is incomplete.")
            records.extend(rows)
            # The configured endpoint determines the dialect. Missing or mixed metadata must not
            # turn an incomplete recovery search into an empty result that permits another creation.
            if trunking:
                meta = page.get("meta")
                if not isinstance(meta, dict) or "next_page_url" not in meta or "next_page_uri" in page:
                    raise ProviderError("Twilio returned invalid pagination metadata; the resource search is incomplete.")
                following = meta["next_page_url"]
            else:
                if "next_page_uri" not in page or "meta" in page:
                    raise ProviderError("Twilio returned invalid pagination metadata; the resource search is incomplete.")
                following = page["next_page_uri"]
            if following is None:
                return records
            if not isinstance(following, str) or not following:
                raise ProviderError("Twilio returned an invalid pagination link.")
            try:
                current = urllib.parse.urljoin(current, following)
            except ValueError:
                raise ProviderError("Twilio returned an invalid pagination link.") from None
        raise ProviderError("Twilio's resource search exceeded the page limit; no new resource was created.")

    @staticmethod
    def _unique(records: list[dict], field: str, value: str) -> dict | None:
        matches = [record for record in records if record.get(field) == value]
        if len(matches) > 1:
            raise ProviderError("Twilio returned multiple matching resources. Review them in the Twilio console before retrying.")
        return matches[0] if matches else None

    # ---- numbers ------------------------------------------------------------------------------
    def search_numbers(self, country: str, area: str | None, limit: int) -> list[Number]:
        params = {"VoiceEnabled": "true", "PageSize": str(limit)}
        if area:
            params["AreaCode"] = area
        found = self._call("GET", self._acct(f"/AvailablePhoneNumbers/{country}/Local.json"), params=params).get("available_phone_numbers", [])
        price = currency = None
        try:
            pr = self._call("GET", f"{PRICING}/PhoneNumbers/Countries/{country}")
            local = next((p for p in pr.get("phone_number_prices", []) if p.get("number_type") == "local"), None)
            price, currency = (float(local["current_price"]) if local else None), pr.get("price_unit")
        except ProviderError:
            pass
        return [Number(n["phone_number"], price, currency, False, n["phone_number"]) for n in found]

    def owned_numbers(self) -> list[Number]:
        """The voice numbers already on the account. Read-only: this is also the free key check."""
        recs = self._list(self._acct("/IncomingPhoneNumbers.json"), "incoming_phone_numbers")
        return [Number(r["phone_number"], None, None, False, r["sid"]) for r in recs if (r.get("capabilities") or {}).get("voice", True)]

    # ---- provisioning -------------------------------------------------------------------------
    def provision(self, journal: Journal, number: Number, line_id: str, sip_user: str, sip_password: str) -> ProvisionResult:
        """One journal step per carrier request, so a crash between any request and its `done` is reconciled by name."""
        self._sensitive = getattr(self, "_sensitive", []) + [sip_password]
        domain_name = self._sip_domain(journal, line_id, sip_user, sip_password)
        st, planned = journal.start("number", phone_number=number.e164)
        if st != "done":
            rec = (self._find_number(number.e164) if st == "started" else None) or self._call(
                "POST", self._acct("/IncomingPhoneNumbers.json"),
                data={"PhoneNumber": number.e164, "FriendlyName": f"ai-phone:{line_id}", "VoiceUrl": _twiml_url(sip_user, domain_name), "VoiceMethod": "GET"})
            journal.finish("number", number_sid=rec["sid"])
        return self._trunk(journal, line_id, number.e164)

    def attach(self, journal: Journal, number: Number, line_id: str, sip_user: str, sip_password: str) -> ProvisionResult:
        """`provision` for a number already on the account: nothing is bought. The number's voice URL is pointed at the
        SIP domain; what it was before is journaled so `release` can put it back."""
        self._sensitive = getattr(self, "_sensitive", []) + [sip_password]
        st, planned = journal.start("claim", phone_number=number.e164)
        if st != "done":  # a read-only lookup, before anything is created
            rec = self._find_number(number.e164)
            if rec is None:
                raise ProviderError(f"{number.e164} is not on this Twilio account")
            if rec.get("trunk_sid") or rec.get("voice_application_sid"):
                raise ProviderError(f"{number.e164} is routed by a SIP trunk or a TwiML app on your Twilio account; "
                                    "remove that in the Twilio console first, then try again")
            # its own key, never `number_sid`: `release` deletes a `number_sid`, and this number is not ours to delete
            journal.finish("claim", brought_number_sid=rec["sid"], number_brought=True,
                           previous_voice_url=rec.get("voice_url") or "", previous_voice_method=rec.get("voice_method") or "POST")
        number_sid = journal.resources()["brought_number_sid"]
        domain_name = self._sip_domain(journal, line_id, sip_user, sip_password)
        st, planned = journal.start("route", brought_number_sid=number_sid)
        if st != "done":  # setting a voice URL is idempotent, so a half-done step is simply repeated
            self._call("POST", self._acct(f"/IncomingPhoneNumbers/{number_sid}.json"),
                       data={"VoiceUrl": _twiml_url(sip_user, domain_name), "VoiceMethod": "GET"})
            journal.finish("route", routed=True)
        return self._trunk(journal, line_id, number.e164)

    def _sip_domain(self, journal: Journal, line_id: str, sip_user: str, sip_password: str) -> str:
        """The credential list, its credential, the registration domain and their mapping; returns the domain name."""
        name = f"ai-phone:{line_id}"
        st, planned = journal.start("credential_list", name=name)
        if st != "done":
            rec = (self._find_credlist(name) if st == "started" else None) or self._call("POST", self._acct("/SIP/CredentialLists.json"), data={"FriendlyName": name})
            journal.finish("credential_list", credential_list_sid=rec["sid"])
        cl_sid = journal.resources()["credential_list_sid"]

        st, planned = journal.start("credential", user=sip_user)
        if st != "done":
            existing = self._find_credential(cl_sid, sip_user) if st == "started" else None
            rec = existing or self._call("POST", self._acct(f"/SIP/CredentialLists/{cl_sid}/Credentials.json"), data={"Username": sip_user, "Password": sip_password})
            journal.finish("credential", credential_sid=rec["sid"], user=sip_user)

        st, planned = journal.start("domain", domain_name=f"ai-phone-{line_id}-{secrets.token_hex(3)}.sip.twilio.com", name=name)
        domain_name = planned["domain_name"]
        if st != "done":
            rec = self._find_domain(domain_name) if st == "started" else None
            for attempt in range(2):
                if rec is not None:
                    break
                try:
                    rec = self._call("POST", self._acct("/SIP/Domains.json"),
                                     data={"DomainName": domain_name, "FriendlyName": name, "SipRegistration": "true", "VoiceMethod": "POST"})
                except ProviderError as err:
                    # A name collision is a 4xx that says so. A 5xx or a transport error is not one (`retryable` is
                    # exactly those): it is retryable under the *same* name, and renaming after one would orphan the
                    # domain the failed request may still have created, since the journal only records the second.
                    if attempt == 1 or err.retryable or "already" not in str(err).lower():
                        raise
                    domain_name = f"ai-phone-{line_id}-{secrets.token_hex(3)}.sip.twilio.com"  # someone else has that name: one fresh suffix
                    journal.doc["steps"][-1]["planned"]["domain_name"] = domain_name  # the journal always names what will be created
                    journal._save()
            journal.finish("domain", domain_sid=rec["sid"], domain_name=domain_name)
        domain_sid = journal.resources()["domain_sid"]

        st, planned = journal.start("domain_mapping", domain_sid=domain_sid, credential_list_sid=cl_sid)
        if st != "done":
            if not (st == "started" and self._domain_has_mapping(domain_sid, cl_sid)):
                self._call("POST", self._acct(f"/SIP/Domains/{domain_sid}/Auth/Registrations/CredentialListMappings.json"), data={"CredentialListSid": cl_sid})
            journal.finish("domain_mapping", mapped=True)
        return domain_name

    def _trunk(self, journal: Journal, line_id: str, e164: str) -> ProvisionResult:
        """The outbound trunk and its credential list; an account without trunking gets an inbound-only line."""
        name = f"ai-phone:{line_id}"
        cl_sid = journal.resources()["credential_list_sid"]
        st, planned = journal.start("trunk", trunk_domain=f"ai-phone-{line_id}-{secrets.token_hex(3)}.pstn.twilio.com", name=name)
        # a resumed inbound-only line has a done trunk step with no domain: plan a fresh one for the retry below
        trunk_domain = planned.get("trunk_domain") or f"ai-phone-{line_id}-{secrets.token_hex(3)}.pstn.twilio.com"
        if st != "done":
            rec = self._find_trunk(name) if st == "started" else None
            if rec is None:
                try:
                    rec = self._call("POST", f"{TRUNKING}/Trunks", data={"FriendlyName": name, "DomainName": trunk_domain})
                except ProviderError as e:
                    # Elastic SIP Trunking is not offered to trial accounts ("SIP trunking is only available after you
                    # upgrade your account"). The line still answers calls through the SIP domain; outbound waits for the
                    # upgrade and `lines resume`, which re-runs this step.
                    if e.status not in (401, 403):
                        raise
                    journal.finish("trunk", trunk_sid=None, trunk_domain=None, outbound_unavailable=str(e))
                    journal.set_state("ready")
                    return self._result(journal, e164)
            journal.finish("trunk", trunk_sid=rec["sid"], trunk_domain=rec.get("domain_name", trunk_domain))
        trunk_sid = journal.resources()["trunk_sid"]
        if not trunk_sid:  # a resumed inbound-only line: try trunking again (the account may have been upgraded since)
            try:
                rec = self._find_trunk(name) or self._call("POST", f"{TRUNKING}/Trunks", data={"FriendlyName": name, "DomainName": trunk_domain})
            except ProviderError as e:
                if e.status not in (401, 403):
                    raise
                journal.set_state("ready")
                return self._result(journal, e164)
            journal.finish("trunk", trunk_sid=rec["sid"], trunk_domain=rec.get("domain_name", trunk_domain))
            trunk_sid = rec["sid"]

        st, planned = journal.start("trunk_credential_list", trunk_sid=trunk_sid, credential_list_sid=cl_sid)
        if st != "done":
            if not (st == "started" and self._trunk_has_credlist(trunk_sid, cl_sid)):
                self._call("POST", f"{TRUNKING}/Trunks/{trunk_sid}/CredentialLists", data={"CredentialListSid": cl_sid})
            journal.finish("trunk_credential_list", attached=True)
        journal.set_state("ready")
        return self._result(journal, e164)

    def poll(self, journal: Journal) -> ProvisionResult:
        e164 = next(s["planned"]["phone_number"] for s in journal.doc["steps"] if s["name"] in ("number", "claim"))
        journal.set_state("ready")
        return self._result(journal, e164)

    def _result(self, journal: Journal, e164: str) -> ProvisionResult:
        r = journal.resources()
        line = {"kind": "trunk", "number": e164, "caller_id": e164, "registrar": f"sip:{r['domain_name']}", "register_via_proxy": False,
                "transport": "udp", "user": r["user"], "trusted_cidrs": list(SIGNALLING_CIDRS),
                "provider": {"name": self.name, "state": "ready",
                             "resources": {k: r[k] for k in ("credential_list_sid", "domain_sid", "number_sid", "brought_number_sid", "trunk_sid")
                                           if r.get(k)}}}
        if r.get("trunk_domain"):
            line.update(outbound_proxy=f"sip:{r['trunk_domain']}", dial_domain=r["trunk_domain"])
            return ProvisionResult("ready", f"{e164} is provisioned", line)
        return ProvisionResult("ready", f"{e164} is provisioned for inbound calls only: Twilio refused to create a SIP trunk "
                                        f"({r.get('outbound_unavailable', 'trunking unavailable')}). Trial accounts cannot use Elastic SIP "
                                        f"Trunking; after upgrading, run `ai-phone lines resume <line> --key-env … --secret-env …` to add outbound.", line)

    def _find_credlist(self, name):
        return self._unique(self._list(self._acct("/SIP/CredentialLists.json"), "credential_lists"), "friendly_name", name)

    def _find_credential(self, cl_sid, user):
        return self._unique(self._list(self._acct(f"/SIP/CredentialLists/{cl_sid}/Credentials.json"), "credentials"), "username", user)

    def _domain_has_mapping(self, domain_sid, cl_sid):
        return any(m.get("sid") == cl_sid or m.get("credential_list_sid") == cl_sid
                   for m in self._list(self._acct(f"/SIP/Domains/{domain_sid}/Auth/Registrations/CredentialListMappings.json"), "contents"))

    def _trunk_has_credlist(self, trunk_sid, cl_sid):
        return any(c.get("sid") == cl_sid for c in self._list(f"{TRUNKING}/Trunks/{trunk_sid}/CredentialLists", "credential_lists"))

    def _find_domain(self, domain_name):
        return self._unique(self._list(self._acct("/SIP/Domains.json"), "domains"), "domain_name", domain_name)

    def _find_number(self, e164):
        return self._unique(self._list(self._acct("/IncomingPhoneNumbers.json"), "incoming_phone_numbers", params={"PhoneNumber": e164}),
                            "phone_number", e164)

    def _find_trunk(self, name):
        return self._unique(self._list(f"{TRUNKING}/Trunks", "trunks"), "friendly_name", name)

    def release(self, journal: Journal) -> ReleaseResult:
        r = journal.resources()
        if r.get("number_brought") and not r.get("brought_number_sid"):  # an older journal kept it under number_sid
            r["brought_number_sid"] = r.pop("number_sid", None)
        leftover = {}
        if r.get("brought_number_sid"):
            # never delete a number ai-phone did not buy: put its voice URL back and leave it on the account
            sid = r["brought_number_sid"]
            try:
                self._call("POST", self._acct(f"/IncomingPhoneNumbers/{sid}.json"),
                           data={"VoiceUrl": r.get("previous_voice_url", ""), "VoiceMethod": r.get("previous_voice_method") or "POST"})
            except ProviderError as e:
                leftover["brought_number_sid"] = f"{sid}: {e} (the number stays on your account; reset its voice URL in the Twilio console)"
        for key, url in (("trunk_sid", lambda s: f"{TRUNKING}/Trunks/{s}"), ("number_sid", lambda s: self._acct(f"/IncomingPhoneNumbers/{s}.json")),
                         ("domain_sid", lambda s: self._acct(f"/SIP/Domains/{s}.json")), ("credential_list_sid", lambda s: self._acct(f"/SIP/CredentialLists/{s}.json"))):
            sid = r.get(key)
            if not sid:  # a `number_sid` is always one ai-phone bought
                continue
            try:
                self._call("DELETE", url(sid))
            except ProviderError as e:
                leftover[key] = f"{sid}: {e}"
        if leftover:
            journal.set_state("rollback_failed")
            return ReleaseResult(False, leftover)
        journal.set_state("pending")
        return ReleaseResult(True, {})

    # ---- caller id ----------------------------------------------------------------------------
    def start_caller_id_verification(self, e164: str) -> VerificationStep:
        rec = self._call("POST", self._acct("/OutgoingCallerIds.json"), data={"PhoneNumber": e164, "FriendlyName": "ai-phone"})
        return VerificationStep("enter_code_on_phone", f"Twilio is calling {e164}. Answer and key in this code: {rec['validation_code']}", {"e164": e164})

    def complete_caller_id_verification(self, step: VerificationStep, code: str | None) -> bool:
        ids = self._call("GET", self._acct("/OutgoingCallerIds.json"), params={"PhoneNumber": step.handle["e164"]}).get("outgoing_caller_ids", [])
        return any(i.get("phone_number") == step.handle["e164"] for i in ids)

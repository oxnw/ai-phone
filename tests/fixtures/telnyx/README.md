# Telnyx contract fixtures

These are synthetic, minimal response shapes, not account exports. All IDs, names,
credentials and timestamps are invented. The mock projects only these documented
fields; unknown request keys do not become response fields.

Sources, pinned when verified:

- [Telnyx OpenAPI](https://github.com/team-telnyx/openapi/blob/8938d09a4c9c7076c562b1609f7383749f6cedb5/openapi/spec3.json): `CredentialConnection`, `CreateCredentialConnectionRequest`, `CredentialInbound`, `CredentialOutbound`, `OutboundVoiceProfile`, `PaginationMeta`.
- [Generated credential SDK](https://github.com/team-telnyx/telnyx-node/blob/55a3f0b396257b2c272487a5ea162642516967d3/src/resources/credential-connections/credential-connections.ts).

Credential create returns the password. Retrieve/list mask it for primary accounts
created from 2026-09-08; an older account can still return it. The masking fixture
is deliberate, and cannot establish a password match.

Documented defaults modeled by the fake are active=true, encode-contact=false,
channel limits=null, conversational traffic, enabled profiles and US/CA destinations.
Requested channel limits of 1 must still be returned as 1; missing fields or defaults
are not substitutes. Credential subdomain receive settings are unsupported.

No coercion from strings to booleans, numbers or IDs is modeled. Profile destination
and tag order may vary without changing membership. The focused fixtures cover the
fields ai-phone uses; they are not a complete Telnyx schema or proof of live calling.

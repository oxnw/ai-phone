# SignalWire response fixtures

These are hand-authored, synthetic responses, not recordings from an account or
copies of request bodies. IDs, numbers, labels and dates are test values. Values
in the endpoint template are an example state, not claims about carrier defaults.
No password is returned by the endpoint response model.

Source: SignalWire's official TypeSpec at commit
[`ae11e126f48bfc0151cb20a8c3778ebf5f2d0db1`](https://github.com/signalwire/docs/tree/ae11e126f48bfc0151cb20a8c3778ebf5f2d0db1/specs/signalwire-rest), checked 2026-10-29:

- [SIP endpoint requests and responses](https://github.com/signalwire/docs/tree/ae11e126f48bfc0151cb20a8c3778ebf5f2d0db1/specs/signalwire-rest/fabric-api/sip-endpoints)
- [Phone-number addresses](https://github.com/signalwire/docs/tree/ae11e126f48bfc0151cb20a8c3778ebf5f2d0db1/specs/signalwire-rest/fabric-api/phone-number-addresses)
- [Phone numbers](https://github.com/signalwire/docs/tree/ae11e126f48bfc0151cb20a8c3778ebf5f2d0db1/specs/signalwire-rest/relay-rest/phone-numbers)
- [SIP profile](https://github.com/signalwire/docs/tree/ae11e126f48bfc0151cb20a8c3778ebf5f2d0db1/specs/signalwire-rest/relay-rest/sip-profile)
- [Verified caller IDs](https://github.com/signalwire/docs/tree/ae11e126f48bfc0151cb20a8c3778ebf5f2d0db1/specs/signalwire-rest/relay-rest/verified-caller-ids)

The test carrier validates the fields used by ai-phone independently of the
adapter and applies explicit state transitions. Unknown or response-only request
fields are rejected in the mock to expose mistakes; this is a strict test policy,
not a claim that every production endpoint rejects every unknown field.

Response metadata and nullable routing fields survive updates. Last-page list
responses have only a `self` link, matching a read-only observation of the API;
the published Fabric endpoint schema also declares `first`, so this deliberately
covers the smaller live shape. Phone addresses use `handler_type`, not `channel`.
DELETE of a phone address unassigns its resource and retains the address and number.
These fixtures check adapter behavior; they do not prove SIP interoperability or
that a live create/update/delete is accepted by the carrier.

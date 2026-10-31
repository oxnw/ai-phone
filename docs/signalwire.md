# SignalWire

SignalWire uses its API to configure a line and SIP/RTP to carry calls. The daemon does not use the carrier API.
Speech models and the current macOS Apple Silicon requirements are unchanged.

## Connect an existing number

Run `uv run ai-phone setup`, choose a phone number, then **SignalWire → Connect with API**. You need:

| Setting | Where it comes from |
|---|---|
| Space host | Your SignalWire dashboard host, such as `example.signalwire.com` |
| Project ID | The project containing the number |
| API token | A token for that project with Numbers and Voice permissions |
| Existing number | A number already owned by that project |

Setup saves credentials in `~/.ai-phone/secrets.env`, readable only by you. Do not paste keys into issue reports.
The API Space host and SIP domain are separate; setup retrieves the SIP domain from the project profile.

From the CLI, with credentials already saved in your environment or `secrets.env`:

```bash
uv run ai-phone lines add signalwire --id signalwire \
  --space example.signalwire.com \
  --key-env SIGNALWIRE_PROJECT_ID --secret-env SIGNALWIRE_API_TOKEN \
  --existing +15555550100
```

The adapter creates a dedicated SIP credential, configures its outbound caller ID, and routes the selected
number's incoming calls to it. It records the old voice route so removing the line can restore it. It does not
change the number's messaging settings or the project's shared SIP defaults. Fresh numbers with an empty default
webhook route are connected through SignalWire's calling-address API; removing the line leaves that calling
address unassigned again. Other existing routes must be restorable before they can be replaced.

Buy new numbers in SignalWire's portal, then connect them as existing numbers. Automated purchasing is unavailable:
the API contract we verified does not provide the country and price information needed by the purchase confirmation.

## Use SIP credentials you already have

The manual trunk setup needs the SIP domain, username and password from SignalWire. Set `from_user: account`.
This makes outbound SIP From use the authenticated username; SignalWire's endpoint `send_as` supplies the number
seen by the recipient. Registration and digest authentication still use the same username and password.

In setup, choose **SignalWire → Use SIP credentials**. The equivalent CLI command is:

```bash
uv run ai-phone lines add trunk --preset signalwire --id signalwire \
  --registrar sip:example.sip.signalwire.com:5061 --user aidemo \
  --password-env AI_PHONE_LINE_SIGNALWIRE_PASSWORD
```

The preset selects account identity, TLS, required SRTP and the verified wildcard policy described below.
Match your endpoint using `--transport` and `--srtp` when its settings differ. The wildcard policy applies only to
TLS trunks. A phone number is optional for manual account-identity trunks.

For an existing manually configured line, the relevant fields are:

```yaml
lines:
  signalwire:
    kind: trunk
    number: "+15555550100"
    user: aidemo
    password_env: AI_PHONE_LINE_SIGNALWIRE_PASSWORD
    registrar: "sip:example.sip.signalwire.com:5061;transport=tls"
    outbound_proxy: "sip:example.sip.signalwire.com:5061;transport=tls"
    register_via_proxy: true
    dial_domain: "example.sip.signalwire.com"
    transport: tls
    tls_hostname_policy: verified_wildcard
    srtp: required
    from_user: account
    accept_invites_from: registered_trunk
```

Use your actual SIP domain and match the endpoint's transport/encryption settings. The API-managed path creates
an endpoint requiring encryption and configures TLS with SRTP. Both paths retain the current PCMU/PCMA codecs.
For manual credentials, set `send_as` in SignalWire's portal to a number you own or have verified there. Editing a
local caller-ID value alone cannot change the provider's endpoint. Manual trunks do not need an API token.

## TLS certificates

The default `tls_hostname_policy: strict` keeps PJSIP's existing hostname checks, which reject wildcard names.
`verified_wildcard` accepts an exact DNS SAN or a wildcard covering one complete label, such as
`*.sip.signalwire.com` for `example.sip.signalwire.com`. The native handshake verifier checks the actual
connection's certificate chain, validity dates and DNS SAN before SIP can be sent or delivered. It rejects
untrusted or incomplete chains, expired certificates, Common Name fallback, partial wildcards and wildcards
spanning multiple labels. Certificate verification remains required.

This policy requires `kind: trunk`, `transport: tls`, `register_via_proxy: true` and an explicit
`outbound_proxy: "sip:DNS-host[:port];transport=tls"`. Use the provider's SIP DNS name, not an IP address or wildcard.
Both REGISTER and outbound calls use that fixed proxy; a connection to a different hostname is rejected.
Each opted-in line has its own TLS transport. Other lines keep their existing verification policy.

Independent incoming TLS connections to this dedicated transport are rejected. Incoming SIP requests on its
already authenticated outgoing connection can proceed through the usual source and call checks. Basic inbound
calling succeeded in the combined-preview tests described below; other carrier/network configurations remain
unverified.

New SignalWire API setups and the manual TLS preset save this policy explicitly. Existing configurations,
including resumed setups, are not automatically changed. To update an existing TLS line:

1. Run `./scripts/install.sh` from the ai-phone checkout. It checks the native verifier version and rebuilds an
   older binding; an outdated or unsupported binding cannot enable this policy.
2. Edit that line in `~/.ai-phone/config.yaml` to include the proxy, proxy-registration and policy fields in the
   example above. Keep its existing username, password reference, number and caller ID.
3. After any active call ends, stop the daemon with Ctrl+C in its running terminal, then start it again with
   `uv run ai-phone run`. Check the line with `uv run ai-phone status`.

This local update does not require recreating the carrier endpoint or changing credentials.

## Change caller ID

For an API-managed line, choose **Change caller ID** in setup. Selecting the line's number or a verified number
updates the endpoint's `send_as` and records the selection locally. A number already owned or verified on the
project needs no verification call. For another number, choosing verification starts a call; enter its code when prompted.

The equivalent CLI commands are:

```bash
uv run ai-phone lines verify-caller-id signalwire +15555550101 \
  --space example.signalwire.com \
  --key-env SIGNALWIRE_PROJECT_ID --secret-env SIGNALWIRE_API_TOKEN
uv run ai-phone lines set-caller-id signalwire +15555550100 \
  --space example.signalwire.com \
  --key-env SIGNALWIRE_PROJECT_ID --secret-env SIGNALWIRE_API_TOKEN
```

Caller-ID verification does not guarantee how every receiving carrier or handset displays the call. If a remote
update succeeds but the local save fails, resume the line to reconcile the pending selection.

## Resume or remove a line

To continue an unfinished existing-number setup in the wizard, choose **SignalWire → Connect with API** and
select the number marked **Resume**. Use the same Space and Project as before.

The equivalent CLI commands are:

```bash
uv run ai-phone lines resume signalwire --space example.signalwire.com \
  --key-env SIGNALWIRE_PROJECT_ID --secret-env SIGNALWIRE_API_TOKEN
uv run ai-phone lines remove signalwire --space example.signalwire.com \
  --key-env SIGNALWIRE_PROJECT_ID --secret-env SIGNALWIRE_API_TOKEN
```

Removal keeps a number you brought, restores its recorded voice route (or unassigns a previously empty calling
address), and removes the SIP resource ai-phone created. It refuses to overwrite a route changed later in the portal. Failed cleanup retains the journal and
reports resources requiring attention. Restart a running daemon after line configuration changes.

## Verification status

On 2026-10-31, the standalone SignalWire branch passed 1,489 unit tests and 52 local SIP/TLS checks against
its rebuilt native binding. The latter include 41 certificate/isolation/lifecycle cases and two registration
tests through the actual Python binding. Local tests use temporary settings, synthetic credentials and fake
models; they do not register with a carrier or place real calls.

On 2026-10-31, real SignalWire-to-Telnyx and Telnyx-to-SignalWire calls between US numbers completed with two local
AIs exchanging speech in both directions. Received caller-ID numbers matched the expected originating numbers,
and both calls ended normally. The inbound test also confirmed automatic answer. These calls used a combined
development preview containing other pending changes, so they establish carrier compatibility for that setup
rather than a live-call test of the standalone SignalWire branch.

This establishes basic carrier calling in the tested setup. Opening greetings partly overlapped, and a forward
arithmetic answer had a transcription mismatch. Transcript checks do not establish perfect transcription,
perceptual voice quality, latency targets or human interruption behavior. Two concurrent local model stacks
make these functional checks unsuitable as performance benchmarks. International destinations, long-running
registration renewal, live
cancellation followed by another call, and carrier-resource cleanup/restoration remain unverified. Caller-ID
numbers observed in these calls do not establish display behavior on every receiving carrier or handset.

Inbound trust uses the configured SIP domain; if the carrier sends from other signaling addresses, those must
be verified and configured explicitly. Do not bypass source checks by selecting unrestricted inbound acceptance.
The calling-address API used for fresh numbers is currently marked beta by SignalWire.

References: [SignalWire SIP credentials](https://signalwire.com/docs/platform/voice/sip/sip-credentials),
[caller ID](https://signalwire.com/docs/platform/voice/how-to-set-caller-id-or-cnam),
[API authorization](https://signalwire.com/docs/apis/authorization),
[calling-address assignment](https://signalwire.com/docs/apis/rest/phone-number-addresses/create-phone-number-address),
and [unassignment](https://signalwire.com/docs/apis/rest/phone-number-addresses/delete-phone-number-address).

"""pjsua2 Account and Call subclasses: decide fast on PJSIP's thread, post events (with line_id) to the asyncio loop."""
from __future__ import annotations

import asyncio
import logging

from ai_phone.telephony import events as ev
from ai_phone.telephony.pj import pj
from ai_phone.telephony.policy import PeerSet, source_allowed

log = logging.getLogger("ai-phone.telephony")


def _post(loop: asyncio.AbstractEventLoop, q: asyncio.Queue, item) -> None:
    loop.call_soon_threadsafe(q.put_nowait, item)


def _reject_incoming(call: "AiCall", source_ip: str, code: int) -> None:
    """Main thread: reject after PJSUA2 has released the incoming request's clone.

    PJSUA2 2.17 frees incoming_data unconditionally after onIncomingCall returns.
    Rejecting synchronously can reset the call and free that clone first. The
    native hangup takes PJSUA's lock, so this thread waits for callback cleanup;
    the queued argument keeps the unclaimed Call alive until then.
    """
    if call.ended:
        return  # the caller cancelled before this queued rejection ran
    call_id, account = call.getId(), call.acc
    op = pj.CallOpParam()
    op.statusCode = code
    try:
        call.hangup(op)
    except pj.Error:
        log.debug("incoming call %s ended before it could be rejected", call_id)
        return
    account.events.put_nowait(ev.CallRejected(call_id, account.line_id, source_ip, code))


class AiCall(pj.Call):
    def __init__(self, acc: "LineAccount", call_id: int = pj.PJSUA_INVALID_ID, direction: str = "in"):
        super().__init__(acc, call_id)
        self.acc = acc
        self.direction = direction
        self.media_connected = False
        self.answered = False
        self.ended = False  # PJSIP reported DISCONNECTED; the daemon releases the one-call claim after its own teardown
        self.cancelled = False  # set by Telephony when it CANCELs an unanswered outbound call
        self.answer_timeout = None  # asyncio TimerHandle for the no-answer CANCEL, owned by this call
        self._media_index = None  # outbound: audio that became active before CONFIRMED (early media) waits here

    def onCallState(self, prm) -> None:
        ci = self.getInfo()
        lid, loop, q = self.acc.line_id, self.acc.loop, self.acc.events
        if ci.state == pj.PJSIP_INV_STATE_EARLY and self.direction == "out":
            _post(loop, q, ev.OutboundProgress(self.getId(), lid, "ringing", int(ci.lastStatusCode), str(ci.lastReason)))
        elif ci.state == pj.PJSIP_INV_STATE_CONFIRMED:
            self.answered = True
            if self.direction == "out":
                _post(loop, q, ev.OutboundProgress(self.getId(), lid, "answered", 200, "OK"))
            _post(loop, q, ev.CallStateChanged(self.getId(), lid, "active"))
            if self._media_index is not None and not self.media_connected:
                self._connect_media(self._media_index)  # early media was held back; the answer is real now
        elif ci.state == pj.PJSIP_INV_STATE_DISCONNECTED:
            code, reason = int(ci.lastStatusCode), str(ci.lastReason)
            self.ended = True
            if self.direction == "out" and not self.answered:
                _post(loop, q, ev.OutboundProgress(self.getId(), lid, self._outcome(code), code, reason))
            _post(loop, q, ev.CallStateChanged(self.getId(), lid, "ended", code, reason))
            self.acc.telephony._call_ended(self)  # timers only; the claim is released by the daemon once teardown is done

    def _outcome(self, code: int) -> str:
        if code in (486, 600):
            return "busy"
        if code == 487 and self.cancelled:
            return "no_answer"
        if 300 <= code < 400:
            return "redirected"
        return "rejected"

    def onCallRedirected(self, prm) -> int:
        return pj.PJSIP_REDIRECT_STOP  # never follow 3xx, not to the next target either

    def _connect_media(self, index: int) -> None:
        am = self.getAudioMedia(index)
        port = self.acc.telephony.port
        am.startTransmit(self.acc.telephony.receive_port)
        port.startTransmit(am)
        info = self.getStreamInfo(index)
        codec, rate = str(info.codecName), int(info.codecClockRate)
        # Logging handlers can block; only the asyncio thread writes the codec report.
        self.acc.loop.call_soon_threadsafe(log.info, "media line=%s codec=%s rate=%d",
                                           self.acc.line_id, codec, rate)
        self.media_connected = True
        _post(self.acc.loop, self.acc.events, ev.MediaActive(self.getId(), self.acc.line_id))

    def onCallMediaState(self, prm) -> None:
        ci = self.getInfo()
        for mi in ci.media:
            if mi.type == pj.PJMEDIA_TYPE_AUDIO and mi.status == pj.PJSUA_CALL_MEDIA_ACTIVE and not self.media_connected:
                if self.direction == "out" and not self.answered:
                    self._media_index = mi.index  # early media (183 with SDP): ignored until the 200 OK
                    continue
                self._connect_media(mi.index)

    def onDtmfDigit(self, prm) -> None:
        _post(self.acc.loop, self.acc.events, ev.Dtmf(self.getId(), self.acc.line_id, prm.digit))


class LineAccount(pj.Account):
    def __init__(self, telephony: "Telephony", spec, peers: PeerSet):
        super().__init__()
        self.telephony, self.spec, self.peers = telephony, spec, peers
        self.line_id = spec.id
        self.loop, self.events = telephony.loop, telephony.events
        self.registered = spec.kind != "trunk"  # sip lines need no registration; trunks flip this in onRegState

    def onRegState(self, prm) -> None:
        ai = self.getInfo()
        self.registered = bool(ai.regIsActive)
        _post(self.loop, self.events, ev.RegState(self.line_id, bool(ai.regIsActive), int(prm.code), str(prm.reason), int(prm.expiration)))

    def onIncomingCall(self, prm) -> None:  # PJSIP thread: decide fast, no models
        t = self.telephony
        call = AiCall(self, prm.callId, "in")
        source_ip = str(prm.rdata.srcAddress).rsplit(":", 1)[0].strip("[]")
        if not t.cfg.calls.allow_inbound or not source_allowed(self.spec.accept_invites_from, source_ip, self.peers):
            self.loop.call_soon_threadsafe(_reject_incoming, call, source_ip, 403)
            return
        if not t.try_claim_call(call):  # one call at a time across every line, atomically against make_call
            self.loop.call_soon_threadsafe(_reject_incoming, call, source_ip, 486)
            return
        op = pj.CallOpParam()
        op.statusCode = pj.PJSIP_SC_RINGING
        call.answer(op)
        _post(self.loop, self.events, ev.IncomingCall(call.getId(), self.line_id, str(call.getInfo().remoteUri), source_ip))

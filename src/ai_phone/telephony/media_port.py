"""AudioMediaPort → rings. Runs on PJSIP's media thread: copy bytes, stamp times, nothing else."""
from __future__ import annotations

import time

from ai_phone.telephony.pj import pj
from ai_phone.telephony.rings import FRAME_BYTES, SILENCE, FrameRing


class ReceiveMediaPort(pj.AudioMediaPort):
    """The native bridge resamples inbound audio before this fixed 16 kHz callback."""

    def __init__(self, rx: FrameRing):
        super().__init__()
        self.rx = rx
        self.last_rx_ns = 0
        fmt = pj.MediaFormatAudio()
        fmt.type = pj.PJMEDIA_TYPE_AUDIO
        fmt.clockRate, fmt.channelCount, fmt.bitsPerSample = 16000, 1, 16
        fmt.frameTimeUsec = 20000
        fmt.avgBps = fmt.maxBps = 16000 * 16
        self.createPort("ai-phone-rx", fmt)

    def onFrameReceived(self, frame) -> None:
        data = bytes(frame.buf)
        if len(data) != FRAME_BYTES:
            data = (data + SILENCE)[:FRAME_BYTES]
        self.last_rx_ns = time.monotonic_ns()
        self.rx.push(data)


class RingMediaPort(pj.AudioMediaPort):
    def __init__(self, rx: FrameRing | None, tx: FrameRing, echo: bool = False, sample_rate: int = 16000):
        super().__init__()
        if sample_rate not in (16000, 48000) or tx.frame_bytes != sample_rate // 50 * 2:
            raise ValueError("playback ring must contain 20 ms frames at 16 or 48 kHz")
        if sample_rate != 16000 and (rx is not None or echo):
            raise ValueError("fullband playback requires a separate 16 kHz receive port")
        self.rx, self.tx, self.echo = rx, tx, echo
        self._silence = bytes(tx.frame_bytes)
        self.last_rx_ns = 0
        self.first_nonsilent_ns: int | None = None
        self._armed = False
        self.playing = False  # set by the controller; silence while listening is not an underrun
        self._flush_requested = False  # set by the controller; honoured here, on the consumer thread
        fmt = pj.MediaFormatAudio()
        fmt.type = pj.PJMEDIA_TYPE_AUDIO
        fmt.clockRate = sample_rate
        fmt.channelCount = 1
        fmt.bitsPerSample = 16
        fmt.frameTimeUsec = 20000
        fmt.avgBps = fmt.maxBps = sample_rate * 16
        self.createPort("ai-phone", fmt)

    def arm_t5(self) -> None:
        self.first_nonsilent_ns = None
        self._armed = True

    def onFrameReceived(self, frame) -> None:
        if self.rx is None:
            return  # This transmit-only port is never connected to incoming media.
        data = bytes(frame.buf)
        if len(data) != FRAME_BYTES:
            data = (data + SILENCE)[:FRAME_BYTES]
        self.last_rx_ns = time.monotonic_ns()
        (self.tx if self.echo else self.rx).push(data)

    @property
    def flush_pending(self) -> bool:
        """True between request_flush() and the PJSIP callback that performs it; frames pushed meanwhile would be lost."""
        return bool(self._flush_requested)

    def request_flush(self) -> None:
        """Ask the PJSIP thread (the tx ring's only consumer) to discard queued audio at its next frame."""
        self._flush_requested = True

    def cancel_flush(self) -> None:
        """Between calls no frame is requested, so a pending flush would hit the next call's greeting; drop it."""
        self._flush_requested = False

    def onFrameRequested(self, frame) -> None:
        if self._flush_requested:
            self.tx.flush()
            self._flush_requested = False  # cleared only after the flush, so flush_pending never lies to the controller
        data = self.tx.pop_or_silence() if self.playing else (self.tx.pop() or self._silence)
        if self._armed and data != self._silence:
            self.first_nonsilent_ns = time.monotonic_ns()
            self._armed = False
        frame.type = pj.PJMEDIA_FRAME_TYPE_AUDIO
        frame.buf = pj.ByteVector(data)
        frame.size = len(data)

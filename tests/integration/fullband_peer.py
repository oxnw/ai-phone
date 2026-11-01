"""Isolated native children for test_fullband_audio; no daemon, models or carriers.

The subject uses production Telephony. The independent peer uses pjsua2 directly
and captures the received 48 kHz PCM before any application resampling. Native
callbacks only copy frames; file I/O and signal checks happen off those threads.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
import math
from pathlib import Path
import resource
import struct
import sys
import time
import wave


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def write_wav(path, frames, rate):
    with wave.open(str(path), 'wb') as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(b''.join(frames))


def tone(rate, hz, seconds, amplitude=6000):
    samples = round(rate * seconds)
    raw = b''.join(struct.pack('<h', round(amplitude * math.sin(2 * math.pi * hz * i / rate)))
                   for i in range(samples)) if hz else bytes(samples * 2)
    size = rate // 50 * 2
    return [raw[i:i + size].ljust(size, b'\0') for i in range(0, len(raw), size)]


async def wait_file(path, timeout=10):
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f'timed out waiting for {path.name}')
        await asyncio.sleep(.01)


def codec_parameters(ep):
    from ai_phone.telephony.pj import pj
    assert callable(getattr(pj, 'ai_phone_opus_version', None)), 'rebuild the required Opus native binding'
    assert pj.ai_phone_opus_version() == 1
    value = ep.codecGetParam('opus/48000/2')
    info, setting = value.info, value.setting
    return {'clock_rate': int(info.clockRate), 'channels': int(info.channelCnt),
            'avg_bps': int(info.avgBps), 'frame_len_ms': int(info.frameLen),
            'frame_len_denum': int(info.frameLenDenum), 'frames_per_packet': int(setting.frmPerPkt),
            'cbr': bool(setting.cbr), 'vad': bool(setting.vad)}


def negotiated(call):
    from ai_phone.telephony.pj import pj
    result = []
    call_info = call.getInfo()  # Own the SWIG parent until its media vector has been read.
    for media in call_info.media:
        if media.type != pj.PJMEDIA_TYPE_AUDIO or media.status != pj.PJSUA_CALL_MEDIA_ACTIVE:
            continue
        stream = call.getStreamInfo(media.index)
        param = stream.audCodecParam
        info, setting = param.info, param.setting
        result.append({'name': str(stream.codecName), 'clock_rate': int(stream.codecClockRate),
                       'pcm_clock_rate': int(info.clockRate), 'channels': int(info.channelCnt),
                       'avg_bps': int(info.avgBps), 'frame_len_ms': int(info.frameLen),
                       'frame_len_denum': int(info.frameLenDenum),
                       'frames_per_packet': int(setting.frmPerPkt),
                       'tx_pt': int(stream.txPt), 'rx_pt': int(stream.rxPt)})
    return result


def port_format(port):
    info = port.getPortInfo()
    fmt = info.format
    return {'rate': int(fmt.clockRate), 'channels': int(fmt.channelCount),
            'bits': int(fmt.bitsPerSample), 'frame_usec': int(fmt.frameTimeUsec)}


async def send_frames(tel, tx, frames):
    index = 0
    deadline = time.monotonic() + len(frames) * .02 + 3
    tel.port.playing = True
    while index < len(frames):
        if time.monotonic() > deadline:
            raise TimeoutError('TX producer did not drain')
        while index < len(frames) and tx.available() < 20:
            tx.push(frames[index])
            index += 1
        await asyncio.sleep(.005)
    while tx.available():
        if time.monotonic() > deadline:
            raise TimeoutError('final TX frames did not drain')
        await asyncio.sleep(.005)
    tel.port.playing = False


async def subject_cycle(args, cycle):
    from ai_phone.config import Config, LineConfig
    from ai_phone.telephony import endpoint, events as ev
    from ai_phone.telephony.rings import FrameRing
    from ai_phone.telephony.pj import pj
    endpoint._system_resolvers = lambda: []

    class LocalTelephony(endpoint.Telephony):
        def _account_config(self, spec):
            config = super()._account_config(spec)
            config.mediaConfig.transportConfig.boundAddress = '127.0.0.1'
            config.natConfig.sipStunUse = pj.PJSUA_STUN_USE_DISABLED
            config.natConfig.mediaStunUse = pj.PJSUA_STUN_USE_DISABLED
            return config

    cfg = Config()
    cfg.codecs = args.subject_codecs.split(',')
    cfg.sip.bind.address = cfg.sip.public_addr = '127.0.0.1'
    cfg.sip.bind.udp_port = cfg.sip.bind.tcp_port = args.subject_port
    cfg.sip.bind.tls_port = args.subject_port + 1
    cfg.sip.stun_server = None
    cfg.log.pjsip_level = 3
    cfg.rtp.port_min = 10000 + (args.subject_port - 5060) * 100
    cfg.rtp.port_max = cfg.rtp.port_min + 20
    line = LineConfig(kind='sip', user='ai', aor=f'sip:ai@127.0.0.1:{args.subject_port}',
                      caller_id=f'sip:ai@127.0.0.1:{args.subject_port}',
                      accept_invites_from='lan', srtp='off')
    spec = endpoint.LineSpec.from_config('test', line, None)
    loop = asyncio.get_running_loop()
    events = asyncio.Queue()
    rx = FrameRing(None, 100)
    tx = FrameRing(None, 100, frame_bytes=cfg.playback_rate // 50 * 2)
    tel = LocalTelephony(cfg, [spec], loop, events, rx, tx)
    report = {'cycle': cycle, 'playback_rate': cfg.playback_rate, 'rx_frame_bytes': rx.frame_bytes,
              'tx_frame_bytes': tx.frame_bytes, 'events': []}
    received = []
    sender = None
    finished = False

    async def drain_rx():
        while not finished:
            while (frame := rx.pop()) is not None:
                received.append(frame)
            await asyncio.sleep(.005)

    async def send():
        await asyncio.sleep(.3)
        rate = cfg.playback_rate
        high = 12000 if rate == 48000 else 6000
        frames = (tone(rate, 1000, .6) + tone(rate, 0, .2) +
                  tone(rate, high, .8) + tone(rate, 0, .4))
        await send_frames(tel, tx, frames)
        if args.flush:
            # More stale audio is queued than can play during the short lead-in.
            # Only the callback consumer may flush the ring; use the public API.
            for frame in tone(rate, 3000, 1.8):
                tx.push(frame)
            tel.port.playing = True
            await asyncio.sleep(.2)
            report['queued_before_flush'] = tx.available()
            report['flush_requested_ns'] = time.monotonic_ns()
            tel.port.request_flush()
            deadline = time.monotonic() + 1
            while tel.port.flush_pending:
                if time.monotonic() > deadline:
                    raise TimeoutError('media callback did not acknowledge flush')
                await asyncio.sleep(.005)
            report['flush_ack_ns'] = time.monotonic_ns()
            report['queued_after_flush'] = tx.available()
            tel.port.playing = False
            await asyncio.sleep(1)
            report['fresh_requested_ns'] = time.monotonic_ns()
            await send_frames(tel, tx, tone(rate, high, .8) + tone(rate, 0, .4))
            report['fresh_drained_ns'] = time.monotonic_ns()
        await asyncio.sleep(.3)
        report['tx_overruns'] = tx.overruns
        if tel.current_call is not None:
            tel.hangup(tel.current_call.getId())

    draining = asyncio.create_task(drain_rx())
    try:
        tel.start()
        report['tx_port'] = port_format(tel.port)
        report['rx_port'] = port_format(tel.receive_port)
        report['separate_receive_port'] = tel.receive_port is not tel.port
        if 'opus' in cfg.codecs:
            report['opus_defaults'] = codec_parameters(tel.ep)
        save(args.directory / f'subject-ready-{cycle}.json', {'ready': True})
        await wait_file(args.directory / f'peer-ready-{cycle}.json')
        if args.direction == 'out':
            tel.make_call('test', f'sip:peer@127.0.0.1:{args.peer_port}', timeout_sec=8)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if sender is not None and sender.done() and sender.exception() is not None:
                raise sender.exception()
            try:
                event = await asyncio.wait_for(events.get(), .1)
            except asyncio.TimeoutError:
                continue
            report['events'].append({'type': type(event).__name__, **asdict(event)})
            if isinstance(event, ev.IncomingCall):
                assert args.direction == 'in'
                assert tel.answer(event.call_id)
            if isinstance(event, ev.MediaActive):
                report['negotiated'] = negotiated(tel.current_call)
                assert sender is None, 'duplicate media activation'
                sender = asyncio.create_task(send())
            if isinstance(event, ev.CallStateChanged) and event.state == 'ended':
                report['ended'] = True
                break
        else:
            raise TimeoutError('subject call did not finish')
        if sender is not None:
            await asyncio.wait_for(sender, 2)
        assert report.get('negotiated'), 'no actual negotiated stream'
    finally:
        if sender is not None and not sender.done():
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)
        finished = True
        await draining
        report['rx_lengths'] = sorted(set(map(len, received)))
        report['rx_frames'] = len(received)
        report['rx_overruns'] = rx.overruns
        tel.stop()  # Joins media clock while both callback ports and rings remain owned.
        await asyncio.sleep(0)
        report['stopped'] = tel.ep is None and tel.port is None and tel.receive_port is None
        for ring in (rx, tx):
            ring.close()
            ring.unlink()
        write_wav(args.directory / f'subject-rx-{cycle}.wav', received, 16000)
        save(args.directory / f'subject-{cycle}.json', report)


async def peer_cycle(args, cycle):
    from ai_phone.telephony.pj import pj
    loop = asyncio.get_running_loop()
    events = asyncio.Queue()
    report = {'cycle': cycle}
    captures, capture_times, capture_lengths = [], [], set()
    unexpected_capture_frames = set()
    capture_none_frames = 0
    source_frame = tone(16000, 2000, .02, amplitude=5000)[0]

    class Capture(pj.AudioMediaPort):
        def __init__(self):
            super().__init__()
            fmt = pj.MediaFormatAudio()
            fmt.type = pj.PJMEDIA_TYPE_AUDIO
            fmt.clockRate, fmt.channelCount, fmt.bitsPerSample = 48000, 1, 16
            fmt.frameTimeUsec = 20000
            self.createPort('independent-fullband-capture', fmt)

        def onFrameReceived(self, frame):
            nonlocal capture_none_frames
            data = bytes(frame.buf)
            frame_type, declared_size = int(frame.type), int(frame.size)
            if frame_type == pj.PJMEDIA_FRAME_TYPE_AUDIO:
                capture_lengths.add(len(data))
                if declared_size != 1920 or len(data) != 1920:
                    unexpected_capture_frames.add((frame_type, declared_size, len(data)))
                    return
            elif frame_type == pj.PJMEDIA_FRAME_TYPE_NONE and declared_size == 0 and not data:
                # A valid empty callback represents one silent 20 ms interval.
                # Preserve its place in the recording and timestamp sequence.
                data = bytes(1920)
                capture_none_frames += 1
            else:
                unexpected_capture_frames.add((frame_type, declared_size, len(data)))
                return
            captures.append(data)
            capture_times.append(time.monotonic_ns())

    class Source(pj.AudioMediaPort):
        def __init__(self):
            super().__init__()
            fmt = pj.MediaFormatAudio()
            fmt.type = pj.PJMEDIA_TYPE_AUDIO
            fmt.clockRate, fmt.channelCount, fmt.bitsPerSample = 16000, 1, 16
            fmt.frameTimeUsec = 20000
            self.createPort('independent-speech-band-source', fmt)

        def onFrameRequested(self, frame):
            frame.type = pj.PJMEDIA_FRAME_TYPE_AUDIO
            frame.buf = pj.ByteVector(source_frame)
            frame.size = 640

    class Call(pj.Call):
        def __init__(self, account, call_id=pj.PJSUA_INVALID_ID):
            super().__init__(account, call_id)
            self.confirmed = False
            self.connected = False
            self.ended = False
            self.audio = None

        def connect(self):
            if not self.confirmed or self.connected:
                return
            info = self.getInfo()
            for item in info.media:
                if item.type == pj.PJMEDIA_TYPE_AUDIO and item.status == pj.PJSUA_CALL_MEDIA_ACTIVE:
                    self.audio = self.getAudioMedia(item.index)
                    self.audio.startTransmit(capture)
                    source.startTransmit(self.audio)
                    self.connected = True
                    loop.call_soon_threadsafe(events.put_nowait, 'media')
                    break

        def onCallState(self, prm):
            info = self.getInfo()
            if info.state == pj.PJSIP_INV_STATE_CONFIRMED:
                self.confirmed = True
                self.connect()
            elif info.state == pj.PJSIP_INV_STATE_DISCONNECTED:
                self.ended = True
                loop.call_soon_threadsafe(events.put_nowait, 'ended')

        def onCallMediaState(self, prm):
            self.connect()

    class Account(pj.Account):
        def __init__(self):
            super().__init__()
            self.call = None

        def onIncomingCall(self, prm):
            self.call = Call(self, prm.callId)
            loop.call_soon_threadsafe(events.put_nowait, 'incoming')

    ep = pj.Endpoint()
    account = capture = source = None
    ep.libCreate()
    try:
        config = pj.EpConfig()
        config.uaConfig.threadCnt = 1
        config.medConfig.clockRate = config.medConfig.sndClockRate = 48000
        config.medConfig.audioFramePtime = 20
        config.medConfig.ecTailLen = 0
        config.medConfig.noVad = True
        config.logConfig.level = 3
        ep.libInit(config)
        wanted = args.peer_codecs.split(',')
        if 'opus' in wanted:
            assert pj.ai_phone_opus_version() == 1, 'Opus is required, not an optional skipped test'
            assert pj.ai_phone_enable_opus() == 0
        ep.audDevManager().setNullDev()
        transport = pj.TransportConfig()
        transport.port = args.peer_port
        transport.boundAddress = transport.publicAddress = '127.0.0.1'
        transport_id = ep.transportCreate(pj.PJSIP_TRANSPORT_UDP, transport)
        ep.libStart()
        available = ep.codecEnum2()
        ids = [str(codec.codecId) for codec in available]
        for name in wanted:
            assert any(codec.split('/')[0] == name for codec in ids), f'{name} missing from native codec factory'
        for codec in ids:
            name = codec.split('/')[0]
            ep.codecSetPriority(codec, 254 - wanted.index(name) if name in wanted else 0)
        if 'opus' in wanted:
            report['opus_defaults'] = codec_parameters(ep)
        capture, source = Capture(), Source()
        report['capture_port'] = port_format(capture)
        account = Account()
        ac = pj.AccountConfig()
        ac.idUri = f'sip:peer@127.0.0.1:{args.peer_port}'
        ac.regConfig.registerOnAdd = False
        ac.sipConfig.transportId = transport_id
        ac.natConfig.sipStunUse = ac.natConfig.mediaStunUse = pj.PJSUA_STUN_USE_DISABLED
        ac.natConfig.iceEnabled = ac.natConfig.turnEnabled = False
        ac.mediaConfig.srtpUse = pj.PJMEDIA_SRTP_DISABLED
        ac.mediaConfig.transportConfig.boundAddress = ac.mediaConfig.transportConfig.publicAddress = '127.0.0.1'
        ac.mediaConfig.transportConfig.port = 10000 + (args.peer_port - 5060) * 100
        ac.mediaConfig.transportConfig.portRange = 20
        account.create(ac)
        save(args.directory / f'peer-ready-{cycle}.json', {'ready': True})
        await wait_file(args.directory / f'subject-ready-{cycle}.json')
        if args.direction == 'in':
            account.call = Call(account)
            account.call.makeCall(f'sip:ai@127.0.0.1:{args.subject_port}', pj.CallOpParam(True))
        while True:
            event = await asyncio.wait_for(events.get(), 18)
            if event == 'incoming':
                assert args.direction == 'out'
                answer = pj.CallOpParam()
                answer.statusCode = 200
                account.call.answer(answer)
            elif event == 'media':
                report['negotiated'] = negotiated(account.call)
            elif event == 'ended':
                report['ended'] = True
                break
        assert report.get('negotiated'), 'peer never obtained an active media stream'
    finally:
        if account is not None and account.call is not None:
            if not account.call.ended:
                try:
                    account.call.hangup(pj.CallOpParam())
                except pj.Error:
                    pass
                until = time.monotonic() + 2
                while not account.call.ended and time.monotonic() < until:
                    await asyncio.sleep(.02)
            if account.call.audio is not None:
                try:
                    account.call.audio.stopTransmit(capture)
                    source.stopTransmit(account.call.audio)
                except pj.Error:
                    pass  # The disconnected stream may already have left the bridge.
                account.call.audio = None
            account.call = None
        if account is not None:
            account.shutdown()
            account = None
        # The independent capture is a Python callback target; keep it alive until
        # the native clock is joined, then release ports before libDestroy.
        ep.audDevManager().setNoDev()
        capture = source = None
        ep.libDestroy()
        ep = None
        await asyncio.sleep(0)
        report.update(stopped=True, capture_lengths=sorted(capture_lengths),
                      capture_none_frames=capture_none_frames,
                      unexpected_capture_frames=[{'type': kind, 'size': size, 'buffer_length': length}
                                                 for kind, size, length in sorted(unexpected_capture_frames)],
                      capture_frames=len(captures), capture_times_ns=capture_times)
        write_wav(args.directory / f'peer-rx-{cycle}.wav', captures, 48000)
        save(args.directory / f'peer-{cycle}.json', report)


async def run(args):
    for cycle in range(args.cycles):
        routine = subject_cycle if args.role == 'subject' else peer_cycle
        await asyncio.wait_for(routine(args, cycle), 25)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--role', choices=('subject', 'peer'), required=True)
    parser.add_argument('--direction', choices=('in', 'out'), required=True)
    parser.add_argument('--subject-codecs', required=True)
    parser.add_argument('--peer-codecs', required=True)
    parser.add_argument('--subject-port', type=int, required=True)
    parser.add_argument('--peer-port', type=int, required=True)
    parser.add_argument('--cycles', type=int, default=1)
    parser.add_argument('--flush', action='store_true')
    args = parser.parse_args()
    assert args.subject_port != 5060 and args.peer_port == args.subject_port + 4
    assert args.cycles in (1, 2)
    sys.path[:0] = [str(args.root / 'src'), str(args.root)]
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    asyncio.run(run(args))


if __name__ == '__main__':
    main()

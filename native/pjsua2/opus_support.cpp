#include "opus_support.hpp"
#include <pjsua-lib/pjsua.h>
#include <pjmedia-codec/opus.h>

int ai_phone_opus_version() {
#if PJ_VERSION_NUM_MAJOR == 2 && PJ_VERSION_NUM_MINOR == 17
    return 1;
#else
    return 0;
#endif
}

int ai_phone_enable_opus() {
    // Codec defaults and fmtp must be fixed before streams or accounts exist.
    if (!ai_phone_opus_version()) return PJ_ENOTSUP;
    if (pjsua_get_state() != PJSUA_STATE_INIT) return PJ_EINVALIDOP;
    pjmedia_endpt* endpoint = pjsua_get_pjmedia_endpt();
    if (!endpoint) return PJ_EINVALIDOP;

    // A bottle built with Opus registers its factory during libInit. The bundled
    // factory is otherwise registered here. Repeated calls never add a duplicate.
    const pj_str_t id = {const_cast<char*>("opus/48000/2"), 12};
    pjmedia_codec_param param;
    pj_status_t status = pjsua_codec_get_param(&id, &param);
    if (status != PJ_SUCCESS) {
        status = pjmedia_codec_opus_init(endpoint);
        if (status != PJ_SUCCESS) return status;
        status = pjsua_codec_get_param(&id, &param);
        if (status != PJ_SUCCESS) return status;
    }

    pjmedia_codec_opus_config cfg;
    status = pjmedia_codec_opus_get_config(&cfg);
    if (status != PJ_SUCCESS) return status;
    cfg.sample_rate = 48000;
    cfg.channel_cnt = 1;
    cfg.frm_ptime = 20;
    cfg.frm_ptime_denum = 1;
    cfg.bit_rate = 64000;
    cfg.packet_loss = 0;
    cfg.complexity = 10;
    cfg.cbr = PJ_TRUE;
    // The upstream setter does not copy these cfg values into CodecParam.
    // Keep the codec manager's defaults/readback consistent with the factory.
    param.setting.cbr = PJ_TRUE;
    param.setting.complexity = 10;
    param.setting.packet_loss = 0;
    param.setting.vad = PJ_FALSE;
    param.setting.frm_per_pkt = 1;
    // Retain PJSIP's packet-loss concealment. This setter also regenerates SDP
    // fmtp; setting only CodecParam would leave the advertised defaults stale.
    return pjmedia_codec_opus_set_default_param(&cfg, &param);
    // The endpoint's codec manager destroys the factory during libDestroy.
}

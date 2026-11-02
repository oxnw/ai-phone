/* Use PJSIP's own adapter when the installed library already supplies it. */
#include <pjmedia-codec/config.h>

#if !defined(PJMEDIA_HAS_OPUS_CODEC) || !PJMEDIA_HAS_OPUS_CODEC
#if PJ_VERSION_NUM_MAJOR != 2 || PJ_VERSION_NUM_MINOR != 17
#error "The bundled Opus adapter requires pjproject 2.17"
#endif
#undef PJMEDIA_HAS_OPUS_CODEC
#define PJMEDIA_HAS_OPUS_CODEC 1
#include "pjproject-opus.c"
#endif

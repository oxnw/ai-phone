#pragma once
#include <pjsua2.hpp>
#include <string>

// Version 1 is the PJSIP 2.17/OpenSSL handshake verifier proved by the native suite.
int ai_phone_tls_verifier_version();
// Returns a PJSUA transport ID, or a negative pj_status_t. The factory binds an
// ephemeral IPv4 port, rejects independent inbound TLS, and verifies its exact
// configured DNS next hop before any SIP application bytes can be exchanged.
int ai_phone_create_verified_tls_transport(const pj::TransportConfig& config,
                                          const std::string& expected_host);
// Begin shutdown without freeing callback storage or closing established TLS.
void ai_phone_deactivate_tls_verifier();
// Must be called only after Endpoint.libDestroy() has returned. False means
// the endpoint is still alive: callback storage has deliberately been retained.
bool ai_phone_reset_tls_verifier();

#if defined(AI_PHONE_TLS_VERIFY_TESTS) && !defined(SWIG)
namespace ai_phone_tls_test {
using Observer = void (*)(const char*, int, int, int, const char*);
void set_observer(Observer observer);
void hold_activation(bool hold);
void activate();
void deactivate(unsigned slot);
int slot_for_factory(const pjsip_tpfactory* factory);
void fail_at(unsigned phase); // 1 before registration, 2 after registration
int last_port();
int last_register_status();
int last_rollback_status();
const char* certificate_identity(const char*, size_t, const char*, size_t);
}
#endif

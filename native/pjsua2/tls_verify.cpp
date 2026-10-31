#include "tls_verify.hpp"
#include <pjsip/sip_transport_tls.h>
#include <pj/ssl_sock.h>
#include <array>
#include <atomic>
#include <mutex>
#include <new>

// Older/different backends retain the ordinary strict transport. Never claim
// this capability on a backend whose handshake callback ordering was not proved.
#if PJ_VERSION_NUM_MAJOR == 2 && PJ_VERSION_NUM_MINOR == 17 && \
    defined(PJ_HAS_SSL_SOCK) && PJ_HAS_SSL_SOCK && \
    PJ_SSL_SOCK_IMP == PJ_SSL_SOCK_IMP_OPENSSL
#define AI_PHONE_VERIFIED_TLS_SUPPORTED 1
#include <openssl/ssl.h>
#include <openssl/pem.h>
#include <openssl/x509v3.h>
#else
#define AI_PHONE_VERIFIED_TLS_SUPPORTED 0
#endif

namespace {
#if AI_PHONE_VERIFIED_TLS_SUPPORTED
constexpr unsigned kSlots=8;
struct Policy {
    bool reserved=false; // Only construction/reset, under lifecycle_mutex.
    std::string hostname; // Immutable until complete endpoint destruction.
    std::atomic<bool> active{false};
    std::atomic<pjsip_tpfactory*> factory{nullptr};
};
std::array<Policy,kSlots> policies;
bool accepting_factories=true; // Protected by lifecycle_mutex.
std::mutex lifecycle_mutex; // Never acquired by a TLS verification callback.
#ifdef AI_PHONE_TLS_VERIFY_TESTS
ai_phone_tls_test::Observer observer=nullptr;
bool hold_active=false;
unsigned fault_phase=0;
int last_bound_port=0, last_registered=0, last_rollback=0;
void observe(const char* kind,int slot,int value,int aux=0,const char* reason="") noexcept {
    if(observer) observer(kind,slot,value,aux,reason);
}
#else
void observe(const char*,int,int,int=0,const char* = "") noexcept {}
#endif
bool dns_name(const std::string& input,std::string& out) {
    if(input.empty() || input.size()>253) return false;
    out.clear(); out.reserve(input.size());
    unsigned label=0; bool has_letter_or_hyphen=false; char previous=0;
    for(char c:input) {
        if(c>='A' && c<='Z') c=char(c-'A'+'a');
        if(c=='.') { if(!label || previous=='-') return false; label=0; }
        else {
            if(!((c>='a'&&c<='z') || (c>='0'&&c<='9') || c=='-')) return false;
            if((!label && c=='-') || ++label>63) return false;
            if(!(c>='0'&&c<='9')) has_letter_or_hyphen=true;
        }
        out+=c; previous=c;
    }
    // Reject numeric/IP-like input too; IP SAN verification is a different policy.
    return label && previous!='-' && has_letter_or_hyphen;
}
bool same_dns_name(const char* actual,const std::string& expected) noexcept {
    if(!actual) return false;
    for(size_t i=0;i<expected.size();++i) {
        char c=actual[i];
        if(!c) return false;
        if(c>='A' && c<='Z') c=char(c-'A'+'a');
        if(c!=expected[i]) return false;
    }
    return actual[expected.size()]=='\0';
}
const char* leaf_identity(const char* pem,size_t length,
                          const char* host,size_t host_length) noexcept {
    if(!pem || !length || length>1024*1024) return "certificate_missing";
    if(!host || !host_length) return "next_hop";
    BIO* bio=BIO_new_mem_buf(pem,int(length));
    if(!bio) return "certificate_parse";
    X509* cert=PEM_read_bio_X509(bio,nullptr,nullptr,nullptr);
    BIO_free(bio);
    if(!cert) return "certificate_parse";
    const int before=X509_cmp_current_time(X509_get0_notBefore(cert));
    const int after=X509_cmp_current_time(X509_get0_notAfter(cert));
    const unsigned flags=X509_CHECK_FLAG_NO_PARTIAL_WILDCARDS |
                         X509_CHECK_FLAG_NEVER_CHECK_SUBJECT;
    const int match=X509_check_host(cert,host,host_length,flags,nullptr);
    X509_free(cert);
    if(before>=0 || after<=0) return "validity";
    if(match!=1) return "dns_san";
    return nullptr;
}
template<unsigned Slot>
pj_bool_t verify(const pjsip_tls_on_verify_param* param) noexcept {
    const Policy& policy=policies[Slot];
    auto reject=[&](const char* why,int status=0) {
        observe("verify",Slot,0,status,why); return PJ_FALSE;
    };
    if(!param || param->tp_dir!=PJSIP_TP_DIR_OUTGOING) return reject("incoming");
    if(!policy.active.load(std::memory_order_acquire) || !policy.factory.load())
        return reject("inactive");
    pj_ssl_sock_info info{};
    if(!param->ssock || pj_ssl_sock_get_info(param->ssock,&info)!=PJ_SUCCESS)
        return reject("socket_info");
    // PJSIP accumulates each OpenSSL chain error BEFORE invoking this callback
    // at every depth. Reject every nonzero status, not just the leaf callback.
    if(info.verify_status) return reject("chain",int(info.verify_status));
    if(!info.native_ssl) return reject("native_ssl");
    const char* actual=SSL_get_servername(static_cast<SSL*>(info.native_ssl),
                                         TLSEXT_NAMETYPE_host_name);
    if(!same_dns_name(actual,policy.hostname)) return reject("next_hop");
    const pj_ssl_cert_info* cert=param->remote_cert_info;
    if(!cert || cert->raw.slen<=0) return reject("certificate_missing");
    const char* failure=leaf_identity(cert->raw.ptr,size_t(cert->raw.slen),
                                     policy.hostname.data(),policy.hostname.size());
    if(failure) return reject(failure);
    if(!policy.active.load(std::memory_order_acquire)) return reject("inactive");
    observe("verify",Slot,1,0,"accepted");
    return PJ_TRUE;
}
#ifdef AI_PHONE_TLS_VERIFY_TESTS
template<unsigned Slot>
void accept_failed(const pjsip_tls_on_accept_fail_param* param) noexcept {
    observe("inbound_rejected",Slot,param?int(param->status):-1,
            param?int(param->last_native_err):0);
}
#endif
using Verify=pj_bool_t (*)(const pjsip_tls_on_verify_param*);
const std::array<Verify,kSlots> callbacks{{verify<0>,verify<1>,verify<2>,verify<3>,
                                         verify<4>,verify<5>,verify<6>,verify<7>}};
#ifdef AI_PHONE_TLS_VERIFY_TEST
#error Use AI_PHONE_TLS_VERIFY_TESTS for the test-only observer build.
#endif
#endif
} // namespace

int ai_phone_tls_verifier_version() { return AI_PHONE_VERIFIED_TLS_SUPPORTED?1:0; }

int ai_phone_create_verified_tls_transport(const pj::TransportConfig& config,
                                          const std::string& expected_host) {
#if !AI_PHONE_VERIFIED_TLS_SUPPORTED
    (void)config; (void)expected_host; return -PJ_ENOTSUP;
#else
    try {
        std::lock_guard<std::mutex> guard(lifecycle_mutex);
        if(!accepting_factories) return -PJ_EINVALIDOP;
        const auto state=pjsua_get_state();
        if(state!=PJSUA_STATE_INIT && state!=PJSUA_STATE_RUNNING) return -PJ_EINVALIDOP;
        std::string hostname;
        if(config.port!=0 || !dns_name(expected_host,hostname)) return -PJ_EINVAL;
        pjsua_transport_config native=config.toPj();
        if(native.tls_setting.ca_list_file.slen==0 && native.tls_setting.ca_list_path.slen==0 &&
           native.tls_setting.ca_buf.slen==0) return -PJ_EINVAL;
        unsigned slot=0;
        while(slot<kSlots && policies[slot].reserved) ++slot;
        if(slot==kSlots) return -PJ_ETOOMANY;
        Policy& policy=policies[slot];
        policy.hostname=std::move(hostname);
        policy.reserved=true;
        // Failure consumes this callback slot until full destruction/reset.
        // No late callback can ever observe a different expected hostname.
        policy.active.store(false);
        native.tls_setting.verify_server=PJ_FALSE;
        native.tls_setting.verify_client=PJ_TRUE;
        native.tls_setting.require_client_cert=PJ_TRUE;
        native.tls_setting.enable_renegotiation=PJ_FALSE;
        native.tls_setting.on_verify_cb=callbacks[slot];
#ifdef AI_PHONE_TLS_VERIFY_TESTS
        using Accept=void (*)(const pjsip_tls_on_accept_fail_param*);
        static const std::array<Accept,kSlots> accept_callbacks{{accept_failed<0>,accept_failed<1>,
            accept_failed<2>,accept_failed<3>,accept_failed<4>,accept_failed<5>,accept_failed<6>,accept_failed<7>}};
        native.tls_setting.on_accept_fail_cb=accept_callbacks[slot];
#endif
        pj_sockaddr local{};
        pj_status_t status=pj_sockaddr_init(pj_AF_INET(),&local,
                           native.bound_addr.slen?&native.bound_addr:nullptr,0);
        if(status!=PJ_SUCCESS) return -status;
        pjsip_host_port published{};
        if(native.public_addr.slen) { published.host=native.public_addr; published.port=0; }
        pjsip_tpfactory* factory=nullptr;
        status=pjsip_tls_transport_start2(pjsua_get_pjsip_endpt(),&native.tls_setting,&local,
                         native.public_addr.slen?&published:nullptr,1,&factory);
        if(status!=PJ_SUCCESS) return -status;
        policy.factory.store(factory,std::memory_order_release);
#ifdef AI_PHONE_TLS_VERIFY_TESTS
        last_bound_port=pj_sockaddr_get_port(&factory->local_addr);
        last_registered=0; last_rollback=0;
        if(fault_phase==1) {
            last_rollback=factory->destroy(factory);
            return last_rollback==PJ_SUCCESS?-PJ_ECANCELLED:-last_rollback;
        }
#endif
        pjsua_transport_id id=PJSUA_INVALID_ID;
        status=pjsua_tpfactory_register(factory,&id);
#ifdef AI_PHONE_TLS_VERIFY_TESTS
        last_registered=status;
#endif
        if(status!=PJ_SUCCESS) {
            const auto rollback=factory->destroy(factory);
#ifdef AI_PHONE_TLS_VERIFY_TESTS
            last_rollback=rollback;
#endif
            return rollback==PJ_SUCCESS?-status:-rollback;
        }
#ifdef AI_PHONE_TLS_VERIFY_TESTS
        if(fault_phase==2) {
            last_rollback=pjsua_transport_close(id,PJ_FALSE);
            return last_rollback==PJ_SUCCESS?-PJ_ECANCELLED:-last_rollback;
        }
        if(!hold_active)
#endif
            policy.active.store(true,std::memory_order_release);
        return id;
    } catch(const std::bad_alloc&) { return -PJ_ENOMEM; }
      catch(...) { return -PJ_EUNKNOWN; }
#endif
}

void ai_phone_deactivate_tls_verifier() {
#if AI_PHONE_VERIFIED_TLS_SUPPORTED
    std::lock_guard<std::mutex> guard(lifecycle_mutex);
    accepting_factories=false;
    for(auto& policy:policies) policy.active.store(false,std::memory_order_release);
#endif
}

bool ai_phone_reset_tls_verifier() {
#if AI_PHONE_VERIFIED_TLS_SUPPORTED
    std::lock_guard<std::mutex> guard(lifecycle_mutex);
    if(pjsua_get_state()!=PJSUA_STATE_NULL) return false;
    // libDestroy has stopped worker threads and drained transports/callbacks.
    for(auto& policy:policies) {
        policy.active.store(false);
        policy.factory.store(nullptr);
        policy.hostname.clear();
        policy.reserved=false;
    }
    accepting_factories=true;
#endif
    return pjsua_get_state()==PJSUA_STATE_NULL;
}

#if defined(AI_PHONE_TLS_VERIFY_TESTS) && AI_PHONE_VERIFIED_TLS_SUPPORTED
namespace ai_phone_tls_test {
void set_observer(Observer fn) { observer=fn; }
void hold_activation(bool hold) { hold_active=hold; }
void activate() {
    std::lock_guard<std::mutex> guard(lifecycle_mutex);
    if(accepting_factories) for(auto& p:policies)
        if(p.reserved && p.factory.load()) p.active.store(true);
}
void deactivate(unsigned slot) { if(slot<kSlots) policies[slot].active.store(false); }
int slot_for_factory(const pjsip_tpfactory* factory) {
    for(unsigned i=0;i<kSlots;++i) if(policies[i].factory.load()==factory) return int(i);
    return -1;
}
void fail_at(unsigned phase) { fault_phase=phase; }
int last_port() { return last_bound_port; }
int last_register_status() { return last_registered; }
int last_rollback_status() { return last_rollback; }
const char* certificate_identity(const char* pem,size_t size,const char* host,size_t length) {
    return leaf_identity(pem,size,host,length);
}
}
#endif

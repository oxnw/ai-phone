// Synthetic native regression driver. Links the SAME production TLS verifier.
#include "tls_verify.hpp"
#include <pjsua-lib/pjsua.h>
#include <pjsip/sip_transport_tls.h>
#include <pjlib-util/resolver.h>
#include <openssl/ssl.h>
#include <openssl/pem.h>
#include <openssl/x509v3.h>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include <poll.h>
#include <unistd.h>
#include <sys/resource.h>

struct Policy {
    std::string expected_host; // Set before factory creation; never modified/reused.
    std::atomic<bool> closed{false};
    pjsua_transport_id transport = PJSUA_INVALID_ID;
    pjsua_acc_id account = PJSUA_INVALID_ID;
};
static std::array<Policy, 2> policies;
static pjsip_tpfactory* strict_factory = nullptr;
static pjsua_transport_id strict_transport = PJSUA_INVALID_ID;
static pjsua_acc_id strict_account = PJSUA_INVALID_ID;
static pjsua_acc_id wrong_hop_account=PJSUA_INVALID_ID;
static unsigned retry_sec=0;
struct Event { const char* kind; int slot, value, aux; const char* reason; };
// Bounded multi-producer, single-consumer log. Each slot is written once;
// no locks/waits or I/O in native callbacks, including under thread_cnt=1.
struct Record { Event event{}; std::atomic<bool> ready{false}; };
static std::array<Record, 65536> events;
static std::atomic<unsigned> next_event{0};
static unsigned read_event=0;
static std::atomic<unsigned> dropped{0};
static void event(const char* kind, int slot, int value, int aux=0,
                  const char* reason="") noexcept {
    unsigned index=next_event.fetch_add(1,std::memory_order_relaxed);
    if(index>=events.size()) { dropped.fetch_add(1,std::memory_order_relaxed); return; }
    events[index].event={kind,slot,value,aux,reason};
    events[index].ready.store(true,std::memory_order_release);
}
static void flush_events() {
    while(read_event<events.size() && events[read_event].ready.load(std::memory_order_acquire)) {
        const auto &e=events[read_event++].event;
        std::cout << "{\"event\":\"" << e.kind << "\",\"slot\":" << e.slot
                  << ",\"value\":" << e.value << ",\"aux\":" << e.aux
                  << ",\"reason\":\"" << e.reason << "\"}\n";
    }
    auto lost=dropped.exchange(0);
    if(lost) std::cout << "{\"event\":\"overflow\",\"value\":" << lost << "}\n";
    std::cout.flush();
}
static int slot_for(pjsip_tpfactory* factory) noexcept {
    if(factory==strict_factory) return -1;
    return ai_phone_tls_test::slot_for_factory(factory);
}
static bool dns_test_host(const std::string& s) {
    if (s.size()<6 || s.size()>253 || s.substr(s.size()-5)!=".test") return false;
    bool label=false; char prev=0;
    for(char c:s) {
        if(c=='.') { if(!label || prev=='-') return false; label=false; }
        else { if(!((c>='a'&&c<='z')||(c>='0'&&c<='9')||c=='-')) return false;
               if(!label&&c=='-') return false; label=true; }
        prev=c;
    }
    return label && prev!='-';
}
using ai_phone_tls_test::certificate_identity;
static int selfcheck(const std::string& pem_path) {
    std::ifstream in(pem_path,std::ios::binary);
    std::ostringstream data; data<<in.rdbuf(); std::string valid=data.str();
    if(!in || valid.empty()) return 2;
    const char* host="tenant.carrier.test";
    const char* malformed="not a certificate";
    const char* truncated="-----BEGIN CERTIFICATE-----\nAA";
    const std::array<std::pair<const char*,bool>,7> results{{
        {"missing",certificate_identity(nullptr,20,host,strlen(host))!=nullptr},
        {"empty",certificate_identity("",0,host,strlen(host))!=nullptr},
        {"malformed",certificate_identity(malformed,strlen(malformed),host,strlen(host))!=nullptr},
        {"truncated",certificate_identity(truncated,strlen(truncated),host,strlen(host))!=nullptr},
        {"missing_host",certificate_identity(valid.data(),valid.size(),nullptr,0)!=nullptr},
        {"valid",certificate_identity(valid.data(),valid.size(),host,strlen(host))==nullptr},
        {"wrong_host",certificate_identity(valid.data(),valid.size(),"wrong.carrier.test",18)!=nullptr}
    }};
    bool all=true;
    for(auto& r:results) {
        std::cout<<"{\"event\":\"selfcheck\",\"case\":\""<<r.first
                 <<"\",\"passed\":"<<(r.second?"true":"false")<<"}\n";
        all=all&&r.second;
    }
    return all?0:1;
}
static void reg_state(pjsua_acc_id id,pjsua_reg_info*) {
    pjsua_acc_info ai{}; pjsua_acc_get_info(id,&ai);
    int slot=id==strict_account?-1:-2;
    for(unsigned i=0;i<policies.size();++i) if(id==policies[i].account) slot=int(i);
    if(id==wrong_hop_account) slot=0;
    event("registration",slot,int(ai.status),int(ai.reg_last_err),
          id==wrong_hop_account?"wrong_hop":"");
}
static void transport_state(pjsip_transport* tp,pjsip_transport_state state,
                            const pjsip_transport_state_info* info) {
    event("transport",slot_for(tp->factory),int(state),info?int(info->status):0);
}
static pj_bool_t receive_request(pjsip_rx_data* data) {
    auto* tp=data->tp_info.transport;
    event("application_request",slot_for(tp->factory),int(data->msg_info.msg->line.req.method.id));
    // A real response proves the request reached the application after TLS.
    if(data->msg_info.msg->line.req.method.id==PJSIP_OPTIONS_METHOD) {
        pjsip_endpt_respond_stateless(pjsua_get_pjsip_endpt(),data,200,nullptr,nullptr,nullptr);
        return PJ_TRUE;
    }
    return PJ_FALSE;
}
static pjsip_module observer;
static void check(pj_status_t status,const char* operation) {
    if(status!=PJ_SUCCESS) {
        std::cerr << "native proof failure: " << operation << " status=" << status << '\n';
        std::exit(2);
    }
}
static pj_str_t str(const std::string& s) { return pj_str(const_cast<char*>(s.c_str())); }
static pjsua_acc_id account(pjsua_transport_id transport,const std::string& host,int peer_port) {
    pjsua_acc_config ac; pjsua_acc_config_default(&ac);
    std::string id="sip:alice@"+host;
    std::string registrar="sip:"+host+":"+std::to_string(peer_port)+";transport=tls";
    std::string proxy=registrar+";lr";
    ac.id=str(id); ac.reg_uri=str(registrar);
    ac.proxy_cnt=1; ac.proxy[0]=str(proxy);
    ac.transport_id=transport;
    ac.register_on_acc_add=PJ_FALSE;
    ac.reg_timeout=120; ac.reg_retry_interval=retry_sec; ac.reg_first_retry_interval=retry_sec;
    ac.reg_retry_random_interval=0;
    ac.allow_contact_rewrite=0; ac.allow_via_rewrite=0; ac.allow_sdp_nat_rewrite=0;
    ac.sip_stun_use=PJSUA_STUN_USE_DISABLED;
    ac.media_stun_use=PJSUA_STUN_USE_DISABLED;
    ac.cred_count=1;
    ac.cred_info[0].scheme=pj_str(const_cast<char*>("digest"));
    ac.cred_info[0].realm=pj_str(const_cast<char*>("*"));
    ac.cred_info[0].username=pj_str(const_cast<char*>("alice"));
    ac.cred_info[0].data_type=PJSIP_CRED_DATA_PLAIN_PASSWD;
    ac.cred_info[0].data=pj_str(const_cast<char*>("synthetic-secret"));
    pjsua_acc_id idout; check(pjsua_acc_add(&ac,PJ_FALSE,&idout),"account"); return idout;
}
static int cycles(const std::map<std::string,std::string>& args) {
    const unsigned count=std::stoul(args.at("--cycles"));
    if(count<2 || count>20 || !args.count("--ca")) return 2;
    ai_phone_tls_test::hold_activation(false);
    for(unsigned cycle=1;cycle<=count;++cycle) {
        check(pjsua_create(),"cycle_create");
        pjsua_config cfg; pjsua_config_default(&cfg); cfg.thread_cnt=1;
        pjsua_logging_config log; pjsua_logging_config_default(&log); log.level=0;log.console_level=0;
        pjsua_media_config media; pjsua_media_config_default(&media);media.thread_cnt=0;media.has_ioqueue=PJ_FALSE;
        check(pjsua_init(&cfg,&log,&media),"cycle_init");
        pj::TransportConfig transport; transport.port=0; transport.boundAddress="127.0.0.1";
        transport.tlsConfig.CaListFile=args.at("--ca");
        if(args.count("--cert")) transport.tlsConfig.certFile=args.at("--cert");
        if(args.count("--key")) transport.tlsConfig.privKeyFile=args.at("--key");
        const int first=ai_phone_create_verified_tls_transport(transport,"tenant.carrier.test");
        const int second=ai_phone_create_verified_tls_transport(transport,"other.carrier.test");
        if(first<0 || second<0) return 2;
        pjsua_transport_info a{},b{};
        check(pjsua_transport_get_info(first,&a),"cycle_info");
        check(pjsua_transport_get_info(second,&b),"cycle_info");
        check(pjsua_start(),"cycle_start");
        const bool rejected_live_reset=!ai_phone_reset_tls_verifier();
        ai_phone_deactivate_tls_verifier();
        check(pjsua_destroy2(PJSUA_DESTROY_NO_NETWORK),"cycle_destroy");
        const bool reset=ai_phone_reset_tls_verifier();
        flush_events();
        std::cout<<"{\"event\":\"cycle_destroyed\",\"cycle\":"<<cycle
                 <<",\"ports\":["<<pj_sockaddr_get_port(&a.local_addr)<<","<<pj_sockaddr_get_port(&b.local_addr)
                 <<"],\"reset\":"<<(reset?"true":"false")
                 <<",\"active_reset_rejected\":"<<(rejected_live_reset?"true":"false")<<"}\n";
        if(!reset || !rejected_live_reset) return 1;
    }
    std::cout<<"{\"event\":\"destroyed\"}\n";
    return 0;
}
int main(int argc,char** argv) {
    rlimit core_limit{0,0};
    if (setrlimit(RLIMIT_CORE,&core_limit)!=0) {
        std::cerr << "Cannot disable core dumps\n"; return 2;
    }
    std::map<std::string,std::string> args;
    for(int i=1;i+1<argc;i+=2) args[argv[i]]=argv[i+1];
    if(ai_phone_tls_verifier_version()!=1) { std::cerr<<"Verified TLS capability unavailable\n"; return 2; }
    ai_phone_tls_test::set_observer(&event);
    if(args.count("--selfcheck")) return selfcheck(args["--selfcheck"]);
    if(args.count("--cycles")) return cycles(args);
    for(auto key:{"--ca","--dns-port","--peer-port"}) if(!args.count(key)) {
        std::cerr<<"Missing "<<key<<'\n'; return 2;
    }
    const std::string host=args.count("--host")?args["--host"]:"tenant.carrier.test";
    const std::string host2=args.count("--host2")?args["--host2"]:"other.carrier.test";
    policies[0].expected_host=args.count("--expected-host")?args["--expected-host"]:host;
    policies[1].expected_host=args.count("--expected-host2")?args["--expected-host2"]:host2;
    for(auto h:{host,host2,policies[0].expected_host,policies[1].expected_host})
        if(!dns_test_host(h)) { std::cerr<<"Only validated synthetic .test DNS hosts allowed\n"; return 2; }
    int dns_port=std::stoi(args["--dns-port"]), peer_port=std::stoi(args["--peer-port"]);
    if(dns_port<1024||dns_port>65535||peer_port<1024||peer_port>65535) return 2;
    unsigned threads=args.count("--threads")?std::stoul(args["--threads"]):0;
    retry_sec=args.count("--retry-sec")?std::stoul(args["--retry-sec"]):0;
    if(threads>1 || retry_sec>30) return 2;
    check(pjsua_create(),"create");
    pjsua_config cfg; pjsua_config_default(&cfg); cfg.thread_cnt=threads;
    cfg.cb.on_reg_state2=&reg_state; cfg.cb.on_transport_state=&transport_state;
    pjsua_logging_config log; pjsua_logging_config_default(&log); log.level=0;log.console_level=0;
    pjsua_media_config media; pjsua_media_config_default(&media);
    media.thread_cnt=0; media.has_ioqueue=PJ_FALSE;
    check(pjsua_init(&cfg,&log,&media),"init");
    pj_dns_resolver* resolver=nullptr;
    check(pjsip_endpt_create_resolver(pjsua_get_pjsip_endpt(),&resolver),"resolver");
    pj_str_t ns=pj_str(const_cast<char*>("127.0.0.1")); pj_uint16_t port=pj_uint16_t(dns_port);
    check(pj_dns_resolver_set_ns(resolver,1,&ns,&port),"resolver_ns");
    check(pjsip_endpt_set_resolver(pjsua_get_pjsip_endpt(),resolver),"resolver_install");
    pjsua_transport_config tc; pjsua_transport_config_default(&tc);
    tc.port=0; tc.bound_addr=ns;
    tc.tls_setting.ca_list_file=str(args["--ca"]);
    if(args.count("--cert")) tc.tls_setting.cert_file=str(args["--cert"]);
    if(args.count("--key")) tc.tls_setting.privkey_file=str(args["--key"]);
    tc.tls_setting.verify_server=PJ_TRUE;
    pj_sockaddr strict_local; check(pj_sockaddr_init(pj_AF_INET(),&strict_local,&ns,0),"strict_local");
    check(pjsip_tls_transport_start2(pjsua_get_pjsip_endpt(),&tc.tls_setting,&strict_local,
                                   nullptr,1,&strict_factory),"strict_factory");
    check(pjsua_tpfactory_register(strict_factory,&strict_transport),"strict_register");
    const std::string fault=args.count("--fail-at")?args["--fail-at"]:"";
    if(!fault.empty() && fault!="before-register" && fault!="after-register" && fault!="capacity") {
        pjsua_destroy2(PJSUA_DESTROY_NO_NETWORK); ai_phone_reset_tls_verifier(); return 2;
    }
    unsigned fillers=0;
    if(fault=="capacity") {
        for(;fillers<32;++fillers) {
            pjsua_transport_id tid=PJSUA_INVALID_ID;
            pj_status_t status=pjsua_transport_create(PJSIP_TRANSPORT_UDP,&tc,&tid);
            if(status==PJ_ETOOMANY) break;
            check(status,"capacity_filler");
        }
    }
    ai_phone_tls_test::hold_activation(true);
    for(unsigned i=0;i<policies.size();++i) {
        pj::TransportConfig guarded;
        guarded.fromPj(tc);
        ai_phone_tls_test::fail_at(i==0 && fault=="before-register"?1:
                                   i==0 && fault=="after-register"?2:0);
        const int id=ai_phone_create_verified_tls_transport(guarded,policies[i].expected_host);
        if(i==0 && !fault.empty()) {
            const int port=ai_phone_tls_test::last_port();
            const int registered=ai_phone_tls_test::last_register_status();
            const int rollback=ai_phone_tls_test::last_rollback_status();
            std::cout<<"{\"event\":\"fault_rollback\",\"phase\":\""<<fault
                     <<"\",\"port\":"<<port<<",\"register_status\":"<<registered
                     <<",\"rollback_status\":"<<rollback<<",\"fillers\":"<<fillers<<"}\n";
            ai_phone_deactivate_tls_verifier();
            const auto destroyed=pjsua_destroy2(PJSUA_DESTROY_NO_NETWORK);
            const bool reset=ai_phone_reset_tls_verifier();
            flush_events();
            std::cout<<"{\"event\":\"destroyed\",\"value\":"<<destroyed
                     <<",\"reset\":"<<(reset?"true":"false")<<"}\n";
            bool expected=fault=="capacity"?id==-PJ_ETOOMANY:id==-PJ_ECANCELLED;
            return expected && rollback==PJ_SUCCESS && destroyed==PJ_SUCCESS && reset?0:1;
        }
        if(id<0) check(-id,"guarded_factory");
        policies[i].transport=id;
    }
    observer.name=pj_str(const_cast<char*>("proof-application")); observer.id=-1;
    observer.priority=PJSIP_MOD_PRIORITY_APPLICATION-1; observer.on_rx_request=&receive_request;
    check(pjsip_endpt_register_module(pjsua_get_pjsip_endpt(),&observer),"observer");
    strict_account=account(strict_transport,host,peer_port);
    policies[0].account=account(policies[0].transport,host,peer_port);
    policies[1].account=account(policies[1].transport,host2,peer_port);
    wrong_hop_account=account(policies[0].transport,host2,peer_port);
    check(pjsua_start(),"start"); check(pjsua_set_null_snd_dev(),"null_audio");
    pjsua_transport_info si{},g0{},g1{};
    check(pjsua_transport_get_info(strict_transport,&si),"strict_info");
    check(pjsua_transport_get_info(policies[0].transport,&g0),"guarded_info");
    check(pjsua_transport_get_info(policies[1].transport,&g1),"guarded_info2");
    std::cout<<"{\"event\":\"ready\",\"strict_port\":"<<pj_sockaddr_get_port(&si.local_addr)
             <<",\"guarded_port\":"<<pj_sockaddr_get_port(&g0.local_addr)
             <<",\"guarded2_port\":"<<pj_sockaddr_get_port(&g1.local_addr)<<"}\n"; std::cout.flush();
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(120);
    bool running=true; std::string buffered;
    while(running && std::chrono::steady_clock::now()<deadline) {
        pjsua_handle_events(10); flush_events();
        pollfd fd{STDIN_FILENO,POLLIN,0};
        if(poll(&fd,1,0)>0 && (fd.revents&(POLLIN|POLLHUP))) {
            char b[1024]; auto n=read(STDIN_FILENO,b,sizeof(b));
            if(n<=0) break;
            buffered.append(b,size_t(n));
            for(size_t pos;(pos=buffered.find('\n'))!=std::string::npos;) {
                std::string line=buffered.substr(0,pos); buffered.erase(0,pos+1);
                if(line=="activate") { ai_phone_tls_test::activate(); event("activated",0,1); }
                else if(line=="deactivate") { ai_phone_deactivate_tls_verifier(); event("deactivated",-1,1); }
                else if(line=="shutdown") { running=false; break; }
                else if(line=="close guarded") {
                    ai_phone_tls_test::deactivate(0);
                    if(!policies[0].closed.exchange(true))
                        event("listener_closed",0,pjsua_transport_close(policies[0].transport,PJ_FALSE));
                }
                else {
                    bool unregister=line.rfind("unregister ",0)==0;
                    bool reg=line.rfind("register ",0)==0;
                    if(reg||unregister) {
                        auto name=line.substr(unregister?11:9);
                        pjsua_acc_id id=name=="strict"?strict_account:
                            name=="guarded"?policies[0].account:
                            name=="guarded2"?policies[1].account:
                            name=="wronghop"?wrong_hop_account:PJSUA_INVALID_ID;
                        if(id!=PJSUA_INVALID_ID) event("command",int(id),
                            pjsua_acc_set_registration(id,unregister?PJ_FALSE:PJ_TRUE));
                    }
                }
                flush_events();
            }
        }
    }
    ai_phone_deactivate_tls_verifier();
    pjsua_destroy2(PJSUA_DESTROY_NO_NETWORK);
    if(!ai_phone_reset_tls_verifier()) return 3;
    flush_events();
    // Production slot storage resets only after complete endpoint destruction.
    std::cout<<"{\"event\":\"destroyed\"}\n";
    return 0;
}

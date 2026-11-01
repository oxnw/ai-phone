#pragma once

// Versioned separately from TLS so an older importable binding is rebuilt.
int ai_phone_opus_version();

// Call on the registered main thread after libInit, before libStart/accounts.
// Returns PJ_SUCCESS or a PJ status; no codec is enabled by importing the module.
int ai_phone_enable_opus();

# KV-cache compatibility: compile-time totality for the provenance key.
# The external key can only be built from a COMPLETE, layered identity;
# dropping a field is a *compile* error (see missing_field.mojo).
# The RepresentationKey mirrors main's kvblock.mojo (shared surface).
# (Mojo nightly 2026-07: `fn` removed -> `def`; `alias` -> `comptime`; std.* )

@fieldwise_init
struct AdapterIdentity(Copyable, Movable):
    var content_hash: String     # stable source, NOT the name
    var generation: Int
    var activation_policy: String
    def parts(self) -> String:
        return self.content_hash + "|" + String(self.generation) + "|" + self.activation_policy

@fieldwise_init
struct SemanticKey(Copyable, Movable):
    var base_model_fp: String
    var adapter: AdapterIdentity
    var rope: String
    var reuse: String
    def digest(self) -> String:
        return "SEM(" + self.base_model_fp + ";" + self.adapter.parts() + ";" + self.rope + ";" + self.reuse + ")"

@fieldwise_init
struct RepresentationKey(Copyable, Movable):   # mirrors main/kvblock.mojo
    var kv_dtype: String
    var quant_codec: String
    var layout: String
    var tp_rank: Int
    def digest(self) -> String:
        return "REP(" + self.kv_dtype + ";" + self.quant_codec + ";" + self.layout + ";" + String(self.tp_rank) + ")"

@fieldwise_init
struct AccessKey(Copyable, Movable):
    var cache_salt: String
    def digest(self) -> String:
        return "ACC(" + self.cache_salt + ")"

@fieldwise_init
struct KVBlockIdentity(Copyable, Movable):
    var semantic: SemanticKey
    var representation: RepresentationKey
    var access: AccessKey

def derive_key(read id: KVBlockIdentity, parent: String, tokens: String) -> UInt64:
    # TOTAL: every layer participates; no subset path exists.
    var d = id.semantic.digest() + id.representation.digest() + id.access.digest() + parent + tokens
    return hash(d)

def main():
    var id_a = KVBlockIdentity(
        SemanticKey("sha256:base", AdapterIdentity("sha256:AAA", 1, "qkvo"), "rope:1e6", "exact"),
        RepresentationKey("bf16", "none", "paged", 0),
        AccessKey("shared-salt"))
    var id_b = KVBlockIdentity(
        SemanticKey("sha256:base", AdapterIdentity("sha256:BBB", 2, "qkvo"), "rope:1e6", "exact"),
        RepresentationKey("bf16", "none", "paged", 0),
        AccessKey("shared-salt"))
    var ka = derive_key(id_a, "genesis", "same-tokens")
    var kb = derive_key(id_b, "genesis", "same-tokens")
    print("key(adapter A) =", ka)
    print("key(adapter B) =", kb)
    if ka != kb:
        print("PASS: distinct adapters -> distinct keys (#44250 prevented)")
    else:
        print("FAIL: collision")

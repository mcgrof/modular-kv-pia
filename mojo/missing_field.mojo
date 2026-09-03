# Proof of "by construction": this OMITS the adapter provenance field.
# It MUST NOT COMPILE -- that is exactly the #44250 failure made un-writable.
@fieldwise_init
struct AdapterIdentity(Copyable, Movable):
    var content_hash: String
    var generation: Int
    var activation_policy: String

@fieldwise_init
struct SemanticKey(Copyable, Movable):
    var base_model_fp: String
    var adapter: AdapterIdentity   # required
    var rope: String
    var reuse: String

def main():
    # BUG: adapter dropped (the LMCache-MP #44250 mistake). Compile error.
    var s = SemanticKey("sha256:base", "rope:1e6", "exact")
    print("this line should be unreachable")

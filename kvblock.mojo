# Shared KV-block primitives in Mojo (branch: main).
# The KVRepresentationKey descriptor -- a compatibility identity layer on one
# side, the codec's format contract on the other. Kept on `main` so both the
# kv-compat and kv-asym-quant branches build on the same definition.
# (Mojo nightly 2026-07: `fn` removed -> all `def`; `alias` -> `comptime`;
#  stdlib namespace is `std.*`.)

@fieldwise_init
struct KVRepresentationKey(Copyable, Movable):
    var kv_dtype: String       # "bf16" | "fp16" | ...
    var quant_codec: String    # "none" | "k16v8" | "vfp8" | ...
    var scale_policy: String
    var page_size: Int
    var layout: String         # "paged" | "contiguous"
    var tp_rank: Int
    var tp_world: Int

    def digest(self) -> String:
        return ("REP(" + self.kv_dtype + ";" + self.quant_codec + ";"
                + self.scale_policy + ";" + String(self.page_size) + ";"
                + self.layout + ";" + String(self.tp_rank) + "/"
                + String(self.tp_world) + ")")

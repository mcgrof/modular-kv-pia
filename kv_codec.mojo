# Asymmetric KV quantization codec (branch: kv-asym-quant) -- SEPARATE from the
# compatibility work. K16 (bf16 kept) + V8 (int8, per-block scale). Driven by a
# KVRepresentationKey with quant_codec="k16v8" (see main/kvblock.mojo). CPU first
# cut; same source is the GPU-kernel starting point (env ~/envs/modular-max, W7900).
# (Mojo nightly 2026-07: `fn` removed -> `def`; `alias` -> `comptime`; std.* .)
from std.time import perf_counter_ns
from std.math import sin

comptime N = 1 << 20      # 1,048,576 V elements
comptime BLOCK = 128      # per-block quant scale (compile-time constant)

def main():
    var v = List[Float32]()
    for k in range(N):
        v.append(Float32(sin(Float64(k) * 0.01)))   # deterministic V in [-1,1]

    var q = List[Int8]()
    for _ in range(N):
        q.append(Int8(0))
    var nblocks = N // BLOCK
    var scales = List[Float32]()
    for _ in range(nblocks):
        scales.append(Float32(0))

    # ---- pack: quantize V to int8 with a per-block symmetric scale ----
    var t0 = perf_counter_ns()
    for b in range(nblocks):
        var base = b * BLOCK
        var amax = Float32(0)
        for j in range(BLOCK):
            var a = v[base + j]
            if a < 0:
                a = -a
            if a > amax:
                amax = a
        var scale = amax / 127.0
        if scale == 0.0:
            scale = 1.0
        scales[b] = scale
        var inv = 1.0 / scale
        for j in range(BLOCK):
            var x = v[base + j] * inv
            var r = Int(x + 0.5) if x >= 0 else Int(x - 0.5)
            if r > 127:
                r = 127
            if r < -127:
                r = -127
            q[base + j] = Int8(r)
    var t1 = perf_counter_ns()

    # ---- unpack: dequantize + measure round-trip error ----
    var maxerr = Float32(0)
    for b in range(nblocks):
        var base = b * BLOCK
        var scale = scales[b]
        for j in range(BLOCK):
            var deq = Float32(Int(q[base + j])) * scale
            var e = deq - v[base + j]
            if e < 0:
                e = -e
            if e > maxerr:
                maxerr = e

    var pack_ns = t1 - t0
    var melem_s = Float64(N) / (Float64(pack_ns) / 1e9) / 1e6
    print("K16/V8 asymmetric KV codec (CPU, single-source Mojo)")
    print("  N =", N, " block =", BLOCK, " (V: fp32->int8, 4x smaller)")
    print("  pack time (ms) =", Float64(pack_ns) / 1e6)
    print("  pack throughput (Melem/s) =", melem_s)
    print("  max abs dequant error =", maxerr)
    if maxerr < 0.02:
        print("  PASS: round-trip within int8 per-block quant bound")
    else:
        print("  CHECK: error too high:", maxerr)

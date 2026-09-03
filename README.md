# modular-kv / kv-asym-quant — Asymmetric KV quantization codec

> **Scope note (read this):** this branch is **NOT** the KV-cache *compatibility*
> work. It is a *different* concern — KV **compression/movement**: packing KV
> bytes smaller (K kept at 16-bit, V quantized to 8-bit) for cheaper storage and
> HBM↔CPU transfer. It is useful and worth keeping, but it was conflated with the
> compatibility contract once, so it lives on its own branch. The compatibility
> work is on `kv-compat`.

Builds on `main` only for the `KVRepresentationKey` descriptor: the codec is
*driven by* `quant_codec="k16v8"` in that descriptor (the descriptor says how the
bytes are encoded; this branch is the code that encodes/decodes them).

## Contents

- `kv_codec.mojo` — single-source asymmetric codec (K16 kept, V fp32→int8,
  per-block scale) with round-trip correctness + a throughput number. CPU first
  cut; the same source is the GPU-kernel starting point.

## Status / next

- CPU first cut works: ~560 Melem/s pack, round-trip max-abs error ~0.004.
- **Next:** port to a gfx1100 GPU kernel via `DeviceContext`, fuse
  paged↔contiguous gather, benchmark HBM↔CPU vs a naive copy+cast. Env:
  `~/envs/modular-max` on prune's W7900 (see the `modular` skill).

## Relationship to your other work

This overlaps the existing **asymmetric-kv** line (K16/V8, V-only FP8, LMCache
serde). Treat this branch as the MAX/Mojo-side codec experiment; fold results
back into that line rather than the compatibility work.

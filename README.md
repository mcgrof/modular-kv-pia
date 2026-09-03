# modular-kv

KV-block primitives for Modular MAX/Mojo, plus two **separate** use cases that
build on them. Split into three branches so the concerns don't get conflated
(they did once — see below).

## Branches

| Branch | What it is |
|---|---|
| `main` | **Shared base.** Generic KV-block primitives only: the canonical, cross-language key encoding and the `KVRepresentationKey` descriptor (how a block's bytes are laid out / encoded). Both use cases import this. Nothing use-case-specific lives here. |
| `kv-compat` | **KV-cache compatibility** (the primary goal). A provenance-complete, layered block *identity* contract so a cached KV block is only reused when it is truly compatible — making the vLLM #44250 LoRA "clash" class un-representable. Conformance suite + Mojo compile-time totality + the KVConnector-v1 provenance-envelope proposal. |
| `kv-asym-quant` | **Asymmetric KV quantization codec** (useful, but a *different* concern — KV *compression/movement*, not compatibility). K16/V8 pack/unpack. Kept on its own branch so it is not mistaken for the compatibility work. |

`main` is the generic trunk; `kv-compat` and `kv-asym-quant` each branch from it
and add only their own code. They share `main`'s descriptor + encoding.

## Why the split

The compatibility idea (KV blocks that know when they're safe to reuse) got
conflated with the asymmetric-quantization codec (KV bytes packed smaller). They
are different: one is **correctness/identity**, the other is
**performance/representation**. The only genuine shared surface is the
`KVRepresentationKey` descriptor — a compatibility *key layer* on one side, and
the thing the codec *consumes to know how to pack* on the other. That shared
surface lives on `main`; everything else is branch-specific.

## Running

- Python (`main`, `kv-compat`): stdlib only. `python3 conformance.py` on
  `kv-compat`.
- Mojo (all branches): needs the MAX/Mojo env on prune's W7900 —
  `~/envs/modular-max` (see the `modular` shared skill). GPU work targets
  gfx1100.

Design docs / writeups live in
`/data/knlp-key-results/modular-kv-provenance-contract-20260707/`.

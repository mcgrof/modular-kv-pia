#!/usr/bin/env python3
"""
Checks for the envelope (branch: kv-compat).

The encoder is the part that has to be right: a connector in another language
reproduces these bytes or the cache silently misses. Where `cbor2` is
installed the encoder is compared against it directly, which is the only test
that can catch a misreading of the specification rather than an inconsistency
with ourselves.

Run:  python3 test_envelope.py   (stdlib only; exit 0 iff all cases pass)
"""

from __future__ import annotations

import hashlib
import sys

from kv_identity import (
    AdapterIdentity, ReuseSemantics, KVSemanticKey, KVAccessKey, KVBlockIdentity,
)
from kvblock import KVRepresentationKey
from kv_envelope import (
    GENESIS_ROOT, SCHEMA_VERSION, KVBlockEnvelope, build_envelope, canonical_cbor,
    chain,
)

TOKENS = b"you are a helpful assistant ... (long prompt) ... answer:"
BASE_FP = "sha256:base-Qwen2.5-7B-Instruct"

ADAPTER_A = AdapterIdentity("my-lora", "sha256:AAA", 1, "qkvo")
ADAPTER_B = AdapterIdentity("my-lora", "sha256:BBB", 2, "qkvo")
ADAPTER_A_RENAMED = AdapterIdentity("renamed", "sha256:AAA", 1, "qkvo")


def rep(quant="none", dtype="bf16", tp_rank=0) -> KVRepresentationKey:
    # World size 2 so both ranks under test are legal; a rank must lie
    # inside its world, and rank 1 of a world of 1 does not exist.
    return KVRepresentationKey(dtype, quant, "per-tensor", 16, "paged",
                               "gqa:4kv/28q", tp_rank, 2, 1)


def ident(adapter, salt="shared-salt", quant="none", dtype="bf16", tp_rank=0):
    return KVBlockIdentity(
        KVSemanticKey(BASE_FP, adapter, "rope:1e6", "attn:full", "tok:qwen2", "",
                      ReuseSemantics.exact()),
        rep(quant, dtype, tp_rank),
        KVAccessKey(salt))


def key(adapter, **kw) -> bytes:
    return build_envelope(ident(adapter, **kw), GENESIS_ROOT, TOKENS).external_key


CASES = []


def case(name, got, note=""):
    CASES.append((name, bool(got), note))


def raises(fn) -> bool:
    try:
        fn()
        return False
    except Exception:
        return True


# --- the encoder ---------------------------------------------------------- #
case("shortest-form integers", canonical_cbor(0) == b"\x00"
     and canonical_cbor(23) == b"\x17"
     and canonical_cbor(24) == b"\x18\x18"
     and canonical_cbor(256) == b"\x19\x01\x00")
case("definite-length text and bytes",
     canonical_cbor("a") == b"\x61a" and canonical_cbor(b"\x01") == b"\x41\x01")
case("map keys sort by ENCODED bytes, not insertion order",
     canonical_cbor({"b": 1, "a": 2}) == canonical_cbor({"a": 2, "b": 1})
     == b"\xa2\x61a\x02\x61b\x01")
case("short keys sort before long ones (length-first ordering)",
     canonical_cbor({"aa": 1, "b": 2}) == b"\xa2\x61b\x02\x62aa\x01")
case("bool is refused rather than colliding with 1",
     raises(lambda: canonical_cbor(True)))

# --- the chain root ------------------------------------------------------- #
case("genesis root is a fixed value, not per-process",
     GENESIS_ROOT == hashlib.sha256(b"KVID/genesis/v1").digest())

# --- identity separation carries into the envelope ------------------------ #
case("distinct adapters -> distinct external keys (#44250)",
     key(ADAPTER_A) != key(ADAPTER_B))
case("same name, different content -> distinct keys (#30931/#42125)",
     key(ADAPTER_A) != key(ADAPTER_B))
case("rename with identical content -> same key (legal sharing preserved)",
     key(ADAPTER_A) == key(ADAPTER_A_RENAMED))
case("representation is folded in: bf16 vs k16v8 -> distinct",
     key(ADAPTER_A, quant="none") != key(ADAPTER_A, quant="k16v8"))
case("representation is folded in: tp_rank differs -> distinct",
     key(ADAPTER_A, tp_rank=0) != key(ADAPTER_A, tp_rank=1))
case("access is folded in: cache_salt differs -> distinct",
     key(ADAPTER_A, salt="t1") != key(ADAPTER_A, salt="t2"))
case("determinism: same inputs -> same key",
     key(ADAPTER_A) == key(ADAPTER_A))

# --- the envelope exposes representation but not identity ----------------- #
env = build_envelope(ident(ADAPTER_A), GENESIS_ROOT, TOKENS)
case("representation metadata is exposed for transport",
     env.representation["kv_dtype"] == "bf16" and env.representation["tp_rank"] == 0)
case("no semantic or access digest is exposed as a separate field",
     not any(k in env.representation for k in ("semantic", "access", "adapter")))
case("envelope serializes to canonical CBOR",
     isinstance(env.to_cbor(), bytes) and env.schema_version == SCHEMA_VERSION)

# --- chaining ------------------------------------------------------------- #
blocks = [b"block-0", b"block-1", b"block-2"]
c1 = chain(ident(ADAPTER_A), blocks)
c2 = chain(ident(ADAPTER_A), blocks)
c3 = chain(ident(ADAPTER_A), [b"block-0", b"CHANGED", b"block-2"])
case("chain is deterministic", [e.external_key for e in c1] == [e.external_key for e in c2])
case("a mid-prefix change propagates to every later block",
     c1[0].external_key == c3[0].external_key
     and c1[1].external_key != c3[1].external_key
     and c1[2].external_key != c3[2].external_key)

# --- cross-implementation agreement --------------------------------------- #
try:
    import cbor2
    samples = [
        0, 23, 24, 255, 256, 65536,
        b"", b"\x00\x01\x02", "", "hello", "unicode: é中",
        [1, 2, 3], {"a": 1, "b": "two"},
        {"schema_version": 1, "external_key": b"\xde\xad", "representation":
         {"kv_dtype": "bf16", "tp_rank": 0, "layout": "paged"}},
    ]
    mismatches = [s for s in samples
                  if canonical_cbor(s) != cbor2.dumps(s, canonical=True)]
    case(f"matches cbor2 canonical output on {len(samples)} samples",
         not mismatches,
         note="" if not mismatches else f"mismatched: {mismatches!r}")
except ImportError:
    CASES.append(("cbor2 cross-check", None, "SKIPPED: cbor2 not installed"))


def main() -> int:
    width = max(len(n) for n, _, _ in CASES)
    ran = [c for c in CASES if c[1] is not None]
    npass = sum(1 for _, ok, _ in ran if ok)
    print("KV block envelope checks")
    print("=" * (width + 14))
    for name, ok, note in CASES:
        tag = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
        line = f"  [{tag}] {name.ljust(width)}"
        if note:
            line += f"   # {note}"
        print(line)
    print("-" * (width + 14))
    print(f"  {npass}/{len(ran)} passed"
          + (f", {len(CASES) - len(ran)} skipped" if len(ran) != len(CASES) else ""))
    return 0 if npass == len(ran) else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
KV-cache compatibility conformance suite (branch: kv-compat).

Reproduces vLLM #44250 / #30931 / #42125 with lossy connector shims, then shows
the layered identity contract prevents them while preserving legal sharing.

Run:  python3 conformance.py   (stdlib only; exit 0 iff all cases pass)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from kvblock import KVRepresentationKey, enc, sha, s   # shared base
from kv_identity import (
    AdapterIdentity, ReuseSemantics, KVSemanticKey, KVAccessKey,
    KVBlockIdentity, derive_key, check_obligations,
)

TOKENS = b"you are a helpful assistant ... (long prompt) ... answer:"
PARENT = "genesis"
BASE_FP = "sha256:base-Qwen2.5-7B-Instruct"

ADAPTER_A = AdapterIdentity("my-lora", "sha256:AAA", 1, "qkvo")
ADAPTER_B = AdapterIdentity("my-lora", "sha256:BBB", 2, "qkvo")  # reload, diff weights
ADAPTER_A_RENAMED = AdapterIdentity("renamed", "sha256:AAA", 1, "qkvo")


def rep(quant="none", dtype="bf16", tp_rank=0) -> KVRepresentationKey:
    # World size 2 so both ranks under test are legal; a rank must lie
    # inside its world, and rank 1 of a world of 1 does not exist.
    return KVRepresentationKey(dtype, quant, "per-tensor", 16, "paged",
                              "gqa:4kv/28q", tp_rank, 2, 1)


def make_identity(adapter, salt="shared-salt", quant="none", dtype="bf16",
                  tp_rank=0) -> KVBlockIdentity:
    return KVBlockIdentity(
        KVSemanticKey(BASE_FP, adapter, "rope:1e6", "attn:full", "tok:qwen2",
                      "", ReuseSemantics.exact()),
        rep(quant, dtype, tp_rank),
        KVAccessKey(salt))


def correct_key(adapter, salt="shared-salt", quant="none", dtype="bf16", tp_rank=0):
    return derive_key(make_identity(adapter, salt, quant, dtype, tp_rank),
                      PARENT, TOKENS)


def lossy_subset_key(adapter, salt="shared-salt"):
    """LMCache-MP #44250: external key re-derived from a SUBSET, dropping adapter."""
    return sha(enc("lossy-subset", s(BASE_FP), s(salt), TOKENS))


def name_keyed_key(adapter, salt="shared-salt"):
    """#30931 / #42125: identity from the adapter LABEL, not a stable source."""
    return sha(enc("name-keyed", s(BASE_FP), s(adapter.name), s(salt), TOKENS))


CASES = []


def case(name, got, expected=True, note=""):
    CASES.append((name, bool(got) == bool(expected), note))


def raises(fn):
    try:
        fn()
        return False
    except Exception:
        return True


# 1. bug class reproduced, then prevented
case("[REPRO] #44250 lossy subset key COLLIDES across adapters",
     lossy_subset_key(ADAPTER_A) == lossy_subset_key(ADAPTER_B),
     note="two adapters -> same external key (the silent bug)")
case("#44250 PREVENTED: contract keys differ across adapters",
     correct_key(ADAPTER_A) != correct_key(ADAPTER_B))
case("[REPRO] #30931/#42125 name-keyed COLLIDES on same-name reload",
     name_keyed_key(ADAPTER_A) == name_keyed_key(ADAPTER_B))
case("#42125 PREVENTED: content+generation separate reloaded adapters",
     correct_key(ADAPTER_A) != correct_key(ADAPTER_B))

# 2. legal sharing preserved
case("legal sharing: identical inputs -> identical key",
     correct_key(ADAPTER_A) == correct_key(ADAPTER_A))
case("label is not identity: rename (same content) -> same key",
     correct_key(ADAPTER_A) == correct_key(ADAPTER_A_RENAMED))

# 3. layer isolation (the flat-key trap)
case("representation isolation: bf16 vs k16v8 -> different key",
     correct_key(ADAPTER_A, quant="none") != correct_key(ADAPTER_A, quant="k16v8"))
case("representation isolation: TP rank differs -> different key",
     correct_key(ADAPTER_A, tp_rank=0) != correct_key(ADAPTER_A, tp_rank=1))
case("access isolation: different cache_salt -> different key",
     correct_key(ADAPTER_A, salt="t1") != correct_key(ADAPTER_A, salt="t2"))

# 4. totality
case("totality: dropping the adapter is un-constructable",
     raises(lambda: KVBlockIdentity(
         KVSemanticKey(BASE_FP, None, "rope:1e6", "attn:full", "tok:qwen2", "",
                       ReuseSemantics.exact()),
         rep(), KVAccessKey("s"))))

# 5. obligation analysis
none_ident = make_identity(AdapterIdentity.none())
case("obligation: adapter active but identity none() -> FAIL",
     raises(lambda: check_obligations(none_ident, {"has_adapter": True})))
case("obligation: quant codec manifest/key mismatch -> FAIL",
     raises(lambda: check_obligations(make_identity(ADAPTER_A, quant="none"),
                                      {"quant_codec": "k16v8"})))
case("obligation: approximate reuse but key says exact -> FAIL",
     raises(lambda: check_obligations(make_identity(ADAPTER_A),
                                      {"approximate_reuse": True})))
case("obligation: fully-specified adapter identity -> PASS",
     not raises(lambda: check_obligations(
         make_identity(ADAPTER_A, quant="k16v8"),
         {"has_adapter": True, "quant_codec": "k16v8"})))


# 6. defects found by independent review (2026-09-04) -- kept as regressions
from dataclasses import replace, fields as _dc_fields  # noqa: E402


@dataclass(frozen=True)
class _ExtendedSemanticKey(KVSemanticKey):
    """A new required dimension, added the way a real one would be."""
    compression_policy_digest: str = ""


def _with_extended(policy: str) -> KVBlockIdentity:
    base = make_identity(ADAPTER_A)
    carried = {f.name: getattr(base.semantic, f.name)
               for f in _dc_fields(KVSemanticKey)}
    return replace(base, semantic=_ExtendedSemanticKey(
        **carried, compression_policy_digest=policy))


case("a new required field reaches the key without editing the digest",
     derive_key(_with_extended("policy-A"), PARENT, TOKENS)
     != derive_key(_with_extended("policy-B"), PARENT, TOKENS),
     note="hand-written digests returned the old value and collided")
case("an undeclared active feature is refused, not ignored",
     raises(lambda: check_obligations(
         make_identity(ADAPTER_A), {"adaptive_compression": "undeclared"})))
case("a rank outside its world is not constructable",
     raises(lambda: KVRepresentationKey("bf16", "none", "per-tensor", 16,
                                        "paged", "gqa:4kv/28q", 1, 1, 1)),
     note="rank 1 of world 1 names a shard that cannot exist")
case("a legal rank inside its world still constructs",
     not raises(lambda: KVRepresentationKey("bf16", "none", "per-tensor", 16,
                                            "paged", "gqa:4kv/28q", 1, 2, 1)))


def main() -> int:
    width = max(len(n) for n, _, _ in CASES)
    npass = sum(ok for _, ok, _ in CASES)
    print("KV-cache compatibility conformance suite")
    print("=" * (width + 12))
    for name, ok, note in CASES:
        line = f"  [{'PASS' if ok else 'FAIL'}] {name.ljust(width)}"
        if note:
            line += f"   # {note}"
        print(line)
    print("-" * (width + 12))
    print(f"  {npass}/{len(CASES)} passed")
    return 0 if npass == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())

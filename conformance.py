#!/usr/bin/env python3
"""
Reproduce the KV-cache reuse bug class (branch: kv-compat).

vLLM #44250 / #30931 / #42125 share one cause: a connector re-derives its own
external cache key from loose request fields, and the derivation silently drops
a field that made two contexts different. Each shim below is that derivation,
written the way the connector wrote it; each case asserts the collision it
produces -- two requests that must not share a cache entry hashing to one key.

This suite passes when the bugs reproduce. It is the "before" half of the
contract that follows, and it fails loudly if a reproduction ever stops
reproducing.

Run:  python3 conformance.py   (stdlib only; exit 0 iff all cases pass)
"""

from __future__ import annotations

import sys

from kvblock import enc, sha, s   # shared base

TOKENS = b"you are a helpful assistant ... (long prompt) ... answer:"
BASE_FP = "sha256:base-Qwen2.5-7B-Instruct"

# An adapter as a connector sees it today: loose request fields, with no type
# saying which of them are load-bearing for cache identity. Same name, different
# weights -- a reload, or two unrelated adapters that happen to share a label.
ADAPTER_A = {"name": "my-lora", "content_hash": "sha256:AAA", "generation": 1}
ADAPTER_B = {"name": "my-lora", "content_hash": "sha256:BBB", "generation": 2}


def lossy_subset_key(adapter, salt="shared-salt"):
    """LMCache-MP #44250: external key re-derived from a SUBSET, dropping adapter."""
    return sha(enc("lossy-subset", s(BASE_FP), s(salt), TOKENS))


def name_keyed_key(adapter, salt="shared-salt"):
    """#30931 / #42125: identity from the adapter LABEL, not a stable source."""
    return sha(enc("name-keyed", s(BASE_FP), s(adapter["name"]), s(salt), TOKENS))


CASES = []


def case(name, got, expected=True, note=""):
    CASES.append((name, bool(got) == bool(expected), note))


case("[REPRO] #44250 lossy subset key COLLIDES across adapters",
     lossy_subset_key(ADAPTER_A) == lossy_subset_key(ADAPTER_B),
     note="two adapters -> same external key (the silent bug)")
case("[REPRO] #30931/#42125 name-keyed COLLIDES on same-name reload",
     name_keyed_key(ADAPTER_A) == name_keyed_key(ADAPTER_B))


def main() -> int:
    width = max(len(n) for n, _, _ in CASES)
    npass = sum(ok for _, ok, _ in CASES)
    print("KV-cache reuse bug-class reproduction")
    print("=" * (width + 12))
    for name, ok, note in CASES:
        line = f"  [{'PASS' if ok else 'FAIL'}] {name.ljust(width)}"
        if note:
            line += f"   # {note}"
        print(line)
    print("-" * (width + 12))
    print(f"  {npass}/{len(CASES)} reproduced")
    return 0 if npass == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())

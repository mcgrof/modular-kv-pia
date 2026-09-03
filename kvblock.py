"""
Shared KV-block primitives (branch: main).

The ONLY code shared by both use cases:
  * a canonical, domain-separated, length-prefixed key encoding that is
    byte-identical across processes/languages (no Python-hash-seed dependence --
    the LMCache #2511 class of "keys silently disagree across processes" bug);
  * KVRepresentationKey -- the descriptor of how a block's bytes are laid
    out/encoded. On the compatibility side this is one identity layer; on the
    codec side it is the thing the packer/unpacker consumes to know the format.

Nothing use-case-specific belongs here. stdlib only.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


# --------------------------------------------------------------------------- #
# Canonical encoding (deterministic, cross-language).
# --------------------------------------------------------------------------- #

def lp(b: bytes) -> bytes:
    """Length-prefix (8-byte big-endian) so concatenation is unambiguous."""
    return len(b).to_bytes(8, "big") + b


def enc(domain: str, *parts: bytes) -> bytes:
    """Domain-separated canonical encoding of an ordered list of byte parts."""
    out = lp(b"KVID/" + domain.encode("utf-8"))
    for p in parts:
        out += lp(p)
    return out


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def s(x: str) -> bytes:
    return x.encode("utf-8")


def i(x: int) -> bytes:
    return int(x).to_bytes(16, "big", signed=True)


# --------------------------------------------------------------------------- #
# The shared descriptor.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class KVRepresentationKey:
    """How a block's bytes are interpreted. Shared surface between the two use
    cases: a compatibility identity layer AND the codec's format contract."""
    kv_dtype: str          # "bf16" | "fp16" | ...
    quant_codec: str       # "none" | "k16v8" | "vfp8" | ...
    scale_policy: str      # "per-tensor" | "per-block" | ...
    page_size: int
    layout: str            # "paged" | "contiguous" | ...
    head_geometry: str     # e.g. "gqa:4kv/28q"
    tp_rank: int
    tp_world: int
    wire_version: int

    def digest(self) -> str:
        return sha(enc(
            "representation",
            s(self.kv_dtype), s(self.quant_codec), s(self.scale_policy),
            i(self.page_size), s(self.layout), s(self.head_geometry),
            i(self.tp_rank), i(self.tp_world), i(self.wire_version),
        ))

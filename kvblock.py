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

    def __post_init__(self) -> None:
        """Reject descriptors that cannot describe a real block.

        Checking that a field is present is not the same as checking that its
        value is possible. Ranks are counted from zero within a world, so rank
        1 of a world of 1 names a shard that cannot exist; a descriptor able to
        hold it will happily key blocks nothing ever produced.
        """
        if self.tp_world < 1:
            raise ValueError(f"tp_world must be >= 1, got {self.tp_world}")
        if not 0 <= self.tp_rank < self.tp_world:
            raise ValueError(
                f"tp_rank must satisfy 0 <= rank < world, got rank "
                f"{self.tp_rank} of world {self.tp_world}")
        if self.page_size < 1:
            raise ValueError(f"page_size must be >= 1, got {self.page_size}")
        if self.wire_version < 1:
            raise ValueError(f"wire_version must be >= 1, got {self.wire_version}")
        for name in ("kv_dtype", "quant_codec", "scale_policy", "layout"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty descriptor")

    def digest(self) -> str:
        return sha(enc(
            "representation",
            s(self.kv_dtype), s(self.quant_codec), s(self.scale_policy),
            i(self.page_size), s(self.layout), s(self.head_geometry),
            i(self.tp_rank), i(self.tp_world), i(self.wire_version),
        ))

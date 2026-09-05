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
from dataclasses import dataclass, fields, is_dataclass
from typing import Any


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


# A field is part of a schema unless it explicitly opts out.  Keeping the
# exclusion beside the declaration makes an accidentally omitted new field
# move the digest in the safe direction.
IDENTITY_EXCLUDED = {"identity": False}


def _encode_value(value: Any) -> bytes:
    if isinstance(value, bool):
        # bool is an int subclass.  Refuse it instead of making True collide
        # with the integer 1 in an identity schema.
        raise TypeError("bool is not an identity field type")
    if isinstance(value, int):
        return enc("integer", i(value))
    if isinstance(value, str):
        return enc("string", s(value))
    if isinstance(value, bytes):
        return enc("bytes", value)
    if isinstance(value, (tuple, list)):
        return enc("sequence", *(_encode_value(item) for item in value))
    if is_dataclass(value):
        return encode_fields(value)
    raise TypeError(f"identity field type is not encodable: {type(value).__name__}")


def encode_fields(obj: Any) -> bytes:
    """Encode every identity-bearing dataclass field, recursively.

    Field names, types, order, and values all participate.  A newly declared
    field therefore changes the bytes without requiring a second hand-written
    field list to be kept in sync.
    """
    if not is_dataclass(obj):
        raise TypeError("encode_fields requires a dataclass instance")
    parts: list[bytes] = []
    for descriptor in fields(obj):
        if not descriptor.metadata.get("identity", True):
            continue
        parts.append(s(descriptor.name))
        parts.append(_encode_value(getattr(obj, descriptor.name)))
    return enc("fields", *parts)


def encode_schema(name: str, version: int, obj: Any) -> bytes:
    """Version and encode a complete dataclass schema."""
    if version < 1:
        raise ValueError(f"schema version must be >= 1, got {version}")
    return enc("schema", s(name), i(version), encode_fields(obj))


# --------------------------------------------------------------------------- #
# The shared descriptor.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class KVScaleMetadata:
    """Resolved scale schema for one component, not data-dependent values."""

    policy: str
    dtype: str
    axis: str
    shape: tuple[int, ...]
    storage_format: str

    def __post_init__(self) -> None:
        if not self.policy:
            raise ValueError("scale policy must be non-empty")
        if any(dimension < 0 for dimension in self.shape):
            raise ValueError(f"scale shape dimensions must be >= 0: {self.shape}")
        if self.policy != "none":
            if not self.shape:
                raise ValueError("scaled data requires a non-empty scale shape")
            for name in ("dtype", "axis", "storage_format"):
                if not getattr(self, name):
                    raise ValueError(
                        f"scaled data requires non-empty scale {name}")

    @staticmethod
    def none() -> "KVScaleMetadata":
        return KVScaleMetadata("none", "", "", (), "")


@dataclass(frozen=True)
class KVComponentFormat:
    """The independently resolved representation of K or V."""

    dtype: str
    codec: str
    packing: str
    byte_order: str
    scale: KVScaleMetadata

    def __post_init__(self) -> None:
        for name in ("dtype", "codec", "packing", "byte_order"):
            if not getattr(self, name):
                raise ValueError(f"component {name} must be non-empty")
        if self.codec == "none" and self.scale.policy != "none":
            raise ValueError("an unencoded component cannot declare scale metadata")

    @staticmethod
    def plain(dtype: str) -> "KVComponentFormat":
        return KVComponentFormat(dtype, "none", "native", "little",
                                 KVScaleMetadata.none())


@dataclass(frozen=True)
class KVShard:
    """Global layer/head ownership assigned to one worker."""

    worker_id: int
    layer_start: int
    layer_stop: int
    kv_head_start: int
    kv_head_stop: int

    def __post_init__(self) -> None:
        if self.worker_id < 0:
            raise ValueError("shard worker_id must be >= 0")
        if not 0 <= self.layer_start < self.layer_stop:
            raise ValueError("a shard must own a non-empty layer range")
        if not 0 <= self.kv_head_start < self.kv_head_stop:
            raise ValueError("a shard must own a non-empty KV-head range")


@dataclass(frozen=True)
class KVPartitionMap:
    """Resolved worker and ownership map, not only rank/world labels."""

    world_size: int
    worker_id: int
    shards: tuple[KVShard, ...]

    def __post_init__(self) -> None:
        if self.world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {self.world_size}")
        if not 0 <= self.worker_id < self.world_size:
            raise ValueError(
                "worker_id must satisfy 0 <= id < world_size, got "
                f"{self.worker_id} of {self.world_size}")
        if not self.shards:
            raise ValueError("partition map must contain at least one shard")
        if any(shard.worker_id >= self.world_size for shard in self.shards):
            raise ValueError("partition map names a worker outside its world")
        if self.worker_id not in {shard.worker_id for shard in self.shards}:
            raise ValueError("partition map has no shard for the active worker")
        canonical = tuple(sorted(
            self.shards,
            key=lambda shard: (
                shard.worker_id,
                shard.layer_start,
                shard.layer_stop,
                shard.kv_head_start,
                shard.kv_head_stop,
            ),
        ))
        if canonical != self.shards:
            raise ValueError("partition shards must be in canonical order")

    @staticmethod
    def uniform(world_size: int, worker_id: int, *, layers: int = 1,
                kv_heads: int = 1) -> "KVPartitionMap":
        if layers < 1 or kv_heads < 1:
            raise ValueError("uniform partition dimensions must be >= 1")
        return KVPartitionMap(
            world_size,
            worker_id,
            tuple(KVShard(rank, 0, layers, rank * kv_heads,
                          (rank + 1) * kv_heads)
                  for rank in range(world_size)),
        )


@dataclass(frozen=True)
class KVRepresentationKey:
    """How a block's K and V bytes are independently interpreted."""

    k: KVComponentFormat
    v: KVComponentFormat
    page_size: int
    layout: str
    head_geometry: str
    partition: KVPartitionMap
    wire_version: int

    SCHEMA_VERSION = 2

    def __post_init__(self) -> None:
        """Reject descriptors that cannot describe a real block.

        Checking that a field is present is not the same as checking that its
        value is possible. Ranks are counted from zero within a world, so rank
        1 of a world of 1 names a shard that cannot exist; a descriptor able to
        hold it will happily key blocks nothing ever produced.
        """
        if self.page_size < 1:
            raise ValueError(f"page_size must be >= 1, got {self.page_size}")
        if self.wire_version < 1:
            raise ValueError(f"wire_version must be >= 1, got {self.wire_version}")
        for name in ("layout", "head_geometry"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty descriptor")

    def digest(self) -> str:
        return sha(encode_schema("representation", self.SCHEMA_VERSION, self))

    @property
    def kv_dtype(self) -> str:
        return self.k.dtype if self.k.dtype == self.v.dtype else (
            f"k:{self.k.dtype}/v:{self.v.dtype}")

    @property
    def quant_codec(self) -> str:
        return self.k.codec if self.k.codec == self.v.codec else (
            f"k:{self.k.codec}/v:{self.v.codec}")

    @property
    def scale_policy(self) -> str:
        return self.k.scale.policy if self.k.scale.policy == self.v.scale.policy else (
            f"k:{self.k.scale.policy}/v:{self.v.scale.policy}")

    @property
    def tp_rank(self) -> int:
        return self.partition.worker_id

    @property
    def tp_world(self) -> int:
        return self.partition.world_size

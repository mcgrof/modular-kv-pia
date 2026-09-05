"""
The envelope that carries a completed block identity across a process or
language boundary (branch: kv-compat).

`kv_identity.py` makes a lossy key impossible to construct in this process.
That guarantee has to survive the trip to a connector written in another
language, which means the key must be reducible to bytes that any producer can
reproduce exactly. Two properties do that work.

The encoding is canonical CBOR. CBOR (RFC 8949) is a binary serialization
format whose deterministic profile fixes one encoding per value: integers take
their shortest form, lengths are definite, and map keys are sorted by their
encoded bytes. Two implementations that follow it emit identical bytes for
identical values, so a Rust or C++ producer agrees with this one without
sharing code. `canonical_cbor()` implements the subset the envelope needs and
is checked against the `cbor2` library where that library is installed.

The chain root is a contract constant, so a key computed on one host means the
same thing on another without relying on engine defaults or process-local
state.  Current vLLM cryptographic hashing also has a shareable default root;
the earlier unconditional random-root claim was stale and is not relied on.

stdlib only.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from kv_identity import KVBlockIdentity

SCHEMA_VERSION = 1

# Deterministic by construction: the root of the parent chain is a fixed
# digest, never a per-process random value.
GENESIS_ROOT = hashlib.sha256(b"KVID/genesis/v1").digest()


# --------------------------------------------------------------------------- #
# Canonical CBOR (RFC 8949 deterministic encoding), for the value kinds the
# envelope uses: unsigned ints, byte strings, text strings, lists and maps.
# --------------------------------------------------------------------------- #

def _head(major: int, length: int) -> bytes:
    """Type byte plus the shortest length encoding that fits."""
    if length < 24:
        return bytes([(major << 5) | length])
    for extra, marker in ((1, 24), (2, 25), (4, 26), (8, 27)):
        if length < (1 << (8 * extra)):
            return bytes([(major << 5) | marker]) + length.to_bytes(extra, "big")
    raise ValueError(f"length too large to encode: {length}")


def canonical_cbor(value) -> bytes:
    """Encode a value in CBOR's deterministic profile."""
    if isinstance(value, bool):
        # Guard: bool is an int subclass, and encoding True as 1 would make
        # True and 1 collide.
        raise TypeError("bool is not encodable here; use an int or a string")
    if isinstance(value, int):
        if value < 0:
            return _head(1, -1 - value)
        return _head(0, value)
    if isinstance(value, bytes):
        return _head(2, len(value)) + value
    if isinstance(value, str):
        raw = value.encode("utf-8")
        return _head(3, len(raw)) + raw
    if isinstance(value, (list, tuple)):
        return _head(4, len(value)) + b"".join(canonical_cbor(v) for v in value)
    if isinstance(value, dict):
        items = [(canonical_cbor(k), canonical_cbor(v)) for k, v in value.items()]
        items.sort(key=lambda kv: kv[0])   # sort by encoded key, not by Python value
        return _head(5, len(items)) + b"".join(k + v for k, v in items)
    raise TypeError(f"not encodable in canonical CBOR: {type(value).__name__}")


def sha256_cbor(value) -> bytes:
    """Digest a value through its canonical encoding."""
    return hashlib.sha256(canonical_cbor(value)).digest()


# --------------------------------------------------------------------------- #
# The envelope.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class KVBlockEnvelope:
    """What the engine hands a connector in place of the materials to build a key.

    `external_key` is opaque and complete: the semantic and access digests are
    folded into it rather than exposed, so there is no field a connector could
    drop or selectively re-hash. `representation` is exposed because a
    connector legitimately needs it to move and transform bytes -- converting
    layout, packing a dtype, selecting a codec -- but never to decide whether
    two blocks are the same block.
    """

    external_key: bytes
    representation: dict
    schema_version: int = SCHEMA_VERSION

    def to_cbor(self) -> bytes:
        return canonical_cbor({
            "schema_version": self.schema_version,
            "external_key": self.external_key,
            "representation": self.representation,
        })

    @staticmethod
    def from_cbor(_raw: bytes) -> "KVBlockEnvelope":
        raise NotImplementedError(
            "decoding is a consumer-side concern and no consumer exists yet; "
            "connectors treat external_key as opaque bytes"
        )

    def hex(self) -> str:
        return self.external_key.hex()


def representation_fields(identity: KVBlockIdentity) -> dict:
    """The transport-visible half of the identity, as plain values."""
    r = identity.representation
    def component_fields(component) -> dict:
        return {
            "dtype": component.dtype,
            "codec": component.codec,
            "packing": component.packing,
            "byte_order": component.byte_order,
            "scale": {
                "policy": component.scale.policy,
                "dtype": component.scale.dtype,
                "axis": component.scale.axis,
                "shape": list(component.scale.shape),
                "storage_format": component.scale.storage_format,
            },
        }

    return {
        "k": component_fields(r.k),
        "v": component_fields(r.v),
        "page_size": r.page_size,
        "layout": r.layout,
        "head_geometry": r.head_geometry,
        "partition": {
            "world_size": r.partition.world_size,
            "worker_id": r.partition.worker_id,
            "shards": [
                {
                    "worker_id": shard.worker_id,
                    "layer_start": shard.layer_start,
                    "layer_stop": shard.layer_stop,
                    "kv_head_start": shard.kv_head_start,
                    "kv_head_stop": shard.kv_head_stop,
                }
                for shard in r.partition.shards
            ],
        },
        "wire_version": r.wire_version,
    }


def build_envelope(identity: KVBlockIdentity, parent_key: bytes,
                   block_tokens: bytes) -> KVBlockEnvelope:
    """Derive a complete external key and wrap it with its representation.

    Every layer participates, as in `kv_identity.derive_key()`; the difference
    is the wire encoding and the fixed chain root, which are what let a
    producer in another language agree byte for byte.
    """
    if not isinstance(identity, KVBlockIdentity):
        raise TypeError("build_envelope requires a fully-formed KVBlockIdentity")
    if not isinstance(parent_key, bytes):
        raise TypeError("parent_key must be bytes; use GENESIS_ROOT for the first block")

    digests = identity.layer_digests()
    key = sha256_cbor({
        "v": SCHEMA_VERSION,
        "semantic": digests["semantic"],
        "representation": digests["representation"],
        "access": digests["access"],
        "parent": parent_key,
        "tokens": block_tokens,
    })
    return KVBlockEnvelope(external_key=key,
                           representation=representation_fields(identity))


def chain(identity: KVBlockIdentity, block_token_list: list[bytes],
          root: bytes = GENESIS_ROOT) -> list[KVBlockEnvelope]:
    """Envelope a whole prefix, each block's key covering its parent's."""
    out: list[KVBlockEnvelope] = []
    parent = root
    for tokens in block_token_list:
        env = build_envelope(identity, parent, tokens)
        out.append(env)
        parent = env.external_key
    return out

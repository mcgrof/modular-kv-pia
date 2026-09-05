"""
KV-cache COMPATIBILITY: a provenance-complete, layered block identity so a
cached KV block is reused only when it is truly compatible with a request.
Makes the vLLM #44250 LoRA "clash" class un-representable. (branch: kv-compat)

Builds on the shared base (`kvblock.py`): reuses the canonical encoding and the
KVRepresentationKey descriptor. Adds the semantic + access layers, the total
key derivation, and the manifest obligation check.

Rules (see docs/ + the design writeup):
  * Three SEPARATE layers, never one flat hash: Semantic / Representation
    (shared, from kvblock) / Access.
  * derive_key() is TOTAL: no external key can be built from a subset -- the
    #44250 failure (connector drops the LoRA dimension) becomes impossible.
  * Identity from STABLE sources (content hash + generation), not names
    (#30931 / #42125).
stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

from kvblock import KVRepresentationKey, enc, sha, s, i  # shared base


# Marks a field as deliberately outside the identity. Anything without it is
# part of the digest, so forgetting the annotation is the safe direction.
IDENTITY_EXCLUDED = {"identity": False}


# --------------------------------------------------------------------------- #
# Adapter identity -- the crux of #44250 / #30931 / #42125.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class AdapterIdentity:
    """`name` is a LABEL, not identity. Identity = (content_hash, generation)."""
    # Excluded from the digest on purpose: an adapter renamed with unchanged
    # weights is the same adapter and must keep its cache. The exclusion is
    # declared on the field so it is visible where the field is, and so that
    # the default for anything added later is to be included.
    name: str = field(metadata=IDENTITY_EXCLUDED)
    content_hash: str    # sha256 of adapter weights -- the STABLE source
    generation: int      # immutable counter, bumped on every (re)load
    activation_policy: str

    @staticmethod
    def none() -> "AdapterIdentity":
        return AdapterIdentity("", "", 0, "none")

    @property
    def is_none(self) -> bool:
        return (self.content_hash == "" and self.generation == 0
                and self.activation_policy == "none")



@dataclass(frozen=True)
class ReuseSemantics:
    mode: str                 # "exact" | "approximate"
    policy_digest: str = ""
    recompute_mask: str = ""
    commit_policy: str = ""

    def __post_init__(self) -> None:
        if self.mode not in ("exact", "approximate"):
            raise ValueError(f"ReuseSemantics.mode invalid: {self.mode!r}")
        if self.mode == "approximate" and not self.policy_digest:
            raise ValueError("approximate reuse requires a policy_digest")

    @staticmethod
    def exact() -> "ReuseSemantics":
        return ReuseSemantics("exact")



@dataclass(frozen=True)
class KVSemanticKey:
    """Fields that change the K/V *values*."""
    base_model_fingerprint: str   # content hash, NOT model name
    adapter: AdapterIdentity      # REQUIRED -- AdapterIdentity.none() explicitly
    rope_semantics: str
    attention_semantics: str
    tokenizer_hash: str
    mm_preprocess_hash: str
    reuse: ReuseSemantics

    def digest(self) -> str:
        return sha(enc("semantic", encode_fields(self)))


@dataclass(frozen=True)
class KVAccessKey:
    cache_salt: str
    tenant: str = ""
    label_isolation: str = ""

    def digest(self) -> str:
        return sha(enc("access", encode_fields(self)))


def _encode_value(value: Any) -> bytes:
    if isinstance(value, bool):
        # bool is an int subclass; encoding True as 1 would collide with the
        # integer, so refuse it rather than silently merge the two.
        raise TypeError("bool is not an identity field type")
    if isinstance(value, int):
        return i(value)
    if isinstance(value, str):
        return s(value)
    if is_dataclass(value):
        return encode_fields(value)
    raise TypeError(f"identity field type is not encodable: {type(value).__name__}")


def encode_fields(obj: Any) -> bytes:
    """Encode every declared field of a dataclass, recursively.

    Deriving the encoding from the field list rather than writing it out by
    hand is what makes a new field impossible to forget. A hand-written digest
    keeps returning the old value when a field is added, so two identities that
    differ only in the new dimension collide -- silently, and in exactly the
    way this whole contract exists to prevent. Field names are encoded
    alongside values so that renaming or reordering also changes the digest.

    A field may be held outside the identity by tagging it with
    IDENTITY_EXCLUDED, which keeps the decision next to the field instead
    of inside a digest function nobody rereads.
    """
    parts: list[bytes] = []
    for f in fields(obj):
        if not f.metadata.get("identity", True):
            continue
        parts.append(s(f.name))
        parts.append(_encode_value(getattr(obj, f.name)))
    return enc("fields", *parts)


def _require_no_none(obj: Any, path: str = "") -> None:
    if is_dataclass(obj):
        for f in fields(obj):
            _require_no_none(getattr(obj, f.name), f"{path}.{f.name}")
    elif obj is None:
        raise ValueError(f"KV identity field is None: {path or '<root>'}")


@dataclass(frozen=True)
class KVBlockIdentity:
    semantic: KVSemanticKey
    representation: KVRepresentationKey   # shared base type
    access: KVAccessKey

    def __post_init__(self) -> None:
        _require_no_none(self)

    def layer_digests(self) -> dict:
        return {"semantic": self.semantic.digest(),
                "representation": self.representation.digest(),
                "access": self.access.digest()}


def derive_key(identity: KVBlockIdentity, parent_key: str,
               block_token_bytes: bytes) -> str:
    """TOTAL external-key derivation -- no subset path (the #44250 failure)."""
    if not isinstance(identity, KVBlockIdentity):
        raise TypeError("derive_key requires a fully-formed KVBlockIdentity")
    d = identity.layer_digests()
    return sha(enc("block/v1", s(d["semantic"]), s(d["representation"]),
                   s(d["access"]), s(parent_key), block_token_bytes))


KNOWN_OBLIGATIONS = frozenset({
    "has_adapter", "quant_codec", "multimodal", "approximate_reuse",
})


def check_obligations(identity: KVBlockIdentity, manifest: dict) -> None:
    """Manifest boundary: assert active dimensions are reflected in the key.

    An unrecognised entry is refused rather than ignored. A checker that skips
    what it does not understand reports success for exactly the case it was
    built to catch: a feature has been switched on, it may well change the
    cached values, and nothing here knows whether the key accounts for it.
    Silence there is indistinguishable from a pass, so a new feature reaches
    production with no obligation attached to it at all.
    """
    unknown = set(manifest) - KNOWN_OBLIGATIONS
    if unknown:
        raise ValueError(
            "OBLIGATION: manifest declares features this checker does not "
            f"know how to verify: {sorted(unknown)}. Add a check, or the key "
            "cannot be claimed complete for them.")

    sem = identity.semantic
    if manifest.get("has_adapter") and sem.adapter.is_none:
        raise ValueError("OBLIGATION: adapter active but identity is none() (#44250)")
    if manifest.get("has_adapter") and not sem.adapter.content_hash:
        raise ValueError("OBLIGATION: adapter active but no content_hash (#42125)")
    if manifest.get("quant_codec", "none") != identity.representation.quant_codec:
        raise ValueError("OBLIGATION: quant_codec manifest/key mismatch")
    if manifest.get("multimodal") and not sem.mm_preprocess_hash:
        raise ValueError("OBLIGATION: multimodal active but no mm_preprocess_hash")
    if manifest.get("approximate_reuse") and sem.reuse.mode != "approximate":
        raise ValueError("OBLIGATION: approximate reuse active but key says exact")

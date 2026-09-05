"""Artifact-bound registration and immutable request descriptors.

Registration consumes the file descriptors the model loader actually opened;
it never fingerprints a path and then asks a different operation to load that
path.  The registry retains duplicate descriptors as lifecycle seals, so an
in-place mutation invalidates the capability before another block key can be
issued.

The wire identity is content-derived and independent of registration order or
local lifecycle counters.  The opaque handle is only a process-local capability
for reaching the resolved contract.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import stat
import threading
from dataclasses import dataclass, field
from typing import BinaryIO, Iterable, Mapping

from kvblock import (
    IDENTITY_EXCLUDED,
    KVRepresentationKey,
    enc,
    encode_schema,
    sha,
)
from kv_identity import (
    AdapterIdentity,
    KVAccessKey,
    KVBlockIdentity,
    KVSemanticKey,
    ReuseSemantics,
    check_obligations,
)


class StaleRegistrationError(RuntimeError):
    """The artifacts behind a registration no longer match its identity."""


@dataclass(frozen=True, order=True)
class ArtifactIdentity:
    """Portable identity of one regular file opened by the loader."""

    content_digest: str
    size: int


@dataclass(frozen=True)
class ArtifactSet:
    """An order-independent multiset of artifact contents for one role."""

    role: str
    members: tuple[ArtifactIdentity, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.role:
            raise ValueError("artifact role must be non-empty")
        if tuple(sorted(self.members)) != self.members:
            raise ValueError("artifact members must be in canonical order")

    @property
    def is_empty(self) -> bool:
        return not self.members

    def digest(self) -> str:
        return sha(encode_schema("artifact-set", self.schema_version, self))


@dataclass(frozen=True)
class ResolvedKVContract:
    """What this engine instance actually loaded and will produce."""

    base_model: ArtifactSet
    adapter: ArtifactSet
    adapter_activation_policy: str
    rope_semantics: str
    attention_semantics: str
    tokenizer_hash: str
    mm_preprocess_hash: str
    representation: KVRepresentationKey
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.base_model.role != "base-model" or self.base_model.is_empty:
            raise ValueError("resolved contract requires loaded base-model artifacts")
        if self.adapter.role != "adapter":
            raise ValueError("adapter artifact set has the wrong role")
        if self.adapter.is_empty != (self.adapter_activation_policy == "none"):
            raise ValueError("adapter artifacts and adapter activation policy disagree")
        for name in ("rope_semantics", "attention_semantics", "tokenizer_hash"):
            if not getattr(self, name):
                raise ValueError(f"resolved contract requires {name}")
        if not isinstance(self.representation, KVRepresentationKey):
            raise TypeError("resolved contract requires a KVRepresentationKey")

    def digest(self) -> str:
        return sha(encode_schema("resolved-kv-contract", self.schema_version, self))


@dataclass(frozen=True)
class RegistrationHandle:
    """Opaque, immutable process-local capability returned by registration."""

    _contract_digest: str
    _registry_nonce: bytes = field(repr=False, metadata=IDENTITY_EXCLUDED)
    _slot: bytes = field(repr=False, metadata=IDENTITY_EXCLUDED)
    _generation: int = field(repr=False, metadata=IDENTITY_EXCLUDED)

    @property
    def contract_digest(self) -> str:
        """Stable public identity; the capability bytes remain private."""
        return self._contract_digest


@dataclass(frozen=True)
class EffectiveInputs:
    """The effective model input and the rules used to interpret it."""

    kind: str
    content_digest: str
    interpretation_digest: str

    def __post_init__(self) -> None:
        for name in ("kind", "content_digest", "interpretation_digest"):
            if not getattr(self, name):
                raise ValueError(f"effective inputs require {name}")

    @staticmethod
    def tokens(token_bytes: bytes, token_mapping_digest: str) -> "EffectiveInputs":
        if not isinstance(token_bytes, bytes):
            raise TypeError("token_bytes must be bytes")
        if not token_mapping_digest:
            raise ValueError("token inputs require a resolved token mapping digest")
        return EffectiveInputs(
            "tokens",
            hashlib.sha256(enc("effective-tokens", token_bytes)).hexdigest(),
            token_mapping_digest,
        )


@dataclass(frozen=True)
class RequestDescriptor:
    """Immutable request state bound to one registered loaded contract."""

    handle: RegistrationHandle
    effective_inputs: EffectiveInputs
    access: KVAccessKey
    reuse: ReuseSemantics
    schema_version: int = 1

    def digest(self) -> str:
        # RegistrationHandle's schema includes only its stable contract digest;
        # the local capability and generation explicitly opt out.
        return sha(encode_schema("request-descriptor", self.schema_version, self))

    def block_identity(self, registry: "ContractRegistry") -> KVBlockIdentity:
        return registry.block_identity(self)

    def issue_envelope(
        self, registry: "ContractRegistry", parent_key: bytes, block_tokens: bytes
    ):
        # Lazy import keeps the contract/identity/envelope dependency acyclic.
        from kv_envelope import build_envelope

        return build_envelope(self.block_identity(registry), parent_key, block_tokens)


def _stat_signature(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _identity_from_fd(fd: int) -> tuple[ArtifactIdentity, tuple[int, ...]]:
    before = os.fstat(fd)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("loaded artifacts must be regular files")

    digest = hashlib.sha256()
    offset = 0
    while True:
        chunk = os.pread(fd, 1024 * 1024, offset)
        if not chunk:
            break
        digest.update(chunk)
        offset += len(chunk)

    after = os.fstat(fd)
    before_signature = _stat_signature(before)
    after_signature = _stat_signature(after)
    if before_signature != after_signature or offset != after.st_size:
        raise StaleRegistrationError("artifact changed while it was fingerprinted")
    return ArtifactIdentity(digest.hexdigest(), offset), after_signature


@dataclass
class _ArtifactSeal:
    fd: int
    identity: ArtifactIdentity
    stat_signature: tuple[int, ...]

    def validate(self) -> None:
        signature = _stat_signature(os.fstat(self.fd))
        if signature == self.stat_signature:
            return
        current, signature = _identity_from_fd(self.fd)
        if current != self.identity:
            raise StaleRegistrationError(
                "a loaded artifact changed after contract registration"
            )
        # Metadata-only changes do not change the artifact the loader read.
        # Remember the new seal so normal per-block validation remains a
        # constant-time fstat rather than rehashing model weights.
        self.stat_signature = signature

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


@dataclass
class _Registration:
    contract: ResolvedKVContract
    seals: tuple[_ArtifactSeal, ...]
    invalid_reason: str = ""


class ContractRegistry:
    """Registers loaded artifacts and validates handles at every key issue."""

    def __init__(self) -> None:
        self._nonce = secrets.token_bytes(32)
        self._entries: dict[bytes, _Registration] = {}
        self._next_generation = 1
        self._lock = threading.RLock()

    @staticmethod
    def _capture(
        streams: Iterable[BinaryIO], role: str
    ) -> tuple[ArtifactSet, tuple[_ArtifactSeal, ...]]:
        seals: list[_ArtifactSeal] = []
        try:
            for stream in streams:
                try:
                    source_fd = stream.fileno()
                except (AttributeError, OSError) as error:
                    raise TypeError(
                        "registration requires the open files used by the loader, "
                        "not paths or operator-written digests"
                    ) from error
                retained_fd = os.dup(source_fd)
                try:
                    identity, signature = _identity_from_fd(retained_fd)
                except Exception:
                    os.close(retained_fd)
                    raise
                seals.append(_ArtifactSeal(retained_fd, identity, signature))
        except Exception:
            for seal in seals:
                seal.close()
            raise

        members = tuple(sorted(seal.identity for seal in seals))
        return ArtifactSet(role, members), tuple(seals)

    def register(
        self,
        *,
        base_artifacts: Iterable[BinaryIO],
        adapter_artifacts: Iterable[BinaryIO] = (),
        adapter_activation_policy: str = "none",
        rope_semantics: str,
        attention_semantics: str,
        tokenizer_hash: str,
        mm_preprocess_hash: str = "",
        representation: KVRepresentationKey,
    ) -> RegistrationHandle:
        """Bind a capability to the exact regular files the loader opened."""
        base, base_seals = self._capture(base_artifacts, "base-model")
        try:
            adapter, adapter_seals = self._capture(adapter_artifacts, "adapter")
            contract = ResolvedKVContract(
                base,
                adapter,
                adapter_activation_policy,
                rope_semantics,
                attention_semantics,
                tokenizer_hash,
                mm_preprocess_hash,
                representation,
            )
        except Exception:
            for seal in base_seals:
                seal.close()
            if "adapter_seals" in locals():
                for seal in adapter_seals:
                    seal.close()
            raise

        with self._lock:
            slot = secrets.token_bytes(32)
            generation = self._next_generation
            self._next_generation += 1
            handle = RegistrationHandle(
                contract.digest(), self._nonce, slot, generation
            )
            self._entries[slot] = _Registration(contract, base_seals + adapter_seals)
            return handle

    def _entry(self, handle: RegistrationHandle) -> _Registration:
        if not isinstance(handle, RegistrationHandle):
            raise TypeError("a RegistrationHandle is required")
        if handle._registry_nonce != self._nonce:
            raise ValueError("registration handle belongs to another registry")
        entry = self._entries.get(handle._slot)
        if entry is None:
            raise StaleRegistrationError("registration handle is closed or unknown")
        if entry.invalid_reason:
            raise StaleRegistrationError(entry.invalid_reason)
        if entry.contract.digest() != handle.contract_digest:
            raise StaleRegistrationError("registration handle contract mismatch")
        return entry

    def validate(self, handle: RegistrationHandle) -> None:
        with self._lock:
            entry = self._entry(handle)
            try:
                for seal in entry.seals:
                    seal.validate()
            except StaleRegistrationError as error:
                entry.invalid_reason = str(error)
                for seal in entry.seals:
                    seal.close()
                raise

    def bind_request(
        self,
        handle: RegistrationHandle,
        *,
        effective_inputs: EffectiveInputs,
        access: KVAccessKey,
        reuse: ReuseSemantics,
        active_features: Mapping[str, object] | None = None,
    ) -> RequestDescriptor:
        """Validate a handle and freeze all request-scoped identity inputs."""
        with self._lock:
            self.validate(handle)
            descriptor = RequestDescriptor(
                handle=handle,
                effective_inputs=effective_inputs,
                access=access,
                reuse=reuse,
            )
            identity = self.block_identity(descriptor)
            entry = self._entry(handle)
            manifest = dict(active_features or {})
            manifest.update(
                {
                    "has_adapter": not entry.contract.adapter.is_empty,
                    "quant_codec": entry.contract.representation.quant_codec,
                    "multimodal": effective_inputs.kind == "multimodal",
                    "approximate_reuse": reuse.mode == "approximate",
                }
            )
            check_obligations(identity, manifest)
            return descriptor

    def block_identity(self, descriptor: RequestDescriptor) -> KVBlockIdentity:
        """Issue a layered identity only while the loaded artifacts stay valid."""
        if not isinstance(descriptor, RequestDescriptor):
            raise TypeError("a RequestDescriptor is required")
        with self._lock:
            self.validate(descriptor.handle)
            entry = self._entry(descriptor.handle)
            contract = entry.contract
            adapter = (
                AdapterIdentity.none()
                if contract.adapter.is_empty
                else AdapterIdentity(
                    "",
                    contract.adapter.digest(),
                    descriptor.handle._generation,
                    contract.adapter_activation_policy,
                )
            )
            semantic = KVSemanticKey(
                contract.base_model.digest(),
                adapter,
                contract.rope_semantics,
                contract.attention_semantics,
                contract.tokenizer_hash,
                contract.mm_preprocess_hash,
                descriptor.reuse,
            )
            return KVBlockIdentity(
                semantic,
                contract.representation,
                descriptor.access,
                descriptor.digest(),
            )

    def close(self, handle: RegistrationHandle) -> None:
        with self._lock:
            if not isinstance(handle, RegistrationHandle):
                raise TypeError("a RegistrationHandle is required")
            if handle._registry_nonce != self._nonce:
                raise ValueError("registration handle belongs to another registry")
            entry = self._entries.pop(handle._slot, None)
            if entry is None:
                raise StaleRegistrationError("registration handle is closed or unknown")
            for seal in entry.seals:
                seal.close()

    def close_all(self) -> None:
        with self._lock:
            for entry in self._entries.values():
                for seal in entry.seals:
                    seal.close()
            self._entries.clear()

    def __enter__(self) -> "ContractRegistry":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close_all()

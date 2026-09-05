#!/usr/bin/env python3
"""Acceptance tests for artifact-bound contract registration."""

from __future__ import annotations

import dataclasses
import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from pathlib import Path

from kv_contract import (
    ContractRegistry,
    EffectiveInputs,
    RequestDescriptor,
    StaleRegistrationError,
)
from kv_envelope import GENESIS_ROOT
from kv_identity import KVAccessKey, ReuseSemantics
from kvblock import (
    KVComponentFormat,
    KVPartitionMap,
    KVRepresentationKey,
    KVScaleMetadata,
)


def representation(
    *, worker_id: int = 0, scale_format: str = "dense-f32-le"
) -> KVRepresentationKey:
    k_format = KVComponentFormat.plain("bf16")
    v_format = KVComponentFormat(
        "int8",
        "symmetric-int8",
        "head-major",
        "little",
        KVScaleMetadata("per-head", "fp32", "kv-head", (4,), scale_format),
    )
    return KVRepresentationKey(
        k_format,
        v_format,
        16,
        "paged-nhd",
        "gqa:4kv/28q",
        KVPartitionMap.uniform(2, worker_id, layers=28, kv_heads=4),
        1,
    )


@dataclass(frozen=True)
class ExtendedRequestDescriptor(RequestDescriptor):
    required_dimension: str = field(kw_only=True)


class ContractRegistrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.base_a = root / "model-a.bin"
        self.base_b = root / "model-b.bin"
        self.adapter_a = root / "adapter-a.bin"
        self.adapter_b = root / "adapter-b.bin"
        self.base_a.write_bytes(b"base shard A")
        self.base_b.write_bytes(b"base shard B")
        self.adapter_a.write_bytes(b"adapter shard A")
        self.adapter_b.write_bytes(b"adapter shard B")
        self.registry = ContractRegistry()

    def tearDown(self) -> None:
        self.registry.close_all()
        self.temp.cleanup()

    def register(self, base_paths=None, adapter_paths=None, registry=None, **kwargs):
        if base_paths is None:
            base_paths = [self.base_a, self.base_b]
        if adapter_paths is None:
            adapter_paths = [self.adapter_a, self.adapter_b]
        target_registry = registry or self.registry
        with ExitStack() as stack:
            base_files = [stack.enter_context(path.open("rb")) for path in base_paths]
            adapter_files = [
                stack.enter_context(path.open("rb")) for path in adapter_paths
            ]
            adapter_policy = kwargs.pop(
                "adapter_activation_policy",
                "qkvo" if adapter_files else "none",
            )
            return target_registry.register(
                base_artifacts=base_files,
                adapter_artifacts=adapter_files,
                adapter_activation_policy=adapter_policy,
                rope_semantics="rope:1e6",
                attention_semantics="causal-full",
                tokenizer_hash="sha256:token-map",
                representation=kwargs.pop("representation", representation()),
                **kwargs,
            )

    def descriptor(self, handle, descriptor_type=RequestDescriptor, **extra):
        common = dict(
            handle=handle,
            effective_inputs=EffectiveInputs.tokens(
                b"the effective token sequence", "sha256:token-map"
            ),
            access=KVAccessKey("shared-salt", "tenant-a", ""),
            reuse=ReuseSemantics.exact(),
        )
        common.update(extra)
        return descriptor_type(**common)

    def envelope_key(self, descriptor, registry=None):
        return descriptor.issue_envelope(
            registry or self.registry, GENESIS_ROOT, b"one token block"
        ).external_key

    def test_registration_is_order_independent_across_workers(self) -> None:
        first = self.register()
        with ContractRegistry() as worker_two:
            second = self.register(
                base_paths=[self.base_b, self.base_a],
                adapter_paths=[self.adapter_b, self.adapter_a],
                registry=worker_two,
            )
            self.assertEqual(first.contract_digest, second.contract_digest)
            self.assertEqual(
                self.envelope_key(self.descriptor(first)),
                self.envelope_key(self.descriptor(second), worker_two),
            )

    def test_in_place_mutation_invalidates_the_handle(self) -> None:
        handle = self.register()
        descriptor = self.descriptor(handle)
        self.envelope_key(descriptor)  # positive control before mutation
        self.adapter_a.write_bytes(b"adapter shard MUTATED")
        with self.assertRaises(StaleRegistrationError):
            self.envelope_key(descriptor)
        with self.assertRaises(StaleRegistrationError):
            self.registry.validate(handle)
        self.registry.close(handle)

    def test_same_path_with_changed_content_gets_new_identity(self) -> None:
        before = self.register()
        self.adapter_a.write_bytes(b"new content at the same path")
        after = self.register()
        self.assertNotEqual(before.contract_digest, after.contract_digest)

    def test_paths_and_operator_digests_are_not_registration_inputs(self) -> None:
        with self.assertRaises(TypeError):
            self.registry.register(
                base_artifacts=[self.base_a],
                rope_semantics="rope:1e6",
                attention_semantics="causal-full",
                tokenizer_hash="sha256:token-map",
                representation=representation(),
            )

    def test_handles_and_request_descriptors_are_immutable(self) -> None:
        handle = self.register()
        descriptor = self.descriptor(handle)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            handle._contract_digest = "forged"
        with self.assertRaises(dataclasses.FrozenInstanceError):
            descriptor.access = KVAccessKey("other")

    def test_handle_is_a_registry_local_capability(self) -> None:
        handle = self.register()
        with ContractRegistry() as other:
            with self.assertRaises(ValueError):
                other.validate(handle)

    def test_no_adapter_is_explicit_in_the_resolved_identity(self) -> None:
        handle = self.register(adapter_paths=[])
        descriptor = self.registry.bind_request(
            handle,
            effective_inputs=EffectiveInputs.tokens(b"tokens", "sha256:token-map"),
            access=KVAccessKey("shared-salt"),
            reuse=ReuseSemantics.exact(),
        )
        self.assertTrue(
            descriptor.block_identity(self.registry).semantic.adapter.is_none
        )

    def test_new_required_descriptor_field_changes_the_key(self) -> None:
        handle = self.register()
        left = self.descriptor(
            handle, ExtendedRequestDescriptor, required_dimension="A"
        )
        right = replace(left, required_dimension="B")
        self.assertNotEqual(left.digest(), right.digest())
        self.assertNotEqual(self.envelope_key(left), self.envelope_key(right))

    def test_schema_version_changes_the_key(self) -> None:
        handle = self.register()
        first = self.descriptor(handle)
        second = replace(first, schema_version=2)
        self.assertNotEqual(self.envelope_key(first), self.envelope_key(second))

    def test_k_and_v_formats_are_independent_and_transport_visible(self) -> None:
        handle = self.register()
        envelope = self.descriptor(handle).issue_envelope(
            self.registry, GENESIS_ROOT, b"one token block"
        )
        self.assertEqual(envelope.representation["k"]["dtype"], "bf16")
        self.assertEqual(envelope.representation["k"]["codec"], "none")
        self.assertEqual(envelope.representation["v"]["dtype"], "int8")
        self.assertEqual(
            envelope.representation["v"]["scale"]["storage_format"],
            "dense-f32-le",
        )

    def test_scale_metadata_changes_the_address(self) -> None:
        first = self.register(
            representation=representation(scale_format="dense-f32-le")
        )
        second = self.register(
            representation=representation(scale_format="packed-f32-le")
        )
        self.assertNotEqual(
            self.envelope_key(self.descriptor(first)),
            self.envelope_key(self.descriptor(second)),
        )

    def test_partition_map_changes_the_address(self) -> None:
        first = self.register(representation=representation(worker_id=0))
        second = self.register(representation=representation(worker_id=1))
        self.assertNotEqual(
            self.envelope_key(self.descriptor(first)),
            self.envelope_key(self.descriptor(second)),
        )

    def test_unknown_active_feature_is_still_refused(self) -> None:
        handle = self.register()
        with self.assertRaisesRegex(ValueError, "does not know how to verify"):
            self.registry.bind_request(
                handle,
                effective_inputs=EffectiveInputs.tokens(b"tokens", "sha256:token-map"),
                access=KVAccessKey("shared-salt"),
                reuse=ReuseSemantics.exact(),
                active_features={"adaptive_compression": "enabled"},
            )

    def test_bind_request_positive_control(self) -> None:
        handle = self.register()
        descriptor = self.registry.bind_request(
            handle,
            effective_inputs=EffectiveInputs.tokens(b"tokens", "sha256:token-map"),
            access=KVAccessKey("shared-salt"),
            reuse=ReuseSemantics.exact(),
        )
        self.assertEqual(descriptor.handle, handle)
        self.envelope_key(descriptor)


if __name__ == "__main__":
    unittest.main(verbosity=2)

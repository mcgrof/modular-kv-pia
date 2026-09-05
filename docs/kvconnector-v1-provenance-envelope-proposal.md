# A provenance envelope at the vLLM KVConnector boundary

**Status: proposal, with the diagnosis and the fix demonstrated on real code.**
Every claim about vLLM below was read from `upstream/main` at commit
`bb4b448f98` on 2026-09-03. The failure and its repair were then reproduced
against that tree's actual `lmcache_mp_connector`: unpatched, three requests
differing only by adapter hand the external cache one identity; with an
87-line change deriving envelope keys, they separate. Evidence, harness and
patch are in
`/data/knlp-key-results/modular-kv-provenance-contract-20260707/connector-evidence-20260903/`.

**What is still not built:** LMCache does not consume the keys — the vLLM side
computes and carries them, and making the far side address entries by them is a
second change. Nothing was served and there is no performance number. Line
numbers below are a convenience and will drift; symbol names are the durable
reference.

**Corrections after independent review (2026-09-04):** this document records a
historical reproduction, not the current baseline. Current vLLM cryptographic
hashing uses a fixed shareable default root; the unconditional random-root claim
below is withdrawn. LMCache already partitions its composite key by dtype,
world size, and worker id; the still-open representation dimensions are codec,
scale policy, layout, and page size. Adapter separation below is a conservative
safety policy, not a claim that every K/V element changes.

## 1. Background

A serving engine caches the attention state of a transformer — the per-token Key
and Value tensors, the "KV cache" — so a later request sharing a prefix can reuse
that work instead of recomputing it. A **connector** is the plug-in that moves
those cached blocks between the engine and some external store: another process,
a CPU-memory tier, an NVMe tier, or a remote cache such as LMCache. vLLM's
plug-in interface for this is the KVConnector v1 API.

Everything a connector stores must be retrievable by a key, and that key decides
correctness rather than merely hit rate. If two genuinely different contexts
produce one key, a request is served another request's attention state: wrong
output, and a cross-user leak if the two requests belong to different people.

The dimension that keeps causing this is the **LoRA adapter** — a small set of
fine-tuning weight deltas applied on top of a base model. An adapter can change
the attention computation and resulting cached state while changing nothing
about the prompt text. A conservative contract therefore does not share entries
between different adapters.

## 2. What vLLM already gets right, and the one place it does not

The engine's own block identity covers four dimensions.
`generate_block_hash_extra_keys()` in `vllm/v1/core/kv_cache_utils.py` folds the
LoRA adapter, multimodal input hashes, the request's `cache_salt`, and
prompt-embedding hashes into each block hash. Blocks chain — each hash covers
its parent — so a divergence anywhere in a prefix propagates to every block
after it.

The adapter dimension is the weak one, and this is worth stating precisely
because it changes what the envelope is for. `_gen_lora_extra_hash_keys()`
returns `[request.lora_request.lora_name]` — the adapter's **name**, a mutable
label — while `lora_int_id` and `lora_path` sit unused on the same object. Two
different adapters loaded under one name therefore hash identically, and so
does a reload carrying new weights. That is #30931 and #42125, present in the
engine itself rather than only in connectors. Measured: request C in the
harness, a reload of `lora-alpha` with a different path and id, produces block
hashes byte-identical to the original in every run.

The encoding is canonical rather than incidental. `BlockHash` is
`NewType("BlockHash", bytes)`, produced by one of the functions in
`vllm/utils/hashing.py`; the CBOR variants (`sha256_cbor`, `xxhash_cbor`) encode
their input with `cbor2.dumps(input, canonical=True)` before digesting. CBOR is
a binary serialization format with a defined canonical form, so two independent
implementations encoding the same values produce the same bytes. That rules out
the ambiguity a naive string concatenation carries, and it does not depend on
Python's randomized `hash()`.

One subtlety worth keeping in view: `cache_salt` is folded only at
`start_token_idx == 0`, so the salt enters the first block and reaches later
blocks through the parent chain rather than being re-folded per block.

## 3. The failure, with the receipt

The API withholds nothing. `get_num_new_matched_tokens(request, ...)`,
`update_state_after_alloc(request, ...)` and `request_finished(request, ...)` all
receive the whole `Request`, and `Request.block_hashes` is a populated list of
those complete hashes.

The connector implicated in #44250 holds them and keys on something else.
In `lmcache_mp_connector.py`, the request tracker captures
`self.block_hashes = ConstantList(request.block_hashes)` (:225) — and then uses
that list only to count blocks (`len(tracker.block_hashes)`, :338 and :400).
The operations that actually store and load are built as
`LoadStoreOp(token_ids=token_ids, block_ids=..., start=..., end=...)` (:353,
:419) with `cache_salt` carried alongside on the metadata (:363, :430). The
adapter dimension appears nowhere in either. Two users with identical prompts
and different adapters therefore produce identical lookup state, which is
#44250 exactly.

Driving that code confirms it. Three requests sharing a 512-token prompt and a
salt, under adapters `lora-alpha`, `lora-beta`, and a reload of `lora-alpha`,
all emit the same external identity — one SHA-256 over the token ids, the
range and the salt — while vLLM's own hashes for the first two differ. The
engine separated them; the connector did not.

The sibling reports are the same mistake at different levels. #30931 and #42125
are the name-as-identity failure described in §2, which the engine still has.
LMCache #2511 is the failure one level down again: two processes hashed
identical content and disagreed.

So the diagnosis is not "the engine fails to compute provenance" and not "the
API fails to pass it along". Both already happen. The defect is that deriving
the external key is left to each connector as an unconstrained local decision,
and the cheapest thing to reach for — the token ids that are right there — is
lossy. Every fix so far has added the one missing field to one connector, which
leaves the next connector free to make the same choice again.

## 4. Why "just use `request.block_hashes`" is close, but not sufficient

It is most of the answer, and any connector doing it today would be better off
than one keying on token ids. Three gaps stop it from being the whole answer,
and they matter specifically because an *external* cache outlives and is shared
beyond the process that wrote it.

**The adapter is identified by a mutable label**, as §2 sets out, so inheriting
the engine's hash inherits #30931 and #42125 along with it. The semantic layer
in `kv_identity.py` uses loaded content as portable identity and keeps the local
lifecycle generation out of cross-worker keys. It is therefore stronger than a
mutable label without confusing process-local reload order with content.

**Historical root result, withdrawn for the current baseline.** The September 3
tree produced different roots in the recorded three-process probe. Current
vLLM's `resolve_none_hash_seed()` gives cryptographic hash functions a fixed,
shareable default when `PYTHONHASHSEED` is unset. The envelope keeps its own
versioned constant, but root instability is no longer a claimed contribution.

**Representation coverage is incomplete, not absent.** LMCache's composite
`CacheEngineKey` already includes dtype, world size, and worker id. The gap to
demonstrate is codec, scale policy, layout, and page size at the composite-key
and reader-validation boundary. The representation schema keeps K and V formats
and scale metadata separate and carries the resolved partition map.

## 5. Proposal

Have the engine hand the connector a completed key rather than the materials to
build one.

```
KVBlockEnvelope (wire v1, canonical CBOR):
  external_key   : bytes   # H(semantic, representation, access, parent, tokens)
                           # opaque and complete; connectors key on this verbatim
  representation :         # non-secret metadata a transport may act on:
      k: {dtype, codec, packing, byte_order, scale}
      v: {dtype, codec, packing, byte_order, scale}
      page_size, layout, head_geometry, partition_map, wire_version
  schema_version : u16
```

`external_key` derives from a deterministic root rather than `NONE_HASH`, and
folds the representation layer that the internal hash omits. The semantic and
access digests fold *into* it and are not exposed as separate fields, so there
is nothing a connector could selectively re-hash — the field it might drop does
not exist at its level. Representation metadata *is* exposed, because a
connector legitimately needs it to move and transform bytes: converting paged to
contiguous layout, packing a dtype, choosing a codec.

The rule that goes with it is short: a connector stores and looks up by
`external_key` verbatim, and never derives identity from token ranges. It reads
`representation` only to decide how to move bytes, never to decide whether two
blocks are the same block.

## 6. This needs no signature break

An earlier draft assumed the API had to grow a parallel `envelopes` list on five
methods. Reading the signatures shows otherwise: the connector already receives
the entire `Request`, so the envelope can be attached to the request or computed
by an engine-side helper the connector calls, with no change to any existing
signature and no compatibility flag. That removes the largest adoption
objection. The remaining work at each call site is a substitution rather than a
plumbing change — match against `external_key` in `get_num_new_matched_tokens`,
store it alongside the block in `update_state_after_alloc`, address external
storage by it in `save_kv_layer` and `start_load_kv`, evict by it in
`request_finished`.

## 7. Demonstrated on the real connector

The change is 87 added lines against `lmcache_mp_connector.py`, with no
deletions: the tracker captures adapter identity, `LMCacheMPRequestMetadata`
gains an `external_keys` field, and the store path derives one complete key per
block. With it applied, the three requests that previously shared one identity
separate — the two different adapters, and the reload the engine itself could
not tell apart.

Two caveats belong with that result. The adapter's content hash is proxied by
`lora_path` and `lora_int_id`, because vLLM computes no hash of adapter
weights; a reload from an unchanged path with changed weights would still
collide, so closing #42125 properly requires the engine to supply a real
content hash. And the representation layer is fixed in the patch rather than
read from the running configuration, so dtype, codec and rank separation is
covered by unit checks rather than by the connector run.

## 8. The conformance test is the part with teeth

A rule that only a maintainer's memory enforces will be broken by the next
connector. Ship a connector-agnostic suite that any connector must pass to
advertise itself as provenance-safe. It should assert that two adapters with
identical tokens and salt produce distinct keys with no cross-adapter hit; that
same-name reloads with different content are distinct; that a differing
`tp_rank`, `kv_dtype` or `quant_codec` is distinct; that everything identical
still shares, so legal reuse is preserved; and that a connector re-deriving from
`(tokens, base, salt)` while an envelope is available *fails*. `conformance.py`
on this branch is the shape of that suite, written against local shims.

## 9. Open questions

Which encoder should be standardized on — `sha256_cbor` is already in-tree and
already used for block hashing, and `xxhash_cbor` sits beside it, so the choice
is between an existing pair rather than a new dependency.

Can the deterministic root be introduced without invalidating existing caches,
or does it need a `schema_version` bump and a migration window?

Can representation metadata ride existing block-hash extra keys instead of a
parallel structure, reducing the API delta further? The extra-keys mechanism
already carries tuples of arbitrary values, so this may be a smaller change than
a new envelope type.

Does folding representation into the key over-partition any cache that is
legitimately shared today — for example between ranks whose blocks really are
interchangeable?

## 10. Internal notes — not part of any upstream text

These are our submission constraints and must not travel into an upstream RFC or
PR description.

The surfaces this touches are ones Modular does not control: vLLM and LMCache
are Python, NIXL and KVBM are Rust and C++. The envelope must therefore be a
language-neutral wire schema rather than a MAX or Mojo type. Modular can ship a
reference producer and the conformance suite and can propose the API change, but
cannot mandate adoption.

vLLM's contribution rules require that a human submit the PR and that AI
assistance be disclosed. Luis submits, AI-disclosed. Keep any upstream PR scoped
to the envelope and the conformance test, and keep private research results out
of it entirely.

Sequencing: land the reference envelope and suite here; prototype against one
real connector, the LMCache path, so the #44250 reproduction flips from a wrong
hit to a correct miss on actual connector code; then propose the API change
upstream; and only then argue for carrying the envelope across engines.

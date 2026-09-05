# modular-kv — KV-cache reuse that cannot cross contexts

*(branch `2026-09-04-feature-resolved-kv-contract`; the shared trunk is
`main`, the codec work is `kv-asym-quant`)*

A serving engine caches the attention state of a transformer — the per-token Key
and Value tensors, the "KV cache" — so that a later request sharing a prefix can
reuse that work instead of recomputing it. Reuse is a lookup by key. That makes
the key load-bearing in a way most caches are not: if two genuinely different
contexts ever encode to the same key, one user is served another user's
attention state. The output is silently wrong, and if the two users are
different people, it is a data leak.

This private branch prototypes a block identity and demonstrates the narrower
guarantees at their actual enforcement points: construction in Python and
required-argument checking in Mojo.

## The bug this exists to end

vLLM issue #44250. Two users share a base model and prompt text but run
different LoRA adapters — small fine-tuning weight deltas that can change the
attention computation and resulting cached state. Conservative separation is a
safety policy, not a claim that every element changes. vLLM's internal block
hash accounted for the adapter by its mutable name. The connector that exports
blocks to an external cache did not: it re-derived its own key from a subset of
the request — tokens, model, cache salt — so both adapters mapped onto one key.

Two siblings in the same family. Issue #30931 keyed on the adapter's *name*, a
mutable label, so two different adapters sharing a name collided. Issue #42125
could not distinguish an adapter from a reload of that adapter carrying new
weights, so stale blocks survived the reload. LMCache issue #2511 is the same
shape one level down: two processes hashed identical content and disagreed,
because Python randomizes `hash()` per process, so the cache silently missed.

Each was fixed by adding the missing field. The bug *class* survived every one
of those fixes, because nothing in the design stops the next field from being
dropped by the next connector. Ending the class means making a lossy key
impossible to write, not adding a fifth field.

## The contract

**Identity is layered.** A `KVBlockIdentity` is three keys, never one flat
bag of fields:

| Layer | Holds | Answers |
|---|---|---|
| Semantic | base-model fingerprint, adapter identity, RoPE and attention semantics, tokenizer hash, multimodal preprocessing hash, reuse semantics | Does this block hold the same *numbers*? |
| Representation | dtype, quantization codec, scale policy, page size, layout, head geometry, tensor-parallel rank and world, wire version | Are the *bytes* encoded the same way? |
| Access | cache salt, tenant, label isolation | Is this block *allowed* to be shared here? |

The separation is what keeps the three concerns from contaminating each other. A
block requantized from bf16 to a smaller codec, or produced by a different
tensor-parallel rank, holds the same semantic content in different bytes: the
representation layer changes the key without touching semantics. Tenancy policy
changes what may be shared without implying anything about the numbers.
`Representation` is the descriptor from `main`, because the codec branch needs
the identical structure to know how to pack and unpack a block.

**Key derivation is total.** `derive_key()` accepts a fully-formed
`KVBlockIdentity` and nothing else. There is no code path that builds an
external key from a subset, which is precisely the path #44250 took. "This
request has no adapter" is spelled explicitly as `AdapterIdentity.none()`, never
as an omitted field, so the absence of an adapter is itself part of the key.

**Identity currently uses content plus a generation counter, not a name.** A
rename therefore preserves legal sharing, while changed content separates.
The generation is process-local, however, so it cannot be portable identity
across workers. Binding identity to what the loader actually opened and keeping
lifecycle invalidation local are the next implementation boundary.

A total key over a *dishonest* identity would still be wrong, so
`check_obligations()` closes the last gap by cross-checking the identity against
what the engine says is active: an adapter running while the identity claims
none, a codec mismatch, a multimodal request with no preprocessing hash,
approximate reuse undeclared in the key.

## What is demonstrated, and where it ran

`conformance.py` is the argument in executable form, and it runs on the Python
standard library alone:

```
python3 conformance.py      # exit 0 iff every check passes
python3 test_envelope.py    # the envelope and its encoder
```

It rebuilds each historical bug as a deliberately lossy key function and shows
those keys colliding, so the suite fails loudly if the reproduction itself ever
stops reproducing. It then shows the contract's keys not colliding on the same
inputs. The remaining checks push in the opposite direction, which matters just
as much: identical requests must still share, including across a pure rename;
each layer in isolation must change the key; and dropping the adapter field must
be rejected rather than quietly hashed as `None`.

The Mojo mirror moves the same guarantee from runtime to compile time. Mojo is
Modular's compiled language; it is built on MLIR, the compiler framework that
also underpins the MAX inference stack, where a model is represented as a graph
of tensor operations and lowered stage by stage toward GPU code with structural
rules checked at every stage. A cache key can be held to that same standard.
`mojo/kv_identity.mojo` declares the adapter as a required field of the semantic
key, so a program that omits it is not a program. `mojo/missing_field.mojo`
commits the #44250 mistake deliberately, and the toolchain refuses it:

```
$ mojo run mojo/missing_field.mojo
missing_field.mojo:18:24: error: no matching function in initialization
    var s = SemanticKey("sha256:base", "rope:1e6", "exact")
missing_field.mojo:10:1: note: candidate not viable: missing required argument: 'reuse'
error: failed to parse the provided Mojo source module
```

That error is the whole point. Inside the engine's own toolchain the bug is not
caught late by a test, a review, or a customer — the program does not build.

Both Mojo files were run on prune under Mojo 1.0.0b3.dev2026070706
(`~/envs/modular-max`, see the `modular` skill): `kv_identity.mojo` prints
distinct keys for two adapters, `missing_field.mojo` fails as quoted. They are
pure identity logic and use no GPU.

## What is not built yet

The guarantee still has to leave this repository, and a Mojo type cannot travel
to the systems that need it — vLLM and LMCache are Python, NIXL is Rust and C++.
`kv_envelope.py` is that crossing: it reduces a complete identity to an opaque
external key plus the representation metadata a transport legitimately needs,
encoded as canonical CBOR so a producer written in another language reproduces
the same bytes, and rooted at a contract constant. `test_envelope.py` checks it
against the `cbor2` library where that is
installed, which is what catches a misreading of the format rather than mere
agreement with ourselves.

**What is not done is the far side.** The envelope has been driven through a
historical vLLM LMCache multi-process connector reproduction, but LMCache does
not yet consume those keys, nothing was served, and there is no performance
number. Current vLLM cryptographic hashing has a shareable default root, so the
older random-root claim is withdrawn. The open milestone is one real external
store/load path with positive controls separating external reuse from the
engine's internal prefix cache.

Two known gaps in the Mojo side, so nobody discovers them the hard way. It is a
demonstrator, not the production producer: it derives its digests by string
concatenation under the builtin `hash()`, which is neither stable across
processes nor the canonical SHA-256 encoding that `main` defines, so it proves
the totality property rather than the wire format. And its representation key is
a local four-field subset rather than the seven-field descriptor in
`main`'s `kvblock.mojo`; the two should be unified before either is trusted as
the shared surface.

## Files

`kv_identity.py` holds the layered identity, the total derivation, and the
obligation check, building on `main`'s `kvblock.py` for the canonical encoding
and the representation descriptor. `conformance.py` is the suite described
above. `kv_envelope.py` reduces an identity to bytes another language can
reproduce, and `test_envelope.py` checks it. `mojo/kv_identity.mojo` and
`mojo/missing_field.mojo` are the compile-time demonstration and its negative
case. `docs/` holds the connector proposal.

Asymmetric KV *quantization* — packing V to int8 while keeping K at 16-bit —
lives on `kv-asym-quant` and is deliberately not here. It makes the cache
smaller and cheaper to move and says nothing about when a block may be reused.
The two lines meet only at the representation descriptor on `main`, and were
split onto separate branches after being conflated once.

Longer-form design notes:
`/data/knlp-key-results/modular-kv-provenance-contract-20260707/`.

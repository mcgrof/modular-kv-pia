# modular-kv-pia

Research prototypes for **Prefix Integrity Analysis (PIA)**: checking whether a
saved transformer key/value cache can be reused without confusing two different
models, adapters, byte layouts, users, or requests.

The repository also contains a separate asymmetric-quantization experiment.
That experiment makes cache data smaller; PIA decides whether cache data is safe
to reuse. They share a description of the stored bytes but solve different
problems.

## Which branch should I read?

| Branch | Purpose | Status |
|---|---|---|
| [`main`](https://github.com/mcgrof/modular-kv-pia/tree/main) | Shared cache-block description and stable encoding used by the experiments. | Start here for the common data model. |
| [`kv-compat`](https://github.com/mcgrof/modular-kv-pia/tree/kv-compat) | First PIA compatibility prototype: complete cache identity, executable collision reproductions, a Python wire envelope, and a Mojo required-field demonstration. | Historical foundation for the newer resolved-request branch. |
| [`2026-09-04-feature-resolved-kv-contract`](https://github.com/mcgrof/modular-kv-pia/tree/2026-09-04-feature-resolved-kv-contract) | Extends `kv-compat` with model-artifact registration, immutable handles, and a request description intended to supply cache addresses to serving systems. | Current scaling prototype; public for collaboration, not production-ready. |
| [`kv-asym-quant`](https://github.com/mcgrof/modular-kv-pia/tree/kv-asym-quant) | Keeps keys at 16 bits while quantizing values to 8 bits in a first-cut Mojo codec. | Separate compression experiment, not a PIA implementation branch. |

[`BRANCHES.md`](BRANCHES.md) explains the ancestry, the integration branches in
vLLM and LMCache, and the current limitations in more detail.

## The shared idea

A cache lookup normally starts from the prompt tokens. Tokens alone are not
enough: the same tokens processed by different model weights, fine-tuning
adapters, attention rules, byte layouts, or access policies can produce cache
objects that must not be mixed.

PIA therefore divides cache identity into three parts:

- **Meaning:** which model computation produced the numbers.
- **Representation:** how those numbers are arranged and encoded as bytes.
- **Access:** which tenant or isolation domain is allowed to reuse them.

The active prototype registers the files actually loaded by the model runtime,
binds a request to that registration, and derives an opaque 32-byte address for
each cache chunk. A connector should carry that address unchanged rather than
rebuilding a weaker address from tokens.

## Current boundary

This is research code. A bounded one-GPU experiment carried an externally
derived address through vLLM and LMCache and restored the stored tensors exactly.
It did not run a complete model server, did not test more than one worker, and
did not measure performance. A September 2026 review also found correctness
problems that must be fixed before an upstream proposal.

The public research status, results, design choices, and work list live on the
[scaling PIA page](https://knlp.io/prefix-integrity-analysis-scaling.html).

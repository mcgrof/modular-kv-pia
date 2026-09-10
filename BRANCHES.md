# Branch map

Last updated: 2026-09-10.

This repository keeps two different research questions on separate branches:

1. Can a serving system tell when a saved key/value cache is safe to reuse?
2. Can it encode that cache in fewer bytes?

The first question is Prefix Integrity Analysis (PIA). The second is cache
compression. They meet at the representation description, which states how the
stored bytes are laid out, but neither question answers the other.

## Ancestry

```text
6141d2a  original shared cache-block primitives
    |
    +-- main (17d9af5)
          |
          +-- kv-compat (e7e47d1)
          |     |
          |     +-- 2026-09-04-feature-resolved-kv-contract (3840c58)
          |
          +-- kv-asym-quant (51d7ea5)
```

Backup tags record earlier rewrites of `kv-compat`. They are not alternative
implementations and are not starting points for new work.

## `main`: common description and encoding

`main` contains the shared cache-block types. Its representation description
records the key and value formats, page size, memory layout, attention-head
geometry, worker partition, and wire-format version. The PIA branches place this
description inside cache identity; the codec branch reads it to decide how to
pack and unpack data.

Use this branch when reviewing the common vocabulary or changing a field shared
by both research lines.

## `kv-compat`: first compatibility prototype

`kv-compat` demonstrates why token-only cache addresses collide and introduces
a complete identity divided into meaning, byte representation, and access
scope. It contains:

- Python reproductions of historical collision classes;
- a chained, opaque cache address and transport envelope;
- a Mojo example showing that a required identity field cannot be omitted; and
- a proposal for passing externally derived addresses through a connector.

It is retained as the readable foundation and parent of the active prototype.
New scaling work should normally start from the resolved-request branch below.

## `2026-09-04-feature-resolved-kv-contract`: active PIA prototype

This branch adds the missing producer side: register the model and adapter files
that were actually opened, issue an immutable registration handle, bind a
request to that handle, and derive the cache address from the resulting request
description.

The corresponding public integration experiments are:

- [vLLM request and connector branch](https://github.com/mcgrof/vllm/tree/2026-09-04-feature-kv-provenance-contract), commit `5726847ea5`;
- [LMCache multiprocess branch](https://github.com/mcgrof/LMCache/tree/2026-09-04-feature-kv-provenance-contract), commit `4a93ed7b`.

These are collaboration branches, not merge proposals. The current interfaces
still need a real model-loader binding, a consistent answer for multi-worker
addresses, partial-prefix fixes, and stronger immutability and encoding checks.
See the [scaling PIA page](https://knlp.io/prefix-integrity-analysis-scaling.html)
for the measured boundary and current work list.

## `kv-asym-quant`: separate compression experiment

This branch contains a first-cut Mojo codec that retains key tensors at 16 bits
and quantizes value tensors to 8 bits. Its purpose is to reduce storage and
transfer bytes. It does not decide whether two requests may share a cache entry.

Use this branch for codec development. Do not stack PIA connector changes on it;
changes needed by both lines belong on `main`.

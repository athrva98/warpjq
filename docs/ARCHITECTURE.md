# Architecture

Numbers here were measured with `WARPJQ_PROFILE=1`, on an RTX 5060 Laptop
(sm_120) under Windows 11 or an H100 80 GB via `scripts/modal_ab.py`, as noted
per figure.

## Pipeline

```
file (mmap or stream)
  -> chunker            newline-aligned, never splits a line
  -> parallel memcpy    into pinned host buffer, slot N
  -> cudaMemcpyAsync    H2D; slot N uploads and computes while N-1 drains
  -> k_nl_count/write   newline scan
  -> k_build_lines      (offset, length) pairs, blank detection
  -> k_eval             validate, resolve paths, predicate, group table
       aggregate  -> k_agg_reduce -> k_agg_final
       streaming  -> DeviceSelect -> k_row_len -> ExclusiveSum -> k_emit
  -> D2H                rows, selected indices, row offsets, fallback list
  -> merge              GPU rows and CPU-finished fallback rows, by line index
```

### Chunking

Chunks end on a newline; the chunker walks back to the last `\n` and carries
the remainder forward. Every consumer sees whole lines, so the kernel needs no
cross-chunk stitching. A line longer than a chunk is not split: the chunker
extends to the next newline, and a chunk exceeding the staging buffer goes to
the CPU engine.

The chunker does not count newlines. It once did, to maintain line numbers for
diagnostics, and that single-threaded pass was also the first touch of each
mapped page, absorbing the page-fault cost: 0.62 s/GB, which made both engines
2.5x slower. The consumer now returns the count it already computes.

### Double buffering

Two slots. Chunk *n* is filled and submitted, then *n-1* is drained. Chunks *n*
and *n-2* share a slot and *n-2* was drained during iteration *n-1*, so a slot
is never refilled while in flight. The pinned buffer stays valid until its slot
is refilled, which is what lets the fallback path read declined lines during
the drain.

`warpjq_submit` synchronises to read the newline count before sizing the
remaining launches, which serialises chunk *n+1*'s copy against *n*'s upload.
~0.05 s/GB. On the roadmap.

## k_eval

One thread per line, each running the whole JSON state machine. A warp stages
its 32 lines through shared memory, so the span is pulled with coalesced
16-byte loads and parsed locally; a warp whose span exceeds the budget parses
from global memory.

Extraction is fused into the validating walk. Previously `d_validate` walked
the line, then `d_extract` walked it again once per needed path, since
last-duplicate-wins prevents `d_object_get` from exiting early. A three-path
projection read every byte four times.

Measured against this shape (H100, 200 MB nginx):

| lever | result |
|---|---|
| shared-memory staging | 26.9 -> 5.29 sectors per request, 65% -> 25% long-scoreboard stalls, ~2x kernel time |
| occupancy | 5120 -> 6144 byte budget lifts 12 -> 16 warps, runtime moves 0.02% |
| instruction count | vectorised scan costs 5.2x instructions for no gain; scalar loop is already ~1.5 inst/byte |

Neither occupancy nor instruction count is the limit. What remains is the
serial dependency chain of one thread walking one line. Shortening it means
warp-cooperative structural indexing, which is a redesign, not a tuning pass.

That redesign is not the next thing to do. On an H100 at 4 GB, `submit` (H2D
plus all kernels) is 0.199 s of a 0.751 s run and the host copy is 0.200 s,
with ~0.35 s of fixed CUDA setup and process cost. Eliminating kernel time
entirely buys 26%; warp-cooperative indexing claims some fraction of what is
left of `submit` after H2D, which is less. Deleting the host copy is worth
more.

Byte equality with the CPU scanner does not depend on the kernel's shape. The
differential tests compare output, so any kernel producing the same bytes
passes them.

## Compiled query

`select(.a.b == 500) | {x: .c}` becomes flat, pointer-free tables uploaded once
per run:

| table | contents |
|---|---|
| `steps` | `{kind, index, key_off, key_len, key_hash}`, one per path segment |
| `paths` | `{step_off, step_count}`, one per interned path |
| `cmps` | `{path, op, lit_kind, lit_off, lit_len, lit_num}` |
| `cond_rpn` | condition in reverse-Polish order |
| `blob` | key names, string literals, projection prefixes |

Two things stay on the host. Key-name escaping: a projection's `{"x":` bytes
are precomputed into the blob, so JSON string escaping lives in one tested
place. Number formatting: aggregate results are formatted host-side, so
jq-compatible rendering is not duplicated in CUDA.

Struct layouts are mirrored by hand in `src/gpu/ffi.rs`. `warpjq_abi_check`
compares `sizeof` on both sides at startup and refuses to run on a mismatch,
because a misaligned struct produces plausible wrong answers rather than a
crash. Note that `uint64_t` is `unsigned long` on Linux and `unsigned long
long` on MSVC, so anything crossing the boundary must be spelled as the header
spells it.

## Correctness

### No DOM

Extracted values are slices of the input, so `1.0`, `0.10` and
`123456789012345678901234567890` survive intact. A DOM would round-trip them
through `f64`. This is also what makes GPU and CPU output byte-comparable:
both hand back offsets into the same bytes.

### The kernel may decline a line

`WARPJQ_LINE_FALLBACK` is a first-class outcome, produced when a number falls
outside the provably correctly-rounded fast path (mantissa >= 2^53, decimal
exponent outside +/-22, or more than 19 significant digits), nesting exceeds
the 64-level stack, a string would need materialising for CSV, or a group key
exceeds 64 KB.

Declined lines are collected with their byte ranges, finished on the CPU by
`exec::cpu::eval_line` (the same function the CPU engine uses) and merged back
by line index. `--stats` reports the rate; it is 0 on all four presets.

The fast-path restriction matters: naive mantissa-times-power-of-ten
accumulation is off by an ULP often enough that `sum(.bytes)` would disagree
with jq on real data. Restricting the kernel to the window where a single
double multiply is provably exact makes disagreement impossible rather than
unlikely.

### Ordering

Output order is input order. Selection is `cub::DeviceSelect::Flagged` over
ascending line indices, which is stable. Rows are written at offsets from an
exclusive prefix sum. Fallback rows are merged by line index, not appended. A
chunk routed to the CPU mid-pipeline drains the in-flight GPU chunk first.

That last point was a bug: streamed input made every chunk exceed the slot
capacity, so every chunk took the CPU path and wrote immediately while chunk 0
sat in flight and drained last. Output contained every line exactly once, in
the wrong order.

### Group table

Open addressing, 65536 entries, keyed on `(kind, decoded key bytes)`. The key
location is packed into the single 64-bit word that `atomicCAS` publishes:

```
[ offset : 40 ][ length : 20 ][ kind : 4 ]
```

so no thread can observe a claimed slot whose key is not yet written, which is
the classic failure of "CAS a flag, then fill the payload". On overflow (more
than 64 probes) the table sets a flag and the host redoes the chunk on the CPU;
it never merges two distinct keys.

## Memory budget

Per slot, at the default 64 MB GPU chunk:

| buffer | size |
|---|---|
| pinned staging, device copy | 64 MB each |
| line offsets and lengths | 8 B x max_lines |
| status and pass flags | 2 B x max_lines |
| selected indices, row offsets | 12 B x max_lines |
| assembled output | 1.5x chunk + 1 MB |
| group table | ~2.6 MB fixed |

`max_lines` is `chunk_bytes / 24`, an assumption rather than a bound: the
shortest legal NDJSON line is `{}`, so a pathological file holds 3x more. The
device reports `chunk_overflow` and the host redoes that chunk on the CPU.

Per-line slot tables are not materialised. `k_eval` resolves paths into
registers and the emit kernel re-resolves for surviving lines only, which costs
less than storing `n_paths x 12 bytes x every line in the chunk`.

## Where the time goes

H100, `select(.status == 500) | count`:

| stage | 2 GB | 4 GB | 8 GB |
|---|---|---|---|
| read and chunk | 0.004 s | 0.015 s | 0.027 s |
| copy to pinned | 0.463 s | 0.981 s | 1.995 s |
| submit (H2D and kernels) | 0.051 s | 0.102 s | 0.203 s |
| wait, merge, write | 0.000 s | 0.000 s | 0.001 s |
| total | 0.845 s | 1.417 s | 2.719 s |

Kernels sustain 42.2 GB/s at every size. The host copy runs at 4.3 GB/s and is
73% of the 8 GB run.

Kernel throughput does not track device tier: RTX 5060 Laptop 20.2 GB/s, L40S
14.3, A100 11.9, H100 29.4 (all at 1 GB). The kernel is latency and clock
bound, so HBM and a high SM count buy it little.

PCIe is not the constraint. An earlier version of this document predicted a
full x16 link would change the result; L40S, A100 and H100 at gen 4 and gen 5
x16, verified under load, did not.

## Measurement notes

`nvidia-smi` reports idle link and clock state. A card at rest downclocks to
gen 1 and P8, and a laptop on battery sits at 6% of its SM clock. Sample under
load, and record clocks alongside timings; `scripts/modal_ab.py` does both.

Run-to-run spread on identical binaries and data is +/-20% best-of-9 on a
shared H100 container, so end-to-end differences below that are not
measurable there.

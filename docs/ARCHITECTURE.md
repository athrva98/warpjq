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

Nothing pathological is required to hit it. `{"a":1}` is seven bytes, so a
file of small records overflows every chunk and runs entirely on the CPU while
`--stats` still names the backend `gpu`. That is why the redone chunks are
counted and reported separately: correct output is not evidence the device did
any work, and this case is invisible otherwise. The same 24-byte assumption is
what let `k_nl_write` run off its buffer before it was given a bound.

Per-line slot tables are not materialised. `k_eval` resolves paths into
registers and the emit kernel re-resolves for surviving lines only, which costs
less than storing `n_paths x 12 bytes x every line in the chunk`.

## Where the time goes

H100, 8 GB, `select(.status == 500) | count`, across the three input paths the
backend has had:

| stage | mmap + copy | pinned reads | + reader thread |
|---|---|---|---|
| read | 0.083 s | 0.427 s | 0.045 s |
| copy to pinned | 2.165 s | none | none |
| submit (H2D and kernels) | 0.185 s | 0.242 s | 0.224 s |
| wait, merge, write | 0.000 s | 0.000 s | 0.000 s |
| total | 2.507 s | 0.995 s | 0.520 s |

The first column mapped the file and memcpy'd each chunk into the pinned
buffer the DMA engine reads from, which cost more than the transfer and the
kernels together. The second reads the file straight into those buffers, so
the bytes land once. The third fills the next buffer on a separate thread
while the device works, which is why `read` drops without the work getting
smaller: what is left is only the part that could not be hidden.

Each column is an A/B against the one before it, on one machine with one file.
Absolute figures move between containers, sometimes by 35%, so the ratios are
the part worth quoting. Registering the mapping with `cudaHostRegister`
instead of copying was also tried and is 2x slower: pinning 8 GB of pages runs
at 5.3 GB/s against 20.8 GB/s for the memcpy it would replace.

Kernel throughput does not track device tier: RTX 5060 Laptop 20.2 GB/s, L40S
14.3, A100 11.9, H100 29.4 (all at 1 GB). The kernel is latency and clock
bound, so HBM and a high SM count buy it little.

PCIe is not the constraint. An earlier version of this document predicted a
full x16 link would change the result; L40S, A100 and H100 at gen 4 and gen 5
x16, verified under load, did not.

## Benchmarks

1 GB of `gen --preset nginx --seed 1`, warm cache, best of three, end-to-end
wall clock including the file read.

RTX 5060 Laptop (sm_120), Windows 11, against jq 1.8.2:

| query | gpu | cpu | jq |
|---|---|---|---|
| `select(.status == 500) \| count` | 0.25 s | 0.33 s | 30.5 s |
| `select(.status == 500) \| sum(.bytes)` | 0.26 s | 0.34 s | 30.6 s |
| `select(.status >= 500) \| {p: .path, b: .bytes}` | 0.36 s | 0.34 s | 40.1 s |
| `select(.status == 500)` | 0.38 s | 0.34 s | 39.6 s |
| `group_by(.host) \| count` (200 MB) | 0.23 s | 0.07 s | 12.0 s |

`group_by` uses a smaller file because it makes jq slurp the whole input into
memory as parsed JSON.

GPU against CPU by input size, `select(.status == 500) | count`, H100 80 GB
with 8 host cores. **The GPU column predates the input path rewrite** and is
kept for the shape of the curve, not the values: 8 GB now runs in 0.52 s to
0.65 s depending on the container, rather than 2.719 s. The sizes between
have not been remeasured.

| input | gpu (old path) | cpu | ratio |
|---|---|---|---|
| 1 MB | 0.217 s | 0.006 s | 0.03x |
| 200 MB | 0.223 s | 0.061 s | 0.27x |
| 1 GB | 0.286 s | 0.254 s | 0.89x |
| 2 GB | 0.845 s | 1.180 s | 1.40x |
| 4 GB | 1.417 s | 2.455 s | 1.73x |
| 8 GB | 2.719 s | 5.451 s | 2.00x |

Against DuckDB 1.5.5 `read_ndjson_auto`, H100, 8 host cores, each timed in a
fresh process so neither side reuses a parse. Answers were compared before
either side was timed and agreed exactly on every query.

| query | 1 GB | | 8 GB | |
|---|---|---|---|---|
| | warpjq | duckdb | warpjq | duckdb |
| `select(.status == 500) \| count` | 0.339 s | 0.367 s | 0.646 s | 1.048 s |
| `select(.status >= 500) \| {p: .path, b: .bytes}` | 0.501 s | 0.397 s | 0.906 s | 1.338 s |
| `group_by(.host) \| count` | 0.362 s | 0.378 s | 0.655 s | 1.103 s |
| `select(.status == 500)` | 0.509 s | 0.470 s | 0.907 s | 2.084 s |

The noise floor is worth knowing before reading any of this. In the run that
produced the 8 GB column, the two warpjq builds under test were behaviourally
identical at that size, since both allocate three slots above 32 chunks, and
they still measured up to 5% apart. Between containers it is worse: DuckDB's
own 8 GB count came out at 0.927 s in one run and 1.048 s in another, 13%
apart, and warpjq's at 0.520 s and 0.646 s, 24% apart. Ratios within a single
run are the only figures here that carry.

DuckDB is doing schema inference, which is what a user typing
`read_ndjson_auto` actually pays, so this is an end-to-end comparison rather
than a pure scan one. Passing explicit columns would make it faster.

Per-device, same 1 GB file, 8 host cores, PCIe sampled under load:

| device | cc | PCIe | kernels | host copy |
|---|---|---|---|---|
| RTX 5060 Laptop | 12.0 | gen 4 x8 | 20.2 GB/s | 7.0 GB/s |
| L40S | 8.9 | gen 4 x16 | 14.3 GB/s | 19.9 GB/s |
| A100 SXM4 40 GB | 8.0 | gen 4 x16 | 11.9 GB/s | 9.7 GB/s |
| H100 80 GB HBM3 | 9.0 | gen 5 x16 | 29.4 GB/s | 21.8 GB/s |

Reproduce with `warpjq bench`, or `scripts/modal_ab.py` for the datacentre
parts, which builds two commits into one image and compares them on one
device.

## Differences from jq

Measured against jq 1.8.2 and checked in CI. Semantics match: filtering,
grouping and every aggregate agree, including across different spellings of
the same string.

Rendering differs because warpjq returns slices of the input where jq
re-serialises. Numbers agree on jq 1.7 and later, which preserves literals;
jq 1.6 prints `123456789012345678901234567890` as
`123456789012345680000000000000`.

| input | jq | warpjq |
|---|---|---|
| `{"s":"\u0041"}` | `"A"` | `"\u0041"` |
| `{"s":"\/slash"}` | `"/slash"` | `"\/slash"` |
| `{"a":1e3}` | `1E+3` | `1e3` |

Normalising would mean re-serialising every value, which costs the number
fidelity above.

Validity differs in the other direction. jq accepts documents RFC 8259
forbids and rewrites the value rather than reporting an error:

| input | jq | warpjq |
|---|---|---|
| `{"a":01}` | `{"a":1}` | rejects |
| `{"a":-01}` | `{"a":-1}` | rejects |
| `{"a":+1}` | `{"a":1}` | rejects |
| `{"a":.5}` | `{"a":0.5}` | rejects |
| `{"a":1.}` | `{"a":1}` | rejects |
| `{"a":Infinity}` | `1.797...e+308` | rejects |
| `{"a":NaN}` | `null` | rejects |
| `{"a":1}{"b":2}` | two values | rejects |

The last is a model difference: jq reads a stream of values, NDJSON is one
value per line. Conversely warpjq accepts a lone high surrogate
(`"\ud83d"`) and a reversed pair, which jq rejects; a lone low surrogate is
accepted by both.

`tests/conformance.rs` asserts this divergence set exactly, so a new one
fails the build rather than going unnoticed.

## Testing

| suite | asserts |
|---|---|
| `differential` | GPU output equals CPU output byte for byte on fuzzed input, ~40 queries by 3 formats, at chunk sizes that force multi-chunk paths and slot reuse; and equals jq on a corpus excluding the spellings above |
| `conformance` | 48 must-accept and 52 must-reject documents through the scanner, both backends and jq |
| `invariants` | output unchanged across chunk size 1 B to 16 MB, thread count 1 to 32, format and backend |
| `cli` | exit codes, stdin, multiple files, every flag, error paths |
| fuzz | scanner, query compiler, and a full end-to-end run over arbitrary bytes |

Release builds compile out `debug_assert!`, so CI runs the suite in both
profiles. A contract violation guarded that way was invisible to the release
run and only surfaced through fuzzing, which builds with debug assertions on.

## Measurement notes

`nvidia-smi` reports idle link and clock state. A card at rest downclocks to
gen 1 and P8, and a laptop on battery sits at 6% of its SM clock. Sample under
load, and record clocks alongside timings; `scripts/modal_ab.py` does both.

Run-to-run spread on identical binaries and data is +/-20% best-of-9 on a
shared H100 container, so end-to-end differences below that are not
measurable there.

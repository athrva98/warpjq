# warpjq architecture

This document explains the pipeline, the memory budget, and, where it
matters, the measurement that produced each decision. Anything stated as a number here
was measured on the development machine (RTX 5060 Laptop, sm_120, 8 GB;
Windows 11; 1 GB `nginx` preset, warm page cache) and can be reproduced with
`WARPJQ_PROFILE=1`.

---

## 1. The pipeline

```
  file (mmap or stream)
        |
        v
   chunker  ── newline-aligned, never splits a line
        |
        v
   parallel memcpy ──> [pinned host buffer, slot N]
        |
        v                             ┌──────────────────────────────┐
   cudaMemcpyAsync (H2D)              │  double buffered: slot N is  │
        |                             │  uploading + computing while │
        v                             │  slot N-1 is being drained   │
   k_build_lines   (offsets, lengths, blank detection)
        |
        v
   k_eval          (validate → resolve paths → predicate → group table)
        |
        +-- aggregate query --> k_agg_reduce -> k_agg_final -> one struct
        |
        +-- streaming query --> DeviceSelect (stable) -> k_row_len
                                     -> ExclusiveSum -> k_emit
        |
        v
   D2H (rows, selected line indices, row offsets, fallback list)
        |
        v
   merge: GPU rows and CPU-finished fallback rows, by line index
        |
        v
   stdout
```

### Chunking

Chunks end on a newline. The chunker walks back from the nominal end to the
last `\n` and carries the remainder forward. Every consumer therefore sees
whole lines and nothing else, which is what lets the kernel assume "one line
per thread" with no cross-chunk stitching.

A line longer than a chunk is not split: the chunker extends forward to the
next newline, and if the result exceeds the GPU staging buffer the whole chunk
is handed to the CPU engine.

**Measured lesson.** The chunker originally counted newlines per chunk to
maintain line numbers for diagnostics. That single-threaded pass over every
byte was *also* the first touch of each mapped page, so it absorbed all the
page-fault cost:

| | before | after |
|---|---|---|
| `read+chunk` stage | 0.62 s (1.7 GB/s) | 0.00 s |
| end-to-end, GPU | 1.25 s | 0.48 s |
| end-to-end, CPU | 1.03 s | 0.41 s |

Removing it made *both* engines ~2.5x faster. The consumer now reports how many
lines it saw (`for_each_chunk`'s closure returns a count), because both engines
already know it as a by-product of work they were doing anyway.

### Double buffering

Two slots. Chunk *n* is filled and submitted, then chunk *n−1* is drained.
Chunk *n* and chunk *n−2* share a slot, and *n−2* was drained during iteration
*n−1*, so a slot is never refilled while it is in flight.

The pinned staging buffer stays valid until its slot is refilled, which is what
lets the fallback path read the original bytes of declined lines during the
drain.

**Known limitation.** `warpjq_submit` currently does a
`cudaStreamSynchronize` to read the newline count back before it can size the
remaining kernel launches. That serialises the copy of chunk *n+1* against the
upload of chunk *n*. It costs ~0.05 s per GB and is on the roadmap.

---

## 2. Why one thread per line

The plan this project started from specified a warp per line building a
simdjson-style structural bitmap. The implementation does not do that, on
purpose.

Log lines are 100 to 800 bytes. A warp-cooperative structural scan computes the
quote/escape mask 32 bytes at a time in parallel, and then **serialises** on one
lane to walk the structural positions and execute the path program. For a
200-byte line that is ~7 parallel steps followed by ~30 serial ones with 31
lanes idle. Thread-per-line keeps every lane doing useful work; the reads are
strided rather than coalesced, but each thread streams sequentially through its
own line, so the bytes fetched are the bytes needed and the cost is latency,
which occupancy hides.

The decisive argument is not performance. It is that thread-per-line is a
direct transliteration of the CPU scanner in `src/json.rs`. Two
implementations of the same algorithm can be asserted byte-equal. Two different
algorithms can only be asserted approximately equal, and "approximately equal to
jq" is not a product.

The warp-cooperative version is on the roadmap for when the input path stops
being the bottleneck. Currently the kernels are 4x faster than the host can
feed them, so making them faster would change nothing.

---

## 3. The compiled query

`select(.a.b == 500) | {x: .c}` becomes flat, pointer-free tables uploaded once
per run:

| table | contents |
|---|---|
| `steps` | `{kind, index, key_off, key_len, key_hash}`, one per path segment |
| `paths` | `{step_off, step_count}`, one per interned path |
| `cmps` | `{path, op, lit_kind, lit_off, lit_len, lit_num}` |
| `cond_rpn` | the condition in reverse-Polish order |
| `blob` | key names, string literals, and projection prefixes |

Two things deliberately stay on the host:

- **Key-name escaping.** A projection's `{"x":` and `,"y":` bytes are
  precomputed into the blob. The kernel copies them verbatim, so JSON string
  escaping lives in exactly one tested place.
- **Number formatting.** Aggregate results are formatted on the host, so
  jq-compatible number rendering is not duplicated in CUDA.

`FlatProgram` is built by `src/query/mod.rs` and consumed by both the lowering
in `src/gpu/lower.rs` and the tests. The struct layouts are mirrored by hand in
`src/gpu/ffi.rs`; `warpjq_abi_check` compares `sizeof` on both sides at startup
and refuses to run on a mismatch, because a misaligned struct would produce
plausible wrong answers rather than a crash.

---

## 4. Correctness by construction

### No DOM

Extracted values are slices of the input. `1.0` stays `1.0`, `0.10` stays
`0.10`, and `123456789012345678901234567890` keeps all thirty digits. A DOM
would round-trip through `f64` and lose digits on exactly the inputs users
notice. This is also what makes the GPU and CPU outputs comparable at the byte
level: both hand back offsets into the same bytes.

**Measured, not assumed.** The justification above was written from memory
before jq was installed on the development machine, which is a bad way to
justify a design. With jq 1.8.2 present it holds: jq preserves `1.0`, `0.10`,
`100`, `9007199254740993` and the thirty-digit integer exactly. There is now a
test (`jq_preserves_number_literals_like_warpjq_does`) that fails if a future jq
starts renormalising them, so the rationale cannot rot silently.

The same measurement found where the design *does* cost compatibility.
jq normalises string escapes on output (`"\u0041"` becomes `"A"`, `"\/"`
becomes `"/"`) and canonicalises exponents, `1e3` to `1E+3`. warpjq preserves
the input spelling in all three cases. Matching jq there would mean
re-serialising every value, which is the DOM round-trip whose absence buys the
number fidelity above, so it is a trade rather than a bug.

The line worth drawing is between *rendering* and *semantics*. Comparison,
grouping and hashing all decode escapes, so `select(.s == "A")` matches
`{"s":"\u0041"}` and `group_by` unifies the two spellings, verified against jq,
byte for byte, including `group_by`'s own output, which round-trips through the
decoder on both sides. Only raw passthrough differs.

### The kernel may decline a line

`WARPJQ_LINE_FALLBACK` is a first-class outcome. It is produced when:

- a number falls outside the provably correctly-rounded fast path
  (mantissa ≥ 2^53, or a decimal exponent outside ±22, or more than 19
  significant digits);
- nesting exceeds the 64-level depth stack;
- a string would need materialising to render as a CSV cell;
- a group key is longer than 64 KB, or the group table overflows.

Declined lines are collected with their byte ranges, finished on the CPU by
`exec::cpu::eval_line`, the *same function* the CPU engine uses, and merged
back by line index. `--stats` reports the rate; on the four built-in presets it
is 0.

The correctly-rounded fast path matters more than it looks. A naive
mantissa-times-power-of-ten accumulation is off by an ULP often enough that
`sum(.bytes)` would disagree with jq on real data. Restricting the kernel to the
window where a single double multiply is provably exact, and deferring
everything else, makes the disagreement impossible rather than unlikely.

### The scanner is iterative, and that is not a style choice

`skip_value` is a state machine over an explicit container stack. The natural
way to write it is mutual recursion (value calls object calls value) and that
is what it was until a code review pointed at it.

A single line of `[[[[…` around 50 KB, comfortably inside the default 64 MB
`--max-line-bytes`, recurses deep enough to overflow the thread stack. That is
not an error that can be caught: it is an abort, and `panic = "abort"` in the
release profile removes even the theoretical option. Any pipeline feeding
warpjq untrusted NDJSON could be killed by one line.

It took the GPU path down with it. The kernel is iterative and correctly
declines anything past its 64-level stack, and hands that line to the CPU
evaluator, which then crashed. The device's careful bounds check delivered the
crash rather than preventing it.

The rewrite keeps the first 64 levels in a single `u64` (one bit per level,
array or object) and spills to a `Vec` beyond that, so the common case still
allocates nothing. Depth is now bounded only by line length, since every level
costs at least one byte.

The lesson worth keeping: the fuzz target for this file *names* "nesting deep
enough to blow a recursive parser's stack" in its header comment, and could not
reach it, because libFuzzer's default `max_len` of 4096 caps depth around 2000.
A guard that documents a risk it cannot exercise is worse than no guard. It
reads as coverage. CI now passes `-max_len=262144`.

### Output capacity is checked on the device, before the store

`out_cap` reserves `chunk * 1.5 + 1 MB`. A projection is not bounded by its
input: six named fields over 26-byte lines expand roughly sevenfold, and the
key names are query-controlled, so the true factor is unbounded.

`k_emit` used to compute `dst = out + row_off[k]` from the prefix sum and write
with no bound check, and the capacity was tested afterwards on the host. By
then the overrun had happened; it surfaced as
`an illegal memory access was encountered`, undefined behaviour detected only
because the driver happened to catch it.

The check now lives in `k_emit`, per row, before any store: a row that would
not fit sets the chunk's overflow flag and is skipped, and the host redoes the
whole chunk on the CPU. The host-side test remains as a backstop and also
degrades to a CPU redo rather than failing the run. Reaching it means device
state is not what we expect, which on a shared card usually means another
process, and dying because something else was using the GPU is not useful
behaviour.

### Ordering

Output order is input order, unconditionally:

- selection is `cub::DeviceSelect::Flagged` over ascending line indices, which
  is stable;
- rows are written at offsets from an exclusive prefix sum, so each row lands
  where it started;
- fallback rows are merged by line index, not appended;
- a chunk routed to the CPU mid-pipeline **drains the in-flight GPU chunk
  first**.

That last point was a real bug. Streamed input made every chunk exceed the slot
capacity (the carried partial line pushed it over), so every chunk went to the
CPU path, which wrote immediately, while chunk 0 sat in flight and was drained
last. The output contained every line exactly once, in the wrong order. Two
fixes: the stream reader now counts the carry against the requested chunk size,
and the CPU path drains before writing.

### The group-by table

Open addressing, 65 536 entries, keyed on `(kind, decoded key bytes)`. The key's
location is packed into the single 64-bit word that `atomicCAS` publishes:

```
[ offset : 40 ][ length : 20 ][ kind : 4 ]
```

so there is no window in which another thread can observe a claimed slot whose
key has not been written yet. That is the classic bug in "CAS a flag, then fill the
payload" tables. On overflow (more than 64 probes) the table sets a flag and
the host redoes the chunk on the CPU; it never merges two distinct keys.

Group keys are compared and hashed over their *decoded* bytes, so `"ab"` and
`"ab"` land in the same group.

**Measured lesson.** `StrIter::next()` originally returned `-1` for
end-of-string, and `R_INVALID` was also `-1`. Every successful string
comparison therefore reported a decode failure the moment it reached the
terminator, and every `group_by` line was declined to the CPU. The output stayed
*correct*, so no differential test could see it. It only showed up as
`100.000% of lines were finished on the CPU` in `--stats` while chasing a
performance question. Distinct sentinels (`STR_END`, `STR_ERR`) fixed it.

---

## 5. Memory budget

Per slot, with the default 64 MB GPU chunk:

| buffer | size |
|---|---|
| pinned staging + device copy | 64 MB each |
| line offsets + lengths | 8 bytes × `max_lines` |
| status + pass flags | 2 bytes × `max_lines` |
| selected indices + row offsets | 12 bytes × `max_lines` |
| assembled output | 1.5 × chunk + 1 MB |
| group table | ~2.6 MB, fixed |

`max_lines` is `chunk_bytes / 24`. That is an assumption, not a bound: the
shortest legal NDJSON line is `{}`, so a pathological file could hold 3x more
lines than the buffers allow. Rather than allocate for a worst case that never
occurs, the device reports `chunk_overflow` and the host redoes that chunk on
the CPU.

Notably, per-line *slot tables* are not materialised. The evaluation kernel
resolves paths into registers, and the emit kernel re-resolves them for the
lines that survived. Re-parsing the survivors costs less than storing
`n_paths × 12 bytes × every line in the chunk`, and it is what keeps the budget
above proportional to the chunk rather than to the query width.

---

## 6. Where the time goes

```
1.07 GB, group_by(.host) | count, warm cache
read+chunk (host)        0.000s
copy to pinned           0.154s     6.99 GB/s
submit (H2D + kernels)   0.053s    20.18 GB/s
wait (sync + D2H)        0.033s    32.45 GB/s
merge + write            0.000s
                        ------
total wall               0.522s     2.06 GB/s
```

The kernels do the actual work (validate every byte of every line, resolve
paths, evaluate predicates, hash and aggregate) at **20 GB/s**. The CPU engine
does the same work at ~2.8 GB/s. The gap between 20 GB/s of capability and
2.06 GB/s of delivered throughput is entirely host-side: a ~7 GB/s copy into
pinned memory, plus ~0.15 s of one-time CUDA context and allocation cost.

The consequence, stated plainly because it is the most useful thing this
document contains: **on consumer laptop hardware, warpjq is input-bound, not
compute-bound.** Optimising the kernels further would change nothing. The three
things that would move the number are all in the input path, and they are the
top three items on the roadmap.

A desktop with PCIe Gen4 x16 and quad-channel memory would feed the GPU
considerably faster; that measurement has not been taken and no claim is made
about it here.

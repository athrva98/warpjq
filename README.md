# warpjq

**Filter, project and aggregate NDJSON using a subset of jq syntax, on the GPU
when you have one and on every core when you don't.**

A single-binary CLI that speaks a strict subset of jq syntax, streams NDJSON
through CUDA kernels, and falls back to a fast multi-core CPU engine on
machines without an NVIDIA GPU. Every result is checked byte for byte against
the CPU engine and against real `jq` on fuzzed input, in CI.

```console
$ warpjq 'select(.status == 500) | count' access.ndjson
59218

$ warpjq 'group_by(.host) | count' access.ndjson
{"host":"api-01","count":740301}
{"host":"web-01","count":739884}
...
```

---

## Benchmarks

1 GB of `warpjq gen --preset nginx`, warm page cache, end-to-end wall clock for
the whole process including reading the file. jq 1.8.2.
Hardware: RTX 5060 Laptop (sm_120, 8 GB), Windows 11.

| query | warpjq (gpu) | warpjq (cpu) | jq 1.8.2 | speedup |
|---|---|---|---|---|
| `select(.status == 500) \| count` | 0.25 s | 0.33 s | 30.5 s | **121x** |
| `select(.status == 500) \| sum(.bytes)` | 0.26 s | 0.34 s | 30.6 s | **119x** |
| `select(.status >= 500) \| {p: .path, b: .bytes}` | 0.36 s | 0.34 s | 40.1 s | **119x** |
| `select(.status == 500)` | 0.38 s | 0.34 s | 39.6 s | **118x** |
| `group_by(.host) \| count` (see note) | 0.23 s | 0.07 s | 12.0 s | **165x** |

The `group_by` row is 200 MB, not 1 GB. `group_by` forces jq to slurp the entire
file into memory as parsed JSON. It does not stream, and it does not scale to
the sizes the rest of this table uses. warpjq streams it.

`warpjq bench` cross-checks every engine's answer before reporting a time. jq
and warpjq both return 59218 for the first row. A tool whose output disagrees
gets a `NOT COMPARABLE` row instead of a number, which is what happens to the
`grep -c` baseline on Windows, where argv quoting mangles its pattern so it
matches nothing and "wins" by doing no work.

### Two honest caveats

**The speedup over jq is real; the speedup from the GPU is not.** Look at the
two warpjq columns: they are the same number. On this hardware the CPU engine
wins as often as the GPU does, and the 1 MB case is 0.001 s on CPU against
0.044 s on GPU, a 40x *loss* that is entirely CUDA setup cost. Essentially all
of the 100x against jq comes from the query compiler, the byte-slice value model
and multi-core execution, not from the kernels.

**The GPU column is noisy.** Repeating an identical benchmark five times gives
the GPU a 0.31 s to 0.45 s spread while the CPU stays within 0.006 s. Treat the
two warpjq columns as tied, and read the per-stage profile below, which is
stable.

Reproduce exactly:

```bash
warpjq gen --preset nginx --size 1GB -o access.ndjson --seed 1
warpjq bench 'select(.status == 500) | count' access.ndjson
```

### Where the time actually goes

`WARPJQ_PROFILE=1` is built into the binary, because the first honest
measurement of this pipeline showed the GPU merely *tying* the CPU and the only
way to find out why was to stop guessing:

```
$ WARPJQ_PROFILE=1 warpjq 'group_by(.host) | count' --backend gpu access.ndjson
warpjq profile: 1.07 GB through the pipeline
warpjq profile: read+chunk (host)        0.000s  8055.08 GB/s
warpjq profile: copy to pinned           0.154s     6.99 GB/s
warpjq profile: submit (H2D+kernels)     0.053s    20.18 GB/s   <-- the GPU
warpjq profile: wait (sync+D2H)          0.033s    32.45 GB/s
warpjq profile: merge+write              0.000s 11582.98 GB/s
```

**The kernels parse, filter and aggregate JSON at ~20 GB/s.** That is roughly 7x
the CPU engine's ~3 GB/s, and roughly 600x jq's ~0.035 GB/s. But the host can
only get bytes into pinned memory at ~7 GB/s, and CUDA context setup costs a
fixed ~0.15 s, so end to end the GPU's advantage over the CPU engine disappears
into the noise.

This is the honest state of the project on consumer laptop hardware, and it is
the single most useful thing in this README: **the kernels are not the
bottleneck; the input path is.** The 100x over jq is already banked by the CPU
engine. The GPU is currently a 20 GB/s engine attached to a 7 GB/s fuel line.
See the [roadmap](#roadmap) for what would move it.

---

## Install

warpjq builds and runs with **no GPU and no CUDA toolkit**. That is deliberate:
the CPU engine is a genuinely fast tool on its own, and nobody should need
hardware to try this.

```bash
# CPU-only. Works everywhere Rust works.
cargo install warpjq-cli

# With CUDA. Needs the CUDA toolkit (12.x) and an NVIDIA GPU of
# compute capability 6.0 or newer.
cargo install warpjq-cli --features cuda
```

From source:

```bash
git clone https://github.com/athrva98/warpjq
cd warpjq
cargo build --release                      # CPU only
cargo build --release --features cuda      # with CUDA kernels
```

The build script finds the CUDA toolkit and, on Windows, MSVC's `cl.exe` on its
own, so a plain `cargo build --features cuda` works outside a Developer Command
Prompt. Set `WARPJQ_CUDA_ARCH=86` (or your architecture) to cut compile time
while developing.

---

## Examples

Every example below runs against a file you can generate locally, with no
downloads:

```bash
warpjq gen --preset nginx --size 1GB -o access.ndjson --seed 1
```

```bash
# 1. Count the 500s.
warpjq 'select(.status == 500) | count' access.ndjson

# 2. Pull out the slow requests, as NDJSON.
warpjq 'select(.duration_ms > 2000) | {t: .ts, path: .path, ms: .duration_ms}' access.ndjson

# 3. The same thing as CSV, for a spreadsheet.
warpjq 'select(.status >= 500) | {ts: .ts, path: .path}' --csv access.ndjson

# 4. Bytes served per host.
warpjq 'group_by(.host) | sum(.bytes)' access.ndjson

# 5. Reach into nested objects (k8s-style logs).
warpjq gen --preset k8s --size 500MB -o pods.ndjson
warpjq 'select(.kubernetes.namespace == "prod") | count' pods.ndjson
```

Presets: `nginx`, `cloudtrail`, `k8s`, and `nested`. The last one is
deliberately hostile (deep nesting, non-ASCII keys, escapes, integers past 2^53)
so that benchmarks are not run exclusively on easy data. `warpjq gen --list`
describes them.

---

## Supported query subset

```text
paths          .a   .a.b   .a[0]   ."odd key"
filters        select(.x == 1)   !=  <  <=  >  >=   and   or   | not
projection     {key: .path, shorthand}
aggregates     count   sum(.f)   min(.f)   max(.f)   avg(.f)
grouping       group_by(.f) | <aggregate>
output         NDJSON (default), --csv, --count
```

Anything outside this is **rejected with a message saying so**, by name, rather
than being silently misread:

```console
$ warpjq 'reduce .[] as $x (0; .+$x)' f.ndjson
warpjq: invalid query: `reduce` is not in the v0.1 subset
  reduce .[] as $x (0; .+$x)
  ^
  help: see the Limitations section of the README for the full v0.1 subset

$ warpjq 'select(status == 500)' f.ndjson
warpjq: invalid query: expected a field path, found `status`
  select(status == 500)
         ^
  help: did you mean `.status`?
```

---

## How it works

**One line, one thread.** Chunks arrive newline-aligned, a CUB stream compaction
turns the newline positions into `(offset, length)` pairs, and each thread walks
one line. The obvious alternative, a warp per line building a simdjson-style
structural bitmap, spends most of its time with 31 lanes idle while lane 0 walks
the bitmap, because log lines are 100 to 800 bytes, not megabytes.
Thread-per-line is also a direct transliteration of the CPU scanner, which is
what makes the two engines byte-comparable rather than approximately equal.

**The query is compiled once, on the host.** `select(.a.b == 500) | {x: .c}`
becomes flat tables (a step list, a comparison table, and the condition in
reverse-Polish order) that upload once and are read by every thread. The kernel
never parses a query and never formats a key name: even the `{"x":` bytes of a
projection are precomputed on the host, so JSON escaping of key names stays in
the one place it is already tested.

**No DOM, ever.** Extracted values are slices of the input, so `1.0`, `0.10` and
`123456789012345678901234567890` come out exactly as they went in. A DOM would
round-trip them through `f64` and lose digits on precisely the inputs people
notice. This is checked against jq rather than assumed:
`jq_preserves_number_literals_like_warpjq_does` asserts that jq 1.8 keeps those
same literals intact, so if a future jq starts renormalising them, the test says
so instead of the README quietly being wrong.

**The kernel is allowed to give up.** Numbers outside the provably
correctly-rounded fast path, nesting deeper than the 64-level stack, and strings
that would need materialising to render as CSV: any line the kernel cannot
decide *exactly* is marked `FALLBACK` and finished on the CPU by the same
evaluator, then merged back in input order. A GPU JSON parser that is subtly
wrong on 0.001% of lines is worse than useless, so "I am not sure" is a
first-class outcome. `--stats` reports how often it happens.

**Output order is input order, always.** Selection is a stable compaction over
ascending line indices, rows are written at offsets from a prefix sum, and
CPU-finished lines are merged by line index. Nothing sorts a stream.

**Double buffering.** While one chunk uploads and computes, the previous one is
drained. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full
pipeline, the memory budget, and the measurements behind each decision.

---

## Correctness

224 tests, 90% line coverage, enforced by a floor in CI.

- **Differential tests**: randomised NDJSON covering unicode, escapes,
  surrogate pairs, integers past 2^53, duplicate keys, deep nesting, empty
  containers, CRLF, blank lines, and deliberately malformed lines, asserting
  **GPU output == CPU output**, byte for byte, across ~40 queries and 3 output
  formats. Small chunk sizes force multi-chunk paths, slot reuse and cross-chunk
  merging, because that seam is where the bugs are.
- **Against real jq**: the same queries over an escape-free corpus assert byte
  equality with `jq` itself, plus dedicated tests that pin every known
  [difference](#differences-from-jq) and assert that query semantics match jq
  even where rendering does not. The excluded spellings are named in the test
  rather than being quietly absent from the corpus.
- **JSON conformance**, in the style of `nst/JSONTestSuite`: an explicit
  accept/reject corpus of 48 must-accept cases, 52 must-reject cases, and the
  implementation-defined ones, run through the scanner, both backends, and jq.
  The set of places warpjq and jq disagree about *validity* is itself asserted,
  so a new divergence fails the build instead of going unnoticed.
- **Invariance tests**: the answer may not depend on how the work was divided.
  Same bytes out at every chunk size from 1 byte to 16 MB, at every thread count
  from 1 to 32, in every output format, on both backends. Every serious bug this
  project has had lived in that seam rather than in the parser.
- **End-to-end CLI tests**: exit codes, stdin, multiple files, every flag, and
  the error paths, driving the real binary, because that is the surface users
  touch.
- **Fuzzing** (`cargo fuzz`): three targets covering the scanner, the query
  compiler, and a full end-to-end run over arbitrary bytes. Run nightly.

Bugs a code review caught that the tests did not, all now covered by regressions
and documented in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md):

1. The CPU scanner was recursive. One ~50 KB line of `[[[[...`, well inside the
   default `--max-line-bytes`, overflowed the stack and **aborted the process**,
   which is not catchable. It took the GPU path down too: the kernel correctly
   declines anything past its 64-level stack and hands the line to that scanner.
   Now iterative.
2. `k_emit` wrote assembled rows from prefix-sum offsets with **no bound
   check**, and the output-capacity test ran on the host afterwards. A six-field
   projection over short lines overran the device buffer, real undefined
   behaviour, observed as `an illegal memory access was encountered`. The check
   is now in the kernel, before the store.
3. Under `--count`, lines the kernel declined were **dropped from the tally**:
   the merge inferred "no row" from "no bytes", and `--count` formats every row
   to zero bytes.

Three more the test suites caught during development:

1. A sentinel collision made every device-side string comparison report
   "undecidable" at the terminator, silently routing 100% of `group_by` to the
   CPU. Output stayed *correct*, so only a performance assertion or a look at
   `--stats` would ever have found it.
2. A chunk routed to the CPU mid-pipeline was written while an earlier GPU chunk
   was still in flight, so its rows came out first. This affected every streamed
   (stdin) input.
3. Aggregates did not span multiple files. `warpjq 'sum(.n)' a.ndjson b.ndjson`
   printed one total per file instead of one total, and `group_by` emitted
   duplicate rows for any key present in more than one file.

---

## Differences from jq

Measured against **jq 1.8.2**, not recalled from memory. Every row below is
pinned by a test that fails if either tool changes.

**Semantics are identical.** Which lines a filter selects, which group a value
lands in, and what `count`, `sum`, `min`, `max` and `avg` return all match jq
exactly, including across different spellings of the same string.
`select(.s == "A")` matches both `{"s":"A"}` and `{"s":"\u0041"}`, and
`group_by` puts them in one group, exactly as jq does. `group_by` output is
byte-identical to jq's, because group keys are decoded and re-encoded on both
sides.

**Rendering differs, by design.** warpjq hands back slices of the input; jq
re-serialises. For numbers those agree. jq preserves `1.0`, `0.10`,
`9007199254740993` and a 30-digit integer, which is what makes the no-DOM design
worth having. For strings they do not:

| input | jq prints | warpjq prints | |
|---|---|---|---|
| `{"s":"plain"}` | `"plain"` | `"plain"` | same |
| `{"s":"tab\there"}` | `"tab\there"` | `"tab\there"` | same |
| `{"s":"café"}` | `"café"` | `"café"` | same |
| `{"s":"\u0041"}` | `"A"` | `"\u0041"` | **differs** |
| `{"s":"\u65e5"}` | `"日"` | `"\u65e5"` | **differs** |
| `{"s":"\/slash"}` | `"/slash"` | `"\/slash"` | **differs** |
| `{"a":1e3}` | `1E+3` | `1e3` | **differs** |

In short: **jq normalises escapes and exponent notation; warpjq preserves what
you gave it.** If your logs come from Python's `json.dumps` at its default
`ensure_ascii=True`, every non-ASCII character is written as `\uXXXX`, and
warpjq will pass those through where jq would decode them. The values are equal,
spelled differently.

Normalising output would mean re-serialising every value, which is exactly the
DOM round-trip that costs the number fidelity above. This is a trade, not an
oversight. It is not currently configurable.

**Validity differs too, and here warpjq is the stricter one.** jq 1.8 accepts
several documents RFC 8259 forbids, and does not merely accept them: it silently
rewrites the value.

| input | jq gives you | warpjq |
|---|---|---|
| `{"a":01}` | `{"a":1}` | rejects the line |
| `{"a":-01}` | `{"a":-1}` | rejects the line |
| `{"a":+1}` | `{"a":1}` | rejects the line |
| `{"a":.5}` | `{"a":0.5}` | rejects the line |
| `{"a":1.}` | `{"a":1}` | rejects the line |
| `{"a":Infinity}` | `1.797...e+308` | rejects the line |
| `{"a":NaN}` | `null` | rejects the line |
| `{"a":1}{"b":2}` | two values | rejects the line |

`01` silently becoming `1` is corruption being papered over, so warpjq refuses
the line instead. The last row is a model difference rather than leniency: jq
reads a stream of values, NDJSON is one value per line.

In the other direction, warpjq accepts a lone high surrogate (`"\ud83d"`) and a
reversed surrogate pair, which jq rejects outright. jq's handling here is
asymmetric: a lone *low* surrogate (`"\ude00"`) is accepted by both.

---

## Limitations

Stated up front, because a tool that overstates itself is worse than a slow one.

- **The GPU is not meaningfully faster than the CPU engine end-to-end.** On the
  development machine the kernels run ~7x faster, but the host feed rate
  (~7 GB/s) and the ~0.15 s CUDA setup cost dominate. On files below roughly
  200 MB the CPU engine wins outright. `--backend auto` is the default and
  simply uses whichever is available; it does **not** yet pick by file size.
- **Output rendering differs from jq** for `\uXXXX` escapes, `\/`, and
  exponent-form numbers. See [Differences from jq](#differences-from-jq). This
  is a deliberate trade for number fidelity, and query semantics are unaffected.
- **NDJSON only.** One JSON value per line. A single multi-gigabyte JSON
  document is not supported.
- **A subset of jq, not jq.** No `reduce`, `def`, `if/then/else`, `map`, regex,
  paths, slices, `.[]` iteration, or field-to-field comparison. The parser names
  each of these individually when it sees them.
- **Type errors skip the line.** jq aborts the whole run on
  `cannot index number with "b"`; warpjq skips the line, counts it, and warns.
  `--strict` aborts instead.
- **Malformed lines are skipped by default**, not fatal, because a 10 GB log
  with three truncated lines is the normal case. `--strict` aborts;
  `--skip-invalid` silences the warning.
- **Aggregates use `f64`**, like jq. Values are only formatted as integers below
  2^53. Pass-through and projection preserve the original bytes exactly.
- **CUDA only.** No ROCm or Metal. The CPU engine keeps AMD and Apple users as
  first-class users meanwhile.
- **Developed on Windows.** The code is portable and CI builds Linux, macOS and
  Windows, but the Linux release path is the least exercised of the three.

---

## Roadmap

Ordered by how much they would actually move the needle, based on the profile
above, not by how interesting they are to write.

- [ ] **Read directly into pinned memory.** The ~7 GB/s host copy is the
      bottleneck. A `read()` straight into the staging buffer removes both the
      copy and the mmap fault storm.
- [ ] **Remove the mid-submit stream sync.** `warpjq_submit` blocks to read the
      newline count. Passing it device-side would let chunk *n+1*'s copy overlap
      chunk *n*'s kernels.
- [ ] **Cut the ~0.15 s CUDA setup**, which is most of the small-file loss.
- [ ] **Auto backend selection by file size**, once the crossover is measured on
      more than one machine.
- [ ] Warp-cooperative structural scan, once the input path stops being the
      limit and the kernels start being it.
- [ ] gzip input (logs are usually gzipped; GPU decompress-then-parse).
- [ ] Regex `test()`.
- [ ] Full-document JSON, ROCm, Metal. Help wanted.

---

## License

MIT. See [LICENSE-MIT](LICENSE-MIT).

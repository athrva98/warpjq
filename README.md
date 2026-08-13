# warpjq

Filter, project and aggregate NDJSON using a subset of jq syntax. CUDA kernels
with a multi-core CPU fallback, so it runs with or without an NVIDIA GPU.

```console
$ warpjq 'select(.status == 500) | count' access.ndjson
59218

$ warpjq 'group_by(.host) | count' access.ndjson
{"host":"api-01","count":740301}
{"host":"web-01","count":739884}
```

---

## Benchmarks

1 GB of `warpjq gen --preset nginx`, warm page cache, best of three runs.
Times are end-to-end wall clock for the whole process, including reading the
file. There is no kernel-only number anywhere in this README.

Measured on an RTX 5060 Laptop GPU (sm_120, 8 GB), Windows 11, against
jq 1.8.2.

| query | warpjq (gpu) | warpjq (cpu) | jq 1.8.2 | speedup |
|---|---|---|---|---|
| `select(.status == 500) \| count` | 0.25 s | 0.33 s | 30.5 s | 121x |
| `select(.status == 500) \| sum(.bytes)` | 0.26 s | 0.34 s | 30.6 s | 119x |
| `select(.status >= 500) \| {p: .path, b: .bytes}` | 0.36 s | 0.34 s | 40.1 s | 119x |
| `select(.status == 500)` | 0.38 s | 0.34 s | 39.6 s | 118x |
| `group_by(.host) \| count` | 0.23 s | 0.07 s | 12.0 s | 165x |

The `group_by` row uses a 200 MB file. `group_by` makes jq slurp the whole
input into memory as parsed JSON, so it does not stream and does not scale to
1 GB. warpjq streams it.

Reproduce:

```bash
warpjq gen --preset nginx --size 1GB -o access.ndjson --seed 1
warpjq bench 'select(.status == 500) | count' access.ndjson
```

`warpjq bench` prints the exact command behind every row and each tool's own
output, and refuses to report a time for any tool whose answer disagrees with
warpjq's.

### GPU against CPU, by input size

The CUDA path is worth using above roughly 1.5 GB and not below it.
`select(.status == 500) | count`, best of three, H100 80 GB HBM3 with 8 host
cores:

| input | warpjq (gpu) | warpjq (cpu) | gpu / cpu |
|---|---|---|---|
| 1 MB | 0.217 s | 0.006 s | 0.03x |
| 200 MB | 0.223 s | 0.061 s | 0.27x |
| 1 GB | 0.286 s | 0.254 s | 0.89x |
| 2 GB | 0.845 s | 1.180 s | 1.40x |
| 4 GB | 1.417 s | 2.455 s | 1.73x |
| 8 GB | 2.719 s | 5.451 s | 2.00x |

CUDA context creation costs about 0.2 s whatever the input size, which is the
entire 1 MB figure. Past the crossover the GPU pipeline runs at roughly 3 GB/s
end-to-end against the CPU engine's 1.6 GB/s.

`--backend auto` uses whichever backend is available and does not yet pick by
file size. On inputs below a gigabyte, `--backend cpu` is faster.

**The speedup over jq is not the GPU's doing.** Both warpjq backends are two
orders of magnitude above jq, so the 100x comes from the query compiler, the
byte-slice value model and multi-core execution. The CUDA path adds a further
2x on large inputs.

### Verified on four architectures

The full suite, 224 tests including the differential comparison against jq,
passes on sm_80, sm_89, sm_90 and sm_120.

| device | cc | PCIe under load | kernels | host copy |
|---|---|---|---|---|
| RTX 5060 Laptop | 12.0 | gen 4 x8 | 20.2 GB/s | 7.0 GB/s |
| L40S | 8.9 | gen 4 x16 | 14.3 GB/s | 19.9 GB/s |
| A100 SXM4 40 GB | 8.0 | gen 4 x16 | 11.9 GB/s | 9.7 GB/s |
| H100 80 GB HBM3 | 9.0 | gen 5 x16 | 29.4 GB/s | 21.8 GB/s |

Kernel throughput does not track device tier. A laptop part beats both an A100
and an L40S. The kernel is one thread per line walking a byte at a time:
scalar, branch-heavy integer work that is latency and clock bound rather than
bandwidth bound, so HBM buys it nothing.

### Where the time goes

`WARPJQ_PROFILE=1` breaks a run into stages. H100, 8 GB:

```
warpjq profile: 8.59 GB through the pipeline
warpjq profile: read+chunk (host)        0.027s   322.65 GB/s
warpjq profile: copy to pinned           1.995s     4.31 GB/s
warpjq profile: submit (H2D+kernels)     0.203s    42.22 GB/s
warpjq profile: wait (sync+D2H)          0.001s 15887.49 GB/s
warpjq profile: merge+write              0.000s 66702.91 GB/s
```

The kernels sustain 42 GB/s. The host copy into pinned memory runs at 4.3 GB/s
and is 73% of the run: the GPU spends 0.203 s on the work and the host spends
1.995 s handing it the bytes. Removing that copy, by reading straight into the
staging buffer, would put the 8 GB run near 0.72 s, or 7.5x the CPU engine
rather than 2.0x. That is the first item on the [roadmap](#roadmap).

An earlier version of this README blamed a laptop's narrow PCIe link and
predicted a datacentre card would change the picture. Three cards with gen 4
and gen 5 x16 links, verified under load, did not. PCIe was not the
constraint.

[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) has the full matrix, the per-device
query breakdown, and instructions for reproducing the datacentre runs.

---

## Install

warpjq builds and runs with no GPU and no CUDA toolkit. The CPU engine is the
default build.

```bash
# CPU only. Works everywhere Rust works.
cargo install warpjq-cli

# With CUDA. Requires the CUDA toolkit (12.x) and an NVIDIA GPU of
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

The build script locates the CUDA toolkit and, on Windows, MSVC's `cl.exe`, so
`cargo build --features cuda` works outside a Developer Command Prompt. Set
`WARPJQ_CUDA_ARCH=86` (or your architecture) to cut compile time during
development.

---

## Usage

Generate a test file with no downloads:

```bash
warpjq gen --preset nginx --size 1GB -o access.ndjson --seed 1
```

```bash
# Count the 500s.
warpjq 'select(.status == 500) | count' access.ndjson

# Extract the slow requests as NDJSON.
warpjq 'select(.duration_ms > 2000) | {t: .ts, path: .path, ms: .duration_ms}' access.ndjson

# The same, as CSV.
warpjq 'select(.status >= 500) | {ts: .ts, path: .path}' --csv access.ndjson

# Bytes served per host.
warpjq 'group_by(.host) | sum(.bytes)' access.ndjson

# Nested paths.
warpjq gen --preset k8s --size 500MB -o pods.ndjson
warpjq 'select(.kubernetes.namespace == "prod") | count' pods.ndjson
```

Presets are `nginx`, `cloudtrail`, `k8s` and `nested`. `nested` contains deep
nesting, non-ASCII keys, escapes and integers past 2^53, so benchmarks are not
run only on flat data. `warpjq gen --list` describes them. The same seed always
produces the same bytes.

Flags: `--csv`, `--count`, `--backend auto|gpu|cpu`, `--strict`,
`--skip-invalid`, `-j/--threads`, `--chunk-size`, `--max-line-bytes`,
`--stats`.

Exit codes follow grep: 0 when rows matched, 1 when none did, 2 for an invalid
query.

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

Constructs outside the subset are rejected by name:

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

**One thread per line.** Chunks arrive newline-aligned, a scan turns newline
positions into `(offset, length)` pairs, and each thread runs the whole JSON
state machine for one line. A warp stages its 32 lines through shared memory
first, so the span is read with coalesced loads and parsed locally.

This is not the fastest possible shape. Occupancy and instruction count are
both measured dead ends on top of it, and what remains is the serial dependency
chain of one thread per line. Shortening that means warp-cooperative structural
indexing, which is a redesign rather than a tuning pass, and on current numbers
the host copy is worth more than anything left in the kernels. See
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

**The query compiles once, on the host.** `select(.a.b == 500) | {x: .c}`
becomes flat tables: a step list, a comparison table, and the condition in
reverse-Polish order. These upload once and every thread reads them. The kernel
never parses a query and never formats a key name. Even the `{"x":` bytes of a
projection are precomputed on the host, keeping JSON escaping of key names in
one place.

**No DOM.** Extracted values are slices of the input, so `1.0`, `0.10` and
`123456789012345678901234567890` come out as they went in. A DOM would
round-trip them through `f64` and lose digits. The test
`jq_preserves_number_literals_like_warpjq_does` asserts jq 1.8 keeps the same
literals intact, so a future jq that renormalises them fails the build.

**The kernel can decline a line.** Numbers outside the provably
correctly-rounded fast path, nesting past the 64-level stack, and strings that
would need materialising for CSV are marked `FALLBACK`, finished on the CPU by
the same evaluator, and merged back in input order. `--stats` reports the rate,
which is 0 on all four generated presets.

**Output order is input order.** Selection is a stable compaction over ascending
line indices, rows are written at offsets from a prefix sum, and CPU-finished
lines merge by line index.

**Double buffering.** One chunk uploads and computes while the previous one
drains. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) has the full pipeline,
the memory budget, and the measurements behind each decision.

---

## Correctness

224 tests, 90% line coverage with a floor enforced in CI.

- **Differential tests.** Randomised NDJSON covering unicode, escapes,
  surrogate pairs, integers past 2^53, duplicate keys, deep nesting, empty
  containers, CRLF, blank lines and malformed lines, asserting GPU output
  equals CPU output byte for byte across ~40 queries and 3 output formats.
  Small chunk sizes force multi-chunk paths, slot reuse and cross-chunk merging.
- **Against jq.** The same queries over a corpus excluding the spellings jq
  renormalises assert byte equality with `jq` itself. Separate tests pin every
  known [difference](#differences-from-jq) and assert semantics match where
  rendering does not. The exclusions are named in the test.
- **JSON conformance**, in the style of `nst/JSONTestSuite`: 48 must-accept
  cases, 52 must-reject cases and the implementation-defined ones, run through
  the scanner, both backends and jq. The set of validity divergences from jq is
  itself asserted, so a new one fails the build.
- **Invariance tests.** Output must not change with chunk size (1 byte to
  16 MB), thread count (1 to 32), output format or backend.
- **CLI tests.** Exit codes, stdin, multiple files, every flag and the error
  paths, driving the built binary.
- **Fuzzing** (`cargo fuzz`): three targets over the scanner, the query compiler
  and a full end-to-end run on arbitrary bytes. Nightly in CI.

Bugs found by review that the tests had missed, now covered by regressions:

1. The CPU scanner was recursive. A 50 KB line of `[[[[...`, well inside the
   default `--max-line-bytes`, overflowed the stack and aborted the process,
   which is not catchable. It took the GPU path down with it, since the kernel
   declines anything past its 64-level stack and hands the line to that scanner.
   Now iterative.
2. `k_emit` wrote assembled rows from prefix-sum offsets with no bound check,
   and the capacity test ran on the host afterwards. A six-field projection over
   short lines overran the device buffer, reported as `an illegal memory access
   was encountered`. The check is now in the kernel, before the store.
3. Under `--count`, lines the kernel declined were dropped from the tally. The
   merge inferred "no row" from "no bytes", and `--count` formats every row to
   zero bytes.

Bugs found by the test suites during development:

1. A sentinel collision made every device-side string comparison report
   "undecidable" at the terminator, routing 100% of `group_by` to the CPU.
   Output stayed correct, so only `--stats` or a performance assertion could
   have caught it.
2. A chunk routed to the CPU mid-pipeline was written while an earlier GPU chunk
   was still in flight, so its rows came out first. This affected every stdin
   input.
3. Aggregates did not span multiple files. `warpjq 'sum(.n)' a.ndjson b.ndjson`
   printed one total per file, and `group_by` emitted duplicate rows for keys
   present in more than one file.

---

## Differences from jq

Measured against jq 1.8.2. Every row is pinned by a test that fails if either
tool changes.

**Semantics match.** Which lines a filter selects, which group a value lands in,
and what `count`, `sum`, `min`, `max` and `avg` return all match jq, including
across different spellings of the same string. `select(.s == "A")` matches both
`{"s":"A"}` and `{"s":"\u0041"}`, and `group_by` puts them in one group.
`group_by` output is byte-identical to jq's, since group keys are decoded and
re-encoded on both sides.

**Rendering differs.** warpjq returns slices of the input; jq re-serialises.
Numbers agree, since jq preserves `1.0`, `0.10`, `9007199254740993` and
30-digit integers. Strings do not:

| input | jq prints | warpjq prints | |
|---|---|---|---|
| `{"s":"plain"}` | `"plain"` | `"plain"` | same |
| `{"s":"tab\there"}` | `"tab\there"` | `"tab\there"` | same |
| `{"s":"café"}` | `"café"` | `"café"` | same |
| `{"s":"\u0041"}` | `"A"` | `"\u0041"` | differs |
| `{"s":"\u65e5"}` | `"日"` | `"\u65e5"` | differs |
| `{"s":"\/slash"}` | `"/slash"` | `"\/slash"` | differs |
| `{"a":1e3}` | `1E+3` | `1e3` | differs |

jq normalises escapes and exponent notation; warpjq preserves the input
spelling. Logs written by Python's `json.dumps` at its default
`ensure_ascii=True` encode every non-ASCII character as `\uXXXX`, and warpjq
passes those through where jq decodes them. The values are equal, spelled
differently. Normalising output would require re-serialising every value, which
costs the number fidelity above. Not currently configurable.

**Validity differs, and warpjq is stricter.** jq 1.8 accepts several documents
RFC 8259 forbids, and rewrites the value rather than reporting an error:

| input | jq gives | warpjq |
|---|---|---|
| `{"a":01}` | `{"a":1}` | rejects the line |
| `{"a":-01}` | `{"a":-1}` | rejects the line |
| `{"a":+1}` | `{"a":1}` | rejects the line |
| `{"a":.5}` | `{"a":0.5}` | rejects the line |
| `{"a":1.}` | `{"a":1}` | rejects the line |
| `{"a":Infinity}` | `1.797...e+308` | rejects the line |
| `{"a":NaN}` | `null` | rejects the line |
| `{"a":1}{"b":2}` | two values | rejects the line |

The last row is a model difference: jq reads a stream of values, NDJSON is one
value per line.

In the other direction, warpjq accepts a lone high surrogate (`"\ud83d"`) and a
reversed surrogate pair, which jq rejects. jq's handling is asymmetric here: a
lone low surrogate (`"\ude00"`) is accepted by both.

---

## Limitations

- **The CUDA path only wins above roughly 1.5 GB.** CUDA context creation
  costs about 0.2 s regardless of input size, and the host copy into pinned
  memory runs at 4.3 GB/s against the kernels' 42 GB/s. Below the crossover
  the CPU engine wins, by 36x at 1 MB. `--backend auto` uses whichever backend
  is available and does not yet pick by file size.
- **Rendering differs from jq** for `\uXXXX` escapes, `\/` and exponent-form
  numbers. See [Differences from jq](#differences-from-jq).
- **NDJSON only.** One JSON value per line. A single multi-gigabyte JSON
  document is not supported.
- **A subset of jq.** No `reduce`, `def`, `if/then/else`, `map`, regex, paths,
  slices, `.[]` iteration or field-to-field comparison.
- **Type errors skip the line.** jq aborts the run on
  `cannot index number with "b"`; warpjq skips the line, counts it and warns.
  `--strict` aborts.
- **Malformed lines are skipped by default.** `--strict` aborts,
  `--skip-invalid` silences the warning.
- **Aggregates use `f64`**, as jq does, and format as integers only below 2^53.
  Pass-through and projection preserve the original bytes.
- **CUDA only.** No ROCm or Metal.
- **Developed on Windows.** CI builds Linux, macOS and Windows, but the Linux
  release path is the least exercised.

---

## Roadmap

Ordered by measured impact on the profile above.

- [ ] Read directly into pinned memory. The host copy is 73% of an 8 GB run on
      an H100, at 4.3 GB/s against the kernels' 42 GB/s. A `read()` into the
      staging buffer removes both the copy and the mmap fault storm, and would
      take that run from 2.0x the CPU engine to roughly 7.5x.
- [ ] Remove the mid-submit stream sync. `warpjq_submit` blocks to read the
      newline count; passing it device-side lets chunk *n+1*'s copy overlap
      chunk *n*'s kernels.
- [ ] Cut the 0.2 s CUDA setup, which is the entire cost of a 1 MB run and
      sets where the crossover lands.
- [ ] Backend selection by file size. The crossover is measured at 1 GB to
      2 GB on four devices, so `auto` has the numbers it needs to choose.
- [ ] Warp-cooperative structural scan, once the input path stops being the
      limit.
- [ ] gzip input, with GPU decompression before parsing.
- [ ] Regex `test()`.
- [ ] Full-document JSON, ROCm, Metal. Help wanted.

---

## License

MIT. See [LICENSE-MIT](LICENSE-MIT).

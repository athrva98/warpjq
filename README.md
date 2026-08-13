# warpjq

Filter, project and aggregate NDJSON using a subset of jq syntax. CUDA kernels
with a multi-core CPU fallback, so it runs with or without an NVIDIA GPU.

```console
$ warpjq 'select(.status == 500) | count' access.ndjson
59218
```

## Install

```bash
cargo install warpjq-cli                    # CPU only, works anywhere
cargo install warpjq-cli --features cuda    # needs CUDA 12.x, cc >= 6.0
```

From source: `cargo build --release [--features cuda]`. The build script
locates the CUDA toolkit and, on Windows, MSVC's `cl.exe`. Set
`WARPJQ_CUDA_ARCH=86` to compile for one architecture instead of six.

## Usage

```bash
warpjq gen --preset nginx --size 1GB -o access.ndjson --seed 1

warpjq 'select(.status == 500) | count'                           access.ndjson
warpjq 'select(.duration_ms > 2000) | {t: .ts, ms: .duration_ms}' access.ndjson
warpjq 'select(.status >= 500) | {ts: .ts, path: .path}' --csv    access.ndjson
warpjq 'group_by(.host) | sum(.bytes)'                            access.ndjson
```

Flags: `--csv`, `--count`, `--backend auto|gpu|cpu`, `--strict`,
`--skip-invalid`, `-j/--threads`, `--chunk-size`, `--max-line-bytes`,
`--stats`. Exit codes follow grep: 0 matched, 1 no match, 2 invalid query.

`gen` presets: `nginx`, `cloudtrail`, `k8s`, `nested`. Same seed, same bytes.
`nested` carries deep nesting, non-ASCII keys, escapes and integers past 2^53.

## Query subset

```text
paths        .a   .a.b   .a[0]   ."odd key"
filters      select(.x == 1)   !=  <  <=  >  >=   and   or   | not
projection   {key: .path, shorthand}
aggregates   count   sum(.f)   min(.f)   max(.f)   avg(.f)
grouping     group_by(.f) | <aggregate>
```

Anything outside the subset is rejected by name with a caret and a hint. No
`reduce`, `def`, `if`, `map`, regex, slices, `.[]`, or field-to-field
comparison.

## Benchmarks

1 GB `gen --preset nginx --seed 1`, warm cache, best of three, end-to-end wall
clock including file read. No kernel-only numbers.

RTX 5060 Laptop (sm_120), Windows 11, jq 1.8.2:

| query | gpu | cpu | jq | speedup |
|---|---|---|---|---|
| `select(.status == 500) \| count` | 0.25 s | 0.33 s | 30.5 s | 121x |
| `select(.status == 500) \| sum(.bytes)` | 0.26 s | 0.34 s | 30.6 s | 119x |
| `select(.status >= 500) \| {p: .path, b: .bytes}` | 0.36 s | 0.34 s | 40.1 s | 119x |
| `select(.status == 500)` | 0.38 s | 0.34 s | 39.6 s | 118x |
| `group_by(.host) \| count` (200 MB) | 0.23 s | 0.07 s | 12.0 s | 165x |

`group_by` uses 200 MB because it makes jq slurp the input into memory. jq
sustains ~0.035 GB/s here, so the speedup over jq does not depend on which
warpjq backend runs.

GPU against CPU, `select(.status == 500) | count`, H100 80 GB, 8 host cores:

| input | gpu | cpu | ratio |
|---|---|---|---|
| 1 MB | 0.217 s | 0.006 s | 0.03x |
| 200 MB | 0.223 s | 0.061 s | 0.27x |
| 1 GB | 0.286 s | 0.254 s | 0.89x |
| 2 GB | 0.845 s | 1.180 s | 1.40x |
| 4 GB | 1.417 s | 2.455 s | 1.73x |
| 8 GB | 2.719 s | 5.451 s | 2.00x |

CUDA context creation costs ~0.2 s regardless of size, which is the whole 1 MB
figure. Crossover is 1 GB to 2 GB. `--backend auto` does not yet pick by size;
below a gigabyte use `--backend cpu`.

Verified on sm_80, sm_89, sm_90 and sm_120. Reproduce with
`warpjq bench <query> <file>`, or `scripts/modal_ab.py` for datacentre parts.

## Differences from jq

Each row is pinned by a test that fails if either tool changes. Measured
against jq 1.8.2.

Semantics match: filtering, grouping and all five aggregates agree, including
across different spellings of the same string.

Rendering differs. warpjq returns slices of the input; jq re-serialises.
Numbers agree on jq 1.7+, which preserves literals; jq 1.6 prints
`123456789012345678901234567890` as `123456789012345680000000000000`. Strings
differ where jq normalises:

| input | jq | warpjq |
|---|---|---|
| `{"s":"\u0041"}` | `"A"` | `"\u0041"` |
| `{"s":"\/slash"}` | `"/slash"` | `"\/slash"` |
| `{"a":1e3}` | `1E+3` | `1e3` |

Normalising would mean re-serialising every value, costing the number fidelity
above. Not configurable.

Validity differs, and warpjq is stricter. jq 1.8 accepts documents RFC 8259
forbids, rewriting the value rather than erroring:

| input | jq | warpjq |
|---|---|---|
| `{"a":01}` | `{"a":1}` | rejects |
| `{"a":+1}` | `{"a":1}` | rejects |
| `{"a":.5}` | `{"a":0.5}` | rejects |
| `{"a":1.}` | `{"a":1}` | rejects |
| `{"a":Infinity}` | `1.797...e+308` | rejects |
| `{"a":NaN}` | `null` | rejects |
| `{"a":1}{"b":2}` | two values | rejects |

The last is a model difference: jq reads a value stream, NDJSON is one value
per line. Conversely warpjq accepts a lone high surrogate (`"\ud83d"`) and a
reversed pair, which jq rejects; a lone low surrogate is accepted by both.

## Correctness

224 tests, 90% line coverage, 85% floor enforced in CI.

| suite | asserts |
|---|---|
| differential | GPU output equals CPU output byte for byte, fuzzed input, ~40 queries by 3 formats, chunk sizes that force multi-chunk paths and slot reuse |
| against jq | byte equality on a corpus excluding the spellings jq renormalises, which are named in the test and asserted separately |
| conformance | 48 must-accept and 52 must-reject documents through the scanner, both backends and jq; the divergence set above is itself asserted |
| invariance | output unchanged across chunk size 1 B to 16 MB, thread count 1 to 32, format and backend |
| CLI | exit codes, stdin, multiple files, every flag, error paths |
| fuzz | scanner, query compiler, and a full end-to-end run on arbitrary bytes |

## Limitations

- The CUDA path only wins above ~1.5 GB.
- NDJSON only. One value per line, no single multi-GB document.
- A subset of jq, not jq.
- Type errors skip the line and warn; jq aborts. `--strict` aborts.
- Malformed lines are skipped by default. `--strict` aborts, `--skip-invalid`
  silences.
- Aggregates use `f64`, as jq does. Pass-through preserves input bytes exactly.
- CUDA only. No ROCm or Metal.

## Roadmap

Ordered by measured impact.

- [ ] Read into pinned memory directly. The host copy is 73% of an 8 GB H100
      run, at 4.3 GB/s against the kernels' 42 GB/s.
- [ ] Remove the mid-submit stream sync that serialises chunk *n+1*'s copy
      against chunk *n*'s kernels.
- [ ] Cut the 0.2 s CUDA setup, which sets where the crossover lands.
- [ ] Backend selection by file size.
- [ ] Warp-cooperative structural indexing. See
      [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for why it is not first.
- [ ] gzip input, regex `test()`, full-document JSON, ROCm, Metal.

## License

MIT. See [LICENSE-MIT](LICENSE-MIT).

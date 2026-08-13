# warpjq

[![CI](https://github.com/athrva98/warpjq/actions/workflows/ci.yml/badge.svg)](https://github.com/athrva98/warpjq/actions/workflows/ci.yml)

Filter, project and aggregate NDJSON with jq syntax. Uses the GPU when there
is one.

```console
$ warpjq 'select(.status == 500) | count' access.ndjson
59218

$ warpjq 'select(.status >= 500) | {t: .ts, path: .path}' access.ndjson
{"t":1700000005,"path":"/health"}
{"t":1700000018,"path":"/api/v1/orders"}

$ warpjq 'group_by(.host) | sum(.bytes)' access.ndjson
{"host":"api-01","sum":144285653795}
{"host":"web-01","sum":144104882301}
```

## Install

```bash
cargo install warpjq-cli                    # CPU only
cargo install warpjq-cli --features cuda    # adds the GPU backend
```

The CUDA feature needs the toolkit installed and a device of compute
capability 6.0 or newer. The build script finds the toolkit, picks the
architectures your `nvcc` supports, and on Windows locates MSVC's `cl.exe`
itself.

## Usage

```
warpjq [OPTIONS] <QUERY> [FILE]...
warpjq gen --preset <NAME> --size <SIZE> -o <FILE>
warpjq bench <QUERY> <FILE>
```

Reads stdin when no file is given. Multiple files are one stream, so
aggregates span all of them.

| option | |
|---|---|
| `--csv` | CSV instead of NDJSON; needs a projection or aggregate |
| `-c`, `--count` | print only how many rows matched |
| `--backend auto\|gpu\|cpu` | default `auto` |
| `--strict` | stop on the first malformed line |
| `--skip-invalid` | skip malformed lines without warning |
| `-j`, `--threads` | CPU worker threads, 0 for one per core |
| `--chunk-size`, `--max-line-bytes` | accept `256MB`, `1GB` |
| `--stats` | timing and throughput to stderr |

Exit codes follow grep: 0 matched, 1 no match, 2 invalid query.

`warpjq gen` writes reproducible test data. Presets are `nginx`,
`cloudtrail`, `k8s` and `nested`; the same seed always produces the same
bytes.

## Syntax

```
paths        .a   .a.b   .a[0]   ."odd key"
filters      select(.x == 1)   !=  <  <=  >  >=   and   or   | not
projection   {key: .path, shorthand}
aggregates   count   sum(.f)   min(.f)   max(.f)   avg(.f)
grouping     group_by(.f) | <aggregate>
```

Constructs outside this subset are rejected by name with a caret and a
suggestion. There is no `reduce`, `def`, `if`, `map`, regex, slicing, `.[]`
iteration, or field-to-field comparison.

## Performance

On 1 GB of nginx-style logs, warpjq runs a filter-and-count in 0.25 s against
30.5 s for jq 1.8.2. Both backends are well over two orders of magnitude
faster than jq, so the margin does not depend on having a GPU.

The CUDA backend overtakes the CPU one above roughly 1.5 GB, reaching 2x at
8 GB. Below that the CPU backend is faster, by a wide margin on small inputs,
because CUDA context creation costs about 0.2 s regardless of input size.
`--backend auto` does not yet choose by size.

`warpjq bench <query> <file>` reproduces these against whatever jq you have
installed. Full tables, per-device figures and the profile breakdown are in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Compatibility with jq

Which lines a filter selects, which group a value lands in, and what the
aggregates return all match jq.

Output rendering differs in one respect: warpjq returns slices of the input
rather than re-serialising, so `\u0041`, `\/` and `1e3` come back as
written where jq prints `A`, `/` and `1E+3`. The values are equal.

Input validity differs in the other direction: jq accepts and silently
rewrites some documents RFC 8259 forbids, such as `{"a":01}` and `{"a":NaN}`.
warpjq rejects those lines.

Both sets are enumerated in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and
checked against jq in CI.

## Limitations

- NDJSON only. One JSON value per line.
- A subset of jq, not jq.
- Malformed lines are skipped with a warning by default; `--strict` aborts.
  A path that hits the wrong type skips the line, where jq aborts the run.
- Aggregates use `f64`, as jq does. Values passing through unchanged keep
  their exact input bytes.
- CUDA only. No ROCm or Metal.

## License

MIT. See [LICENSE-MIT](LICENSE-MIT).

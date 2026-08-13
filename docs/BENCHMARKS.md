# Benchmarks

Every number here is end-to-end wall clock for the whole process, including
reading the file. There is no kernel-only timing anywhere in this document.

Reproduce any of it with:

```bash
warpjq gen --preset nginx --size 1GB -o access.ndjson --seed 1
warpjq bench 'select(.status == 500) | count' access.ndjson
```

The datacentre runs come from `scripts/modal_bench.py`, which builds a pinned
commit from the public repository on a Modal container, runs the full test
suite on the device, and then benchmarks.

---

## 1. Against jq

1 GB of `warpjq gen --preset nginx --seed 1`, warm page cache, best of three
runs, on an RTX 5060 Laptop GPU (sm_120, 8 GB) under Windows 11, against
jq 1.8.2.

| query | warpjq (gpu) | warpjq (cpu) | jq 1.8.2 | speedup |
|---|---|---|---|---|
| `select(.status == 500) \| count` | 0.25 s | 0.33 s | 30.5 s | 121x |
| `select(.status == 500) \| sum(.bytes)` | 0.26 s | 0.34 s | 30.6 s | 119x |
| `select(.status >= 500) \| {p: .path, b: .bytes}` | 0.36 s | 0.34 s | 40.1 s | 119x |
| `select(.status == 500)` | 0.38 s | 0.34 s | 39.6 s | 118x |
| `group_by(.host) \| count` (200 MB) | 0.23 s | 0.07 s | 12.0 s | 165x |

`group_by` uses a 200 MB file because it makes jq slurp the whole input into
memory as parsed JSON. It does not stream and does not scale to 1 GB.

jq sustains about 0.035 GB/s on this workload. Both warpjq backends are two
orders of magnitude above that, so the speedup over jq is not sensitive to
which backend runs.

---

## 2. GPU against CPU, by input size

This is the interesting axis, and the one where an earlier version of this
document was wrong.

`select(.status == 500) | count`, best of three, H100 80 GB HBM3 with 8 host
cores. Sizes at and below 1 GB are from the three-device sweep in section 3;
2 GB and above are from a separate scaling run on the H100.

| input | warpjq (gpu) | warpjq (cpu) | gpu / cpu |
|---|---|---|---|
| 1 MB | 0.217 s | 0.006 s | 0.03x |
| 50 MB | 0.216 s | 0.019 s | 0.09x |
| 200 MB | 0.223 s | 0.061 s | 0.27x |
| 1 GB | 0.286 s | 0.254 s | 0.89x |
| 2 GB | 0.845 s | 1.180 s | **1.40x** |
| 4 GB | 1.417 s | 2.455 s | **1.73x** |
| 8 GB | 2.719 s | 5.451 s | **2.00x** |

The crossover is between 1 GB and 2 GB. Below it the CPU engine wins, by 36x
at 1 MB. Above it the GPU wins, reaching 2.0x at 8 GB and still climbing
slowly.

The shape is a fixed cost plus a faster marginal rate. CUDA context creation
and buffer allocation cost about 0.2 s regardless of input size, which is the
entire 1 MB figure. After that the GPU pipeline runs at roughly 3 GB/s
end-to-end against the CPU engine's 1.6 GB/s at these sizes.

---

## 3. Across devices

Same 1 GB file, same queries, 8 host cores, jq 1.7.1. PCIe link sampled while
a transfer was in flight, because an idle card downclocks its link and reports
a misleadingly narrow one.

| device | cc | PCIe under load | kernels (GB/s) | host copy (GB/s) | tests |
|---|---|---|---|---|---|
| RTX 5060 Laptop | 12.0 | gen 4 x8 | 20.2 | 7.0 | 224 passed |
| L40S | 8.9 | gen 4 x16 | 14.3 | 19.9 | 224 passed |
| A100 SXM4 40 GB | 8.0 | gen 4 x16 | 11.9 | 9.7 | 224 passed |
| H100 80 GB HBM3 | 9.0 | gen 5 x16 | 29.4 | 21.8 | 224 passed |

At 1 GB the CPU engine wins on all three datacentre parts:

| query, 1 GB | L40S | A100 | H100 |
|---|---|---|---|
| `select \| count` | 0.281 / **0.272** | 0.423 / **0.299** | 0.297 / **0.288** |
| `select \| sum(.bytes)` | 0.281 / **0.263** | 0.424 / **0.265** | 0.294 / **0.276** |
| projection | 0.449 / **0.261** | 0.682 / **0.265** | 0.428 / **0.259** |
| passthrough | 0.438 / **0.266** | 0.686 / **0.265** | 0.489 / **0.275** |
| `group_by \| count` | 0.197 / **0.055** | 0.283 / **0.057** | 0.231 / **0.061** |

Times are gpu / cpu; the winner is in bold. This is consistent with section 2:
1 GB sits just below the crossover, so the fixed CUDA cost has not amortised.

**Kernel throughput does not track device tier.** An RTX 5060 Laptop part beats
both an A100 and an L40S at the kernel stage. The kernel is one thread per
line walking a byte at a time: scalar, branch-heavy integer work that is
latency and clock bound rather than bandwidth bound. It gets nothing from HBM
and little from the A100's 108 SMs at a lower clock. The H100 leads on clock
and cache, not on memory bandwidth.

---

## 4. Where the time goes at scale

H100, `select(.status == 500) | count`, from `WARPJQ_PROFILE=1`:

| stage | 2 GB | 4 GB | 8 GB |
|---|---|---|---|
| read and chunk | 0.004 s | 0.015 s | 0.027 s |
| copy to pinned | 0.463 s | 0.981 s | **1.995 s** |
| submit (H2D and kernels) | 0.051 s | 0.102 s | 0.203 s |
| wait (sync and D2H) | 0.000 s | 0.000 s | 0.001 s |
| merge and write | 0.000 s | 0.000 s | 0.000 s |
| **total wall** | 0.845 s | 1.417 s | 2.719 s |

**The kernels sustain 42.2 GB/s at every size.** The host copy into pinned
memory runs at 4.3 GB/s and accounts for 73% of the 8 GB run. The GPU spends
0.203 s doing the work and the host spends 1.995 s handing it the bytes.

That single stage is the whole optimisation target. Reading straight into the
pinned staging buffer, instead of faulting an mmap in and then copying it,
would remove almost all of it. If it went to zero, the 8 GB run would be about
0.72 s against the CPU engine's 5.45 s, or 7.5x rather than 2.0x.

---

## 5. What this means

- **Against jq, warpjq is roughly 100x faster**, and that holds on either
  backend. Nothing in this document threatens that number.
- **The CUDA path is worth using above about 1.5 GB**, and not below it. The
  default `--backend auto` does not yet pick by size, which is a known gap.
- **The GPU is not the source of the speedup over jq.** The query compiler,
  the byte-slice value model and multi-core execution are. The CUDA path adds
  a further 2x on large inputs.
- **The bottleneck is the host copy, not PCIe and not the kernels.** An earlier
  version of this document blamed a laptop's narrow PCIe link and predicted a
  datacentre card would fix it. Three datacentre cards with gen 4 and gen 5 x16
  links, verified under load, did not fix it. That hypothesis was wrong.
- **The kernels are correct on four architectures.** 224 tests, including the
  full differential suite against jq, pass on sm_80, sm_89, sm_90 and sm_120.

---

## Reproducing the datacentre numbers

```bash
pip install modal && modal setup
modal run scripts/modal_bench.py --gpus L40S,A100,H100   # sections 3 and 4
modal run scripts/modal_bench.py::scale                  # section 2
```

The script pins a commit hash, builds it from the public repository, and runs
the test suite on the device before reporting any timing. A device that fails
the tests reports the failure instead of a benchmark.

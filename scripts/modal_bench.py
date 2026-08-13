"""Benchmark warpjq on datacentre GPUs via Modal.

The numbers in the README were taken on an RTX 5060 Laptop, where the host
cannot feed the kernels fast enough for the CUDA path to beat the CPU engine.
This runs the identical benchmark on parts with a full PCIe link and far more
memory bandwidth, to find out whether that conclusion is a property of the
design or of the laptop.

It also runs the full test suite on each device. The kernels are compiled for
sm_120 locally and have never executed on any other architecture, so agreement
between backends on sm_80 and sm_90 is worth as much as the timings.

    modal run scripts/modal_bench.py                 # all GPU types
    modal run scripts/modal_bench.py --gpus A100     # just one
"""

import json
import subprocess

import modal

# Pin the commit so a re-run measures the same code.
COMMIT = "f99e1269880febf6a637fb050dad378210a586e4"
REPO = "https://github.com/athrva98/warpjq"

# sm_80 A100, sm_89 L40S and Ada, sm_90 H100/H200.
CUDA_ARCHS = "80,89,90"

app = modal.App("warpjq-bench")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.2-devel-ubuntu24.04", add_python="3.11"
    )
    .apt_install("git", "curl", "build-essential", "jq", "pkg-config", "bc")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
        "| sh -s -- -y --profile minimal --default-toolchain stable",
        f"git clone {REPO} /warpjq && cd /warpjq && git checkout {COMMIT}",
        # Build once, in the image layer, so every GPU type reuses it.
        f"cd /warpjq && . $HOME/.cargo/env && WARPJQ_CUDA_ARCH='{CUDA_ARCHS}' "
        "cargo build --release -p warpjq-cli --features cuda",
        f"cd /warpjq && . $HOME/.cargo/env && WARPJQ_CUDA_ARCH='{CUDA_ARCHS}' "
        "cargo build --release --tests --features warpjq-cli/cuda",
    )
)

QUERIES = [
    "select(.status == 500) | count",
    "select(.status == 500) | sum(.bytes)",
    "select(.status >= 500) | {p: .path, b: .bytes}",
    "select(.status == 500)",
]


def sh(cmd: str, **kw) -> str:
    """Run a command under the cargo environment and return its combined output."""
    full = f'. $HOME/.cargo/env && cd /warpjq && {cmd}'
    p = subprocess.run(
        ["bash", "-lc", full], capture_output=True, text=True, **kw
    )
    return (p.stdout or "") + (p.stderr or "")


def device_facts() -> dict:
    q = (
        "name,memory.total,driver_version,compute_cap,"
        "pcie.link.gen.max,pcie.link.width.max,"
        "pcie.link.gen.current,pcie.link.width.current"
    )
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    keys = [
        "name", "memory", "driver", "compute_cap",
        "pcie_gen_max", "pcie_width_max",
        "pcie_gen_now", "pcie_width_now",
    ]
    return dict(zip(keys, [v.strip() for v in out.split(",")]))


def pcie_under_load(warpjq: str, path: str) -> str:
    """PCIe link state sampled while a transfer is actually running.

    An idle card downclocks its link, so `nvidia-smi` at rest reports
    something like "gen 1 x8" on a card wired for gen 4 x16. Reading it
    during a run is the only way to know what the transfers actually got.
    """
    proc = subprocess.Popen(
        ["bash", "-lc",
         f". $HOME/.cargo/env && cd /warpjq && for i in 1 2 3 4 5; do "
         f"{warpjq} 'select(.status == 500) | count' --backend gpu {path} "
         "> /dev/null; done"],
    )
    best = ""
    while proc.poll() is None:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=pcie.link.gen.current,pcie.link.width.current",
             "--format=csv,noheader"],
            capture_output=True, text=True,
        ).stdout.strip()
        if out and out > best:
            best = out
    proc.wait()
    return best


def run_on_device(label: str) -> dict:
    result = {"gpu": label, "device": device_facts()}

    cpus = subprocess.run(["nproc"], capture_output=True, text=True).stdout.strip()
    result["cpu_cores"] = cpus
    result["jq_version"] = subprocess.run(
        ["jq", "--version"], capture_output=True, text=True
    ).stdout.strip()

    W = "./target/release/warpjq"

    # Correctness first. A fast wrong answer is not a result.
    print(f"[{label}] running the test suite", flush=True)
    tests = sh(
        ". $HOME/.cargo/env && "
        f"WARPJQ_CUDA_ARCH='{CUDA_ARCHS}' cargo test --workspace "
        "--features warpjq-cli/cuda --release -- --test-threads=1",
        timeout=3600,
    )
    passed = sum(
        int(line.split()[3])
        for line in tests.splitlines()
        if line.startswith("test result:")
    )
    result["tests_passed"] = passed
    result["tests_failed"] = "FAILED" in tests
    if result["tests_failed"]:
        result["test_output"] = tests[-4000:]

    print(f"[{label}] generating data", flush=True)
    sh(f"{W} gen --preset nginx --size 1GB -o /tmp/big.ndjson --seed 1")
    sh(f"{W} gen --preset nginx --size 200MB -o /tmp/small.ndjson --seed 1")
    sh("cat /tmp/big.ndjson > /dev/null")

    result["pcie_under_load"] = pcie_under_load(W, "/tmp/big.ndjson")

    # Per-stage profile, which is where the laptop result came from.
    print(f"[{label}] profiling", flush=True)
    result["profile"] = sh(
        f"WARPJQ_PROFILE=1 {W} 'group_by(.host) | count' "
        "--backend gpu --stats /tmp/big.ndjson > /dev/null"
    )

    # Raw per-backend timings, so the GPU-vs-CPU margin is visible without
    # depending on jq being present.
    timings = {}
    for q in QUERIES + ["group_by(.host) | count"]:
        f = "/tmp/small.ndjson" if q.startswith("group_by") else "/tmp/big.ndjson"
        for backend in ("gpu", "cpu"):
            best = None
            for _ in range(3):
                out = sh(
                    f"{W} '{q}' --backend {backend} --stats {f} > /dev/null"
                )
                for line in out.splitlines():
                    if " in " in line and "GB/s" in line:
                        secs = float(line.split(" in ")[1].split("s ")[0])
                        best = secs if best is None else min(best, secs)
            timings[f"{q} [{backend}]"] = best
    result["timings"] = timings

    # The full comparison table, including jq.
    print(f"[{label}] benchmarking against jq", flush=True)
    tables = {}
    for q in QUERIES:
        tables[q] = sh(
            f"{W} bench '{q}' /tmp/big.ndjson --runs 3 --warmup 1", timeout=1800
        )
    tables["group_by(.host) | count"] = sh(
        f"{W} bench 'group_by(.host) | count' /tmp/small.ndjson "
        "--runs 1 --warmup 0",
        timeout=1800,
    )
    result["tables"] = tables

    # Small-file crossover, where the laptop loses badly to CUDA setup cost.
    sh(f"{W} gen --preset nginx --size 1MB -o /tmp/tiny.ndjson --seed 3")
    sh(f"{W} gen --preset nginx --size 50MB -o /tmp/mid.ndjson --seed 3")
    cross = {}
    for size, f in (("1MB", "/tmp/tiny.ndjson"), ("50MB", "/tmp/mid.ndjson"),
                    ("200MB", "/tmp/small.ndjson"), ("1GB", "/tmp/big.ndjson")):
        sh(f"cat {f} > /dev/null")
        for backend in ("gpu", "cpu"):
            best = None
            for _ in range(3):
                out = sh(
                    f"{W} 'select(.status == 500) | count' "
                    f"--backend {backend} --stats {f} > /dev/null"
                )
                for line in out.splitlines():
                    if " in " in line and "GB/s" in line:
                        secs = float(line.split(" in ")[1].split("s ")[0])
                        best = secs if best is None else min(best, secs)
            cross[f"{size} [{backend}]"] = best
    result["crossover"] = cross

    return result


@app.function(image=image, gpu="A100", cpu=8.0, memory=32768, timeout=5400)
def bench_a100():
    return run_on_device("A100-40GB")


@app.function(image=image, gpu="H100", cpu=8.0, memory=32768, timeout=5400)
def bench_h100():
    return run_on_device("H100")


@app.function(image=image, gpu="L40S", cpu=8.0, memory=32768, timeout=5400)
def bench_l40s():
    return run_on_device("L40S")


@app.local_entrypoint()
def main(gpus: str = "A100,H100,L40S"):
    wanted = [g.strip().upper() for g in gpus.split(",") if g.strip()]
    fns = {"A100": bench_a100, "H100": bench_h100, "L40S": bench_l40s}

    results = []
    for name in wanted:
        fn = fns.get(name)
        if fn is None:
            print(f"unknown gpu {name}, skipping")
            continue
        print(f"=== dispatching {name} ===", flush=True)
        results.append(fn.remote())

    with open("modal_results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    for r in results:
        d = r["device"]
        print("=" * 72)
        print(f"{r['gpu']}: {d['name']}  cc {d['compute_cap']}  {d['memory']}")
        print(
            f"  PCIe idle: gen {d['pcie_gen_now']} x{d['pcie_width_now']} | "
            f"under load: {r.get('pcie_under_load', '?')} | "
            f"max: gen {d['pcie_gen_max']} x{d['pcie_width_max']}"
        )
        print(f"  host: {r['cpu_cores']} cores, {r['jq_version']}")
        status = "FAILED" if r["tests_failed"] else "passed"
        print(f"  tests: {r['tests_passed']} {status}")
        print("  profile:")
        for line in r["profile"].splitlines():
            if "profile:" in line or "GB/s |" in line:
                print("   ", line.strip())
        print("  timings (best of 3, seconds):")
        for k, v in r["timings"].items():
            print(f"    {k:<52} {v}")
        print("  crossover:")
        for k, v in r["crossover"].items():
            print(f"    {k:<20} {v}")
    print("\nwrote modal_results.json")


@app.function(image=image, gpu="H100", cpu=8.0, memory=65536, timeout=5400)
def scaling():
    """Find the size where the CUDA path overtakes the CPU engine, if it does.

    At 1 GB the GPU loses on every device tested. The per-stage profile says
    the kernels are fast and the fixed CUDA setup is roughly 0.2 s, which
    predicts a crossover somewhere above 1 GB. Measure it rather than
    extrapolate from two points.
    """
    W = "./target/release/warpjq"
    out = {"device": device_facts(), "sizes": {}}

    for size in ("2GB", "4GB", "8GB"):
        sh(f"{W} gen --preset nginx --size {size} -o /tmp/scale.ndjson --seed 1")
        sh("cat /tmp/scale.ndjson > /dev/null")
        row = {}
        for backend in ("gpu", "cpu"):
            best = None
            for _ in range(3):
                t = sh(
                    f"{W} 'select(.status == 500) | count' "
                    f"--backend {backend} --stats /tmp/scale.ndjson > /dev/null"
                )
                for line in t.splitlines():
                    if " in " in line and "GB/s" in line:
                        secs = float(line.split(" in ")[1].split("s ")[0])
                        best = secs if best is None else min(best, secs)
            row[backend] = best
        row["profile"] = sh(
            f"WARPJQ_PROFILE=1 {W} 'select(.status == 500) | count' "
            "--backend gpu /tmp/scale.ndjson > /dev/null"
        )
        out["sizes"][size] = row
        print(f"[scaling] {size}: gpu={row['gpu']} cpu={row['cpu']}", flush=True)
        sh("rm -f /tmp/scale.ndjson")

    return out


@app.local_entrypoint()
def scale():
    r = scaling.remote()
    print(f"device: {r['device']['name']}")
    for size, row in r["sizes"].items():
        speedup = row["cpu"] / row["gpu"] if row["gpu"] else 0
        print(f"  {size:<5} gpu={row['gpu']:<7} cpu={row['cpu']:<7} "
              f"gpu is {speedup:.2f}x the cpu")
        for line in row["profile"].splitlines():
            if "profile:" in line:
                print("      ", line.strip())

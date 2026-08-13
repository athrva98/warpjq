"""A/B two commits on datacentre GPUs, on identical hardware and data.

Both commits are built into the same image, so a run compares two binaries on
one device with one dataset rather than comparing numbers taken on different
machines at different times.

The local attempt at this A/B was taken on a laptop on battery, with the GPU
in P8 at 180 MHz of a 3090 MHz SM clock and 405 MHz of a 12001 MHz memory
clock. Every number from it was meaningless. This records the clocks and the
active throttle reasons alongside the timings so that cannot happen quietly.

    modal run scripts/modal_ab.py                       # all three GPUs
    modal run scripts/modal_ab.py --gpus H100           # one
"""

import json
import subprocess

import modal

BASE = "fb39309d2d06c8bc8d24d0c9b81b4094766305d8"  # main, before the rewrite
HEAD = "fc2e7b1ad7b9b1a7f29b2f37e511cb79a1963ecf"  # kernel-optimization
REPO = "https://github.com/athrva98/warpjq"

CUDA_ARCHS = "80,89,90"

app = modal.App("warpjq-ab")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.2-devel-ubuntu24.04", add_python="3.11"
    )
    .apt_install("git", "curl", "build-essential", "jq", "pkg-config")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
        "| sh -s -- -y --profile minimal --default-toolchain stable",
        f"git clone {REPO} /base && cd /base && git checkout {BASE}",
        f"git clone {REPO} /head && cd /head && git checkout {HEAD}",
        f"cd /base && . $HOME/.cargo/env && WARPJQ_CUDA_ARCH='{CUDA_ARCHS}' "
        "cargo build --release -p warpjq-cli --features cuda",
        f"cd /head && . $HOME/.cargo/env && WARPJQ_CUDA_ARCH='{CUDA_ARCHS}' "
        "cargo build --release -p warpjq-cli --features cuda",
        f"cd /head && . $HOME/.cargo/env && WARPJQ_CUDA_ARCH='{CUDA_ARCHS}' "
        "cargo build --release --tests --features warpjq-cli/cuda",
    )
)

BASE_BIN = "/base/target/release/warpjq"
HEAD_BIN = "/head/target/release/warpjq"

QUERIES = [
    "select(.status == 500) | count",
    "select(.status >= 500) | {p: .path, b: .bytes}",
    "group_by(.host) | count",
    "select(.status == 500)",
]


def sh(cmd: str, cwd: str = "/head", **kw) -> str:
    full = f". $HOME/.cargo/env && cd {cwd} && {cmd}"
    p = subprocess.run(["bash", "-lc", full], capture_output=True, text=True, **kw)
    return (p.stdout or "") + (p.stderr or "")


def clocks() -> dict:
    q = (
        "clocks.current.sm,clocks.max.sm,clocks.current.memory,"
        "clocks.max.memory,power.draw,power.max_limit,pstate,"
        "clocks_throttle_reasons.active"
    )
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout.strip()
    keys = ["sm_now", "sm_max", "mem_now", "mem_max", "power", "power_max",
            "pstate", "throttle"]
    return dict(zip(keys, [v.strip() for v in out.split(",")]))


def timed(binary: str, query: str, path: str, runs: int = 7) -> float:
    """Best-of-N wall clock for one binary, from --stats."""
    best = None
    for _ in range(runs):
        out = sh(f"{binary} '{query}' --backend gpu --stats {path} > /dev/null")
        for line in out.splitlines():
            if " in " in line and "GB/s" in line:
                secs = float(line.split(" in ")[1].split("s ")[0])
                best = secs if best is None else min(best, secs)
    return best


def stage_profile(binary: str, query: str, path: str) -> dict:
    out = sh(
        f"WARPJQ_PROFILE=1 {binary} '{query}' --backend gpu {path} > /dev/null"
    )
    stages = {}
    for line in out.splitlines():
        if "warpjq profile:" not in line or "through the pipeline" in line:
            continue
        body = line.split("warpjq profile:", 1)[1].strip()
        parts = body.rsplit(None, 3)
        if len(parts) >= 2:
            stages[parts[0].strip()] = parts[1]
    return stages


def run_ab(label: str) -> dict:
    result = {"gpu": label, "clocks_idle": clocks()}

    # The branch has never run on this architecture. Correctness first.
    tests = sh(
        f"WARPJQ_CUDA_ARCH='{CUDA_ARCHS}' cargo test --workspace "
        "--features warpjq-cli/cuda --release -- --test-threads=1",
        timeout=3600,
    )
    result["tests_passed"] = sum(
        int(l.split()[3]) for l in tests.splitlines() if l.startswith("test result:")
    )
    result["tests_failed"] = "FAILED" in tests
    if result["tests_failed"]:
        result["test_output"] = tests[-4000:]

    result["sizes"] = {}
    for size in ("200MB", "1GB", "4GB"):
        sh(f"{HEAD_BIN} gen --preset nginx --size {size} -o /tmp/ab.ndjson --seed 1")
        sh("cat /tmp/ab.ndjson > /dev/null")

        # Both binaries must agree before either is timed.
        mismatches = []
        for q in QUERIES:
            sh(f"{BASE_BIN} '{q}' --backend gpu /tmp/ab.ndjson > /tmp/b.out")
            sh(f"{HEAD_BIN} '{q}' --backend gpu /tmp/ab.ndjson > /tmp/h.out")
            if "differ" in sh("cmp /tmp/b.out /tmp/h.out || echo differ"):
                mismatches.append(q)

        rows = {}
        for q in QUERIES:
            b = timed(BASE_BIN, q, "/tmp/ab.ndjson")
            h = timed(HEAD_BIN, q, "/tmp/ab.ndjson")
            rows[q] = {"base": b, "head": h}

        result["sizes"][size] = {
            "mismatches": mismatches,
            "timings": rows,
            "profile_base": stage_profile(BASE_BIN, QUERIES[1], "/tmp/ab.ndjson"),
            "profile_head": stage_profile(HEAD_BIN, QUERIES[1], "/tmp/ab.ndjson"),
            "clocks_after": clocks(),
        }
        print(f"[{label}] {size} done", flush=True)
        sh("rm -f /tmp/ab.ndjson")

    return result


@app.function(image=image, gpu="L40S", cpu=8.0, memory=65536, timeout=7200)
def ab_l40s():
    return run_ab("L40S")


@app.function(image=image, gpu="A100", cpu=8.0, memory=65536, timeout=7200)
def ab_a100():
    return run_ab("A100-40GB")


@app.function(image=image, gpu="H100", cpu=8.0, memory=65536, timeout=7200)
def ab_h100():
    return run_ab("H100")


@app.local_entrypoint()
def main(gpus: str = "L40S,A100,H100"):
    fns = {"L40S": ab_l40s, "A100": ab_a100, "H100": ab_h100}
    results = [
        fns[g.strip().upper()].remote()
        for g in gpus.split(",")
        if g.strip().upper() in fns
    ]

    with open("modal_ab_results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    for r in results:
        c = r["clocks_idle"]
        print("=" * 78)
        print(f"{r['gpu']}  sm {c['sm_now']}/{c['sm_max']}  "
              f"mem {c['mem_now']}/{c['mem_max']}  {c['pstate']}  "
              f"throttle {c['throttle']}")
        status = "FAILED" if r["tests_failed"] else "passed"
        print(f"  tests on this arch: {r['tests_passed']} {status}")
        for size, s in r["sizes"].items():
            if s["mismatches"]:
                print(f"  {size}: OUTPUT MISMATCH on {s['mismatches']}")
            print(f"  --- {size} (clocks after: sm {s['clocks_after']['sm_now']}, "
                  f"{s['clocks_after']['pstate']}) ---")
            for q, t in s["timings"].items():
                sp = t["base"] / t["head"] if t["head"] else 0
                print(f"    {q:<46} {t['base']:>7.3f} -> {t['head']:>7.3f}  {sp:.2f}x")
            print(f"    stages, projection query:")
            for k in s["profile_base"]:
                print(f"      {k:<24} {s['profile_base'][k]:>8} -> "
                      f"{s['profile_head'].get(k, '?'):>8}")
    print("\nwrote modal_ab_results.json")

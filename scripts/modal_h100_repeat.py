"""Re-run the one unexplained row from the A/B, across fresh containers.

At 200 MB on an H100, three of four queries showed the kernel-optimization
branch 16% to 23% slower than main, while the same queries were flat on an
L40S and an A100, and flat on the H100 itself at 1 GB and 4 GB. That was one
container with no repeats, so it is equally consistent with a noisy neighbour.

`max_inputs=1` retires each container after one input, so the repeats land on
different machines rather than re-measuring the same one. Sizes bracket
200 MB, since the original oddity was size-specific.
"""

import json
import statistics
import subprocess

import modal

BASE = "fb39309d2d06c8bc8d24d0c9b81b4094766305d8"
HEAD = "fc2e7b1ad7b9b1a7f29b2f37e511cb79a1963ecf"
REPO = "https://github.com/athrva98/warpjq"
CUDA_ARCHS = "80,89,90"

app = modal.App("warpjq-h100-repeat")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.2-devel-ubuntu24.04", add_python="3.11"
    )
    .apt_install("git", "curl", "build-essential", "pkg-config")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
        "| sh -s -- -y --profile minimal --default-toolchain stable",
        f"git clone {REPO} /base && cd /base && git checkout {BASE}",
        f"git clone {REPO} /head && cd /head && git checkout {HEAD}",
        f"cd /base && . $HOME/.cargo/env && WARPJQ_CUDA_ARCH='{CUDA_ARCHS}' "
        "cargo build --release -p warpjq-cli --features cuda",
        f"cd /head && . $HOME/.cargo/env && WARPJQ_CUDA_ARCH='{CUDA_ARCHS}' "
        "cargo build --release -p warpjq-cli --features cuda",
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


def sh(cmd: str) -> str:
    p = subprocess.run(
        ["bash", "-lc", f". $HOME/.cargo/env && cd /head && {cmd}"],
        capture_output=True, text=True,
    )
    return (p.stdout or "") + (p.stderr or "")


def timed(binary: str, query: str, path: str, runs: int = 9):
    """Best of N, plus the median, so an outlier-driven result is visible."""
    times = []
    for _ in range(runs):
        out = sh(f"{binary} '{query}' --backend gpu --stats {path} > /dev/null")
        for line in out.splitlines():
            if " in " in line and "GB/s" in line:
                times.append(float(line.split(" in ")[1].split("s ")[0]))
    return {"best": min(times), "median": statistics.median(times)}


@app.function(image=image, gpu="H100", cpu=8.0, memory=65536,
              timeout=5400, max_inputs=1)
def one_container(trial: int) -> dict:
    host = subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip()
    out = {"trial": trial, "host": host, "sizes": {}}

    for size in ("100MB", "200MB", "400MB"):
        sh(f"{HEAD_BIN} gen --preset nginx --size {size} -o /tmp/r.ndjson --seed 1")
        sh("cat /tmp/r.ndjson > /dev/null")
        rows = {}
        # Interleave base and head so a drifting machine perturbs both equally
        # rather than whichever ran second.
        for q in QUERIES:
            b = timed(BASE_BIN, q, "/tmp/r.ndjson")
            h = timed(HEAD_BIN, q, "/tmp/r.ndjson")
            b2 = timed(BASE_BIN, q, "/tmp/r.ndjson")
            h2 = timed(HEAD_BIN, q, "/tmp/r.ndjson")
            rows[q] = {
                "base": min(b["best"], b2["best"]),
                "head": min(h["best"], h2["best"]),
                "base_median": statistics.median([b["median"], b2["median"]]),
                "head_median": statistics.median([h["median"], h2["median"]]),
            }
        out["sizes"][size] = rows
        sh("rm -f /tmp/r.ndjson")
        print(f"[trial {trial}] {size} done", flush=True)
    return out


@app.local_entrypoint()
def main(trials: int = 5):
    results = list(one_container.map(range(trials)))
    with open("modal_h100_repeat.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    hosts = {r["host"] for r in results}
    print(f"{len(results)} trials across {len(hosts)} distinct containers\n")

    for size in ("100MB", "200MB", "400MB"):
        print(f"=== {size} ===")
        for q in QUERIES:
            ratios = [
                r["sizes"][size][q]["base"] / r["sizes"][size][q]["head"]
                for r in results
            ]
            med_ratios = [
                r["sizes"][size][q]["base_median"] / r["sizes"][size][q]["head_median"]
                for r in results
            ]
            lo, hi = min(ratios), max(ratios)
            verdict = "flat"
            if hi < 0.95:
                verdict = "HEAD SLOWER on every trial"
            elif lo > 1.05:
                verdict = "head faster on every trial"
            elif lo < 0.95:
                verdict = "mixed, some trials slower"
            print(f"  {q:<46} best {statistics.median(ratios):.2f}x "
                  f"[{lo:.2f}-{hi:.2f}]  median {statistics.median(med_ratios):.2f}x"
                  f"  {verdict}")
        print()

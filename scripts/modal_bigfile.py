"""Does memory stay flat as the input grows?

The pinned pool is meant to be a fixed 128 MB (two 64 MB slots) regardless of
input size, with the file streaming through it. That is what the code says;
this checks it against a file several times larger than anything the pipeline
has actually run on, and watches peak RSS and locked memory rather than
trusting the reading.

A pool that scaled with the file, or a mapping being faulted in behind our
back, would show up as RSS tracking file size.

    modal run scripts/modal_bigfile.py
"""

import json
import subprocess

import modal

HEAD = "0c33562"
REPO = "https://github.com/athrva98/warpjq"
CUDA_ARCHS = "80,89,90"

app = modal.App("warpjq-bigfile")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.2-devel-ubuntu24.04", add_python="3.11"
    )
    .apt_install("git", "curl", "build-essential", "pkg-config", "time")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
        "| sh -s -- -y --profile minimal --default-toolchain stable",
        f"git clone {REPO} /head && cd /head && git checkout {HEAD}",
        f"cd /head && . $HOME/.cargo/env && WARPJQ_CUDA_ARCH='{CUDA_ARCHS}' "
        "cargo build --release -p warpjq-cli --features cuda",
    )
)

W = "/head/target/release/warpjq"


def sh(cmd: str, timeout: int = 7200) -> str:
    p = subprocess.run(
        ["bash", "-lc", f". $HOME/.cargo/env && {cmd}"],
        capture_output=True, text=True, timeout=timeout,
    )
    return (p.stdout or "") + (p.stderr or "")


def measured(query: str, path: str) -> dict:
    """Peak RSS via /usr/bin/time, which reports the high-water mark."""
    out = sh(f"/usr/bin/time -v {W} '{query}' --backend gpu --stats {path} > /dev/null")
    res = {"peak_rss_mb": None, "secs": None, "answer_ok": "Command exited" not in out}
    for line in out.splitlines():
        if "Maximum resident set size" in line:
            res["peak_rss_mb"] = round(int(line.split(":")[1].strip()) / 1024, 1)
        if " in " in line and "GB/s" in line:
            res["secs"] = float(line.split(" in ")[1].split("s ")[0])
    return res


@app.function(image=image, gpu="H100", cpu=8.0, memory=32768,
              timeout=10800, ephemeral_disk=1048576)
def run():
    # Deliberately less RAM (32 GB) than the largest file, so anything that
    # scales with input size cannot hide in spare memory.
    out = {"ram_mb": 32768, "sizes": {}}
    for size in ("2GB", "8GB", "32GB", "64GB"):
        sh(f"{W} gen --preset nginx --size {size} -o /tmp/big.ndjson --seed 1")
        actual = sh("stat -c %s /tmp/big.ndjson").strip()
        row = {
            "bytes": actual,
            "count": measured("select(.status == 500) | count", "/tmp/big.ndjson"),
            "group": measured("group_by(.host) | count", "/tmp/big.ndjson"),
        }
        # The answer has to stay right, not just the memory stay flat.
        row["answer"] = sh(
            f"{W} 'select(.status == 500) | count' --backend gpu /tmp/big.ndjson"
        ).strip()
        row["answer_cpu"] = sh(
            f"{W} 'select(.status == 500) | count' --backend cpu /tmp/big.ndjson"
        ).strip()
        out["sizes"][size] = row
        print(f"{size}: {row}", flush=True)
        sh("rm -f /tmp/big.ndjson")
    return out


@app.local_entrypoint()
def main():
    r = run.remote()
    with open("modal_bigfile.json", "w", encoding="utf-8") as fh:
        json.dump(r, fh, indent=2)

    print(f"\ncontainer RAM: {r['ram_mb'] / 1024:.0f} GB, H100\n")
    print(f"  {'size':>6} {'file bytes':>14} {'peak RSS':>10} {'secs':>8}  {'answers match':>14}")
    for size, row in r["sizes"].items():
        c = row["count"]
        match = "yes" if row["answer"] == row["answer_cpu"] else "NO -- DIFFER"
        rss = c["peak_rss_mb"]
        print(f"  {size:>6} {row['bytes']:>14} {rss:>9} MB {c['secs'] or 0:>7.2f}s  {match:>14}")
    print("\n  A flat peak RSS across sizes means the pool is fixed and the")
    print("  file is streaming. RSS tracking file size would mean it is not.")

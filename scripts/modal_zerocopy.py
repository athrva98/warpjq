"""Does removing the staging copy actually move the wall clock?

The copy was 73% of an 8 GB run in an earlier profile, which is the whole
reason for the zero-copy path. But that profile was taken in a 64 GB
container under page-cache pressure, and a later 128 GB run did the same
8 GB in 0.884 s total, so the copy cannot have been 2 s there. The share is
a function of memory pressure, not a constant, and the only way to know what
pinning buys is to measure both paths on the same machine and file.

One binary runs both: WARPJQ_NO_PIN=1 forces the staging path. So this is an
A/B of two code paths on identical hardware, data and page-cache state,
rather than two builds compared across runs.

DuckDB is measured alongside as the bar that matters.

    modal run scripts/modal_zerocopy.py
"""

import json
import subprocess

import modal

COMMIT = "126d896078e4d17350645a247b8efe19de7dd7ac"
REPO = "https://github.com/athrva98/warpjq"
CUDA_ARCHS = "80,89,90"

app = modal.App("warpjq-zerocopy")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.2-devel-ubuntu24.04", add_python="3.11"
    )
    .apt_install("git", "curl", "build-essential", "pkg-config")
    .pip_install("duckdb")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
        "| sh -s -- -y --profile minimal --default-toolchain stable",
        f"git clone {REPO} /warpjq && cd /warpjq && git checkout {COMMIT}",
        f"cd /warpjq && . $HOME/.cargo/env && WARPJQ_CUDA_ARCH='{CUDA_ARCHS}' "
        "cargo build --release -p warpjq-cli --features cuda",
    )
)

W = "/warpjq/target/release/warpjq"

CASES = [
    ("count matching", "select(.status == 500) | count",
     "select count(*) from read_ndjson_auto('{f}') where status = 500"),
    ("projection", "select(.status >= 500) | {p: .path, b: .bytes}",
     "select path, bytes from read_ndjson_auto('{f}') where status >= 500"),
    ("group by", "group_by(.host) | count",
     "select host, count(*) from read_ndjson_auto('{f}') group by host order by host"),
    ("passthrough", "select(.status == 500)",
     "select * from read_ndjson_auto('{f}') where status = 500"),
]


def sh(cmd: str, env: str = "", timeout: int = 3600) -> str:
    p = subprocess.run(
        ["bash", "-lc", f". $HOME/.cargo/env && {env} {cmd}"],
        capture_output=True, text=True, timeout=timeout,
    )
    return (p.stdout or "") + (p.stderr or "")


def timed(query: str, path: str, env: str = "", runs: int = 7):
    times = []
    for _ in range(runs):
        out = sh(f"{W} '{query}' --backend gpu --stats {path} > /dev/null", env)
        for line in out.splitlines():
            if " in " in line and "GB/s" in line:
                times.append(float(line.split(" in ")[1].split("s ")[0]))
    return min(times) if times else None


def profile(query: str, path: str, env: str = "") -> dict:
    out = sh(f"WARPJQ_PROFILE=1 {W} '{query}' --backend gpu {path} > /dev/null", env)
    stages = {}
    for line in out.splitlines():
        if "warpjq profile:" not in line or "through the pipeline" in line:
            continue
        body = line.split("warpjq profile:", 1)[1].strip()
        parts = body.rsplit(None, 3)
        if len(parts) >= 2:
            stages[parts[0].strip()] = parts[1]
    return stages


def duckdb_time(sql: str, path: str, runs: int = 5):
    with open("/tmp/dq.py", "w") as fh:
        fh.write(
            "import duckdb, time, sys\n"
            "t = time.perf_counter()\n"
            "r = duckdb.sql(sys.argv[1]).fetchall()\n"
            "print(time.perf_counter() - t)\n"
        )
    times = []
    for _ in range(runs):
        out = sh(f"python3 /tmp/dq.py {json.dumps(sql.format(f=path))}")
        try:
            times.append(float(out.splitlines()[0]))
        except (ValueError, IndexError):
            return None
    return min(times)


@app.function(image=image, gpu="H100", cpu=8.0, memory=131072, timeout=7200)
def run():
    out = {"sizes": {}}
    for size in ("1GB", "8GB"):
        sh(f"{W} gen --preset nginx --size {size} -o /tmp/z.ndjson --seed 1")
        sh("cat /tmp/z.ndjson > /dev/null")

        rows = {}
        for label, wq, sql in CASES:
            # Both paths must produce identical bytes before either is timed.
            sh(f"{W} '{wq}' --backend gpu /tmp/z.ndjson > /tmp/pin.out")
            sh(f"{W} '{wq}' --backend gpu /tmp/z.ndjson > /tmp/stg.out",
               env="WARPJQ_NO_PIN=1")
            same = "differ" not in sh("cmp /tmp/pin.out /tmp/stg.out || echo differ")
            rows[label] = {
                "pinned": timed(wq, "/tmp/z.ndjson"),
                "staging": timed(wq, "/tmp/z.ndjson", env="WARPJQ_NO_PIN=1"),
                "duckdb": duckdb_time(sql, "/tmp/z.ndjson"),
                "identical": same,
            }
            print(f"[{size}] {label}: {rows[label]}", flush=True)

        out["sizes"][size] = {
            "timings": rows,
            "profile_pinned": profile(CASES[0][1], "/tmp/z.ndjson"),
            "profile_staging": profile(CASES[0][1], "/tmp/z.ndjson", "WARPJQ_NO_PIN=1"),
        }
        sh("rm -f /tmp/z.ndjson")
    return out


@app.local_entrypoint()
def main():
    r = run.remote()
    with open("modal_zerocopy.json", "w", encoding="utf-8") as fh:
        json.dump(r, fh, indent=2)

    for size, s in r["sizes"].items():
        print(f"=== {size} on H100 ===")
        print(f"  {'query':<16} {'staging':>9} {'pinned':>9} {'speedup':>9} "
              f"{'duckdb':>9} {'pinned vs duckdb':>18}")
        for label, t in s["timings"].items():
            p, g, d = t["pinned"], t["staging"], t["duckdb"]
            flag = "" if t["identical"] else "  OUTPUT DIFFERS"
            vs = f"{d / p:.2f}x" if d and p else "n/a"
            print(f"  {label:<16} {g:>9.3f} {p:>9.3f} {g / p:>8.2f}x "
                  f"{(d or 0):>9.3f} {vs:>18}{flag}")
        print("  stages, staging -> pinned:")
        keys = list(s["profile_staging"]) or list(s["profile_pinned"])
        for k in keys:
            print(f"    {k:<24} {s['profile_staging'].get(k, '?'):>8} -> "
                  f"{s['profile_pinned'].get(k, '?'):>8}")
        print()

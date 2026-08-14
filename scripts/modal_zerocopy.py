"""A/B two commits of the GPU input path, with DuckDB as the reference.

Both are built into one image and run against the same file on the same
machine with the same page-cache state. That matters more than it sounds:
the same binary measured 0.995 s and 0.737 s for an 8 GB count in two
different containers. Only the within-run ratio means anything, so never
compare a number here against one from another run.

Set BASE and HEAD to the commits under test. Output has to be byte identical
between them before either is timed.

    modal run scripts/modal_zerocopy.py
"""

import json
import subprocess

import modal

BASE = "0aa36df"   # three slots always
HEAD = "6dc2dc9"   # slot count sized from input length
REPO = "https://github.com/athrva98/warpjq"
CUDA_ARCHS = "80,89,90"

app = modal.App("warpjq-inputpath")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.2-devel-ubuntu24.04", add_python="3.11"
    )
    .apt_install("git", "curl", "build-essential", "pkg-config")
    .pip_install("duckdb")
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


def timed(W: str, query: str, path: str, env: str = "", runs: int = 7):
    times = []
    for _ in range(runs):
        out = sh(f"{W} '{query}' --backend gpu --stats {path} > /dev/null", env)
        for line in out.splitlines():
            if " in " in line and "GB/s" in line:
                times.append(float(line.split(" in ")[1].split("s ")[0]))
    return min(times) if times else None


def profile(W: str, query: str, path: str, env: str = "") -> dict:
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
    out = {"base": BASE, "head": HEAD, "sizes": {}}
    for size in ("1GB", "8GB"):
        sh(f"{HEAD_BIN} gen --preset nginx --size {size} -o /tmp/z.ndjson --seed 1")
        sh("cat /tmp/z.ndjson > /dev/null")

        rows = {}
        for label, wq, sql in CASES:
            # Both paths must produce identical bytes before either is timed.
            sh(f"{HEAD_BIN} '{wq}' --backend gpu /tmp/z.ndjson > /tmp/pin.out")
            sh(f"{BASE_BIN} '{wq}' --backend gpu /tmp/z.ndjson > /tmp/stg.out")
            same = "differ" not in sh("cmp /tmp/pin.out /tmp/stg.out || echo differ")
            rows[label] = {
                "head": timed(HEAD_BIN, wq, "/tmp/z.ndjson"),
                "base": timed(BASE_BIN, wq, "/tmp/z.ndjson"),
                "duckdb": duckdb_time(sql, "/tmp/z.ndjson"),
                "identical": same,
            }
            print(f"[{size}] {label}: {rows[label]}", flush=True)

        out["sizes"][size] = {
            "timings": rows,
            "profile_head": profile(HEAD_BIN, CASES[0][1], "/tmp/z.ndjson"),
            "profile_base": profile(BASE_BIN, CASES[0][1], "/tmp/z.ndjson"),
        }
        sh("rm -f /tmp/z.ndjson")
    return out


@app.local_entrypoint()
def main():
    r = run.remote()
    with open("modal_zerocopy.json", "w", encoding="utf-8") as fh:
        json.dump(r, fh, indent=2)

    print(f"base {r['base']} -> head {r['head']}, H100\n")
    for size, s in r["sizes"].items():
        print(f"=== {size} ===")
        print(f"  {'query':<16} {'base':>9} {'head':>9} {'speedup':>9} "
              f"{'duckdb':>9} {'head vs duckdb':>16}")
        for label, t in s["timings"].items():
            b, h, d = t["base"], t["head"], t["duckdb"]
            flag = "" if t["identical"] else "  OUTPUT DIFFERS"
            vs = f"{d / h:.2f}x" if d and h else "n/a"
            print(f"  {label:<16} {b:>9.3f} {h:>9.3f} {b / h:>8.2f}x "
                  f"{(d or 0):>9.3f} {vs:>16}{flag}")
        print("  stages, base -> head:")
        for k in (list(s["profile_base"]) or list(s["profile_head"])):
            print(f"    {k:<24} {s['profile_base'].get(k, '?'):>8} -> "
                  f"{s['profile_head'].get(k, '?'):>8}")
        print()

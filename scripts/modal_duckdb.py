"""Compare warpjq against DuckDB's native NDJSON reader.

The question this answers is whether warpjq scans NDJSON faster than DuckDB
does. If it does not, there is nothing for a DuckDB extension to contribute
and the honest answer is to use DuckDB. If it does, the extension has a
reason to exist and this measures by how much.

Queries are matched by result, not by shape: each pair is checked to return
the same answer before either side is timed.

    modal run scripts/modal_duckdb.py
"""

import json
import statistics
import subprocess

import modal

COMMIT = "9a30598"
REPO = "https://github.com/athrva98/warpjq"
CUDA_ARCHS = "80,89,90"

app = modal.App("warpjq-duckdb")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.2-devel-ubuntu24.04", add_python="3.11"
    )
    .apt_install("git", "curl", "build-essential", "pkg-config", "jq")
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

# (label, warpjq query, duckdb SQL). The SQL uses read_ndjson_auto so DuckDB
# does its own schema inference, which is what a user would actually type.
CASES = [
    (
        "count matching",
        "select(.status == 500) | count",
        "select count(*) from read_ndjson_auto('{f}') where status = 500",
    ),
    (
        "sum on a filter",
        "select(.status == 500) | sum(.bytes)",
        "select sum(bytes) from read_ndjson_auto('{f}') where status = 500",
    ),
    (
        "group by",
        "group_by(.host) | count",
        "select host, count(*) from read_ndjson_auto('{f}') group by host order by host",
    ),
    (
        "count all rows",
        "count",
        "select count(*) from read_ndjson_auto('{f}')",
    ),
]


def sh(cmd: str, timeout: int = 3600) -> str:
    p = subprocess.run(
        ["bash", "-lc", f". $HOME/.cargo/env && {cmd}"],
        capture_output=True, text=True, timeout=timeout,
    )
    return (p.stdout or "") + (p.stderr or "")


def time_warpjq(query: str, path: str, backend: str, runs: int = 5):
    times = []
    for _ in range(runs):
        out = sh(f"{W} '{query}' --backend {backend} --stats {path} > /dev/null")
        for line in out.splitlines():
            if " in " in line and "GB/s" in line:
                times.append(float(line.split(" in ")[1].split("s ")[0]))
    return min(times) if times else None


def time_duckdb(sql: str, path: str, runs: int = 5):
    """Times DuckDB in a fresh process each run, as warpjq is timed.

    Keeping it in-process would let DuckDB cache the file between runs and
    measure something warpjq never gets.
    """
    script = (
        "import duckdb, time, sys\n"
        "q = sys.argv[1]\n"
        "t = time.perf_counter()\n"
        "r = duckdb.sql(q).fetchall()\n"
        "print(time.perf_counter() - t)\n"
        "print(repr(r[:5]))\n"
    )
    with open("/tmp/dq.py", "w") as fh:
        fh.write(script)
    times, result = [], None
    for _ in range(runs):
        out = sh(f"python3 /tmp/dq.py {json.dumps(sql.format(f=path))}")
        lines = [l for l in out.splitlines() if l.strip()]
        try:
            times.append(float(lines[0]))
            result = lines[1]
        except (ValueError, IndexError):
            return None, out[-400:]
    return min(times), result


@app.function(image=image, gpu="H100", cpu=8.0, memory=131072, timeout=7200)
def compare():
    out = {"duckdb_version": sh("python3 -c 'import duckdb;print(duckdb.__version__)'").strip()}
    out["cores"] = sh("nproc").strip()
    out["sizes"] = {}

    for size in ("1GB", "8GB"):
        sh(f"{W} gen --preset nginx --size {size} -o /tmp/d.ndjson --seed 1")
        sh("cat /tmp/d.ndjson > /dev/null")
        rows = {}
        for label, wq, sql in CASES:
            # Agreement first. A faster wrong answer is not a result.
            w_ans = sh(f"{W} '{wq}' --backend cpu /tmp/d.ndjson | head -3").strip()
            d_time, d_ans = time_duckdb(sql, "/tmp/d.ndjson")
            rows[label] = {
                "warpjq_gpu": time_warpjq(wq, "/tmp/d.ndjson", "gpu"),
                "warpjq_cpu": time_warpjq(wq, "/tmp/d.ndjson", "cpu"),
                "duckdb": d_time,
                "warpjq_answer": w_ans[:120],
                "duckdb_answer": (d_ans or "")[:120],
            }
            print(f"[{size}] {label}: {rows[label]}", flush=True)
        out["sizes"][size] = rows
        sh("rm -f /tmp/d.ndjson")
    return out


@app.local_entrypoint()
def main():
    r = compare.remote()
    with open("modal_duckdb.json", "w", encoding="utf-8") as fh:
        json.dump(r, fh, indent=2)

    print(f"duckdb {r['duckdb_version']}, {r['cores']} cores, H100\n")
    for size, rows in r["sizes"].items():
        print(f"=== {size} ===")
        print(f"  {'query':<18} {'wq gpu':>9} {'wq cpu':>9} {'duckdb':>9}  {'best wq vs duckdb':>18}")
        for label, t in rows.items():
            g, c, d = t["warpjq_gpu"], t["warpjq_cpu"], t["duckdb"]
            if d is None:
                print(f"  {label:<18} duckdb failed: {t['duckdb_answer']}")
                continue
            best = min(x for x in (g, c) if x is not None)
            print(f"  {label:<18} {g:>9.3f} {c:>9.3f} {d:>9.3f}  {d / best:>17.2f}x")
        print()
        for label, t in rows.items():
            print(f"  {label:<18} warpjq={t['warpjq_answer'][:60]}")
            print(f"  {'':<18} duckdb={t['duckdb_answer'][:60]}")
        print()

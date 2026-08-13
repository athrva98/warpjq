# Correctness + performance gate for CUDA kernel work.
#
#   powershell -File scripts/check_kernels.ps1
#
# Runs three things:
#   1. The full test suite with --test-threads=1.
#   2. A GPU-vs-CPU sweep over every dumped corpus x query pair, comparing
#      stdout byte for byte.
#   3. nsys per-kernel timings on a fixed workload.
#
# Why --test-threads=1: the differential suite is flaky under its default
# parallelism *on main*, at roughly 40% of runs on this machine (6/15 with an
# unmodified tree). Several tests run GPU work concurrently in one process and
# something in that path is order-dependent. Single-threaded it is reliable,
# so that is the gate until the underlying race is found -- see the note in
# docs/BENCHMARKS.md. Do not read a green parallel run as a pass, and do not
# read a red one as a regression, without checking main the same way.

$ErrorActionPreference = "Continue"
$env:WARPJQ_CUDA_ARCH = "120"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Output "== build =="
cargo build --release --features cuda 2>&1 | Select-String "error|warning: unused" | Select-Object -First 20

Write-Output "`n== tests (single-threaded) =="
$raw = cargo test --release --features cuda -- --test-threads=1 2>&1 | Out-String
($raw -split "`n" | Select-String "^test result") -join "`n"
if ($raw -cmatch "FAILED") { Write-Output "TESTS FAILED"; ($raw -split "`n" | Select-String "panicked at|GPU and CPU disagree" | Select-Object -First 6) }

Write-Output "`n== GPU vs CPU sweep =="
$w = Join-Path $root "target\release\warpjq.exe"
$corpora = Get-ChildItem (Join-Path $root "crates\warpjq-core\corpus_*.ndjson") -ErrorAction SilentlyContinue
if (-not $corpora) { Write-Output "  (no corpora; run the tmp_dump_corpora test)" }
$queries = @('.', '.status', '.host', '.msg', '.nested.deep.v', '.arr[1]', '.arr[9]', '.missing',
             'select(.status == 500)', 'select(.status >= 404)', 'select(.flag)', 'select(.flag | not)',
             'select(.flag == null)', 'select(.nested.deep.v > 250)',
             '{h: .host, s: .status}', '{v: .nested.deep.v, a: .arr[0]}',
             'count', 'select(.status == 500) | count', 'sum(.bytes)', 'min(.bytes)', 'max(.bytes)',
             'group_by(.host) | count', 'group_by(.host) | sum(.bytes)')
$bad = 0; $n = 0
foreach ($c in $corpora) {
  foreach ($q in $queries) {
    # Small chunks exercise multi-chunk merging, which is where ordering bugs live.
    $g = & $w --backend gpu --chunk-size 65536 -j 4 $q $c.FullName 2>$null | Out-String
    $cpu = & $w --backend cpu --chunk-size 65536 -j 4 $q $c.FullName 2>$null | Out-String
    $n++
    if ($g -ne $cpu) { $bad++; Write-Output ("  MISMATCH {0} :: {1}" -f $c.Name, $q) }
  }
}
Write-Output ("  {0}/{1} query x corpus pairs agree" -f ($n - $bad), $n)

Write-Output "`n== kernel timings =="
$bench = Join-Path $root "bench200.ndjson"
if (Test-Path $bench) {
  $nsys = "C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.1.3\target-windows-x64\nsys.exe"
  & $nsys profile --force-overwrite=true -o nsysrep --trace=cuda $w --backend gpu 'select(.status == 500) | count' $bench 2>&1 | Out-Null
  & $nsys stats --force-export=true --report cuda_gpu_kern_sum nsysrep.nsys-rep 2>$null |
    Select-String -Pattern "k_eval|k_build_lines|k_agg|DeviceSelectSweep|Time \(%\)"
} else {
  Write-Output "  (bench200.ndjson missing; run: warpjq gen --preset nginx --size 200MB -o bench200.ndjson --seed 1)"
}

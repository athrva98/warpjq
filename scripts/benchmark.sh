#!/usr/bin/env bash
# Reproduces the benchmark matrix in the README.
#
#   ./scripts/benchmark.sh [sizes...]      default: 1GB 10GB
#
# Generates its own data, so there is nothing to download and everyone runs the
# same bytes. Prints the hardware and cache state alongside the numbers,
# because a benchmark that does not say what it ran on is not a result.
set -euo pipefail

WARPJQ="${WARPJQ:-./target/release/warpjq}"
SIZES=("${@:-1GB 10GB}")
# Split a single "1GB 10GB" argument into words.
read -r -a SIZES <<< "${SIZES[*]}"
DATADIR="${WARPJQ_BENCH_DIR:-./bench-data}"
SEED="${WARPJQ_BENCH_SEED:-1}"

command -v "$WARPJQ" >/dev/null 2>&1 || [ -x "$WARPJQ" ] || {
  echo "warpjq not found at '$WARPJQ'. Build it first:" >&2
  echo "  cargo build --release --features cuda" >&2
  exit 1
}

echo "=================================================================="
echo " warpjq benchmark"
echo "=================================================================="
echo "binary:   $("$WARPJQ" --version)"
echo "date:     $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "seed:     $SEED"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "gpu:      $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1)"
else
  echo "gpu:      none detected"
fi
if [ -r /proc/cpuinfo ]; then
  echo "cpu:      $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')"
elif command -v sysctl >/dev/null 2>&1; then
  echo "cpu:      $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
fi
for tool in jq jaq; do
  if command -v "$tool" >/dev/null 2>&1; then
    echo "$tool:       $("$tool" --version 2>&1 | head -1)"
  else
    echo "$tool:       not installed (its row will be skipped)"
  fi
done
echo

mkdir -p "$DATADIR"

# One query per interesting shape: a pure filter, a filter + reduction, a
# grouped reduction, and a projection that has to assemble output.
QUERIES=(
  'select(.status == 500) | count'
  'select(.status == 500) | sum(.bytes)'
  'group_by(.host) | count'
  'select(.status >= 500) | {t: .ts, p: .path, b: .bytes}'
)

for size in "${SIZES[@]}"; do
  file="$DATADIR/nginx-$size.ndjson"
  if [ ! -f "$file" ]; then
    echo "--- generating $size of nginx logs (seed $SEED) ---"
    "$WARPJQ" gen --preset nginx --size "$size" -o "$file" --seed "$SEED"
  fi

  echo
  echo "##################################################################"
  echo "# $size  ($(du -h "$file" | cut -f1) on disk)"
  echo "##################################################################"

  # Warm the page cache once for the whole size, then let `bench` do its own
  # warmup per engine. Cold-cache numbers need `echo 3 > /proc/sys/vm/drop_caches`
  # between runs and should be reported separately.
  cat "$file" > /dev/null

  for q in "${QUERIES[@]}"; do
    echo
    echo "### $q"
    "$WARPJQ" bench "$q" "$file" --runs 3 --warmup 1
  done
done

echo
echo "=================================================================="
echo " Notes"
echo "=================================================================="
cat <<'EOF'
* Every timing is end-to-end wall clock for the whole process, including
  reading the file. There is no kernel-only number anywhere in this output.
* Page cache is warm. For cold numbers, drop caches between runs:
    sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
* Rows whose output disagrees with warpjq's are reported as NOT COMPARABLE
  rather than timed.
* WARPJQ_PROFILE=1 breaks a single GPU run into read / copy / kernel / drain
  stages, which is the fastest way to see whether you are compute- or
  input-bound on your hardware.
EOF

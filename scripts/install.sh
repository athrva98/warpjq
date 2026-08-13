#!/usr/bin/env sh
# warpjq installer.
#
#   curl -fsSL https://raw.githubusercontent.com/athrva98/warpjq/main/scripts/install.sh | sh
#
# Installs a prebuilt binary when one matches this machine, and otherwise
# builds from source with cargo. Detects an NVIDIA GPU and picks the CUDA build
# only when one is present *and* the CUDA toolkit is available, because
# useful without either, so a missing GPU is never an error.
#
# Environment:
#   WARPJQ_VERSION   version to install (default: latest release)
#   WARPJQ_PREFIX    install directory (default: ~/.local/bin)
#   WARPJQ_CUDA      force "1" or "0" instead of auto-detecting
set -eu

REPO="athrva98/warpjq"
PREFIX="${WARPJQ_PREFIX:-$HOME/.local/bin}"
VERSION="${WARPJQ_VERSION:-latest}"

say()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

need() { command -v "$1" >/dev/null 2>&1; }

detect_target() {
  os="$(uname -s)"
  arch="$(uname -m)"
  case "$os" in
    Linux)  os_part="unknown-linux-gnu" ;;
    Darwin) os_part="apple-darwin" ;;
    MINGW*|MSYS*|CYGWIN*) os_part="pc-windows-msvc" ;;
    *) die "unsupported operating system: $os" ;;
  esac
  case "$arch" in
    x86_64|amd64) arch_part="x86_64" ;;
    aarch64|arm64) arch_part="aarch64" ;;
    *) die "unsupported architecture: $arch" ;;
  esac
  printf '%s-%s' "$arch_part" "$os_part"
}

want_cuda() {
  if [ "${WARPJQ_CUDA:-}" = "1" ]; then return 0; fi
  if [ "${WARPJQ_CUDA:-}" = "0" ]; then return 1; fi
  # Both a device and a toolkit are needed. Either alone means the CPU build.
  need nvidia-smi && nvidia-smi >/dev/null 2>&1 && need nvcc
}

download() {
  url="$1"; out="$2"
  if need curl; then curl -fsSL "$url" -o "$out"
  elif need wget; then wget -qO "$out" "$url"
  else die "need curl or wget"; fi
}

install_from_source() {
  need cargo || die "no prebuilt binary for this platform and cargo is not installed.
  Install Rust from https://rustup.rs and re-run, or build manually:
    git clone https://github.com/$REPO && cd warpjq && cargo build --release"
  if want_cuda; then
    say "building from source with CUDA support"
    cargo install --git "https://github.com/$REPO" warpjq-cli --features cuda --root "$(dirname "$PREFIX")" --force
  else
    say "building from source (CPU only)"
    cargo install --git "https://github.com/$REPO" warpjq-cli --root "$(dirname "$PREFIX")" --force
  fi
}

main() {
  target="$(detect_target)"
  if want_cuda; then
    flavour="cuda"
    say "detected an NVIDIA GPU and the CUDA toolkit; installing the CUDA build"
  else
    flavour="cpu"
    if need nvidia-smi && ! need nvcc; then
      warn "found an NVIDIA GPU but no CUDA toolkit; installing the CPU build.
         Install the CUDA toolkit and re-run to get the GPU engine."
    fi
    say "installing the CPU build ($target)"
  fi

  if [ "$VERSION" = "latest" ]; then
    url="https://github.com/$REPO/releases/latest/download/warpjq-$target-$flavour.tar.gz"
  else
    url="https://github.com/$REPO/releases/download/$VERSION/warpjq-$target-$flavour.tar.gz"
  fi

  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT

  if download "$url" "$tmp/warpjq.tar.gz" 2>/dev/null; then
    tar -xzf "$tmp/warpjq.tar.gz" -C "$tmp"
    mkdir -p "$PREFIX"
    install -m 0755 "$tmp/warpjq" "$PREFIX/warpjq" 2>/dev/null \
      || { cp "$tmp/warpjq" "$PREFIX/warpjq" && chmod 0755 "$PREFIX/warpjq"; }
    say "installed to $PREFIX/warpjq"
  else
    warn "no prebuilt binary at $url"
    install_from_source
  fi

  case ":$PATH:" in
    *":$PREFIX:"*) ;;
    *) warn "$PREFIX is not on your PATH. Add it with:
         echo 'export PATH=\"$PREFIX:\$PATH\"' >> ~/.profile" ;;
  esac

  if "$PREFIX/warpjq" --version >/dev/null 2>&1; then
    say "$("$PREFIX/warpjq" --version) is ready"
    printf '\nTry it:\n  warpjq gen --preset nginx --size 200MB -o access.ndjson\n'
    printf "  warpjq 'select(.status == 500) | count' access.ndjson\n"
    printf "  warpjq bench 'select(.status == 500) | count' access.ndjson\n\n"
  else
    die "the installed binary did not run; please open an issue at https://github.com/$REPO/issues"
  fi
}

main "$@"

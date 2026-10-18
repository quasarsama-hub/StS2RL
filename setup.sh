#!/usr/bin/env bash
# One-shot setup for Linux / macOS / WSL2.
#
#   bash setup.sh            # CPU torch
#   TORCH_CUDA=cu128 bash setup.sh   # CUDA torch wheel (pick the tag matching your driver)
#
# What it does, every step idempotent:
#   1. fetch the simulator submodule (evangambit/sts2, pinned)
#   2. apply our patches to it (patches/sts2/*.patch)
#   3. install the .NET 9 SDK into ~/.dotnet if `dotnet` is missing
#   4. build the NativeAOT simulator library
#   5. create .venv and install Python dependencies
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
SIM="$ROOT/third_party/sts2"
cd "$ROOT"

echo "==> submodule"
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --depth 1 third_party/sts2

echo "==> patches"
for patch in "$ROOT"/patches/sts2/*.patch; do
  if git -C "$SIM" apply --reverse --check "$patch" >/dev/null 2>&1; then
    echo "    already applied: $(basename "$patch")"
  else
    git -C "$SIM" apply "$patch"
    echo "    applied: $(basename "$patch")"
  fi
done

echo "==> .NET 9 SDK"
if ! command -v dotnet >/dev/null 2>&1 || ! dotnet --list-sdks 2>/dev/null | grep -q '^9\.'; then
  if [ ! -x "$HOME/.dotnet/dotnet" ] || ! "$HOME/.dotnet/dotnet" --list-sdks | grep -q '^9\.'; then
    curl -sSL -o /tmp/dotnet-install.sh https://dot.net/v1/dotnet-install.sh
    bash /tmp/dotnet-install.sh --channel 9.0 --install-dir "$HOME/.dotnet"
  fi
  export DOTNET_ROOT="$HOME/.dotnet"
  export PATH="$HOME/.dotnet:$PATH"
fi
dotnet --version

echo "==> native simulator build"
# NativeAOT needs a C toolchain: clang + zlib on Linux, Xcode CLT on macOS.
bash "$SIM/scripts/build.sh"

echo "==> python environment"
PY="${PYTHON:-python3}"
if command -v uv >/dev/null 2>&1; then
  [ -d .venv ] || uv venv --python 3.12 .venv
  INSTALL="uv pip install --python .venv/bin/python"
else
  [ -d .venv ] || "$PY" -m venv .venv
  .venv/bin/python -m pip install --upgrade pip >/dev/null
  INSTALL=".venv/bin/python -m pip install"
fi
TORCH_INDEX="https://download.pytorch.org/whl/${TORCH_CUDA:-cpu}"
$INSTALL --index-url "$TORCH_INDEX" "torch>=2.4"
$INSTALL -e "$ROOT"

echo
echo "Done. Activate with:  source .venv/bin/activate"
echo "Smoke test:           python -m sts2rl.evaluate --agent heuristic --runs 4"

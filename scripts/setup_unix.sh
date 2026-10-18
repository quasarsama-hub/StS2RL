#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
python3 -m venv .venv
.venv/bin/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e '.[dev]'
mkdir -p .external
if [[ ! -d .external/sts2-rl-agent ]]; then
  git clone --no-checkout https://github.com/vvsuby/sts2-rl-agent.git .external/sts2-rl-agent
  git -C .external/sts2-rl-agent checkout --detach 04a62c35b0ee264f5e4a3b5c6587f7438f1a7e0c
fi
.venv/bin/python -c 'from spire2_rl.experiment import provenance; from pathlib import Path; print(provenance(Path(".external/sts2-rl-agent").resolve()))'
echo 'Ready: .venv/bin/spire2-rl train --experiment experiments/ironclad-a0 --steps 20000'

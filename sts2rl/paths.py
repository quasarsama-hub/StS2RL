"""Where things live, and the one sys.path edit that makes the simulator importable."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIM_ROOT = Path(os.environ.get("STS2_SIM_ROOT", ROOT / "third_party" / "sts2"))
GENERATED = SIM_ROOT / "src" / "Sts2Emulator" / "Generated"
RUNS_DIR = Path(os.environ.get("STS2RL_RUNS", ROOT / "runs"))

_sim_src = str(SIM_ROOT / "src")
if _sim_src not in sys.path:
    sys.path.insert(0, _sim_src)

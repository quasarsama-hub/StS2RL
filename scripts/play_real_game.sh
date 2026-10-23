#!/usr/bin/env bash
# Let the agent play your copy of Slay the Spire 2 -- the one-command version.
#
#   bash scripts/play_real_game.sh                                   # find the game
#   bash scripts/play_real_game.sh "/path/to/Slay the Spire 2"
#
# What it does, each step only if it is not done yet:
#   1. sets this repo up (setup.sh: simulator, Python environment)
#   2. installs the STS2MCP mod into the game's mods/ folder (two files from its latest
#      release; github.com/Gennadiyev/STS2MCP) -- the mod lets a program read and play the
#      game over http://localhost:15526
#   3. on Windows (WSL), turns on mirrored networking so WSL can reach that port
#   4. waits for the game, then for you to start an Ironclad Ascension 8 run
#   5. hands the run to the agent, which plays until the run ends or the simulator
#      disagrees with the game; the log it writes is what to send back for fixing
#
# Before the first run: back up your save, and consider turning Steam Cloud off while a
# bot plays. Ascension 8 has to be unlocked in your profile -- it is the only one the
# simulator models.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
URL="${STS2MCP_URL:-http://localhost:15526}"
MOD_RELEASE="https://github.com/Gennadiyev/STS2MCP/releases/latest/download"
PY="$ROOT/.venv/bin/python"

say() { printf '\n==> %s\n' "$*"; }

say "1/5  this repo"
if [ -x "$PY" ] && "$PY" -c "import sts2rl.sim" 2>/dev/null; then
  echo "    already set up"
else
  # What a fresh Ubuntu (the WSL default) lacks: the simulator's NativeAOT build needs
  # clang and zlib, and Python needs its venv module.
  if command -v apt-get >/dev/null 2>&1; then
    need=()
    command -v git >/dev/null 2>&1 || need+=(git)
    command -v curl >/dev/null 2>&1 || need+=(curl)
    command -v clang >/dev/null 2>&1 || need+=(clang)
    dpkg -s zlib1g-dev >/dev/null 2>&1 || need+=(zlib1g-dev)
    python3 -c "import ensurepip" >/dev/null 2>&1 || need+=(python3-venv)
    if [ "${#need[@]}" -gt 0 ]; then
      echo "    installing ${need[*]} (sudo may ask for your password)"
      sudo apt-get update -qq
      sudo apt-get install -y "${need[@]}"
    fi
  fi
  bash setup.sh
fi

say "2/5  the STS2MCP mod"
in_wsl=0
grep -qi microsoft /proc/version 2>/dev/null && in_wsl=1
GAME_DIR="${1:-${STS2_GAME_DIR:-}}"
if [ -z "$GAME_DIR" ]; then
  for candidate in \
    "/mnt/c/Program Files (x86)/Steam/steamapps/common/Slay the Spire 2" \
    "/mnt/d/SteamLibrary/steamapps/common/Slay the Spire 2" \
    "/mnt/e/SteamLibrary/steamapps/common/Slay the Spire 2" \
    "$HOME/.steam/steam/steamapps/common/Slay the Spire 2" \
    "$HOME/.local/share/Steam/steamapps/common/Slay the Spire 2" \
    "$HOME/Library/Application Support/Steam/steamapps/common/Slay the Spire 2"; do
    if [ -d "$candidate" ]; then
      GAME_DIR="$candidate"
      break
    fi
  done
fi
if [ -z "$GAME_DIR" ] || [ ! -d "$GAME_DIR" ]; then
  echo "Could not find Slay the Spire 2. Pass its folder, in quotes:" >&2
  echo "  bash scripts/play_real_game.sh \"/mnt/c/Program Files (x86)/Steam/steamapps/common/Slay the Spire 2\"" >&2
  echo "(Steam: right-click the game > Manage > Browse local files shows it.)" >&2
  exit 1
fi
MODS_DIR="$GAME_DIR/mods"
[ -d "$GAME_DIR/SlayTheSpire2.app" ] && MODS_DIR="$GAME_DIR/SlayTheSpire2.app/Contents/MacOS/mods"
if [ -f "$MODS_DIR/STS2_MCP.dll" ] && [ -f "$MODS_DIR/STS2_MCP.json" ]; then
  echo "    already installed in $MODS_DIR"
else
  mkdir -p "$MODS_DIR"
  for file in STS2_MCP.dll STS2_MCP.json; do
    curl -fsSL -o "$MODS_DIR/$file" "$MOD_RELEASE/$file"
  done
  echo "    installed into $MODS_DIR"
  echo "    In the game: Settings > Mods, enable mods (accept the consent dialog once)."
fi

say "3/5  the connection to the game"
if [ "$in_wsl" = 1 ]; then
  win_home="$(cmd.exe /c 'echo %USERPROFILE%' 2>/dev/null | tr -d '\r' || true)"
  if [ -n "$win_home" ]; then
    cfg="$(wslpath "$win_home")/.wslconfig"
    if [ -f "$cfg" ] && grep -qi '^[[:space:]]*networkingMode[[:space:]]*=[[:space:]]*mirrored' "$cfg"; then
      echo "    WSL mirrored networking is on"
    else
      # Add the setting under [wsl2], creating the section if the file has none.
      if [ -f "$cfg" ] && grep -qi '^\[wsl2\]' "$cfg"; then
        awk 'BEGIN{done=0} {print} tolower($0) ~ /^\[wsl2\]/ && !done {print "networkingMode=mirrored"; done=1}' "$cfg" > "$cfg.new"
        mv "$cfg.new" "$cfg"
      else
        printf '\n[wsl2]\nnetworkingMode=mirrored\n' >> "$cfg"
      fi
      echo "    Turned on WSL mirrored networking in $cfg, so WSL can reach the game."
      echo "    It takes effect after a WSL restart: in Windows PowerShell run"
      echo "        wsl --shutdown"
      echo "    then open WSL again and re-run this script."
      exit 0
    fi
  fi
fi

api() { curl -fsS -m 5 "$URL$1" 2>/dev/null; }
if ! api / >/dev/null; then
  echo "    Start Slay the Spire 2 (with mods enabled). Waiting for it on $URL ..."
  until api / >/dev/null; do sleep 3; done
fi
echo "    connected: $(api / | head -c 120)"

say "4/5  the run"
run_seed() {
  api /api/v1/compendium | "$PY" -c \
    'import json,sys; r=(json.load(sys.stdin).get("current_run") or {}); s=r.get("seed"); print(s if isinstance(s,str) else "")' \
    2>/dev/null || true
}
seed="$(run_seed)"
if [ -z "$seed" ]; then
  echo "    In the game, start a NEW run: Ironclad, Ascension 8."
  echo "    Stop on the first screen (Neow's offer) -- don't pick anything. Waiting ..."
  until [ -n "$seed" ]; do sleep 3; seed="$(run_seed)"; done
fi
echo "    run in progress, seed $seed"
echo "    It must be a fresh Ironclad Ascension 8 run still on Neow's first screen."
read -r -p "    Press Enter to let the agent play (Ctrl+C to stop) ... " _

say "5/5  the agent plays"
status=0
"$PY" -m sts2rl.bridge.live --model "$ROOT/models/best.pt" --attach --url "$URL" || status=$?
log="$(ls -t "$ROOT"/runs/live/*.jsonl 2>/dev/null | head -1 || true)"
echo
if [ -f "$log" ]; then
  echo "The full log of this run is $log"
  echo "If it stopped early (a 'desync' -- the simulator disagreed with the game), send that"
  echo "file back: it names the exact floor and the difference, which is what gets fixed."
fi
exit "$status"

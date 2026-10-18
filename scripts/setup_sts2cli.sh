#!/usr/bin/env bash
# Set up sts2-cli -- the game's own engine, headless -- for the real-engine bench
# (python -m sts2rl.bridge.cli_eval).
#
#   bash scripts/setup_sts2cli.sh                                  # find the game
#   bash scripts/setup_sts2cli.sh "/path/to/Slay the Spire 2"
#   STS2CLI_DIR=~/sts2-cli bash scripts/setup_sts2cli.sh            # where to put it
#
# sts2-cli (github.com/wuhao21/sts2-cli, MIT) ships no game code: it copies sts2.dll and
# friends out of YOUR install, IL-patches the copy so it runs without Godot, and builds a
# small .NET program around it. Your game files are only read.
#
# Each sts2-cli release is written against one game build, and the engine's API moves
# between builds. Two pinned revisions are tried in turn; the first that builds against
# your game AND opens a run the way the simulator does (python -m sts2rl.bridge.cli
# --smoke) is kept:
#   d11aa88  (2026-05-30)  written for Steam build 23372702 (v0.106.1)
#   084d1aa  (2026-09-07)  written for v0.111.0 (beta)
# The public v0.107.1 (Steam build 23811903) sits between the two.
#
# Needs git, python3 and the .NET 9 SDK (setup.sh installs one into ~/.dotnet).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CLI_DIR="${STS2CLI_DIR:-$ROOT/third_party/sts2-cli}"
REPO="https://github.com/wuhao21/sts2-cli.git"
REVISIONS=(d11aa883b582dd68bd39b331f3370746b30d447e 084d1aa3d8e118ca7ce8d8774ad16d6be9c92367)

if [ -x "$HOME/.dotnet/dotnet" ]; then
  export DOTNET_ROOT="$HOME/.dotnet"
  export PATH="$HOME/.dotnet:$PATH"
fi
if ! command -v dotnet >/dev/null 2>&1 || ! dotnet --list-sdks 2>/dev/null | grep -q '^9\.'; then
  echo "The .NET 9 SDK is needed; run 'bash setup.sh' first (it installs one into ~/.dotnet)." >&2
  exit 1
fi

PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || PY=python3

echo "==> game"
GAME_DIR="${1:-${STS2_GAME_DIR:-}}"
if [ -z "$GAME_DIR" ]; then
  for candidate in \
    "/mnt/c/Program Files (x86)/Steam/steamapps/common/Slay the Spire 2" \
    "/mnt/d/SteamLibrary/steamapps/common/Slay the Spire 2" \
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
  echo "Could not find Slay the Spire 2. Pass its folder:" >&2
  echo "  bash scripts/setup_sts2cli.sh \"/mnt/c/Program Files (x86)/Steam/steamapps/common/Slay the Spire 2\"" >&2
  echo "(Steam: right-click the game > Manage > Browse local files shows it.)" >&2
  exit 1
fi
if ! find "$GAME_DIR" -name sts2.dll -print -quit | grep -q .; then
  echo "No sts2.dll under $GAME_DIR -- is that the game's install folder?" >&2
  exit 1
fi
echo "    $GAME_DIR"

echo "==> sts2-cli"
if [ ! -d "$CLI_DIR/.git" ]; then
  git clone --quiet "$REPO" "$CLI_DIR"
else
  git -C "$CLI_DIR" fetch --quiet origin
fi

for rev in "${REVISIONS[@]}"; do
  echo "==> trying sts2-cli ${rev:0:7}"
  git -C "$CLI_DIR" checkout --quiet --force "$rev"
  git -C "$CLI_DIR" clean -fdxq  # a failed attempt must not leave its build or DLLs behind
  if (cd "$CLI_DIR" && bash ./setup.sh "$GAME_DIR") && "$PY" -m sts2rl.bridge.cli --smoke "$CLI_DIR"; then
    echo "$rev" > "$CLI_DIR/.sts2rl-revision"
    echo
    echo "sts2-cli ${rev:0:7} built against your game in $CLI_DIR"
    echo "Smoke test (heuristic agent, 4 runs):"
    echo "  python -m sts2rl.bridge.cli_eval --cli \"$CLI_DIR\" --agent heuristic --runs 4 --sims 16"
    echo "The trained agent on held-out seeds:"
    echo "  python -m sts2rl.bridge.cli_eval --cli \"$CLI_DIR\" --model models/best.pt --runs 32 --split test"
    exit 0
  fi
  echo "    sts2-cli ${rev:0:7} does not build or run against this game version"
done

echo >&2
echo "No pinned sts2-cli revision builds against your game. If the game updated, sts2-cli's" >&2
echo "newest commit may support it: git -C \"$CLI_DIR\" checkout origin/main && (cd \"$CLI_DIR\" && bash setup.sh \"$GAME_DIR\")" >&2
exit 1

# StS2RL — a self-improving Slay the Spire 2 agent

An agent that learns to play **Slay the Spire 2** by playing itself in a fast, headless
simulator of the game, gets measurably better with every training iteration, and can
then play the real game through the STS2MCP mod.

```
            ┌──────────────── self-play (N worker processes) ────────────────┐
            │                                                               │
 seeds ──▶  │  run simulator (C#, NativeAOT, exact game RNG)                 │
            │     ├─ combat: information-set MCTS over resampled clones      │
            │     │          + native heuristic rollouts (0.2 ms each)       │
            │     └─ everything else: score each option's resulting state    │
            │                with the value network, pick the best           │
            └───────────────┬───────────────────────────────────────────────┘
                            │ every visited state + how the run ended
                            ▼
                 replay buffer ──▶ train value net (TD(λ)) ──▶ evaluate on fixed seeds
                            ▲                                        │
                            └──────────── better decisions ◀─────────┘
```

## How it works

**One learned function drives every decision.** The value network looks at a run state —
deck, relics, potions, HP, gold, floor, the map ahead, the act boss, and the fight in
progress — and predicts how well the run will end. Card rewards, pathing, shops, rest
sites, events, Neow and potion use are all chosen by forking the simulator, trying each
option, and asking the network which resulting state is best. Improving that one function
improves all of those decisions at once, and each training iteration's games become the
next iteration's training data.

**Combat is searched, not guessed.** Each move runs information-set MCTS: every simulation
forks the real position with everything the player hasn't seen (draw order, enemy rolls)
re-randomised, so the search can't cheat by reading the future. Branches are keyed by
*which card at which target* rather than hand slot, so five Strikes are one branch.
Leaves are valued in the same units as the rest of the run — the value network's
estimate for "this HP, these potions left" — so the search knows when spending a potion or
HP is worth it for the run, not just for this fight.

**It starts from a sensible prior and learns corrections.** The network is residual on a
hand-written heuristic (progress, HP, potions, a little per non-basic card and per
upgrade) with zero-initialised output, so iteration 0 plays like the heuristic agent
instead of randomly, and training only has to learn what the heuristic gets wrong — which
cards are worth taking, which path to walk, when an elite is worth the HP. (Without this,
an untrained network's near-random preferences made it skip card rewards by coin flip
and play worse than the heuristic it was supposed to improve on.)

**The simulator** is [evangambit/sts2](https://github.com/evangambit/sts2) (MIT), a C#
re-implementation of StS2 v0.107.1 checked against the live game (4,800 tests, per-card live
captures, exact RNG, exact map generation). It's pinned as a git submodule; this repo
carries four patches for it (`patches/sts2/`):

| patch | what it does |
|---|---|
| `0001-multi-act-runs` | Runs used to end at the second floor of Act 2 (a per-act floor check read a cumulative floor counter), and Act 3's ancients had no options, which stranded every run. Now runs play all three acts. |
| `0002-fast-id-lookups` | `Cards.Get(id)` was a linear scan on every call; it dominated the step cost. Now O(1). **A card play went from 174 µs to 8 µs; end turn from 122 µs to 16 µs.** |
| `0003-agent-exports-and-crash-guards` | Native heuristic combat rollouts (20× faster than driving them from Python), and engine exceptions reported as an error status instead of aborting the process. |
| `0004-end-turn-flush-snapshot` | Fixes a crash ("Collection was modified") when an end-of-turn exhaust draws a card mid-flush (e.g. Dark Embrace + Ethereal); the drawn card is flushed instead of vanishing. |

The simulator's own test suite (4,801 tests) passes with all four patches applied.

## Results so far

All numbers are from fixed evaluation seeds (`EVAL000000…`), **Ironclad, Ascension 8**,
greedy play. "Act 1 clear" = beat the Act 1 boss. 95% Wilson intervals in brackets.

RESULTS_TABLE

For scale: the strongest public StS2 bot ([STS2FableBot](https://github.com/cicero225/STS2FableBot),
hand-tuned search + heuristics, ~4,000 live runs of development) reports ~20% full-run wins
at **Ascension 0**. This agent trains at Ascension 8 (tougher enemies, less starting HP,
Ascender's Bane in the deck), because that's what the simulator's live captures pin down.

## Quick start

Linux, macOS or WSL2. Needs git, a C toolchain (clang + zlib on Linux, Xcode CLT on macOS)
and Python 3.11+. The script installs the .NET 9 SDK into `~/.dotnet` if it's missing.

```bash
git clone --recurse-submodules https://github.com/quasarsama-hub/StS2RL
cd StS2RL
bash setup.sh                       # submodule + patches + native build + .venv
source .venv/bin/activate

python -m sts2rl.evaluate --agent heuristic --runs 16     # smoke test
python -m pytest tests -q                                  # 40 tests, ~30 s
```

### Train

```bash
python -m sts2rl.train --name main --workers $(nproc) --iterations 200
```

Each iteration plays `--runs-per-iter` self-play runs, trains, then evaluates on 64 fixed
seeds and keeps the best checkpoint. Everything goes to `runs/main/`
(`metrics.jsonl`, `latest.pt`, `best.pt`, data shards). Stop it any time; the same command
resumes.

It's CPU-bound (search), not GPU-bound: throughput scales with cores. Rough numbers:

| machine | self-play runs/hour (24 sims) |
|---|---|
| 4 cores (this repo's development box) | ~800 early, fewer as runs get longer |
| 16 cores | ~3,000+ |

Knobs that matter most:

* `--sims` (combat simulations per move in self-play) and `--eval-sims`. More is stronger
  and slower; 24–64 for training, 100–400 for evaluation/live play.
* `--runs-per-iter`: more data per iteration = steadier value estimates.
* `--max-act 1` to train on Act 1 only (the part of the simulator verified against the
  live game) before taking on the whole run.

### Evaluate, watch, benchmark

```bash
python -m sts2rl.report   runs/main                                     # learning curve
python -m sts2rl.evaluate --model runs/main/best.pt --runs 200 --sims 128
python -m sts2rl.watch    --model runs/main/best.pt --seed ANYSEED     # see its reasoning
python -m sts2rl.bench    --sims 0,32,100                               # combat-only benchmark
```

`watch` prints every non-combat choice with the value the network gave each option.

## Playing the real game

The bridge (`sts2rl/bridge/live.py`) drives the game through
[STS2MCP](https://github.com/Gennadiyev/STS2MCP), a mod that exposes the game over a local
HTTP API. The simulator runs as a **shadow** of the real run: same seed, same character,
same ascension, so the two stay in lockstep. The agent decides on the shadow exactly as
it does in training and the bridge sends the game the matching command. After every
step it compares the game with the shadow — floor, HP, gold, hand, enemy HP — and
**stops the moment they disagree**, because a diverged simulator would mislead the agent.

**On Windows:** run everything (training and the bridge) inside WSL2 (Ubuntu), and let WSL
reach the game's `localhost:15526` by enabling mirrored networking — put this in
`%UserProfile%\.wslconfig`, then run `wsl --shutdown` once:

```ini
[wsl2]
networkingMode=mirrored
```

Check with `curl -s http://localhost:15526/` from the WSL shell while the game is running.

1. Install STS2MCP into the game's `mods/` folder and enable mods. To let the bridge start
   *seeded* runs itself, use a fork with custom-run support
   ([evangambit/STS2MCP](https://github.com/evangambit/STS2MCP)); with the stock mod,
   use `--attach` instead.
2. Back up your save and consider turning off Steam Cloud while a bot plays.
3. Play **Ironclad on Ascension 8** (what the simulator models).

```bash
# bridge starts a seeded custom run
python -m sts2rl.bridge.live --model runs/main/best.pt --seed MYSEED001 --abandon-existing
# or: start an Ironclad A8 run yourself, stop on the first Neow screen, then
python -m sts2rl.bridge.live --model runs/main/best.pt --attach
```

Every step is logged to `runs/live/<seed>.jsonl`. `--sims 200` (default) is stronger than
training play; raise it if you don't mind slower turns.

**How it's been tested:** I can't run the game in this environment, so the bridge was
checked against 30 full runs recorded from the real game (shipped with the simulator).
Replaying each through the shadow simulator, **all 30 stayed in sync for every step
(~3,000 decisions)**, and the bridge's sim-action → game-command mapping reproduced the
command the real game received **every time**. The live HTTP loop itself reuses the
simulator project's capture driver, which has played many live runs. What has *not* been
exercised yet is shop purchases and multi-card selection screens against the live game.

## Limitations, honestly

* **Act 1 is verified against the live game; Acts 2–3 are modelled from the decompiled
  source but not yet checked against live captures.** Expect the agent's Act 2–3 play to
  be only as good as the simulator there, and expect the live bridge to stop on a desync
  more often in later acts. Act 3's three ancients aren't implemented in the simulator,
  so the agent skips their blessings (harder than the real game, never easier).
* **Ironclad, Ascension 8 only.** The simulator's run setup is Ironclad at A8. The other
  four characters' cards are all implemented in the simulator; adding them is mostly
  run-setup work there.
* **Combat search uses a hand-written rollout policy.** The next big strength gain is
  AlphaZero-style: a learned policy/value network for combat used as the search prior and
  leaf evaluator (the simulator project measured +2 HP/fight from just the policy prior).
* "Extraordinarily good and consistent" is a compute question from here: the loop
  improves with more self-play, and it parallelises across as many cores as you give it.

## Layout

```
setup.sh                 one-shot setup
patches/sts2/            our patches to the simulator
third_party/sts2/        the simulator (git submodule, evangambit/sts2)
sts2rl/
  sim.py                 fast handle on the native run simulator
  gamedata.py            cards / relics / potions / encounters from the generated tables
  combat.py              information-set MCTS for combat
  features.py            run-state encoding (+ map lookahead)
  model.py               value network (residual on the heuristic prior)
  agent.py               the full-run agent
  selfplay.py            crash-tolerant multi-process self-play
  dataset.py             replay buffer + TD(lambda) targets
  train.py               the improvement loop
  evaluate.py            fixed-seed evaluation with confidence intervals
  report.py              learning curve from a training run
  watch.py               narrated single run
  bench.py               combat benchmark on hard fights
  bridge/live.py         real-game play through STS2MCP with a shadow simulator
tests/                   unit tests + real-game-trace bridge tests
```

## Credits

The simulator is [evangambit/sts2](https://github.com/evangambit/sts2), forked from
[Zamiell/slay-the-spire-2-emulator](https://github.com/Zamiell/slay-the-spire-2-emulator)
(both MIT). The game bridge builds on [STS2MCP](https://github.com/Gennadiyev/STS2MCP) (MIT).
Slay the Spire 2 is © Mega Crit; this project is unaffiliated and for personal/research use.

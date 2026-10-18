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
carries five patches for it (`patches/sts2/`):

| patch | what it does |
|---|---|
| `0001-multi-act-runs` | Runs used to end at the second floor of Act 2 (a per-act floor check read a cumulative floor counter), and Act 3's ancients had no options, which stranded every run. Now runs play all three acts. |
| `0002-fast-id-lookups` | `Cards.Get(id)` was a linear scan on every call; it dominated the step cost. Now O(1). **A card play went from 174 µs to 8 µs; end turn from 122 µs to 16 µs.** |
| `0003-agent-exports-and-crash-guards` | Native heuristic combat rollouts (20× faster than driving them from Python), and engine exceptions reported as an error status instead of aborting the process. |
| `0004-end-turn-flush-snapshot` | Fixes a crash ("Collection was modified") when an end-of-turn exhaust draws a card mid-flush (e.g. Dark Embrace + Ethereal); the drawn card is flushed instead of vanishing. |
| `0005-golden-path-unknown-rooms` | Golden Compass's single-column map typed its "?" rooms as *no node*, which the map screen can't move onto, so every Golden Compass run got stuck on its second floor. |

The simulator's own test suite passes with all five patches applied (4,802 tests, including a
regression test patch 5 adds).

## Results so far

**Ironclad, Ascension 8**, greedy play, simulator runs of the full three-act game. "Act 1
cleared" = beat the Act 1 boss. 95% Wilson intervals in brackets.

**Held-out evaluation** — 128 seeds (`EVAL001000…`) never used for training or for picking
the checkpoint. Same seeds for every row, so the comparison is paired:

| agent | combat sims / move | run score | Act 1 cleared | mean floor reached |
|---|---|---|---|---|
| heuristic, no training | 48 | 0.135 | 14% [9–21%] | 13.8 |
| **trained** (`models/best.pt`) | 48 | 0.180 | **44% [36–52%]** | 18.4 |
| **trained**, deeper search | 128 | 0.195 | **55% [47–64%]** | 19.9 |

Training roughly **tripled the Act 1 clear rate** after ~1,800 self-play runs (about 30
minutes on 4 CPU cores), and more search per move adds another ~11 points. No run has
cleared Act 2 yet: deaths moved from Act 1 elites (Terror Eel, Phantasmal Gardeners) to
Act 1 bosses (Vantom, Kin, Lagavulin Matriarch) and early Act 2 fights.

**The learning curve** (64 selection seeds, 48 sims; ±~6 points of noise per row):

| iteration | self-play runs | score | Act 1 cleared | mean floor |
|---|---|---|---|---|
| heuristic | — | 0.129 | 12% | 13.2 |
| 0 | 200 | 0.143 | 9% | 14.6 |
| 1 | 400 | 0.178 | 33% | 18.2 |
| 3 | 798 | 0.184 | 41% | 18.7 |
| 8 | 1,796 | 0.193 | 45% | 19.7 |
| 10 | 2,196 | 0.192 | 48% | 19.6 |
| 14 | 2,996 | 0.170 | 31% | 17.4 |

Most of the gain came in the first ~800 runs, then it plateaued around 40–45% at this
scale (iterations 6–14 are within noise of each other). Getting past the Act 1 bosses is
now a combat-strength problem more than a deck-building one; see *Pushing it further*.
`python -m sts2rl.report runs/<name>` prints this table for your own runs.

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
python -m sts2rl.watch --model models/best.pt --seed ANYSEED   # the trained agent
```

`models/best.pt` is the checkpoint from the results below; you can continue training from
it (see `models/README.md`) instead of starting over.

### Train

```bash
python -m sts2rl.train --name main --workers $(nproc) --iterations 200
```

Each iteration plays `--runs-per-iter` self-play runs, trains, then evaluates on 64 fixed
seeds and keeps the best checkpoint. Everything goes to `runs/main/`
(`metrics.jsonl`, `latest.pt`, `best.pt`, data shards). Stop it any time; the same command
resumes.

It's CPU-bound (search), not GPU-bound: throughput scales with cores. Rough numbers:

| machine | self-play runs per hour (24 sims, at the current skill level) |
|---|---|
| 4 cores (this repo's development box) | ~4,000 including training and evaluation time |
| 16 cores | ~15,000 |

Runs get longer as the agent gets better (a full three-act run is ~3× an Act 1 death),
so expect these to fall over time.

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
python -m sts2rl.bridge.live --model models/best.pt --seed MYSEED001 --abandon-existing
# or: start an Ironclad A8 run yourself, stop on the first Neow screen, then
python -m sts2rl.bridge.live --model models/best.pt --attach
```

Every step is logged to `runs/live/<seed>.jsonl`. `--sims 200` (default) is stronger than
training play; raise it if you don't mind slower turns.

**How it's been tested:** the bridge has not yet been run against the live game. It was
checked against 30 full runs recorded from the real game (shipped with the simulator).
Replaying each through the shadow simulator, **all 30 stayed in sync for every step
(~3,000 decisions)**, and the bridge's sim-action → game-command mapping reproduced the
command the real game received **every time**. The live HTTP loop itself reuses the
simulator project's capture driver, which recorded those runs. What has *not* been
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

## Pushing it further

In rough order of expected payoff:

1. **More self-play, in bigger batches.** The curve above came from a 4-core box in under
   an hour and plateaued because each iteration's 200 runs carry little signal about
   individual cards. Give it 16–64 cores; raise `--runs-per-iter` to 500–1000 and
   `--eval-runs` to 200 so improvements are measured, not guessed.
2. **A learned combat network** (AlphaZero-style policy prior + leaf value, trained on the
   search's own visit counts). Deaths now concentrate on act bosses, which are long
   fights where a hand-written rollout policy is the weakest link.
3. **Deeper search where it matters.** Bosses already get 4× the simulations; live play
   defaults to 200 per move. Evaluation is cheap to run at 400+.
4. **Simulator coverage.** Every Act 2–3 divergence fixed upstream (or by a patch here)
   makes the agent's later-act play transfer better to the real game. The bridge's
   desync log (`runs/live/*.jsonl`) is exactly the evidence needed to find them.

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
models/                  trained checkpoint (best.pt) and its held-out evaluation
results/                 records of the development training run (curve, logs, data)
```

## Credits

The simulator is [evangambit/sts2](https://github.com/evangambit/sts2), forked from
[Zamiell/slay-the-spire-2-emulator](https://github.com/Zamiell/slay-the-spire-2-emulator)
(both MIT). The game bridge builds on [STS2MCP](https://github.com/Gennadiyev/STS2MCP) (MIT).
Slay the Spire 2 is © Mega Crit; this project is unaffiliated and for personal/research use.

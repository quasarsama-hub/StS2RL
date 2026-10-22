# StS2RL — a self-improving Slay the Spire 2 agent

An agent that learns to play **Slay the Spire 2** by playing itself in a fast, headless
simulator of the game, gets measurably better with every training iteration, and can
then play the real game — through the STS2MCP mod, or headless on the game's own engine
through sts2-cli, hundreds of runs at a time.

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
carries fourteen patches for it (`patches/sts2/`):

| patch | what it does |
|---|---|
| `0001-multi-act-runs` | Runs used to end at the second floor of Act 2 (a per-act floor check read a cumulative floor counter), and Act 3's ancients had no options, which stranded every run. Now runs play all three acts. |
| `0002-fast-id-lookups` | `Cards.Get(id)` was a linear scan on every call; it dominated the step cost. Now O(1). **A card play went from 174 µs to 8 µs; end turn from 122 µs to 16 µs.** |
| `0003-agent-exports-and-crash-guards` | Native heuristic combat rollouts (20× faster than driving them from Python), and engine exceptions reported as an error status instead of aborting the process. |
| `0004-end-turn-flush-snapshot` | Fixes a crash ("Collection was modified") when an end-of-turn exhaust draws a card mid-flush (e.g. Dark Embrace + Ethereal); the drawn card is flushed instead of vanishing. |
| `0005-golden-path-unknown-rooms` | Golden Compass's single-column map typed its "?" rooms as *no node*, which the map screen can't move onto, so every Golden Compass run got stuck on its second floor. |
| `0006-treasure-chest-relic` | Treasure chests paid their gold but **never their relic**: the game's `treasure_room_relics` RNG stream and shared relic bag were built and cloned but never drawn from, so every simulated run was a relic or two short by mid-Act 1. Chests now offer their relic on the reward screen (rarity rolled on that stream, drawn from the shared bag). Checked against every chest in the recorded real-game runs: **11 of 11 offer exactly the relic the game offered** (Vajra, Red Mask, Oddly Smooth Stone, Stone Cracker, Blood Vial, …). Also, Silver Crucible ("the first Treasure Chest you open is empty") emptied every chest; now only the first. |
| `0007-fourth-reward-observation-offset` | Lasting Candy's fourth card-reward option was written into the deck block of the observation instead of its own (an offset used without the run block's base), so nothing could read it. |
| `0008-reward-skip-abandons-the-screen` | Skipping a rewards screen gave up only its queued potions. An unclaimed card reward, the offers queued behind it, Kaleidoscope's, the relic and bonus relics, Prayer Wheel's reward still owed and Lasting Candy's option all stayed in the state and came back on a later screen that never had them (an unclaimed Prayer Wheel reward added up fight after fight). Now a skip abandons what is on the screen, as in the game, and only that: it is also the plain Proceed off an emptied screen, and an Orrery bought at the shop still owes its five. Neow's Bones' screen can no longer be skipped (the game offers it WithSkippingDisallowed). |
| `0009-rest-site-proceed-after-an-option` | A rest site's Proceed was offered from the start, so a run could leave with nothing done; the game keeps the button off until an option is taken (all 27 recorded rest sites). The trained agent took that way out in a few held-out runs (EVAL001019 left two rest sites, at 71/80 and 38/80 HP). Also, a rest left by Dream Catcher's card reward kept its result page pending, and the next rest site's first move did nothing but leave. |
| `0010-direct-card-rewards-offer-every-screen` | The Trial (a Nondescript defendant found guilty: two card rewards) and Colorful Philosophers (three) offered only the first; answering it went straight to the map, and the rest stayed queued for whatever rewards screen came next. Now each is offered in turn, as sts2-cli shows them. |
| `0011-screens-opened-from-rewards-return-to-them` | A relic claimed off a rewards screen that opens a screen of its own (Precise Scissors, Pomander, Lead Paperweight, …) now comes back to the rewards screen; it went to the map, Neow or the event instead, and the rest of the screen was lost — a fight's card reward, or Neow's Bones' second relic and its curse (whose flag then held Proceed off on later screens). Also: an elite's relic flag no longer outlives its screen (the next rest-site or Trial card reward handed out a free relic), Dream Catcher's card shares Tiny Mailbox's rewards screen instead of vanishing, and Smith is spent like every other rest option, so Miniature Tent's second option survives it. |
| `0012-only-modelled-potions-can-be-drunk` | **55 of the 63 potions have no effect in the simulator yet** (it says so: "populated incrementally as potions are reverse-engineered"), but every one could be drunk — as a no-op that only emptied the slot, which the game never does. The trained agent drank them 546 times on the 128 test seeds, and on the real engine every such potion would end the run as a desync. Now only the eight modelled ones (Block, Clarity, Duplicator, Gigantification, Powdered Demise, Radiant Tincture, Regen, Shackling) can be drunk; the agent holds the rest. Fairy in a Bottle is never drunk by hand (it is Automatic) and keeps its revive. |
| `0013-relics-claimed-off-a-rewards-screen` | 0011 decided "back to the rewards screen" from leftover reward state, which can outlive its screen (The Future of Potions left a card reward flagged), so a later shop removal or event selection could land on a phantom rewards screen; now only a screen a rewards screen opened goes back to it. A relic claimed off a rewards screen also does what its pickup does however it was obtained: Small Capsule and Toy Box queue their relics ahead of the screen's others instead of wiping them (Neow's Bones' second relic), Scroll Boxes opens its bundle, Lost Coffer and Kaleidoscope add their rewards, and Neow's Bones' curse lands after a removal its relic opened (Precise Scissors could remove it). And a removal the Forbidden Grimoire owes after a boss no longer ends the run at that boss. |
| `0014-modelled-potions-match-their-text` | Three of the eight drinkable potions were incomplete: Clarity Extract now also draws its card at once, Radiant Tincture gives its energy at once, and Block Potion's 12 Block is unpowered (Frail, Dexterity and Vambrace changed it). |

The simulator's own test suite passes with all fourteen patches applied (4,837 tests, including
regression tests patches 5–14 add).

## Results so far

**Ironclad, Ascension 8**, greedy play, simulator runs of the full three-act game. "Act 1
cleared" = beat the Act 1 boss. 95% Wilson intervals in brackets.

**Held-out evaluation** — 128 test seeds (`EVAL001000…`), never used for training or for
picking the checkpoint. Same seeds for every row, so the comparison is paired. Measured
with all fourteen simulator patches, and with the agent unable to walk away from an event (a
move the simulator allows and the game does not; see the sts2-cli section)
(`results/dev-run/final-eval-patch0006/`):

| agent | combat sims / move | run score | Act 1 cleared | Act 2 cleared | mean floor reached |
|---|---|---|---|---|---|
| heuristic, no training | 48 | 0.142 | 24% [18–32%] | 0 | 14.5 |
| **trained** (`models/best.pt`) | 48 | 0.195 | **55% [47–64%]** | 1 run | 19.9 |
| **trained**, deeper search | 128 | 0.207 | **62% [54–70%]** | 2 runs | 21.1 |

Paired, at 48 sims: the trained agent clears Act 1 on 71 seeds to the heuristic's 31 (68
seeds differ, exact McNemar p < 0.001), and its run score is higher by +0.053 [+0.038,
+0.068]. Training **more than doubled the Act 1 clear rate** after ~1,800 self-play runs
(about 30 minutes on 4 CPU cores). Deeper search gets further on the same seeds (run score
+0.012 [+0.002, +0.022], p = 0.01; 1.2 floors) but does not clear Act 1 measurably more
often (80 vs 71 seeds, 29 differ, p = 0.14). Deaths moved from Act 1 hallway fights
(Phantasmal Gardeners, Terror Eel, Bygone Effigy, Phrog Parasite) to the Act 1 bosses
(Vantom, Lagavulin Matriarch), Slumbering Beetle and the early Act 2.

The checkpoint was trained *before* patches 0006–0014 and the event fix. The trained rows
read 44% / 55% with none of them (`results/dev-run/final-eval/`), 52% / 66% with the chest
relics but the walk-away still allowed (commit 8f987fd), 59% / 56% with rest sites still
offering Proceed from the start (commit d2235b2), 59% / 67% with every potion still
drinkable (commit 0cade94) and 56% / 63% before patches 0013–0014 (commit 60afc78); the
heuristic went 14% → 27% → 26% → 25% → 25% → 25% → 24%. The later fixes change the move
only at a few rest sites, potions or relic screens, yet most runs play out differently
after each: every candidate move the agent weighs draws from its random stream, so one
candidate fewer re-rolls every decision after it. On 128 seeds a single rate is good to
about ±8 points (95%), so read the history as "chest relics help a lot, the rest is
noise" rather than to the point. (A measurement in between, commit 068659c, read 55% /
59% and 30% because of a bug that let the agent walk away from events that build their
options at run time.)

**The learning curve** (64 validation seeds, 48 sims, simulator before patch 0006; ±~6
points of noise per row):

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
python -m pytest -q                                        # 77 tests, ~45 s
python -m sts2rl.watch --model models/best.pt --seed ANYSEED   # the trained agent
```

`models/best.pt` is the checkpoint from the results below; you can continue training from
it (see `models/README.md`) instead of starting over.

### Train

```bash
python -m sts2rl.train --name main --workers $(nproc) --iterations 200
```

Each iteration plays `--runs-per-iter` self-play runs, trains, then evaluates on the
**validation** seeds and puts the new checkpoint up against the current champion on those
same seeds. It replaces the champion (`best.pt`) only if its run score is higher with
p < 0.05 (paired sign-flip permutation test over seeds) *and* it shows no significant
drop in Act 1 clears or wins (exact McNemar). A checkpoint that is merely not worse stays
out — that is how noise becomes "the best checkpoint". Each test stands on its own, so over
a long run an occasional false promotion is still expected: confirm a final champion on
the test split. The first champion is the checkpoint the run starts from (`best.pt`, or a
`latest.pt` you put there to continue from), else the untrained heuristic. Everything goes
to `runs/main/` (`metrics.jsonl`, `latest.pt`, `best.pt`, `champion.json`, data shards).
Stop it any time; the same command resumes — the champion's cached results are reused
only while the settings, the simulator build, the agent code and `best.pt` itself are
unchanged.

Seeds come in two disjoint splits: **validation** (`EVAL000000–000999`) is what training
selects on; **test** (`EVAL001000+`) is never touched by training and is only for final
numbers (`--split test`). Evaluation is deterministic per (checkpoint, seed, settings), so
comparisons are paired:

```bash
python -m sts2rl.compare --a models/best.pt --b runs/main/best.pt --runs 200 --split test
```

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
* `--eval-runs`: the champion gate's sample size. 64 only detects big jumps; 200+ lets
  smaller real improvements through.
* `--max-act 1` to train on Act 1 only (the part of the simulator verified against the
  live game) before taking on the whole run.

### Evaluate, watch, benchmark

```bash
python -m sts2rl.report   runs/main                                     # learning curve
python -m sts2rl.evaluate --model runs/main/best.pt --runs 200 --sims 128 --split test
python -m sts2rl.compare  --a heuristic --b runs/main/best.pt --runs 128 --split test
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
step it compares the game with the shadow — floor, HP, gold, relics, hand, enemy HP — and
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

## Testing on the real game engine, headless (sts2-cli)

[sts2-cli](https://github.com/wuhao21/sts2-cli) (MIT) runs **your installed game's own
code** — it copies `sts2.dll` out of your Steam folder, patches the copy to run without
Godot, and plays a run over stdin/stdout. No window, no mod, no clicking, so the agent can
play hundreds of real-engine runs unattended. sts2-cli can't fork a run, so the simulator
rides along as a lockstep shadow exactly as in the live bridge: every screen is matched,
the shadow steps through what the engine resolves by itself (gold/relic/potion pickups,
chests, "proceed" pages), and after every step HP, gold, floor, deck, relics, potions, the
hand, energy and every enemy's HP are compared.

That makes it two tools in one:

* **A fidelity test of the simulator against the game you own.** The first disagreement on
  each seed ends that run with its exact floor and reason ("enemy hp: game=[12] sim=[18]")
  and a full trace. That is precisely the evidence needed to fix the simulator.
* **Real-engine results.** Seeds and agent seeds match `sts2rl.evaluate` (and so do the
  default `--sims`/`--samples`), so each run is the twin of the simulator run with the same
  settings, and a run that stays in sync to the end finishes exactly where the simulator
  said it would (checked: 80 of 80 runs identical against the stand-in engine below).
  Outcomes are reported only over the runs that were still in sync when each was decided.

```bash
bash scripts/setup_sts2cli.sh            # finds the game (WSL: /mnt/c/Program Files (x86)/Steam/...),
                                         # or pass its folder as the first argument
python -m sts2rl.bridge.cli_eval --cli third_party/sts2-cli --agent heuristic --runs 4 --sims 16
python -m sts2rl.bridge.cli_eval --cli third_party/sts2-cli --model models/best.pt --runs 32 --split test
```

Each sts2-cli revision is written against one game build. The setup script tries two
pinned revisions — `d11aa88` (for v0.106.1) and `084d1aa` (for v0.111.0 beta); the public
v0.107.1 sits between them — and keeps the first that builds against your game *and*
opens a run exactly as the simulator does (`python -m sts2rl.bridge.cli --smoke`). Output
lands in `runs/cli/<timestamp>/`: one JSONL trace per seed and `summary.json`.

**How it's been tested:** not yet against the real engine — there is no game install where
this was written. Every mapping was checked line by line against sts2-cli's source
(`RunSimulator.cs`), and the bench was tested against `tests/fake_sts2cli.py`, a
simulator-backed stand-in that speaks sts2-cli's protocol and keeps its habits: 80 full
runs (48 heuristic, 32 with the trained model) in sync from Neow to the end through act
transitions, shops, events, rest sites and multi-card selections, every one finishing
exactly where its simulator twin does; a deliberately injected 1-HP error is caught on
the floor it appears.

What the bench has to work around, and how:

* **sts2-cli's own departures from the game.** It heals between acts the moment the boss
  dies and rounds the heal *up*; the game rounds down (a capture: 264/280 → 276), so the
  setup script patches sts2-cli (`patches/sts2-cli/`). It skips the result pages of events
  and rest sites, and makes a forced one-card pick without a screen; the shadow steps
  through those too. It shows only a fight's first card reward (Prayer Wheel, White Star or
  The Hunt's extra ones are dropped; an event's are all shown) and a rest site's first
  option (Miniature Tent): the shadow forfeits the same, and each run counts its forfeits —
  a run with one can end differently from its twin.
* **Moves the simulator has and the game doesn't** (rerolling or sacrificing a card reward,
  which sts2-cli has no command for) are never chosen: the agent evaluates them but takes
  its best real move, and each run counts how often that happened. Walking away from an
  event is gone from the agent altogether — the game has no such button, except where an
  event really offers one (Welcome to Wongos' Leave) — and so is leaving a rest site before
  taking an option (simulator patch 0009). The stand-in engine refuses the latter, though
  sts2-cli itself would force its way to the map.
* **What counts as what.** A move the engine refuses although the simulator allowed it is a
  desync, as is any disagreement in the comparison. sts2-cli itself failing (an exception,
  whether it arrives with a stack trace or as the plain message sts2-cli writes when it
  catches the game's own — "Buy relic failed: …" —, a timeout, an unknown screen) is an
  engine error, and a run that crashes its worker
  process is re-run alone and, if it crashes again, recorded as one; the others carry on.
* **Not modelled by the simulator, so they end a run as a desync:** Act 3's ancients, and
  the card Welcome to Wongos' Leave downgrades (the game's option text says it does; which
  card, off which random stream, is not known without the game's code).
  **Checked only in part:** a selection screen of more than 10 cards, whose cards past the
  10th the simulator's observation does not show (they are matched by position alone).

## Limitations, honestly

* **Act 1 is verified against the live game; Acts 2–3 are modelled from the decompiled
  source but not yet checked against live captures.** Expect the agent's Act 2–3 play to
  be only as good as the simulator there, and expect the live bridge to stop on a desync
  more often in later acts. Act 3's three ancients aren't implemented in the simulator,
  so the agent skips their blessings (harder than the real game, never easier).
* **Most potions are not modelled.** Only 8 of the 63 have an effect in the simulator, so
  the agent drinks only those and carries the rest (simulator patch 0012). Every potion it
  holds is a fight it plays without help the real game would give. sts2-cli reports each
  potion's numbers (`vars` in its state), which is the data to port the rest from.
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
   makes the agent's later-act play transfer better to the real game. The sts2-cli
   bench's traces (`runs/cli/*/`) and the live bridge's desync log (`runs/live/*.jsonl`)
   are exactly the evidence needed to find them.

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
  evaluate.py            fixed-seed evaluation (validation / test splits)
  stats.py               paired statistics: McNemar, permutation test, the champion gate
  compare.py             paired comparison of two agents on the same seeds
  report.py              learning curve from a training run
  watch.py               narrated single run
  bench.py               combat benchmark on hard fights
  bridge/live.py         real-game play through STS2MCP with a shadow simulator
  bridge/cli.py          real-engine play through sts2-cli with a shadow simulator
  bridge/cli_eval.py     batch real-engine evaluation: fidelity + results
scripts/setup_sts2cli.sh fetch, patch and build sts2-cli against your game
tests/                   unit tests, real-game-trace bridge tests, sts2-cli bench tests
models/                  trained checkpoint (best.pt) and its held-out evaluation
results/                 records of the development training run (curve, logs, data)
```

## Credits

The simulator is [evangambit/sts2](https://github.com/evangambit/sts2), forked from
[Zamiell/slay-the-spire-2-emulator](https://github.com/Zamiell/slay-the-spire-2-emulator)
(both MIT). The game bridge builds on [STS2MCP](https://github.com/Gennadiyev/STS2MCP) (MIT); the
headless real-engine bench on [sts2-cli](https://github.com/wuhao21/sts2-cli) (MIT).
Slay the Spire 2 is © Mega Crit; this project is unaffiliated and for personal/research use.

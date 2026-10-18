# Spire 2 RL — Ironclad A0

A repeat-run learner with Maskable PPO, saved policy/optimizer checkpoints,
episode journals, disjoint training/validation/test seeds, paired evaluation,
and a conservative champion gate. One character, one ascension level.

**Current boundary:** PPO trains in a pinned external Python STS2 emulator.
Transfer to the installed game is unverified. The native game runner has a
tested transport adapter and an optional Windows build/startup probe; a native
Gym observation/action adapter and live parity checks are still needed before
these checkpoints can play or train in the actual game.

## Windows first run

Install Python 3.12 x64 and Git for Windows if missing. Extract this project,
open PowerShell in its folder, then run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
.\.venv\Scripts\spire2-rl.exe train --experiment experiments/ironclad-a0 --steps 20000
.\.venv\Scripts\spire2-rl.exe evaluate --experiment experiments/ironclad-a0 --episodes 100
```

Setup detects the Steam library and GPU, uses an isolated CPU environment,
and fetches the exact external backend revisions in `backends.lock.json`.
The first 20,000 steps are an operational milestone, not an expectation of
full-run wins. PPO rounds the budget up to a complete rollout.

Run the same `train` command again to continue learning from the latest weights
and optimizer. Existing experiments keep their saved configuration. Active
episodes restart on new seeds; resume is not an exact mid-combat replay.
Reservation files prevent seed reuse after an interrupted process.
Strict resume currently requires the same checkout location, Python/dependency
versions and source fingerprints. Moving an existing experiment to another
machine is not supported; start a new local experiment after extracting the
starter. Included verification results are evidence from this workspace.

Native setup additionally needs .NET 9 SDK and your installed STS2:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1 -BuildNative
.\.venv\Scripts\spire2-rl.exe native-probe --config artifacts/windows-install.json
```

This copies and patches assemblies inside `.external`; setup does not modify
the Steam installation. Build/startup success does not establish mechanics
parity. See [Windows details](docs/windows.md) and [native limitations](docs/native-backend.md).

## What repeated runs change

Each rollout records visible observations, legal actions, rewards and outcomes.
PPO updates the policy and value network from those trajectories, then saves
weights and optimizer state. The next batch uses the updated policy. The
runner never asks an LLM to rewrite its code after a loss.

The default objective is sparse: +1 for winning the selected scope, -1 for
game death. Full runs cover three acts. There is a separately labeled starter
combat diagnostic and optional Act 1 curriculum; those wins never count as
full-run wins. `--reward-shaping` creates a separate experimental objective,
but champion selection still uses actual wins and floor reached.

The policy sees current intents, current hand, owned deck/relic/potion features,
visible map information and the offered choices. It does not see RNG state,
hidden draw order or internal future enemy state. Card IDs use learned
embeddings; public choices are encoded alongside their action slots.

Simulator errors abort and are logged separately. Step limits are truncations,
not deaths. Native transport deadlines kill the worker and record an abort;
an action with uncertain delivery is never blindly sent twice.

## Measuring progress

```powershell
.\.venv\Scripts\spire2-rl.exe status --experiment experiments/ironclad-a0
.\.venv\Scripts\spire2-rl.exe evaluate --experiment experiments/ironclad-a0 --episodes 200 --split validation
```

Evaluation uses the same held-out seeds for candidate and baseline and reports
win intervals, paired win/floor differences, timeouts and aborts. The default
baseline is legal random actions. Comparing against a stronger fair heuristic
is an important next learning milestone; beating random alone establishes
very little.

`promote --experiment ... --report <evaluation.json>` updates the champion
pointer only when at least 100 paired validation runs pass the recorded gate.
When a champion exists, evaluation must compare against that checkpoint, using
`--baseline <absolute champion ZIP path>`. Test seeds are reserved for final
assessment and cannot promote a checkpoint. Repeated validation selection can
still overfit that partition; use the test split sparingly.

Artifacts inside each experiment:

- `experiment.json`: environment, objective, seed scheme and dependency provenance.
- `episodes.jsonl`: completed and interrupted training episodes.
- `checkpoints/`, `latest.json`: checksum-verified policy/optimizer snapshots.
- `seed-cursors/`: durable next episode indices per worker.
- `evaluations/`: paired held-out episode records and confidence intervals.
- `sessions.jsonl`: training budget, duration and actual parameter change.
- `champion.json`: accepted checkpoint and evidence, when the gate passes.

If an engine exception occurs, inspect its recorded seed/action rather than
disable strict mode. If learning runs but strength does not improve, examine
observations, credit assignment and a fair imitation warm start. See the
[learning milestones and risks](docs/learning-plan.md).

The current sparse objective leaves budget truncations neutral. A policy can
therefore stall instead of risking death; an earlier observation schema's
candidate did this on 5/100 validation seeds and was rejected. The packaged
revision exposes selection prompts and bounds; its first candidate had zero
stalls in 100 validation runs, mean floor 4.00 versus random's 3.45, and zero
full-run wins. See the [measured results](docs/initial-results.md). A separate experiment with
an explicit decision cost or fair imitation warm start is the next learning
step. Do not silently change the objective of a saved experiment.

## Development

Use `scripts/setup_unix.sh` on Linux/macOS, or create a venv, install CPU torch
2.8.0 and `pip install -e '.[dev]'`, then fetch the emulator at the lockfile
revision. The default checkout path is `.external/sts2-rl-agent`; `--upstream`
and `SPIRE2_UPSTREAM_PATH` can select another clean pinned checkout.

```bash
SPIRE2_UPSTREAM_PATH=/path/to/sts2-rl-agent .venv/bin/python -m pytest
```

Tests cover engine errors, legal action handling, visibility, durable seed
reservations, corruption detection, uncertainty and native child-process
protocol failures. Native protocol tests use controlled child processes;
they do not substitute for live Windows/game integration tests.

## Sources and attribution

- [vvsuby/sts2-rl-agent](https://github.com/vvsuby/sts2-rl-agent): external emulator,
  based on [zhiyue/sts2-rl-agent](https://github.com/zhiyue/sts2-rl-agent), references
  game v0.106.1. No explicit upstream license; fetched separately, not bundled.
- [wuhao21/sts2-cli](https://github.com/wuhao21/sts2-cli): MIT native headless
  runner; scheduling/stub adaptations still require parity validation.
- [sb3-contrib MaskablePPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html).

No game assemblies or decompiled source are distributed with this project.
Slay the Spire 2 belongs to Mega Crit.

## Combined learning sessions

For an existing event-isolated learner, [learning sessions](docs/learning-sessions.md)
combine source verification, a fixed training allowance, one starting/final
comparison, bounded timeout diagnosis and compact reports:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_learning.ps1
```

Repeating the same command resumes or displays its completed result. An explicit
later session preserves the previous final model and PPO optimizer.

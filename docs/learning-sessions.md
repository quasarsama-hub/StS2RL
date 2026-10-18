# One command for a complete learning session

From the project folder, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_learning.ps1
```

The default starts from the existing event-isolated cycle 1 policy in
`experiments/ironclad-a0-act1-event-isolated-cycles/cycle-0001`. It verifies the
saved evidence and model, diagnoses recorded timeouts, and creates a separate
learning session. It retains the policy, value network, PPO Adam state,
cumulative decision counter and training seed high-water marks. The simulator,
action masks, rewards, PPO settings, four environments with 256-decision
rollouts, two CPU training threads, 4,000-decision episode cap and deterministic
evaluation stay unchanged.

The default requests **240,000 additional decisions**. One fresh 1,000-seed
evaluation block is fixed before using it. The starting model plays that block
once; after the fixed training allowance, the final model plays it once.
Intermediate checkpoints are recovery and infrastructure records, never models
selected using the held-out results. Three previous 80,000-decision cycles would
run 6,000 evaluation games; this session runs 2,000.

Each training chunk subtracts the actual decisions already collected from one
total target. PPO finishes complete 1,024-decision rollouts, so total rounding
overrun stays below one rollout instead of accumulating separately per chunk.
The optimization removes repeated evaluation and manual steps. It does not
establish stronger play or change the learning algorithm.

## Resume and later sessions

Repeat the same command to resume its remaining original allowance or display
the cached completed result. The model, sample size, schedule and budget are
frozen; the command does not append another allowance after completion.

A later session explicitly starts from the preceding session's saved final
model and optimizer:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_learning.ps1 -Session 2
```

Choose a larger allowance before starting that session:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_learning.ps1 -Session 2 -Steps 1000000
```

`Steps` means additional simulated game decisions collected by PPO on the PC.
It is independent of Codex tokens and API credits. More experience can still
plateau. Saved weights, value network, optimizer and counter survive an
interruption; active games and incomplete rollouts restart on fresh training
seeds. The launcher resolves the project path and its own Python executable, so
it can be invoked by absolute path from any PowerShell directory.

## Automatic diagnostics and reports

The original cycle's incomplete evaluation status and continuation gate remain
unchanged. This is a new, fixed training protocol. Its primary endpoint is
**Act 1 cleared within the unchanged decision cap**, divided by all attempted
evaluation seeds. Timeouts stay a separate outcome and count as nonclears.
They are never renamed deaths or removed from this endpoint's denominator.
Completed-game floor averages are descriptive diagnostics.

Each diagnostic stage replays at most four recorded failures with the exact
frozen policy, seed and original cap, retaining a bounded public-state tail.
Cases not replayed are disclosed. Reproduced legal budget timeouts can be
recorded while the fixed session learns further. A replay mismatch, simulator
exception or logging failure pauses for diagnosis. No forced actions, new
inference rules, reward changes or episode-cap extensions are applied.

The terminal shows concise progress and heartbeats. The detailed log and
`report.txt`, `report.json`, raw episode evidence and `diagnostics.zip` are saved
locally. The terminal prints their locations. The compact diagnostic bundle
contains reports, protocols, journals, bounded trace summaries and a log tail;
it excludes model checkpoints, game files and large action/observation catalogs.
Send the report after completion or the bundle if help is needed.

## Evidence and scope

The final matched comparison reports an exact paired McNemar test and a paired
bootstrap. Resampling the counts of differences -1, 0 and +1 has the same
empirical bootstrap distribution as resampling individual pairs, with much
less Python work. Random streams differ. A zero-width interval does not prove
equal playing strength. Evidence concerns this prespecified starting/final
comparison, not the whole sequence of experiments.

See [release verification](learning-sessions-verification.json) for the new
tests and actual local runtime checks. This release was rebuilt from the user's
uploaded recovery ZIP; its verification belongs to these new source bytes.
The user's trained checkpoint stays on Windows. Native-game transfer remains
unverified.

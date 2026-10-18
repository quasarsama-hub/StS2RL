# Run another bounded Act 1 learning cycle

The completed repaired continuation improved survival on 1,000 clean matched
validation runs: mean floor 9.303 before training versus 10.219 afterward. The
paired gain was 0.916 floors, with a 95% interval of [0.628, 1.210]. Act 1 clears
rose from 13 to 17, but the exact paired test did not establish a clear-rate
improvement (p=0.572). The resulting policy cleared only 1.7% of these runs.

This is evidence of progress from additional repeated-run RL. It justifies
another bounded trial; it does not establish reliable winning or guarantee that
more training will continue helping. The completed training journal recorded
10 wins and 618 deaths, with no timeouts or engine aborts. These changing-policy
training outcomes identify positive reward examples, rather than estimate the
final policy's strength.

## Start cycle 1

From the existing project directory:

```powershell
cd "C:\Users\Junhyuk\Downloads\spire2-rl-starter\spire2-rl"
.\.venv\Scripts\python.exe .\scripts\act1_cycles.py --cycle 1
```

Cycle 1 starts from the completed checkpoint in
`experiments/ironclad-a0-act1-rl-continuation`. New cycles are stored separately
under `experiments/ironclad-a0-act1-cycles`. Each cycle freezes its starting
policy, plays 1,000 fresh validation runs, performs a fixed additional 80,000
PPO decisions, and plays another 1,000 runs on the same validation seeds.
The final printed JSON has stage `after_rl_cycle`. Cycle 1's summary is saved at
`experiments/ironclad-a0-act1-cycles/cycle-0001/cycle-progress.json`; its full
matched evidence is retained alongside it in `cycle-evidence.json`.

The first new evaluation block starts at validation index 2,700, following the
completed continuation's block from 1,700 through 2,699. Later cycles use the
next unused block. Validation runs add no training steps. Both compared policies
use the same finite-selection and potion-capacity interfaces.

Repeating the same cycle command resumes that cycle or reads its completed
result. It does not give an already completed cycle another training budget.
When another cycle is justified, its command uses the next number:

```powershell
.\.venv\Scripts\python.exe .\scripts\act1_cycles.py --cycle 2
```

Each new cycle starts from its immediate predecessor's completed checkpoint.
It preserves policy weights, value weights, PPO Adam state, and cumulative
timestep count. Fresh training episodes begin beyond the predecessor's seed
cursor high-water marks. There is no new imitation, teacher collection, reward
change, or automatic champion promotion.

The requested budget rounds up to complete 1,024-decision PPO rollouts. Starting
from the reported count of 161,848, an uninterrupted additional 80,000-decision
cycle is expected to add 80,896 decisions and finish at 242,744. An interrupted
partial rollout can change this rounding. A resumed game does not restore its
previous active episode or unoptimized rollout exactly.

## Interpret each result

The paired floor interval measures survival progress. An interval entirely above
zero supports a floor gain for that fixed comparison. An exact paired McNemar
test assesses Act 1 clears separately; surviving longer does not prove a higher
clear rate. Timeouts or aborted runs prevent an all-clean positive-gain result.

The launcher checks the preceding cycle's raw matched evaluation evidence before
allowing another cycle. A clean measured gain in floors or clears can justify
another fixed trial; an established clear-rate regression, infrastructure
failure, or inconclusive result requires inspection instead of an automatic
budget extension. This is a continuation rule, not proof of clear-rate
equivalence or noninferiority.

Each reported 95% interval and exact test applies to its individual comparison.
Repeated cycles and decisions based on their results do not turn those intervals
into a single 95% guarantee for the entire training series. Do not combine gains
from different seed blocks as though they came from one paired experiment.

Keep completed checkpoints and their raw evidence. If progress stalls, use the
saved outcomes to investigate combat, deck building, paths, and boss losses
before increasing the budget again. Wins here are emulator Act 1 clears; native
game transfer and visible gameplay remain unverified.

## Launcher verification

The new launcher passed 31 targeted tests covering preserved policy and Adam
state, fixed-budget resumes, fresh seed blocks, raw progress checks, source
immutability, and completed-result caching. Two isolated short local cycles
also exercised actual PPO training and 400 simulated evaluation runs, with no
timeouts or aborts. These used a separate local policy, not the user's weights,
and tested runtime behavior rather than learning strength. The production
positive eligibility path was exercised with persisted test fixtures; an actual
ineligible local parent was rejected without creating another cycle.

See [verification details](act1-cycles-verification.json) for file hashes and
the exact scope of those checks.

# Check the frozen policies after an inconclusive cycle

Cycle 2 completed cleanly but did not establish further improvement on its
1,000 matched validation pairs. Act 1 clears rose from 37 to 50 (3.7% to 5.0%),
with a paired gain interval of [-0.5, +3.0] percentage points and an exact
two-sided McNemar p-value of 0.1766. Average floor changed from 10.842 to 10.620;
the gain interval was [-0.584, +0.121]. Neither established improvement nor an
established regression follows from these results.

Only four seeds were cleared by both policies: the trained policy newly cleared
46 seeds but lost 33 previously successful ones. This could reflect different
or riskier choices, but the aggregate outcomes alone do not identify the cause.
The unchanged continuation gate correctly keeps cycle 3 paused.

## One fresh fixed comparison

Run this command after installing the additive diagnostic patch:

```powershell
cd "C:\Users\Junhyuk\Downloads\spire2-rl-starter\spire2-rl"
.\.venv\Scripts\python.exe .\scripts\check_act1_cycle.py --cycle 2 --episodes 3000
```

The checker freezes cycle 2's exact initial and final checkpoints (reported
counters 242,744 and 323,640), then plays
3,000 matched pairs (6,000 simulated game runs) on fresh validation seeds. For
the reported sequence, the block begins at index 4,700. Each paired run uses the
same repaired interface and seed, with deterministic inference. No PPO updates
or new demonstrations occur. The models, optimizer, training journals, cycle
reports, and champion pointer retain their existing bytes.

The primary measure is Act 1 clear rate, assessed with the existing exact paired
two-sided McNemar test. A positive direction and p-value below 0.05 support a
clear-rate gain for this comparison. Floor reached is secondary; a positive
floor interval does not establish better winning. Infrastructure errors remain
separate and prevent an all-clean positive-gain claim.

The protocol fixes both models, the sample size, inference, seed block, and
checker version before gameplay. Repeating the command reads its completed
report, or restarts an interrupted incomplete evaluation with those same frozen
models and seeds. Changing the sample size or schedule for this diagnostic is
rejected. Do not increase the sample size after seeing the result. This fresh
check is assessed separately from the earlier data-dependent cycle comparison;
do not pool the two into a larger significance claim or treat individual tests
as a global confidence guarantee for the full learning series.

The final printed stage is `frozen_cycle_strength_check`. Full raw rows and
checksums are stored under
`experiments/ironclad-a0-act1-cycles/cycle-0002/diagnostics/cycle-strength-<model hash>/`.
The original cycle result and its training gate remain unchanged; this tool
does not start or authorize another training cycle automatically.

If the fixed fresh sample remains inconclusive, inspect losing trajectories
and combat/deck/path decisions before allocating more training. A statistically
supported gain would justify considering another bounded trial. All clears are
emulator Act 1 clears; native game transfer remains unverified.

## Verification boundary

The targeted tests use persisted model fixtures with real tensors and optimizer
state and verify model
and seed identity, immutable fixed protocols, repeated cached results, malformed
or changed evidence, interrupted evaluation recovery, and unchanged training
files. A separate short local-policy smoke check exercises actual frozen-model
gameplay and raw report validation. Those checks establish runtime behavior;
they do not evaluate the user's policy or run the full 3,000-pair command.

See [release verification](cycle-strength-check-verification.json) for measured
test counts, source hashes, and the exact local smoke scope.

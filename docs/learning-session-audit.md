# Diagnose the saved learner with one frozen controller audit

The first combined learning session supplied 71 successful training runs but
did not establish better held-out play: 47/1,000 Act 1 clears before versus
40/1,000 after, with paired clear-rate change -0.007, interval [-0.024, 0.011]
and exact paired p=0.4998969. The final learner reached the boss stage in 331
runs and cleared in 40; 291 runs died there. These outcomes do not identify
whether combat tactics, earlier attrition or deck construction caused failure.

The uploaded bundle contains no PPO entropy/KL/value-loss/explained-variance
metrics. Policy collapse and critic failure are hypotheses, not diagnoses.
All three supplied exact timeouts are combat stalls; there were no timeouts in
training or the final evaluation. They cannot explain the general low clear
rate. Training took 538.72 seconds for 240,640 decisions, about 446.69/s.

## Run the fixed audit

From the project folder:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check_learning.ps1
```

The launcher resolves the project and its existing Python executable. The
default uses saved session 1. It verifies the source models, metadata and raw
results, then replays the **first 100 ordered seeds of its existing matched evaluation
block with five controllers**. The cohort is selected by index, without
filtering for outcomes:

| Controller | Combat, including combat selection prompts | Other phases |
| --- | --- | --- |
| Previous policy | Frozen starting learner | Frozen starting learner |
| Current policy | Frozen final learner | Frozen final learner |
| Fair teacher | Existing visible-information heuristic | Same heuristic |
| Teacher combat | Heuristic | Frozen final learner |
| Teacher noncombat | Frozen final learner | Heuristic |

This is 500 simulated games and **zero additional training decisions**. The
existing teacher uses no lookahead. Each controller gets its own freshly reset
games and private event objects. The original 4,000-decision cap, observations,
rewards and action masks stay unchanged. The learner's weights, optimizer,
training seed cursors, completed report and checkpoint pointers stay unchanged.

The audit fixes its source, cohort and implementation before playing. Repeating
the same command resumes cached controller results or reads a completed audit;
it does not add training. To inspect a later completed session, choose its
number explicitly, for example `-Session 2`. Do not change the sample size after
seeing outcomes. These seeds already belong to the completed evaluation block and remain
reserved by its original protocol. The audit does not consume a new validation
block or change the completed source report. Current and previous policy
replays must match all seven original gameplay fields; a mismatch requires
inspection.

The output includes a short human report and a compact diagnostic ZIP. The
terminal prints their locations. It records actual fight/boss identity, entry
HP and permanent public deck details, potions, room/card/rest choices, unused
energy and end-turn flags. Flags locate decisions to inspect; they do not
automatically label unused cards or potions as mistakes. Observational errors
are disclosed separately from game failures.

## Interpret the comparison

If replacing combat decisions helps substantially while changing other choices
does not, combat execution is a useful next training target. If changing other
choices helps, inspect routing, draft, rest and event decisions together with
boss-entry HP/deck. Improvements in both hybrids suggest interacting weaknesses.
If the teacher also performs poorly, audit its own limitations and simulator
behavior before using it as a stronger training target.

Replacing decisions changes downstream encounters and decks. The hybrids do
not fight identical restored boss states, and conditional boss-entry statistics
compare different policy-dependent arrivals. This small audit is for diagnosis;
it is not a new learning-strength, promotion or global statistical test. Better
teacher-assisted play does not establish that RL learned that behavior.

One later hypothesis is sparse long-horizon credit: the original code defaults
to GAE lambda 0.95, while clears average about 232 decisions. A longer-credit
lambda 1.0 trial would require a fixed equal-budget unchanged-control branch,
new training metrics and a fresh final clear-rate comparison. The current bundle
does not establish that changing lambda is the right remedy.

See [release verification](learning-session-audit-verification.json) for the
actual source bytes, targeted tests and separate local game runtime checks.
The user's trained models stay on Windows. Native-game transfer is unverified.

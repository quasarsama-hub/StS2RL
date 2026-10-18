# First measured experiment

These are measurements from this workspace, not upstream benchmark claims or
results from the installed game. Backend: the pinned Python emulator, Ironclad
A0, three modeled acts, sparse rewards, four sequential vector environments.

The first policy trained for 2,048 decisions, was saved and reloaded, then
trained for another 18,432 decisions. Both sessions changed policy parameters;
the saved optimizer and total-step counter continued correctly. The training
journal contains 236 completed runs, all deaths, plus eight unfinished episodes
discarded at the two session boundaries. No engine exception occurred.

On 100 paired validation reset seeds:

| Outcome | Trained policy | Legal random baseline |
| --- | ---: | ---: |
| Full-run wins | 0 | 0 |
| Game deaths | 95 | 100 |
| Decision-budget timeouts | 5 | 0 |
| Engine/transport aborts | 0 | 0 |
| Mean floor among completed games | 6.147 | 3.450 |

The paired comparison excludes the five incomplete pairs. On the remaining
95 seeds, the mean floor difference was +2.695, with a paired bootstrap 95%
interval [2.200, 3.221]. This is an early floor-progress diagnostic, not an
estimated full-run win improvement. Excluding timed-out runs can bias the
floor diagnostic; the acceptance gate therefore rejected the checkpoint.
The trained policy used deterministic inference; the random baseline used
seeded legal-action sampling.

A recorded timeout seed, `905902561`, showed repeated selection/deselection
of a Defend during a pending deck choice. The adapter exposed the selection
flag but omitted the prompt before confirmation became available and omitted
the required selection bounds. That observation gap prompted the public-v2 /
public-slot-v3 update: all active selection prompts, bounds and counts now
reach the policy. The original measurement used the previous observation
schema. A fresh experiment is required after that change; previous model
weights are not silently reused against a changed schema.

`artifacts/verification.json` and `artifacts/validation-report.json` contain
the final packaged revision's test, training and evaluation measurements.
Native Windows startup and real-game parity have not been exercised here.

## Final packaged revision

The corrected 32-feature choice schema was trained from scratch for the same
2,048 + 18,432 decision budgets. Training completed 225 games, all deaths,
with eight unfinished session-boundary episodes and zero engine failures.

On the same 100 validation seeds, the final policy and random baseline both
had zero wins and 100 deaths. Both had **zero timeouts and zero engine aborts**.
The final policy reached mean floor **4.00**, compared with **3.45** for random.
The paired mean difference was +0.55, bootstrap 95% interval [0.24, 0.85].
This passed the initial floor-progress gate and was saved as the first local
emulator champion. It remains a weak policy, and no native-game performance
has been established. It must beat this checkpoint, then a stronger fair
baseline, in subsequent validation. Test seeds have not been used.

The different observation dimensions also changed network initialization;
the old and new policies do not isolate the causal effect of the observation
fix. Zero observed stalls on 100 seeds does not guarantee stall-free behavior.

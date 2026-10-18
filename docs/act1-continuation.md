# Continue learning after the verified potion repair

The latest frozen-policy comparison supports better survival among completed
runs: 8.481 mean floors before RL versus 9.362 after it, with a paired gain of
0.879 and a 95% interval of [0.553, 1.205]. Two candidate engine errors prevented
the checker's all-clean flag from passing. The 12 versus 6 Act 1 clears remain
too few to establish a clear-rate improvement (exact paired p=0.210).

The user's repair replay reproduced both original full-inventory potion pickup
errors. With the same saved policy and only the capacity mask added, those runs
completed as deaths on floors 13 and 16. That verifies the specific interface
repair. It does not show additional learning, and these known failure seeds are
not a fresh strength test. The original evaluation and replay evidence remain
intact.

The next bounded experiment performs another 80,000 PPO decisions, without more
imitation, teacher demonstrations, reward changes, or policy redesign. This asks
whether repeated-run RL can further improve the existing learner after the
verified action contract repair.

## Run the continuation

From the existing project directory:

```powershell
cd "C:\Users\Junhyuk\Downloads\spire2-rl-starter\spire2-rl"
.\.venv\Scripts\python.exe .\scripts\continue_act1_rl.py --new-steps 80000 --episodes 1000 --validation-start 1700
```

The defaults read the source at
`experiments/ironclad-a0-act1-rl-trial` and create the separate experiment
`experiments/ironclad-a0-act1-rl-continuation`. The launcher finds the latest
saved potion repair summary and validates it before migration. To select a particular saved
summary, pass its full path with `--repair-check`.

The source policy weights, value weights, PPO Adam optimizer state and timestep
counter are transferred together. The source checkpoint stays frozen. With the
reported source count of 80,952, the requested total is 160,952 decisions. PPO
collects complete 1,024-decision rollouts, so an uninterrupted run is expected to
finish at 161,848: **80,896 additional decisions**. The output reports source,
additional and cumulative counts separately. A resumed partial rollout can
change the exact rounding; it is not bitwise restoration of an active game.

The capacity mask gives the new experiment a distinct identity. The source is
verified under its original factory first; the original experiment is never
reinitialized with the repaired factory. Training starts on new seeds after the
source's saved seed cursor high-water marks. Active episodes and pending rollout
observations restart at the boundary; optimizer state remains restored.

## What the comparison measures

The experiment freezes the transferred learner before additional training and
evaluates it on 1,000 fresh validation seeds starting at index 1,700. It then
trains the bounded budget and evaluates the resulting learner on those same
seeds. Both policies use the same potion-capacity mask. This produces **2,000
evaluation runs** in total, plus the runs collected for training.

Both evaluations use deterministic inference. Validation runs do not add PPO
training steps. The schedule is checked against earlier saved evaluation seeds
and pinned together with the budget and evidence for later resumption. Reusing
the same command resumes unfinished work or reads completed evidence; it does
not silently grant another 80,000 steps.

The primary result is the paired change from the frozen 80,952-step learner to
the continued learner. A paired bootstrap interval reports floor progress;
an exact paired McNemar test assesses the rare Act 1 clear outcomes. The two
questions are reported separately. A positive floor gain does not establish a
higher clear rate. Timeouts and aborted runs remain explicit and prevent an
all-clean positive-gain claim.

The training record identifies wins, deaths, timeouts, interrupted sessions and
ordinary unfinished episodes discarded at a budget boundary. Training policies
change over time, so those counts locate reward examples rather than estimate
the final model's playing strength. A counted win does not prove every related
transition reached an optimizer update, especially after an interruption.

The launcher never promotes a champion automatically. The same saved model can
continue learning, while a failed or inconclusive strength check remains a real
result. If this bounded trial plateaus or regresses, repeatedly expanding its
budget without investigating the failure would not demonstrate progress.
Wins here are simulated Act 1 clears; native game transfer and visible gameplay
remain unverified.

## Verification

Thirty focused tests verify exact source weights, Adam moments and counter
preservation; raw replay evidence and corruption rejection; seed high-water
marks; separate identities; atomic initialization; matched repaired evaluation;
fixed-budget resume; and the absence of imitation and champion promotion.

A separate local operational pilot loaded an existing short policy with the
actual SB3 loader, retained its weights and optimizer exactly, performed 1,024
additional PPO decisions, and played 100 validation runs before and after.
All 200 runs finished without timeouts or infrastructure aborts. That pilot
exercised the lower-level loader, factory, trainer and evaluator; its repair
eligibility gate was tested separately using persisted evidence fixtures.
It used different weights and a smaller budget than the user's forthcoming
trial, so it is not a playing-strength prediction.

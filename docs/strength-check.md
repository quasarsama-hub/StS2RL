# Check the frozen RL trial before allocating more training

The user's first bounded PPO trial improved the point estimate from 8.67 to 9.38 mean floors and from zero to one Act 1 clear across 100 matched validation seeds. Its floor interval [-0.36, 1.78] and clear-rate interval [0, 0.03] did not establish a reliable gain. The next measurement is one larger paired evaluation of the existing models, rather than additional training.

```powershell
.\.venv\Scripts\python.exe .\scripts\check_act1_trial.py --episodes 1000 --validation-start 700
```

This runs 1,000 fresh validation seeds for both the saved cloned initializer and the completed RL candidate: **2,000 simulated runs total**. It prints progress every 50 pairs. The factory retains the trial's original recipe and plan so its fingerprint stays compatible. The larger evaluation has its own frozen protocol; it does not change the training budget or the original 100-run comparison.

Training, imitation, optimizer updates, checkpoint replacement, and champion promotion are not called. The tool saves its protocol and raw evidence under `experiments/ironclad-a0-act1-rl-trial/diagnostics/strength-<candidate-checksum-prefix>/`. Repeating the same command reuses a completed report after checking its models, seeds and statistics. Sample size and seed schedule cannot be changed for that candidate after the diagnostic begins. Interrupted evaluations can be rerun under that same protocol; they do not resume or extend training.

The output separates three findings:

- `establishes_positive_floor_gain`: whether the paired floor interval lies above zero, with complete accounting.
- `establishes_positive_act1_clear_gain`: whether an exact paired clear-outcome test supports more clears, with complete accounting. It uses the exact two-sided McNemar test at 0.05; the bootstrap clear-rate interval is still shown, but cannot by itself establish the rare-win claim.
- `training_record`: win/death/timeout counts from changing training policies, budget-boundary discards, other abort reasons, completed/interrupted sessions, saved interruption step counts, and the latest decision count's remainder within a rollout.

A nonzero remainder such as 80,952 modulo 1,024 = 56 can occur after interruption and resumption during a partial rollout. The session evidence should confirm the actual history; the total does not imply that every collected transition was used for optimization. Training-episode counts show whether successful examples occurred, not held-out playing strength.

If a gain remains unresolved after this predeclared sample, do not select a larger sample merely because the first result was inconclusive. Assess the observed effect, rare-clear counts and available training successes before choosing the next learning experiment. No clear-rate equivalence or noninferiority claim is made. Wins remain emulator Act 1 clears; native transfer is unverified.

## Local verification

The checker played 2,000 simulated runs here with a separate short-trial checkpoint. The two policies each cleared Act 1 once in 1,000 runs, and the checker detected a small floor regression. That is an operational test on different models, not an estimate of the user's model. The actual pilot preceded the final guards for missing source evidence and malformed training accounting, and the exact rare-clear test; focused tests verify those refinements as well as frozen policies, paired seeds, cached-evidence integrity, unchanged training artifacts, and omission of all training/BC/promotion calls.

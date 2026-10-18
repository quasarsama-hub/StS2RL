# Exploratory Act 1 PPO continuation

The user's imitation result improved mean floor from 5.50 to 8.52 across 100 matched seeds: paired gain +3.02, 95% interval [2.01, 3.97]. Act 1 clears changed from 1 to 0. The clear-rate difference was -0.01, with a bootstrap interval [-0.03, 0]. This supports a progression gain, not preserved or improved winning ability. The original runner performed zero PPO decisions because it used the champion-promotion rule as a training-entry gate.

This additive launcher separates exploratory training from promotion. It can try one fixed PPO budget from the frozen cloned policy when the source has at least 100 clean matched episodes, a positive lower floor-gain interval, and no wholly negative clear-rate interval. That rule does not establish safety or noninferiority. The existing champion-promotion rule is unchanged, and this launcher never updates a champion.

```powershell
.\.venv\Scripts\python.exe .\scripts\act1_rl_trial.py --target-steps 80000 --episodes 100
```

Default source: `experiments/ironclad-a0-act1`. Separate destination: `experiments/ironclad-a0-act1-rl-trial`. Keep the same command to resume. Source, target budget, and evaluation schedule are fixed at trial creation. PPO rollout rounding can slightly exceed the target; repeating the command trains only the remaining total. No budget extension is automatic.

The source experiment, its pre-imitation model, its cloned model, and its raw evaluation evidence stay intact. Trial initialization strictly copies both policies into fresh optimizer snapshots under the same Act 1 observation/action schema and a separate experiment identity. It does not collect demonstrations or repeat imitation. Training cursors continue beyond the source's durable frontier. Validation starts at index 500 by default and must be disjoint from the source's imitation evaluations.

The trial evaluates both frozen references before PPO, then evaluates the final policy on the same fresh seeds. Read `experiments/ironclad-a0-act1-rl-trial/trial-progress.json`:

- `additional_rl_comparison`: final PPO policy versus the frozen cloned initializer; this isolates RL gains.
- `incumbent_comparison`: final policy versus the model before imitation; this measures overall performance under the same Act 1 objective.
- `new_rl_steps`: PPO decisions in this trial, excluding previous training and imitation.
- `champion_updated`: always false.

More Act 1 clears are the primary goal. If the trial again produces no positive training outcomes and no held-out improvement, repeating an unchanged budget is not justified. A nearer combat curriculum or a different reward design would need its own measured experiment. Wins here are emulator Act 1 clears, not full-game wins; native-game transfer remains unverified.

## Verification and limits

A full 80,896-decision pilot here on a different cloned checkpoint showed 6.87 to 6.97 mean floors, paired gain +0.10 with interval [-0.75, 0.95], and zero Act 1 clears in both 100-run validation samples. Its 885 completed training episodes all ended in death. The pilot ran cleanly but did not establish RL progress. This is not the user's stronger 8.52-floor checkpoint, and it is not a prediction that their trial will succeed.

The pilot used an isolated copy of the local source with its frozen clone restored as the initial policy; the original local experiment was untouched and the source copy's durable cursor frontier was retained. A subsequent guard rejects overlapping evaluation schedules. The final release bytes were exercised in a separate 1,024-decision end-to-end run with 100 matched validation seeds starting at index 600. All 26 focused tests passed. They cover the reported 1-to-0 clear case, refusal to claim promotion or noninferiority, raw evidence checks, source preservation, fresh optimizers, fixed budgets, disjoint validation seeds, cache recovery, both frozen references, and omission of all imitation and champion-update calls.

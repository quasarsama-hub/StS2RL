# Bounded Act 1 learning experiment

The user's latest matched combat audit found 5.38 mean floors for the 160,768-step policy, 5.00 for the 80,896-step policy, 6.14 for a simple legal-card controller, and 8.94 for the public rule teacher. The paired gain from the last PPO batch was +0.38 floors, with a 95% interval of [-0.19, 0.96]. The teacher cleared Act 1 on 4 of 100 seeds. All 1,390 completed training episodes were deaths; none supplied a full-run success reward. These observations justify one bounded change of learning setup, not a claim that more full-run training will work.

Run from the project directory:

```powershell
.\.venv\Scripts\python.exe .\scripts\learn_act1.py --source experiments/ironclad-a0-add-only --experiment experiments/ironclad-a0-act1 --target-steps 80000 --episodes 100
```

This creates a separate experiment and imports the source policy weights. It collects 256 complete teacher episodes on a reserved block of **training** seeds beyond the source's durable seed frontier. Supervised labels contain only public observations, legal action masks, and teacher actions during ordinary combat. Pending choices and noncombat actions drive the teacher's collection trajectories but are not cloned. Losing teacher episodes can still supply combat demonstrations. The learner's shared network can change noncombat behavior as a result of combat training; gameplay evaluation measures that effect.

Cloning uses 12 supervised epochs, a separate Adam optimizer, and a holdout group of training-seed episodes for action-prediction diagnostics. Its updates do not count as PPO decisions. The cloned checkpoint starts PPO with a fresh optimizer.

The objective then becomes **clear Act 1**, with the existing +1 success / -1 death rewards and no shaping. The observation/action schema and finite card-selection interface remain the same. The PPO learning rate is 0.0001. The source experiment, checkpoint, and seed cursors stay intact.

The script evaluates the imported policy, cloned policy, and teacher on the same 100 fresh validation seeds starting at index 400. If cloning does not establish gameplay improvement, if the teacher produces no Act 1 successes during collection, or if validation contains incomplete runs, it stops before PPO. If those checks pass, it runs up to 80,000 new PPO decisions and evaluates again on the same seeds. PPO rollout rounding can exceed the requested target slightly. Repeating the command resumes toward the same total rather than adding another 80,000 decisions.

Read `experiments/ironclad-a0-act1/act1-progress.json`:

- `before_imitation`: imported policy under the Act 1 objective.
- `after_imitation_before_rl`: supervised learning only.
- `after_rl`: policy after the bounded repeated-run PPO phase.
- `imitation_comparison`: cloning versus imported weights.
- `additional_rl_comparison`: PPO versus the saved cloned checkpoint.
- `additional_rl_establishes_more_act1_clears`: whether the paired confidence interval supports a positive Act 1 clear-rate gain from PPO.

Wins here mean Act 1 clears in the Python emulator. They do not mean full-game wins or verified play in the installed game. The teacher is a measured heuristic, not an expert or a guarantee of success. No additional training budget is allocated automatically.

Initialization publishes the new experiment only after both step-zero checkpoints and the demonstration dataset are complete. If initialization is cancelled, it removes its staging directory; retrying may regenerate that same deterministic demonstration block. Once published, cached evaluation evidence, dataset bytes, checkpoints, recipe, and source hashes are checked before resuming.

## Local verification

The complete workflow was exercised here with a different source checkpoint from the user's Windows model, the default 256 demonstration episodes / 12 cloning epochs, 100 matched validation episodes, and a shortened 2,048-decision PPO phase. The teacher supplied 24,624 ordinary-combat labels and cleared Act 1 in 13 of its 256 training episodes. Imitation raised the learner's mean floor from 4.81 to 7.34; the paired gain was +2.53, with a 95% interval of [1.76, 3.27]. Neither learner checkpoint cleared Act 1 in that validation sample. The short PPO phase then averaged 7.20 floors; its incremental gain was -0.14, with interval [-0.63, 0.34]. That does not establish additional RL progress.

The focused compatibility, imitation, and workflow checks passed 110 tests. This verifies a functioning experiment and an imitation gain on the local model. It does not predict the outcome of an 80,000-decision trial on the user's checkpoint, prove that Act 1 PPO will improve, or verify native-game transfer.

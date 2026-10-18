# A controlled card-choice representation trial

This trial responds to the frozen behavior audit of the 645,176-decision learner. In its 100 replayed runs it skipped 368 of 551 nonempty normal card rewards (66.8%) and chose healing at all 169 recorded rest sites. It reached 38 bosses and cleared five; 22 boss entries had at least 90% HP and 17 of those runs died. Some boss decks contained no nonbasic attacks after repeatedly skipping attack offers. These are descriptive observations on an already-used evaluation subset, not a fresh strength claim or proof that every skip or heal was a mistake.

The existing encoder already receives permanent deck counts, upgrades, enemies and map information. Its hand cards have learned identity embeddings, but offered-card identity is a normalized numeric index in the flat choice table. The new extractor connects those visible choice identities to the existing hand embeddings and a bank of static public card properties. This tests a concrete representation limitation rather than simply giving the same setup more time.

## Run on Windows

Extract the additive update into the existing project. From any PowerShell directory:

```powershell
powershell -ExecutionPolicy Bypass -File "C:\Users\Junhyuk\Downloads\spire2-rl-starter\spire2-rl\scripts\train_choice_features.ps1"
```

The default source is the completed `experiments/ironclad-a0-learning-sessions/session-0001`. The new sibling folder is `session-choice-features-trial`. The source files, model and original learner remain intact. The launcher accepts `-Session`, `-Steps` and `-Episodes`; once initialized those choices are fixed. Running the same command resumes unfinished work. A completed rerun verifies cached evidence without training or rewriting it. Do not run another learning session concurrently.

## What learns

Both branches inherit the same policy, critic, Adam moments and step counters, and the same next training-seed cursors. The control continues the original PPO. The treatment adds an initially zero residual to the original feature extractor. Every original parameter and optimizer state is copied by name; the original Adam options are retained and new parameters begin with empty optimizer state. Initial masked action probabilities and critic values must match exactly. A saved treatment uses an importable Python class, so later processes can reload it and resume its optimizer.

The treatment decodes only card IDs already present in the public choice table. It reuses the hand-card embedding and adds public type, printed cost, base damage/block, target type, exhaust/retain/ethereal and explicit draw fields, with separate base/upgraded rows. A small shared slot network feeds a zero-initialized residual projection. The static bank reads the pinned emulator's public card definitions; it does not inspect draw order or simulate future outcomes. The bank builder restores engine card-instance allocation and random-number states.

All 581 real card definitions are represented. The generic placeholder has no definition and contributes no residual. Conditional effects are not fully encoded. Choices exposing only a bundle hash, or missing an upgrade flag, do not gain information that the observation does not contain. The intervention adds capacity as well as card semantics; a gain would establish the usefulness of this package, not isolate the contribution of each descriptor.

PPO rewards, discount, GAE, learning rate, action masks, observations, four-environment rollout, game rules and training seed scheme remain the same. There is no teacher training, forced card pickup, extra healing rule, search or reward shaping. The learner can still skip cards or heal when it judges those actions useful.

## Fixed comparison and cost

Each branch receives 240,000 additional decisions by default, in recoverable chunks of at most 80,000. Actual decisions can exceed the target by less than one 1,024-decision rollout. From the uploaded source counter of 645,176, the expected default final counter is 885,816 per branch: 240,640 new decisions each, or 481,280 in total. The user model itself remains on Windows; cloud verification uses separate short local policies.

Both training allowances finish before either final evaluation is inspected. There is one matched comparison of the frozen treatment and continued control on 1,000 fresh validation seeds. There is no repeated starting evaluation, intermediate checkpoint selection, automatic allowance extension or promotion. The starting index is chosen from the full available lineage and existing reservations (9,800 for the supplied history if no later block exists). The immutable `session-protocol.json` is reserved before training inside the existing learning series, so future ordinary sessions also skip this block.

The primary outcome is Act 1 clears within the unchanged 4,000-decision cap, across all attempted seeds. Timeouts count as failed clears; simulator errors or exact source/timeout replay mismatches pause the trial. Paired confidence intervals and the exact paired McNemar test compare treatment with the equally trained control. Floor reached is descriptive. A single trial describes these trained policies; it is not a replicated guarantee about the learning method.

On the user's previous CPU throughput, the two training allowances alone would take roughly 18 minutes; evaluation and initialization add time. The actual duration depends on hardware and learned episode length. Both branches run in sequence to keep CPU settings and resource contention controlled. This spends more compute than one continuation, but answers whether the representation change helps beyond ordinary continued training.

## Outputs

The launcher prints the final report or a pause reason. Under the new trial folder:

- `report.txt` and `report.json`: the clear-rate comparison and actual decision counts.
- `diagnostics.zip`: bounded metadata, raw final outcomes, replay evidence and per-update PPO telemetry; no model or game distribution.
- `branches/control` and `branches/treatment`: separate models, journals and checkpoints.
- `branches/*/telemetry/chunk-*/progress.csv` and `progress.json`: entropy, approximate KL, clip fraction, policy/value losses and explained variance, including the final PPO update.

If the treatment does not improve clear rate, retain the report and both branches. This trial does not silently promote a model or try another budget. Everything remains emulator Act 1; transfer to the installed Slay the Spire 2 game is unverified.

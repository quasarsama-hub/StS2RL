# Verify the potion pickup contract before continuing training

The user's 1,000-pair frozen-policy evaluation found a paired mean floor gain of
0.879, with a 95% interval of [0.553, 1.205], across 998 completed pairs. The
remaining two candidate runs aborted. The checker deliberately requires all
pairs to complete before setting its positive-gain flags, so the false floor
flag does not erase the progression signal in completed runs. Act 1 clears were
12 versus 6; the exact paired clear test gave p=0.210, so a higher clear rate
remains unresolved. The current learner still rarely clears Act 1.

The training journal includes three wins and one interrupted session. A single
exception causes the training runner to abort every active environment with the
same reason. Four identical abort entries do not imply four independent potion
failures. The saved interruption at 73,784 decisions followed by seven full
1,024-step rollouts explains the final 80,952 count. Recorded wins locate reward
examples; they do not prove every winning transition reached an optimizer update.

The pinned simulator has a reproducible action contract defect. It advertises
`pick_potion` whenever a potion reward is offered, even when all inventory slots
are full. Its reward handler cannot add the potion, returns `success=False`, and
advances the reward chain. The guarded adapter correctly reports this rejection
as an infrastructure error. It must not become a fabricated gameplay death or
an automatically substituted skip action.

`PotionCapacityMask` checks only the current visible inventory and capacity. It
masks the original pickup slot when no empty slot exists, while leaving its
choice descriptor and every action slot in place. A pickup with available room,
combat actions, other choices, observations and rewards stay unchanged. The
policy explicitly chooses its next legal action. The wrapper has a separate
backend identity and propagates other simulator failures.

The defect was reproduced using a manually staged real-engine reward state.
That verifies the engine contract, not the user's exact trajectory. The
following command determines whether the two actual evaluation failures match:

```powershell
.\.venv\Scripts\python.exe .\scripts\check_potion_failures.py
```

It reads the frozen strength report automatically and replays only its aborted
or timed-out rows. With two candidate aborts, that is four simulated runs: each
seed once through the original interface and once through the capacity mask,
using the same saved policy. It logs public inventory **before** the rejected
action can mutate the reward state. The complete traces and an independent
summary go under `experiments/ironclad-a0-act1-rl-trial/diagnostics/potion-<time>/`.

`repair_verified` requires every original failure to reproduce exactly, exhibit
an advertised potion pickup with every capacity slot occupied, and complete as
a genuine win or death under the repair with a recorded masked pickup decision.
An unrelated error, a nonreproduced failure, a repaired timeout or abort, or no
observed repair intervention leaves the flag false. An empty failure list also
cannot verify a repair.

The command does no training and does not overwrite the original strength
report, checkpoints, seed cursors or champion. A repaired outcome on these known
seeds is interface regression evidence, not additional learning or a new
strength comparison. Any subsequent training with this mask must use a new
experiment identity and compare the current frozen learner against its later
checkpoint under the same repaired interface on a predeclared fresh seed set.
Emulator Act 1 performance remains distinct from native-game transfer.

Exit code 0 means the specific repair was verified; 2 means it was not verified;
1 reports a setup/evidence error, and 130 an interruption.

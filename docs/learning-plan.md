# What this starter can establish

The starter trains one persistent Ironclad A0 policy against a pinned Python
emulator. It can establish that experience changes the policy, that training
resumes from saved weights and optimizer state, and that a checkpoint improves
on separate evaluation seeds. None of those results establishes native STS2
playing strength until the native backend passes conformance checks.

The emulator pin is `04a62c35b0ee264f5e4a3b5c6587f7438f1a7e0c` from
[`vvsuby/sts2-rl-agent`](https://github.com/vvsuby/sts2-rl-agent). Its reference
game version is recorded separately from the current installed game. Keep the
upstream checkout clean; corrections belong in the local adapter and its tests.

## Initial experiment

Use all three modeled acts, Neow enabled, Ironclad, and ascension zero. An
act-limited episode is a separate curriculum experiment: a win after one act
must never enter the three-act win-rate table. The single-combat backend is an
integration diagnostic, not evidence of a competent run policy.

The initial CPU settings are deliberately modest:

| Setting | Initial value | Reason |
| --- | --- | --- |
| Algorithm | MaskablePPO | Learns only among the currently legal actions |
| Parallel environments | 4, sequential `DummyVecEnv` | Modest memory and simple failure accounting |
| CPU threads | 2 | Avoids excessive thread overhead for small policies |
| Rollout steps per environment | 256 | 1,024 transitions per update |
| Batch size | 128 | Eight minibatches per rollout |
| Epochs per rollout | 4 | Keeps the initial compute budget bounded |
| Learning rate | 0.0003 | Conservative initial PPO default |
| Discount | 0.999 | Preserves more signal across long runs |
| Entropy coefficient | 0.01 | Maintains exploration among legal options |
| Policy layers | 128, 64 | A small CPU baseline; complexity comes later |
| Feature encoder | Card embedding 16; public features 256 | Treats card IDs as identities and retains visible choices |
| Maximum decisions per episode | 4,000 | Caps a stuck policy without classifying it as death |

These are starting settings, not a proven optimum. Record measured transitions
per second before projecting larger budgets. Four `DummyVecEnv` workers do not
mean four simulator processes run simultaneously.

The default reward is sparse: +1 for a terminal win, -1 for actual death, and no
fabricated death penalty for a time limit. Sparse full-run PPO may make little
progress in the first 100,000 decisions. An optional shaped-reward experiment
provides intermediate feedback but changes the training objective; give it its
own experiment directory and configuration fingerprint.

Upstream shaping includes +0.03 per advanced floor, +0.05 per combat victory,
+0.3 per act transition, and positive-only enemy HP reduction times 0.002.
Because damage to healing or spawning enemies can accumulate without a win,
shaped return cannot be the promotion criterion. Do not silently turn shaping
on when resuming a sparse experiment.

## What counts as progress

1. **Integration check:** reset and replay identical seeded traces; all selected
   actions are legal; errors are recorded as aborted episodes; a healthy time
   limit is recorded as timeout. This is required before a long training job.
2. **First 20,000 decisions:** confirm completed episodes, finite optimization,
   changed parameters, committed checkpoints, and successful resume. This
   establishes an operating learner, not improved playing strength.
3. **Continue the same experiment to 100,000 decisions:** evaluate checkpoints
   on the same validation reset seeds against the initial checkpoint and random
   legal baseline. Inspect paired floor progress, death/win outcomes, timeouts,
   and simulator failures. Do not promise full-run wins at this budget.
4. **When validation improves:** compare against the current champion rather
   than repeatedly beating the random baseline. Use untouched test seeds for a
   final assessment after choosing the checkpoint.

Start with 100 paired validation episodes for a usable measurement. Fewer
episodes are useful for a smoke check but will not reliably resolve small
improvements. Increase the evaluation set when the result is uncertain rather
than repeatedly inspecting a handful of favorable seeds. If all policies have
zero full-run wins, paired floor progress is an early diagnostic; it does not
mean the run objective has been solved.

Training, validation, and test reset seeds are separate. Upstream `reset(seed)`
uses that seed to draw an internal emulator run seed; the logged reset seed is
the reproducible identifier for this experiment, not a native STS2 seed string.
Resume restores weights and optimizer state and starts fresh episodes on new
training seeds. It does not restore a partly played simulator state or promise
bitwise-identical interrupted rollouts.

## Fair teacher option

The upstream `HeuristicBot(use_lookahead=False)` can inform a future
adapter-aware baseline or behavior-cloning warm start. Its scoring reads current hand,
current enemy intents, visible powers/HP, known deck size, and current public
reward/shop/map options. This audit found no draw-order or future-RNG reads in
that mode. Its fixed noncombat action slots do not match this starter's dynamic
public decision slots, so do not call it directly against the starter. A small
diagnostic also found it repeatedly toggling a pending deck selection rather
than completing the choice. Fair information access does not make it a reliable
teacher. Translate its rules into the current action API and evaluate complete
strict runs before using it for training data.

Do not use `use_lookahead=True` for fair teacher data. It deep-copies the actual
run manager and executes candidate actions with the actual hidden draw order
and RNG. That is an oracle search, not a player-information baseline.

Behavior cloning is optional. If introduced, record its teacher version and
dataset seed split, retain a separate untrained baseline, and evaluate the
cloned policy before RL. A stronger initial imitation score is not evidence
that subsequent reinforcement learning improved it.

## Simulator readiness issues to keep visible

The upstream environment silently converts exceptions into gameplay losses
unless `STS2_STRICT=1`; the local adapter must keep strict behavior active and
must never train on those fabricated losses. Strict mode alone does not catch
every semantic problem: upstream card rejection can substitute an end-turn,
and direct potion execution requires manager bookkeeping when combat ends.
The adapter's regression checks must cover both transitions.

The upstream fixed action layout can omit public choices: card bundles and
card-reward sacrifice are not represented by its original reward slots, and
shop slots expose at most nine purchases at once. It also offers a skip slot
for rewards that may be non-skippable. The local adapter addresses these with
ordered public noncombat decision slots and checks rejection explicitly.
Combat retains the fixed upstream layout. A state with more public decisions
than the fixed action-space capacity still raises an error. An apparently
valid mask is not proof of native game parity.

Current policy observations must remain independent of hidden draw order and
future random outcomes. Current map node types and current enemy intents are
visible. Future encounter identity, future rewards, actual RNG state, and
future monster state-machine information must not become policy inputs.

If the first 100,000 decisions show no held-out improvement, inspect episode
outcomes and action traces before increasing scale. Change one measured cause
at a time in a new experiment: observation omissions, legal-action translation,
reward objective, exploration, or a teacher warm start. Code-generation loops
and more training steps alone are not a progress criterion.

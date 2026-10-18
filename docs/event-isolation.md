# Independent event state for repeated-run learning

The fixed fresh cycle 2 comparison did not establish additional learning
strength: 123 to 130 Act 1 clears in 3,000 matched pairs (4.10% to 4.33%),
exact paired p=0.681. The clear-rate gain interval was [-0.733, +1.167]
percentage points. Floor reached changed by -0.106, with an interval including
zero. This does not establish equivalence, a permanent plateau, or a loss of
the earlier measured gains. The next ordinary cycle remains paused.

During the planned behavior audit, a separate simulator ownership defect was
reproduced. The upstream event registry stores mutable event instances shared
by all same-process games. The original trainer uses four `DummyVecEnv` games.
An event's choices, RNG, pending resolver or page progress can therefore be
overwritten by another game. This occurs without an exception being logged.

Two concrete reproductions in the pinned backend:

| Scenario | Original behavior | Isolated behavior |
| --- | --- | --- |
| Seed 42 offers Lost Coffer; seed 43 then resets | Seed 42 selects its displayed option but receives Neow's Torment | Seed 42 receives its displayed Lost Coffer |
| Two seed-44 games decipher Tablet of Truth | Same initial 80 MaxHP becomes 77 in one game and 74 in the other | Both games end at 77 MaxHP |

This is a training integration defect. Its contribution to the observed learning
curve has not been measured. The 3,000-pair check used sequential games, so the
demonstrated interleaving fault does not invalidate that comparison. Pending
choices from an abandoned run can also remain in the original registry;
the repair creates fresh instances per run rather than copying live callbacks.

## Verify the repair with the current frozen policy

After extracting the additive patch into the existing project, run:

```powershell
cd "C:\Users\Junhyuk\Downloads\spire2-rl-starter\spire2-rl"
.\.venv\Scripts\python.exe .\scripts\check_cycle_event_isolation.py --cycle 2 --episodes 100
```

The command first reproduces both original faults and checks their isolated
counterparts. It then plays 100 fresh matched pairs using the exact same saved
323,640-decision policy on the original and isolated backends. This adds no RL
steps. The expected first fresh index is 7,700, following the completed
3,000-pair diagnostic. A previously reserved behavioral audit can move this
block forward; the command checks reservations rather than assuming an index.

The printed stage is `cycle_event_isolation_check`. `status: verified` requires
the original counterexamples, their corrected results, and clean isolated
policy replays. This establishes that narrow repair, not improved playing
strength. The protocol pins the checkpoint, backend identities, sample, seed
block and source hashes. Repeating the same command reads completed evidence;
changed models, schedules or cached statistics are rejected.

The wrapper supplies each run with fresh instances of all 68 registered event
classes. During reset, step and mask calls it scopes the registry under the
existing engine lock and restores the exact prior registry afterward, including
on exceptions. Event selection, game actions, rewards and public spaces remain
delegated to the original engine. The repaired backend has a separate identity;
the old trainer is preserved for reproducing its evidence.

## One bounded trial with corrected event ownership

After repair verification, use the new branch:

```powershell
.\.venv\Scripts\python.exe .\scripts\isolated_act1_cycles.py --cycle 1
```

The default source is
`experiments/ironclad-a0-act1-cycles/cycle-0002`. Corrected cycles are stored
under `experiments/ironclad-a0-act1-event-isolated-cycles`. The initial policy,
value network, PPO Adam state, cumulative counter and seed high-water marks
are retained. The configuration, finite-selection interface, potion mask,
reward objective and 1,024-decision rollout size are retained. Only event
ownership is corrected; no imitation is repeated.

Each corrected cycle freezes its starting policy, evaluates 1,000 fresh matched
seeds before and after one requested 80,000-decision budget, and stores its raw
evidence separately. An uninterrupted first trial is expected to add 80,896
decisions, from 323,640 to 404,536. Repeating the same cycle resumes its remaining
budget or reads its completed result. A later cycle requires clean raw evidence
of further progress; the launcher does not automatically add another budget
or promote a champion.

The first repaired trial is a deliberate test of a corrected environment,
rather than an override of the inconclusive original cycle's gate. Its result
must be judged on the repaired before/after comparison. A repair alone does not
guarantee more wins. If learning still stalls, use behavioral evidence to choose
one measured bottleneck before changing the policy or objective.

## Behavioral evidence

The optional read-only command below audits the original frozen cycle 2
policies and three public-information controller variants:

```powershell
.\.venv\Scripts\python.exe .\scripts\audit_act1_cycle.py --cycle 2 --episodes 100
```

It compares the current policy, its frozen predecessor, a heuristic teacher,
the teacher controlling only ordinary combat, and the teacher controlling all
other decisions. Combat selection prompts belong to the latter category.
Reports include death rooms/enemies, HP entering elites and bosses, rest/path/
card choices, unused energy/cards, potions left at death and bounded public-state
examples. These are descriptive diagnostics: leaving a card or potion unused
can be correct, and changing a controller can also change its path and fights.
This audit does not authorize another training budget or claim a new strength
gain from its small sample.

## Verification scope

Targeted tests cover the real-engine Neow and Tablet faults, fresh per-run
callbacks, registry restoration after normal/error/nested calls, unchanged
sequential gameplay, cached diagnostic evidence, model identity and fixed
budgets. Separate local smoke runs exercise actual frozen PPO policy replay,
the five diagnostic controllers, and corrected-policy migration/training.
These use a separate short local policy, not the user's checkpoint.

All 71 targeted tests passed. The actual corrected PPO pilot restored policy,
value and Adam tensors exactly, then trained 1,024 decisions with four distinct
Neow instances. Its 50 before/after gameplay runs completed without simulator
aborts or timeouts; completed-result replay left all files unchanged. These
small checks verify operation and do not estimate learning strength.

See [release verification](event-isolation-verification.json) for measured
counts, exact file hashes and runtime limits. The repair covers event-instance
ownership; other simulator globals and native-game transfer remain unverified.

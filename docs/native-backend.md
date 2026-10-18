# Native backend boundary

`spire2_rl.native.NativeRunner` adapts the inspected `wuhao21/sts2-cli`
JSON-lines protocol. It does not implement a native Gym training environment.
It can probe the installed runner, capture raw decisions, and deliver explicit
actions. The executable requires the user's own local game assemblies and the
upstream setup/build; no game code or assets are included here.

After completing upstream setup, prefer a direct built DLL command:

```python
from spire2_rl.native import NativeRunner, NativeAbort

command = ["dotnet", "/path/to/sts2-cli/src/Sts2Headless/bin/Debug/net9.0/Sts2Headless.dll"]
try:
    with NativeRunner(command, cwd="/path/to/sts2-cli") as game:
        state = game.reset("training-000001", character="Ironclad", ascension=0)
        # Do not skip the opening event: select from this returned decision.
        print(state["decision"], state.get("options"))
        revision = game.revision
        # The controller must derive option_index from the actual current options.
        state = game.act("choose_option", {"option_index": 0}, expected_revision=revision)
except NativeAbort as abort:
    # Quarantine the episode with infrastructure metadata; never label a loss.
    print(abort.as_dict())
```

Use the actual assembly output path reported by the upstream build. `dotnet run
--no-build --project /path/to/Sts2Headless.csproj` is also accepted as an argument
vector. POSIX workers use isolated process groups so aborts kill launcher children.
Windows tree cleanup uses bounded `taskkill /T /F`; its behavior still needs testing
on a user's native installation. Command strings are never run through a shell.

The wire starts with `{"type":"ready","version":"0.2.0"}`. This is the
**CLI protocol version**, not evidence of the installed game's build. Reset sends
`start_run` with character, string seed, ascension and language. Every reset cold
starts a process, avoiding repeated-run singleton/event-handler carryover. Actions
use `{"cmd":"action","action":"...","args":{...}}` and return an ordinary
`type=decision` payload. Allowed reported phases are `event_choice`, `map_select`,
`combat_play`, `card_reward`, `card_select`, `bundle_select`, `rest_site`, `shop`,
and `game_over`. Game-over has an explicit `victory` Boolean; HP is not a win flag.

The upstream action names include `play_card`, `end_turn`, `select_map_node`,
`choose_option`, `select_card_reward`, `skip_card_reward`, `select_cards`,
`skip_select`, `select_bundle`, `use_potion`, `discard_potion`, `buy_card`,
`buy_relic`, `buy_potion`, `remove_card`, `leave_room`, and `proceed`. Selection
indices are a **comma-separated string**, e.g. `{"indices":"0,2"}`, because
upstream command parsing does not accept a native array for this argument. Card,
enemy, potion, reward and option indices must be re-derived from the latest raw
payload; they are not stable IDs across actions.

The transport has one in-flight command and bounded stdin writes, stdout reads,
response size, diagnostics, startup and shutdown. It rejects a stale local
revision before delivery. A successful response reconciles the next reported
state and records `last_transition.changed`. Its hash includes the whole reported
payload; hash equality does not prove equality of hidden engine state. A higher
controller must decide how to handle legal no-ops and repeated state/action cycles.

There is **no request ID, idempotency key, action receipt, or general observe
command** in this native CLI. A timeout or broken pipe after sending can mean the
action already executed. The adapter kills that worker, marks delivery `unknown`,
and never repeats the action. Engine error responses, unsupported decisions and
malformed output also invalidate the episode; none become training defeats.
Close sends native `quit`, checks the acknowledgement and exit, and kills the
worker on failure. The adapter never requests a save path; any native profile/save side effects still
require inspection on the user's pinned installation.

Current upstream scope has important policy limits: treasure rooms automatically
choose the first offered relic and return to the map. Those choices are not
available to a learner through this adapter. Its native save support checkpoints
map/pre-room state, not arbitrary mid-combat state. Native scheduler patches,
Godot stubs and test lifecycle still require parity audits on a pinned game build.

Mock-process tests here verify transport behavior only. No game assemblies or
native runtime are available in this execution environment, so native compatibility,
action legality, deterministic gameplay and real-game throughput remain untested.

Protocol sources inspected: upstream [Program.cs](https://github.com/wuhao21/sts2-cli/blob/main/src/Sts2Headless/Program.cs),
[RunSimulator.cs](https://github.com/wuhao21/sts2-cli/blob/main/src/Sts2Headless/RunSimulator.cs),
and the [v0.111.0 compatibility report](https://github.com/wuhao21/sts2-cli/blob/main/docs/compatibility-v0.111.0.md).

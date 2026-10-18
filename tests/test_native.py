"""Protocol tests use a real child process implementing the inspected CLI wire.

They verify pipe deadlines and single-delivery behavior without game assemblies;
they do not establish compatibility or parity with the real game.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

from spire2_rl.native import AbortCategory, NativeAbort, NativeRunner, decision_hash


FAKE_CLI = r'''
import json, os, subprocess, sys, time
mode = os.environ.get("FAKE_MODE", "normal")
path = os.environ["FAKE_COMMAND_LOG"]
def emit(value):
    print(json.dumps(value), flush=True)
def state(phase="event_choice", step=0, **extra):
    return dict(type="decision", decision=phase, player={"hp": 80},
                context={"seed": "test", "step": step}, options=[{"index": 0}], **extra)
if mode == "startup_hang":
    time.sleep(20)
if mode == "bad_ready":
    emit({"type": "ready"})
elif mode == "wrong_version":
    emit({"type": "ready", "version": "9.9.9"})
else:
    print("A harmless dotnet startup banner", flush=True)
    emit({"type": "ready", "version": "0.2.0"})
step = 0
for line in sys.stdin:
    cmd = json.loads(line)
    with open(path, "a") as log:
        log.write(json.dumps(cmd) + "\n")
    if cmd["cmd"] == "quit":
        if mode == "quit_hang":
            time.sleep(20)
        if mode == "bad_quit":
            emit({"type": "quit_result", "success": False})
        else:
            emit({"type": "quit_result", "success": True, "save": None})
        break
    if cmd["cmd"] == "start_run":
        if mode == "launcher_child":
            child = "import sys,time; f=open(sys.argv[1], 'a'); exec('while True:\\n f.write(\\\"tick\\\\n\\\"); f.flush(); time.sleep(0.02)')"
            subprocess.Popen([sys.executable, "-u", "-c", child, path + ".ticks"],
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if mode == "stderr_flood":
            for i in range(300):
                print("diagnostic " + "x" * 1000, file=sys.stderr, flush=True)
        if mode == "unsupported":
            emit(state("unknown"))
        elif mode == "oversize":
            emit(state(description="x" * 5000))
        elif mode == "bad_payload":
            emit({"type": "decision", "decision": "event_choice"})
        elif mode == "nonfinite":
            print('{"type":"decision","decision":"event_choice","energy":NaN}', flush=True)
        else:
            emit(state())
        if mode == "block_input":
            time.sleep(20)
        continue
    if cmd["cmd"] == "action":
        if mode in ("action_hang", "launcher_child"):
            time.sleep(20)
        if mode == "action_eof":
            sys.exit(4)
        if mode == "malformed":
            print('{"type":broken}', flush=True)
        elif mode == "engine_error":
            emit({"type": "error", "message": "action failed after partial mutation", "stack_trace": "native trace"})
        elif mode == "game_over":
            emit(state("game_over", victory=False))
        elif mode == "bad_game_over":
            emit(state("game_over"))
        elif mode == "noop":
            emit(state())
        else:
            step += 1
            emit(state("combat_play", step, energy=3-step, hand=[{"index": 0, "id": "CARD.STRIKE"}]))
'''


@pytest.fixture
def factory(tmp_path: Path):
    runners: list[NativeRunner] = []
    log = tmp_path / "commands.jsonl"

    def make(mode: str = "normal", **kwargs) -> NativeRunner:
        options = {"timeout_s": 0.5, "startup_timeout_s": 1.0, "close_timeout_s": 0.3}
        options.update(kwargs)
        runner = NativeRunner(
            [sys.executable, "-u", "-c", FAKE_CLI],
            env={"FAKE_MODE": mode, "FAKE_COMMAND_LOG": str(log)},
            **options,
        )
        runners.append(runner)
        return runner

    yield make, log
    for runner in runners:
        try:
            runner.close()
        except NativeAbort:
            pass  # A deliberate bad/hanging shutdown has already killed its worker.


def commands(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_actual_start_action_and_quit_wire_without_autoplay(factory):
    make, log = factory
    runner = make()
    with runner:
        initial = runner.reset(123, character="Silent", ascension=2, lang="en")
        assert initial["decision"] == "event_choice"
        assert commands(log) == [{"cmd": "start_run", "character": "Silent", "seed": "123", "ascension": 2, "lang": "en"}]
        state = runner.act("choose_option", {"option_index": 0}, expected_revision=runner.revision)
        assert state["decision"] == "combat_play"
        assert runner.protocol_version == "0.2.0"
        assert runner.last_transition is not None and runner.last_transition.changed
    assert commands(log)[-2:] == [{"cmd": "action", "action": "choose_option", "args": {"option_index": 0}}, {"cmd": "quit"}]
    assert runner.pid is None and runner.current is None


def test_reset_cold_starts_new_worker_and_revision_never_reuses(factory):
    make, log = factory
    runner = make()
    runner.reset("first")
    first_pid, first_revision = runner.pid, runner.revision
    runner.act("choose_option", {"option_index": 0})
    runner.reset("second")
    assert runner.pid != first_pid
    assert runner.revision > first_revision
    assert [c["cmd"] for c in commands(log)] == ["start_run", "action", "quit", "start_run"]
    assert runner.last_transition is None


def test_stale_action_is_rejected_before_delivery_without_poisoning_worker(factory):
    make, log = factory
    runner = make()
    runner.reset("test")
    with pytest.raises(NativeAbort) as caught:
        runner.act("choose_option", {"option_index": 0}, expected_revision=runner.revision - 1)
    assert caught.value.category == AbortCategory.STALE_ACTION
    assert caught.value.delivery == "not_sent"
    assert len(commands(log)) == 1
    assert runner.pid is not None
    runner.act("choose_option", {"option_index": 0}, expected_revision=runner.revision)


def test_timeout_aborts_unknown_delivery_and_never_retries(factory):
    make, log = factory
    runner = make("action_hang", timeout_s=0.1)
    runner.reset("test")
    started = time.monotonic()
    with pytest.raises(NativeAbort) as caught:
        runner.act("choose_option", {"option_index": 0})
    assert time.monotonic() - started < 1.0
    assert caught.value.category == AbortCategory.TRANSPORT_TIMEOUT
    assert caught.value.delivery == "unknown"
    assert runner.pid is None and runner.current is None
    assert [c["cmd"] for c in commands(log)] == ["start_run", "action"]
    with pytest.raises(NativeAbort) as second:
        runner.act("choose_option", {"option_index": 0})
    assert second.value.category == AbortCategory.NO_ACTIVE_RUN


def test_stdin_write_is_also_bounded(factory):
    make, log = factory
    runner = make("block_input", timeout_s=0.1, max_command_bytes=1024 * 1024)
    runner.reset("test")
    started = time.monotonic()
    with pytest.raises(NativeAbort) as caught:
        runner.act("choose_option", {"padding": "x" * 500000})
    assert time.monotonic() - started < 1.0
    assert caught.value.category == AbortCategory.TRANSPORT_TIMEOUT
    assert caught.value.delivery == "unknown"
    assert runner.pid is None
    assert [c["cmd"] for c in commands(log)] == ["start_run"]


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group cleanup; Windows tree cleanup needs native-host verification")
def test_abort_kills_launcher_children_not_only_direct_worker(factory):
    make, log = factory
    runner = make("launcher_child", timeout_s=0.1)
    runner.reset("test")
    ticks = Path(str(log) + ".ticks")
    deadline = time.monotonic() + 1.0
    while not ticks.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert ticks.exists(), "child process must start before testing containment"
    with pytest.raises(NativeAbort):
        runner.act("end_turn")
    after_abort = ticks.read_text()
    time.sleep(0.1)
    assert ticks.read_text() == after_abort
    assert runner.pid is None


@pytest.mark.parametrize(
    "mode,category,delivery",
    [
        ("action_eof", AbortCategory.PROCESS_EXIT, "unknown"),
        ("malformed", AbortCategory.PROTOCOL_ERROR, "unknown"),
        ("engine_error", AbortCategory.ENGINE_ERROR, "response_received"),
        ("bad_game_over", AbortCategory.PROTOCOL_ERROR, "response_received"),
    ],
)
def test_action_failures_are_infrastructure_aborts_not_losses(factory, mode, category, delivery):
    make, log = factory
    runner = make(mode)
    runner.reset("test")
    with pytest.raises(NativeAbort) as caught:
        runner.act("choose_option", {"option_index": 0})
    assert caught.value.category == category
    assert caught.value.delivery == delivery
    assert runner.current is None and runner.pid is None
    assert len(commands(log)) == 2
    if mode == "engine_error":
        assert caught.value.payload["stack_trace"] == "native trace"
        assert caught.value.as_dict()["category"] == "engine_error"


@pytest.mark.parametrize(
    "mode,category",
    [
        ("bad_ready", AbortCategory.PROTOCOL_ERROR),
        ("wrong_version", AbortCategory.PROTOCOL_ERROR),
        ("unsupported", AbortCategory.UNSUPPORTED_STATE),
        ("bad_payload", AbortCategory.PROTOCOL_ERROR),
        ("nonfinite", AbortCategory.PROTOCOL_ERROR),
    ],
)
def test_ready_and_decision_validation(factory, mode, category):
    make, _ = factory
    runner = make(mode)
    with pytest.raises(NativeAbort) as caught:
        runner.reset("test")
    assert caught.value.category == category
    assert runner.pid is None and runner.current is None


def test_startup_timeout(factory):
    make, log = factory
    runner = make("startup_hang", startup_timeout_s=0.1)
    with pytest.raises(NativeAbort) as caught:
        runner.reset("test")
    assert caught.value.category == AbortCategory.STARTUP_TIMEOUT
    assert caught.value.delivery == "not_sent"
    assert runner.pid is None and commands(log) == []


def test_oversized_response_cannot_fill_unbounded_memory(factory):
    make, _ = factory
    runner = make("oversize", max_line_bytes=1024)
    with pytest.raises(NativeAbort) as caught:
        runner.reset("test")
    assert caught.value.category == AbortCategory.PROTOCOL_ERROR
    assert "byte limit" in str(caught.value)
    assert runner.pid is None


def test_stderr_flood_is_drained_and_diagnostics_are_bounded(factory):
    make, _ = factory
    runner = make("stderr_flood")
    assert runner.reset("test")["decision"] == "event_choice"
    assert len(runner.diagnostics) <= 64
    assert any(line.startswith("stderr: diagnostic") for line in runner.diagnostics)


def test_noop_reconciliation_preserves_state_without_faking_progress(factory):
    make, _ = factory
    runner = make("noop")
    initial = runner.reset("test")
    revision = runner.revision
    result = runner.act("end_turn")
    assert result == initial
    assert runner.revision == revision + 1
    assert runner.last_transition is not None and not runner.last_transition.changed


def test_current_and_returned_payloads_do_not_mutate_transport_state(factory):
    make, _ = factory
    runner = make()
    state = runner.reset("test")
    state["player"]["hp"] = -900
    view = runner.current
    assert view["player"]["hp"] == 80
    view["player"]["hp"] = 0
    assert runner.current["player"]["hp"] == 80


def test_gameover_uses_explicit_victory_and_blocks_further_actions(factory):
    make, log = factory
    runner = make("game_over")
    runner.reset("test")
    terminal = runner.act("choose_option", {"option_index": 0})
    assert terminal["decision"] == "game_over"
    assert terminal["victory"] is False
    assert terminal["player"]["hp"] == 80  # Never infer victory from HP.
    with pytest.raises(NativeAbort) as caught:
        runner.act("end_turn")
    assert caught.value.category == AbortCategory.NO_ACTIVE_RUN
    assert len(commands(log)) == 2


@pytest.mark.parametrize("mode", ["bad_quit", "quit_hang"])
def test_explicit_close_reaps_worker_even_when_quit_fails(factory, mode):
    make, _ = factory
    runner = make(mode)
    runner.reset("test")
    with pytest.raises(NativeAbort) as caught:
        runner.close()
    assert caught.value.category == AbortCategory.SHUTDOWN_FAILURE
    assert runner.pid is None and runner.current is None
    runner.close()  # Already closed is idempotent.


def test_context_exit_does_not_mask_an_existing_exception(factory):
    make, _ = factory
    runner = make("quit_hang")
    with pytest.raises(ValueError, match="original problem"):
        with runner:
            runner.reset("test")
            raise ValueError("original problem")
    assert runner.pid is None


def test_argument_limits_and_actual_selection_string_wire(factory):
    make, log = factory
    runner = make(max_command_bytes=256)
    runner.reset("test")
    with pytest.raises(ValueError):
        runner.act("select_cards", {"indices": [0, 1]})
    with pytest.raises(ValueError):
        runner.act("select_cards", {"indices": "x" * 400})
    assert len(commands(log)) == 1
    runner.act("select_cards", {"indices": "0,1"})
    assert commands(log)[-1] == {"cmd": "action", "action": "select_cards", "args": {"indices": "0,1"}}


def test_missing_executable_is_an_explicit_startup_failure(tmp_path):
    runner = NativeRunner([str(tmp_path / "missing-dotnet")])
    with pytest.raises(NativeAbort) as caught:
        runner.reset("test")
    assert caught.value.category == AbortCategory.STARTUP_FAILURE
    assert runner.pid is None


def test_decision_hash_includes_same_count_card_changes():
    left = {"hand": [{"id": "STRIKE"}], "energy": 3}
    right = {"hand": [{"id": "DEFEND"}], "energy": 3}
    assert decision_hash(left) != decision_hash(right)
    assert decision_hash(left) == decision_hash({"energy": 3, "hand": [{"id": "STRIKE"}]})


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_timeout_configuration_must_be_finite_and_positive(value):
    with pytest.raises(ValueError):
        NativeRunner([sys.executable], timeout_s=value)

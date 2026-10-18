"""Frozen session audits preserve learners and compare complete decision phases.

Controlled games exercise persisted protocols/checkpoint bytes and real Torch
policy/value/Adam state. A separate release runtime check covers actual games.
"""
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import inspect
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest
import torch

from spire2_rl import runner
from spire2_rl.experiment import atomic_json, read_json, sha256
from spire2_rl.potion_capacity import PotionCapacityMask
from spire2_rl.study import evaluation_seeds

from test_act1_continuation import load_script
from test_act1_cycle_audit import audit_row
from test_act1_trial import bytes_in
from test_isolated_act1_cycles import PreservedPolicy
from test_learning_session import sessions, setup as learning_setup, tensors_equal


@pytest.fixture(scope="module")
def session_auditor():
    return load_script("audit_learning_session.py")


@pytest.fixture
def audit_setup(session_auditor, learning_setup, monkeypatch):
    """Produce a coherent source through the existing combined-session APIs."""
    audit, setup = session_auditor, learning_setup
    setup.module.progress(setup.source, setup.series, setup.upstream,
                          session=1, steps=1500, chunk_steps=700, episodes=1000)
    source = setup.series / "session-0001"
    # Retain real source/protocol validation while installing this fixture's
    # controlled public environment and serialized policy implementation.
    monkeypatch.setitem(audit.sessions, "load_source", setup.module.load_source)
    monkeypatch.setitem(audit.sessions, "factory", setup.module.factory)
    calls = {"played": [], "loaded": [], "closed": 0, "outcomes": {},
             "observer_errors": {}, "mutate": None, "interrupt": None,
             "bad_counter": False, "invalid_rows": None}
    training_calls = list(setup.calls["training"])

    class Policy:
        @classmethod
        def load(cls, path, **kwargs):
            assert kwargs.get("device") == "cpu"
            calls["loaded"].append(Path(path))
            model = PreservedPolicy.load(path)
            model.policy.set_training_mode = model.policy.train
            if calls["bad_counter"]:
                model.num_timesteps += 1
            return model

    class Teacher:
        def __init__(self):
            self.bot = SimpleNamespace(use_lookahead=False)
            self.decisions, self.phases = Counter(), Counter()
        def select(self, env):
            return 1

    def play(env, seeds, *, model, label, controller=None, on_row=None):
        assert isinstance(env, audit.LearningAuditObserver)
        layers, current = [], env.env
        while hasattr(current, "env"):
            layers.append(current)
            current = current.env
        assert sum(isinstance(layer, setup.Isolation) for layer in layers) == 1
        assert sum(isinstance(layer, PotionCapacityMask) for layer in layers) == 1
        calls["played"].append((label, list(seeds), model, controller))
        rows = []
        for index, seed in enumerate(seeds):
            outcomes = calls["outcomes"].get(label, ["death", "win"])
            row = audit_row(seed, label, outcome=outcomes[index % len(outcomes)],
                            errors=calls["observer_errors"].get(label))
            if label in {"current", "previous"}:
                original = read_json(source / ("after_rl.json" if label == "current" else "before_rl.json"))
                frozen = next(item for item in original["episodes"] if item["seed"] == seed)
                row.update({key: frozen[key] for key in ("seed", "outcome", "floor", "hp", "steps", "return", "error")})
                row["act1_cleared"] = row["outcome"] == "win"
            if row["outcome"] == "win":
                row["floor"] = 16
            elif row["outcome"] == "timeout":
                row.update(steps=4000, hp=10, **{"return": 0})
            row["audit"].update(public_choices=[], policy_probes=[])
            for entry in row["audit"]["combat_entries"]:
                entry.update(max_hp=80, deck_size=1, permanent_deck=[{"card_id": "STRIKE_IRONCLAD", "upgraded": False}],
                             relics=["BURNING_BLOOD"], potions=[None, "FIRE_POTION"],
                             enemies=["VANTOM"], enemy_hp=[100], player_powers={})
            if calls["invalid_rows"]:
                calls["invalid_rows"](row)
            rows.append(row)
            if on_row is not None:
                on_row(row)
            if calls["mutate"]:
                calls["mutate"]()
            if calls["interrupt"] == label:
                calls["interrupt"] = None
                raise KeyboardInterrupt
        return rows

    def forbidden(*args, **kwargs):
        pytest.fail("A frozen behavior audit must never train, imitate, or promote")

    monkeypatch.setattr(audit, "MaskablePPO", Policy)
    monkeypatch.setattr(audit, "FairTeacher", Teacher)
    monkeypatch.setattr(audit, "play_controller", play)
    monkeypatch.setattr(runner, "train", forbidden)
    monkeypatch.setattr(runner, "promote", forbidden)
    return SimpleNamespace(module=audit, learning=setup, source=source, upstream=setup.upstream,
                           calls=calls, training_calls=training_calls)


def execute(setup, **kwargs):
    return setup.module.audit(setup.source, setup.upstream, **kwargs)


def report_at(result):
    return read_json(Path(result["report_path"]).with_name("report.json"))


def test_default_five_controllers_replay_ordered_fixed_block_and_actual_frozen_models(audit_setup):
    setup, audit = audit_setup, audit_setup.module
    source_bytes = bytes_in(setup.source)
    lineage = read_json(setup.source / "session-lineage.json")
    latest = read_json(setup.source / "latest.json")
    previous_model = PreservedPolicy.load(setup.source / lineage["initial_checkpoint"]["model"])
    current_model = PreservedPolicy.load(setup.source / latest["model"])
    result = execute(setup)
    assert inspect.signature(audit.audit).parameters["episodes"].default == 100
    assert result["additional_training_steps"] == 0
    assert result["champion_updated"] is False and result["training_gate_unchanged"] is True
    assert result["validation_start"] == 8800 and result["episodes_per_controller"] == 100
    assert [call[0] for call in setup.calls["played"]] == list(audit.LABELS)
    expected = evaluation_seeds("validation", 100, 7, start=8800)
    assert all(call[1] == expected for call in setup.calls["played"])
    assert expected == evaluation_seeds("validation", 1000, 7, start=8800)[:100]
    current, previous, teacher, combat, noncombat = setup.calls["played"]
    assert current[2].num_timesteps == latest["num_timesteps"]
    assert previous[2].num_timesteps == lineage["initial_checkpoint"]["num_timesteps"]
    tensors_equal(current[2], current_model)
    tensors_equal(previous[2], previous_model)
    assert current[3] is previous[3] is None
    assert teacher[2] is combat[2] is noncombat[2] is None
    assert combat[3].model is current[2] and combat[3].mode == "teacher_combat"
    assert noncombat[3].model is current[2] and noncombat[3].mode == "teacher_noncombat"
    assert all(bytes_in(setup.source)[name] == data for name, data in source_bytes.items())
    assert setup.learning.calls["training"] == setup.training_calls
    protocol = read_json(Path(result["report_path"]).with_name("protocol.json"))
    assert protocol["current_checkpoint"] == latest
    assert protocol["previous_checkpoint"] == lineage["initial_checkpoint"]
    assert protocol["use_lookahead"] is False


def test_completed_repeat_is_byte_identical_without_gameplay(audit_setup):
    first = execute(audit_setup, episodes=3)
    original = bytes_in(audit_setup.source)
    second = execute(audit_setup, episodes=3)
    assert second == first and bytes_in(audit_setup.source) == original
    assert len(audit_setup.calls["played"]) == 5


@pytest.mark.parametrize("phase,pending", [("COMBAT", False), ("COMBAT", True),
                                           ("EVENT", True), ("MAP", False), ("REST", False)])
def test_phase_hybrids_include_pending_combat_choices(session_auditor, phase, pending):
    obs, masks = {"visible": np.array([1])}, np.array([False, True, True])
    calls = {"model": 0, "teacher": 0}
    class Model:
        def predict(self, supplied, *, action_masks, deterministic):
            assert supplied is obs and action_masks is masks and deterministic is True
            calls["model"] += 1
            return np.int64(1), None
    class Teacher:
        def select(self, supplied):
            assert supplied is env
            calls["teacher"] += 1
            return 2
    combat = SimpleNamespace(pending_choice=object() if pending else None)
    mgr = SimpleNamespace(phase=phase, PHASE_COMBAT="COMBAT", get_combat_state=lambda: combat)
    env = SimpleNamespace(raw_env=SimpleNamespace(_mgr=mgr), current_observation=obs,
                          action_masks=lambda: masks)
    is_combat = phase == "COMBAT"
    assert session_auditor.HybridController(Model(), Teacher(), "teacher_combat").select(env) == (2 if is_combat else 1)
    assert session_auditor.HybridController(Model(), Teacher(), "teacher_noncombat").select(env) == (1 if is_combat else 2)
    assert calls == {"model": 1, "teacher": 1}
    with pytest.raises(ValueError, match="Unknown|mode|category"):
        session_auditor.HybridController(Model(), Teacher(), "all_choices")


@pytest.mark.parametrize("changed", [{"episodes": 4}, {"start": 8801}])
def test_existing_audit_sample_and_schedule_are_fixed(audit_setup, changed):
    execute(audit_setup, episodes=3)
    original = bytes_in(audit_setup.source)
    with pytest.raises(ValueError):
        execute(audit_setup, **({"episodes": 3} | changed))
    assert bytes_in(audit_setup.source) == original and len(audit_setup.calls["played"]) == 5


def test_indices_outside_frozen_validation_block_rejected_before_any_gameplay(audit_setup):
    with pytest.raises(ValueError, match="frozen|subset|source|block|range"):
        execute(audit_setup, episodes=3, start=9800)
    assert not audit_setup.calls["played"]


def test_interrupted_controller_resumes_only_missing_seed_suffix(audit_setup):
    setup = audit_setup
    setup.calls["interrupt"] = "fair_teacher"
    interrupted = execute(setup, episodes=3)
    assert interrupted["status"] == "interrupted"
    prefix = [(label, seeds) for label, seeds, _, _ in setup.calls["played"]]
    assert [label for label, _ in prefix] == ["current", "previous", "fair_teacher"]
    result = execute(setup, episodes=3)
    assert result["additional_training_steps"] == 0
    labels = [call[0] for call in setup.calls["played"]]
    assert labels == ["current", "previous", "fair_teacher", "fair_teacher", "teacher_combat", "teacher_noncombat"]
    expected = evaluation_seeds("validation", 3, 7, start=8800)
    assert setup.calls["played"][3][1] == expected[1:]
    folder = Path(result["report_path"]).parent
    assert all(len(read_json(folder / "controllers" / f"{label}.json")["episodes"]) == 3 for label in setup.module.LABELS)


@pytest.mark.parametrize("tamper", ["protocol", "seed", "label", "clear", "nan_floor"])
def test_completed_cached_raw_evidence_tampering_is_rejected(audit_setup, tamper):
    result = execute(audit_setup, episodes=3)
    folder = Path(result["report_path"]).parent
    path = folder / "controllers" / "current.json"
    data = read_json(path)
    if tamper == "protocol":
        data["protocol_sha256"] = "replaced-protocol"
    elif tamper == "seed":
        data["episodes"][0]["seed"] += 1
    elif tamper == "label":
        data["episodes"][0]["policy"] = "teacher_combat"
    elif tamper == "clear":
        data["episodes"][0]["act1_cleared"] = not data["episodes"][0]["act1_cleared"]
    else:
        data["episodes"][0]["floor"] = float("nan")
    if tamper == "nan_floor":
        path.write_text(__import__("json").dumps(data))
    else:
        atomic_json(path, data)
    with pytest.raises(ValueError):
        execute(audit_setup, episodes=3)
    assert len(audit_setup.calls["played"]) == 5


@pytest.mark.parametrize("tamper", ["source_protocol", "source_raw", "checkpoint", "counter"])
def test_incompatible_source_or_embedded_counter_stops_before_play(audit_setup, tamper):
    setup = audit_setup
    if tamper == "source_protocol":
        path = setup.source / "session-protocol.json"
        data = read_json(path); data["plan"]["new_steps"] += 1; atomic_json(path, data)
    elif tamper == "source_raw":
        path = setup.source / "after_rl.json"
        data = read_json(path); data["episodes"][0]["seed"] += 1; atomic_json(path, data)
    elif tamper == "checkpoint":
        latest = read_json(setup.source / "latest.json")
        (setup.source / latest["model"]).write_bytes(b"replaced model")
    else:
        setup.calls["bad_counter"] = True
    with pytest.raises(ValueError):
        execute(setup, episodes=3)
    assert not setup.calls["played"]


def test_learning_file_mutation_during_play_is_detected(audit_setup):
    setup = audit_setup
    setup.calls["mutate"] = lambda: (setup.source / "episodes.jsonl").write_text("changed journal\n")
    result = execute(setup, episodes=3)
    assert result["status"] == "paused" and result["pause_reason"]
    assert not Path(result["report_path"]).with_name("completion.json").exists()


@pytest.mark.parametrize("failure", ["aborted", "observer"])
def test_game_and_observer_errors_block_diagnostic_conclusions(audit_setup, failure):
    if failure == "aborted":
        audit_setup.calls["outcomes"]["fair_teacher"] = ["aborted", "death"]
    else:
        audit_setup.calls["observer_errors"]["current"] = [{"stage": "public_snapshot", "error": "missing public field"}]
    result = execute(audit_setup, episodes=3)
    assert result["status"] == "paused"
    assert result["observer_or_game_errors"] is True and result["playing_strength_claim"] is False
    assert result["paired_descriptive"] == {}
    assert result["additional_training_steps"] == 0 and result["champion_updated"] is False
    assert result["training_gate_unchanged"] is True


def test_legal_timeout_remains_explicit_nonclear_in_all_attempt_denominator(audit_setup):
    audit_setup.calls["outcomes"]["teacher_combat"] = ["timeout", "win", "death"]
    result = execute(audit_setup, episodes=3)
    summary = report_at(result)["controllers"]["teacher_combat"]
    gameplay = summary["gameplay"]
    assert gameplay["act1_clears"] == 1 and gameplay["attempts"] == 3
    assert gameplay["clear_rate"] == pytest.approx(1/3)
    assert gameplay["outcomes"]["timeout"] == 1
    assert result["status"] == "completed"


def test_compact_bundle_contains_evidence_and_excludes_models(audit_setup):
    result = execute(audit_setup, episodes=3)
    bundle = Path(result["diagnostics_zip"])
    assert bundle.is_file()
    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
        assert any(name.endswith("report.json") for name in names)
        assert any(name.endswith("protocol.json") for name in names)
        assert any("controllers/current.json" in name for name in names)
        assert not any(name.endswith(".zip") or "/checkpoints/" in name for name in names)
    assert Path(result["report_path"]).is_file()


@pytest.mark.parametrize("tamper", ["seed", "label", "clear", "outcome"])
def test_partial_controller_cache_is_validated_before_resume(audit_setup, tamper):
    setup = audit_setup
    setup.calls["interrupt"] = "fair_teacher"
    interrupted = execute(setup, episodes=3)
    assert interrupted["status"] == "interrupted"
    paths = list(setup.source.parent.parent.rglob("controllers/fair_teacher.json"))
    assert len(paths) == 1
    path = paths[0]; payload = read_json(path)
    assert len(payload["episodes"]) == 1
    row = payload["episodes"][0]
    if tamper == "seed":
        row["seed"] += 1
    elif tamper == "label":
        row["policy"] = "current"
    elif tamper == "clear":
        row["act1_cleared"] = not row["act1_cleared"]
    else:
        row["outcome"] = "invented_success"
    payload["evidence_sha256"] = setup.module._payload_digest(payload)
    atomic_json(path, payload)
    try:
        rejected = execute(setup, episodes=3)
    except ValueError:
        pass
    else:
        assert rejected["status"] == "paused" and rejected["pause_reason"]
        assert rejected["paired_descriptive"] == {} and rejected["playing_strength_claim"] is False
    assert len(setup.calls["played"]) == 3


@pytest.mark.parametrize("bad", ["wrong_label", "invalid_outcome", "negative_steps"])
def test_malformed_new_episode_is_rejected_without_completed_report(audit_setup, bad):
    def mutate(row):
        if bad == "wrong_label":
            row["policy"] = "unplanned_controller"
        elif bad == "invalid_outcome":
            row["outcome"] = "success"
        else:
            row["steps"] = -1
    audit_setup.calls["invalid_rows"] = mutate
    existing = set(audit_setup.source.parent.parent.rglob("completion.json"))
    try:
        result = execute(audit_setup, episodes=3)
    except ValueError:
        pass
    else:
        assert result["status"] == "paused" and result["pause_reason"]
    assert set(audit_setup.source.parent.parent.rglob("completion.json")) == existing


def public_combat_env():
    """Expose player-visible fields, with traps on hidden simulator state."""
    strike = SimpleNamespace(card_id=SimpleNamespace(name="STRIKE_IRONCLAD"), upgraded=False)
    bash = SimpleNamespace(card_id=SimpleNamespace(name="BASH"), upgraded=True)
    potions = [None, SimpleNamespace(potion_id="FIRE_POTION")]
    player = SimpleNamespace(current_hp=37, max_hp=80, block=4, powers={"STRENGTH": SimpleNamespace(amount=2)})
    permanent = SimpleNamespace(current_hp=37, max_hp=80, deck=[strike, strike, bash],
                                relics=["BURNING_BLOOD"], potions=potions)
    enemy = SimpleNamespace(monster_id="VANTOM", current_hp=110, max_hp=150, block=3,
                            powers={"WEAK": SimpleNamespace(amount=1)})
    class Combat:
        pending_choice = None
        energy = 2
        hand = [strike]
        enemies = [enemy]
        @property
        def draw_pile(self):
            pytest.fail("Observer must not inspect hidden draw order")
        @property
        def rng(self):
            pytest.fail("Observer must not inspect simulator RNG")
    combat = Combat(); combat.player = player; combat.potions = potions
    mgr = SimpleNamespace(phase="COMBAT", PHASE_COMBAT="COMBAT", PHASE_EVENT="EVENT",
                          current_room=SimpleNamespace(room_type=SimpleNamespace(name="BOSS")),
                          run_state=SimpleNamespace(total_floor=16, player=permanent),
                          get_combat_state=lambda: combat)
    class Env:
        raw_env = SimpleNamespace(_mgr=mgr)
        def reset(self, *, seed=None):
            return {"visible": np.array([1], dtype=np.float32)}, {"floor": 16, "phase": "COMBAT"}
        def action_masks(self):
            return np.array([True, True, True])
    return Env(), mgr, combat


def test_observer_captures_public_boss_entry_inventory_without_hidden_reads(session_auditor):
    env, mgr, combat = public_combat_env()
    observer = session_auditor.LearningAuditObserver(env)
    observation, info = observer.reset(seed=7)
    assert observer.current_observation is observation
    assert not observer.episode_audit["observer_errors"]
    assert len(observer.episode_audit["combat_entries"]) == 1
    snapshot = observer.episode_audit["combat_entries"][0]
    assert snapshot["room_type"] == "BOSS" and snapshot["floor"] == 16
    assert snapshot["hp"] == 37 and snapshot["max_hp"] == 80 and snapshot["deck_size"] == 3
    assert snapshot["permanent_deck"] == [
        {"card_id": "BASH", "upgraded": True, "count": 1},
        {"card_id": "STRIKE_IRONCLAD", "upgraded": False, "count": 2}]
    assert snapshot["relics"] == ["BURNING_BLOOD"]
    assert snapshot["potions"] == [None, "FIRE_POTION"]
    assert snapshot["player_powers"] == [{"power_id": "STRENGTH", "amount": 2}]
    assert snapshot["enemy_state"] == [{"enemy_id": "VANTOM", "hp": 110, "max_hp": 150,
                                         "block": 3, "powers": [{"power_id": "WEAK", "amount": 1}]}]
    # Taking a snapshot must not mutate the underlying visible state.
    assert combat.player.current_hp == 37 and combat.player.powers["STRENGTH"].amount == 2


def test_observer_snapshot_fault_is_disclosed_without_breaking_game_reset(session_auditor):
    env, mgr, combat = public_combat_env()
    del combat.enemies[0].current_hp
    observer = session_auditor.LearningAuditObserver(env)
    observation, info = observer.reset(seed=7)
    assert info["floor"] == 16 and observer.current_observation is observation
    errors = observer.episode_audit["observer_errors"]
    assert len(errors) == 1 and errors[0]["stage"] == "reset_combat_entry"
    assert "AttributeError" in errors[0]["error"]
    assert observer.episode_audit["combat_entries"] == []


def test_optional_policy_probe_is_deterministic_bounded_and_no_grad(session_auditor):
    env, mgr, combat = public_combat_env()
    calls = []
    class Policy:
        def obs_to_tensor(self, observation):
            calls.append("observation")
            return torch.as_tensor(observation["visible"]), False
        def get_distribution(self, observation, *, action_masks):
            assert not torch.is_grad_enabled()
            assert action_masks.tolist() == [True, True, True]
            calls.append("distribution")
            return SimpleNamespace(entropy=lambda: torch.tensor(0.5))
        def predict_values(self, observation):
            assert not torch.is_grad_enabled()
            calls.append("value")
            return torch.tensor(-0.25)
    observer = session_auditor.LearningAuditObserver(env, model=SimpleNamespace(policy=Policy()))
    observer.reset(seed=7)
    rng_state = torch.get_rng_state().clone()
    observer._probe()
    assert torch.equal(rng_state, torch.get_rng_state())
    assert len(observer.episode_audit["policy_probes"]) == 1
    probe = observer.episode_audit["policy_probes"][0]
    assert probe["masked_entropy"] == 0.5 and probe["critic_value"] == -0.25
    assert probe["legal_actions"] == 3
    observer._steps = 1
    observer._probe()
    assert len(calls) == 3
    for index in range(session_auditor.PROBE_LIMIT + 2):
        observer._steps = index * session_auditor.PROBE_STRIDE
        observer._probe()
    assert len(observer.episode_audit["policy_probes"]) == session_auditor.PROBE_LIMIT


@pytest.mark.parametrize("field", ["seed", "outcome", "floor", "hp", "steps", "return", "error"])
def test_frozen_replay_parity_checks_every_original_game_field(audit_setup, field):
    setup = audit_setup
    original = read_json(setup.source / "after_rl.json")["episodes"][0]
    row = audit_row(original["seed"], "current")
    row.update({key: original[key] for key in setup.module.GAME_FIELDS})
    row["audit"].update(public_choices=[], policy_probes=[])
    row.update(act1_cleared=False, matches_source_result=True,
               original_result={key: original[key] for key in setup.module.GAME_FIELDS})
    if field == "outcome":
        row[field] = "win"; row["act1_cleared"] = True
    elif field == "error":
        row[field] = "Simulated changed error"
    else:
        row[field] += 1
    payload = {"protocol_sha256": "immutable", "label": "current", "complete": True,
               "episodes": [row], "teacher_decisions": {}, "teacher_phases": {}}
    payload["evidence_sha256"] = setup.module._payload_digest(payload)
    with pytest.raises(ValueError, match="source|seeds|classification|infrastructure error"):
        setup.module._validate_payload(payload, "current", [original["seed"]], "immutable",
                                       originals=[original], cap=4000)


def test_frozen_gameplay_replay_mismatch_pauses_and_discloses_it(audit_setup):
    def change_action_effect(row):
        if row["policy"] == "current":
            row["steps"] += 1
    audit_setup.calls["invalid_rows"] = change_action_effect
    result = execute(audit_setup, episodes=3)
    assert result["status"] == "paused" and result["observer_or_game_errors"] is True
    assert result["playing_strength_claim"] is False and result["paired_descriptive"] == {}
    assert not Path(result["report_path"]).with_name("completion.json").exists()
    cached = read_json(Path(result["report_path"]).parent / "controllers/current.json")
    assert cached["episodes"][0]["matches_source_result"] is False
    assert cached["episodes"][0]["original_result"]["steps"] + 1 == cached["episodes"][0]["steps"]


def test_explicit_cohort_is_only_contiguous_subset_of_original_order(audit_setup):
    result = execute(audit_setup, episodes=3, start=8805)
    assert result["validation_start"] == 8805
    expected = evaluation_seeds("validation", 1000, 7, start=8800)[5:8]
    assert all(call[1] == expected for call in audit_setup.calls["played"])
    protocol = read_json(Path(result["report_path"]).with_name("protocol.json"))
    assert "already" in protocol["cohort_scope"] and "not a fresh" in protocol["cohort_scope"]

"""Numerical, replay-accounting, and safe-bundle checks for the new workflow."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest


SCRIPT_DIR = Path(__file__).parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


report = load("learning_session_report")
trace = load("learning_session_trace")


def row(seed, outcome="death", floor=5, **kwargs):
    return {"seed": seed, "outcome": outcome, "floor": floor, "hp": 0.0,
            "steps": 40, "return": -1.0, "error": None, **kwargs}


def replayed(original, *, result=None, match=True, **kwargs):
    played = dict(original if result is None else result)
    return {"original_result": original, "replayed_result": played,
            "matches_original_result": match, "logging_errors": [], **kwargs}


def test_all_attempt_denominator_keeps_timeout_and_abort():
    rows = [row(1, "win"), row(2), row(3, "timeout", 7), row(4, "aborted", 3, error="engine failed")]
    saved = copy.deepcopy(rows)
    result = report.summarize_attempts(rows)
    assert result["clear_rate"] == 0.25
    assert result["episodes"] == 4 and result["act1_clears"] == 1
    assert result["outcomes"] == {"win": 1, "death": 1, "timeout": 1, "aborted": 1}
    assert result["mean_floor_completed"] == 5 and result["mean_floor_observed"] == 17 / 3
    assert result["infrastructure_failure_rate"] == 0.25
    assert rows == saved


@pytest.mark.parametrize("mutate", [
    {"seed": True}, {"seed": -1}, {"seed": 2**31}, {"outcome": "crash"},
    {"floor": float("nan")}, {"floor": float("inf")}, {"floor": -1},
    {"floor": True}, {"steps": -1}, {"return": float("nan")}, {"hp": float("inf")},
    {"error": "failed"},
])
def test_invalid_episode_evidence_rejected(mutate):
    with pytest.raises(ValueError):
        report.checked_rows([{**row(1), **mutate}])


def test_duplicate_and_mismatched_seed_lists_rejected():
    with pytest.raises(ValueError):
        report.checked_rows([row(1), row(1)])
    with pytest.raises(ValueError):
        report.compare_attempts([row(1), row(2)], [row(2), row(1)])
    with pytest.raises(ValueError):
        report.summarize_attempts([])


def test_known_exact_paired_test_and_clear_difference():
    before = [row(seed, "win" if 38 <= seed < 67 else "death") for seed in range(1000)]
    after = [row(seed, "win" if seed < 38 else "death") for seed in range(1000)]
    result = report.compare_attempts(before, after)
    paired = result["paired_clear_evidence"]
    assert paired["candidate_only_clears"] == 38
    assert paired["baseline_only_clears"] == 29
    assert paired["exact_two_sided_p"] == pytest.approx(0.3284320626971897)
    assert result["rl_act1_clear_rate_gain"] == 0.009
    assert not result["establishes_positive_act1_clear_gain"]


def test_bootstrap_matches_known_empirical_distribution():
    # Two paired values +1 and 0: bootstrap mean has probabilities .25/.50/.25.
    before = [row(1), row(2)]
    after = [row(1, "win"), row(2)]
    result = report.compare_attempts(before, after, bootstrap_samples=10000)
    assert result["rl_act1_clear_rate_gain_ci95"] == [0, 1]
    assert result["rl_act1_clear_rate_gain"] == 0.5


def test_positive_and_negative_clear_evidence():
    before = [row(seed) for seed in range(100)]
    after = [row(seed, "win" if seed < 30 else "death") for seed in range(100)]
    positive = report.compare_attempts(before, after)
    negative = report.compare_attempts(after, before)
    assert positive["establishes_positive_act1_clear_gain"]
    assert positive["rl_act1_clear_rate_gain_ci95"][0] > 0
    assert not negative["establishes_positive_act1_clear_gain"]
    assert negative["rl_act1_clear_rate_gain_ci95"][1] < 0


def test_no_discordant_wins_does_not_claim_equivalence():
    rows = [row(seed) for seed in range(20)]
    result = report.compare_attempts(rows, rows)
    assert result["rl_act1_clear_rate_gain_ci95"] == [0, 0]
    assert result["paired_clear_evidence"]["exact_two_sided_p"] == 1
    assert not result["establishes_positive_act1_clear_gain"]
    assert "does not establish equal ability" in result["note"]


def test_abort_blocks_strength_even_with_large_clear_gain():
    before = [row(seed) for seed in range(100)]
    after = [row(seed, "win" if seed < 70 else "death") for seed in range(100)]
    after[-1] = row(99, "aborted", error="broken simulator")
    result = report.compare_attempts(before, after)
    assert result["infrastructure_failure_count"] == 1
    assert result["rl_act1_clear_rate_gain"] == 0.7
    assert not result["paired_clear_evidence"]["supports_more_clears"]
    assert not result["establishes_positive_act1_clear_gain"]


def test_reproduced_legal_timeout_remains_nonclear_without_pause():
    original = row(1, "timeout", steps=4000, **{"return": 0.0})
    result = report.classify_replays([replayed(original, original_decision_cap=4000)])
    assert result["matched_timeouts"] == 1
    assert not result["requires_pause"]


@pytest.mark.parametrize("case", ["mismatch", "abort", "logging", "cap"])
def test_replay_contract_failures_pause(case):
    original = row(1, "timeout", steps=4000)
    item = replayed(original, original_decision_cap=4000)
    if case == "mismatch":
        item["replayed_result"]["return"] = 1.0
    elif case == "abort":
        item["replayed_result"] = row(1, "aborted", error="failure")
    elif case == "logging":
        item["logging_errors"] = [{"error": "unserializable public state"}]
    else:
        item["original_decision_cap"] = 4001
    assert report.classify_replays([item])["requires_pause"]


def test_report_zip_has_evidence_and_excludes_model_catalogs_and_symlinks(tmp_path):
    (tmp_path / "session-lineage.json").write_text('{"source": "frozen"}')
    (tmp_path / "before_rl.json").write_text('{"raw": "evaluation"}')
    (tmp_path / "episodes.jsonl").write_text('{"outcome":"death"}\n')
    (tmp_path / "session.log").write_bytes(b"a" * (report.MAX_LOG_BYTES + 20) + b"LOGEND")
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints/model.zip").write_bytes(b"model")
    traces = tmp_path / "diagnostics" / "seed-1"
    traces.mkdir(parents=True)
    (traces / "summary.json").write_text('{"outcome":"timeout"}')
    (traces / "actions.jsonl").write_text('{"secret":"catalog"}')
    (traces / "observations.jsonl").write_text('{"secret":"catalog"}')
    external = tmp_path.parent / "outside-evidence.json"
    external.write_text('{"outside":true}')
    (tmp_path / "protocol.json").symlink_to(external)
    paths = report.write_report(tmp_path, {"status": "paused", "pause_reason": "diagnostic check"})
    assert Path(paths["text_report"]).is_file()
    with zipfile.ZipFile(paths["diagnostics_zip"]) as bundle:
        names = bundle.namelist()
        assert "before_rl.json" in names and "episodes.jsonl" in names
        assert "diagnostics/seed-1/summary.json" in names
        assert not any("model" in name or "actions" in name or "observations" in name for name in names)
        assert "protocol.json" not in names
        assert bundle.read("logs/session.log.tail.txt").endswith(b"LOGEND")
        assert len(bundle.read("logs/session.log.tail.txt")) == report.MAX_LOG_BYTES
        manifest = json.loads(bundle.read("bundle-manifest.json"))
        assert manifest["omitted"][0]["reason"] == "symlink or outside report root"


def test_bundle_size_limit_disclosed(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "MAX_FILE_BYTES", 200)
    (tmp_path / "before_rl.json").write_text("x" * 201)
    paths = report.write_report(tmp_path, {"status": "completed"})
    with zipfile.ZipFile(paths["diagnostics_zip"]) as bundle:
        manifest = json.loads(bundle.read("bundle-manifest.json"))
        assert "before_rl.json" not in bundle.namelist()
        assert any(row["path"] == "before_rl.json" and row["reason"] == "bundle size bound" for row in manifest["omitted"])


def test_explicit_session_log_tail_supported_for_staged_pause_report(tmp_path):
    folder = tmp_path / "reports" / "session-0001"
    log = tmp_path / "session-0001.log"
    log.write_text("original session console output")
    paths = report.write_report(folder, {"status": "paused", "log_path": str(log)})
    with zipfile.ZipFile(paths["diagnostics_zip"]) as bundle:
        assert bundle.read("log-tail.txt") == log.read_bytes()


def test_arbitrary_external_file_cannot_be_selected_as_log(tmp_path):
    external = tmp_path / "credentials.txt"
    external.write_text("do not include this")
    paths = report.write_report(tmp_path / "session-0001", {"status": "paused", "log_path": str(external)})
    with zipfile.ZipFile(paths["diagnostics_zip"]) as bundle:
        assert "log-tail.txt" not in bundle.namelist()
        assert "outside allowed session layout" in json.loads(bundle.read("bundle-manifest.json"))["omitted"][0]["reason"]


def test_bundle_manifest_and_entry_counts_are_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "MAX_BUNDLE_FILES", 2)
    monkeypatch.setattr(report, "MAX_OMITTED_RECORDS", 1)
    for seed in range(5):
        folder = tmp_path / "diagnostics" / f"seed-{seed}"
        folder.mkdir(parents=True)
        (folder / "summary.json").write_text('{"outcome":"timeout"}')
    paths = report.write_report(tmp_path, {"status": "completed"})
    with zipfile.ZipFile(paths["diagnostics_zip"]) as bundle:
        manifest = json.loads(bundle.read("bundle-manifest.json"))
        assert len(bundle.namelist()) == 3  # Two evidence entries and the manifest.
        assert len(manifest["omitted"]) == 1
        assert manifest["omitted_files"] == 5 and manifest["omitted_records_truncated"] == 4


def test_archive_size_reserves_room_for_manifest_and_headers(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "MAX_BUNDLE_BYTES", 600 * 1024)
    (tmp_path / "before_rl.json").write_bytes(b"x" * (100 * 1024))
    paths = report.write_report(tmp_path, {"status": "completed"})
    assert Path(paths["diagnostics_zip"]).stat().st_size <= 600 * 1024
    with zipfile.ZipFile(paths["diagnostics_zip"]) as bundle:
        assert "before_rl.json" not in bundle.namelist()
        assert json.loads(bundle.read("bundle-manifest.json"))["omitted_files"] == 1


class FakeEnvironment:
    def __init__(self, outcome="timeout", fail_step=None):
        self.actions, self.outcome, self.fail_step = [], outcome, fail_step

    def reset(self, *, seed):
        return {"visible": [1, 2]}, {"floor": 5, "hp": 17.0, "phase": "choice"}

    def action_masks(self):
        return [True, False]

    def step(self, action):
        if self.fail_step == len(self.actions):
            raise RuntimeError("engine failed")
        self.actions.append(action)
        done = len(self.actions) == 3
        return {"visible": [1, 2]}, 0.25, done and self.outcome != "timeout", done and self.outcome == "timeout", {
            "floor": 5, "hp": 17.0, "phase": "choice", "outcome": self.outcome}


class FakePolicy:
    def predict(self, obs, *, action_masks, deterministic):
        assert deterministic is True and action_masks == [True, False]
        return 0, None


def identity(value):
    import numpy as np
    if isinstance(value, dict):
        return {key: identity(item) for key, item in value.items()}
    return value.tolist() if isinstance(value, np.ndarray) else value


def test_logger_failure_preserves_all_original_game_actions_and_return():
    def broken(env, action):
        raise RuntimeError("logger failed")
    plain, logged = FakeEnvironment(), FakeEnvironment()
    original = trace._run(plain, FakePolicy(), 12, cap=3, describe=lambda env, action: {}, serialize=identity)
    result = trace._run(logged, FakePolicy(), 12, cap=3, describe=broken, serialize=identity)
    assert {key: result[key] for key in trace.GAME_FIELDS} == {key: original[key] for key in trace.GAME_FIELDS}
    assert plain.actions == logged.actions == [0, 0, 0]
    assert result["return"] == 0.75 and result["logging_error_count"] == 3
    assert original["most_repeated_public_observation_action_visits"] == 3


def test_game_failure_retains_original_error_and_last_completed_info():
    env = FakeEnvironment(fail_step=1)
    result = trace._run(env, FakePolicy(), 12, cap=3, describe=lambda env, action: {}, serialize=identity)
    assert result["outcome"] == "aborted"
    assert result["error"] == "RuntimeError: engine failed"
    assert result["error_stage"] == "game_step" and result["steps"] == 1
    assert result["return"] == 0.25 and result["floor"] == 5 and result["hp"] == 17


def test_replay_tail_bound_does_not_change_game_path():
    env = FakeEnvironment()
    result = trace._run(env, FakePolicy(), 12, cap=3, tail=1, describe=lambda env, action: {}, serialize=identity)
    assert len(result["last_decisions"]) == 1 and result["last_decisions"][0]["step"] == 2
    assert len(env.actions) == 3

"""Promotion must verify episode evidence and the exact assessed artifacts."""

import hashlib
import json
from pathlib import Path

import pytest

from spire2_rl.runner import promote
from spire2_rl.study import evaluation_seeds, paired_compare


IDENTITY = "promotion-test-environment"


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint(root: Path, name: str) -> tuple[Path, dict]:
    path = root / "checkpoints" / f"{name}.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"fake weights for {name}".encode())
    state = {
        "model": str(path.relative_to(root)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "fingerprint": IDENTITY,
        "num_timesteps": 123,
        "next_indices": [100],
        "interrupted": False,
    }
    write_json(path.with_suffix(".json"), state)
    return path.resolve(), state


def setup_report(root: Path, previous: dict | None = None) -> tuple[Path, dict]:
    write_json(root / "experiment.json", {
        "fingerprint": IDENTITY,
        "backend": {"backend": "test-fixture", "real_game_transfer": "unverified"},
    })
    candidate_path, candidate_state = checkpoint(root, "candidate")
    if previous is None:
        baseline_label = "random"
        baseline_identity = {"kind": "random"}
    else:
        write_json(root / "champion.json", previous)
        baseline_path = (root / previous["checkpoint"]["model"]).resolve()
        baseline_label = str(baseline_path)
        baseline_identity = {
            "kind": "checkpoint",
            "path": str(baseline_path),
            "sha256": previous["checkpoint"]["sha256"],
        }
    seeds = evaluation_seeds("validation", 100)
    candidate_rows = [
        {"seed": seed, "policy": candidate_path.name, "outcome": "win", "floor": 50, "hp": 20}
        for seed in seeds
    ]
    baseline_rows = [
        {"seed": seed, "policy": baseline_label, "outcome": "death", "floor": 5, "hp": 0}
        for seed in seeds
    ]
    report = {
        "fingerprint": IDENTITY,
        "checkpoint": candidate_state,
        "backend": {"backend": "test-fixture", "real_game_transfer": "unverified"},
        "split": "validation",
        "seed_start": 0,
        "baseline_identity": baseline_identity,
        "candidate_episodes": candidate_rows,
        "baseline_episodes": baseline_rows,
        "comparison": paired_compare(candidate_rows, baseline_rows, bootstrap_samples=100),
    }
    path = root / "evaluations" / "validation.json"
    write_json(path, report)
    return path, report


def previous_champion(root: Path) -> dict:
    _, state = checkpoint(root, "incumbent")
    return {"checkpoint": state, "evaluation": "older-validation.json", "backend": {"backend": "test-fixture"}}


def test_clean_validation_can_promote_first_champion_without_loading_a_model(tmp_path):
    path, report = setup_report(tmp_path)
    champion = promote(tmp_path, path)
    assert champion["checkpoint"] == report["checkpoint"]
    assert champion["evaluation"] == str(path.resolve())
    assert champion["previous"] is None
    assert read_json(tmp_path / "champion.json") == champion


def test_forged_promotion_flag_cannot_override_timeout_episode_evidence(tmp_path):
    path, report = setup_report(tmp_path)
    assert report["comparison"]["promote"] is True
    # Leave the old successful comparison in place, but invalidate every actual run.
    for row in report["candidate_episodes"]:
        row["outcome"] = "timeout"
    write_json(path, report)
    with pytest.raises(ValueError):
        promote(tmp_path, path)
    assert not (tmp_path / "champion.json").exists()
    assert not (tmp_path / ".running.lock").exists()


@pytest.mark.parametrize("change", ["test_split", "report_fingerprint", "checkpoint_fingerprint", "checkpoint_bytes"])
def test_wrong_split_or_changed_evaluation_identity_cannot_create_champion(tmp_path, change):
    path, report = setup_report(tmp_path)
    if change == "test_split":
        report["split"] = "test"
    elif change == "report_fingerprint":
        report["fingerprint"] = "other-experiment"
    elif change == "checkpoint_fingerprint":
        # Matching report/manifest fingerprints cannot authorize a checkpoint from another run.
        report["checkpoint"]["fingerprint"] = "other-experiment"
    else:
        (tmp_path / report["checkpoint"]["model"]).write_bytes(b"changed after evaluation")
    write_json(path, report)
    with pytest.raises(ValueError):
        promote(tmp_path, path)
    assert not (tmp_path / "champion.json").exists()


def test_existing_champion_is_replaced_only_after_comparison_against_its_exact_artifact(tmp_path):
    previous = previous_champion(tmp_path)
    path, report = setup_report(tmp_path, previous)
    champion = promote(tmp_path, path)
    assert champion["checkpoint"] == report["checkpoint"]
    assert champion["previous"] == previous
    assert read_json(tmp_path / "champion.json") == champion


@pytest.mark.parametrize("change", ["different_path", "different_checksum", "missing_identity", "changed_baseline_bytes"])
def test_stale_or_unbound_baseline_cannot_replace_existing_champion(tmp_path, change):
    previous = previous_champion(tmp_path)
    path, report = setup_report(tmp_path, previous)
    champion_path = tmp_path / "champion.json"
    champion_before = champion_path.read_bytes()
    baseline_path = Path(report["baseline_identity"]["path"])
    if change == "different_path":
        # Even identical bytes at a different path are not the incumbent named in the report.
        alternate = tmp_path / "checkpoints" / "different-incumbent.zip"
        alternate.write_bytes(baseline_path.read_bytes())
        report["baseline_identity"]["path"] = str(alternate.resolve())
    elif change == "different_checksum":
        report["baseline_identity"]["sha256"] = "0" * 64
    elif change == "missing_identity":
        del report["baseline_identity"]
    else:
        # A formerly correct digest becomes stale when the incumbent file changes.
        baseline_path.write_bytes(b"changed incumbent after evaluation")
    write_json(path, report)
    with pytest.raises(ValueError):
        promote(tmp_path, path)
    assert champion_path.read_bytes() == champion_before
    assert not (tmp_path / ".running.lock").exists()

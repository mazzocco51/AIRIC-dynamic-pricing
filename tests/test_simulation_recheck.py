"""Independent Nov 18 checks for market invariants and study reproducibility."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from hackathon_kit.cli import toml_text
from hackathon_kit.config import load_config
from hackathon_kit.simulator import resolve_opponent_configs, run_simulation


@pytest.fixture
def study():
    path = Path(__file__).resolve().parents[1] / "scripts" / "competition_study.py"
    spec = importlib.util.spec_from_file_location("competition_study_recheck", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evaluation_market_is_frozen_for_all_profiles(study, tmp_path):
    selected = load_config(profile="competitive")
    selected["market"].update(turns=7, initial_cash=92.0, capacity=3)
    path = tmp_path / "selected.toml"
    path.write_text(toml_text(selected), encoding="utf-8")
    configs, seeds = study.study_configurations("evaluation", path)
    assert len(seeds) == 20 and min(seeds) == 300000
    assert set(configs) == {"competitive", "baseline", "exp3", "gp"}
    assert all(config["market"] == selected["market"] for config in configs.values())
    configs["baseline"]["market"]["capacity"] = 2
    assert configs["competitive"]["market"]["capacity"] == 3


def test_matchups_are_invariant_to_focal_strategy_and_tuning(study):
    baseline = load_config(profile="baseline")
    tuned = load_config(profile="competitive")
    tuned["agent"].update(demand_prior=1.5, demand_optimism=0.0, cash_fraction=0.2)
    manifest = study.protocol_manifest({"baseline": baseline, "tuned": tuned})
    assert manifest["baseline"] == manifest["tuned"]
    assert len(manifest["baseline"]["strong"]["stable"]) == 3
    assert manifest["baseline"]["default"]["aggressive"] != manifest["baseline"]["default"]["stable"]


def test_paired_study_rejects_market_mismatch(study):
    first = load_config()
    second = deepcopy(first)
    second["market"]["initial_cash"] += 1
    with pytest.raises(ValueError, match="same market"):
        study.protocol_manifest({"first": first, "second": second})


def test_unknown_roster_is_not_silently_defaulted(study):
    with pytest.raises(ValueError, match="Unknown study roster"):
        study.run_job(("baseline", load_config(), "strnog", "stable", 0))


def test_default_roster_metadata_matches_actual_episode():
    cfg = load_config()
    cfg["market"]["turns"] = 2
    events = []
    run_simulation(cfg, "aggressive", seed=18, emit=events.append)
    assert events[0]["opponent_configs"] == resolve_opponent_configs(cfg, "aggressive")
    original = deepcopy(cfg)
    roster = resolve_opponent_configs(cfg, "stable")
    roster[0]["market"]["capacity"] += 1
    assert cfg == original


def test_default_roster_cannot_inherit_competitor_tuned_values():
    cfg = load_config(profile="competitive")
    cfg["agent"].update(cash_fraction=0.1, target_stock=8)
    normal = load_config()
    assert resolve_opponent_configs(cfg, "stable") == resolve_opponent_configs(normal, "stable")


def test_development_and_evaluation_seeds_are_disjoint(study, tmp_path):
    development_configs, development_seeds = study.study_configurations("development")
    selected = next(iter(development_configs.values()))
    path = tmp_path / "selected.toml"
    path.write_text(toml_text(selected), encoding="utf-8")
    _, evaluation_seeds = study.study_configurations("evaluation", path)
    assert set(development_seeds).isdisjoint(evaluation_seeds)
    assert len(list(study.jobs_for("candidate", selected, development_seeds))) == 28


def test_worker_failure_is_checkpointed_and_marked_failed(study, tmp_path, monkeypatch):
    class FailingPool:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def map(self, fn, jobs, **kwargs):
            yield {"profile": "first", "profit": 0, "seed": 200}
            raise RuntimeError("synthetic worker failure")

    out = tmp_path / "study-failed"
    monkeypatch.setattr(study, "ProcessPoolExecutor", FailingPool)
    monkeypatch.setattr(sys, "argv", ["competition_study.py", "development", "--out", str(out)])
    with pytest.raises(RuntimeError, match="synthetic worker"):
        study.main()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["completed_jobs"] == 1
    assert "synthetic worker failure" in manifest["error"]
    assert len(manifest["study_sha256"]) == 64
    assert "resolved_matchups" in manifest
    assert len((out / "partial-runs.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    assert not (out / "selected.toml").exists()


def test_trace_callbacks_cannot_mutate_episode_or_returned_results():
    cfg = load_config()
    cfg["market"]["turns"] = 2
    reference, reference_state = run_simulation(cfg, seed=19)

    def mutate_snapshot(event):
        if event["type"] == "header":
            event["config"]["market"]["turns"] = 1
            event["opponent_configs"][0]["agent"]["margin"] = 99
        elif event["type"] == "final":
            event["metrics"]["profit"] = 123456
            event["state"]["pending"]["injected"] = {}

    metrics, state = run_simulation(cfg, seed=19, emit=mutate_snapshot)
    assert {k: v for k, v in metrics.items() if not k.startswith("decision_ms")} == {
        k: v for k, v in reference.items() if not k.startswith("decision_ms")}
    assert state == reference_state

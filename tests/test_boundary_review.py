from copy import deepcopy
from dataclasses import asdict
import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

from hackathon_kit.config import load_config, validate_config, merge
from hackathon_kit.contracts import Action, Feedback, JsonAdapter, Observation
from hackathon_kit.cli import execute, build_parser, paired_comparisons, replay, save_run, source_fingerprint


def payload(**changes):
    result = dict(decision_id="turn:0", turn=0, phase="pricing", cash=20.0,
                  raw={"metal": 2, "plastic": 2}, stock={"widget": 1, "gadget": 0},
                  capacity=5, raw_cost={"metal": 2.0, "plastic": 1.0},
                  stock_cost={"widget": 4.0, "gadget": 0.0})
    result.update(changes)
    return result


@pytest.mark.parametrize("field,value", [
    ("cash", True), ("cash", "20"), ("cash", float("nan")),
    ("cash", 10**1000), ("decision_id", 1), ("raw", []),
    ("stock_cost", {"widget": True, "gadget": 0}),
    ("previous_prices", {"widget": float("nan")}),
    ("previous_prices", {"missing": 10}), ("price_min", True),
    ("bid_max", False), ("remaining_budget", "unknown"),
])
def test_malformed_observation_rejected(field, value):
    with pytest.raises(ValueError):
        JsonAdapter(load_config()).decode_observation(payload(**{field: value}))


def test_missing_price_limits_use_config_not_hardcoded_defaults():
    config = load_config()
    config["market"].update(price_min=100.0, price_max=200.0)
    obs = JsonAdapter(config).decode_observation(payload())
    assert (obs.price_min, obs.price_max) == (100, 200)
    assert obs.remaining_budget is None


def test_impossible_observed_tick_interval():
    config = load_config()
    config["market"]["price_tick"] = 1.0
    with pytest.raises(ValueError, match="tick"):
        JsonAdapter(config).decode_observation(payload(price_min=5.1, price_max=5.9))


@pytest.mark.parametrize("updates", [
    {"conversion_feedback": "false"}, {"reserve": {"metal": True, "plastic": 1}},
    {"production_cost": {"widget": False, "gadget": 1}}, {"initial_cash": True},
    {"price_min": 5.1, "price_max": 5.9, "price_tick": 1.0},
    {"recipes": {"widget": [], "gadget": {"metal": 1}}},
])
def test_configuration_types_and_tick_lattice(updates):
    config = load_config()
    config["market"].update(updates)
    with pytest.raises(ValueError):
        validate_config(config)


def test_scalar_configuration_section_is_rejected():
    with pytest.raises(ValueError):
        validate_config(merge(load_config(), {"market": 3}))
    with pytest.raises(ValueError):
        merge(load_config(), ["market"])


@pytest.mark.parametrize("feedback", [
    Feedback("x", "pricing", rewards={"widget": float("inf")}),
    Feedback("x", "pricing", sales={"widget": -1}),
    Feedback("x", "pricing", sales={"widget": True}),
    Feedback("x", "pricing", censored={"widget": "false"}),
    Feedback("x", "pricing", successes={"widget": 1}),
    Feedback("x", "pricing", successes={"widget": 2}, trials={"widget": 1}),
    Feedback("x", "auction", won=None),
    Feedback("x", "pricing", won=False),
    Feedback("x", "production", costs={"widget": -1}),
    Feedback("x", "pricing", rewards={"unknown": 1}),
])
def test_malformed_feedback_rejected(feedback):
    with pytest.raises(ValueError):
        JsonAdapter(load_config()).validate_feedback(feedback)


def test_boolean_action_price_and_bid_rejected():
    adapter = JsonAdapter(load_config())
    obs = adapter.decode_observation(payload(phase="auction", resource="metal"))
    with pytest.raises(ValueError):
        adapter.validate_action(obs, Action(obs.decision_id, bids={"metal": True}))
    obs = adapter.decode_observation(payload())
    with pytest.raises(ValueError):
        adapter.validate_action(obs, Action(obs.decision_id, prices={"widget": True, "gadget": 8}))


def test_empty_trace_fails_cleanly(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    with pytest.raises(ValueError, match="Empty trace"):
        replay(path)


def test_empty_existing_state_must_not_silently_reset(tmp_path):
    state = tmp_path / "state.json"
    state.write_text("{}")
    request = tmp_path / "input.json"
    request.write_text(json.dumps({"observation": payload()}))
    args = build_parser().parse_args(["act", "--input", str(request), "--state", str(state)])
    with pytest.raises(ValueError, match="state"):
        execute(args)
    assert state.read_text() == "{}"


def test_seed_overlap_rejected_before_creating_directory(tmp_path):
    destination = tmp_path / "overlap"
    args = build_parser().parse_args(["benchmark", "--out", str(destination),
                                     "--dev-start", "50", "--dev-seeds", "10", "--eval-start", "55"])
    with pytest.raises(ValueError, match="overlap"):
        execute(args)
    assert not destination.exists()


def test_paired_deltas_use_matching_seeds_not_unpaired_variance():
    rows = [dict(profile=p, split="evaluation", scenario="stable", seed=s, profit=v)
            for p, s, v in [("baseline", 1, 10), ("baseline", 2, 100), ("ucb", 2, 105), ("ucb", 1, 15)]]
    result = paired_comparisons(rows)[0]
    assert result["profit_delta_mean"] == 5
    assert result["profit_delta_se"] == 0
    assert result["positive_delta_fraction"] == 1


def test_tune_preserves_config_and_custom_grid(tmp_path, monkeypatch):
    config = tmp_path / "base.toml"
    config.write_text('[agent]\npricing="exp3ix"\n')
    recorded = []
    def fake_run(cfg, scenario, seed, out, opponent_configs=None):
        recorded.append((deepcopy(cfg), seed))
        return {"profit": cfg["agent"]["eta"]}
    monkeypatch.setattr("hackathon_kit.cli.save_run", fake_run)
    args = build_parser().parse_args(["tune", "--config", str(config), "--out", str(tmp_path / "tuning"),
                                     "--dev-seeds", "1", "--dev-start", "17", "--parameter", "eta=[0.1,0.2]"])
    execute(args)
    selected = load_config(tmp_path / "tuning" / "selected.toml")
    assert selected["agent"]["pricing"] == "exp3ix"
    assert selected["agent"]["eta"] == 0.2
    assert {seed for _, seed in recorded} == {17}


def test_source_manifest_and_failed_run_status(tmp_path, monkeypatch):
    assert len(source_fingerprint()) == 64
    def failure(*args, **kwargs):
        raise ValueError("injected failure")
    monkeypatch.setattr("hackathon_kit.cli.run_simulation", failure)
    with pytest.raises(ValueError, match="injected"):
        save_run(load_config(), "stable", 0, tmp_path / "failure")
    manifest = json.loads((tmp_path / "failure" / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["code_sha256"] == source_fingerprint()

from copy import deepcopy
import json

import pytest

from hackathon_kit.cli import build_parser, execute, paired_comparisons, replay, save_run, strict_json_loads
from hackathon_kit.config import load_config, validate_config
from hackathon_kit.contracts import JsonAdapter


def payload():
    return dict(decision_id="0", turn=0, phase="pricing", cash=20.0,
                raw={"metal": 2, "plastic": 2}, stock={"widget": 1, "gadget": 0},
                capacity=5, raw_cost={"metal": 2.0, "plastic": 1.0},
                stock_cost={"widget": 4.0, "gadget": 0.0})


@pytest.mark.parametrize("text", ['{"cash": 100, "cash": 1}', '{"x": NaN}',
                                  '{"x": Infinity}', '{"x": -Infinity}', '{"x": 1e999}',
                                  '{"observation": {"turn": 0, "turn": 1}}'])
def test_json_ambiguity_and_nonfinite_values_are_rejected(text):
    with pytest.raises(ValueError):
        strict_json_loads(text)


def test_existing_null_state_is_not_a_new_episode(tmp_path):
    state = tmp_path / "state.json"
    state.write_text("null")
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"observation": payload()}))
    args = build_parser().parse_args(["act", "--state", str(state), "--input", str(request)])
    with pytest.raises(ValueError, match="state"):
        execute(args)
    assert state.read_text() == "null"


@pytest.mark.parametrize("value", [[], None, "header", 3])
def test_nonobject_trace_header_rejected_without_attribute_error(tmp_path, value):
    trace = tmp_path / "trace.jsonl"
    trace.write_text(json.dumps(value) + "\n")
    with pytest.raises(ValueError, match="header"):
        replay(trace)


@pytest.mark.parametrize("section,key,value", [
    ("market", "initial_cash", 10 ** 1000),
    ("market", "turns", 10 ** 1000),
    ("market", "reserve", {"metal": 10 ** 1000, "plastic": 1}),
    ("market", "supply", {"metal": 10 ** 1000, "plastic": 1}),
    ("market", "initial_raw", {"metal": 10 ** 1000, "plastic": 1}),
    ("market", "recipes", {"widget": {"metal": 10 ** 1000}, "gadget": {"metal": 1}}),
    ("agent", "reward_low", -1e308),
    ("agent", "bid_reward_low", -1e308),
])
def test_unrepresentable_numeric_configuration_rejected(section, key, value):
    config = load_config()
    config[section][key] = value
    if key == "reward_low":
        config[section]["reward_high"] = 1e308
    if key == "bid_reward_low":
        config[section]["bid_reward_high"] = 1e308
    with pytest.raises(ValueError):
        validate_config(config)


def test_auction_resource_has_correct_type_and_phase():
    adapter = JsonAdapter(load_config())
    observation = payload()
    observation.update(phase="auction", resource=[])
    with pytest.raises(ValueError, match="resource"):
        adapter.decode_observation(observation)
    observation.update(phase="pricing", resource="metal")
    with pytest.raises(ValueError, match="phase"):
        adapter.decode_observation(observation)


@pytest.mark.parametrize("duplicate", [False, True])
def test_unmatched_or_duplicate_seeds_cannot_produce_paired_claim(duplicate):
    rows = [{"profile": "baseline", "split": "evaluation", "scenario": "stable", "seed": 1, "profit": 5},
            {"profile": "ucb", "split": "evaluation", "scenario": "stable", "seed": 1 if duplicate else 2, "profit": 7}]
    if duplicate:
        rows.append(deepcopy(rows[-1]))
    with pytest.raises(ValueError, match="matching seed"):
        paired_comparisons(rows)


def test_incompatible_roster_rejected_before_benchmark_output(tmp_path):
    out = tmp_path / "benchmark"
    args = build_parser().parse_args(["benchmark", "--out", str(out), "--opponents", "pacing"])
    with pytest.raises(ValueError, match="Pacing"):
        execute(args)
    assert not out.exists()


def test_persist_failure_records_failed_manifest(tmp_path, monkeypatch):
    config = load_config()
    config["market"]["turns"] = 1
    def failed_csv(*args):
        raise OSError("disk full during metrics persistence")
    monkeypatch.setattr("hackathon_kit.cli.csv_write", failed_csv)
    with pytest.raises(OSError, match="disk full"):
        save_run(config, "stable", 0, tmp_path / "run")
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert len(manifest["resolved_opponent_configs"]) == 3
    assert len(manifest["agent_seeds"]) == 4

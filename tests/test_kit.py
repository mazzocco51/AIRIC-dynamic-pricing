from copy import deepcopy
from dataclasses import asdict
import json
import subprocess
import sys

import numpy as np
import pytest

from hackathon_kit.agent import MarketAgent
from hackathon_kit.cli import replay, save_run, toml_text
from hackathon_kit.config import SCENARIOS, load_config, validate_config
from hackathon_kit.contracts import Action, Feedback, JsonAdapter, Observation
from hackathon_kit.learners import Learner
from hackathon_kit.simulator import resolve_auction, run_simulation


def observation(phase="pricing", identifier="0", **overrides):
    values = dict(decision_id=identifier, turn=0, phase=phase, cash=100.0,
                  raw={"metal": 5, "plastic": 5}, stock={"widget": 4, "gadget": 4},
                  raw_cost={"metal": 2.0, "plastic": 1.0}, stock_cost={"widget": 4.0, "gadget": 6.5}, capacity=5)
    values.update(overrides)
    return Observation(**values)


@pytest.mark.parametrize("kind,payment", [("first", 8), ("second", 5)])
def test_auction_payment(kind, payment):
    assert resolve_auction([3, 8, 5], 2, kind, 0.1) == (1, payment)
    assert resolve_auction([0, 0], 0, kind, 0.1) == (None, 0)
    assert resolve_auction([1, 0], 2, kind, 0.1) == (None, 0)
    assert resolve_auction([8, 8], 2, kind, 0.9) == (1, 8)


def test_second_price_single_bidder_reserve():
    assert resolve_auction([8, 0], 2, "second", 0) == (0, 2)


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("profile", ["baseline", "adaptive", "ucb", "exp3", "gp", "conversion"])
def test_simulation_constraints_accounting_and_privacy(profile, scenario):
    cfg = load_config(profile=profile)
    cfg["market"]["turns"] = 4
    records = []
    metrics, state = run_simulation(cfg, scenario, seed=42, emit=records.append)
    assert metrics["invalid_actions"] == 0
    assert metrics["min_cash"] >= 0
    assert metrics["final_equity"] == pytest.approx(metrics["initial_equity"] + metrics["profit"])
    assert not state["pending"]
    for row in records:
        if row["type"] == "decision":
            obs = row["observation"]
            assert not {"demand", "future", "opponents", "utilities"} & obs.keys()
            assert set(obs["raw"]) == {"metal", "plastic"}


@pytest.mark.parametrize("config", ["configs/first-price.toml", "configs/pacing.toml"])
def test_special_buying_modes(config):
    cfg = load_config(config)
    cfg["market"]["turns"] = 6
    metrics, _ = run_simulation(cfg, seed=5)
    if cfg["market"]["budget_mode"] == "fixed":
        assert metrics["procurement_spend"] <= cfg["market"]["fixed_budget"]


def test_milp():
    pytest.importorskip("scipy")
    cfg = load_config(profile="milp")
    cfg["market"]["turns"] = 3
    assert run_simulation(cfg)[0]["invalid_actions"] == 0


@pytest.mark.parametrize("updates", [
    {"buying": "pacing"}, {"buying": "learned"}, {"window": 2}, {"cash_fraction": 2},
    {"reward_high": -2000}, {"gp_noise": 0},
])
def test_incompatible_configuration(updates):
    cfg = load_config()
    cfg["agent"].update(updates)
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_ts_requires_counts():
    cfg = load_config(profile="conversion")
    cfg["market"]["conversion_feedback"] = False
    with pytest.raises(ValueError, match="successes"):
        validate_config(cfg)


def test_delayed_and_duplicate_feedback():
    agent = MarketAgent(load_config(profile="adaptive"))
    agent.act(observation(identifier="early"))
    agent.act(observation(identifier="late"))
    arm = agent.pending["early"]["learning"]["price:widget"]["arm"]
    feedback = Feedback("early", "pricing", rewards={"widget": -3.0, "gadget": 8.0})
    agent.observe(feedback)
    snapshot = agent.state_dict()
    agent.observe(feedback)
    assert agent.state_dict() == snapshot
    assert agent.learners["price:widget"].history[0][1] == arm
    assert "late" in agent.pending
    with pytest.raises(ValueError, match="unknown"):
        agent.observe(Feedback("missing", "pricing"))


def test_feedback_validation_is_atomic():
    agent = MarketAgent(load_config(profile="adaptive"))
    agent.act(observation())
    before = agent.state_dict()
    with pytest.raises(ValueError):
        agent.observe(Feedback("0", "pricing", rewards={"widget": 4}))
    assert agent.state_dict() == before


def test_restart_restores_rng_and_pending():
    cfg = load_config(profile="exp3")
    agent = MarketAgent(cfg, 19)
    agent.act(observation(identifier="pending"))
    clone = MarketAgent(cfg, 999)
    clone.load_state_dict(json.loads(json.dumps(agent.state_dict())))
    feedback = Feedback("pending", "pricing", rewards={"widget": -1.0, "gadget": 15.0})
    agent.observe(feedback)
    clone.observe(feedback)
    for i in range(8):
        obs = observation(identifier=str(i))
        assert agent.act(obs) == clone.act(obs)
    assert agent.state_dict() == clone.state_dict()


def test_censored_ts_does_not_learn_demand():
    learner = Learner("conversion_ts", [5, 8], load_config()["agent"])
    decision = learner.choose(np.random.default_rng(0))
    learner.update(decision, 5, censored=True)
    assert learner.history == []
    decision = learner.choose(np.random.default_rng(1))
    with pytest.raises(ValueError, match="successes"):
        learner.update(decision, 5)
    learner.update(decision, 5, successes=2, trials=10)
    assert learner.history[0][-2:] == [2, 10]


def test_sliding_window_expiration_and_negative_rewards():
    cfg = load_config()["agent"]
    cfg["window"] = 2
    learner = Learner("swucb", [5, 8], cfg, -10, 10)
    rng = np.random.default_rng(0)
    old = learner.choose(rng)
    learner.update(old, -5)
    assert learner.history[0][2] == 0.25
    for _ in range(3):
        learner.choose(rng)
    assert learner.history == []
    learner.update(old, 3)
    assert learner.history == []
    with pytest.warns(RuntimeWarning):
        assert learner.normalize(100) == 1


def test_exp3_numerical_stability():
    learner = Learner("exp3ix", [1, 2], load_config()["agent"])
    rng = np.random.default_rng(8)
    for _ in range(2000):
        decision = learner.choose(rng)
        learner.update(decision, -1000)
    assert np.all(np.isfinite(learner.log_weights))


@pytest.mark.parametrize("obs,action", [
    (observation(), Action("0", prices={"widget": 31, "gadget": 10})),
    (observation(), Action("0", prices={"widget": 7.001, "gadget": 10})),
    (observation("production"), Action("0", production={"widget": 6})),
    (observation("production", raw={"metal": 0, "plastic": 0}), Action("0", production={"widget": 1})),
    (observation("production", cash=0), Action("0", production={"widget": 1})),
    (observation("auction", resource="metal"), Action("0", bids={"metal": 101})),
])
def test_invalid_actions_rejected(obs, action):
    with pytest.raises(ValueError):
        JsonAdapter(load_config()).validate_action(obs, action)


def test_missing_observation_stays_unknown():
    obs = JsonAdapter(load_config()).decode_observation(asdict(observation()))
    assert obs.remaining_budget is None
    assert obs.resource is None
    assert obs.previous_prices == {}


def test_save_run_replay_and_reproducibility(tmp_path):
    cfg = load_config(profile="exp3")
    cfg["market"]["turns"] = 3
    metrics = save_run(cfg, "shock", 17, tmp_path / "run")
    result = replay(tmp_path / "run" / "trace.jsonl.gz")
    assert result["state_match"]
    again, _ = run_simulation(cfg, "shock", 17)
    assert {k: v for k, v in metrics.items() if not k.startswith("decision_ms")} == {
        k: v for k, v in again.items() if not k.startswith("decision_ms")}
    assert load_config(tmp_path / "run" / "config.toml") == cfg


def test_cli_act_clean_stdout_and_idempotent_restart(tmp_path):
    command = [sys.executable, "-m", "hackathon_kit", "act", "--input", "examples/observation.json",
               "--state", str(tmp_path / "agent.json")]
    first = subprocess.run(command, capture_output=True, text=True, check=True)
    second = subprocess.run(command, capture_output=True, text=True, check=True)
    assert json.loads(first.stdout) == json.loads(second.stdout)
    assert set(json.loads(first.stdout)) == {"decision_id", "bids", "production", "prices"}


def test_cli_benchmark_and_tune(tmp_path):
    base = [sys.executable, "-m", "hackathon_kit"]
    subprocess.run(base + ["benchmark", "--out", str(tmp_path / "bench"), "--turns", "2",
                   "--dev-seeds", "1", "--eval-seeds", "1", "--profiles", "baseline", "adaptive",
                   "--scenarios", "stable"], capture_output=True, check=True)
    manifest = json.loads((tmp_path / "bench" / "manifest.json").read_text())
    assert set(manifest["development_seeds"]).isdisjoint(manifest["evaluation_seeds"])
    subprocess.run(base + ["tune", "--out", str(tmp_path / "tune"), "--turns", "2",
                   "--dev-seeds", "1", "--windows", "10", "--cash-fractions", "0.4"], capture_output=True, check=True)
    assert load_config(tmp_path / "tune" / "selected.toml")["agent"]["window"] == 10


def test_custom_market_from_toml(tmp_path):
    config_path = tmp_path / "custom.toml"
    config_path.write_text('''
[market]
turns = 2
[market.reserve]
wood = 1.0
[market.supply]
wood = 6
[market.initial_raw]
wood = 2
[market.recipes.chair]
wood = 2
[market.production_cost]
chair = 0.5
''')
    cfg = load_config(config_path)
    assert set(cfg["market"]["recipes"]) == {"chair"}
    assert run_simulation(cfg)[0]["invalid_actions"] == 0


def test_opponents_do_not_inherit_tuned_parameters(monkeypatch):
    import hackathon_kit.simulator as simulator
    from hackathon_kit.config import DEFAULT
    configs = []
    original = simulator.MarketAgent
    def capture(config, seed):
        configs.append(deepcopy(config))
        return original(config, seed)
    monkeypatch.setattr(simulator, "MarketAgent", capture)
    cfg = load_config(profile="adaptive")
    cfg["agent"]["cash_fraction"] = 0.1
    cfg["market"]["turns"] = 1
    run_simulation(cfg)
    assert configs[0]["agent"]["cash_fraction"] == 0.1
    assert all(c["agent"]["cash_fraction"] == DEFAULT["agent"]["cash_fraction"] for c in configs[1:])


def test_exploratory_low_price_does_not_block_restocking():
    agent = MarketAgent(load_config(profile="adaptive"))
    agent.last_prices = {"widget": 5.0, "gadget": 5.0}
    obs = observation("auction", resource="metal", raw={"metal": 0, "plastic": 0},
                      stock={"widget": 0, "gadget": 0})
    assert agent.act(obs).bids["metal"] >= agent.m["reserve"]["metal"]


def test_only_realized_prices_inform_input_values():
    agent = MarketAgent(load_config(profile="exp3"), 2)
    action = agent.act(observation())
    assert agent.last_prices == {}
    agent.observe(Feedback("0", "pricing", rewards={"widget": 0, "gadget": 5}, sales={"widget": 0, "gadget": 1}))
    assert agent.last_prices == {"gadget": action.prices["gadget"]}

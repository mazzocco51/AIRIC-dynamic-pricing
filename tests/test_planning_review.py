from copy import deepcopy
from dataclasses import asdict
import json
import math
import numpy as np
import pytest

from hackathon_kit.agent import MarketAgent
from hackathon_kit.config import load_config
from hackathon_kit.contracts import Feedback, Observation, JsonAdapter
from hackathon_kit.planning import DemandModel, poisson_log_survival
from hackathon_kit.simulator import run_simulation


def obs(identifier="p", phase="pricing", **kwargs):
    args = dict(decision_id=identifier, turn=0, phase=phase, cash=100., capacity=5,
                stock={"widget": 2, "gadget": 2}, raw={"metal": 1, "plastic": 2},
                raw_cost={"metal": 2., "plastic": 1.}, stock_cost={"widget": 4., "gadget": 6.5})
    args.update(kwargs)
    return Observation(**args)


def test_poisson_tail_rare_events_distinguishable():
    values = poisson_log_survival(np.array([1.5, 3., 12., 24.]), 100)
    assert np.all(np.isfinite(values))
    assert np.all(np.diff(values) > 0)
    assert values[0] < -200
    scipy = pytest.importorskip("scipy.stats")
    np.testing.assert_allclose(values, scipy.poisson.logsf(99, [1.5, 3, 12, 24]), rtol=1e-12)
    np.testing.assert_allclose(poisson_log_survival(np.array([10000.]), 10000), scipy.poisson.logsf(9999, 10000), rtol=1e-10)


def test_censored_posterior_does_not_treat_stockout_as_demand():
    cfg = load_config(profile="competitive")
    censored = DemandModel(cfg["market"], cfg["agent"])
    exact = deepcopy(censored)
    censored.update(censored.anchor, None, 2, 2, True)
    exact.update(exact.anchor, None, 2, 3, False)
    assert censored.weights @ censored.levels > exact.weights @ exact.levels
    censored.update(censored.anchor, None, 100, 100, True)
    assert censored.weights @ censored.levels > 20


def test_expected_capped_revenue_analytic():
    cfg = load_config(profile="competitive")
    cfg["agent"]["demand_optimism"] = 0
    model = DemandModel(cfg["market"], cfg["agent"])
    revenue, _ = model.revenue_curve([model.anchor], 1, None)
    expected = model.anchor * (model.weights @ (1 - np.exp(-model.levels)))
    assert revenue[1] == pytest.approx(expected)


def test_competitive_dynamic_price_bounds():
    agent = MarketAgent(load_config(profile="competitive"))
    observation = obs(price_min=8.2, price_max=13.4)
    action = agent.act(observation)
    assert all(8.2 <= p <= 13.4 for p in action.prices.values())
    JsonAdapter(agent.config).validate_action(observation, action)


def test_changed_id_context_or_feedback_rejected():
    agent = MarketAgent(load_config(profile="competitive"))
    agent.act(obs())
    with pytest.raises(ValueError, match="different observation"):
        agent.act(obs(cash=99))
    feedback = Feedback("p", "pricing", sales={"widget": 1, "gadget": 2}, censored={"widget": False, "gadget": True})
    agent.observe(feedback)
    agent.observe(feedback)
    feedback.sales["widget"] = 0
    with pytest.raises(ValueError, match="Conflicting"):
        agent.observe(feedback)


def test_invalid_state_and_feedback_are_atomic():
    cfg = load_config(profile="competitive")
    agent = MarketAgent(cfg)
    agent.act(obs())
    before = agent.state_dict()
    invalid = deepcopy(before)
    invalid["models"]["widget"]["weights"][0] = float("nan")
    with pytest.raises(ValueError):
        agent.load_state_dict(invalid)
    assert agent.state_dict() == before
    with pytest.raises(ValueError):
        agent.observe(Feedback("p", "pricing", sales={"widget": 1, "gadget": 3}, censored={"widget": False, "gadget": True}))
    assert agent.state_dict() == before


def test_competitive_delayed_feedback_and_restart():
    cfg = load_config(profile="competitive")
    agent = MarketAgent(cfg)
    agent.act(obs("old"))
    agent.act(obs("new", turn=1))
    cloned = MarketAgent(cfg)
    cloned.load_state_dict(json.loads(json.dumps(agent.state_dict())))
    for identity in ("new", "old"):
        feedback = Feedback(identity, "pricing", sales={"widget": 2, "gadget": 0}, censored={"widget": True, "gadget": False})
        agent.observe(feedback)
        cloned.observe(feedback)
    assert agent.act(obs("next", turn=2)) == cloned.act(obs("next", turn=2))
    assert agent.state_dict() == cloned.state_dict()


@pytest.mark.parametrize("scenario", ["stable", "shock", "scarcity", "aggressive"])
@pytest.mark.parametrize("auction", ["first", "second"])
def test_competitive_completes_constraints(scenario, auction):
    cfg = load_config(profile="competitive")
    cfg["market"].update(turns=6, auction=auction)
    metrics, _ = run_simulation(cfg, scenario, 47)
    assert metrics["invalid_actions"] == 0
    assert metrics["final_equity"] == pytest.approx(metrics["initial_equity"] + metrics["profit"])


def test_low_cash_preserves_feasibility():
    cfg = load_config(profile="competitive")
    agent = MarketAgent(cfg)
    observation = obs("auction", "auction", cash=0.5, resource="metal", raw={"metal": 0, "plastic": 0})
    assert agent.act(observation).bids["metal"] == 0


def test_baseline_profile_clears_specialized_modes(tmp_path):
    from hackathon_kit.cli import toml_text
    path = tmp_path / "competitive.toml"
    path.write_text(toml_text(load_config(profile="competitive")))
    cfg = load_config(path, "baseline")
    assert [cfg["agent"][k] for k in ["pricing", "buying", "production"]] == ["fixed", "prudent", "greedy"]

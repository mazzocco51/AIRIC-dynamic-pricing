from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest

from hackathon_kit.config import load_config
from hackathon_kit.learners import Learner
from hackathon_kit.planning import DemandModel, joint_plan


def test_high_rate_capped_sales_recovers_probability_mass():
    cfg = load_config(profile="competitive")
    cfg["agent"]["demand_optimism"] = 0.
    model = DemandModel(cfg["market"], cfg["agent"])
    model.rates = lambda prices, competitor: np.full((25, 1), 1000.)
    revenues, _ = model.revenue_curve([10.], 2000, None)
    assert revenues[-1] == pytest.approx(10000., rel=1e-8)
    assert revenues[1000] < 10000.
    assert np.all(np.diff(revenues) >= -1e-10)


def test_small_beam_reserves_capacity_for_later_valuable_product():
    cfg = load_config(profile="competitive")
    market, settings = cfg["market"], cfg["agent"]
    settings["planner_limit"] = 1
    obs = SimpleNamespace(capacity=1, stock={"gadget": 0, "widget": 0},
                          raw={"metal": 5, "plastic": 5}, raw_cost={"metal": 0., "plastic": 0.},
                          cash=100., remaining_budget=None)
    curves = {"gadget": ([0., 5.], [10., 10.]), "widget": ([0., 100.], [20., 20.])}
    plan = joint_plan(obs, market, settings, curves, market["reserve"], purchasing=False)
    assert plan["quantities"] == {"gadget": 0, "widget": 1}
    assert not plan["exact"] and plan["pruned_states"] > 0


def test_planner_memory_depends_on_feasible_not_nominal_capacity():
    cfg = load_config(profile="competitive")
    obs = SimpleNamespace(capacity=10**9, stock={"gadget": 0, "widget": 0},
                          raw={"metal": 5, "plastic": 5}, raw_cost={"metal": 2., "plastic": 1.},
                          cash=100., remaining_budget=None)
    curves = {p: ([0., 10., 20., 30.], [10.] * 4) for p in obs.stock}
    plan = joint_plan(obs, cfg["market"], cfg["agent"], curves, cfg["market"]["reserve"], purchasing=False)
    assert sum(plan["quantities"].values()) <= 6


def test_exp3_overflow_rejected_without_partial_update():
    settings = deepcopy(load_config()["agent"])
    settings.update(eta=1e308, gamma=1e-300)
    learner = Learner("exp3ix", [1., 2.], settings)
    decision = learner.choose(np.random.default_rng(1))
    decision["probability"] = 1e-300
    before = learner.state_dict()
    with pytest.raises(ValueError, match="overflow"):
        learner.update(decision, settings["reward_low"])
    assert learner.state_dict() == before


def test_checkpoint_rejects_colliding_pending_identifiers():
    learner = Learner("ucb", [1., 2.], load_config()["agent"])
    learner.choose(np.random.default_rng(2))
    state = learner.state_dict()
    state["pending"]["00"] = state["pending"]["0"]
    with pytest.raises(ValueError):
        learner.load_state_dict(state)

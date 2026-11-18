from copy import deepcopy
import json
import pytest

from hackathon_kit.agent import MarketAgent
from hackathon_kit.config import load_config
from hackathon_kit.contracts import Observation, JsonAdapter
from hackathon_kit.prices import round_to_tick, tick_interval, is_tick_aligned


def observation(phase="pricing", **extra):
    fields = dict(decision_id="test", turn=0, phase=phase, cash=100., capacity=5,
                  raw={"metal": 0, "plastic": 0}, stock={"widget": 2, "gadget": 2},
                  raw_cost={"metal": 2., "plastic": 1.}, stock_cost={"widget": 4., "gadget": 6.5})
    fields.update(extra)
    return Observation(**fields)


@pytest.mark.parametrize("profile", ["exp3", "ucb", "adaptive", "gp", "conversion"])
def test_failed_second_choice_rolls_back_first_choice_and_randomness(profile, monkeypatch):
    agent = MarketAgent(load_config(profile=profile), 81)
    original = agent.state_dict()
    key = list(agent.learners)[1]
    def fail(*args):
        raise ValueError("injected second product failure")
    monkeypatch.setattr(agent.learners[key], "choose", fail)
    with pytest.raises(ValueError, match="injected"):
        agent.act(observation())
    assert agent.state_dict() == original


@pytest.mark.parametrize("mutator", [
    lambda s: s["learners"]["price:widget"].update(pending={}),
    lambda s: s["pending"]["test"]["learning"]["price:widget"].update(arm=99),
    lambda s: s["pending"]["test"].update(offered={}),
    lambda s: s["pending"].update(test=None),
    lambda s: s["actions"]["test"]["prices"].update(widget=float("nan")),
    lambda s: s["observations"].update(test="bad hash"),
])
def test_inconsistent_checkpoint_is_rejected_atomically(mutator):
    agent = MarketAgent(load_config(profile="exp3"))
    agent.act(observation())
    before = agent.state_dict()
    broken = deepcopy(before)
    mutator(broken)
    with pytest.raises(ValueError):
        agent.load_state_dict(broken)
    assert agent.state_dict() == before


def test_valid_checkpoint_keeps_pending_cross_references():
    agent = MarketAgent(load_config(profile="competitive"))
    agent.act(observation())
    restored = MarketAgent(agent.config)
    restored.load_state_dict(json.loads(json.dumps(agent.state_dict())))
    assert restored.state_dict() == agent.state_dict()


@pytest.mark.parametrize("tick,low,high,price", [
    (1e-12, 1e-12, 9e-12, 4.5e-12), (0.05, 0.1, 0.3, 0.299),
    (0.07, 0.14, 0.28, 0.27), (1e-11, 5.0, 6.0, 5.12345678901),
])
def test_price_tick_precision(tick, low, high, price):
    result = round_to_tick(price, low, high, tick)
    assert low <= result <= high
    assert is_tick_aligned(result, tick)


def test_tick_interval_does_not_expand_small_bounds_with_absolute_epsilon():
    with pytest.raises(ValueError, match="No legal"):
        tick_interval(1.1e-12, 1.9e-12, 1e-12)


def test_tiny_tick_agent_action_is_legal():
    config = load_config()
    config["market"].update(price_min=1e-12, price_max=9e-12, price_tick=1e-12)
    agent = MarketAgent(config)
    obs = observation()
    action = agent.act(obs)
    JsonAdapter(config).validate_action(obs, action)


@pytest.mark.parametrize("fixed_budget", [None, 20.0])
def test_planned_bid_reserves_remaining_units_of_the_same_material(monkeypatch, fixed_budget):
    cfg = load_config(profile="competitive")
    cfg["agent"].update(cash_fraction=1., bid_value_fraction=1.)
    agent = MarketAgent(cfg)
    plan = {"requirements": {"metal": 3, "plastic": 0}, "score": 91., "processing": 3., "revenue": 100.}
    monkeypatch.setattr("hackathon_kit.agent.joint_plan", lambda *args, **kwargs: plan)
    obs = observation("auction", cash=20., resource="metal", remaining_budget=fixed_budget)
    cap = agent.planned_bid(JsonAdapter(cfg).decode_observation(__import__('dataclasses').asdict(obs)))
    # 3 production + 2 further metal units at 2 each must still be affordable.
    assert cap + 3 + 2 * 2 <= 20

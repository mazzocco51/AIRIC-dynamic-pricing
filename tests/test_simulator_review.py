"""Physics, accounting and cross-play regressions independent of strategy quality."""
from copy import deepcopy
import math

import pytest

from hackathon_kit.config import load_config
from hackathon_kit.contracts import Action
from hackathon_kit.simulator import resolve_auction, run_simulation


@pytest.mark.parametrize("bids,reserve,draw", [
    ([], 1, 0), ([math.nan, 2], 1, 0), ([math.inf, 2], 1, 0),
    ([-1, 2], 1, 0), ([True, 2], 1, 0), ([1, 2], math.nan, 0),
    ([1, 2], -1, 0), ([1, 2], 1, -0.01), ([1, 2], 1, 1),
])
def test_auction_rejects_malformed_inputs(bids, reserve, draw):
    with pytest.raises(ValueError):
        resolve_auction(bids, reserve, "second", draw)


def test_auction_tie_partitions_are_symmetric():
    winners = [resolve_auction([4, 4, 4], 1, "second", (i + 0.5) / 300)[0] for i in range(300)]
    assert [winners.count(i) for i in range(3)] == [100, 100, 100]
    assert resolve_auction([0, 0], 0, "second", 0.5) == (None, 0)


def test_custom_crossplay_roster_is_recorded_and_private():
    cfg = load_config(profile="baseline")
    cfg["market"]["turns"] = 3
    opponents = []
    for pricing in ("gpucb", "exp3ix"):
        opponent = deepcopy(cfg)
        opponent["agent"]["pricing"] = pricing
        opponents.append(opponent)
    events = []
    metrics, _ = run_simulation(cfg, seed=42, opponent_configs=opponents, emit=events.append)
    assert events[0]["opponent_configs"] == opponents
    assert len(events[0]["agent_seeds"]) == 3
    assert len(events[-1]["players"]) == 3
    assert 1 <= metrics["rank"] <= 3
    assert 0 <= metrics["win_share"] <= 1
    for event in events:
        if event["type"] == "decision":
            assert "opponent_configs" not in event["observation"]
            assert "players" not in event["observation"]
    assert cfg["agent"]["pricing"] == "fixed"  # Input configurations remain untouched.


def test_crossplay_requires_the_same_market():
    cfg = load_config()
    opponent = deepcopy(cfg)
    opponent["market"]["initial_cash"] += 1
    with pytest.raises(ValueError, match="markets must match"):
        run_simulation(cfg, opponent_configs=[opponent])
    with pytest.raises(ValueError, match="at least one"):
        run_simulation(cfg, opponent_configs=[])


def test_resource_dictionary_order_does_not_change_the_market():
    cfg = load_config(profile="exp3")
    cfg["market"]["turns"] = 4
    reversed_cfg = deepcopy(cfg)
    reversed_cfg["market"]["supply"] = dict(reversed(list(cfg["market"]["supply"].items())))
    first, _ = run_simulation(cfg, seed=9)
    second, _ = run_simulation(reversed_cfg, seed=9)
    assert {k: v for k, v in first.items() if not k.startswith("decision_ms")} == {
        k: v for k, v in second.items() if not k.startswith("decision_ms")}


def test_tracing_does_not_change_simulation_outcomes():
    cfg = load_config(profile="exp3")
    cfg["market"]["turns"] = 4
    trace = []
    traced, first_state = run_simulation(cfg, seed=11, emit=trace.append)
    untraced, second_state = run_simulation(cfg, seed=11)
    assert first_state == second_state
    assert {k: v for k, v in traced.items() if not k.startswith("decision_ms")} == {
        k: v for k, v in untraced.items() if not k.startswith("decision_ms")}


def test_storage_limited_liability_is_explicit_and_accounted(monkeypatch):
    import hackathon_kit.simulator as simulator

    class InactiveAgent:
        def __init__(self, config, seed):
            self.config = config

        def act(self, obs):
            if obs.phase == "auction":
                return Action(obs.decision_id, bids={obs.resource: 0.0})
            if obs.phase == "production":
                return Action(obs.decision_id)
            return Action(obs.decision_id, prices={p: 30.0 for p in obs.stock})

        def observe(self, feedback):
            pass

        def state_dict(self):
            return {}

    monkeypatch.setattr(simulator, "MarketAgent", InactiveAgent)
    cfg = load_config()
    cfg["market"].update(turns=2, holding_cost=100.0)
    events = []
    metrics, _ = run_simulation(cfg, emit=events.append)
    assert metrics["final_cash"] == 0
    assert metrics["profit"] == -150
    assert metrics["storage_paid"] == 150
    assert metrics["storage_waived"] == 650
    assert metrics["final_equity"] == 6
    assert metrics["rank"] == 1
    assert metrics["win_share"] == 0.25
    assert all(player["final_equity"] == 6 for player in events[-1]["players"])

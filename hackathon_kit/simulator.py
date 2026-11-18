"""Toy market: public observations never contain demand draws or rival inventories."""
from copy import deepcopy
from dataclasses import asdict
import math
from numbers import Real
import time
import numpy as np

from .agent import MarketAgent
from .config import DEFAULT, SCENARIOS, validate_config
from .contracts import Feedback, JsonAdapter, Observation


def resolve_auction(bids, reserve, kind, tie_draw):
    """Resolve one unit; zero is the explicit abstention bid, even at reserve zero."""
    if kind not in ("first", "second"):
        raise ValueError("Unknown auction")
    bids = list(bids)
    def finite_number(value):
        return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
    if not bids or any(not finite_number(bid) or bid < 0 for bid in bids):
        raise ValueError("Auction bids must be finite, nonnegative real numbers")
    if not finite_number(reserve) or reserve < 0:
        raise ValueError("Auction reserve must be finite and nonnegative")
    if not finite_number(tie_draw) or not 0 <= tie_draw < 1:
        raise ValueError("Tie draw must be in [0, 1)")
    eligible = [i for i, bid in enumerate(bids) if bid > 0 and bid >= reserve]
    if not eligible:
        return None, 0.0
    best = max(bids[i] for i in eligible)
    tied = [i for i in eligible if bids[i] == best]
    winner = tied[min(len(tied) - 1, int(tie_draw * len(tied)))]
    payment = best if kind == "first" else max(reserve, max((b for i, b in enumerate(bids) if i != winner), default=0))
    return winner, float(payment)


def inventory_add(player, kind, item, quantity, unit_cost):
    cost_key = "raw_cost" if kind == "raw" else "stock_cost"
    old = player[kind][item]
    if quantity:
        player[cost_key][item] = (old * player[cost_key][item] + quantity * unit_cost) / (old + quantity)
        player[kind][item] += quantity


def equity(player):
    return player["cash"] + sum(q * player["raw_cost"][r] for r, q in player["raw"].items()) + sum(
        q * player["stock_cost"][p] for p, q in player["stock"].items())


def resolve_opponent_configs(config, scenario, opponent_configs=None):
    """Resolve a complete, independent roster for both simulation and study manifests."""
    if scenario not in SCENARIOS:
        raise ValueError("Unknown scenario")
    validate_config(config)
    if opponent_configs is None:
        opponents = []
        for kind in ("fixed", "random", "reactive"):
            opponent = deepcopy(config)
            opponent["agent"] = deepcopy(DEFAULT["agent"])
            opponent["agent"].update(pricing=kind, buying="prudent", production="greedy")
            if scenario == "aggressive":
                opponent["agent"].update(margin=0.2, bid_value_fraction=1.0, target_stock=5)
            opponents.append(opponent)
    else:
        if not isinstance(opponent_configs, (list, tuple)) or not opponent_configs:
            raise ValueError("Supply at least one opponent configuration")
        opponents = deepcopy(list(opponent_configs))
    for opponent in opponents:
        validate_config(opponent)
        if opponent["market"] != config["market"]:
            raise ValueError("Opponents may override agent settings only; markets must match")
    return opponents


def run_simulation(config, scenario="stable", seed=0, emit=None, opponent_configs=None):
    """Run a market episode with optional agent-only opponent configurations.

    Each supplied configuration must describe the same market. The focal agent
    remains player zero; per-player outcomes are written only to the final trace,
    never to observations. Omitting opponents preserves the demonstration roster.
    """
    if scenario not in SCENARIOS:
        raise ValueError("Unknown scenario")
    validate_config(config)
    if type(seed) is not int or seed < 0:
        raise ValueError("Simulation seed must be a nonnegative integer")
    config = deepcopy(config)
    m = config["market"]
    resources, products = sorted(m["reserve"]), sorted(m["recipes"])
    adapter = JsonAdapter(config)
    # Exogenous draws are independent from strategy random generators.
    auction_rng, demand_rng = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2)]
    opponents = resolve_opponent_configs(config, scenario, opponent_configs)
    agent_seeds = [seed + 10000 * (i + 1) for i in range(1 + len(opponents))]
    agents = [MarketAgent(c, agent_seed) for c, agent_seed in zip([config] + opponents, agent_seeds)]
    players = []
    for _ in agents:
        players.append({"cash": m["initial_cash"], "raw": {r: m["initial_raw"].get(r, 0) for r in resources},
                        "stock": {p: 0 for p in products}, "raw_cost": dict(m["reserve"]),
                        "stock_cost": {p: 0.0 for p in products}, "spent": 0.0,
                        "profit": 0.0, "min_cash": m["initial_cash"], "sold": 0,
                        "storage_paid": 0.0, "storage_waived": 0.0})
    previous = [{} for _ in agents]
    initial_equity = equity(players[0])
    times = []
    if emit is not None:
        emit({"type": "header", "schema": 1, "config": deepcopy(config), "scenario": scenario, "seed": seed,
              "opponent_configs": deepcopy(opponents), "agent_seeds": list(agent_seeds),
              "market_model": "multinomial_logit_no_stockout_substitution_v1",
              "initial_state": agents[0].state_dict()})

    def observe_player(i, turn, phase, suffix, resource=None):
        player = players[i]
        obs = Observation(
            decision_id=f"{turn}:{phase}:{suffix}:{i}", turn=turn, phase=phase,
            cash=player["cash"], raw=dict(player["raw"]), stock=dict(player["stock"]),
            capacity=m["capacity"], raw_cost=dict(player["raw_cost"]), stock_cost=dict(player["stock_cost"]),
            previous_prices=dict(previous[i]), resource=resource,
            remaining_budget=max(0.0, m["fixed_budget"] - player["spent"]) if m["budget_mode"] == "fixed" else None,
            price_min=m["price_min"], price_max=m["price_max"], bid_max=player["cash"])
        # Round-trip through the same public adapter used by the CLI.
        return adapter.decode_observation(asdict(obs))

    def decide(i, obs):
        start = time.perf_counter()
        action = agents[i].act(obs)
        elapsed = time.perf_counter() - start
        adapter.validate_action(obs, action)
        if i == 0:
            times.append(elapsed)
            if emit is not None:
                emit({"type": "decision", "observation": asdict(obs), "action": asdict(action)})
        return action

    def deliver(i, feedback):
        agents[i].observe(feedback)
        if i == 0 and emit is not None:
            emit({"type": "feedback", "feedback": asdict(feedback)})

    def audit():
        for player in players:
            if not all(math.isfinite(value) for value in (player["cash"], player["profit"], equity(player))):
                raise AssertionError("Non-finite accounting value")
            # Arithmetic at an exact budget boundary can leave a tiny negative
            # residue; it is not credit and must not leak into the next observation.
            if -1e-9 <= player["cash"] < 0:
                player["cash"] = 0.0
            player["min_cash"] = min(player["min_cash"], player["cash"])
            if player["cash"] < -1e-7 or any(q < 0 for q in (*player["raw"].values(), *player["stock"].values())):
                raise AssertionError("Negative cash/inventory")
            if abs(equity(player) - initial_equity - player["profit"]) > 1e-6:
                raise AssertionError("Accounting identity violated")
            if m["budget_mode"] == "fixed" and player["spent"] > m["fixed_budget"] + 1e-7:
                raise AssertionError("Fixed purchasing budget exceeded")

    for turn in range(m["turns"]):
        for resource in resources:
            count = m["supply"][resource]
            supply = max(1, count // 2) if scenario == "scarcity" else count
            for unit in range(supply):
                observations = [observe_player(i, turn, "auction", f"{resource}:{unit}", resource) for i in range(len(agents))]
                actions = [decide(i, obs) for i, obs in enumerate(observations)]
                bids = [action.bids[resource] for action in actions]
                winner, payment = resolve_auction(bids, m["reserve"][resource], m["auction"], auction_rng.random())
                if winner is not None:
                    player = players[winner]
                    player["cash"] -= payment
                    player["spent"] += payment
                    inventory_add(player, "raw", resource, 1, payment)
                for i, action in enumerate(actions):
                    deliver(i, Feedback(action.decision_id, "auction", costs={resource: payment if winner == i else 0.0}, won=winner == i))
                audit()
        observations = [observe_player(i, turn, "production", "all") for i in range(len(agents))]
        actions = [decide(i, obs) for i, obs in enumerate(observations)]
        for i, action in enumerate(actions):
            player = players[i]
            costs = {}
            for product, q in action.production.items():
                recipe = m["recipes"][product]
                basis = m["production_cost"][product] + sum(n * player["raw_cost"][r] for r, n in recipe.items())
                for r, n in recipe.items():
                    player["raw"][r] -= n * q
                costs[product] = m["production_cost"][product] * q
                player["cash"] -= costs[product]
                inventory_add(player, "stock", product, q, basis)
            deliver(i, Feedback(action.decision_id, "production", costs=costs))
        audit()
        observations = [observe_player(i, turn, "pricing", "all") for i in range(len(agents))]
        actions = [decide(i, obs) for i, obs in enumerate(observations)]
        feedbacks = [Feedback(a.decision_id, "pricing") for a in actions]
        for product in products:
            prices = np.array([a.prices[product] for a in actions])
            sensitivity = 0.22 if scenario == "shock" and turn >= m["turns"] // 2 else 0.12
            utilities = np.exp(2.0 - sensitivity * prices)
            # Demand is drawn before stock checks; unmet demand goes to the outside option.
            probabilities = np.append(utilities, 1.0)
            probabilities /= probabilities.sum()
            choices = np.searchsorted(np.cumsum(probabilities), demand_rng.random(m["customers"]))
            for i, player in enumerate(players):
                demand = int(np.sum(choices == i))
                available = player["stock"][product]
                sold = min(available, demand)
                revenue = sold * prices[i]
                profit = revenue - sold * player["stock_cost"][product]
                player["cash"] += revenue
                player["stock"][product] -= sold
                player["profit"] += profit
                player["sold"] += sold
                feedbacks[i].sales[product] = sold
                feedbacks[i].rewards[product] = float(profit)
                feedbacks[i].costs[product] = sold * player["stock_cost"][product]
                # Conservatively mark equality too: an observer cannot infer unmet demand.
                feedbacks[i].censored[product] = sold == available
                if m["conversion_feedback"] and sold < available:
                    feedbacks[i].successes[product] = sold
                    feedbacks[i].trials[product] = m["customers"]
        for i, player in enumerate(players):
            # Limited-liability toy rule: storage fee cannot overdraw cash.
            requested_fee = m["holding_cost"] * (sum(player["raw"].values()) + sum(player["stock"].values()))
            fee = min(player["cash"], requested_fee)
            player["cash"] -= fee
            player["profit"] -= fee
            player["storage_paid"] += fee
            player["storage_waived"] += requested_fee - fee
            deliver(i, feedbacks[i])
            previous[i] = {p: float(np.mean([a.prices[p] for j, a in enumerate(actions) if j != i])) for p in products}
        audit()
    player = players[0]
    profits = [float(p["profit"]) for p in players]
    best_profit = max(profits)
    leaders = [i for i, profit in enumerate(profits) if math.isclose(profit, best_profit, rel_tol=0, abs_tol=1e-8)]
    metrics = {"scenario": scenario, "seed": seed, "profit": float(player["profit"]),
               "initial_equity": float(initial_equity),
               "final_equity": float(equity(player)), "final_cash": float(player["cash"]),
               "min_cash": float(player["min_cash"]), "raw_remaining": sum(player["raw"].values()),
               "stock_remaining": sum(player["stock"].values()), "units_sold": player["sold"],
               "procurement_spend": player["spent"], "invalid_actions": 0,
               "rank": 1 + sum(profit > profits[0] + 1e-8 for profit in profits[1:]),
               "win_share": 1.0 / len(leaders) if 0 in leaders else 0.0,
               "storage_paid": player["storage_paid"], "storage_waived": player["storage_waived"],
               "decision_ms_mean": float(np.mean(times) * 1000),
               "decision_ms_max": float(np.max(times) * 1000)}
    state = agents[0].state_dict()
    if emit is not None:
        per_player = [{"player": i, "profit": float(p["profit"]), "final_equity": float(equity(p)),
                       "final_cash": float(p["cash"]), "units_sold": p["sold"],
                       "raw": dict(p["raw"]), "stock": dict(p["stock"])} for i, p in enumerate(players)]
        emit({"type": "final", "metrics": dict(metrics), "players": per_player, "state": deepcopy(state)})
    return metrics, state

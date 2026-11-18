from copy import deepcopy
from dataclasses import asdict
import math
import hashlib
import json
import numpy as np

from .config import validate_config
from .contracts import Action, Feedback, JsonAdapter
from .learners import Learner
from .planning import DemandModel, joint_plan
from .prices import round_to_tick


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


class MarketAgent:
    def __init__(self, config, seed=0):
        validate_config(config)
        self.config = deepcopy(config)
        self.m, self.a = self.config["market"], self.config["agent"]
        self.adapter = JsonAdapter(self.config)
        self.reset(seed)

    def reset(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.pending = {}
        self.completed = set()
        self.actions = {}
        self.observations = {}
        self.feedback_digests = {}
        self.learners = {}
        self.pacing_lambda = 1.0
        self.last_prices = {}
        self.price_turns = {}
        self.models = {p: DemandModel(self.m, self.a) for p in self.m["recipes"]} if self.a["pricing"] == "inventory" else {}
        self.input_prices = dict(self.m["reserve"])
        self.curve_cache = {}
        if self.a["pricing"] in ("ucb", "swucb", "exp3ix", "gpucb", "conversion_ts"):
            grid = self.price_grid()
            for p in self.m["recipes"]:
                self.learners[f"price:{p}"] = Learner(self.a["pricing"], grid, self.a)
        if self.a["buying"] == "learned":
            for r in self.m["reserve"]:
                self.learners[f"bid:{r}"] = Learner("exp3ix", [0.0, 0.4, 0.6, 0.8, 1.0], self.a,
                                                   self.a["bid_reward_low"], self.a["bid_reward_high"])

    def price_grid(self):
        return sorted(set(self.round_price(float(x)) for x in np.linspace(
            self.m["price_min"], self.m["price_max"], self.a["price_levels"])))

    def round_price(self, price, minimum=None, maximum=None):
        tick = self.m["price_tick"]
        lo = self.m["price_min"] if minimum is None else minimum
        hi = self.m["price_max"] if maximum is None else maximum
        return round_to_tick(price, lo, hi, tick)

    def estimated_cost(self, p, obs):
        if obs.stock[p] > 0:
            return obs.stock_cost[p]
        return self.m["production_cost"][p] + sum(
            q * obs.raw_cost[r] for r, q in self.m["recipes"][p].items())

    def expected_price(self, p, obs):
        baseline = self.round_price(self.estimated_cost(p, obs) * (1 + self.a["margin"]))
        # One exploratory low selling price must not permanently stop restocking.
        return max(baseline, self.last_prices.get(p, baseline))

    def resource_value(self, resource, obs):
        # Allocate product contribution in proportion to replacement costs, so
        # expensive complementary inputs are not valued like cheap ones.
        values = []
        for p, recipe in self.m["recipes"].items():
            if resource not in recipe:
                continue
            total = sum(q * max(obs.raw_cost[r], self.m["reserve"][r]) for r, q in recipe.items())
            weight = max(obs.raw_cost[resource], self.m["reserve"][resource]) / total if total > 0 else 1 / sum(recipe.values())
            values.append((self.expected_price(p, obs) - self.m["production_cost"][p]) * weight)
        return max(0.0, max(values, default=0.0))

    def act(self, obs):
        # Failed decisions must not consume randomness or exploration slots.
        phase = getattr(obs, "phase", None)
        relevant = [key for key in self.learners if (phase == "pricing" and key.startswith("price:"))
                    or (phase == "auction" and key == f"bid:{getattr(obs, 'resource', None)}")]
        snapshots = {key: self.learners[key].state_dict() for key in relevant}
        rng_state = deepcopy(self.rng.bit_generator.state)
        try:
            return self._act(obs)
        except Exception:
            self.rng.bit_generator.state = rng_state
            for key, state in snapshots.items():
                self.learners[key].load_state_dict(state)
            self.curve_cache.clear()
            raise

    def _act(self, obs):
        obs = self.adapter.decode_observation(asdict(obs))
        observation_digest = fingerprint(asdict(obs))
        if obs.decision_id in self.actions:
            if self.observations[obs.decision_id] != observation_digest:
                raise ValueError("Decision ID reused with a different observation")
            return Action(**deepcopy(self.actions[obs.decision_id]))
        if obs.phase == "pricing":
            for key, learner in self.learners.items():
                if key.startswith("price:") and any(p < obs.price_min or p > obs.price_max for p in learner.arms):
                    raise ValueError("Observed price bounds changed: update grid/config and reset agent")
        action = Action(obs.decision_id)
        pending = {"phase": obs.phase, "learning": {}, "resource": obs.resource,
                   "offered": dict(obs.stock), "previous_prices": dict(obs.previous_prices), "turn": obs.turn}
        if obs.phase == "auction":
            resource = obs.resource
            if self.a["buying"] == "planned":
                action.bids[resource] = self.planned_bid(obs)
                pending["value"] = action.bids[resource]
                self.adapter.validate_action(obs, action)
                self.pending[obs.decision_id] = pending
                self.actions[obs.decision_id] = asdict(action)
                self.observations[obs.decision_id] = observation_digest
                return action
            need = sum(max(0, self.a["target_stock"] - obs.stock[p]) * recipe.get(resource, 0)
                       for p, recipe in self.m["recipes"].items()) - obs.raw[resource]
            value = self.resource_value(resource, obs)
            limit = min(obs.cash * self.a["cash_fraction"], value * self.a["bid_value_fraction"],
                        obs.bid_max if obs.bid_max is not None else obs.cash,
                        obs.remaining_budget if obs.remaining_budget is not None else obs.cash)
            bid = max(0.0, limit) if need > 0 else 0.0
            if self.a["buying"] == "pacing":
                bid = min(bid, value / (1 + self.pacing_lambda))
            elif self.a["buying"] == "learned" and need > 0 and limit > 0:
                key = f"bid:{resource}"
                learner = self.learners[key]
                decision = learner.choose(self.rng)
                bid *= learner.arms[decision["arm"]]
                pending["learning"][key] = decision
            pending["value"] = value
            action.bids[resource] = float(bid)
        elif obs.phase == "production":
            action.production = self.plan_production(obs)
        else:
            for product in self.m["recipes"]:
                key = f"price:{product}"
                if self.a["pricing"] == "inventory":
                    curves = self.demand_curves(obs)
                    price = curves[product][1][obs.stock[product]]
                elif key in self.learners:
                    learner = self.learners[key]
                    # Changed bounds require rebuilding the configured action grid.
                    if any(p < obs.price_min or p > obs.price_max for p in learner.arms):
                        raise ValueError("Observed price bounds changed: update grid/config and reset agent")
                    if obs.stock[product] > 0:
                        decision = learner.choose(self.rng, self.estimated_cost(product, obs))
                        price = learner.arms[decision["arm"]]
                        pending["learning"][key] = decision
                    else:
                        price = self.expected_price(product, obs)
                elif self.a["pricing"] == "random":
                    price = self.rng.uniform(obs.price_min, obs.price_max)
                elif self.a["pricing"] == "reactive":
                    cost = self.estimated_cost(product, obs)
                    previous = obs.previous_prices.get(product)
                    price = max(cost * 1.1, previous * 0.95) if previous is not None else cost * 1.4
                else:
                    price = self.estimated_cost(product, obs) * (1 + self.a["margin"])
                action.prices[product] = self.round_price(price, obs.price_min, obs.price_max)
        self.adapter.validate_action(obs, action)
        self.pending[obs.decision_id] = pending
        self.actions[obs.decision_id] = asdict(action)
        self.observations[obs.decision_id] = observation_digest
        return action

    def demand_curves(self, obs):
        curves = {}
        grid = sorted(set(self.round_price(float(p), obs.price_min, obs.price_max)
                          for p in np.linspace(obs.price_min, obs.price_max, self.a["price_levels"])))
        for product, model in self.models.items():
            maximum = obs.stock[product] + min(obs.capacity, self.a["target_stock"])
            # Memory is bounded to one cached curve per product.
            key = (model.updates, maximum, obs.previous_prices.get(product), tuple(grid))
            previous = self.curve_cache.get(product)
            if previous is None or previous[0] != key:
                previous = (key, model.revenue_curve(grid, maximum, obs.previous_prices.get(product)))
                self.curve_cache[product] = previous
            curves[product] = previous[1]
        return curves

    def planned_bid(self, obs):
        prices = {r: max(self.m["reserve"][r], self.input_prices[r]) for r in self.m["reserve"]}
        plan = joint_plan(obs, self.m, self.a, self.demand_curves(obs), prices, purchasing=True)
        resource = obs.resource
        missing = {r: max(0, plan["requirements"][r] - obs.raw[r]) for r in obs.raw}
        if not missing[resource] or plan["score"] <= 0:
            return 0.0
        complements = sum(q * prices[r] for r, q in missing.items() if r != resource)
        existing_basis = sum(min(plan["requirements"][r], obs.raw[r]) * obs.raw_cost[r] for r in obs.raw)
        value = max(0, (plan["revenue"] - plan["processing"] - complements - existing_basis) / missing[resource])
        # Reserve enough money for production and the other missing inputs.
        # This auction buys ONE unit. Reserve the remaining units of this same
        # input as well as other components, otherwise the bundle can be stranded.
        reserved_inputs = complements + (missing[resource] - 1) * prices[resource]
        affordable = max(0, obs.cash * self.a["cash_fraction"] - plan["processing"] - reserved_inputs)
        if obs.remaining_budget is not None:
            affordable = min(affordable, max(0, obs.remaining_budget - reserved_inputs))
        cap = min(affordable, value * self.a["bid_value_fraction"],
                  obs.bid_max if obs.bid_max is not None else obs.cash)
        if self.m["auction"] == "first":
            cap = min(cap, max(self.m["reserve"][resource], prices[resource] * 1.05))
        return float(max(0, cap))

    def plan_production(self, obs):
        if self.a["production"] == "planned":
            return joint_plan(obs, self.m, self.a, self.demand_curves(obs), self.input_prices,
                              purchasing=False)["quantities"]
        products = list(self.m["recipes"])
        remaining = {p: max(0, self.a["target_stock"] - obs.stock[p]) for p in products}
        margins = {p: self.expected_price(p, obs) - self.m["production_cost"][p]
                   - sum(q * obs.raw_cost[r] for r, q in self.m["recipes"][p].items()) for p in products}
        if self.a["production"] == "milp":
            from scipy.optimize import Bounds, LinearConstraint, milp
            rows = [[self.m["recipes"][p].get(r, 0) for p in products] for r in obs.raw]
            rows += [[1] * len(products), [self.m["production_cost"][p] for p in products]]
            bounds = list(obs.raw.values()) + [obs.capacity, obs.cash]
            result = milp(-np.array([margins[p] for p in products]), integrality=np.ones(len(products)),
                          bounds=Bounds(np.zeros(len(products)), [remaining[p] for p in products]),
                          constraints=LinearConstraint(rows, np.zeros(len(rows)), bounds),
                          options={"time_limit": 1.0})
            if result.x is not None:
                candidate = {p: int(round(x)) for p, x in zip(products, result.x)}
                try:
                    self.adapter.validate_action(obs, Action(obs.decision_id, production=candidate))
                    return candidate
                except ValueError:
                    pass  # Numerical or solver limit: deterministic greedy fallback.
        stock, cash, capacity = dict(obs.raw), obs.cash, obs.capacity
        output = {p: 0 for p in products}
        while capacity > 0:
            feasible = [p for p in products if remaining[p] > 0 and margins[p] > 0
                        and self.m["production_cost"][p] <= cash + 1e-9
                        and all(stock[r] >= q for r, q in self.m["recipes"][p].items())]
            if not feasible:
                break
            def score(p):
                burden = max(q / max(1, stock[r]) for r, q in self.m["recipes"][p].items())
                return margins[p] / burden
            product = max(feasible, key=score)
            output[product] += 1
            remaining[product] -= 1
            capacity -= 1
            cash -= self.m["production_cost"][product]
            for r, q in self.m["recipes"][product].items():
                stock[r] -= q
        return output

    def observe(self, feedback: Feedback):
        self.adapter.validate_feedback(feedback)
        digest = fingerprint(asdict(feedback))
        if feedback.decision_id in self.completed:
            if self.feedback_digests[feedback.decision_id] != digest:
                raise ValueError("Conflicting duplicate feedback")
            return
        if feedback.decision_id not in self.pending:
            raise ValueError("Feedback for an unknown decision")
        pending = self.pending[feedback.decision_id]
        if feedback.phase != pending["phase"]:
            raise ValueError("Feedback phase mismatch")
        # Validate the whole feedback before any update: one final feedback per decision.
        for key in pending["learning"]:
            product = key.split(":", 1)[1]
            if key.startswith("price:"):
                if product not in feedback.rewards or not math.isfinite(feedback.rewards[product]):
                    raise ValueError("Missing/non-finite realized product reward")
                if self.a["pricing"] == "conversion_ts":
                    if product not in feedback.censored:
                        raise ValueError("Conversion feedback must declare censorship")
                    if not feedback.censored[product]:
                        s, n = feedback.successes.get(product), feedback.trials.get(product)
                        if type(s) is not int or type(n) is not int or not 0 <= s <= n:
                            raise ValueError("Missing/invalid conversion counts")
        if feedback.phase == "auction":
            resource = pending["resource"]
            if type(feedback.won) is not bool or resource not in feedback.costs:
                raise ValueError("Auction feedback needs outcome and explicit cost")
            cost = feedback.costs[resource]
            if not math.isfinite(cost) or cost < 0:
                raise ValueError("Invalid auction cost")
            bid = self.actions[feedback.decision_id]["bids"][resource]
            if (not feedback.won and cost != 0) or cost > bid + 1e-8:
                raise ValueError("Auction payment inconsistent with submitted bid/outcome")
        for product, sold in feedback.sales.items():
            if sold > pending["offered"][product]:
                raise ValueError("Sales exceed offered stock")
        staged_models = {}
        if feedback.phase == "pricing" and self.models:
            for product, model in self.models.items():
                if product not in feedback.sales or product not in feedback.censored:
                    raise ValueError("Inventory pricing needs product sales and censorship flags")
                staged = deepcopy(model)
                staged.update(self.actions[feedback.decision_id]["prices"][product],
                              pending["previous_prices"].get(product), feedback.sales[product],
                              pending["offered"][product], feedback.censored[product])
                staged_models[product] = staged
        staged_learners = {}
        for key, decision in pending["learning"].items():
            item = key.split(":", 1)[1]
            learner = deepcopy(self.learners[key])
            if key.startswith("price:"):
                learner.update(decision, feedback.rewards[item], feedback.successes.get(item),
                               feedback.trials.get(item), feedback.censored.get(item, False))
            else:
                # Procurement proxy, deliberately not presented as final company profit.
                reward = pending["value"] * int(feedback.won) - feedback.costs[item]
                learner.update(decision, reward)
            staged_learners[key] = learner
        self.learners.update(staged_learners)
        self.models.update(staged_models)
        if feedback.phase == "auction" and self.a["buying"] == "planned":
            resource = pending["resource"]
            bid = self.actions[feedback.decision_id]["bids"][resource]
            old = self.input_prices[resource]
            if feedback.won:
                self.input_prices[resource] = 0.8 * old + 0.2 * cost
            elif bid > 0:
                self.input_prices[resource] = max(old, 0.8 * old + 0.2 * bid * 1.1)
        if feedback.phase == "pricing":
            # A quoted but unsold price is not evidence of realizable input value.
            for product, quantity in feedback.sales.items():
                if quantity > 0 and product in self.actions[feedback.decision_id]["prices"] and pending["turn"] >= self.price_turns.get(product, -1):
                    self.last_prices[product] = self.actions[feedback.decision_id]["prices"][product]
                    self.price_turns[product] = pending["turn"]
        if feedback.phase == "auction" and self.a["buying"] == "pacing":
            rho = self.m["fixed_budget"] / (self.m["turns"] * sum(self.m["supply"].values()))
            self.pacing_lambda = float(np.clip(self.pacing_lambda + self.a["eta"] * (cost - rho), 0, 1 / rho))
        self.completed.add(feedback.decision_id)
        self.feedback_digests[feedback.decision_id] = digest
        del self.pending[feedback.decision_id]

    def state_dict(self):
        return deepcopy({"schema": 2, "config": self.config, "rng": self.rng.bit_generator.state,
                         "pending": self.pending, "completed": sorted(self.completed), "actions": self.actions,
                         "observations": self.observations, "feedback_digests": self.feedback_digests,
                         "models": {p: m.state_dict() for p, m in self.models.items()}, "input_prices": self.input_prices,
                         "learners": {k: v.state_dict() for k, v in self.learners.items()},
                         "pacing_lambda": self.pacing_lambda, "last_prices": self.last_prices, "price_turns": self.price_turns})

    def load_state_dict(self, state):
        state = deepcopy(state)
        expected_keys = set(self.state_dict())
        if not isinstance(state, dict) or set(state) != expected_keys:
            raise ValueError("Invalid agent checkpoint fields")
        for key in expected_keys - {"schema", "completed", "pacing_lambda"}:
            if not isinstance(state[key], dict):
                raise ValueError(f"Invalid checkpoint mapping: {key}")
        if not isinstance(state["completed"], list) or any(not isinstance(x, str) or not x for x in state["completed"]):
            raise ValueError("Invalid completed decision IDs")
        if len(set(state["completed"])) != len(state["completed"]):
            raise ValueError("Duplicate completed decision IDs")
        if state["schema"] != 2 or state["config"] != self.config:
            raise ValueError("State version/configuration mismatch; reset or use matching config")
        staged = MarketAgent(self.config)
        if set(state["learners"]) != set(staged.learners) or set(state["models"]) != set(staged.models):
            raise ValueError("State has missing/extra learners or demand models")
        if set(state["actions"]) != set(state["observations"]) or set(state["completed"]) != set(state["feedback_digests"]):
            raise ValueError("Invalid state decision/digest indexes")
        if set(state["pending"]) & set(state["completed"]) or set(state["actions"]) != set(state["pending"]) | set(state["completed"]):
            raise ValueError("Invalid pending/completed partition")
        if set(state["input_prices"]) != set(self.m["reserve"]) or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in state["input_prices"].values()):
            raise ValueError("Invalid input price estimates")
        if type(state["pacing_lambda"]) not in (int, float) or not math.isfinite(state["pacing_lambda"]) or state["pacing_lambda"] < 0:
            raise ValueError("Invalid pacing state")
        if set(state["last_prices"]) - set(self.m["recipes"]) or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in state["last_prices"].values()):
            raise ValueError("Invalid realized price estimates")
        if set(state["price_turns"]) - set(self.m["recipes"]) or any(type(v) is not int or v < 0 for v in state["price_turns"].values()):
            raise ValueError("Invalid realized price timestamps")
        for decision_id, data in state["pending"].items():
            if not isinstance(decision_id, str) or not isinstance(data, dict) or data.get("phase") not in ("auction", "production", "pricing"):
                raise ValueError("Invalid pending decision")
            if type(data.get("turn")) is not int or data["turn"] < 0 or set(data.get("learning", {})) - set(self.learners):
                raise ValueError("Invalid pending decision context")
        for decision_id, action in state["actions"].items():
            if not isinstance(action, dict) or action.get("decision_id") != decision_id or set(action) != {"decision_id", "bids", "production", "prices"}:
                raise ValueError("Invalid cached action")
        staged.rng.bit_generator.state = state["rng"]
        staged.pending, staged.actions = state["pending"], state["actions"]
        staged.observations, staged.feedback_digests = state["observations"], state["feedback_digests"]
        staged.completed = set(state["completed"])
        staged.pacing_lambda, staged.last_prices = state["pacing_lambda"], state["last_prices"]
        staged.price_turns = state["price_turns"]
        staged.input_prices = state["input_prices"]
        for key, data in state["learners"].items():
            staged.learners[key].load_state_dict(data)
        for key, data in state["models"].items():
            staged.models[key].load_state_dict(data)
        staged._validate_checkpoint_decisions()
        self.__dict__.update(staged.__dict__)

    def _validate_checkpoint_decisions(self):
        products, resources = set(self.m["recipes"]), set(self.m["reserve"])
        for collection in (self.observations, self.feedback_digests):
            for digest in collection.values():
                if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                    raise ValueError("Invalid checkpoint fingerprint")
        expected_pending = {key: {} for key in self.learners}
        for identity, action in self.actions.items():
            if not isinstance(identity, str) or not identity:
                raise ValueError("Invalid cached action identity")
            for field, allowed in (("bids", resources), ("production", products), ("prices", products)):
                values = action[field]
                if not isinstance(values, dict) or set(values) - allowed:
                    raise ValueError("Invalid cached action keys")
                if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values.values()):
                    raise ValueError("Invalid cached action values")
                if field == "production" and any(type(v) is not int for v in values.values()):
                    raise ValueError("Invalid cached production quantities")
            if sum(bool(action[k]) for k in ("bids", "production", "prices")) > 1:
                raise ValueError("Cached action mixes phases")
            if action["prices"] and (set(action["prices"]) != products or any(v <= 0 for v in action["prices"].values())):
                raise ValueError("Invalid cached pricing action")
        for identity, pending in self.pending.items():
            offered = pending.get("offered")
            previous = pending.get("previous_prices")
            decisions = pending.get("learning")
            if not isinstance(offered, dict) or set(offered) != products or any(type(v) is not int or v < 0 for v in offered.values()):
                raise ValueError("Invalid checkpoint offered inventory")
            if not isinstance(previous, dict) or set(previous) - products or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in previous.values()):
                raise ValueError("Invalid checkpoint public prices")
            if not isinstance(decisions, dict):
                raise ValueError("Invalid checkpoint learning decisions")
            phase = pending["phase"]
            action = self.actions[identity]
            if phase == "auction":
                if pending.get("resource") not in resources or set(action["bids"]) != {pending["resource"]}:
                    raise ValueError("Checkpoint auction resource mismatch")
                if type(pending.get("value")) not in (int, float) or not math.isfinite(pending["value"]) or pending["value"] < 0:
                    raise ValueError("Invalid checkpoint auction value")
            elif pending.get("resource") is not None:
                raise ValueError("Resource supplied outside auction in checkpoint")
            field = {"auction": "bids", "production": "production", "pricing": "prices"}[phase]
            if any(action[f] for f in ("bids", "production", "prices") if f != field):
                raise ValueError("Pending phase differs from cached action")
            for key, choice in decisions.items():
                learner = self.learners[key]
                if not isinstance(choice, dict) or set(choice) != {"time", "arm", "probability"}:
                    raise ValueError("Invalid checkpoint learner choice")
                t, arm = choice["time"], choice["arm"]
                if type(t) is not int or not 0 <= t < learner.clock or type(arm) is not int or not 0 <= arm < len(learner.arms):
                    raise ValueError("Invalid checkpoint learner time/arm")
                if t in expected_pending[key]:
                    raise ValueError("Multiple pending decisions reference the same learner pull")
                if learner.kind == "exp3ix":
                    probability = choice["probability"]
                    if type(probability) not in (int, float) or not math.isfinite(probability) or not 0 < probability <= 1:
                        raise ValueError("Invalid checkpoint sampling probability")
                elif choice["probability"] is not None:
                    raise ValueError("Unexpected checkpoint sampling probability")
                kind, item = key.split(":", 1)
                if kind == "price" and (phase != "pricing" or action["prices"].get(item) != learner.arms[arm]):
                    raise ValueError("Checkpoint learner arm differs from submitted price")
                if kind == "bid" and (phase != "auction" or item != pending["resource"]):
                    raise ValueError("Checkpoint bidding learner resource mismatch")
                expected_pending[key][t] = arm
        for key, learner in self.learners.items():
            # Sliding windows may already have discarded overdue pending pulls.
            relevant = {t: arm for t, arm in expected_pending[key].items()
                        if learner.kind != "swucb" or t >= learner.clock - learner.cfg["window"]}
            actual = {t: arm for t, arm in learner.pending.items()
                      if learner.kind != "swucb" or t >= learner.clock - learner.cfg["window"]}
            if actual != relevant:
                raise ValueError("Agent and learner pending checkpoints disagree")

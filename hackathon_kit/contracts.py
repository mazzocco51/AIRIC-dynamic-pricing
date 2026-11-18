from dataclasses import asdict, dataclass, field
from typing import Protocol
import math
from numbers import Real
from .prices import tick_interval, is_tick_aligned


def finite_number(value):
    try:
        return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def mapping(value, name, keys=None):
    if not isinstance(value, dict) or any(not isinstance(k, str) or not k for k in value):
        raise ValueError(f"{name} must be a dictionary with nonempty string keys")
    if keys is not None and set(value) - set(keys):
        raise ValueError(f"Unknown key in {name}")
    return value


@dataclass
class Observation:
    decision_id: str
    turn: int
    phase: str
    cash: float
    raw: dict[str, int]
    stock: dict[str, int]
    capacity: int
    raw_cost: dict[str, float]
    stock_cost: dict[str, float]
    # Only public information. Missing optional measurements stay None.
    previous_prices: dict[str, float] = field(default_factory=dict)
    resource: str | None = None
    remaining_budget: float | None = None
    price_min: float | None = None
    price_max: float | None = None
    bid_max: float | None = None


@dataclass
class Action:
    decision_id: str
    bids: dict[str, float] = field(default_factory=dict)
    production: dict[str, int] = field(default_factory=dict)
    prices: dict[str, float] = field(default_factory=dict)


@dataclass
class Feedback:
    decision_id: str
    phase: str
    rewards: dict[str, float] = field(default_factory=dict)
    sales: dict[str, int] = field(default_factory=dict)
    costs: dict[str, float] = field(default_factory=dict)
    censored: dict[str, bool] = field(default_factory=dict)
    successes: dict[str, int] = field(default_factory=dict)
    trials: dict[str, int] = field(default_factory=dict)
    won: bool | None = None


class Agent(Protocol):
    def reset(self, seed: int = 0) -> None: ...
    def act(self, observation: Observation) -> Action: ...
    def observe(self, feedback: Feedback) -> None: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict) -> None: ...


class MarketAdapter(Protocol):
    def decode_observation(self, payload: dict) -> Observation: ...
    def encode_action(self, action: Action) -> dict: ...
    def check_compatibility(self) -> None: ...


class JsonAdapter:
    """Internal sample protocol; replace at the official integration boundary."""
    def __init__(self, config):
        self.config = config

    def check_compatibility(self):
        from .config import validate_config
        validate_config(self.config)

    def decode_observation(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("Observation must be an object")
        try:
            obs = Observation(**payload)
        except TypeError as exc:
            raise ValueError(f"Invalid observation fields: {exc}") from exc
        if obs.phase not in ("auction", "production", "pricing"):
            raise ValueError("Unknown phase")
        if not isinstance(obs.decision_id, str) or not obs.decision_id or type(obs.turn) is not int or obs.turn < 0:
            raise ValueError("Invalid decision identifier/turn")
        if not finite_number(obs.cash) or obs.cash < 0:
            raise ValueError("Cash must be finite and nonnegative")
        if type(obs.capacity) is not int or not finite_number(obs.capacity) or obs.capacity < 0:
            raise ValueError("Invalid capacity")
        m = self.config["market"]
        for name in ("raw", "stock", "raw_cost", "stock_cost", "previous_prices"):
            mapping(getattr(obs, name), name)
        if set(obs.raw) != set(m["reserve"]) or set(obs.stock) != set(m["recipes"]):
            raise ValueError("Observation inventory does not match configured market")
        for inventory in (obs.raw, obs.stock):
            if any(type(q) is not int or not finite_number(q) or q < 0 for q in inventory.values()):
                raise ValueError("Inventory quantities must be nonnegative integers")
        for costs, inventory in ((obs.raw_cost, obs.raw), (obs.stock_cost, obs.stock)):
            if set(costs) != set(inventory) or any(not finite_number(v) or v < 0 for v in costs.values()):
                raise ValueError("Explicit nonnegative inventory cost estimates required")
        mapping(obs.previous_prices, "previous_prices", m["recipes"])
        if any(not finite_number(v) or v < 0 for v in obs.previous_prices.values()):
            raise ValueError("Observed competitor prices must be finite and nonnegative")
        obs.price_min = m["price_min"] if obs.price_min is None else obs.price_min
        obs.price_max = m["price_max"] if obs.price_max is None else obs.price_max
        if not finite_number(obs.price_min) or not finite_number(obs.price_max) or not 0 < obs.price_min < obs.price_max:
            raise ValueError("Invalid observed price bounds")
        tick = m["price_tick"]
        tick_interval(obs.price_min, obs.price_max, tick)
        for value in (obs.remaining_budget, obs.bid_max):
            if value is not None and (not finite_number(value) or value < 0):
                raise ValueError("Invalid observed budget/bid limit")
        if obs.phase == "auction" and (not isinstance(obs.resource, str) or obs.resource not in m["reserve"]):
            raise ValueError("Auction resource missing or unknown")
        if obs.phase != "auction" and obs.resource is not None:
            raise ValueError("Resource auction field supplied for a different phase")
        if m["budget_mode"] == "fixed" and obs.remaining_budget is None:
            raise ValueError("Fixed budget mode requires observed remaining_budget")
        return obs

    def validate_feedback(self, feedback):
        if not isinstance(feedback, Feedback):
            raise ValueError("Feedback must implement the Feedback contract")
        if not isinstance(feedback.decision_id, str) or not feedback.decision_id:
            raise ValueError("Feedback requires a nonempty string decision ID")
        if feedback.phase not in ("auction", "production", "pricing"):
            raise ValueError("Unknown feedback phase")
        products, resources = set(self.config["market"]["recipes"]), set(self.config["market"]["reserve"])
        for name in ("rewards", "sales", "censored", "successes", "trials"):
            mapping(getattr(feedback, name), name, products)
        mapping(feedback.costs, "costs", products | resources)
        if any(not finite_number(v) for v in feedback.rewards.values()):
            raise ValueError("Feedback rewards must be finite numbers")
        if any(not finite_number(v) or v < 0 for v in feedback.costs.values()):
            raise ValueError("Feedback costs must be finite nonnegative numbers")
        for name in ("sales", "successes", "trials"):
            if any(type(v) is not int or v < 0 for v in getattr(feedback, name).values()):
                raise ValueError(f"Feedback {name} must contain nonnegative integers")
        if any(type(v) is not bool for v in feedback.censored.values()):
            raise ValueError("Censorship flags must be boolean")
        if set(feedback.successes) != set(feedback.trials) or any(
            value > feedback.trials[p] for p, value in feedback.successes.items()
        ):
            raise ValueError("Conversion successes require matching trials and successes <= trials")
        if feedback.won is not None and type(feedback.won) is not bool:
            raise ValueError("Auction outcome must be boolean or unknown")
        if feedback.phase == "auction" and type(feedback.won) is not bool:
            raise ValueError("Auction feedback requires an observed outcome")
        if feedback.phase != "auction" and feedback.won is not None:
            raise ValueError("Auction outcome supplied for a different phase")
        return feedback

    def encode_action(self, action):
        return asdict(action)

    def validate_action(self, obs, action):
        m = self.config["market"]
        if not isinstance(obs, Observation) or obs.phase not in ("auction", "production", "pricing"):
            raise ValueError("Invalid observation for action validation")
        if not isinstance(action, Action):
            raise ValueError("Action must implement the Action contract")
        if not isinstance(action.decision_id, str) or not action.decision_id:
            raise ValueError("Action requires a nonempty string decision ID")
        for name in ("bids", "production", "prices"):
            mapping(getattr(action, name), name)
        if action.decision_id != obs.decision_id:
            raise ValueError("Mismatched decision ID")
        allowed = {"auction": "bids", "production": "production", "pricing": "prices"}[obs.phase]
        for name in ("bids", "production", "prices"):
            if name != allowed and getattr(action, name):
                raise ValueError("Action contains a different phase")
        if obs.phase == "auction":
            if set(action.bids) != {obs.resource}:
                raise ValueError("Exactly one resource bid required")
            bid = action.bids[obs.resource]
            limit = min(obs.cash, obs.bid_max if obs.bid_max is not None else obs.cash,
                        obs.remaining_budget if obs.remaining_budget is not None else obs.cash)
            if not finite_number(bid) or not 0 <= bid <= limit + 1e-8:
                raise ValueError("Bid exceeds available funds/limit")
        elif obs.phase == "production":
            if set(action.production) - set(m["recipes"]):
                raise ValueError("Unknown product")
            if any(type(q) is not int or q < 0 for q in action.production.values()):
                raise ValueError("Production must use integer nonnegative quantities")
            if sum(action.production.values()) > obs.capacity:
                raise ValueError("Production capacity exceeded")
            for r in obs.raw:
                if sum(q * m["recipes"][p].get(r, 0) for p, q in action.production.items()) > obs.raw[r]:
                    raise ValueError("Raw resource constraint violated")
            if sum(q * m["production_cost"][p] for p, q in action.production.items()) > obs.cash + 1e-8:
                raise ValueError("Production cash constraint violated")
        else:
            if set(action.prices) != set(m["recipes"]):
                raise ValueError("All product prices required")
            tick = m["price_tick"]
            low = m["price_min"] if obs.price_min is None else obs.price_min
            high = m["price_max"] if obs.price_max is None else obs.price_max
            for price in action.prices.values():
                if not finite_number(price) or not low <= price <= high:
                    raise ValueError("Price out of bounds")
                if not is_tick_aligned(price, tick):
                    raise ValueError("Price violates tick size")

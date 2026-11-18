"""Explicit defaults for the toy market, never claims about the official game."""
from copy import deepcopy
import math
import tomllib
from .prices import tick_interval


def finite_number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False

PROFILES = {
    "baseline": {"pricing": "fixed"},
    "adaptive": {"pricing": "swucb"},
    "ucb": {"pricing": "ucb"},
    "exp3": {"pricing": "exp3ix"},
    "gp": {"pricing": "gpucb"},
    "conversion": {"pricing": "conversion_ts"},
    "firstprice": {"pricing": "swucb", "buying": "learned"},
    "pacing": {"pricing": "swucb", "buying": "pacing"},
    "milp": {"pricing": "swucb", "production": "milp"},
    "competitive": {"pricing": "inventory", "buying": "planned", "production": "planned"},
}
# Profiles are complete strategy combinations; applying baseline to a competitive
# configuration must not accidentally retain its specialized buying/production.
PROFILES = {name: {"pricing": "fixed", "buying": "prudent", "production": "greedy", **settings}
            for name, settings in PROFILES.items()}
SCENARIOS = ("stable", "shock", "scarcity", "aggressive")

DEFAULT = {
    "market": {
        "turns": 40, "initial_cash": 150.0, "capacity": 5,
        "auction": "second", "budget_mode": "replenishing", "fixed_budget": 80.0,
        "reserve": {"metal": 2.0, "plastic": 1.0},
        "initial_raw": {"metal": 2, "plastic": 2},
        "supply": {"metal": 5, "plastic": 5},
        "recipes": {"widget": {"metal": 1, "plastic": 1},
                    "gadget": {"metal": 2, "plastic": 1}},
        "production_cost": {"widget": 1.0, "gadget": 1.5},
        "price_min": 5.0, "price_max": 30.0, "price_tick": 0.01,
        "customers": 24, "conversion_feedback": True,
        "holding_cost": 0.02,
    },
    "agent": {
        "pricing": "fixed", "buying": "prudent", "production": "greedy",
        "margin": 0.6, "target_stock": 3, "cash_fraction": 0.4,
        "bid_value_fraction": 0.7, "price_levels": 7, "window": 20,
        "reward_low": -1000.0, "reward_high": 1000.0,
        "bid_reward_low": -30.0, "bid_reward_high": 30.0,
        "eta": 0.1, "gamma": 0.05, "gp_length": 0.25,
        "gp_noise": 0.01, "gp_beta": 2.0, "gp_history": 128,
        "demand_decay": 0.97, "demand_optimism": 0.3, "demand_prior": 3.0,
        "price_elasticity": 0.12, "competition_weight": 0.5, "planner_limit": 256,
    },
}


def merge(base, override):
    if not isinstance(base, dict) or not isinstance(override, dict):
        raise ValueError("Configuration sections must be tables")
    result = deepcopy(base)
    for key, value in override.items():
        if key not in result:
            raise ValueError(f"Unknown configuration key: {key}")
        # Replace complete market maps to allow new resource/product names.
        if key in ("reserve", "supply", "recipes", "production_cost", "initial_raw"):
            result[key] = deepcopy(value)
        else:
            result[key] = merge(result[key], value) if isinstance(value, dict) else value
    return result


def load_config(path=None, profile=None):
    config = deepcopy(DEFAULT)
    if path:
        with open(path, "rb") as stream:
            config = merge(config, tomllib.load(stream))
    if profile:
        if profile not in PROFILES:
            raise ValueError(f"Unknown profile: {profile}")
        config["agent"].update(PROFILES[profile])
    validate_config(config)
    return config


def validate_config(config):
    if not isinstance(config, dict) or set(config) != set(DEFAULT):
        raise ValueError("Configuration must contain exactly market and agent tables")
    dynamic_maps = {"reserve", "supply", "recipes", "production_cost", "initial_raw"}
    for section, defaults in DEFAULT.items():
        values = config[section]
        if not isinstance(values, dict) or set(values) != set(defaults):
            raise ValueError(f"Invalid keys in configuration table {section}")
        for key, default in defaults.items():
            value = values[key]
            if key in dynamic_maps:
                if not isinstance(value, dict) or any(not isinstance(k, str) or not k for k in value):
                    raise ValueError(f"{section}.{key} must be a table with nonempty string keys")
            elif type(default) is bool:
                if type(value) is not bool:
                    raise ValueError(f"{section}.{key} must be boolean")
            elif type(default) is int:
                if type(value) is not int:
                    raise ValueError(f"{section}.{key} must be an integer")
            elif type(default) is float:
                if not finite_number(value):
                    raise ValueError(f"{section}.{key} must be a finite number")
            elif type(default) is str and not isinstance(value, str):
                raise ValueError(f"{section}.{key} must be a string")
    m, a = config["market"], config["agent"]
    for key in ("turns", "capacity", "customers"):
        if type(m[key]) is not int or m[key] <= 0:
            raise ValueError(f"market.{key} must be a positive integer")
    for key in ("target_stock", "price_levels", "window", "gp_history", "planner_limit"):
        if type(a[key]) is not int or a[key] <= 0:
            raise ValueError(f"agent.{key} must be a positive integer")
    if a["price_levels"] < 2 or a["window"] < a["price_levels"]:
        raise ValueError("At least two price levels; window must cover all levels")
    for section in (m, a):
        for key, value in section.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and not finite_number(value):
                raise ValueError(f"Non-finite configuration: {key}")
    if m["auction"] not in ("first", "second"):
        raise ValueError("auction must be first or second")
    if m["budget_mode"] not in ("fixed", "replenishing"):
        raise ValueError("budget_mode must be fixed or replenishing")
    if not 0 < m["price_min"] < m["price_max"] or m["price_tick"] <= 0:
        raise ValueError("Invalid price bounds or tick")
    tick_interval(m["price_min"], m["price_max"], m["price_tick"])
    if m["initial_cash"] <= 0 or m["fixed_budget"] <= 0 or m["holding_cost"] < 0:
        raise ValueError("Cash/budget must be positive and holding cost nonnegative")
    if set(m["reserve"]) != set(m["supply"]):
        raise ValueError("Supply and reserve resource names differ")
    if not m["reserve"] or not m["recipes"]:
        raise ValueError("At least one resource and one product required")
    if set(m["initial_raw"]) - set(m["reserve"]) or any(type(q) is not int or not finite_number(q) or q < 0 for q in m["initial_raw"].values()):
        raise ValueError("Invalid initial raw inventory; configure it when replacing resources")
    for resource, count in m["supply"].items():
        if type(count) is not int or not finite_number(count) or count <= 0 or not finite_number(m["reserve"][resource]) or m["reserve"][resource] < 0:
            raise ValueError("Invalid supply or reserve")
    if set(m["recipes"]) != set(m["production_cost"]):
        raise ValueError("Missing production costs")
    for product, recipe in m["recipes"].items():
        if not isinstance(recipe, dict) or not recipe or any(r not in m["reserve"] or type(q) is not int or not finite_number(q) or q <= 0 for r, q in recipe.items()):
            raise ValueError(f"Invalid recipe: {product}")
        cost = m["production_cost"][product]
        if not finite_number(cost) or cost < 0:
            raise ValueError("Invalid production cost")
    if a["pricing"] not in ("fixed", "ucb", "swucb", "exp3ix", "gpucb", "conversion_ts", "random", "reactive", "inventory"):
        raise ValueError("Unknown pricing strategy")
    if a["production"] not in ("greedy", "milp", "planned") or a["buying"] not in ("prudent", "pacing", "learned", "planned"):
        raise ValueError("Unknown production or buying strategy")
    if a["buying"] == "pacing" and (m["auction"] != "second" or m["budget_mode"] != "fixed"):
        raise ValueError("Pacing requires second-price auctions and a fixed purchasing budget")
    if a["buying"] == "learned" and m["auction"] != "first":
        raise ValueError("Learned discrete bids require first-price auctions")
    if a["pricing"] == "conversion_ts" and not m["conversion_feedback"]:
        raise ValueError("Conversion TS requires observed successes and trials")
    for low, high in (("reward_low", "reward_high"), ("bid_reward_low", "bid_reward_high")):
        if not a[low] < a[high] or not math.isfinite(a[high] - a[low]):
            raise ValueError(f"{low} must be below {high} with a finite reward span")
    for key in ("cash_fraction", "bid_value_fraction"):
        if not 0 < a[key] <= 1:
            raise ValueError(f"{key} must be in (0, 1]")
    if a["margin"] < 0 or any(a[k] <= 0 for k in ("eta", "gamma", "gp_length", "gp_noise", "gp_beta")):
        raise ValueError("Invalid learning parameter or margin")
    if not 0 < a["demand_decay"] <= 1 or a["demand_optimism"] < 0 or a["demand_prior"] <= 0 or a["price_elasticity"] <= 0 or a["competition_weight"] < 0:
        raise ValueError("Invalid demand model parameter")
    planned = (a["pricing"] == "inventory", a["production"] == "planned", a["buying"] == "planned")
    if any(planned) and not all(planned):
        raise ValueError("Inventory pricing and planned buying/production must be selected together")
    if a["production"] == "milp":
        try:
            from scipy.optimize import milp  # noqa: F401
        except ImportError as exc:
            raise ValueError("MILP requires pip install '.[optimize]'") from exc

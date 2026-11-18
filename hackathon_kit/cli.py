import argparse
from copy import deepcopy
import csv
from dataclasses import asdict
import gzip
import hashlib
import itertools
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import tempfile
import tomllib

import numpy as np

from . import __version__
from .agent import MarketAgent
from .config import DEFAULT, PROFILES, SCENARIOS, load_config, validate_config
from .contracts import Feedback, JsonAdapter
from .simulator import run_simulation


def strict_json_loads(text):
    """Do not silently repair contradictory keys or non-JSON floating values."""
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    def constant(value):
        raise ValueError(f"Non-finite JSON value: {value}")
    def floating(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"JSON number outside finite numeric range: {value}")
        return number
    return json.loads(text, object_pairs_hook=object_pairs, parse_constant=constant, parse_float=floating)


def json_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(data, indent=2, allow_nan=False)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(serialized)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def toml_text(config):
    lines = []
    def visit(table, path):
        if path:
            lines.append("[" + ".".join(json.dumps(part) for part in path) + "]")
        for key, value in table.items():
            if not isinstance(value, dict):
                lines.append(f"{json.dumps(key)} = {json.dumps(value)}")
        lines.append("")
        for key, value in table.items():
            if isinstance(value, dict):
                visit(value, path + [key])
    visit(config, [])
    result = "\n".join(lines)
    assert tomllib.loads(result) == config
    return result


def csv_write(path, rows):
    if not rows:
        raise ValueError("No results to write")
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def versions():
    result = {"kit": __version__, "python": platform.python_version(), "numpy": np.__version__,
              "platform": platform.platform()}
    for package in ("scipy", "matplotlib", "pytest"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def source_fingerprint():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def save_run(config, scenario, seed, directory, opponent_configs=None):
    validate_config(config)
    if type(seed) is not int or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    if scenario not in SCENARIOS:
        raise ValueError("Unknown scenario")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "config.toml").write_text(toml_text(config), encoding="utf-8")
    manifest = {"versions": versions(), "code_sha256": source_fingerprint(), "seed": seed, "scenario": scenario,
                "opponent_configs": opponent_configs, "status": "running"}
    json_write(directory / "manifest.json", manifest)
    try:
        with gzip.open(directory / "trace.jsonl.gz", "wt", encoding="utf-8") as stream:
            def emit(row):
                if row.get("type") == "header":
                    manifest["resolved_opponent_configs"] = row.get("opponent_configs")
                    manifest["agent_seeds"] = row.get("agent_seeds")
                stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
            metrics, state = run_simulation(config, scenario, seed, emit, opponent_configs=opponent_configs)
        csv_write(directory / "metrics.csv", [metrics])
        json_write(directory / "state.json", state)
        json_write(directory / "manifest.json", {**manifest, "status": "complete"})
    except Exception as exc:
        json_write(directory / "manifest.json", {**manifest, "status": "failed", "error": str(exc)})
        raise
    return metrics


def summarize(rows):
    result = []
    keys = sorted(set((row["profile"], row["split"], row["scenario"]) for row in rows))
    for profile, split, scenario in keys:
        group = [r for r in rows if (r["profile"], r["split"], r["scenario"]) == (profile, split, scenario)]
        row = {"profile": profile, "split": split, "scenario": scenario, "runs": len(group)}
        metrics = ["profit", "final_equity", "final_cash", "min_cash", "raw_remaining", "stock_remaining",
                   "invalid_actions", "decision_ms_mean", "decision_ms_max"]
        metrics += [name for name in ("units_sold", "procurement_spend", "rank", "win_share") if all(name in r for r in group)]
        for metric in metrics:
            values = [r[metric] for r in group]
            row[metric + "_mean"] = statistics.mean(values)
            row[metric + "_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
        row["profit_p10"] = float(np.quantile([r["profit"] for r in group], 0.1))
        row["profit_se"] = row["profit_sd"] / len(group) ** 0.5
        result.append(row)
    return result


def replay(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    count = 0
    finished = False
    with opener(path, "rt", encoding="utf-8") as stream:
        line = next(stream, None)
        if line is None:
            raise ValueError("Empty trace")
        header = strict_json_loads(line)
        if not isinstance(header, dict) or header.get("type") != "header" or type(header.get("schema")) is not int or header["schema"] != 1:
            raise ValueError("Unsupported trace header")
        agent = MarketAgent(header["config"])
        agent.load_state_dict(header["initial_state"])
        adapter = JsonAdapter(header["config"])
        for line in stream:
            row = strict_json_loads(line)
            if not isinstance(row, dict) or "type" not in row:
                raise ValueError("Trace records must be typed JSON objects")
            if finished:
                raise ValueError("Unexpected data after final record")
            if row["type"] == "decision":
                action = agent.act(adapter.decode_observation(row["observation"]))
                if asdict(action) != row["action"]:
                    raise ValueError(f"Replay differs at {action.decision_id}")
                count += 1
            elif row["type"] == "feedback":
                agent.observe(adapter.validate_feedback(Feedback(**row["feedback"])))
            elif row["type"] == "final":
                if agent.state_dict()["pending"]:
                    raise ValueError("Trace ended with unresolved decisions")
                if agent.state_dict() != row["state"]:
                    raise ValueError("Final reconstructed state differs")
                finished = True
            else:
                raise ValueError("Unknown trace record")
    if not finished:
        raise ValueError("Incomplete trace: no final record")
    return {"verified_decisions": count, "state_match": True}


def configured(args, profile=None):
    config = load_config(args.config, profile if profile is not None else getattr(args, "profile", None))
    if getattr(args, "turns", None) is not None:
        config["market"]["turns"] = args.turns
    validate_config(config)
    return config


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be positive")
    return number


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Must be nonnegative")
    return number


def seed_ranges(dev_start, dev_count, eval_start=None, eval_count=0):
    development = range(dev_start, dev_start + dev_count)
    evaluation = range(eval_start or 0, (eval_start or 0) + eval_count)
    if eval_count and max(development.start, evaluation.start) < min(development.stop, evaluation.stop):
        raise ValueError("Development and evaluation seed ranges overlap")
    return development, evaluation


def paired_comparisons(rows):
    if not rows:
        raise ValueError("Paired comparison requires results")
    profiles = list(dict.fromkeys(r["profile"] for r in rows))
    reference = "baseline" if "baseline" in profiles else profiles[0]
    results = []
    for profile in profiles:
        if profile == reference:
            continue
        for split, scenario in sorted(set((r["split"], r["scenario"]) for r in rows)):
            base = {r["seed"]: r for r in rows if (r["profile"], r["split"], r["scenario"]) == (reference, split, scenario)}
            target = [r for r in rows if (r["profile"], r["split"], r["scenario"]) == (profile, split, scenario)]
            base_count = sum((r["profile"], r["split"], r["scenario"]) == (reference, split, scenario) for r in rows)
            target_seeds = {r["seed"] for r in target}
            if len(base) != base_count or len(target_seeds) != len(target) or target_seeds != set(base) or not target:
                raise ValueError("Paired comparisons require exactly one result per matching seed")
            differences = [r["profit"] - base[r["seed"]]["profit"] for r in target]
            sd = statistics.stdev(differences) if len(differences) > 1 else 0.0
            results.append({"profile": profile, "reference": reference, "split": split, "scenario": scenario,
                            "pairs": len(differences), "profit_delta_mean": statistics.mean(differences),
                            "profit_delta_sd": sd, "profit_delta_se": sd / len(differences) ** 0.5,
                            "positive_delta_fraction": sum(d > 0 for d in differences) / len(differences)})
    return results


def build_parser():
    parser = argparse.ArgumentParser(description="Offline market-agent toolkit (internal sample protocol)")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("simulate", "benchmark", "tune", "validate", "act"):
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path)
        if name != "benchmark":
            p.add_argument("--profile", choices=PROFILES)
        if name in ("simulate", "benchmark", "tune"):
            p.add_argument("--out", type=Path, required=True)
            p.add_argument("--turns", type=positive)
            p.add_argument("--opponents", nargs="+", choices=PROFILES,
                           help="Explicit opponent profiles; omitted uses the scenario default roster")
        if name == "simulate":
            p.add_argument("--scenario", choices=SCENARIOS, default="stable")
            p.add_argument("--seed", type=nonnegative, default=0)
        if name == "benchmark":
            p.add_argument("--profiles", nargs="+", choices=PROFILES, default=["baseline", "adaptive", "ucb", "exp3"])
            p.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
            p.add_argument("--dev-seeds", type=positive, default=20)
            p.add_argument("--eval-seeds", type=positive, default=20)
            p.add_argument("--dev-start", type=nonnegative, default=0)
            p.add_argument("--eval-start", type=nonnegative, default=100000)
        if name == "tune":
            p.add_argument("--windows", nargs="+", type=positive, default=[10, 20, 40])
            p.add_argument("--cash-fractions", nargs="+", type=float, default=[0.25, 0.4])
            p.add_argument("--dev-seeds", type=positive, default=20)
            p.add_argument("--dev-start", type=nonnegative, default=0)
            p.add_argument("--parameter", action="append", default=[], metavar="NAME=JSON_LIST",
                           help='Override default grid, e.g. demand_decay=[0.9,0.97]')
        if name == "act":
            p.add_argument("--input", type=Path, help="JSON envelope; otherwise stdin")
            p.add_argument("--state", type=Path, required=True, help="Persistent JSON agent state")
            p.add_argument("--seed", type=nonnegative, default=0)
    p = sub.add_parser("replay")
    p.add_argument("trace", type=Path)
    return parser


def opponent_configs(config, names):
    if not names:
        return None
    result = []
    for name in names:
        opponent = deepcopy(config)
        opponent["agent"] = {**deepcopy(DEFAULT["agent"]), **PROFILES[name]}
        validate_config(opponent)
        result.append(opponent)
    return result


def execute(args):
    if args.command == "validate":
        config = configured(args)
        agent = MarketAgent(config)
        return {"valid": True, "strategies": config["agent"],
                "requirements": {**{k: l.requirements for k, l in agent.learners.items()},
                                 **{f"demand:{p}": {"feedback": ["sales", "censored"],
                                                       "observation": ["stock", "price bounds", "raw_cost", "stock_cost"]}
                                    for p in agent.models}}}
    if args.command == "act":
        data = strict_json_loads(args.input.read_text(encoding="utf-8-sig") if args.input else sys.stdin.read())
        if not isinstance(data, dict) or not isinstance(data.get("observation"), dict) or not isinstance(data.get("feedback", []), list):
            raise ValueError("act expects an observation object and optional feedback list")
        state_exists = args.state.exists()
        state = strict_json_loads(args.state.read_text(encoding="utf-8-sig")) if state_exists else None
        if state_exists and (not isinstance(state, dict) or "config" not in state):
            raise ValueError("Malformed saved agent state")
        config = state["config"] if state is not None and not args.config and not args.profile else configured(args)
        agent = MarketAgent(config, args.seed)
        if state is not None:
            agent.load_state_dict(state)
        for feedback in data.get("feedback", []):
            agent.observe(JsonAdapter(config).validate_feedback(Feedback(**feedback)))
        obs = JsonAdapter(config).decode_observation(data["observation"])
        action = agent.act(obs)
        json_write(args.state, agent.state_dict())
        return asdict(action)
    if args.command == "simulate":
        config = configured(args)
        return save_run(config, args.scenario, args.seed, args.out, opponent_configs(config, args.opponents))
    if args.command == "replay":
        return replay(args.trace)
    if args.command == "benchmark":
        development, evaluation = seed_ranges(args.dev_start, args.dev_seeds, args.eval_start, args.eval_seeds)
        configurations = [(profile, configured(args, profile)) for profile in dict.fromkeys(args.profiles)]
        rosters = {profile: opponent_configs(config, args.opponents) for profile, config in configurations}
        args.out.mkdir(parents=True, exist_ok=False)
        rows = []
        for profile, config in configurations:
            for scenario in dict.fromkeys(args.scenarios):
                for split, seeds in (("development", development), ("evaluation", evaluation)):
                    print(f"{profile} / {scenario} / {split}", file=sys.stderr)
                    for seed in seeds:
                        metrics = save_run(config, scenario, seed, args.out / profile / scenario / f"{split}-{seed}",
                                           rosters[profile])
                        rows.append({"profile": profile, "split": split, **metrics})
        csv_write(args.out / "runs.csv", rows)
        csv_write(args.out / "summary.csv", summarize(rows))
        paired = paired_comparisons(rows)
        if paired:
            csv_write(args.out / "paired.csv", paired)
        json_write(args.out / "manifest.json", {"versions": versions(), "code_sha256": source_fingerprint(), "development_seeds": list(development),
                   "evaluation_seeds": list(evaluation), "profiles": args.profiles, "opponents": args.opponents})
        return {"runs": len(rows), "summary": str(args.out / "summary.csv")}
    if args.command == "tune":
        if args.dev_start + args.dev_seeds > 100000:
            raise ValueError("Tuning must not use evaluation seeds")
        base = configured(args, args.profile or ("adaptive" if not args.config else None))
        grid = {"window": args.windows, "cash_fraction": args.cash_fractions}
        if args.parameter:
            grid = {}
            for entry in args.parameter:
                name, separator, raw = entry.partition("=")
                if not separator or name not in base["agent"] or name in grid:
                    raise ValueError("Each --parameter requires a unique agent option NAME=JSON_LIST")
                values = strict_json_loads(raw)
                if not isinstance(values, list) or not values:
                    raise ValueError("Parameter grid values must be a nonempty JSON list")
                grid[name] = values
        candidates = []
        for values in itertools.product(*grid.values()):
            config = deepcopy(base)
            config["agent"].update(dict(zip(grid, values)))
            validate_config(config)
            candidates.append(config)
        rosters = [opponent_configs(config, args.opponents) for config in candidates]
        args.out.mkdir(parents=True, exist_ok=False)
        results = []
        for index, config in enumerate(candidates):
            print(f"Candidate {index + 1}/{len(candidates)}", file=sys.stderr)
            metrics = [save_run(config, scenario, seed, args.out / f"candidate-{index}" / scenario / str(seed),
                                rosters[index])
                       for scenario in SCENARIOS for seed in range(args.dev_start, args.dev_start + args.dev_seeds)]
            results.append({"candidate": index, "window": config["agent"]["window"],
                            "cash_fraction": config["agent"]["cash_fraction"],
                            "parameters": json.dumps({key: config["agent"][key] for key in grid}, sort_keys=True),
                            "profit_mean": statistics.mean(r["profit"] for r in metrics),
                            "profit_sd": statistics.stdev(r["profit"] for r in metrics)})
        best = max(results, key=lambda r: r["profit_mean"])
        csv_write(args.out / "candidates.csv", results)
        (args.out / "selected.toml").write_text(toml_text(candidates[best["candidate"]]), encoding="utf-8")
        json_write(args.out / "selection.json", {"criterion": "mean profit across all development scenarios/seeds",
                   "development_seeds": list(range(args.dev_start, args.dev_start + args.dev_seeds)),
                   "evaluation_seeds_used": [], "selected": best, "versions": versions(), "code_sha256": source_fingerprint(),
                   "opponents": args.opponents, "opponent_configs": rosters[best["candidate"]]})
        return {"selected": best, "config": str(args.out / "selected.toml"),
                "note": "Selection uses development data only; evaluate on held-out seeds."}


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        result = execute(args)
        print(json.dumps(result, allow_nan=False))
    except (ValueError, KeyError, TypeError, OSError, ImportError, OverflowError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

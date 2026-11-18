"""Frozen development/holdout protocol, parallel on Windows. Run from project root."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hackathon_kit.cli import csv_write, json_write, opponent_configs, source_fingerprint, toml_text, versions
from hackathon_kit.config import SCENARIOS, load_config, validate_config
from hackathon_kit.simulator import resolve_opponent_configs, run_simulation


def matchup_opponents(config, roster, scenario):
    if roster == "selfplay":
        opponents = [deepcopy(config) for _ in range(3)]
    elif roster == "strong":
        opponents = opponent_configs(config, ["exp3", "gp", "competitive"])
    elif roster == "default":
        opponents = None
    else:
        raise ValueError(f"Unknown study roster: {roster}")
    return resolve_opponent_configs(config, scenario, opponents)


def run_job(job):
    label, config, roster, scenario, seed = job
    opponents = matchup_opponents(config, roster, scenario)
    metrics, _ = run_simulation(config, scenario, seed, opponent_configs=opponents)
    return {"profile": label, "roster": roster, **metrics}


def study_configurations(stage, selection=None):
    """Freeze one market across contestants before creating any paired jobs."""
    configurations = {}
    if stage == "development":
        seeds = list(range(200, 204))
        for prior in (1.5, 3.0):
            for optimism in (0.0, 0.3):
                cfg = load_config(profile="competitive")
                cfg["agent"].update(demand_prior=prior, demand_optimism=optimism)
                validate_config(cfg)
                configurations[f"prior{prior}-optimism{optimism}"] = cfg
    elif stage == "evaluation":
        if selection is None:
            raise ValueError("evaluation requires the frozen selection TOML")
        selected = load_config(selection)
        seeds = list(range(300000, 300020))
        for profile in ("baseline", "exp3", "gp"):
            cfg = load_config(profile=profile)
            cfg["market"] = deepcopy(selected["market"])
            validate_config(cfg)
            configurations[profile] = cfg
        configurations["competitive"] = selected
    else:
        raise ValueError(f"Unknown study stage: {stage}")
    return configurations, seeds


def protocol_manifest(configurations):
    """Record actual opponents, not names whose meaning might change later."""
    markets = [config["market"] for config in configurations.values()]
    if not markets or any(market != markets[0] for market in markets[1:]):
        raise ValueError("Every study contestant must use the same market")
    matchups = {}
    for label, config in configurations.items():
        matchups[label] = {}
        for roster in ("default", "strong"):
            matchups[label][roster] = {scenario: matchup_opponents(config, roster, scenario)
                                      for scenario in (SCENARIOS if roster == "default" else SCENARIOS[:3])}
    # Paired score differences are meaningful only against identical opponents.
    for roster in ("default", "strong"):
        first = next(iter(matchups.values()))[roster]
        if any(value[roster] != first for value in matchups.values()):
            raise ValueError("Paired contestants must face the same resolved opponent roster")
    return matchups


def jobs_for(label, config, seeds):
    for roster in ("default", "strong"):
        for scenario in (SCENARIOS if roster == "default" else SCENARIOS[:3]):
            for seed in seeds:
                yield label, config, roster, scenario, seed


def summarize(rows):
    result = []
    for profile, roster, scenario in sorted(set((r["profile"], r["roster"], r["scenario"]) for r in rows)):
        group = [r for r in rows if (r["profile"], r["roster"], r["scenario"]) == (profile, roster, scenario)]
        profits = [r["profit"] for r in group]
        result.append({"profile": profile, "roster": roster, "scenario": scenario, "runs": len(group),
                       "profit_mean": statistics.mean(profits), "profit_se": statistics.stdev(profits) / len(group)**0.5 if len(group)>1 else 0,
                       "win_share": statistics.mean(r["win_share"] for r in group),
                       "rank_mean": statistics.mean(r["rank"] for r in group),
                       "cash_mean": statistics.mean(r["final_cash"] for r in group),
                       "invalid_actions": sum(r["invalid_actions"] for r in group),
                       "decision_ms_mean": statistics.mean(r["decision_ms_mean"] for r in group)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["development", "evaluation"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("workers must be in 1..16")
    if args.stage == "evaluation" and args.selection is None:
        parser.error("evaluation requires the frozen --selection TOML from development")
    configurations, seeds = study_configurations(args.stage, args.selection)
    matchups = protocol_manifest(configurations)
    jobs = [job for label, cfg in configurations.items() for job in jobs_for(label, cfg, seeds)]
    if args.stage == "evaluation":
        jobs += [("competitive", configurations["competitive"], "selfplay", "stable", seed) for seed in seeds]
        matchups["competitive"]["selfplay"] = {"stable": matchup_opponents(configurations["competitive"], "selfplay", "stable")}
    args.out.mkdir(parents=True, exist_ok=False)
    study_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest = {"stage": args.stage, "seeds": seeds, "versions": versions(), "code_sha256": source_fingerprint(),
                "study_sha256": study_hash, "resolved_matchups": matchups,
                "configurations": configurations, "jobs": len(jobs),
                "selection_rule": "maximum development win share; tie-break mean profit", "status": "running"}
    if args.selection is not None:
        manifest["selection_sha256"] = hashlib.sha256(args.selection.read_bytes()).hexdigest()
    json_write(args.out / "manifest.json", manifest)
    rows = []
    try:
        with (args.out / "partial-runs.jsonl").open("w", encoding="utf-8") as checkpoint:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for row in pool.map(run_job, jobs, chunksize=2):
                    rows.append(row)
                    checkpoint.write(json.dumps(row, allow_nan=False) + "\n")
                    checkpoint.flush()
                    if len(rows) % 20 == 0:
                        print(f"{len(rows)}/{len(jobs)}", flush=True)
        if source_fingerprint() != manifest["code_sha256"] or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != study_hash:
            raise RuntimeError("Source changed during the study; rerun with frozen code")
    except BaseException as exc:
        manifest.update(status="failed", completed_jobs=len(rows), error=f"{type(exc).__name__}: {exc}")
        json_write(args.out / "manifest.json", manifest)
        raise
    csv_write(args.out / "runs.csv", rows)
    csv_write(args.out / "summary.csv", summarize(rows))
    if args.stage == "development":
        scores = []
        for label in configurations:
            subset = [r for r in rows if r["profile"] == label]
            scores.append((statistics.mean(r["win_share"] for r in subset), statistics.mean(r["profit"] for r in subset), label))
        winner = max(scores)[2]
        (args.out / "selected.toml").write_text(toml_text(configurations[winner]), encoding="utf-8")
        manifest.update(selected=winner, scores=scores)
    else:
        paired = []
        for reference in ("baseline", "exp3", "gp"):
            for roster in ("default", "strong"):
                for scenario in (SCENARIOS if roster == "default" else SCENARIOS[:3]):
                    group = [r for r in rows if r["roster"] == roster and r["scenario"] == scenario]
                    rival = {r["seed"]: r for r in group if r["profile"] == reference}
                    differences = [r["profit"] - rival[r["seed"]]["profit"] for r in group if r["profile"] == "competitive"]
                    paired.append({"reference": reference, "roster": roster, "scenario": scenario,
                                   "pairs": len(differences), "delta_mean": statistics.mean(differences),
                                   "delta_se": statistics.stdev(differences)/len(differences)**0.5})
        csv_write(args.out / "paired.csv", paired)
    manifest["status"] = "complete"
    json_write(args.out / "manifest.json", manifest)
    print(json.dumps({"stage": args.stage, "runs": len(rows)}))


if __name__ == "__main__":
    main()

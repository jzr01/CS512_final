"""Generate paired-seed experiment suites, with optional per-condition training."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import yaml
from distributed_swarm.config import load_config
from distributed_swarm.training.ppo import train
from .evaluate import evaluate, write_results
from .ablation import variants


def cases(base, spec):
    yield "baseline", deepcopy(base)
    for loss in spec["packet_losses"]:
        for delay in spec["max_delays"]:
            for mode in ("naive", "timestamp", "confidence"):
                config = deepcopy(base)
                config.setdefault("network", {}).update(packet_loss=loss, min_delay=0., max_delay=delay)
                config.setdefault("consistency", {}).update(mode=mode, filter_stale=None)
                yield f"consistency_{mode}_loss{loss}_delay{delay}", config
    for name, filtering, clocks in [("unfiltered",False,False), ("sequence",True,False), ("sequence_clock",True,True)]:
        config = deepcopy(base)
        config.setdefault("consistency", {}).update(mode="timestamp", filter_stale=filtering, logical_clocks=clocks)
        yield f"ordering_{name}", config
    for count in spec["crash_counts"]:
        config = deepcopy(base)
        config.setdefault("network", {})["crashes"] = [dict(robot_id=i, time=spec["fault_time"]) for i in range(count)]
        yield f"crashes_{count}", config
    for name, config in variants(base):
        yield f"ablation_{name}", config
    for name in ("outage", "partition"):
        config = deepcopy(base)
        network = config.setdefault("network", {})
        interval = dict(start=spec["fault_time"], end=spec["recovery_time"])
        if name == "outage":
            network["disconnections"] = [{**interval, "robots": [0]}]
        else:
            n = base.get("num_agents", 5)
            network["partitions"] = [{**interval, "groups": [list(range(n//2)), list(range(n//2, n))]}]
        yield name, config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["2d","3d"], default="2d")
    parser.add_argument("--config")
    parser.add_argument("--suite", default=str(Path(__file__).parents[1] / "configs/experiments.yaml"))
    parser.add_argument("--checkpoint", help="Fixed-policy robustness evaluation")
    parser.add_argument("--train", action="store_true", help="Train separate policies for every case and seed")
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--output-dir", default="results/experiments")
    parser.add_argument("--limit", type=int, help="Limit cases for smoke tests")
    parser.add_argument("--stochastic", action="store_true", help="Sample policy actions during evaluation")
    args = parser.parse_args()
    if args.train and args.checkpoint:
        parser.error("choose per-case training or fixed checkpoint evaluation")
    base = load_config(args.config, args.kind)
    spec = yaml.safe_load(Path(args.suite).read_text())
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "suite.json").write_text(json.dumps(dict(kind=args.kind, spec=spec, base=base, stochastic=args.stochastic,
        protocol="per_condition_training" if args.train else "fixed_policy" if args.checkpoint else "random_baseline"), indent=2))
    all_rows = []
    for idx, (name, config) in enumerate(cases(base["env"], spec)):
        if args.limit is not None and idx >= args.limit:
            break
        config_file = out / f"{name}.yaml"
        config_file.write_text(yaml.safe_dump({**base, "env": config}))
        for seed in spec["seeds"]:
            checkpoint = args.checkpoint
            if args.train:
                train_args = ["--config", str(config_file), "--seed", str(seed), "--run-dir", str(out / name / f"seed{seed}")]
                if args.total_timesteps:
                    train_args.extend(["--total-timesteps", str(args.total_timesteps)])
                checkpoint = train(args.kind, train_args)
            rows = evaluate(args.kind, config, checkpoint, spec["episodes"], spec["evaluation_seed"] + seed * 1000,
                            stochastic=args.stochastic)
            for row in rows:
                row.update(case=name, training_seed=seed if args.train else None,
                           evaluation_seed_group=seed, checkpoint=str(checkpoint) if checkpoint else "")
            all_rows.extend(rows)
            write_results(all_rows, out / "episodes.csv")
            print(f"completed {name} seed={seed}", flush=True)


if __name__ == "__main__":
    main()

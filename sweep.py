from dotenv import load_dotenv
from pathlib import Path
import argparse
import copy
import itertools
import yaml


DEFAULT_ENTITY = "munasir"
DEFAULT_PROJECT = "llm-ca-evolution"
CONFIG_DIR = Path("configurations")


def flatten_keys(data, prefix=""):
    keys = []
    if not isinstance(data, dict):
        return keys
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            keys.extend(flatten_keys(value, path))
        else:
            keys.append(path)
    return keys


def get_nested(config, dotted_key):
    current = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"Unknown sweep key '{dotted_key}'")
        current = current[part]
    return current


def set_nested(config, dotted_key, value):
    parts = dotted_key.split(".")
    current = config
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"Unknown sweep key '{dotted_key}'")
        current = current[part]
    if not isinstance(current, dict) or parts[-1] not in current:
        raise KeyError(f"Unknown sweep key '{dotted_key}'")
    current[parts[-1]] = value


def parse_scalar(raw):
    text = raw.strip()
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("null", "none"):
        return None
    try:
        if any(token in lowered for token in (".", "e")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def parse_sweep_arg(raw):
    if "=" not in raw:
        raise ValueError(f"Sweep argument must look like key=v1,v2 got '{raw}'")
    key, values = raw.split("=", 1)
    key = key.strip()
    parsed = [parse_scalar(item) for item in values.split(",") if item.strip() != ""]
    if not key or not parsed:
        raise ValueError(f"Sweep argument must look like key=v1,v2 got '{raw}'")
    return key, parsed


def parse_shard(raw):
    if "/" not in raw:
        raise ValueError(f"Shard must look like 2/4 got '{raw}'")
    index, count = raw.split("/", 1)
    index, count = int(index), int(count)
    if count < 1 or index < 1 or index > count:
        raise ValueError(f"Shard must look like 2/4 with 1 <= index <= count got '{raw}'")
    return index - 1, count


def select_shard(jobs, shard):
    index, count = shard
    # Round robin so every shard gets a similar mix of cheap and expensive jobs.
    return [job for position, job in enumerate(jobs) if position % count == index]


def parse_seeds(raw):
    seeds = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        seeds.append(int(item))
    if not seeds:
        raise ValueError("Select at least one seed")
    return seeds


def load_yaml(path):
    with open(path, "r") as file:
        return yaml.safe_load(file)


def is_experiment_config(path):
    data = load_yaml(path)
    return isinstance(data, dict) and "environment" in data and "evolution" in data


def discover_configs(paths=None):
    if paths:
        resolved = []
        for raw in paths:
            path = Path(raw)
            if path.is_dir():
                resolved.extend(sorted(path.glob("*.yaml")))
            else:
                resolved.append(path)
        return resolved

    if not CONFIG_DIR.exists():
        raise FileNotFoundError(f"No config directory at {CONFIG_DIR}")
    return [path for path in sorted(CONFIG_DIR.glob("*.yaml")) if is_experiment_config(path)]


def load_sweep_file(path):
    data = load_yaml(path) or {}
    if not isinstance(data, dict):
        raise ValueError("Sweep file must be a YAML mapping")

    sweep = data.get("sweep", {})
    if sweep is None:
        sweep = {}
    if not isinstance(sweep, dict):
        raise ValueError("sweep: must be a mapping of dotted keys to lists of values")

    parsed_sweep = {}
    for key, values in sweep.items():
        if not isinstance(values, list) or len(values) == 0:
            raise ValueError(f"Sweep key '{key}' must be a non-empty list")
        parsed_sweep[str(key)] = values

    seeds = data.get("seeds")
    if seeds is not None:
        if not isinstance(seeds, list) or len(seeds) == 0:
            raise ValueError("seeds: must be a non-empty list")
        seeds = [int(seed) for seed in seeds]

    configs = data.get("configs")
    entity = data.get("entity")
    project = data.get("project")
    return parsed_sweep, seeds, configs, entity, project


def expand_jobs(config_paths, sweep, seeds):
    keys = list(sweep.keys())
    if keys:
        combos = [dict(zip(keys, values)) for values in itertools.product(*[sweep[key] for key in keys])]
    else:
        combos = [{}]

    jobs = []
    for config_path in config_paths:
        base = load_yaml(config_path)
        for overrides in combos:
            for key in overrides:
                get_nested(base, key)
            for seed in seeds:
                jobs.append({
                    "config_path": config_path,
                    "overrides": overrides,
                    "seed": seed,
                })
    return jobs


def apply_job(base_config, job):
    config = copy.deepcopy(base_config)
    for key, value in job["overrides"].items():
        set_nested(config, key, value)
    config["evolution"]["seed"] = int(job["seed"])
    return config


def run_name(job):
    # No seed in the name, so the repeats of one combination share it and group in W&B.
    stem = Path(job["config_path"]).stem
    parts = [stem]
    for key, value in job["overrides"].items():
        parts.append(f"{key.split('.')[-1]}={value}")
    return "_".join(str(part) for part in parts)


def format_job(job):
    overrides = ", ".join(f"{key}={value}" for key, value in job["overrides"].items()) or "(base config)"
    return f"{job['config_path']} | {overrides} | seed={job['seed']}"


def log_generation(generation, ga):
    import wandb

    metrics = ga.fitness_stats()
    metrics["generation"] = generation
    wandb.log(metrics, step=generation)


def run_job(job, entity, project):
    import wandb
    from genetic_algorithm import GeneticAlgorithm

    base_config = load_yaml(job["config_path"])
    config = apply_job(base_config, job)
    wandb_config = {
        "config_path": str(job["config_path"]),
        "seed": job["seed"],
        "overrides": job["overrides"],
        **config,
    }
    run = wandb.init(
        entity=entity,
        project=project,
        name=run_name(job),
        group=Path(job["config_path"]).stem,
        job_type="sweep",
        config=wandb_config,
        reinit=True,
    )
    try:
        ga = GeneticAlgorithm(config)
        best = ga.run(on_generation=log_generation)
        wandb.summary["best_fitness"] = None if best is None else best.fitness_value
        return best
    except Exception:
        wandb.summary["failed"] = True
        raise
    finally:
        run.finish()


def build_parser():
    parser = argparse.ArgumentParser(
        description="Sweep CA evolution configs. You choose the keys, values, and seeds.",
        epilog=(
            "Examples:\n"
            "  uv run sweep.py --seeds 42,7\n"
            "  uv run sweep.py --seeds 1,2,3 --sweep ca.local_functions=1,4,8 --sweep llm.code_context=true,false\n"
            "  uv run sweep.py --configs configurations/binary.yaml --seeds 42 --sweep ca.global_functions=0,4\n"
            "  uv run sweep.py --sweep-file my_sweep.yaml\n"
            "  uv run sweep.py --sweep-file my_sweep.yaml --shard 2/4\n"
            "  uv run sweep.py --list-keys"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-c", "--configs", nargs="*", default=None,
        help="Config YAML files or directories. Default: all experiment configs in configurations/",
    )
    parser.add_argument(
        "-s", "--sweep", action="append", default=[],
        help="Dotted config key and comma-separated values, e.g. ca.local_functions=1,4,8. Repeat for more keys.",
    )
    parser.add_argument(
        "--sweep-file", type=str, default=None,
        help="YAML with optional configs, seeds, sweep, entity, and project. CLI flags override the file.",
    )
    parser.add_argument(
        "--seeds", type=str, default=None,
        help="Comma-separated seeds, e.g. 42,7,123",
    )
    parser.add_argument(
        "--shard", type=str, default=None,
        help="Run one slice of the job grid, e.g. 2/4. Jobs are dealt round-robin across shards.",
    )
    parser.add_argument("--count", action="store_true", help="Print how many jobs the grid has and exit")
    parser.add_argument("--entity", type=str, default=None, help=f"W&B entity. Default: {DEFAULT_ENTITY}")
    parser.add_argument("--project", type=str, default=None, help=f"W&B project. Default: {DEFAULT_PROJECT}")
    parser.add_argument("--dry-run", action="store_true", help="Print the job grid without running")
    parser.add_argument("--list-keys", action="store_true", help="Print sweepable dotted keys and exit")
    return parser


if __name__ == "__main__":
    load_dotenv()
    args = build_parser().parse_args()

    file_sweep, file_seeds, file_configs, file_entity, file_project = {}, None, None, None, None
    if args.sweep_file:
        file_sweep, file_seeds, file_configs, file_entity, file_project = load_sweep_file(args.sweep_file)

    config_paths = discover_configs(args.configs if args.configs is not None else file_configs)
    if not config_paths:
        raise SystemExit("No config files found")

    if args.list_keys:
        print("Sweepable keys (from selected configs):")
        for path in config_paths:
            keys = flatten_keys(load_yaml(path))
            print(f"\n{path}")
            for key in keys:
                print(f"  {key}")
        raise SystemExit(0)

    sweep = dict(file_sweep)
    for raw in args.sweep:
        key, values = parse_sweep_arg(raw)
        sweep[key] = values

    if args.seeds is not None:
        seeds = parse_seeds(args.seeds)
    elif file_seeds is not None:
        seeds = file_seeds
    else:
        raise SystemExit("Select seeds with --seeds or a seeds list in --sweep-file")

    entity = args.entity or file_entity or DEFAULT_ENTITY
    project = args.project or file_project or DEFAULT_PROJECT

    jobs = expand_jobs(config_paths, sweep, seeds)
    if args.count:
        print(len(jobs))
        raise SystemExit(0)

    total = len(jobs)
    if args.shard:
        jobs = select_shard(jobs, parse_shard(args.shard))

    print(f"W&B: {entity}/{project}")
    if args.shard:
        print(f"Shard {args.shard}: {len(jobs)} of {total} job(s)")
    print(f"{len(jobs)} job(s) across {len(config_paths)} config(s), {len(seeds)} seed(s)")
    for index, job in enumerate(jobs, start=1):
        print(f"  [{index}/{len(jobs)}] {format_job(job)}")

    if args.dry_run:
        raise SystemExit(0)

    failures = []
    for index, job in enumerate(jobs, start=1):
        print(f"\nRunning [{index}/{len(jobs)}] {format_job(job)}")
        try:
            run_job(job, entity, project)
        except Exception as error:
            print(f"Failed [{index}/{len(jobs)}]: {error}")
            failures.append((job, error))

    if failures:
        print(f"\n{len(failures)} job(s) failed:")
        for job, error in failures:
            print(f"  {format_job(job)} -> {error}")
        raise SystemExit(1)

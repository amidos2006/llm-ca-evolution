from dotenv import load_dotenv
from pathlib import Path
import argparse
import json
import math
import re
import sys
import yaml

import numpy as np
import pcg_benchmark
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evolution.ca_chromosome import CAChromosome

RUN_DIR_RE = re.compile(
    r"^(?P<game>.+)_\d{8}-\d{6}_lf-(?P<lf>\d+)_gf-(?P<gf>\d+)_"
    r"steps-(?P<steps>\d+)_cc-(?P<cc>true|false)$"
)
DEFAULT_RUNS = 30
GRID_COLUMNS = 6
MONTAGE_SCALE = 2
LABEL_HEIGHT = 22
CELL_PADDING = 6


def load_yaml(path):
    with open(path, "r") as file:
        return yaml.safe_load(file)


def resolve_path(raw):
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def resolve_chromosome_path(raw):
    path = resolve_path(raw)
    if path.is_dir():
        candidate = path / "chromosome_0.json"
        if candidate.exists():
            return candidate
        raise SystemExit(f"No chromosome_0.json in {path}")
    if path.suffix.lower() == ".json":
        return path
    sibling = path.with_suffix(".json")
    if sibling.exists():
        print(f"{path.name} is not JSON; using {sibling.name}")
        return sibling
    raise SystemExit(
        f"{path} is not a chromosome JSON. Pass chromosome_*.json "
        f"(sweep folders also have a chromosome_0.png render next to it)."
    )


def load_chromosome_json(path):
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except UnicodeDecodeError:
        raise RuntimeError(
            f"{path} is not text/JSON (it looks like a binary file such as a PNG). "
            "Pass the chromosome_*.json, not the rendered image."
        )
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{path} is not valid chromosome JSON: {error}")


def infer_run_settings(chromosome_path):
    for parent in chromosome_path.resolve().parents:
        match = RUN_DIR_RE.match(parent.name)
        if match:
            return {
                "game": match.group("game"),
                "local_functions": int(match.group("lf")),
                "global_functions": int(match.group("gf")),
                "steps": int(match.group("steps")),
                "code_context": match.group("cc") == "true",
                "run_dir": parent,
            }
    return None


def default_config_path(game):
    path = PROJECT_ROOT / "configurations" / f"{game}.yaml"
    if path.exists():
        return path
    return None


def resolve_prompt_paths(config):
    for key in ("system_prompt", "user_function_prompt", "user_execute_prompt"):
        raw = config["llm"][key]
        path = Path(raw)
        if not path.is_absolute():
            config["llm"][key] = str(PROJECT_ROOT / path)


def build_config(chromosome_data, chromosome_path, config_path=None, steps=None):
    inferred = infer_run_settings(chromosome_path)
    if config_path:
        config_path = resolve_path(config_path)
    elif inferred:
        config_path = default_config_path(inferred["game"])
        if config_path is None:
            raise RuntimeError(
                f"No configurations/{inferred['game']}.yaml for run folder {inferred['run_dir'].name}"
            )
    else:
        raise RuntimeError(
            "Pass --config, or use a chromosome saved under a sweep folder "
            "named like sokoban_YYYYMMDD-HHMMSS_lf-1_gf-5_steps-10_cc-true"
        )

    if not config_path.exists():
        raise RuntimeError(f"No config at {config_path}")

    config = load_yaml(config_path)
    resolve_prompt_paths(config)

    local_functions = chromosome_data.get("local_functions") or []
    global_functions = chromosome_data.get("global_functions") or []
    config["ca"]["local_functions"] = len(local_functions)
    config["ca"]["global_functions"] = len(global_functions)

    if steps is not None:
        config["ca"]["steps"] = int(steps)
    elif inferred is not None:
        config["ca"]["steps"] = inferred["steps"]

    if inferred is not None:
        config["llm"]["code_context"] = inferred["code_context"]
        mismatches = []
        if inferred["local_functions"] != len(local_functions):
            mismatches.append(
                f"lf folder={inferred['local_functions']} json={len(local_functions)}"
            )
        if inferred["global_functions"] != len(global_functions):
            mismatches.append(
                f"gf folder={inferred['global_functions']} json={len(global_functions)}"
            )
        if mismatches:
            print(f"Warning: chromosome counts differ from the run folder ({', '.join(mismatches)})")

    return config, inferred


def load_chromosome(config, chromosome_data):
    chromosome = CAChromosome(
        None,
        config,
        local_functions=chromosome_data.get("local_functions") or [],
        global_functions=chromosome_data.get("global_functions") or [],
        execution_function=chromosome_data.get("execute_function") or "",
    )
    chromosome.fitness_value = chromosome_data.get("fitness")
    return chromosome


def step_usage(step_fraction, max_steps):
    # execute() returns `step / ca.steps` from a 0-based loop, so 0.0 is the first update.
    step_index = int(round(float(step_fraction) * max_steps))
    steps_used = min(max_steps, max(1, step_index + 1))
    return steps_used, 100.0 * steps_used / max_steps


def replay_summary_path(chromosome_path):
    return Path(chromosome_path).parent / f"{Path(chromosome_path).stem}_replay" / "summary.json"


def average_steps_from_summary(data):
    if data.get("average_steps_used") is not None:
        return float(data["average_steps_used"])
    used = [
        generation["steps_used"]
        for generation in data.get("generations") or []
        if generation.get("steps_used") is not None
    ]
    if used:
        return float(np.mean(used))
    percent = data.get("average_percent_steps")
    max_steps = data.get("max_steps")
    if percent is not None and max_steps:
        return float(percent) / 100.0 * float(max_steps)
    return None


def read_cached_summary(chromosome_path, runs):
    path = replay_summary_path(chromosome_path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if int(data.get("runs") or 0) != int(runs):
        return None
    if average_steps_from_summary(data) is None:
        return None
    return data


def write_summary(chromosome_path, summary, overwrite=False):
    path = replay_summary_path(chromosome_path)
    if path.exists() and not overwrite:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = dict(summary)
    serializable["chromosome"] = str(chromosome_path)
    with open(path, "w") as file:
        json.dump(serializable, file, indent=2)
    return path


def measure_chromosome(chromosome_path, runs=DEFAULT_RUNS, seed=None, steps=None, config_path=None):
    """Run execute() `runs` times with no rendering. Returns a summary dict."""
    chromosome_path = Path(chromosome_path)
    chromosome_data = load_chromosome_json(chromosome_path)
    config, _inferred = build_config(
        chromosome_data, chromosome_path, config_path=config_path, steps=steps,
    )
    max_steps = int(config["ca"]["steps"])
    seed = int(config["evolution"]["seed"]) if seed is None else int(seed)

    chromosome = load_chromosome(config, chromosome_data)
    compile_errors = chromosome.compile()
    if compile_errors > 0.0:
        raise RuntimeError(f"Chromosome failed to compile (error fraction {compile_errors:.3f})")
    if chromosome.execute_function is None:
        raise RuntimeError("Chromosome compiled but has no execute() function")

    env = pcg_benchmark.make(config["environment"]["name"])
    env.content_space.seed(seed)

    generations = []
    percents = []
    steps_used_list = []
    completed_count = 0
    error_count = 0
    for index in range(runs):
        errors, level, step_fraction = chromosome.execute(env)
        record = {
            "run": index,
            "error": False,
            "completed": False,
            "quality": None,
            "steps_used": None,
            "max_steps": max_steps,
            "step_fraction": None,
            "percent": None,
        }
        if errors > 0.0 or level is None or step_fraction is None or step_fraction < 0:
            record["error"] = True
            error_count += 1
            generations.append(record)
            continue
        used, percent = step_usage(step_fraction, max_steps)
        quality, *_ = env.evaluate(level)
        completed = bool(quality >= 1.0)
        record.update({
            "completed": completed,
            "quality": float(quality),
            "steps_used": used,
            "step_fraction": float(step_fraction),
            "percent": percent,
        })
        percents.append(percent)
        steps_used_list.append(used)
        completed_count += int(completed)
        generations.append(record)

    return {
        "chromosome": str(chromosome_path),
        "environment": config["environment"]["name"],
        "seed": seed,
        "runs": runs,
        "max_steps": max_steps,
        "successful_runs": len(steps_used_list),
        "completed_runs": completed_count,
        "error_runs": error_count,
        "average_percent_steps": float(np.mean(percents)) if percents else None,
        "average_steps_used": float(np.mean(steps_used_list)) if steps_used_list else None,
        "generations": generations,
    }


def render_level(env, level, path):
    image = env.render(level)
    image.save(path)
    return image


def font():
    for name in ("DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, 14)
        except OSError:
            continue
    return ImageFont.load_default()


def labeled_tile(image, text, width, height, fill):
    canvas = Image.new("RGB", (width, height + LABEL_HEIGHT), (18, 18, 18))
    resized = image.convert("RGB").resize((width, height), Image.NEAREST)
    canvas.paste(resized, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, height + 4), text, fill=fill, font=font())
    return canvas


def save_montage(records, output_dir, tile_size):
    if not records:
        return None, None
    columns = min(GRID_COLUMNS, len(records))
    rows = math.ceil(len(records) / columns)
    cell_w = tile_size[0] * MONTAGE_SCALE
    cell_h = tile_size[1] * MONTAGE_SCALE
    page_w = columns * (cell_w + CELL_PADDING) + CELL_PADDING
    page_h = rows * (cell_h + LABEL_HEIGHT + CELL_PADDING) + CELL_PADDING
    page = Image.new("RGB", (page_w, page_h), (8, 8, 8))
    placeholder = Image.new("RGB", (cell_w, cell_h), (40, 40, 40))
    for index, record in enumerate(records):
        image = record.get("image") or placeholder
        if record["error"]:
            label, fill = f"#{index:02d} error", (220, 80, 80)
        elif record["completed"]:
            label, fill = f"#{index:02d} {record['steps_used']}/{record['max_steps']}", (120, 200, 120)
        else:
            label, fill = f"#{index:02d} {record['steps_used']}/{record['max_steps']}", (230, 200, 80)
        tile = labeled_tile(image, label, cell_w, cell_h, fill)
        row, col = divmod(index, columns)
        x = CELL_PADDING + col * (cell_w + CELL_PADDING)
        y = CELL_PADDING + row * (cell_h + LABEL_HEIGHT + CELL_PADDING)
        page.paste(tile, (x, y))
    png_path = output_dir / "montage.png"
    pdf_path = output_dir / "montage.pdf"
    page.save(png_path)
    page.save(pdf_path, "PDF", resolution=150)
    return png_path, pdf_path


def default_output_dir(chromosome_path):
    return chromosome_path.parent / f"{chromosome_path.stem}_replay"


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Load a saved CA chromosome JSON, run its execute() function, "
            "report how much of the CA step budget each generation used, "
            "and render every generated level."
        )
    )
    parser.add_argument(
        "chromosome", type=str,
        help="Path to a chromosome_*.json file, a generation folder, or the sibling chromosome_0.png",
    )
    parser.add_argument(
        "-c", "--config", type=str, default=None,
        help="Experiment YAML. Inferred from the sweep folder name when omitted.",
    )
    parser.add_argument(
        "-n", "--runs", type=int, default=DEFAULT_RUNS,
        help=f"How many times to generate (default: {DEFAULT_RUNS})",
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        help="Override ca.steps. Default: sweep folder value, else the config file.",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Directory for PNGs and summary.json. Default: <chromosome>_replay/ next to the JSON.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Seed for env.content_space. Default: evolution.seed from the config.",
    )
    return parser


if __name__ == "__main__":
    load_dotenv(PROJECT_ROOT / ".env")
    args = build_parser().parse_args()
    chromosome_path = resolve_chromosome_path(args.chromosome)
    if not chromosome_path.exists():
        raise SystemExit(f"No chromosome at {chromosome_path}")

    try:
        chromosome_data = load_chromosome_json(chromosome_path)
        config, inferred = build_config(
            chromosome_data, chromosome_path, config_path=args.config, steps=args.steps,
        )
    except RuntimeError as error:
        raise SystemExit(error)
    max_steps = int(config["ca"]["steps"])
    seed = args.seed if args.seed is not None else int(config["evolution"]["seed"])
    output_dir = resolve_path(args.output) if args.output else default_output_dir(chromosome_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "generations"
    images_dir.mkdir(parents=True, exist_ok=True)

    chromosome = load_chromosome(config, chromosome_data)
    compile_errors = chromosome.compile()
    if compile_errors > 0.0:
        raise SystemExit(f"Chromosome failed to compile (error fraction {compile_errors:.3f})")
    if chromosome.execute_function is None:
        raise SystemExit("Chromosome compiled but has no execute() function")

    env = pcg_benchmark.make(config["environment"]["name"])
    env.content_space.seed(seed)

    records = []
    percents = []
    tile_size = None
    for index in range(args.runs):
        errors, level, step_fraction = chromosome.execute(env)
        record = {
            "run": index,
            "error": False,
            "completed": False,
            "quality": None,
            "steps_used": None,
            "max_steps": max_steps,
            "step_fraction": None,
            "percent": None,
            "image_path": None,
        }
        if errors > 0.0 or level is None or step_fraction is None or step_fraction < 0:
            record["error"] = True
            print(f"[{index:02d}] runtime error")
            records.append(record)
            continue

        steps_used, percent = step_usage(step_fraction, max_steps)
        quality, *_ = env.evaluate(level)
        completed = bool(quality >= 1.0)
        image_path = images_dir / f"generation_{index:02d}.png"
        image = None
        try:
            image = render_level(env, level, image_path)
            if tile_size is None:
                tile_size = image.size
            record["image_path"] = str(image_path.relative_to(output_dir))
        except Exception as error:
            print(f"[{index:02d}] render failed: {error}")

        record.update({
            "completed": completed,
            "quality": float(quality),
            "steps_used": steps_used,
            "step_fraction": float(step_fraction),
            "percent": percent,
            "image": image,
        })
        percents.append(percent)
        status = "complete" if completed else "incomplete"
        print(
            f"[{index:02d}] {status} in {steps_used}/{max_steps} steps "
            f"({percent:.1f}%), quality={quality:.3f}"
        )
        records.append(record)

    average = float(np.mean(percents)) if percents else None
    completed_count = sum(1 for record in records if record["completed"])
    error_count = sum(1 for record in records if record["error"])
    summary = {
        "chromosome": str(chromosome_path),
        "environment": config["environment"]["name"],
        "seed": seed,
        "runs": args.runs,
        "max_steps": max_steps,
        "successful_runs": len(percents),
        "completed_runs": completed_count,
        "error_runs": error_count,
        "average_percent_steps": average,
        "average_steps_used": float(np.mean([
            record["steps_used"] for record in records if record["steps_used"] is not None
        ])) if any(record["steps_used"] is not None for record in records) else None,
        "generations": [
            {key: value for key, value in record.items() if key != "image"}
            for record in records
        ],
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as file:
        json.dump(summary, file, indent=2)

    montage_png, montage_pdf = None, None
    if tile_size is not None:
        montage_png, montage_pdf = save_montage(records, output_dir, tile_size)

    print()
    if average is None:
        print("No successful generations")
    else:
        print(
            f"Average steps to complete across {len(percents)} generation(s): "
            f"{average:.2f}% of {max_steps} CA steps"
        )
        print(f"Completed (quality >= 1): {completed_count}/{args.runs}")
        if error_count:
            print(f"Runtime errors: {error_count}/{args.runs}")
    print(f"Renders: {images_dir}")
    if montage_png is not None:
        print(f"Montage: {montage_png}")
        print(f"Montage PDF: {montage_pdf}")
    print(f"Summary: {summary_path}")

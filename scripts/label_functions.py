from dotenv import load_dotenv
from pathlib import Path
import argparse
import csv
import json
import re
import sys
import time

import anthropic
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evolution import parallel
from replay_chromosome import RUN_DIR_RE
from utils import usage

DEFAULT_MODEL = "claude-opus-4-8"
DEFAULT_GENERATION = 50
DEFAULT_KIND = "both"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_EFFORT = "low"
DEFAULT_MAX_FUNCTIONS = 15
GAME_ORDER = ("binary", "zelda", "sokoban")
RUN_PREFIX_RE = re.compile(r"^(binary|zelda|sokoban)(_|$)")
FUNCTION_NAME_RE = re.compile(r"^def\s+(\w+)\s*\(", re.MULTILINE)
JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

KIND_HELP = {
    "local": (
        "a local helper. It receives a small binary neighborhood of one tile type "
        "around a single cell and returns a scalar feature."
    ),
    "global": (
        "a global helper. It receives a binary mask of one tile type over the whole "
        "map and returns a scalar feature."
    ),
}

SYSTEM_PROMPT = """\
You label helper functions from evolved cellular-automata level generators.
Each function is a small feature detector: it sees a binary grid of one tile type
and returns a number the generator later uses to decide the next tile.

Give the function a short general name (a tag) such as "counting function",
"path function", "connectivity function", "border detector", "clustering function",
or "distance function". Invent a similarly terse tag if none of those fit.
Then write one or two sentences explaining what it computes.

Reply with a JSON object only, no markdown fences:
{"tag": "...", "explanation": "..."}
"""


def resolve_path(raw):
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def default_output_dir():
    return PROJECT_ROOT / "results" / "function_labels"


def load_game_descriptions():
    descriptions = {}
    config_dir = PROJECT_ROOT / "configurations"
    if not config_dir.exists():
        return descriptions
    for path in config_dir.glob("*.yaml"):
        try:
            data = yaml.safe_load(path.read_text())
        except (OSError, yaml.YAMLError):
            continue
        environment = data.get("environment") or {}
        name = str(environment.get("name") or path.stem)
        game = name.rsplit("-", 1)[0] if "-" in name else name
        descriptions[game] = environment.get("description") or ""
        descriptions[path.stem] = environment.get("description") or ""
    return descriptions


def parse_run_folder(path):
    match = RUN_DIR_RE.match(path.name)
    if not match:
        return {"game": path.name.split("_", 1)[0], "local": None, "global": None}
    return {
        "game": match.group("game"),
        "local": int(match.group("lf")),
        "global": int(match.group("gf")),
    }


def is_game_run_dir(path):
    return path.is_dir() and RUN_PREFIX_RE.match(path.name) is not None


def discover_run_dirs(path):
    if is_game_run_dir(path):
        return [path]
    if not path.is_dir():
        return []
    return sorted(
        child for child in path.iterdir()
        if is_game_run_dir(child)
    )


def expand_folder_args(paths):
    folders = []
    seen = set()
    for raw in paths:
        path = resolve_path(raw)
        if not path.is_dir():
            raise SystemExit(f"Not a directory: {path}")
        found = discover_run_dirs(path)
        if not found:
            raise SystemExit(
                f"No binary/zelda/sokoban run folders in {path}. "
                "Pass the results directory or a specific run folder."
            )
        for folder in found:
            key = str(folder.resolve())
            if key not in seen:
                seen.add(key)
                folders.append(folder)
    return folders


def generation_dir(run_dir, generation):
    if run_dir.name == f"generation_{generation}" and run_dir.is_dir():
        return run_dir
    candidate = run_dir / f"generation_{generation}"
    if candidate.is_dir():
        return candidate
    return None


def best_chromosome_path(directory):
    # Population is saved fitness-descending, so chromosome_0 is the elite.
    if directory is None:
        return None
    path = directory / "chromosome_0.json"
    return path if path.exists() else None


def normalize_code(code):
    return "\n".join(line.rstrip() for line in str(code).strip().splitlines()).strip()


def count_lines(code):
    if not code:
        return 0
    return len(code.splitlines())


def function_name(code, fallback):
    match = FUNCTION_NAME_RE.search(code)
    return match.group(1) if match else fallback


def selected_kinds(kind):
    if kind == "both":
        return ("local", "global")
    return (kind,)


def collect_functions(run_dir, generation, kinds):
    folder = parse_run_folder(run_dir)
    chromosome_path = best_chromosome_path(generation_dir(run_dir, generation))
    if chromosome_path is None:
        return None
    with open(chromosome_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    records = []
    for kind in kinds:
        sources = data.get(f"{kind}_functions") or []
        for index, code in enumerate(sources):
            text = normalize_code(code)
            if not text:
                continue
            records.append({
                "folder": run_dir.name,
                "game": folder["game"],
                "chromosome": 0,
                "kind": kind,
                "index": index,
                "name": function_name(text, f"{kind}_{index}"),
                "code": text,
                "lines": count_lines(text),
                "fitness": data.get("fitness"),
            })
    return records


def cache_key(record):
    return f"{record['game']}|{record['kind']}|{record['code']}"


def load_cache(path):
    if path is None or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_cache(path, cache):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2) + "\n")


def extract_json(text):
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = JSON_OBJECT_RE.search(stripped)
        if match:
            return json.loads(match.group(0))
        raise


def label_prompt(record, descriptions):
    game = record["game"]
    description = descriptions.get(game, "")
    game_line = f"Game: {game}."
    if description:
        game_line += f" {description}"
    return (
        f"{game_line}\n"
        f"This is {KIND_HELP[record['kind']]}\n\n"
        f"Function `{record['name']}`:\n"
        f"```python\n{record['code']}\n```"
    )


def call_claude(client, model, max_tokens, effort, prompt):
    attempt = 0
    while True:
        try:
            with parallel.api_slot():
                response = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    output_config={"effort": effort},
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
            usage.record(response)
            text = next((block.text for block in response.content if block.type == "text"), None)
            if text is None:
                raise RuntimeError(
                    f"No text block in Claude response "
                    f"(stop_reason={response.stop_reason}, "
                    f"blocks={[block.type for block in response.content]})"
                )
            parsed = extract_json(text)
            tag = str(parsed.get("tag") or "").strip()
            explanation = str(parsed.get("explanation") or "").strip()
            if not tag or not explanation:
                raise ValueError(f"Incomplete JSON label: {parsed}")
            return tag, explanation
        except Exception as error:
            attempt += 1
            delay = parallel.backoff_delay(attempt)
            print(f"Error labeling function: {error}. Retrying in {delay:.1f}s...")
            time.sleep(delay)


def unique_jobs(records):
    jobs = []
    seen = set()
    for record in records:
        key = cache_key(record)
        if key in seen:
            continue
        seen.add(key)
        jobs.append(record)
    return jobs


def apply_labels(records, labels):
    labeled = []
    for record in records:
        tag, explanation = labels[cache_key(record)]
        labeled.append({
            "folder": record["folder"],
            "game": record["game"],
            "chromosome": record["chromosome"],
            "kind": record["kind"],
            "index": record["index"],
            "name": record["name"],
            "tag": tag,
            "explanation": explanation,
            "lines": record["lines"],
            "code": record["code"],
            "fitness": record["fitness"],
        })
    return labeled


def write_json(path, records):
    path.write_text(json.dumps(records, indent=2) + "\n")


def write_csv(path, records):
    fields = ("folder", "chromosome", "kind", "name", "lines", "tag", "explanation")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def folder_line_summary(records):
    total = sum(record["lines"] for record in records)
    by_kind = {}
    for record in records:
        by_kind[record["kind"]] = by_kind.get(record["kind"], 0) + record["lines"]
    parts = [f"{len(records)} function(s), {total} line(s)"]
    if len(by_kind) > 1:
        kind_text = ", ".join(f"{kind} {count}" for kind, count in by_kind.items())
        parts.append(f"({kind_text})")
    return total, " ".join(parts)


def count_by_game(folder_to_game):
    counts = {game: 0 for game in GAME_ORDER}
    for game in folder_to_game.values():
        counts[game] = counts.get(game, 0) + 1
    return counts


def high_level_summary(records, folder_to_game, skipped=0):
    instances = {"local": 0, "global": 0}
    types = {"local": set(), "global": set()}
    for record in records:
        kind = record["kind"]
        instances[kind] = instances.get(kind, 0) + 1
        tag = record.get("tag")
        if tag:
            types.setdefault(kind, set()).add(normalize_tag(tag))
    return {
        "experiments": len(folder_to_game),
        "skipped": skipped,
        "games": count_by_game(folder_to_game),
        "local_instances": instances.get("local", 0),
        "global_instances": instances.get("global", 0),
        "local_types": len(types.get("local", ())),
        "global_types": len(types.get("global", ())),
    }


def format_game_counts(games):
    return ", ".join(f"{game} {games.get(game, 0)}" for game in GAME_ORDER)


def format_summary_lines(summary):
    return [
        f"Experiments: {summary['experiments']} ({format_game_counts(summary['games'])})",
        (
            f"Local functions: {summary['local_instances']} "
            f"({summary['local_types']} types)"
        ),
        (
            f"Global functions: {summary['global_instances']} "
            f"({summary['global_types']} types)"
        ),
    ]


def write_summary(path, summary):
    lines = ["# Function summary", ""]
    lines.extend(f"- {line}" for line in format_summary_lines(summary))
    if summary.get("skipped"):
        lines.append(
            f"- Skipped: {summary['skipped']} folder(s) without "
            "generation_50/chromosome_0.json"
        )
    lines.append("")
    path.write_text("\n".join(lines))


def write_markdown(path, records, summary=None):
    lines = ["# Function labels", ""]
    if summary:
        lines.append("## High-level view")
        lines.append("")
        lines.extend(f"- {line}" for line in format_summary_lines(summary))
        lines.append("")
    current_folder = None
    current_chromosome = None
    folder_records = []

    def flush_folder():
        if not folder_records:
            return
        _total, folder_summary = folder_line_summary(folder_records)
        lines.append(f"*{folder_summary}*")
        lines.append("")

    for record in records:
        if record["folder"] != current_folder:
            flush_folder()
            current_folder = record["folder"]
            current_chromosome = None
            folder_records = []
            lines.append(f"## {current_folder}")
            lines.append("")
        folder_records.append(record)
        if record["chromosome"] != current_chromosome:
            current_chromosome = record["chromosome"]
            lines.append(f"### chromosome_{current_chromosome}")
            lines.append("")
        unit = "line" if record["lines"] == 1 else "lines"
        lines.append(f"- **{record['name']}** — {record['tag']} ({record['lines']} {unit})")
        lines.append(f"  {record['explanation']}")
        lines.append("")
    flush_folder()
    path.write_text("\n".join(lines).rstrip() + "\n")


def latex_escape(text):
    mapping = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(mapping.get(char, char) for char in str(text))


def normalize_tag(tag):
    return " ".join(str(tag).lower().split())


def display_tag(tag):
    return normalize_tag(tag).title()


def mean_std(values):
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return None, None
    if array.size < 2:
        return float(array.mean()), None
    return float(array.mean()), float(array.std(ddof=1))


def format_lines_cell(mean, std):
    if mean is None:
        return "---"
    if std is None:
        return f"${mean:.0f}$"
    return f"${mean:.0f} \\pm {std:.0f}$"


def format_percent_cell(value):
    if value is None:
        return "---"
    return f"${value:.0f}\\%$"


def pick_description(group):
    explanations = [record["explanation"].strip() for record in group if record.get("explanation")]
    if not explanations:
        return ""
    return min(explanations, key=len)


def folder_games(records, folders=None):
    games = {}
    if folders:
        for path in folders:
            games[path.name] = parse_run_folder(path)["game"]
    for record in records:
        games.setdefault(record["folder"], record["game"])
    return games


def aggregate_function_types(records, kind, folder_to_game, max_types):
    group = [record for record in records if record["kind"] == kind]
    by_tag = {}
    for record in group:
        by_tag.setdefault(normalize_tag(record["tag"]), []).append(record)

    runs_by_game = {}
    for folder, game in folder_to_game.items():
        runs_by_game.setdefault(game, set()).add(folder)

    rows = []
    for tag, members in by_tag.items():
        present = {}
        for record in members:
            present.setdefault(record["game"], set()).add(record["folder"])
        percents = {}
        for game in GAME_ORDER:
            total = len(runs_by_game.get(game, ()))
            if total == 0:
                percents[game] = None
            else:
                percents[game] = 100.0 * len(present.get(game, ())) / total
        mean, std = mean_std([record["lines"] for record in members])
        rows.append({
            "tag": tag,
            "name": display_tag(tag),
            "description": pick_description(members),
            "mean": mean,
            "std": std,
            "percents": percents,
            "run_count": sum(len(folders) for folders in present.values()),
            "occurrence_count": len(members),
        })
    rows.sort(key=lambda row: (-row["run_count"], -row["occurrence_count"], row["name"]))
    if max_types and max_types > 0:
        rows = rows[:max_types]
    return rows


def build_function_table(records, kind, folder_to_game, max_types):
    rows = aggregate_function_types(records, kind, folder_to_game, max_types)
    if not rows:
        return None
    total_runs = len(folder_to_game)
    games = format_game_counts(count_by_game(folder_to_game))
    header = "Name & Description & \\# Lines & Binary & Zelda & Sokoban"
    lines = [
        "\\begin{table}",
        "    \\centering",
        "    \\begin{tabular}{|l|p{4.6cm}|c|ccc|}",
        "        \\hline",
        f"        {header} \\\\",
        "        \\hline",
        "        \\hline",
    ]
    for row in rows:
        percents = " & ".join(format_percent_cell(row["percents"][game]) for game in GAME_ORDER)
        lines.append(
            "        "
            f"{latex_escape(row['name'])} & {latex_escape(row['description'])} & "
            f"{format_lines_cell(row['mean'], row['std'])} & {percents} \\\\"
        )
    lines.extend([
        "        \\hline",
        "    \\end{tabular}",
        "    \\caption{The percentage of time that the "
        f"{kind} function appears in all the experiments and runs "
        f"(total of {total_runs}; {games}) and the average number of lines "
        "for each of these functions.}",
        f"    \\label{{tab:{kind}_functions}}",
        "\\end{table}",
        "",
    ])
    return "\n".join(lines)


def write_function_tables(output_dir, records, folder_to_game, max_types):
    written = []
    kinds = []
    for kind in ("global", "local"):
        if any(record["kind"] == kind for record in records):
            kinds.append(kind)
    for kind in kinds:
        table = build_function_table(records, kind, folder_to_game, max_types)
        if not table:
            continue
        path = output_dir / f"{kind}_functions.tex"
        path.write_text(table)
        written.append(path)
    return written


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Label the best generation_50 chromosome in each Binary, Zelda, and "
            "Sokoban run. Pass the results directory to pick up every matching "
            "folder, then write tags, line counts, a high-level summary, and "
            "LaTeX tables whose percentages use all experiments per game."
        ),
        epilog=(
            "Examples:\n"
            "  uv run scripts/label_functions.py results\n"
            "  uv run scripts/label_functions.py results --kind global --max-functions 15\n"
            "  uv run scripts/label_functions.py results --dry-run\n"
            "  uv run scripts/label_functions.py results --from-labels results/function_labels/labels.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "folders", nargs="*",
        help="The results directory, or one or more run folders. Directories are "
             "scanned for subfolders whose names start with binary, zelda, or sokoban.",
    )
    parser.add_argument(
        "--kind", choices=("local", "global", "both"), default=DEFAULT_KIND,
        help="Which helpers to label. Default: both.",
    )
    parser.add_argument(
        "--generation", type=int, default=DEFAULT_GENERATION,
        help=f"Generation folder to read. Default: {DEFAULT_GENERATION}",
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default=None,
        help="Where to write labels and LaTeX tables. Default: results/function_labels/",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Claude model. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
        help=f"max_tokens for each labeling call. Default: {DEFAULT_MAX_TOKENS}",
    )
    parser.add_argument(
        "--effort", type=str, default=DEFAULT_EFFORT,
        help=f"Claude effort. Default: {DEFAULT_EFFORT}",
    )
    parser.add_argument(
        "--max-concurrency", type=int, default=6,
        help="In-flight API calls. Default: 6",
    )
    parser.add_argument(
        "--max-functions", type=int, default=DEFAULT_MAX_FUNCTIONS,
        help="Maximum function types (rows) per LaTeX table, most common first. "
             f"Default: {DEFAULT_MAX_FUNCTIONS}",
    )
    parser.add_argument(
        "--from-labels", type=str, default=None,
        help="Build outputs from an existing labels.json. Pass the results "
             "directory as well so percentages use every experiment per game.",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Ignore and do not update the label cache in the output directory.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List the functions that would be labeled without calling the API.",
    )
    return parser


def write_outputs(output_dir, labeled, folder_to_game, max_functions, write_labels=True, skipped=0):
    summary = high_level_summary(labeled, folder_to_game, skipped)
    if write_labels:
        json_path = output_dir / "labels.json"
        csv_path = output_dir / "labels.csv"
        markdown_path = output_dir / "labels.md"
        write_json(json_path, labeled)
        write_csv(csv_path, labeled)
        write_markdown(markdown_path, labeled, summary)
        print(f"JSON: {json_path}")
        print(f"CSV: {csv_path}")
        print(f"Markdown: {markdown_path}")
    summary_path = output_dir / "summary.md"
    write_summary(summary_path, summary)
    print(f"Summary: {summary_path}")
    for line in format_summary_lines(summary):
        print(line)
    if summary.get("skipped"):
        print(
            f"Skipped: {summary['skipped']} folder(s) without "
            "generation_50/chromosome_0.json"
        )
    for path in write_function_tables(output_dir, labeled, folder_to_game, max_functions):
        print(f"Table: {path}")


if __name__ == "__main__":
    load_dotenv(PROJECT_ROOT / ".env")
    args = build_parser().parse_args()
    output_dir = resolve_path(args.output_dir) if args.output_dir else default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    folders = expand_folder_args(args.folders) if args.folders else []
    if folders:
        discovered = count_by_game({folder.name: parse_run_folder(folder)["game"] for folder in folders})
        print(
            f"Discovered {len(folders)} run folder(s) "
            f"({format_game_counts(discovered)})"
        )

    if args.from_labels:
        labels_path = resolve_path(args.from_labels)
        labeled = json.loads(labels_path.read_text())
        if not labeled:
            raise SystemExit(f"No records in {labels_path}")
        skipped = 0
        completed = []
        for folder in folders:
            if best_chromosome_path(generation_dir(folder, args.generation)) is None:
                skipped += 1
            else:
                completed.append(folder)
        folder_to_game = folder_games(labeled, completed or None)
        write_outputs(
            output_dir, labeled, folder_to_game, args.max_functions,
            write_labels=False, skipped=skipped,
        )
        raise SystemExit(0)

    if not folders:
        raise SystemExit(
            "Pass the results directory (or run folders), "
            "or --from-labels results/function_labels/labels.json"
        )

    kinds = selected_kinds(args.kind)
    records = []
    completed = []
    skipped = 0
    for folder in folders:
        found = collect_functions(folder, args.generation, kinds)
        if found is None:
            skipped += 1
            continue
        completed.append(folder)
        records.extend(found)

    print(
        f"Using {len(completed)} experiment(s) with "
        f"generation_{args.generation}/chromosome_0.json"
    )
    if skipped:
        print(f"Skipped {skipped} folder(s) without a final chromosome")
    if not records:
        raise SystemExit("No functions found")

    jobs = unique_jobs(records)
    print(f"{len(records)} occurrence(s), {len(jobs)} unique function(s) to label")
    cache_path = None if args.no_cache else output_dir / "label_cache.json"
    folder_to_game = folder_games(records, completed)

    if args.dry_run:
        summary = high_level_summary(records, folder_to_game, skipped)
        for line in format_summary_lines(summary):
            print(line)
        if len(jobs) <= 40:
            for record in jobs:
                preview = record["code"].splitlines()[0] if record["code"] else ""
                print(f"  {record['folder']} {record['name']} ({record['lines']} lines): {preview}")
        else:
            print(f"Omitting {len(jobs)} unique function listing; pass fewer folders to preview.")
        raise SystemExit(0)

    cache = load_cache(cache_path)
    pending = [record for record in jobs if cache_key(record) not in cache]
    print(f"{len(jobs) - len(pending)} cached, {len(pending)} API call(s)")

    parallel.configure(args.max_concurrency)
    client = anthropic.Client()
    descriptions = load_game_descriptions()
    mark = usage.totals()

    def label_one(record):
        tag, explanation = call_claude(
            client, args.model, args.max_tokens, args.effort,
            label_prompt(record, descriptions),
        )
        print(f"  {record['folder']} {record['name']}: {tag}")
        return cache_key(record), tag, explanation

    if pending:
        for key, tag, explanation in parallel.run_parallel(label_one, pending):
            cache[key] = [tag, explanation]
        if cache_path is not None:
            save_cache(cache_path, cache)

    missing = [record for record in jobs if cache_key(record) not in cache]
    if missing:
        raise SystemExit(f"Missing labels for {len(missing)} function(s)")

    labels = {key: tuple(value) for key, value in cache.items()}
    labeled = apply_labels(records, labels)
    write_outputs(output_dir, labeled, folder_to_game, args.max_functions, skipped=skipped)
    spent = usage.delta(mark)
    if spent["calls"]:
        print(usage.describe(spent))

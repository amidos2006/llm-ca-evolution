from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from pathlib import Path
import argparse
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from replay_chromosome import (
    RUN_DIR_RE,
    average_steps_from_summary,
    measure_chromosome,
    read_cached_summary,
    write_summary,
)

DEFAULT_ENTITY = "munasir"
DEFAULT_REPLAY_RUNS = 30
HISTORY_SAMPLES = 2000
HISTORY_WORKERS = 8
PANEL_WIDTH = 5.2
PANEL_HEIGHT = 4.0
# Wong, Nature Methods 2011. Skip the yellow — it vanishes on white. First two (blue,
# vermillion) stay separable in greyscale and for the common colour-vision deficiencies.
SERIES_COLOURS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#E69F00",
    "#332288",
)
# Two-sided 95% t critical values by degrees of freedom, so the bands stay honest at 2-5
# seeds without pulling in scipy. Anything larger falls back to the normal 1.96.
T_CRITICAL_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
}

TABLE_HEADER = ["Game", "Local functions", "Global functions", "Best fitness", "Avg. iterations"]


def config_value(config, dotted_key):
    """Configs come back nested, but a flat 'ca.steps' key shows up on older runs."""
    if dotted_key in config:
        return config[dotted_key]
    current = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def game_name(run):
    name = config_value(run.config, "environment.name")
    if name:
        # sokoban-v0 -> sokoban
        text = str(name)
        return text.rsplit("-", 1)[0] if "-" in text else text
    if run.group:
        return str(run.group)
    return str(run.name).split("_", 1)[0]


def as_number(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def collect_runs(entity, project, metric):
    import wandb

    api = wandb.Api()
    collected, skipped = [], 0
    for run in api.runs(f"{entity}/{project}"):
        summary = dict(run.summary)
        value = summary.get(metric)
        if run.state != "finished" or summary.get("failed") or value is None:
            skipped += 1
            continue
        local = as_number(config_value(run.config, "ca.local_functions"))
        globals_ = as_number(config_value(run.config, "ca.global_functions"))
        steps = as_number(config_value(run.config, "ca.steps"))
        if local is None or globals_ is None or steps is None:
            skipped += 1
            continue
        collected.append({
            "run": run,
            "key": (game_name(run), local, globals_, steps),
            "value": float(value),
        })
    return collected, skipped


def group_runs(records):
    # code_context is not one of the table columns, so runs that differ only in it
    # land in the same cell and are averaged together.
    grouped = {}
    for record in records:
        grouped.setdefault(record["key"], []).append(record)
    return dict(sorted(grouped.items()))


def mean_std(values):
    array = np.asarray(values, dtype=float)
    if array.size < 2:
        return float(array.mean()), None
    return float(array.mean()), float(array.std(ddof=1))


def confidence_interval(values):
    """Half-width of the 95% CI. Seed counts are small, so use t rather than 1.96."""
    array = np.asarray(values, dtype=float)
    if array.size < 2:
        return 0.0
    critical = T_CRITICAL_95.get(array.size - 1, 1.96)
    return float(critical * array.std(ddof=1) / np.sqrt(array.size))


def format_cell(mean, std):
    if mean is None:
        return "---"
    if std is None:
        return f"${mean:.3f}$"
    return f"${mean:.3f} \\pm {std:.3f}$"


def generation_number(path):
    return int(path.name.split("_", 1)[1])


def best_chromosome(run_dir):
    generations = []
    for child in run_dir.iterdir():
        if not (child.is_dir() and child.name.startswith("generation_")):
            continue
        chromosome = child / "chromosome_0.json"
        if not chromosome.exists():
            continue
        try:
            generations.append((generation_number(child), chromosome))
        except ValueError:
            continue
    if not generations:
        return None
    return max(generations, key=lambda item: item[0])[1]


def matching_run_dirs(results_dir, game, local, globals_, steps):
    matches = []
    if not results_dir.exists():
        return matches
    for path in results_dir.iterdir():
        if not path.is_dir():
            continue
        match = RUN_DIR_RE.match(path.name)
        if not match:
            continue
        if (
            match.group("game") == game
            and int(match.group("lf")) == local
            and int(match.group("gf")) == globals_
            and int(match.group("steps")) == steps
        ):
            matches.append(path)
    return sorted(matches, key=lambda path: path.name)


def chromosomes_for_group(results_dir, key, seed_count):
    game, local, globals_, steps = key
    chromosomes = []
    for run_dir in matching_run_dirs(results_dir, game, local, globals_, steps):
        chromosome = best_chromosome(run_dir)
        if chromosome is not None:
            chromosomes.append(chromosome)
    if seed_count and len(chromosomes) > seed_count:
        chromosomes = chromosomes[-seed_count:]
    return chromosomes


def iteration_average(chromosome_path, runs):
    cached = read_cached_summary(chromosome_path, runs)
    if cached is not None:
        return average_steps_from_summary(cached)
    summary = measure_chromosome(chromosome_path, runs=runs)
    write_summary(chromosome_path, summary, overwrite=False)
    return average_steps_from_summary(summary)


def collect_iterations(grouped, results_dir, runs):
    values = {}
    for key, records in grouped.items():
        chromosomes = chromosomes_for_group(results_dir, key, len(records))
        if not chromosomes:
            print(f"No local chromosomes for {key[0]} lf={key[1]} gf={key[2]} steps={key[3]}")
            values[key] = []
            continue
        averages = []
        for chromosome in chromosomes:
            print(f"Replaying {chromosome} ({runs} generations, no montage)")
            try:
                average = iteration_average(chromosome, runs)
            except Exception as error:
                print(f"  failed: {error}")
                continue
            if average is None:
                print("  no successful generations")
                continue
            print(f"  average iterations: {average:.3f}")
            averages.append(average)
        values[key] = averages
    return values


def build_table(grouped, metric, iterations=None, replay_runs=DEFAULT_REPLAY_RUNS):
    rows = []
    for (game, local, globals_, _steps), records in grouped.items():
        mean, std = mean_std([record["value"] for record in records])
        iteration_values = None if iterations is None else iterations.get((game, local, globals_, _steps))
        if iteration_values:
            iteration_mean, iteration_std = mean_std(iteration_values)
        else:
            iteration_mean, iteration_std = None, None
        rows.append([
            game, str(local), str(globals_),
            format_cell(mean, std),
            format_cell(iteration_mean, iteration_std),
        ])

    seed_counts = sorted({len(records) for records in grouped.values()})
    if len(seed_counts) == 1:
        seeds = f"{seed_counts[0]} seeds"
    else:
        seeds = f"{seed_counts[0]}--{seed_counts[-1]} seeds"
    caption = (
        f"Mean $\\pm$ standard deviation of {metric.replace('_', ' ')} "
        f"over {seeds} per configuration, and mean CA iterations to complete "
        f"a level over {replay_runs} generations per seed."
    )

    lines = [
        "% Requires \\usepackage{booktabs} in the Overleaf preamble.",
        "\\begin{table}[ht]",
        "  \\centering",
        "  \\begin{tabular}{lrrcc}",
        "    \\toprule",
        "    " + " & ".join(TABLE_HEADER) + " \\\\",
        "    \\midrule",
    ]
    lines.extend("    " + " & ".join(row) + " \\\\" for row in rows)
    lines.extend([
        "    \\bottomrule",
        "  \\end{tabular}",
        f"  \\caption{{{caption}}}",
        "  \\label{tab:best-fitness}",
        "\\end{table}",
        "",
    ])
    return "\n".join(lines)


def history_curve(run, metric):
    # One sampled request per run; scan_history pages and is far too slow over a whole sweep.
    rows = run.history(keys=["generation", metric], samples=HISTORY_SAMPLES, pandas=False)
    generations, values = [], []
    for row in rows:
        generation, value = row.get("generation"), row.get(metric)
        if generation is None or value is None:
            continue
        generations.append(int(generation))
        values.append(float(value))
    return generations, values


def fetch_curves(records, metric):
    with ThreadPoolExecutor(max_workers=HISTORY_WORKERS) as pool:
        return list(pool.map(lambda record: history_curve(record["run"], metric), records))


def aggregate_curves(records, metric):
    """Mean and 95% CI half-width per generation; a run that died early only counts where it logged."""
    per_generation = {}
    for generations, values in fetch_curves(records, metric):
        for generation, value in zip(generations, values):
            per_generation.setdefault(generation, []).append(value)
    if not per_generation:
        return None

    generations = sorted(per_generation)
    means, intervals = [], []
    for generation in generations:
        values = per_generation[generation]
        means.append(float(np.mean(values)))
        intervals.append(confidence_interval(values))
    return np.array(generations), np.array(means), np.array(intervals)


def axis_limits(curves):
    """Frame the mean curves. A single wide band may clip rather than flatten every panel."""
    means = np.concatenate([mean for _, mean, _ in curves]) if curves else np.array([])
    if means.size == 0:
        return None
    low, high = float(means.min()), float(means.max())
    padding = max((high - low) * 0.35, 0.02)
    return low - padding, high + padding


def short_project_name(project):
    prefix = "llm-ca-evolution-"
    return project[len(prefix):] if project.startswith(prefix) else project


def default_output_dir(projects):
    export = PROJECT_ROOT / "results" / "wandb_export"
    if len(projects) == 1:
        return export / projects[0]
    return export / ("compare_" + "__".join(short_project_name(project) for project in projects))


def grouped_game_names(grouped):
    return {key[0] for key in grouped}


def comparison_labels(project_groups):
    """Prefer the game name; disambiguate with the project when two series share a game."""
    primary = []
    for _project, grouped in project_groups:
        names = grouped_game_names(grouped)
        primary.append(next(iter(names)) if len(names) == 1 else None)
    labels = []
    for (project, _grouped), game in zip(project_groups, primary):
        if game is not None and primary.count(game) == 1:
            labels.append(game.capitalize())
        elif game is not None:
            labels.append(f"{game.capitalize()} ({short_project_name(project)})")
        else:
            labels.append(short_project_name(project))
    return labels


def records_for_cell(grouped, local, global_):
    """All seeds for this (lf, gf), including distinct step budgets if a project ran more than one."""
    records = []
    for (_game, cell_local, cell_global, _steps), group in grouped.items():
        if cell_local == local and cell_global == global_:
            records.extend(group)
    return records


def series_colours(count):
    if count <= len(SERIES_COLOURS):
        return list(SERIES_COLOURS[:count])
    import matplotlib
    cmap = matplotlib.colormaps["tab10"]
    return [cmap(index % 10) for index in range(count)]


def style_panel(axes):
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        axes.spines[spine].set_linewidth(0.8)
        axes.spines[spine].set_color("0.25")
    axes.tick_params(length=3.5, width=0.8, color="0.25", labelsize="large", pad=3)
    axes.set_axisbelow(True)
    axes.grid(True, axis="y", color="0.85", linewidth=0.8)
    axes.grid(False, axis="x")


def plot_montage(series, metric, output_dir, stem):
    """Small multiples: one panel per (lf, gf), one coloured mean+CI line per series entry."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = set()
    for item in series:
        cells.update(item["curves"])
    if not cells:
        return []

    locals_ = sorted({local for local, _global in cells})
    globals_ = sorted({global_ for _local, global_ in cells})
    colours = series_colours(len(series))
    label = metric.replace("_", " ")
    figure, grid = plt.subplots(
        len(locals_), len(globals_),
        figsize=(PANEL_WIDTH * len(globals_), PANEL_HEIGHT * len(locals_)),
        sharex=True, sharey=True, squeeze=False,
    )
    figure.set_facecolor("white")

    for row, local in enumerate(locals_):
        for column, global_ in enumerate(globals_):
            axes = grid[row][column]
            axes.set_facecolor("white")
            axes.set_title(f"lf = {local},  gf = {global_}", fontsize="large", pad=8, color="0.15")
            style_panel(axes)
            drawn = 0
            # Bands first so overlapping CIs never paint over the mean lines.
            for index, item in enumerate(series):
                curve = item["curves"].get((local, global_))
                if curve is None:
                    continue
                generations, means, intervals = curve
                axes.fill_between(
                    generations, means - intervals, means + intervals,
                    color=colours[index], alpha=0.18, linewidth=0, zorder=1,
                )
                drawn += 1
            for index, item in enumerate(series):
                curve = item["curves"].get((local, global_))
                if curve is None:
                    continue
                generations, means, _intervals = curve
                axes.plot(
                    generations, means, color=colours[index], linewidth=2.4,
                    solid_capstyle="round", zorder=2, label=item["label"],
                )
            if drawn == 0:
                axes.text(0.5, 0.5, "no runs", transform=axes.transAxes,
                          ha="center", va="center", fontsize="large", color="0.5")

    # A single noisy band would otherwise stretch the shared axis and flatten every panel.
    all_curves = [curve for item in series for curve in item["curves"].values()]
    limits = axis_limits(all_curves)
    if limits is not None:
        grid[0][0].set_ylim(*limits)

    handles = [
        plt.Line2D([], [], color=colours[index], linewidth=2.4, label=item["label"],
                   solid_capstyle="round")
        for index, item in enumerate(series)
    ]
    figure.supylabel(f"Mean {label}", fontsize="x-large", color="0.15")
    # Reserve a strip at the bottom, then stack the x label and the legend inside it.
    # tight_layout does not lay out either one, so both get an explicit position.
    figure.tight_layout(rect=(0.01, 0.07, 1, 0.995), h_pad=1.15, w_pad=0.7)
    figure.supxlabel("Generation", y=0.042, fontsize="x-large", color="0.15")
    figure.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0),
        ncols=max(len(series), 1), frameon=False, fontsize="x-large",
        handlelength=2.6, handletextpad=0.7, columnspacing=1.8,
    )

    written = []
    for suffix in ("pdf", "png"):
        path = output_dir / f"{stem}.{suffix}"
        figure.savefig(path, bbox_inches="tight", dpi=200, facecolor="white")
        written.append(path)
    plt.close(figure)
    return written


def plot_game(game, groups, metric, output_dir):
    # Small multiples: a panel per (lf, gf) holds only the steps lines, so the whole
    # figure needs one legend of a few entries instead of one entry per combination.
    by_steps = {}
    for (_game, local, global_, steps), records in groups.items():
        curve = aggregate_curves(records, metric)
        if curve is None:
            continue
        by_steps.setdefault(steps, {})[(local, global_)] = curve
    if not by_steps:
        return []
    series = [
        {"label": f"steps={steps}", "curves": by_steps[steps]}
        for steps in sorted(by_steps)
    ]
    return plot_montage(series, metric, output_dir, f"fitness_curves_{game}")


def plot_projects(project_groups, metric, output_dir):
    """Same lf×gf montage as plot_game, but one mean+CI line per project. Legend is the game name."""
    labels = comparison_labels(project_groups)
    cells = {
        (local, global_)
        for _project, grouped in project_groups
        for (_game, local, global_, _steps) in grouped
    }
    series = []
    for label, (_project, grouped) in zip(labels, project_groups):
        curves = {}
        for local, global_ in cells:
            records = records_for_cell(grouped, local, global_)
            if not records:
                continue
            curve = aggregate_curves(records, metric)
            if curve is not None:
                curves[(local, global_)] = curve
        series.append({"label": label, "curves": curves})
    if not any(item["curves"] for item in series):
        return []
    return plot_montage(series, metric, output_dir, "fitness_curves_compare")


def scatter_points(groups):
    """Mean metric per (local, global); seeds and steps are averaged together."""
    by_cell = {}
    for (_game, local, globals_, _steps), records in groups.items():
        by_cell.setdefault((local, globals_), []).extend(record["value"] for record in records)
    return [
        (local, globals_, mean_std(values)[0])
        for (local, globals_), values in sorted(by_cell.items())
    ]


def plot_fitness_scatter(game, groups, metric, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = scatter_points(groups)
    if not points:
        return []

    locals_ = np.array([local for local, _globals, _mean in points], dtype=float)
    globals_ = np.array([globals_value for _local, globals_value, _mean in points], dtype=float)
    means = np.array([mean for _local, _globals, mean in points], dtype=float)

    label = metric.replace("_", " ")
    figure, axes = plt.subplots(figsize=(PANEL_WIDTH, PANEL_HEIGHT))
    scatter = axes.scatter(
        locals_, globals_, c=means, cmap="viridis", s=220,
        edgecolors="0.2", linewidths=0.6, zorder=3,
    )
    colorbar = figure.colorbar(scatter, ax=axes)
    colorbar.set_label(f"Mean {label}", fontsize="x-large")
    colorbar.ax.tick_params(labelsize="large")

    axes.set_xlabel("Local functions", fontsize="x-large")
    axes.set_ylabel("Global functions", fontsize="x-large")
    axes.set_xticks(sorted(set(locals_)))
    axes.set_yticks(sorted(set(globals_)))
    axes.xaxis.set_major_formatter(plt.FormatStrFormatter("%d"))
    axes.yaxis.set_major_formatter(plt.FormatStrFormatter("%d"))
    axes.tick_params(labelsize="x-large")
    axes.grid(True, alpha=0.3, zorder=0)

    x_span = max(locals_.max() - locals_.min(), 1.0)
    y_span = max(globals_.max() - globals_.min(), 1.0)
    axes.set_xlim(locals_.min() - 0.12 * x_span, locals_.max() + 0.12 * x_span)
    axes.set_ylim(globals_.min() - 0.12 * y_span, globals_.max() + 0.12 * y_span)

    figure.tight_layout()

    written = []
    for suffix in ("pdf", "png"):
        path = output_dir / f"fitness_scatter_{game}.{suffix}"
        figure.savefig(path, bbox_inches="tight", dpi=150)
        written.append(path)
    plt.close(figure)
    return written


def build_parser():
    parser = argparse.ArgumentParser(
        description="Download W&B runs, group the seed repeats, and write a LaTeX table plus fitness plots.",
        epilog=(
            "Examples:\n"
            "  uv run scripts/compile_results.py --project llm-ca-evolution-sokoban-opus\n"
            "  uv run scripts/compile_results.py --project llm-ca-evolution-zelda-opus-2-2 llm-ca-evolution-binary-opus-2-1\n"
            "  uv run scripts/compile_results.py --project my-project --output-dir results/export"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-p", "--project", dest="projects", nargs="+", action="extend", required=True,
        help="W&B project name(s). Pass two or more to also write a comparison montage.",
    )
    parser.add_argument("-e", "--entity", type=str, default=DEFAULT_ENTITY, help=f"W&B entity. Default: {DEFAULT_ENTITY}")
    parser.add_argument(
        "-o", "--output-dir", type=str, default=None,
        help="Where to write the table and figures. Default: results/wandb_export/<project>, "
             "or results/wandb_export/compare_<projects> when comparing.",
    )
    parser.add_argument(
        "-m", "--metric", type=str, default="best_fitness",
        help="Summary and history key to aggregate. Default: best_fitness",
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Local sweep results to replay for average iterations. Default: results/",
    )
    parser.add_argument(
        "--replay-runs", type=int, default=DEFAULT_REPLAY_RUNS,
        help=f"How many times to execute each best chromosome. Default: {DEFAULT_REPLAY_RUNS}",
    )
    return parser


if __name__ == "__main__":
    load_dotenv(PROJECT_ROOT / ".env")
    args = build_parser().parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args.projects)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records, skipped = [], 0
    project_groups = []
    for project in args.projects:
        print(f"W&B: {args.entity}/{project}")
        project_records, project_skipped = collect_runs(args.entity, project, args.metric)
        skipped += project_skipped
        for record in project_records:
            record["project"] = project
        records.extend(project_records)
        grouped_project = group_runs(project_records)
        if grouped_project:
            project_groups.append((project, grouped_project))
        print(
            f"  {len(project_records)} run(s) in {len(grouped_project)} configuration(s), "
            f"{project_skipped} skipped"
        )

    if not records:
        raise SystemExit(f"No usable runs found ({skipped} skipped)")

    grouped = group_runs(records)
    print(f"{len(records)} run(s) in {len(grouped)} configuration(s), {skipped} skipped")

    results_dir = Path(args.results_dir) if args.results_dir else PROJECT_ROOT / "results"
    if not results_dir.is_absolute():
        results_dir = PROJECT_ROOT / results_dir
    iterations = collect_iterations(grouped, results_dir, args.replay_runs)

    table_path = output_dir / "fitness_table.tex"
    table_path.write_text(build_table(grouped, args.metric, iterations, args.replay_runs))
    print(f"Table: {table_path}")

    games = {}
    for key, group_records in grouped.items():
        games.setdefault(key[0], {})[key] = group_records

    for game, groups in games.items():
        for path in plot_game(game, groups, args.metric, output_dir):
            print(f"Plot: {path}")
        for path in plot_fitness_scatter(game, groups, args.metric, output_dir):
            print(f"Plot: {path}")

    if len(project_groups) >= 2:
        for path in plot_projects(project_groups, args.metric, output_dir):
            print(f"Plot: {path}")

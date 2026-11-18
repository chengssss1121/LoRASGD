#!/usr/bin/env python3
"""Plot EXP1 and compare held-out and training factor gradients.

The revised held-out figure shows (a) the cumulative test/train squared-norm
ratio, (b) normalized cumulative gradient inner product, and (c) the original
cumulative squared vector gap. All three are computed per seed before seed
averaging. Sparse pre-update checkpoints use interval weights. Alignment is
the cosine between weighted concatenated gradient sequences, not the mean
of checkpoint cosines. Existing saved squared norms and vector gaps suffice;
no training or gradient evaluations are repeated.

To reproduce the original three-panel training and two-panel held-out figures
with linear axes, run:
    python3 EXP1_Plot_Cumulative.py --linear --heldout-layout original
This writes to figures_linear/ inside the selected run, leaving figures/ intact.
The saved measurements and interval-weighted averages are unchanged.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np


CHECKPOINT_METRICS = (
    "train_loss",
    "factor_spectral_sq",
    "train_grad_sq",
    "test_loss",
    "test_grad_sq",
    "gradient_gap_sq",
)
RUNNING_AVERAGE_METRICS = ("train_grad_sq", "test_grad_sq", "gradient_gap_sq")

SCHEDULE_STYLES = {
    "constant_small": ("#0072b2", "-"),
    "constant_medium": ("#d55e00", "--"),
    "constant_large": ("#009e73", "-."),
    "decay": ("#8b4da5", ":"),
}


def read_series(path, expected_steps, metrics):
    values = {metric: [] for metric in metrics}
    steps = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"step", *metrics}
        if not required.issubset(reader.fieldnames or []):
            missing = sorted(required.difference(reader.fieldnames or []))
            raise ValueError(f"Missing CSV columns in {path}: {', '.join(missing)}")
        for row_number, row in enumerate(reader, start=2):
            try:
                steps.append(int(row["step"]))
                for metric in metrics:
                    value = float(row[metric])
                    if not math.isfinite(value) or value < 0:
                        raise ValueError(f"Invalid {metric}: {value}")
                    if metric == "step_size" and value == 0:
                        raise ValueError("Expected a positive step size")
                    values[metric].append(value)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{row_number}: {exc}") from exc
    if not np.array_equal(steps, expected_steps):
        raise ValueError(f"Unexpected step grid in {path}")
    return {metric: np.asarray(series, dtype=float) for metric, series in values.items()}


def read_run(run_dir):
    with (run_dir / "experiment.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("experiment") != "EXP1"
        or manifest.get("status") != "completed"
    ):
        raise ValueError("Expected a completed EXP1 experiment with schema_version=1")

    steps = manifest["steps"]
    eval_every = manifest["eval_every"]
    seeds = manifest["seeds"]
    schedules = manifest["schedules"]
    schedule_ids = [schedule["id"] for schedule in schedules]
    if type(steps) is not int or steps < 1 or type(eval_every) is not int or eval_every < 1:
        raise ValueError("Expected positive integers for steps and eval_every")
    if not seeds or any(type(seed) is not int for seed in seeds) or len(seeds) != len(set(seeds)):
        raise ValueError("Expected a nonempty list of distinct integer seeds")
    if (
        not schedule_ids
        or len(schedule_ids) != len(set(schedule_ids))
        or not set(schedule_ids).issubset(SCHEDULE_STYLES)
        or any(not isinstance(schedule["label"], str) or not schedule["label"] for schedule in schedules)
    ):
        raise ValueError("Expected distinct, labeled EXP1 schedules")

    checkpoint_steps = np.array([*range(0, steps, eval_every), steps])
    update_steps = np.arange(1, steps + 1)
    expected_runs = {(seed, schedule_id) for seed in seeds for schedule_id in schedule_ids}
    runs = {}
    folders = set()
    for entry in manifest["runs"]:
        key = (entry["seed"], entry["schedule"])
        if key not in expected_runs or key in runs:
            raise ValueError(f"Unexpected or duplicate run {key}")
        folder = (run_dir / entry["folder"]).resolve()
        if not folder.is_relative_to(run_dir) or folder in folders:
            raise ValueError(f"Expected a distinct run folder inside {run_dir}: {folder}")
        folders.add(folder)
        checkpoints = read_series(folder / "checkpoints.csv", checkpoint_steps, CHECKPOINT_METRICS)
        updates = read_series(folder / "updates.csv", update_steps, ("step_size",))
        runs[key] = {**checkpoints, **updates}
    missing_runs = expected_runs.difference(runs)
    if missing_runs:
        raise ValueError(f"Missing paired runs: {sorted(missing_runs)}")
    return manifest, checkpoint_steps, update_steps, runs


def draw_panel(ax, x, metric, schedules, seeds, runs, *, log_scale=False):
    curves = []
    for schedule in schedules:
        schedule_id = schedule["id"]
        measurements = np.stack([runs[seed, schedule_id][metric] for seed in seeds])
        mean_curve = measurements.mean(axis=0)
        spread = measurements.std(axis=0, ddof=1) if len(seeds) > 1 else None
        curves.append((schedule, mean_curve, spread))

    if log_scale:
        ax.set_yscale("log", nonpositive="mask")
        means = np.concatenate([mean for _, mean, _ in curves])
        positives = means[means > 0]
        if not positives.size:
            ax.set_ylim(1e-12, 1)
            ax.text(0.5, 0.5, "All measured values are zero", ha="center", va="center",
                    transform=ax.transAxes)
            return
        # A log axis has no ordinate for zero. Omit zero means and stop
        # sample-SD bands that cross zero at the visible lower boundary.
        floor = float(positives.min()) * 0.7
    else:
        ax.set_yscale("linear")
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

    for schedule, mean, spread in curves:
        color, linestyle = SCHEDULE_STYLES[schedule["id"]]
        shown = np.where(mean > 0, mean, np.nan) if log_scale else mean
        ax.plot(x, shown, color=color, linestyle=linestyle, linewidth=1.55,
                marker="o" if x.size == 1 else None, label=schedule["label"])
        if spread is not None and np.any(spread > 0):
            lower = np.maximum(mean - spread, 0)
            if log_scale:
                lower = np.maximum(lower, floor)
            ax.fill_between(x, lower, mean + spread,
                            where=mean > 0 if log_scale else None,
                            color=color, alpha=0.16, linewidth=0)
    if log_scale:
        ax.set_ylim(bottom=floor)


def make_figure(panels, manifest, runs, steps_by_metric, *, note=None):
    schedules, seeds = manifest["schedules"], manifest["seeds"]
    width = 10.6 if len(panels) == 3 else 8.7
    with plt.rc_context({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
    }):
        figure, axes = plt.subplots(
            1, len(panels), figsize=(width, 3.3), sharex=True, squeeze=False
        )
        axes[0, 0].xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        for ax, (title, metric, ylabel, log_scale) in zip(axes[0], panels):
            draw_panel(ax, steps_by_metric[metric], metric, schedules, seeds, runs, log_scale=log_scale)
            ax.set_title(title, loc="left", pad=6)
            ax.set_ylabel(ylabel)
            ax.set_xlim(0, manifest["steps"])
            ax.set_xlabel("Training updates")
            ax.grid(axis="y", alpha=0.22)
        # The legend stays complete if any log-scale panel has only zeros.
        handles = [
            plt.Line2D([], [], color=SCHEDULE_STYLES[schedule["id"]][0],
                       linestyle=SCHEDULE_STYLES[schedule["id"]][1],
                       label=schedule["label"], linewidth=1.55)
            for schedule in schedules
        ]
        figure.legend(handles=handles, loc="upper center", ncol=len(handles),
                      frameon=False, bbox_to_anchor=(0.5, 0.995))
        if note:
            figure.text(0.5, 0.025, note, ha="center", fontsize=8)
        figure.tight_layout(rect=(0, 0.07 if note else 0, 1, 0.88))
    return figure


def running_pre_update_averages(checkpoint_steps, runs):
    # Step c is measured before update c+1. Use that measurement for every
    # update through the next checkpoint, including a shortened last interval.
    interval_lengths = np.diff(checkpoint_steps)
    completed_updates = checkpoint_steps[1:]
    return {
        key: {
            f"mean_{metric}": np.cumsum(values[metric][:-1] * interval_lengths) / completed_updates
            for metric in RUNNING_AVERAGE_METRICS
        }
        for key, values in runs.items()
    }


def heldout_comparison_metrics(checkpoint_steps, runs):
    """Derive magnitude and direction comparisons from matched squared norms.

    At each checkpoint, <test, train> = (test_sq + train_sq - gap_sq)/2.
    The final checkpoint belongs to no preceding pre-update interval.
    Source norms may be computed in float32, so tiny Cauchy--Schwarz violations
    from rounding are corrected and counted. Larger violations raise an error.
    Zero-denominator comparisons are missing, never regularized by an epsilon.
    """
    steps = np.asarray(checkpoint_steps, dtype=np.longdouble)
    if (steps.ndim != 1 or len(steps) < 2 or not np.isfinite(steps).all()
            or steps[0] != 0 or np.any(np.diff(steps) <= 0)):
        raise ValueError("Expected increasing checkpoints starting at zero.")
    weights, elapsed = np.diff(steps), steps[1:]
    result = {}
    for key, values in runs.items():
        train, test, gap = [np.asarray(values[name], dtype=np.longdouble)
                            for name in RUNNING_AVERAGE_METRICS]
        if any(x.shape != steps.shape or not np.isfinite(x).all() or np.any(x < 0)
               for x in (train, test, gap)):
            raise ValueError(f"Run {key}: invalid matched gradient measurements.")
        inner = (train + test - gap) / 2
        limit = np.sqrt(train) * np.sqrt(test)
        tolerance = 64 * np.finfo(np.float32).eps * np.maximum.reduce([train, test, gap])
        bad = np.abs(inner) > limit + tolerance
        if np.any(bad):
            checkpoint = int(steps[np.flatnonzero(bad)[0]])
            raise ValueError(
                f"Run {key}, checkpoint {checkpoint}: train_grad_sq, test_grad_sq and "
                "gradient_gap_sq violate the squared-distance identity. The gap must "
                "be the squared norm of the vector difference at the same iterate.")
        repairs = int(np.count_nonzero(np.abs(inner) > limit))
        inner = np.clip(inner, -limit, limit)
        train_sum = np.cumsum(weights * train[:-1])
        test_sum = np.cumsum(weights * test[:-1])
        gap_sum = np.cumsum(weights * gap[:-1])
        inner_sum = np.cumsum(weights * inner[:-1])
        ratio = np.full(elapsed.shape, np.nan, dtype=np.longdouble)
        alignment = ratio.copy()
        np.divide(test_sum, train_sum, out=ratio, where=train_sum > 0)
        denominator = np.sqrt(train_sum) * np.sqrt(test_sum)
        np.divide(inner_sum, denominator, out=alignment, where=denominator > 0)
        finite = np.isfinite(alignment)
        if np.any(np.abs(alignment[finite]) > 1 + 1e-12):
            raise ValueError(f"Run {key}: invalid cumulative gradient alignment.")
        alignment[finite] = np.clip(alignment[finite], -1, 1)
        result[key] = {name: np.asarray(value, dtype=np.float64) for name, value in {
            "heldout_train_ratio": ratio,
            "gradient_alignment": alignment,
            "mean_gradient_gap_sq": gap_sum / elapsed,
            "cumulative_train_grad_sq": train_sum,
            "cumulative_test_grad_sq": test_sum,
            "cumulative_gradient_inner_product": inner_sum,
        }.items()}
        result[key]["roundoff_repairs"] = repairs
    return result


def make_heldout_figure(manifest, completed_steps, runs, comparisons, *, linear=False):
    """Use individual-seed curves for ratios and alignment; SD for the gap."""
    schedules, seeds = manifest["schedules"], manifest["seeds"]
    exact = manifest["eval_every"] == 1
    prefix = "" if exact else "Checkpoint estimate of\n"
    labels = (
        prefix + r"$\frac{\sum_{k=1}^t\|\nabla J_{S_{\mathrm{test}}}(B_k,A_k)\|_F^2}{\sum_{k=1}^t\|\nabla J_S(B_k,A_k)\|_F^2}$",
        prefix + "normalized cumulative inner product",
        prefix + r"$\frac{1}{t}\sum_{k=1}^t\|(\nabla J_{S_{\mathrm{test}}}-\nabla J_S)(B_k,A_k)\|_F^2$",
    )
    titles = ("(a) Relative held-out stationarity", "(b) Gradient alignment",
              "(c) Cumulative gradient gap")
    with plt.rc_context({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white",
                         "pdf.fonttype": 42}):
        figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.1), sharex=True)
        for ax, metric in zip(axes[:2], ("heldout_train_ratio", "gradient_alignment")):
            any_valid = False
            for schedule in schedules:
                schedule_id = schedule["id"]
                values = np.stack([comparisons[seed, schedule_id][metric] for seed in seeds])
                color, style = SCHEDULE_STYLES[schedule_id]
                for curve in values:
                    ax.plot(completed_steps, curve, color=color, linestyle=style,
                            linewidth=0.7, alpha=0.18)
                # Require every seed at a point: undefined ratios are gaps,
                # and do not silently change the seed sample size.
                valid = np.isfinite(values).all(axis=0)
                mean = np.full(values.shape[1], np.nan)
                mean[valid] = values[:, valid].mean(axis=0)
                any_valid |= bool(np.any(valid))
                ax.plot(completed_steps, mean, color=color, linestyle=style, linewidth=1.9)
            ax.axhline(1.0, color="#666666", linestyle=(0, (2, 3)), linewidth=1.0, zorder=0)
            if not any_valid:
                ax.text(0.5, 0.5, "Undefined: zero gradient denominators",
                        transform=ax.transAxes, ha="center", fontsize=8)
            ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        draw_panel(axes[2], completed_steps, "mean_gradient_gap_sq", schedules, seeds,
                   runs, log_scale=not linear)
        for ax, title, label in zip(axes, titles, labels):
            ax.set_title(title, loc="left", pad=9)
            ax.set_ylabel(label)
            ax.set_xlabel("Training updates")
            ax.set_xlim(0, manifest["steps"])
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
            ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            ax.grid(axis="y", alpha=0.2)
        handles = [plt.Line2D([], [], color=SCHEDULE_STYLES[item["id"]][0],
                             linestyle=SCHEDULE_STYLES[item["id"]][1], linewidth=1.9,
                             label=item["label"]) for item in schedules]
        figure.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
                      bbox_to_anchor=(0.5, 0.995), fontsize=9.4)
        figure.tight_layout(rect=(0, 0, 1, 0.88))
    return figure


def write_heldout_comparisons(figure_dir, manifest, completed_steps, comparisons, *, linear=False):
    metrics = ("heldout_train_ratio", "gradient_alignment", "mean_gradient_gap_sq",
               "cumulative_train_grad_sq", "cumulative_test_grad_sq",
               "cumulative_gradient_inner_product")
    path = figure_dir / "EXP1_heldout_comparison.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("schedule", "seed", "step", *metrics))
        writer.writeheader()
        for schedule in manifest["schedules"]:
            for seed in manifest["seeds"]:
                values = comparisons[seed, schedule["id"]]
                for i, step in enumerate(completed_steps):
                    row = {"schedule": schedule["id"], "seed": seed, "step": int(step)}
                    row.update({name: float(values[name][i]) if np.isfinite(values[name][i]) else ""
                                for name in metrics})
                    writer.writerow(row)
    exact = manifest["eval_every"] == 1
    details = {
        "averaging": "exact pre-update sums" if exact else "left-checkpoint interval-weighted estimates",
        "timing": "Checkpoint c represents gradients before update c+1 through the next checkpoint. The final checkpoint is excluded from prior intervals. Step zero averages are undefined and omitted.",
        "panel_a": "Sum of held-out squared factor-gradient norms divided by the sum of training squared factor-gradient norms. Reference 1 means equal cumulative squared norms.",
        "panel_b": "Sum of train-test gradient inner products divided by the square root of the product of the two sums of squared gradient norms. This is cosine similarity of the weighted concatenated sequences, not mean checkpoint cosine similarity. Reference 1 means positive proportionality of the concatenated sequences.",
        "inner_product": "(train_grad_sq + test_grad_sq - gradient_gap_sq)/2, using the squared vector difference at the same iterate.",
        "panel_c": "Cumulative mean squared train-test factor-gradient gap, using the same intervals.",
        "aggregation": "Compute quantities separately per seed before taking arithmetic seed means. A mean point requires defined values for all seeds.",
        "axis_scale": "All axes linear" if linear else "Linear axes except logarithmic panel-c y-axis",
        "uncertainty": "Panels a,b: faint individual-seed curves and thick means. Panel c: mean +/- one sample standard deviation, with the lower band clipped at zero on linear axes or at the visible lower boundary on log axes. Source values are unchanged.",
        "zero_denominators": "Missing, not regularized. Empty CSV cells and line gaps.",
        "pointwise_roundoff_repairs": sum(x["roundoff_repairs"] for x in comparisons.values()),
        "roundoff_policy": "Only Cauchy-Schwarz excursions within 64 times float32 epsilon times the largest saved squared quantity are corrected. Substantive inconsistencies raise errors.",
        "interpretation": "Agreement of norms alone does not establish directional agreement. These are finite test-set diagnostics at observed iterates, not a population convergence-rate test.",
        "caption": "Held-out factor-gradient comparisons under four learning-rate schedules: (a) cumulative held-out-to-training squared-gradient ratio; (b) normalized cumulative train-test gradient inner product; (c) cumulative mean squared gradient gap. All cumulative quantities use interval-weighted checkpoints. In (a) and (b), thick curves show means across seeds, faint curves show individual runs, and horizontal lines mark one. In (c), shading indicates one sample standard deviation.",
    }
    (figure_dir / "EXP1_heldout_plot_details.json").write_text(
        json.dumps(details, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {path}")


def save_figure(figure, output_stem):
    try:
        for extension in ("pdf", "png"):
            path = output_stem.with_suffix(f".{extension}")
            figure.savefig(path, dpi=240, bbox_inches="tight")
            print(f"Saved {path}")
    finally:
        plt.close(figure)


def write_endpoints(path, manifest, checkpoint_steps, runs):
    total_steps = manifest["steps"]
    before_updates = checkpoint_steps < total_steps
    tail_length = (total_steps + 9) // 10
    tail = before_updates & (checkpoint_steps >= total_steps - tail_length)
    averaging = "per_update" if manifest["eval_every"] == 1 else "checkpoint"
    averaged_metrics = RUNNING_AVERAGE_METRICS
    fields = [
        "schedule", "seed", "steps", "averaging",
        "pre_update_measurements", "last_10pct_pre_update_measurements",
        *[f"final_{metric}" for metric in CHECKPOINT_METRICS],
        *[f"pre_update_mean_{metric}" for metric in averaged_metrics],
        *[f"last_10pct_pre_update_mean_{metric}" for metric in averaged_metrics],
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for schedule in manifest["schedules"]:
            for seed in manifest["seeds"]:
                values = runs[seed, schedule["id"]]
                row = {
                    "schedule": schedule["id"],
                    "seed": seed,
                    "steps": total_steps,
                    "averaging": averaging,
                    "pre_update_measurements": int(before_updates.sum()),
                    "last_10pct_pre_update_measurements": int(tail.sum()),
                }
                for metric in CHECKPOINT_METRICS:
                    row[f"final_{metric}"] = values[metric][-1]
                for metric in averaged_metrics:
                    row[f"pre_update_mean_{metric}"] = values[metric][before_updates].mean()
                    row[f"last_10pct_pre_update_mean_{metric}"] = (
                        values[metric][tail].mean() if tail.any() else ""
                    )
                writer.writerow(row)


def plot_results(run_dir, *, heldout_only=False, output_dir=None, linear=False,
                 heldout_layout="comparison"):
    """Read a completed run and return the path of its figures directory."""
    if heldout_layout not in ("original", "comparison"):
        raise ValueError("heldout_layout must be 'original' or 'comparison'")
    run_dir = Path(run_dir).expanduser().resolve()
    manifest, checkpoint_steps, _, runs = read_run(run_dir)
    default_directory = "figures_linear" if linear else "figures"
    figure_dir = Path(output_dir).expanduser().resolve() if output_dir else run_dir / default_directory
    figure_dir.mkdir(parents=True, exist_ok=True)

    steps_by_metric = dict.fromkeys(CHECKPOINT_METRICS, checkpoint_steps)
    for metric in RUNNING_AVERAGE_METRICS:
        steps_by_metric[f"mean_{metric}"] = checkpoint_steps[1:]
    for key, averages in running_pre_update_averages(checkpoint_steps, runs).items():
        runs[key].update(averages)
    empirical_average_label = r"$\frac{1}{t}\sum_{k=1}^t\|\nabla J_S(B_k,A_k)\|_F^2$"
    heldout_average_label = r"$\frac{1}{t}\sum_{k=1}^t\|\nabla J_{S_{\mathrm{test}}}(B_k,A_k)\|_F^2$"
    gap_average_label = r"$\frac{1}{t}\sum_{k=1}^t\|(\nabla J_{S_{\mathrm{test}}}-\nabla J_S)(B_k,A_k)\|_F^2$"
    if manifest["eval_every"] == 1:
        average_note = "Running averages use the gradient measured before each update."
    else:
        average_note = "Running averages use the pre-update gradient at the start of each checkpoint interval."

    main_panels = (
        ("(a) Training objective", "train_loss", "Full training cross-entropy", False),
        ("(b) Factor growth", "factor_spectral_sq",
         r"$\|A_t\|_2^2+\|B_t\|_2^2$", False),
        ("(c) Average empirical stationarity", "mean_train_grad_sq", empirical_average_label, not linear),
    )
    average_panels = (
        ("(a) Empirical factor gradient", "mean_train_grad_sq", empirical_average_label, not linear),
        ("(b) Average gradient gap", "mean_gradient_gap_sq", gap_average_label, not linear),
    )
    heldout_panels = (
        ("(a) Average held-out stationarity", "mean_test_grad_sq", heldout_average_label, not linear),
        ("(b) Average gradient gap", "mean_gradient_gap_sq", gap_average_label, not linear),
    )
    if not heldout_only:
        for filename, panels in (("EXP1_main", main_panels), ("EXP1_running_averages", average_panels)):
            figure = make_figure(panels, manifest, runs, steps_by_metric, note=average_note)
            save_figure(figure, figure_dir / filename)
    if heldout_layout == "original":
        figure = make_figure(heldout_panels, manifest, runs, steps_by_metric, note=average_note)
    else:
        comparisons = heldout_comparison_metrics(checkpoint_steps, runs)
        figure = make_heldout_figure(manifest, checkpoint_steps[1:], runs, comparisons, linear=linear)
        write_heldout_comparisons(figure_dir, manifest, checkpoint_steps[1:], comparisons, linear=linear)
    save_figure(figure, figure_dir / "EXP1_heldout")
    plot_settings = {
        "run_directory": str(run_dir),
        "heldout_layout": heldout_layout,
        "y_axes": "all linear" if linear else "stationarity and gradient-gap panels logarithmic",
        "x_axes": "all linear",
        "averaging": average_note,
        "seed_aggregation": "Arithmetic mean after computing each seed's cumulative averages",
        "bands": "One sample standard deviation, lower edge clipped at zero on linear axes or at the visible lower boundary on log axes. Comparison panels a,b show individual seeds instead.",
    }
    (figure_dir / "EXP1_plot_settings.json").write_text(
        json.dumps(plot_settings, indent=2) + "\n", encoding="utf-8")

    endpoints_path = figure_dir / "EXP1_endpoints.csv"
    write_endpoints(endpoints_path, manifest, checkpoint_steps, runs)
    print(f"Saved {endpoints_path}")
    if manifest["eval_every"] == 1:
        print("Endpoint averages use every pre-update iterate (steps 0 through T-1).")
    else:
        print("Endpoint averages use measured pre-update checkpoints (steps < T).")
    return figure_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="Path to a completed EXP1 run")
    parser.add_argument("--heldout-only", action="store_true", help="Only regenerate the held-out figure")
    parser.add_argument("--output-dir", type=Path,
                        help="Default: figures_linear with --linear; otherwise figures, inside the run")
    parser.add_argument("--linear", action="store_true",
                        help="Use linear axes in every panel; retain the original squared-gradient averages")
    parser.add_argument("--heldout-layout", choices=("original", "comparison"), default="comparison",
                        help="original: two panels as in the paper; comparison: ratio, alignment, and gap")
    args = parser.parse_args()
    if args.run_dir is None:
        pointer = Path(__file__).resolve().with_name("EXP1.latest.txt")
        try:
            latest = pointer.read_text(encoding="utf-8").strip()
        except OSError:
            parser.error("Give --run-dir or complete a training run first.")
        if not latest:
            parser.error("EXP1.latest.txt is empty. Give --run-dir.")
        args.run_dir = pointer.parent / Path(latest).expanduser()
    try:
        plot_results(args.run_dir, heldout_only=args.heldout_only, output_dir=args.output_dir,
                     linear=args.linear, heldout_layout=args.heldout_layout)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"EXP1 plot error: {exc}\n")


if __name__ == "__main__":
    main()

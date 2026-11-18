#!/usr/bin/env python3
"""Plot the fixed-learning-rate ViT factor-growth and gradient-noise experiment."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np


HERE = Path(__file__).resolve().parent
TRAJECTORY_METRICS = (
    "train_loss", "test_loss", "factor_sq", "train_grad_sq", "test_grad_sq",
    "gradient_gap_sq",
)
NOISE_METRICS = (
    "factor_sq", "weight_noise_sq", "factor_noise_sq", "weight_noise_se", "factor_noise_se",
)
RATIOS = ("amplification", "normalized_amplification")
COLORS = ("#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#56b4e9")
LINESTYLES = ("-", "--", "-.", ":")


def integer_list(manifest, name, minimum):
    values = manifest[name]
    if (not isinstance(values, list) or not values
            or any(type(value) is not int or value < minimum for value in values)
            or len(set(values)) != len(values)):
        raise ValueError(f"Manifest {name} must contain distinct integers >= {minimum}.")
    return values


def nonnegative(row, name):
    value = float(row[name])
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return value


def csv_rows(path, required):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = set(required).difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {', '.join(sorted(missing))}.")
        for number, row in enumerate(reader, 2):
            yield number, row


def close(observed, expected, name):
    if not math.isclose(observed, expected, rel_tol=1e-4, abs_tol=1e-12):
        raise ValueError(f"{name} does not agree with its recorded components.")


def read_run(run_dir):
    with (run_dir / "experiment.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("experiment") != "ViT_Noise_EXP" or manifest.get("status") != "completed":
        raise ValueError("Expected a completed ViT_Noise_EXP experiment, not the older horizon sweep.")
    seeds = integer_list(manifest, "seeds", 0)
    steps = integer_list(manifest, "checkpoints", 0)
    batches = integer_list(manifest, "diagnostic_batches", 1)
    if steps != sorted(steps) or steps[0] != 0 or steps[-1] != manifest["steps"]:
        raise ValueError("Checkpoint list must be sorted and include initialization and the final update.")
    if type(manifest["steps"]) is not int or manifest["steps"] < 1:
        raise ValueError("Manifest steps must be a positive integer.")
    if type(manifest["diagnostic_pairs"]) is not int or manifest["diagnostic_pairs"] < 1:
        raise ValueError("Manifest diagnostic_pairs must be a positive integer.")
    lr = float(manifest["lr"])
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("Manifest lr must be finite and positive.")

    expected_trajectory = {(seed, step) for seed in seeds for step in steps}
    trajectory = {}
    path = run_dir / "ViT_trajectory.csv"
    required = {"seed", "step", "lr", *TRAJECTORY_METRICS}
    for line, row in csv_rows(path, required):
        try:
            key = (int(row["seed"]), int(row["step"]))
            if key not in expected_trajectory or key in trajectory:
                raise ValueError(f"Unexpected or repeated seed/checkpoint {key}.")
            close(float(row["lr"]), lr, "Learning rate")
            trajectory[key] = {name: nonnegative(row, name) for name in TRAJECTORY_METRICS}
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path}:{line}: {exc}") from exc
    missing = expected_trajectory.difference(trajectory)
    if missing:
        raise ValueError(f"{path}: missing {len(missing)} rows, including {min(missing)}.")

    expected_noise = {(seed, step, batch) for seed in seeds for step in steps for batch in batches}
    noise = {}
    path = run_dir / "ViT_noise.csv"
    required = {"seed", "step", "diagnostic_batch_size", "pairs", *NOISE_METRICS, *RATIOS}
    for line, row in csv_rows(path, required):
        try:
            key = (int(row["seed"]), int(row["step"]), int(row["diagnostic_batch_size"]))
            if key not in expected_noise or key in noise:
                raise ValueError(f"Unexpected or repeated seed/checkpoint/diagnostic batch {key}.")
            if int(row["pairs"]) != manifest["diagnostic_pairs"]:
                raise ValueError("Diagnostic pair count differs from the manifest.")
            metrics = {name: nonnegative(row, name) for name in NOISE_METRICS}
            close(metrics["factor_sq"], trajectory[key[:2]]["factor_sq"], "Factor size")
            bound = metrics["factor_sq"] * metrics["weight_noise_sq"]
            if metrics["factor_noise_sq"] > bound * (1 + 1e-4) + 1e-12:
                raise ValueError("Factor noise exceeds (||A_t||_2^2 + ||B_t||_2^2) times weight noise.")
            for name, denominator in (
                    ("amplification", metrics["weight_noise_sq"]),
                    ("normalized_amplification", bound)):
                if denominator == 0:
                    if row[name].strip():
                        raise ValueError(f"Undefined {name} must be an empty CSV cell.")
                    metrics[name] = np.nan
                else:
                    metrics[name] = nonnegative(row, name)
                    close(metrics[name], metrics["factor_noise_sq"] / denominator, name)
            noise[key] = metrics
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path}:{line}: {exc}") from exc
    missing = expected_noise.difference(noise)
    if missing:
        raise ValueError(f"{path}: missing {len(missing)} rows, including {min(missing)}.")
    return manifest, trajectory, noise


def summarize(samples):
    """Require all seeds at a point; undefined ratios are omitted, not set to zero."""
    samples = np.asarray(samples, dtype=np.float64)
    valid = np.isfinite(samples).all(axis=0)
    mean = np.full(samples.shape[1], np.nan)
    sd = np.full(samples.shape[1], np.nan)
    mean[valid] = samples[:, valid].mean(axis=0)
    sd[valid] = samples[:, valid].std(axis=0, ddof=1) if len(samples) > 1 else 0
    return mean, sd


def checkpoint_empirical_average(steps, values):
    """Average pre-update squared norms using each left checkpoint's interval.

    ``values`` has shape (seeds, checkpoints). At step t, the recorded value
    represents the state after t updates, hence the gradient before update
    t+1. The final state's gradient does not enter the average through t.
    Sparse checkpoints give a piecewise-constant approximation, not an
    exact or unbiased all-update average. Unit-spaced checkpoints are exact.
    """
    steps = np.asarray(steps, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if (steps.ndim != 1 or len(steps) < 2 or not np.isfinite(steps).all()
            or steps[0] != 0 or np.any(np.diff(steps) <= 0)
            or np.any(steps != np.floor(steps))):
        raise ValueError("Checkpoints must be increasing integer update counts starting at zero.")
    if (values.ndim != 2 or values.shape[0] < 1 or values.shape[1] != len(steps)
            or not np.isfinite(values).all() or np.any(values < 0)):
        raise ValueError("Squared empirical gradients must be a finite nonnegative seed-by-checkpoint array.")
    weighted = values[:, :-1] * np.diff(steps)[None, :]
    return np.cumsum(weighted, axis=1) / steps[None, 1:]


def draw_curve(ax, steps, samples, label, color, linestyle="-", linewidth=1.9, band=True):
    mean, sd = summarize(samples)
    ax.plot(steps, mean, label=label, color=color, linestyle=linestyle, linewidth=linewidth)
    if band and np.any(sd > 0):
        # These are standard-deviation bands, not confidence intervals.
        ax.fill_between(steps, mean - sd, mean + sd, color=color, alpha=0.14, linewidth=0)
    return mean, sd


def log_if_positive(ax, statistics, include=None):
    lower, upper = [], []
    for mean, sd in statistics:
        valid = np.isfinite(mean) & np.isfinite(sd)
        lower.extend((mean[valid] - sd[valid]).tolist())
        upper.extend((mean[valid] + sd[valid]).tolist())
    if include is not None:
        lower.append(include)
        upper.append(include)
    if lower and min(lower) > 0 and max(upper) > 4 * min(lower):
        ax.set_yscale("log")


def decorate(ax, title, ylabel, steps):
    ax.set_title(title, loc="left", pad=10)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Training updates")
    ax.set_xlim(steps[0], steps[-1])
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(formatter)
    ax.grid(axis="y", which="major", alpha=0.2)
    ax.margins(y=0.08)


def save(figure, output):
    for extension in ("png", "pdf"):
        path = output.with_suffix(f".{extension}")
        figure.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
        print(f"Saved {path}", flush=True)
    plt.close(figure)


def plot_training(manifest, trajectory, output):
    seeds, steps = manifest["seeds"], manifest["checkpoints"]

    def samples(name):
        return [[trajectory[seed, step][name] for step in steps] for seed in seeds]

    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.0))
    draw_curve(axes[0], steps, samples("train_loss"), "Training", COLORS[0])
    decorate(axes[0], "(a) Training objective", "Full training cross-entropy", steps)
    draw_curve(axes[1], steps, samples("factor_sq"), r"$\|A_t\|_2^2+\|B_t\|_2^2$", COLORS[0])
    decorate(axes[1], "(b) Factor growth", r"$\|A_t\|_2^2+\|B_t\|_2^2$", steps)
    averages = checkpoint_empirical_average(steps, samples("train_grad_sq"))
    exact = bool(np.all(np.diff(steps) == 1))
    stats = [draw_curve(axes[2], steps[1:], averages, "Empirical", COLORS[0])]
    log_if_positive(axes[2], stats)
    ylabel = r"$\frac{1}{t}\sum_{k=1}^{t}\|\nabla J_S(B_k,A_k)\|_F^2$"
    if not exact:
        ylabel = "Checkpoint estimate of\n" + ylabel
    decorate(axes[2], "(c) Averaged empirical stationarity", ylabel, steps)
    figure.tight_layout()
    save(figure, output / "ViT_training")
    method = "exact_pre_update_mean" if exact else "left_checkpoint_interval_weighted_estimate"
    with (output / "ViT_empirical_average.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("seed", "step", "average_train_grad_sq", "averaging_method"))
        writer.writeheader()
        for seed, curve in zip(seeds, averages):
            for step, value in zip(steps[1:], curve):
                writer.writerow({"seed": seed, "step": step, "average_train_grad_sq": float(value),
                                 "averaging_method": method})


def select_noise_reference_batch(manifest, requested=None):
    """Choose a diagnostic size from the design, never from the plotted values."""
    batches = sorted(manifest["diagnostic_batches"])
    if requested is not None:
        if requested not in batches:
            raise ValueError(f"Noise reference batch must be one of {batches}.")
        return requested
    training_batch = manifest.get("batch_size")
    return training_batch if training_batch in batches else batches[len(batches) // 2]


def plot_noise(manifest, trajectory, noise, output, reference_batch):
    seeds, steps, batches = manifest["seeds"], manifest["checkpoints"], manifest["diagnostic_batches"]
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.4))
    names = ("weight_noise_sq", "factor_noise_sq")
    titles = ("(a) Weight-gradient noise", "(b) Factor-gradient noise")
    labels = (r"Estimate of $\mathbb{E}\|\xi_t\|_F^2$",
              "Estimate of\n" + r"$\mathbb{E}[\|B_t^\top\xi_t\|_F^2+\|\xi_t A_t^\top\|_F^2]$")
    handles = []
    statistics = [[], []]
    for index, batch in enumerate(batches):
        color = COLORS[index % len(COLORS)]
        linestyle = LINESTYLES[index % len(LINESTYLES)]
        label = f"Diagnostic batch size {batch}"
        handles.append(Line2D([], [], color=color, linestyle=linestyle, label=label, linewidth=1.9))
        for panel, (ax, name) in enumerate(zip(axes[:2], names)):
            samples = [[noise[seed, step, batch][name] for step in steps] for seed in seeds]
            statistics[panel].append(draw_curve(ax, steps, samples, label, color, linestyle))
    for index, (ax, title, ylabel) in enumerate(zip(axes[:2], titles, labels)):
        log_if_positive(ax, statistics[index])
        decorate(ax, title, ylabel, steps)

    # At a fixed iterate the common inverse-batch-size scaling cancels from
    # the normalized ratio. Show one design-selected batch and seed variation,
    # instead of three nearly identical batch-size curves.
    ratios = np.asarray([[noise[seed, step, reference_batch]["normalized_amplification"]
                          for step in steps] for seed in seeds])
    color = COLORS[batches.index(reference_batch) % len(COLORS)]
    linestyle = LINESTYLES[batches.index(reference_batch) % len(LINESTYLES)]
    for curve in ratios:
        axes[2].plot(steps, curve, color=color, linewidth=0.9, alpha=0.22)
    ratio_mean, _ = summarize(ratios)
    axes[2].plot(steps, ratio_mean, color=color, linestyle=linestyle, linewidth=2.1)
    axes[2].axhline(1.0, color="#333333", linestyle=":", linewidth=1.4)
    finite = ratios[np.isfinite(ratios)]
    if finite.size and finite.min() > 0 and max(1.0, finite.max()) > 4 * finite.min():
        axes[2].set_yscale("log")
    decorate(axes[2], "(c) Fraction of variance bound",
             "Factor-gradient variance / upper bound", steps)
    axes[2].text(0.04, 0.85, f"Diagnostic batch size {reference_batch}",
                 transform=axes[2].transAxes, va="top", fontsize=9,
                 bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2})
    handles.append(Line2D([], [], color="#333333", linestyle=":", linewidth=1.4,
                          label="Upper bound 1 in (c)"))
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.997),
                  ncol=min(4, len(handles)), frameon=False, fontsize=9.2)
    legend_height = 0.11 * math.ceil(len(handles) / 4)
    figure.tight_layout(rect=(0, 0, 1, 1 - legend_height))
    save(figure, output / "ViT_noise")
    with (output / "ViT_noise_bound_fraction.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("seed", "step", "diagnostic_batch_size",
                                                    "fraction_of_variance_bound"))
        writer.writeheader()
        for seed, curve in zip(seeds, ratios):
            for step, value in zip(steps, curve):
                writer.writerow({"seed": seed, "step": step, "diagnostic_batch_size": reference_batch,
                                 "fraction_of_variance_bound": float(value) if np.isfinite(value) else ""})


def plot_results(run_dir, output_dir=None, noise_reference_batch=None):
    run_dir = Path(run_dir).expanduser().resolve()
    manifest, trajectory, noise = read_run(run_dir)
    reference_batch = select_noise_reference_batch(manifest, noise_reference_batch)
    output = Path(output_dir).expanduser().resolve() if output_dir else run_dir / "figures"
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42}):
        plot_training(manifest, trajectory, output)
        plot_noise(manifest, trajectory, noise, output, reference_batch)
    spacing = np.diff(manifest["checkpoints"])
    exact = bool(np.all(spacing == 1))
    details = {
        "source_run": str(run_dir),
        "seed_count": len(manifest["seeds"]),
        "training_panel_c": {
            "quantity": "(1/t) sum_{k=1}^t ||grad J_S(B_k,A_k)||_F^2",
            "averaging_method": "exact_pre_update_mean" if exact else "left_checkpoint_interval_weighted_estimate",
            "description": "Each saved left-checkpoint squared empirical gradient is weighted by the number of updates until the next checkpoint. The final checkpoint gradient is excluded from preceding intervals. At t=0 the average is undefined and not plotted.",
            "limitation": None if exact else "Sparse measurements give a piecewise-constant approximation, not an exact or unbiased all-update average. Discretization error is not measured by the seed standard-deviation band.",
            "checkpoint_spacing_min": int(spacing.min()),
            "checkpoint_spacing_max": int(spacing.max()),
            "uncertainty": "One sample standard deviation across per-seed averages.",
            "csv": "ViT_empirical_average.csv",
        },
        "noise_panels_a_b": "Means across seeds, with one sample standard deviation. Both variances are estimated from the same independent minibatch pairs.",
        "noise_panel_c": {
            "x": "Training updates",
            "y": "estimated factor-gradient variance / ((||A_t||_2^2 + ||B_t||_2^2) * estimated weight-gradient variance)",
            "diagnostic_batch_size": reference_batch,
            "selection": "Explicit argument, otherwise the training batch size when available, otherwise the middle sorted diagnostic size. Selection does not depend on outcomes.",
            "summary": "The thick curve is the mean of per-seed normalized ratios at each checkpoint; faint curves show individual seeds. No SD band is used in this panel. Undefined ratios are missing, with line breaks. The mean requires every seed to have a defined ratio.",
            "upper_bound": "Horizontal line at 1. The paired estimator satisfies the same algebraic bound; this is a diagnostic of bound tightness, not an independent convergence test.",
            "csv": "ViT_noise_bound_fraction.csv",
        },
    }
    (output / "plot_details.json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="completed run directory, default ViT_Noise.latest.txt")
    parser.add_argument("--output-dir", type=Path, help="default: the run directory's figures folder")
    parser.add_argument("--noise-reference-batch", type=int,
                        help="diagnostic size for noise panel (c); default: training size if measured, else middle diagnostic size")
    args = parser.parse_args()
    try:
        run_dir = args.run_dir
        if run_dir is None:
            pointer = HERE / "ViT_Noise.latest.txt"
            if not pointer.is_file():
                raise ValueError(f"{pointer} is missing. Run ViT_Run.py first, or specify --run-dir.")
            latest = pointer.read_text(encoding="utf-8").strip()
            if not latest:
                raise ValueError(f"Empty latest-run pointer: {pointer}")
            run_dir = HERE / Path(latest).expanduser()
        plot_results(run_dir, args.output_dir, args.noise_reference_batch)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"ViT plotting error: {exc}\n")


if __name__ == "__main__":
    main()

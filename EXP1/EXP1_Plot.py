#!/usr/bin/env python3
"""Plot the revised EXP1: minibatch loss, factor growth, and a trajectory average.

Save beside EXP1.sh and EXP1_Run.py inside LoRASGD/EXP1/.
Run: python EXP1_Plot.py
Or:  python EXP1_Plot.py --run-dir /path/to/saved/experiment
For the theorem's squared-gradient average, add --gradient-stat mean_squared.

The default is the arithmetic mean of full empirical factor-gradient NORMS.
Each gradient is evaluated on all training examples at a pre-update iterate.
Every training step contributes once. This is neither a minibatch-gradient
average nor the norm of an averaged gradient vector. No smoothing is applied.
Old EXP1 logs containing only epoch-end gradients cannot recover this average.
This file only reads saved results and never starts training.
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

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 12,
    "axes.linewidth": 0.9, "axes.facecolor": "white",
    "figure.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42,
    "path.simplify": False,
})

RANK_COLORS = ((4, "#0072BD"), (8, "#E00000"), (16, "#00A000"))


def read_csv(path, required):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Missing columns {sorted(missing)} in {path}. "
                "Use results from the revised EXP1_Run.py; epoch-end gradients "
                "cannot recover an average over every training step."
            )
        return list(reader)


def finite_column(rows, key, path):
    values = [float(row[key]) for row in rows]
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError(f"Invalid {key} measurements in {path}")
    return values


def running_mean(values):
    total = 0.0
    means = []
    for count, value in enumerate(values, start=1):
        total += value
        means.append(total / count)
    return means


def read_results(run):
    """Validate complete, paired runs and reconstruct both trajectory averages."""
    run = Path(run)
    path = run / "experiment.json"
    if not path.is_file():
        raise ValueError(
            f"Missing {path}. This plotter needs the revised EXP1 results with "
            "a full empirical gradient at every training step."
        )
    manifest = json.loads(path.read_text())
    if (manifest.get("schema_version") != 2 or manifest.get("experiment") != "EXP1"
            or manifest.get("ranks") != [4, 8, 16]):
        raise ValueError("Expected revised EXP1 schema 2 with widths 4, 8, and 16.")
    if manifest.get("status") != "completed":
        raise ValueError("EXP1 training is not marked completed.")
    epochs = int(manifest["epochs"])
    n_train = int(manifest["train_examples"])
    batch_size = int(manifest["batch_size"])
    expected = int(manifest["horizon_updates"])
    lr = float(manifest["lr"])
    if min(epochs, n_train, batch_size, expected) < 1 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("Invalid experiment counts or learning rate.")
    per_epoch = math.ceil(n_train / batch_size)
    if expected != epochs * per_epoch:
        raise ValueError("Training horizon does not match the nominal epoch count.")
    entries = manifest["runs"]
    if (len(entries) != 3 or {entry["rank"] for entry in entries} != {4, 8, 16}
            or len({entry["folder"] for entry in entries}) != 3):
        raise ValueError("Expected three distinct adapter-width runs.")
    by_rank = {entry["rank"]: entry for entry in entries}
    common_settings = common_batches = common_base = None
    curves = {}

    for rank, color in RANK_COLORS:
        entry = by_rank[rank]
        folder = run / entry["folder"]
        if entry.get("status") != "completed":
            raise ValueError(f"Run is not completed: {folder}")
        config = json.loads((folder / "config.json").read_text())
        summary = json.loads((folder / "summary.json").read_text())
        if (config["rank"] != rank or config["schedule"] != "constant"
                or config["sampling"] != "iid" or config["lambda_product"] != 0
                or any(config[key] != manifest[key] for key in
                       ("epochs", "train_examples", "test_examples", "batch_size", "seed", "data_sha256"))
                or not math.isclose(float(config["actual_lr"]), lr, rel_tol=1e-12)):
            raise ValueError(f"Unexpected experiment settings in {folder}")
        if summary["status"] != "completed" or summary["completed_updates"] != expected:
            raise ValueError(f"Training is incomplete: {folder}")
        settings = tuple(config[key] for key in (
            "epochs", "train_examples", "test_examples", "batch_size", "data_sha256",
            "actual_lr", "seed", "sampling", "schedule", "init_std", "lambda_product"))
        if common_settings is not None and settings != common_settings:
            raise ValueError("The runs changed settings other than adapter width.")
        common_settings = settings
        batches = summary["minibatch_sha256"]
        base = summary["base_parameters_sha256"]
        if not batches or not base or not summary["initial_parameters_sha256"]:
            raise ValueError(f"Missing pairing hashes in {folder}")
        if common_batches is not None and batches != common_batches:
            raise ValueError("The runs used different minibatch sampling sequences.")
        if common_base is not None and base != common_base:
            raise ValueError("The runs used different frozen base classifiers.")
        common_batches, common_base = batches, base

        update_path = folder / "updates.csv"
        updates = read_csv(update_path, (
            "step", "lr", "batch_loss_before_step", "full_factor_gradient_sq",
            "full_factor_gradient_norm", "running_mean_gradient_sq", "running_mean_gradient_norm"))
        steps = [int(row["step"]) for row in updates]
        if steps != list(range(1, expected + 1)):
            raise ValueError(f"Missing or duplicate per-step gradient measurements in {update_path}")
        if any(not math.isclose(float(row["lr"]), lr, rel_tol=1e-12) for row in updates):
            raise ValueError(f"Learning rate changed during training: {folder}")
        losses = finite_column(updates, "batch_loss_before_step", update_path)
        squared = finite_column(updates, "full_factor_gradient_sq", update_path)
        norms = finite_column(updates, "full_factor_gradient_norm", update_path)
        if any(not math.isclose(norm * norm, sq, rel_tol=1e-10, abs_tol=1e-14)
               for norm, sq in zip(norms, squared)):
            raise ValueError(f"Inconsistent gradient norms and squared norms in {update_path}")
        mean_norms, mean_squared = running_mean(norms), running_mean(squared)
        for column, computed in (("running_mean_gradient_norm", mean_norms),
                                 ("running_mean_gradient_sq", mean_squared)):
            recorded = finite_column(updates, column, update_path)
            if any(not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-14)
                   for a, b in zip(recorded, computed)):
                raise ValueError(f"Incorrect saved trajectory average {column} in {update_path}")

        pair_path = folder / "pair_metrics.csv"
        pairs = read_csv(pair_path, ("step", "a_spectral_sq", "b_spectral_sq", "spectral_sum"))
        pair_steps = [int(row["step"]) for row in pairs]
        if pair_steps != list(range(0, expected + 1)):
            raise ValueError(f"Missing or duplicate per-step factor norms in {pair_path}")
        a_sq = finite_column(pairs, "a_spectral_sq", pair_path)
        b_sq = finite_column(pairs, "b_spectral_sq", pair_path)
        spectral_sum = finite_column(pairs, "spectral_sum", pair_path)
        if any(not math.isclose(a + b, total, rel_tol=1e-10, abs_tol=1e-12)
               for a, b, total in zip(a_sq, b_sq, spectral_sum)):
            raise ValueError(f"Inconsistent squared spectral-norm sum in {pair_path}")
        curves[rank] = {
            "color": color, "steps": steps, "losses": losses,
            "factor_steps": pair_steps, "spectral_sum": spectral_sum,
            "mean_norm": mean_norms, "mean_squared": mean_squared,
        }
    return manifest, curves


def plot_results(run, script_dir, gradient_stat="mean_norm"):
    if gradient_stat not in {"mean_norm", "mean_squared"}:
        raise ValueError("gradient_stat must be mean_norm or mean_squared")
    run, script_dir = Path(run), Path(script_dir)
    manifest, curves = read_results(run)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), sharex=True)
    for rank, _ in RANK_COLORS:
        curve = curves[rank]
        color, steps = curve["color"], curve["steps"]
        label = fr"$r={rank}$"
        axes[0].plot([step - 1 for step in steps], curve["losses"],
                     color=color, linewidth=0.8, label=label)
        axes[1].plot(curve["factor_steps"], curve["spectral_sum"],
                     color=color, linewidth=1.5, label=label)
        axes[2].plot(steps, curve[gradient_stat], color=color, linewidth=1.5, label=label)
        print(f"r={rank}: {manifest['epochs']} epochs, {steps[-1]} updates, "
              f"final {gradient_stat}={curve[gradient_stat][-1]:.6g}")

    lr = float(manifest["lr"])
    fig.suptitle("ResNet-18 pretrained on ImageNet" + fr" ($\eta={lr:g}$)", fontsize=16)
    axes[0].set_title("(a) Minibatch loss", fontsize=14, pad=9)
    axes[1].set_title("(b) Factor growth", fontsize=14, pad=9)
    axes[0].set_ylabel("Cross-entropy loss", fontsize=16, labelpad=7)
    axes[1].set_ylabel(r"$\|A_t\|_2^2+\|B_t\|_2^2$", fontsize=18, labelpad=8)
    if gradient_stat == "mean_norm":
        axes[2].set_title("(c) Average empirical gradient norm", fontsize=14, pad=9)
        axes[2].set_ylabel(r"$\frac{1}{t}\sum_{k=1}^{t}\|\nabla J_S(B_k,A_k)\|_F$",
                           fontsize=16, labelpad=8)
        filename = "loss_factor_growth_average_gradient_norm"
    else:
        axes[2].set_title("(c) Average squared empirical gradient norm", fontsize=13, pad=9)
        axes[2].set_ylabel(r"$\frac{1}{t}\sum_{k=1}^{t}\|\nabla J_S(B_k,A_k)\|_F^2$",
                           fontsize=16, labelpad=8)
        filename = "loss_factor_growth_average_squared_gradient_norm"
    for ax, location in zip(axes, ("upper right", "upper left", "best")):
        ax.set_xlabel("Training steps", fontsize=16, labelpad=7)
        ax.set_xlim(0, manifest["horizon_updates"])
        maximum = max(max(line.get_ydata()) for line in ax.lines)
        ax.set_ylim(0, 1.06 * maximum if maximum > 0 else 1)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        ax.xaxis.set_major_formatter(formatter)
        ax.tick_params(direction="in", length=4, width=0.8, labelsize=12)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
        ax.grid(False)
        ax.legend(title="Adapter width", loc=location, frameon=False,
                  fontsize=12, title_fontsize=12, handlelength=2.3)
    fig.tight_layout(rect=(0, 0, 1, 0.97), w_pad=2.2)
    destinations = []
    for output in (run / "figures", script_dir / "figures" / f"EXP1_{manifest['epochs']}epochs"):
        output.mkdir(parents=True, exist_ok=True)
        for extension in ("png", "pdf"):
            path = output / f"{filename}.{extension}"
            fig.savefig(path, dpi=300)
            destinations.append(path)
            print("Saved:", path)
    plt.close(fig)
    return destinations


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="Saved revised EXP1 experiment")
    parser.add_argument("--gradient-stat", choices=("mean_norm", "mean_squared"),
                        default="mean_norm", help="Trajectory average shown in panel (c)")
    args = parser.parse_args()
    if args.run_dir is None:
        pointer = script_dir / "EXP1.latest.txt"
        if not pointer.is_file():
            parser.error("EXP1.latest.txt is missing. Supply --run-dir.")
        run = Path(pointer.read_text().strip()).expanduser()
        if not run.is_absolute():
            run = script_dir / run
    else:
        run = args.run_dir.expanduser()
    run = run.resolve()
    print("Plotting saved EXP1 results:", run)
    print("Averaging a full empirical factor-gradient measurement from every training step.")
    plot_results(run, script_dir, args.gradient_stat)


if __name__ == "__main__":
    main()

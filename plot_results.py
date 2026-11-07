#!/usr/bin/env python3
"""Plot exact factor measurements; aggregate pairs and seeds explicitly."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_run(path):
    path = Path(path)
    with (path / "pair_metrics.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No norm measurements in {path}")
    raw_config = json.loads((path / "config.json").read_text())
    config = {**raw_config.get("arguments", {}), **raw_config}
    if "arguments" in raw_config:
        config.setdefault("model", "resnet18")
        config.setdefault("data_kind", raw_config.get("dataset", "unspecified"))
        config.setdefault("actual_lr", raw_config.get("optimizer", {}).get("lr", config.get("lr")))
    by_step = defaultdict(list)
    for row in rows:
        by_step[int(row["step"])].append(row)
    steps = np.array(sorted(by_step))
    pair_count = len({row["pair"] for row in by_step[int(steps[0])]})
    names = sorted({row["pair"] for row in rows})
    if any(len(by_step[int(step)]) != pair_count for step in steps):
        raise ValueError(f"Inconsistent pair count across checkpoints in {path}")
    sums = {}
    for field in ["a_spectral_sq", "b_spectral_sq", "spectral_sum", "frobenius_sum"]:
        sums[field] = np.array([sum(float(row[field]) for row in by_step[int(step)]) for step in steps])
    heatmap = np.array([[float(next(row["spectral_sum"] for row in by_step[int(step)] if row["pair"] == name))
                         for step in steps] for name in names])
    if not config.get("data_sha256") and (path / "dataset_indices.npz").exists():
        import hashlib
        digest = hashlib.sha256((config.get("data_kind", "") + str(config.get("data_preprocessing", ""))).encode())
        digest.update((path / "dataset_indices.npz").read_bytes())
        if "synthetic" in str(config.get("data_kind", "")).lower():
            digest.update(str(config.get("seed", "")).encode())
        config["data_sha256"] = digest.hexdigest()
    return {"path": path, "config": config, "steps": steps, "sums": sums,
            "pair_count": pair_count, "pair_names": names, "heatmap": heatmap, "rows": rows}


def group_key(run):
    c = run["config"]
    # A configuration grouping, not an assertion that all these models share one theorem.
    return tuple(str(c.get(k, "unspecified")) for k in ["model", "data_kind", "data_sha256", "rank", "lr",
                                                       "actual_lr", "init_std", "sampling", "schedule", "lambda_product"])


def save(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-layer", action="store_true", help="Also save A/B/sum plot for every pair in every run")
    args = parser.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output folder must be empty; choose a new --output")
    runs = [read_run(path) for path in args.runs]
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    groups = defaultdict(list)
    for run in runs:
        groups[group_key(run)].append(run)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.1), constrained_layout=True)
    fields = ["a_spectral_sq", "b_spectral_sq", "spectral_sum"]
    titles = [r"$\sum_\ell\|A^{(\ell)}\|_2^2$", r"$\sum_\ell\|B^{(\ell)}\|_2^2$",
              r"$\sum_\ell(\|A^{(\ell)}\|_2^2+\|B^{(\ell)}\|_2^2)$"]
    if all(run["pair_count"] == 1 for run in runs):
        titles = [r"$\|A_t\|_2^2$", r"$\|B_t\|_2^2$", r"$\|A_t\|_2^2+\|B_t\|_2^2$"]
    for key, group in groups.items():
        common = group[0]["steps"]
        for run in group[1:]:
            common = np.intersect1d(common, run["steps"])
        if len(common) < 2:
            raise ValueError("Seed aggregation needs at least two common logged updates")
        config = group[0]["config"]
        label = f"rank={config.get('rank')}, lr={float(config.get('actual_lr', config.get('lr', 0))):.3g}, n={len(group)}"
        for ax, field in zip(axes, fields):
            values = np.stack([run["sums"][field][np.searchsorted(run["steps"], common)] for run in group])
            mean = values.mean(axis=0)
            line, = ax.plot(common, mean, label=label)
            if len(group) > 1:
                sd = values.std(axis=0, ddof=1)
                ax.fill_between(common, np.maximum(0, mean - sd), mean + sd, color=line.get_color(), alpha=0.16)
            # C_T is a fixed-horizon, single-pair reference ONLY when settings match.
            if field == "spectral_sum" and len(group) == 1 and group[0]["pair_count"] == 1:
                c = config
                if c.get("matches_theorem_step_and_sampling"):
                    ax.axhline(c["reference_fixed_horizon_C_T"], linestyle="--", color=line.get_color(),
                               alpha=0.65, label="fixed-horizon theory C_T")
    for ax, title in zip(axes, titles):
        ax.set(title=title, xlabel="Optimizer updates (0 = initialization)")
        ax.grid(alpha=0.22)
    axes[-1].legend(fontsize=8)
    fig.suptitle("Exact numerical spectral norms; bands = mean ± 1 sample SD across seeds", fontsize=12)
    save(fig, args.output, "spectral_growth")

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    summaries = []
    for run in runs:
        total = run["sums"]["spectral_sum"]
        if total[0] <= 0:
            raise ValueError("Cannot normalize by zero initial spectral sum")
        label = run["path"].name
        ax.plot(run["steps"], total / total[0], label=label)
        summaries.append({"run": str(run["path"]), "pairs": run["pair_count"], "initial_spectral_sum": total[0],
                          "final_spectral_sum": total[-1], "maximum_logged_spectral_sum": total.max(),
                          "final_over_initial": total[-1] / total[0],
                          "logging_every_update": run["config"].get("log_every", 1) == 1})
    ax.set(xlabel="Optimizer updates", ylabel="Aggregate spectral sum / its initial value",
           title="Relative growth (each run uses its own initialization)")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=7, ncol=2)
    save(fig, args.output, "relative_growth")
    with (args.output / "growth_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)
    (args.output / "plot_metadata.json").write_text(json.dumps({
        "runs": [str(path) for path in args.runs], "seed_band": "mean plus/minus one sample standard deviation",
        "aggregation": "sum over factor pairs, separately summing each pair's squared spectral norm",
        "not_global_stacked_spectral_norm": True, "no_growth_exponent_fitted": True}, indent=2) + "\n")
    for i, run in enumerate(runs):
        if run["pair_count"] > 1:
            fig, ax = plt.subplots(figsize=(10, max(3.5, 0.23 * run["pair_count"])), constrained_layout=True)
            image = ax.imshow(np.log1p(run["heatmap"]), aspect="auto", interpolation="nearest")
            ticks = np.linspace(0, len(run["steps"]) - 1, min(6, len(run["steps"]))).astype(int)
            ax.set_xticks(ticks, [str(run["steps"][t]) for t in ticks])
            ax.set_yticks(range(len(run["pair_names"])), run["pair_names"], fontsize=7)
            ax.set(xlabel="Optimizer updates", title=f"Per-pair growth: {run['path'].name}")
            fig.colorbar(image, ax=ax, label="log(1 + pair spectral sum)")
            save(fig, args.output, f"run_{i:02d}_layer_heatmap")
        if args.per_layer:
            for j, name in enumerate(run["pair_names"]):
                rows = [row for row in run["rows"] if row["pair"] == name]
                steps = [int(row["step"]) for row in rows]
                fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
                for field, label in zip(fields, ["A squared spectral norm", "B squared spectral norm", "Sum"]):
                    ax.plot(steps, [float(row[field]) for row in rows], label=label)
                ax.set(xlabel="Optimizer updates", title=name)
                ax.legend(); ax.grid(alpha=0.22)
                save(fig, args.output, f"run_{i:02d}_pair_{j:03d}")
    print(f"Saved plots to {args.output}")


if __name__ == "__main__":
    main()

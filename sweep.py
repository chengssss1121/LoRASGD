#!/usr/bin/env python3
"""Run matched ordinary-SGD constant-learning-rate/seed comparisons."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--features", type=Path)
    source.add_argument("--synthetic", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--learning-rates", type=float, nargs="+", default=[0.03, 0.05, 0.07])
    p.add_argument("--seeds", type=int, nargs="+", default=[13, 23, 37])
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--init-std", type=float, default=0.01)
    p.add_argument("--lambda-product", type=float, default=0.0)
    p.add_argument("--max-steps", type=int)
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        p.error("Output folder must be empty")
    args.output.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("train_logistic.py")
    jobs = []
    for lr in args.learning_rates:
        for seed in args.seeds:
            run = args.output / f"rank{args.rank}_lr{lr:g}_seed{seed}"
            cmd = [sys.executable, str(script), "--output", str(run), "--lr", str(lr), "--seed", str(seed),
                   "--rank", str(args.rank), "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
                   "--init-std", str(args.init_std), "--lambda-product", str(args.lambda_product)]
            cmd += ["--synthetic"] if args.synthetic else ["--features", str(args.features)]
            if args.max_steps:
                cmd += ["--max-steps", str(args.max_steps)]
            result = subprocess.run(cmd, check=False)
            jobs.append({"run": str(run), "command": cmd, "exit_code": result.returncode})
    (args.output / "sweep_manifest.json").write_text(json.dumps(jobs, indent=2) + "\n")
    if any(job["exit_code"] != 0 for job in jobs):
        raise SystemExit("At least one run failed; retained in sweep_manifest.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Simultaneous LoRA-SGD on cached CIFAR-10 features or offline smoke data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np

from lora_growth.logistic import evaluate, initialize, make_synthetic, simultaneous_step
from lora_growth.metrics import pair_metrics


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON defaults; explicit CLI arguments override them")
    parser.add_argument("--features", type=Path, help="NPZ with x_train,y_train,x_test,y_test")
    parser.add_argument("--synthetic", action="store_true", help="Use labeled offline smoke data, not CIFAR-10")
    parser.add_argument("--output", type=Path, default=Path("results/logistic"))
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--sampling", choices=["iid", "shuffle"], default="iid")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--data-seed", type=int, default=1729)
    parser.add_argument("--init-std", type=float, default=0.01)
    parser.add_argument("--lambda-product", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=1, help="Full evaluation every this many epochs")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--schedule", choices=["constant", "theory-general"], default="constant")
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--n-train", type=int, default=512)
    parser.add_argument("--n-test", type=int, default=256)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--n-classes", type=int, default=10)
    known, _ = parser.parse_known_args(argv)
    if known.config:
        config = json.loads(known.config.read_text())
        allowed = {action.dest for action in parser._actions} - {"help", "config"}
        if not isinstance(config, dict) or set(config) - allowed:
            parser.error("Config must be a JSON object with recognized argument names")
        parser.set_defaults(**config)
    args = parser.parse_args(argv)
    args.output = Path(args.output)
    if args.features is not None:
        args.features = Path(args.features)
    if bool(args.features) == bool(args.synthetic):
        parser.error("Choose exactly one of --features FILE and --synthetic")
    for name in ["rank", "epochs", "batch_size", "log_every", "eval_every", "n_train", "n_test", "feature_dim", "n_classes"]:
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if not np.isfinite(args.lr) or args.lr <= 0 or not np.isfinite(args.init_std) or args.init_std <= 0:
        parser.error("--lr and --init-std must be finite and positive")
    if not np.isfinite(args.lambda_product) or args.lambda_product < 0:
        parser.error("--lambda-product must be finite and nonnegative")
    if not 0 < args.delta < 1:
        parser.error("--delta must be between zero and one")
    if args.schedule == "theory-general" and args.sampling != "iid":
        parser.error("theory-general requires --sampling iid")
    return args


def load_features(args):
    if args.synthetic:
        return make_synthetic(args.n_train, args.n_test, args.feature_dim, args.n_classes, args.data_seed)
    with np.load(args.features, allow_pickle=False) as data:
        arrays = []
        for key in ["x_train", "y_train", "x_test", "y_test"]:
            if key not in data:
                raise ValueError(f"Missing array {key!r} in {args.features}")
            arrays.append(data[key])
    return tuple(arrays)


def validate_data(arrays):
    x_train, y_train, x_test, y_test = arrays
    for x, y in [(x_train, y_train), (x_test, y_test)]:
        if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or len(x) == 0:
            raise ValueError("Each nonempty x must be Nxd, with an N-vector of labels")
        if not np.issubdtype(y.dtype, np.integer) or not np.isfinite(x).all():
            raise ValueError("Features must be finite and labels must have an integer dtype")
        if np.min(y) < 0:
            raise ValueError("Labels must be zero-based nonnegative integers")
    if x_train.shape[1] != x_test.shape[1]:
        raise ValueError("Train and test feature dimensions differ")
    classes = int(max(np.max(y_train), np.max(y_test))) + 1
    if classes < 2:
        raise ValueError("At least two classes are required")
    return (np.asarray(x_train, dtype=np.float64), np.asarray(y_train, dtype=np.int64),
            np.asarray(x_test, dtype=np.float64), np.asarray(y_test, dtype=np.int64), classes)


def array_digest(arrays):
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def main(argv=None):
    args = parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output folder is nonempty: {args.output}; use a new --output")
    x_train, y_train, x_test, y_test, classes = validate_data(load_features(args))
    W0, bias, A, B = initialize(x_train.shape[1], classes, args.rank, args.init_std, args.seed)
    # Independent index RNG: initialization or logging cannot alter sample draws.
    index_rng = np.random.default_rng(np.random.SeedSequence([args.seed, 90210]))
    batches_per_epoch = math.ceil(len(y_train) / args.batch_size)
    horizon = args.epochs * batches_per_epoch
    if args.max_steps is not None:
        horizon = min(horizon, args.max_steps)
    initial = pair_metrics(A, B)
    init_train = evaluate(x_train, y_train, W0, bias, A, B, args.lambda_product)
    C0 = 1.0 + init_train["loss"] + initial["spectral_sum"]
    max_x_sq = float(np.max(np.sum(x_train**2, axis=1)))
    # Conservative fixed-training-set bounds for multiclass softmax CE.
    # Its logit Hessian norm is <= 1/2. The product regularizer cancels in noise.
    L = max(1.0, 0.5 * max_x_sq + args.lambda_product)
    b = math.sqrt(2.0 * max_x_sq)
    sigma_sq = 2.0 * max_x_sq
    theoretical_lr = 1.0 / (64.0 * ((b + 8.0 * L * math.sqrt(args.rank) * C0)**2
                                    + 2.0 * sigma_sq) * horizon**0.75 * math.log(4.0 * horizon / args.delta))
    lr = theoretical_lr if args.schedule == "theory-general" else args.lr
    matches_cap = lr <= theoretical_lr * (1 + 1e-12) and args.sampling == "iid"
    fixed_horizon_cap = 2.0 * initial["spectral_sum"] + horizon**0.25
    args.output.mkdir(parents=True, exist_ok=True)
    resolved = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    resolved.update({"actual_lr": lr, "horizon_updates": horizon, "train_examples": len(y_train),
                     "test_examples": len(y_test), "input_dimension": x_train.shape[1], "classes": classes,
                     "data_sha256": array_digest((x_train, y_train, x_test, y_test)),
                     "C0": C0, "L_fixed_dataset_bound": L, "b_fixed_dataset_bound": b,
                     "sigma_squared_fixed_dataset_bound": sigma_sq, "theory_lr_cap": theoretical_lr,
                     "matches_theorem_step_and_sampling": matches_cap, "reference_fixed_horizon_C_T": fixed_horizon_cap,
                     "paper_index_convention": "logged step k equals paper iterate t=k+1",
                     "lora_scaling": 1.0, "momentum": 0.0, "weight_decay": 0.0,
                     "gradient_clipping": False, "python": platform.python_version(), "numpy": np.__version__,
                     "command": sys.argv, "model": "logistic_classifier",
                     "data_kind": "synthetic_smoke" if args.synthetic else "cached_features",
                     "source_sha256": {str(path.relative_to(Path(__file__).parent)): hashlib.sha256(path.read_bytes()).hexdigest()
                                       for path in [Path(__file__), Path(__file__).parent / "lora_growth/logistic.py",
                                                    Path(__file__).parent / "lora_growth/metrics.py"]}})
    if args.features and args.features.with_suffix(".json").exists():
        resolved["feature_extraction_metadata"] = json.loads(args.features.with_suffix(".json").read_text())
    (args.output / "config.json").write_text(json.dumps(resolved, indent=2) + "\n")
    np.savez(args.output / "initial_factors.npz", A=A, B=B, W0=W0, bias=bias)
    start_time = time.perf_counter()
    max_sum = initial["spectral_sum"]
    max_imbalance_drift = 0.0
    initial_gram = A @ A.T - B.T @ B
    step = 0
    status = "completed"
    metric_fields = list(initial)
    with ((args.output / "pair_metrics.csv").open("w", newline="") as pair_file,
          (args.output / "updates.csv").open("w", newline="") as update_file,
          (args.output / "metrics.csv").open("w", newline="") as eval_file):
        pairs = csv.DictWriter(pair_file, fieldnames=["step", "epoch", "pair", "lr", "batch_loss"] + metric_fields
                               + ["running_max_spectral_sum", "imbalance_drift_spectral", "theory_cap_C_T"])
        updates = csv.DictWriter(update_file, fieldnames=["step", "epoch", "lr", "batch_loss_before_step",
                                                        "sample_factor_gradient_sq_before_step", "batch_accuracy_before_step"])
        evaluations = csv.DictWriter(eval_file, fieldnames=["step", "epoch", "train_loss", "train_accuracy", "test_loss",
                                                           "test_accuracy", "full_factor_gradient_sq"])
        pairs.writeheader(); updates.writeheader(); evaluations.writeheader()

        def log_pair(epoch, batch_loss=""):
            nonlocal max_sum, max_imbalance_drift
            values = pair_metrics(A, B)
            max_sum = max(max_sum, values["spectral_sum"])
            drift = float(np.linalg.norm(A @ A.T - B.T @ B - initial_gram, ord=2))
            max_imbalance_drift = max(max_imbalance_drift, drift)
            pairs.writerow({"step": step, "epoch": epoch, "pair": "classifier", "lr": lr,
                            "batch_loss": batch_loss, **values, "running_max_spectral_sum": max_sum,
                            "imbalance_drift_spectral": drift, "theory_cap_C_T": fixed_horizon_cap if matches_cap else ""})
            pair_file.flush()

        def log_evaluation(epoch):
            train = evaluate(x_train, y_train, W0, bias, A, B, args.lambda_product)
            test = evaluate(x_test, y_test, W0, bias, A, B, args.lambda_product)
            evaluations.writerow({"step": step, "epoch": epoch, "train_loss": train["loss"],
                                  "train_accuracy": train["accuracy"], "test_loss": test["loss"],
                                  "test_accuracy": test["accuracy"], "full_factor_gradient_sq": train["full_factor_gradient_sq"]})
            eval_file.flush()
            return train

        log_pair(0)
        log_evaluation(0)
        last_logged = 0
        for epoch in range(1, args.epochs + 1):
            order = index_rng.permutation(len(y_train)) if args.sampling == "shuffle" else None
            for batch in range(batches_per_epoch):
                if step >= horizon:
                    break
                indices = (index_rng.integers(len(y_train), size=args.batch_size) if order is None
                           else order[batch * args.batch_size:(batch + 1) * args.batch_size])
                A_next, B_next, loss, gradient_sq, accuracy = simultaneous_step(
                    x_train[indices], y_train[indices], W0, bias, A, B, lr, args.lambda_product)
                if not (np.isfinite(A_next).all() and np.isfinite(B_next).all() and np.isfinite(loss)
                        and np.isfinite(gradient_sq)):
                    status = "nonfinite_update"
                    break
                A, B = A_next, B_next
                step += 1
                updates.writerow({"step": step, "epoch": epoch, "lr": lr,
                                  "batch_loss_before_step": loss, "sample_factor_gradient_sq_before_step": gradient_sq,
                                  "batch_accuracy_before_step": accuracy})
                if step % args.log_every == 0 or step == horizon:
                    log_pair(epoch, loss)
                    last_logged = step
            update_file.flush()
            if status != "completed":
                break
            if epoch % args.eval_every == 0 or step == horizon:
                train = log_evaluation(epoch)
                print(f"epoch={epoch} updates={step}/{horizon} loss={train['loss']:.6f} "
                      f"spectral_sum={pair_metrics(A, B)['spectral_sum']:.6g}", flush=True)
            if step >= horizon:
                break
        if last_logged != step:
            log_pair(epoch)
        final = pair_metrics(A, B)
    np.savez(args.output / "final_factors.npz", A=A, B=B, W0=W0, bias=bias)
    summary = {"status": status, "completed_updates": step, "requested_updates": horizon,
               "initial_spectral_sum": initial["spectral_sum"], "final_spectral_sum": final["spectral_sum"],
               "max_logged_spectral_sum": max_sum, "max_logged_imbalance_drift": max_imbalance_drift,
               "log_every": args.log_every, "maximum_covers_every_update": args.log_every == 1,
               "growth_ratio_final_over_initial": final["spectral_sum"] / initial["spectral_sum"],
               "elapsed_seconds": time.perf_counter() - start_time,
               "theory_cap_C_T": fixed_horizon_cap if matches_cap else None,
               "logged_cap_exceeded": max_sum > fixed_horizon_cap if matches_cap else None,
               "final_train": evaluate(x_train, y_train, W0, bias, A, B, args.lambda_product)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved {args.output}; status={status}")
    if status != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

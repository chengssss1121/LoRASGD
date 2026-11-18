#!/usr/bin/env python3
"""Train one constant-step ViT adapter and measure checkpoint noise amplification.

The cache is produced by ViT_Prepare.py. Only the two factors of the last
MLP output projection are trained. Diagnostic minibatches never update them.
"""

import argparse
import csv
import json
from math import isfinite
from pathlib import Path
from tempfile import NamedTemporaryFile, mkdtemp

import numpy as np

HERE = Path(__file__).resolve().parent
TRAJECTORY_FIELDS = (
    "seed", "step", "lr", "train_loss", "test_loss", "factor_sq",
    "train_grad_sq", "test_grad_sq", "gradient_gap_sq",
)
NOISE_FIELDS = (
    "seed", "step", "diagnostic_batch_size", "pairs", "factor_sq",
    "weight_noise_sq", "factor_noise_sq", "weight_noise_se", "factor_noise_se",
    "amplification", "normalized_amplification",
)

class Arrays:
    """Share the same calculations between NumPy CPU and PyTorch CUDA."""

    def __init__(self, device, dtype):
        self.xp = np
        self.device = "cpu"
        self.dtype = dtype
        if device != "cpu":
            try:
                import torch
            except ImportError:
                if device != "auto":
                    raise ValueError("CUDA training requires a GPU-enabled PyTorch installation.")
                return
            if torch.cuda.is_available():
                self.xp = torch
                self.device = "cuda" if device == "auto" else device
            elif device != "auto":
                raise ValueError("CUDA is unavailable. Use --device cpu or select a GPU machine.")

    def float(self, array):
        if self.xp is np:
            return np.asarray(array, dtype=self.dtype)
        return self.xp.as_tensor(array, dtype=getattr(self.xp, self.dtype), device=self.device)

    def index(self, array):
        if self.xp is np:
            return np.asarray(array, dtype=np.int64)
        return self.xp.as_tensor(array, dtype=self.xp.long, device=self.device)

    def zeros(self, shape):
        if self.xp is np:
            return np.zeros(shape, dtype=self.dtype)
        return self.xp.zeros(shape, dtype=getattr(self.xp, self.dtype), device=self.device)

    def double(self, array):
        if self.xp is np:
            return np.asarray(array, dtype=np.float64)
        return array.to(dtype=self.xp.float64)

    def numpy(self, array):
        if self.xp is np:
            return np.asarray(array)
        return array.detach().cpu().numpy()

    def row_indices(self, count):
        if self.xp is np:
            return np.arange(count)
        return self.xp.arange(count, device=self.device)


def take(split, indices):
    return tuple(array[indices] for array in split)


def load_cache(path, arrays):
    with np.load(path, allow_pickle=False) as cache:
        splits = {}
        shape = None
        for name in ("pilot", "train", "test"):
            u, residual, labels = (cache[f"{name}_{field}"] for field in ("u", "r", "y"))
            if u.ndim != 2 or residual.ndim != 2 or labels.ndim != 1 or not len(labels):
                raise ValueError(f"Expected two feature matrices and nonempty 1-D labels for {name}.")
            if len(u) != len(labels) or len(residual) != len(labels):
                raise ValueError(f"Feature and label sizes differ in {name}.")
            dims = (u.shape[1], residual.shape[1])
            if min(dims) < 1 or (shape is not None and dims != shape):
                raise ValueError(f"Inconsistent feature widths in {name}.")
            shape = dims
            if labels.dtype.kind not in "iu" or np.any(labels < 0):
                raise ValueError(f"Expected nonnegative integer labels in {name}.")
            if not np.isfinite(u).all() or not np.isfinite(residual).all():
                raise ValueError(f"Nonfinite features in {name}.")
            splits[name] = arrays.float(u), arrays.float(residual), arrays.index(labels)
        gamma, beta = cache["ln_weight"], cache["ln_bias"]
        epsilon = float(cache["ln_eps"].item())
        if gamma.shape != (shape[1],) or beta.shape != gamma.shape:
            raise ValueError("LayerNorm parameters do not match the residual width.")
        if not np.isfinite(gamma).all() or not np.isfinite(beta).all() or not 0 < epsilon < float("inf"):
            raise ValueError("Invalid LayerNorm parameters.")
        classes = 1 + max(int(splits[name][2].max().item()) for name in ("pilot", "train"))
        if classes < 2 or int(splits["test"][2].max().item()) >= classes:
            raise ValueError("At least two training classes are required, with no unseen test labels.")
    normalization = (arrays.float(gamma), arrays.float(beta), epsilon)
    return splits, normalization, classes


def layer_norm(xp, residual, normalization):
    gamma, beta, epsilon = normalization
    centered = residual - residual.mean(1)[:, None]
    inv_std = 1 / xp.sqrt((centered * centered).mean(1)[:, None] + epsilon)
    unit = centered * inv_std
    return unit * gamma + beta, unit, inv_std


def cross_entropy(arrays, logits, labels, gradient=False, loss=True):
    xp = arrays.xp
    indices = arrays.row_indices(len(labels))
    shifted = logits - xp.amax(logits, 1)[:, None]
    exponentials = xp.exp(shifted)
    denominator = exponentials.sum(1)
    loss_sum = (xp.log(denominator) - shifted[indices, labels]).sum() if loss else None
    if not gradient:
        return loss_sum, None
    derivative = exponentials / denominator[:, None]
    derivative[indices, labels] -= 1
    return loss_sum, derivative


def warm_head(arrays, split, normalization, classes, epochs, seed):
    xp = arrays.xp
    _, residual, labels = split
    x, _, _ = layer_norm(xp, residual, normalization)
    parameters = [arrays.zeros((classes, x.shape[1])), arrays.zeros((classes,))]
    first = [arrays.zeros(weight.shape) for weight in parameters]
    second = [arrays.zeros(weight.shape) for weight in parameters]
    rng = np.random.default_rng(seed)
    update = 0
    for _ in range(epochs):
        order = arrays.index(rng.permutation(len(labels)))
        for start in range(0, len(labels), 32):
            indices = order[start:start + 32]
            x_batch, y_batch = x[indices], labels[indices]
            weight, bias = parameters
            _, derivative = cross_entropy(
                arrays, x_batch @ weight.T + bias, y_batch, gradient=True, loss=False,
            )
            derivative /= len(y_batch)
            gradients = derivative.T @ x_batch, derivative.sum(0)
            update += 1
            for i, gradient in enumerate(gradients):
                first[i] = 0.9 * first[i] + 0.1 * gradient
                second[i] = 0.999 * second[i] + 0.001 * (gradient * gradient)
                first_hat = first[i] / (1 - 0.9**update)
                second_hat = second[i] / (1 - 0.999**update)
                parameters[i] -= 0.001 * first_hat / (xp.sqrt(second_hat) + 1e-8)
    return tuple(parameters)


class Objective:
    def __init__(self, arrays, normalization, head, eval_batch_size):
        self.arrays = arrays
        self.normalization = normalization
        self.weight, self.bias = head
        self.eval_batch_size = eval_batch_size

    def batch(self, split, a, b, gradient=False, loss=True, ambient=False):
        """Return sums of per-sample loss and factor gradients for this batch."""
        u, residual, labels = split
        projection = u @ a.T
        x, unit, inv_std = layer_norm(
            self.arrays.xp, residual + projection @ b.T, self.normalization,
        )
        loss_sum, derivative = cross_entropy(
            self.arrays, x @ self.weight.T + self.bias, labels, gradient=gradient or ambient, loss=loss,
        )
        if not gradient and not ambient:
            return loss_sum, None
        dx = (derivative @ self.weight) * self.normalization[0]
        # Backpropagate through LayerNorm and then through the two factors.
        dr = inv_std * (dx - dx.mean(1)[:, None] - unit * (dx * unit).mean(1)[:, None])
        if ambient:
            return loss_sum, dr
        return loss_sum, ((dr @ b).T @ u, dr.T @ projection)

    def evaluate(self, split, a, b, gradient=False):
        """Compute a full-split mean loss or mean factor gradient in chunks."""
        count = len(split[2])
        if gradient:
            grad_a, grad_b = self.arrays.zeros(a.shape), self.arrays.zeros(b.shape)
        else:
            loss_sum = self.arrays.zeros(())
        for start in range(0, count, self.eval_batch_size):
            batch = take(split, slice(start, start + self.eval_batch_size))
            loss, derivatives = self.batch(batch, a, b, gradient=gradient, loss=not gradient)
            if gradient:
                grad_a += derivatives[0]
                grad_b += derivatives[1]
            else:
                loss_sum += loss
        if gradient:
            return grad_a / count, grad_b / count
        return float(loss_sum.item()) / count

    def metrics(self, split, a, b):
        """Full-split loss and mean factor gradient, accumulated in float64."""
        count = len(split[2])
        loss_total = self.arrays.double(self.arrays.zeros(()))
        grad_a = self.arrays.double(self.arrays.zeros(a.shape))
        grad_b = self.arrays.double(self.arrays.zeros(b.shape))
        for start in range(0, count, self.eval_batch_size):
            loss, gradients = self.batch(take(split, slice(start, start + self.eval_batch_size)),
                                         a, b, gradient=True, loss=True)
            loss_total += self.arrays.double(loss)
            grad_a += self.arrays.double(gradients[0])
            grad_b += self.arrays.double(gradients[1])
        return float(loss_total.item()) / count, (grad_a / count, grad_b / count)

    def step(self, split, a, b, step_size):
        _, (grad_a, grad_b) = self.batch(split, a, b, gradient=True, loss=False)
        scale = step_size / len(split[2])
        return a - scale * grad_a, b - scale * grad_b


def initialized_factors_and_batches(arrays, split, rank, count, batch_size, seed):
    init_seed, draw_seed = np.random.SeedSequence([seed, 0]).spawn(2)
    input_size, hidden_size = split[0].shape[1], split[1].shape[1]
    a = arrays.float(np.random.default_rng(init_seed).normal(0, 0.01, (rank, input_size)))
    b = arrays.zeros((hidden_size, rank))
    draws = np.random.default_rng(draw_seed).integers(len(split[2]), size=(count, batch_size))
    return a, b, arrays.index(draws)


def squared_pair(pair):
    return float(sum((part * part).sum() for part in pair).item())


def factor_scale(arrays, a, b):
    a64, b64 = arrays.double(a), arrays.double(b)
    return float((arrays.xp.linalg.eigvalsh(a64 @ a64.T)[-1]
                  + arrays.xp.linalg.eigvalsh(b64.T @ b64)[-1]).item())


def paired_noise_values(arrays, u1, dr1, u2, dr2, a, b):
    """Return half squared ambient/factor gradient differences in float64.

    For independent minibatch gradients g1 and g2 with the same mean,
    E[||g1-g2||^2 / 2] = E[||g1-E[g1]||^2]. The factor map is evaluated
    at the same frozen A,B for both minibatches. All reductions and products
    used for noise measurement are float64, independent of training dtype.
    """
    if len(u1) != len(u2) or not len(u1):
        raise ValueError("Noise estimation requires two equally sized nonempty minibatches.")
    xp = arrays.xp
    u1, u2, dr1, dr2, a, b = (
        arrays.double(value) for value in (u1, u2, dr1, dr2, a, b)
    )
    count = len(u1)
    if xp is np:
        u = np.concatenate((u1, u2), axis=0)
        s = np.concatenate((dr1, -dr2), axis=0) / count
    else:
        u = xp.cat((u1, u2), dim=0)
        s = xp.cat((dr1, -dr2), dim=0) / count
    # The gradient difference is s.T @ u. A sample-space identity avoids a 768x3072 matrix
    # whenever its two smaller Gram products require fewer multiplications.
    if len(u) * (u.shape[1] + s.shape[1]) <= u.shape[1] * s.shape[1]:
        products = (s @ s.T) * (u @ u.T)
        weight_sq = float(products.sum().item())
        rounding_scale = float(abs(products).sum().item())
        if weight_sq < -1e-10 * max(rounding_scale, np.finfo(float).tiny):
            raise FloatingPointError("Negative noise norm beyond float64 rounding tolerance.")
        weight_sq = max(weight_sq, 0.0)
    else:
        delta = s.T @ u
        weight_sq = float((delta * delta).sum().item())
    factor_a = (s @ b).T @ u
    factor_b = s.T @ (u @ a.T)
    factor_sq = squared_pair((factor_a, factor_b))
    return 0.5 * weight_sq, 0.5 * factor_sq


def noise_diagnostic(objective, train, a, b, batch_size, pairs, seed, step):
    """An unbiased paired-minibatch estimator of conditional gradient variance."""
    if pairs < 2:
        raise ValueError("At least two diagnostic pairs are required to estimate Monte Carlo uncertainty.")
    arrays = objective.arrays
    rng = np.random.default_rng(np.random.SeedSequence([seed, 739, step, batch_size]))
    samples = np.empty((pairs, 2), dtype=np.float64)
    c = factor_scale(arrays, a, b)
    for pair in range(pairs):
        indices = rng.integers(len(train[2]), size=(2, batch_size))
        split1 = take(train, arrays.index(indices[0]))
        split2 = take(train, arrays.index(indices[1]))
        _, dr1 = objective.batch(split1, a, b, loss=False, ambient=True)
        _, dr2 = objective.batch(split2, a, b, loss=False, ambient=True)
        weight, factor = paired_noise_values(arrays, split1[0], dr1, split2[0], dr2, a, b)
        if not isfinite(weight) or not isfinite(factor):
            raise FloatingPointError("Nonfinite noise measurement.")
        bound = c * weight
        tolerance = 1e-9 * max(bound, factor, np.finfo(float).tiny)
        if factor > bound + tolerance:
            raise FloatingPointError("Paired factor noise exceeds its spectral-norm upper bound.")
        samples[pair] = weight, factor
    mean = samples.mean(axis=0)
    se = samples.std(axis=0, ddof=1) / np.sqrt(pairs)
    amplification = float(mean[1] / mean[0]) if mean[0] > 0 else None
    normalized = float(mean[1] / (c * mean[0])) if c > 0 and mean[0] > 0 else None
    return {
        "seed": seed, "step": step, "diagnostic_batch_size": batch_size,
        "pairs": pairs, "factor_sq": c, "weight_noise_sq": float(mean[0]),
        "factor_noise_sq": float(mean[1]), "weight_noise_se": float(se[0]),
        "factor_noise_se": float(se[1]), "amplification": amplification,
        "normalized_amplification": normalized,
    }


def pick_learning_rate(args, objective, pilot_fit, pilot_eval):
    if args.lr is not None:
        return args.lr, []
    a0, b0, batches = initialized_factors_and_batches(
        objective.arrays, pilot_fit, args.rank, args.pilot_steps, args.batch_size, args.pilot_seed,
    )
    scores = []
    for candidate in args.lr_candidates:
        a, b = a0, b0
        try:
            for indices in batches:
                a, b = objective.step(take(pilot_fit, indices), a, b, candidate)
            validation_loss = objective.evaluate(pilot_eval, a, b)
        except FloatingPointError:
            validation_loss = float("nan")
        scores.append({"lr": candidate,
                       "validation_loss": validation_loss if isfinite(validation_loss) else None})
        print(f"Pilot constant lr={candidate:g}: validation loss={validation_loss:.6g}", flush=True)
    finite_scores = [row for row in scores if row["validation_loss"] is not None]
    if not finite_scores:
        raise FloatingPointError("No finite pilot candidate. Use smaller --lr-candidates.")
    best = min(finite_scores, key=lambda row: row["validation_loss"])
    return best["lr"], scores


def write_atomic(path, contents):
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(contents)
    temporary.replace(path)


def make_run_dir(requested):
    if requested is None:
        directory = HERE / "results"
        directory.mkdir(exist_ok=True)
        return Path(mkdtemp(prefix="ViT_Noise_", dir=directory))
    directory = requested.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError(f"Run directory is not empty: {directory}")
    return directory


def run(args):
    arrays = Arrays(args.device, args.dtype)
    if arrays.xp is not np:
        # Do not silently use TF32's reduced mantissa in diagnostic gradients.
        arrays.xp.backends.cuda.matmul.allow_tf32 = False
        arrays.xp.backends.cudnn.allow_tf32 = False
    cache = args.cache.expanduser().resolve()
    if not cache.is_file():
        raise ValueError(f"Cache missing: {cache}. Run ViT_Prepare.py or pass --cache PATH.")
    splits, normalization, classes = load_cache(cache, arrays)
    pilot = splits["pilot"]
    if len(pilot[2]) < 2:
        raise ValueError("The pilot split requires at least two samples.")
    order = arrays.index(np.random.default_rng(args.pilot_seed).permutation(len(pilot[2])))
    validation_size = max(1, len(pilot[2]) // 5)
    pilot_eval = take(pilot, order[:validation_size])
    pilot_fit = take(pilot, order[validation_size:])
    print(f"Using {arrays.device}; warming the frozen head on {len(pilot_fit[2])} pilot samples.", flush=True)
    head = warm_head(arrays, pilot_fit, normalization, classes, args.head_epochs, args.pilot_seed)
    objective = Objective(arrays, normalization, head, args.eval_batch_size)
    lr, scores = pick_learning_rate(args, objective, pilot_fit, pilot_eval)
    print(f"Constant learning rate: {lr:g}. Diagnostic batch sizes do not change training.", flush=True)
    train, test = splits["train"], splits["test"]
    checkpoints = np.unique(np.linspace(0, args.steps, min(args.checkpoints, args.steps) + 1,
                                       dtype=np.int64)).tolist()
    checkpoint_set = set(checkpoints)
    directory = make_run_dir(args.run_dir)
    print(f"Run directory: {directory}", flush=True)
    manifest = {
        "schema_version": 1, "experiment": "ViT_Noise_EXP", "status": "running",
        "cache": str(cache), "device": arrays.device, "dtype": args.dtype,
        "rank": args.rank, "batch_size": args.batch_size, "steps": args.steps,
        "seeds": args.seeds, "checkpoints": checkpoints,
        "diagnostic_batches": args.diagnostic_batches, "diagnostic_pairs": args.diagnostic_pairs,
        "lr": lr, "lr_candidates": args.lr_candidates, "lr_source": "user" if args.lr else "pilot",
        "pilot_scores": scores, "pilot_seed": args.pilot_seed, "pilot_steps": args.pilot_steps,
        "head_epochs": args.head_epochs, "head_fit_size": len(pilot_fit[2]),
        "pilot_eval_size": len(pilot_eval[2]),
        "split_sizes": {name: len(split[2]) for name, split in splits.items()},
        "eval_batch_size": args.eval_batch_size,
        "sampling": "IID with replacement for training and each diagnostic minibatch",
        "metric_timing": "instantaneous at initialization and after listed update counts",
        "noise_estimator": "mean of half squared differences of independent minibatch-gradient pairs",
        "ratio_note": "ratio of paired variance estimates; not an unbiased estimator of the variance ratio",
        "diagnostic_dtype": "float64 products and reductions; forward/adjoints use training dtype",
        "factor_scale": "squared spectral norm of A plus squared spectral norm of B",
        "gradient_note": "full-split mean factor gradients, not minibatch gradients or cumulative averages",
    }
    manifest_path = directory / "experiment.json"
    write_atomic(manifest_path, json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(directory / "frozen_head.npz", weight=arrays.numpy(head[0]), bias=arrays.numpy(head[1]))
    try:
        with (directory / "ViT_trajectory.csv").open("x", newline="", encoding="utf-8") as trajectory_stream, \
             (directory / "ViT_noise.csv").open("x", newline="", encoding="utf-8") as noise_stream:
            trajectory_writer = csv.DictWriter(trajectory_stream, fieldnames=TRAJECTORY_FIELDS)
            noise_writer = csv.DictWriter(noise_stream, fieldnames=NOISE_FIELDS)
            trajectory_writer.writeheader()
            noise_writer.writeheader()
            for seed in args.seeds:
                a, b, batches = initialized_factors_and_batches(
                    arrays, train, args.rank, args.steps, args.batch_size, seed,
                )
                initial_a, initial_b = arrays.numpy(a).copy(), arrays.numpy(b).copy()
                for step in range(args.steps + 1):
                    if step in checkpoint_set:
                        train_loss, train_gradient = objective.metrics(train, a, b)
                        test_loss, test_gradient = objective.metrics(test, a, b)
                        gap = tuple(g_test - g_train for g_test, g_train in zip(test_gradient, train_gradient))
                        row = {
                            "seed": seed, "step": step, "lr": lr, "train_loss": train_loss,
                            "test_loss": test_loss, "factor_sq": factor_scale(arrays, a, b),
                            "train_grad_sq": squared_pair(train_gradient),
                            "test_grad_sq": squared_pair(test_gradient),
                            "gradient_gap_sq": squared_pair(gap),
                        }
                        if any(not isfinite(value) or value < 0 for value in row.values()):
                            raise FloatingPointError("Nonfinite or negative checkpoint metric. Try a smaller --lr.")
                        trajectory_writer.writerow(row)
                        trajectory_stream.flush()
                        print(f"Seed {seed}, step {step}/{args.steps}: train loss={train_loss:.6g}, "
                              f"||A_t||_2^2 + ||B_t||_2^2={row['factor_sq']:.6g}, "
                              f"||grad J_S(B_t,A_t)||_F^2={row['train_grad_sq']:.6g}", flush=True)
                        for diagnostic_batch in args.diagnostic_batches:
                            noise = noise_diagnostic(objective, train, a, b, diagnostic_batch,
                                                     args.diagnostic_pairs, seed, step)
                            noise_writer.writerow(noise)
                            noise_stream.flush()
                            print(f"  Diagnostic batch size {diagnostic_batch}: weight variance={noise['weight_noise_sq']:.5g}, "
                                  f"factor variance={noise['factor_noise_sq']:.5g}", flush=True)
                    if step < args.steps:
                        a, b = objective.step(take(train, batches[step]), a, b, lr)
                np.savez_compressed(directory / f"seed_{seed}_factors.npz", initial_A=initial_a,
                                    initial_B=initial_b, final_A=arrays.numpy(a), final_B=arrays.numpy(b))
    except (Exception, KeyboardInterrupt):
        manifest["status"] = "failed"
        write_atomic(manifest_path, json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        raise
    manifest["status"] = "completed"
    write_atomic(manifest_path, json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    write_atomic(HERE / "ViT_Noise.latest.txt", str(directory) + "\n")
    print(f"Completed. Saved trajectory, noise diagnostics, and factors in {directory}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    previous_cache = HERE.parent / "ViT_EXP" / "data" / "vit_cifar100.npz"
    parser.add_argument("--cache", type=Path,
                        default=previous_cache if previous_cache.is_file() else HERE / "data" / "vit_cifar100.npz")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 14, 15, 16, 17])
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--checkpoints", type=int, default=20,
                        help="Number of intervals; both endpoints are evaluated (default 21 checkpoints)")
    parser.add_argument("--lr", type=float, help="Use this constant learning rate and skip pilot selection")
    parser.add_argument("--lr-candidates", type=float, nargs="+", default=[0.003, 0.01, 0.03])
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64, help="Training minibatch size")
    parser.add_argument("--diagnostic-batches", type=int, nargs="+", default=[16, 64, 256])
    parser.add_argument("--diagnostic-pairs", type=int, default=32,
                        help="Independent pairs per diagnostic batch size at each checkpoint")
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--head-epochs", type=int, default=1)
    parser.add_argument("--pilot-seed", type=int, default=2027)
    parser.add_argument("--pilot-steps", type=int, default=200)
    args = parser.parse_args()
    if args.device not in ("auto", "cpu", "cuda") and not (
        args.device.startswith("cuda:") and args.device[5:].isdigit()
    ):
        parser.error("--device must be auto, cpu, cuda, or cuda:N")
    for name in ("steps", "checkpoints", "rank", "batch_size", "diagnostic_pairs",
                 "eval_batch_size", "head_epochs", "pilot_steps"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.diagnostic_pairs < 2:
        parser.error("--diagnostic-pairs must be at least 2 to estimate Monte Carlo uncertainty")
    for name, minimum in (("seeds", 0), ("diagnostic_batches", 1)):
        values = getattr(args, name)
        if any(value < minimum for value in values) or len(set(values)) != len(values):
            parser.error(f"--{name.replace('_', '-')} requires distinct integers >= {minimum}")
    if args.pilot_seed < 0:
        parser.error("--pilot-seed must be nonnegative")
    rates = args.lr_candidates + ([args.lr] if args.lr is not None else [])
    if not all(isfinite(eta) and eta > 0 for eta in rates):
        parser.error("Learning rates must be positive and finite")
    args.diagnostic_batches.sort()
    return args


def main():
    args = parse_args()
    try:
        run(args)
    except (OSError, ValueError, KeyError, FloatingPointError, RuntimeError, ImportError) as error:
        raise SystemExit(f"ViT_Run.py: {error}") from None


if __name__ == "__main__":
    main()

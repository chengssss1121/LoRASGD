#!/usr/bin/env python3
"""Train four paired LoRA-SGD schedules on frozen CIFAR-10 features.

Use a .npz or .pt cache with train_features, train_labels, test_features,
and test_labels. If the cache is absent, extract features using torchvision.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import tempfile
from urllib.parse import urlparse

import numpy as np


HERE = Path(__file__).resolve().parent
SCHEDULES = (
    ("constant_small", "Constant 0.01697", 0.01697),
    ("constant_medium", "Constant 0.03000", 0.03000),
    ("constant_large", "Constant 0.05317", 0.05317),
    ("decay", "Decaying (starts at 0.05317)", 0.05317),
)
CHECKPOINT_FIELDS = (
    "step", "train_loss", "factor_spectral_sq", "train_grad_sq",
    "test_loss", "test_grad_sq", "gradient_gap_sq",
)


class Arrays:
    """Use NumPy on CPU or PyTorch on CUDA with the same update rule."""

    def __init__(self, device, dtype):
        self.device = device
        self.dtype = dtype
        self.xp = np
        if device == "cpu":
            return
        try:
            import torch
        except ImportError as exc:
            if device == "auto":
                self.device = "cpu"
                return
            raise ValueError("CUDA training requires PyTorch with CUDA support.") from exc
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device != "cpu":
            if not torch.cuda.is_available():
                raise ValueError("CUDA is unavailable. Use --device cpu or --device auto.")
            self.xp = torch

    def float(self, value):
        if self.xp is np:
            return np.asarray(value, dtype=self.dtype)
        return self.xp.as_tensor(value, dtype=getattr(self.xp, self.dtype), device=self.device)

    def index(self, value):
        if self.xp is np:
            return np.asarray(value, dtype=np.int64)
        return self.xp.as_tensor(value, dtype=self.xp.long, device=self.device)

    def residual_and_loss(self, logits, labels, need_loss):
        if self.xp is np:
            rows = np.arange(len(labels))
            shifted = logits - logits.max(axis=1, keepdims=True)
            residual = np.exp(shifted)
            total = residual.sum(axis=1, keepdims=True)
            loss = (np.log(total[:, 0]) - shifted[rows, labels]).sum().item() if need_loss else 0.0
            residual /= total
        else:
            rows = self.xp.arange(len(labels), device=self.device)
            if need_loss:
                log_probs = self.xp.log_softmax(logits, dim=1)
                loss = -log_probs[rows, labels].sum().item()
                residual = log_probs.exp()
            else:
                residual = self.xp.softmax(logits, dim=1)
                loss = 0.0
        residual[rows, labels] -= 1
        return residual, loss


def unpack_features(data):
    if isinstance(data, (list, tuple)) and len(data) == 4:
        return tuple(np.asarray(part) for part in data)

    def read(mapping, *keys):
        for key in keys:
            if key in mapping:
                return mapping[key]
        raise ValueError(f"Feature cache has none of these keys: {', '.join(keys)}")

    def read_split(split, short):
        if split in data:
            entry = data[split]
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                return np.asarray(entry[0]), np.asarray(entry[1])
            x = read(entry, "features", "x", "X")
            y = read(entry, "labels", "y", "targets")
        else:
            x = read(data, f"{split}_features", f"{split}_x", f"{short}_x", f"x_{split}", f"X_{split}")
            y = read(data, f"{split}_labels", f"{split}_y", f"{short}_y", f"y_{split}", f"{split}_targets")
        return np.asarray(x), np.asarray(y)

    train_x, train_y = read_split("train", "tr")
    test_x, test_y = read_split("test", "te")
    return train_x, train_y, test_x, test_y


def extract_features(path, data_dir, download, device):
    try:
        import torch
        from torchvision import datasets, models
    except ImportError as exc:
        raise ValueError("Feature extraction needs PyTorch and torchvision. Pass --features to reuse a cache.") from exc

    weights = models.ResNet18_Weights.IMAGENET1K_V1
    transform = weights.transforms()
    try:
        datasets_by_split = {
            split: datasets.CIFAR10(data_dir, train=(split == "train"), download=download, transform=transform)
            for split in ("train", "test")
        }
    except RuntimeError as exc:
        if not download:
            raise ValueError(f"CIFAR-10 is unavailable in {data_dir}. Use --download or pass --features.") from exc
        raise
    weights_file = Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(weights.url).path).name
    if not weights_file.is_file() and not download:
        raise ValueError("Pretrained ResNet-18 weights are missing. Use --download or pass --features.")
    model = models.resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.to(device).eval()
    data = {}
    print(f"Extracting CIFAR-10 ResNet-18 features to {path}", flush=True)
    with torch.inference_mode():
        for split, dataset in datasets_by_split.items():
            feature_parts, label_parts = [], []
            loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
            for images, labels in loader:
                feature_parts.append(model(images.to(device)).cpu().numpy())
                label_parts.append(labels.numpy())
            data[f"{split}_features"] = np.concatenate(feature_parts)
            data[f"{split}_labels"] = np.concatenate(label_parts)

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            if path.suffix.lower() == ".npz":
                np.savez(handle, **data)
            else:
                torch.save({key: torch.from_numpy(value) for key, value in data.items()}, handle)
            handle.flush()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return unpack_features(data)


def load_features(path, data_dir, download, device):
    suffix = path.suffix.lower()
    if suffix not in {".npz", ".pt", ".pth"}:
        raise ValueError("--features must point to a .npz, .pt, or .pth file.")
    if not path.exists():
        return extract_features(path, data_dir, download, device)
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            return unpack_features(data)
    try:
        import torch
    except ImportError as exc:
        raise ValueError("Reading a .pt/.pth feature cache requires PyTorch.") from exc
    return unpack_features(torch.load(path, map_location="cpu", weights_only=True))


def validate_features(parts, arrays):
    train_x, train_y, test_x, test_y = parts
    for split, x, y in (("train", train_x, train_y), ("test", test_x, test_y)):
        if x.ndim != 2 or min(x.shape) == 0 or len(x) != y.size or x.shape[1] != train_x.shape[1]:
            raise ValueError(f"{split}: expected nonempty features [examples, dimension] and one label per row.")
        if not np.issubdtype(x.dtype, np.number) or np.iscomplexobj(x) or not np.isfinite(x).all():
            raise ValueError(f"{split}: features must have finite real entries.")
        if not np.issubdtype(y.dtype, np.integer):
            raise ValueError(f"{split}: labels must be integer class labels.")
    classes = np.unique(train_y)
    if len(classes) < 2 or not np.isin(test_y, classes).all():
        raise ValueError("Need at least two training classes and no unseen test classes.")
    train = arrays.float(train_x), arrays.index(np.searchsorted(classes, train_y.reshape(-1)))
    test = arrays.float(test_x), arrays.index(np.searchsorted(classes, test_y.reshape(-1)))
    return train, test, len(classes)


def squared_norm(value):
    return (value * value).sum().item()


def full_objective(arrays, data, weight, chunk_size):
    x, y = data
    total_loss = 0.0
    gradient = weight * 0.0
    for start in range(0, len(y), chunk_size):
        x_chunk = x[start:start + chunk_size]
        y_chunk = y[start:start + chunk_size]
        residual, loss = arrays.residual_and_loss(x_chunk @ weight.T, y_chunk, need_loss=True)
        total_loss += loss
        gradient += residual.T @ x_chunk
    return total_loss / len(y), gradient / len(y)


def factor_gradient(gradient, a, b):
    return b.T @ gradient, gradient @ a.T


def checkpoint(arrays, train, test, base, a, b, chunk_size):
    weight = base + b @ a
    train_loss, train_grad = full_objective(arrays, train, weight, chunk_size)
    test_loss, test_grad = full_objective(arrays, test, weight, chunk_size)
    train_a, train_b = factor_gradient(train_grad, a, b)
    test_a, test_b = factor_gradient(test_grad, a, b)
    eigvalsh = arrays.xp.linalg.eigvalsh
    factor_size = eigvalsh(a @ a.T)[-1].item() + eigvalsh(b.T @ b)[-1].item()
    return (
        train_loss, max(0.0, factor_size), squared_norm(train_a) + squared_norm(train_b),
        test_loss, squared_norm(test_a) + squared_norm(test_b),
        squared_norm(test_a - train_a) + squared_norm(test_b - train_b),
    )


def step_size(schedule, initial, t, steps):
    if schedule == "decay":
        return initial / math.sqrt(1 + (t - 1) / (steps / 10))
    return initial


def train_schedule(args, arrays, train, test, base, initial_a, initial_b, batches, schedule, folder):
    schedule_id, _, initial_step_size = schedule
    a = initial_a + 0.0
    b = initial_b + 0.0
    with (folder / "checkpoints.csv").open("x", newline="") as checkpoint_file, \
            (folder / "updates.csv").open("x", newline="") as update_file:
        checkpoints = csv.writer(checkpoint_file)
        updates = csv.writer(update_file)
        checkpoints.writerow(CHECKPOINT_FIELDS)
        updates.writerow(("step", "step_size"))
        for step in range(args.steps + 1):
            if step % args.eval_every == 0 or step == args.steps:
                values = checkpoint(arrays, train, test, base, a, b, args.eval_batch_size)
                if any(not math.isfinite(value) for value in values):
                    raise ValueError(f"Non-finite checkpoint at update {step} in {folder}.")
                checkpoints.writerow((step, *values))
            if step == args.steps:
                break
            batch = batches[step]
            x, y = train[0][batch], train[1][batch]
            eta = step_size(schedule_id, initial_step_size, step + 1, args.steps)
            residual, _ = arrays.residual_and_loss(x @ (base + b @ a).T, y, need_loss=False)
            grad_a, grad_b = factor_gradient((residual.T @ x) / len(y), a, b)
            a -= eta * grad_a
            b -= eta * grad_b
            updates.writerow((step + 1, eta))


def atomic_text(path, content):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        tmp = Path(handle.name)
        try:
            handle.write(content)
            handle.flush()
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)


def experiment(args, arrays, train, test, n_classes, run_dir):
    input_dim = train[0].shape[1]
    init_bound = 1 / math.sqrt(input_dim)
    settings = {
        "schema_version": 1,
        "experiment": "EXP1",
        "status": "completed",
        "steps": args.steps,
        "eval_every": args.eval_every,
        "rank": args.rank,
        "batch_size": args.batch_size,
        "seeds": args.seeds,
        "schedules": [{"id": key, "label": label} for key, label, _ in SCHEDULES],
        "runs": [],
        "features": str(args.features),
        "train_examples": len(train[1]),
        "test_examples": len(test[1]),
        "device": arrays.device,
        "dtype": arrays.dtype,
        "sampling": "iid_with_replacement",
    }
    for seed in args.seeds:
        init_rng, batch_rng = [np.random.default_rng(stream) for stream in np.random.SeedSequence(seed).spawn(2)]
        base = arrays.float(init_rng.uniform(-init_bound, init_bound, (n_classes, input_dim)))
        a_init = arrays.float(init_rng.normal(0.0, 0.01, (args.rank, input_dim)))
        b_init = arrays.float(np.zeros((n_classes, args.rank)))
        batches = arrays.index(batch_rng.integers(len(train[1]), size=(args.steps, args.batch_size), dtype=np.int64))
        for schedule in SCHEDULES:
            schedule_id, label, _ = schedule
            folder = run_dir / f"seed_{seed}" / schedule_id
            folder.mkdir(parents=True)
            print(f"Training seed {seed}, {label} ({arrays.device}, {args.steps} updates)", flush=True)
            train_schedule(args, arrays, train, test, base, a_init, b_init, batches, schedule, folder)
            settings["runs"].append({"seed": seed, "schedule": schedule_id, "folder": str(folder.relative_to(run_dir))})
    atomic_text(run_dir / "experiment.json", json.dumps(settings, indent=2) + "\n")
    atomic_text(HERE / "EXP1.latest.txt", str(run_dir) + "\n")


def options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=HERE / "data" / "cifar10_resnet18_features.npz",
                        help="Path to a combined .npz or .pt feature cache")
    parser.add_argument("--data-dir", type=Path, default=HERE / "data", help="CIFAR-10 dataset root for feature extraction")
    parser.add_argument("--download", action="store_true", help="Download CIFAR-10 and ResNet-18 weights if needed")
    parser.add_argument("--run-dir", type=Path, help="New or empty results directory (default: create one under EXP1/results)")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=19600, help="Number of updates per schedule and seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 14, 15, 16, 17])
    parser.add_argument("--eval-every", type=int, default=98, help="Updates between exact full-gradient evaluations")
    parser.add_argument("--eval-batch-size", type=int, default=4096, help="Chunk size for full-dataset evaluations")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a specific CUDA device such as cuda:0")
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--plot", action="store_true", help="Generate the plots after training")
    args = parser.parse_args()
    if min(args.rank, args.batch_size, args.steps, args.eval_every, args.eval_batch_size) < 1:
        parser.error("Rank, batch sizes, update count, and evaluation interval must be positive.")
    if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be distinct nonnegative integers.")
    if args.device not in {"auto", "cpu", "cuda"} and not (
        args.device.startswith("cuda:") and args.device[5:].isdigit()
    ):
        parser.error("--device must be auto, cpu, cuda, or cuda:<device index>.")
    return args


def main():
    args = options()
    try:
        arrays = Arrays(args.device, args.dtype)
        args.features = args.features.expanduser().resolve()
        parts = load_features(args.features, args.data_dir.expanduser(), args.download, arrays.device)
        train, test, n_classes = validate_features(parts, arrays)
        if args.run_dir is None:
            results_root = HERE / "results"
            results_root.mkdir(exist_ok=True)
            run_dir = Path(tempfile.mkdtemp(prefix="EXP1_", dir=results_root))
        else:
            run_dir = args.run_dir.expanduser().resolve()
            run_dir.mkdir(parents=True, exist_ok=True)
            if any(run_dir.iterdir()):
                raise ValueError(f"Choose a new or empty --run-dir. {run_dir} already contains files.")
        lock = run_dir / ".running"
        lock.touch(exist_ok=False)
        print(f"Results directory: {run_dir}", flush=True)
        try:
            experiment(args, arrays, train, test, n_classes, run_dir)
        finally:
            lock.unlink(missing_ok=True)
        if args.plot:
            from EXP1_Plot import plot_results

            plot_results(run_dir)
        print(f"Completed EXP1: {run_dir}", flush=True)
    except (OSError, RuntimeError, ValueError, ImportError) as exc:
        raise SystemExit(f"EXP1: {exc}") from exc


if __name__ == "__main__":
    main()

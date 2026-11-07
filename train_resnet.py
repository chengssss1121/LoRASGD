#!/usr/bin/env python3
"""Simultaneous LoRA-SGD on CIFAR-10 with a frozen-base ResNet-18.

Defaults adapt the ResNet-18 experiment of Mu and Klabjan: rank 20,
batch size 512, 250 epochs, and no rank-dependent adapter scaling. The
classifier construction, Gaussian standard deviation, preprocessing, and IID
sampling convention are explicit reconstruction choices, not unspecified claims
of exact reproduction. BatchNorm statistics remain frozen via model.eval().
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("results/resnet_r20_lr004_seed13"))
    parser.add_argument("--rank", type=int, default=20)
    parser.add_argument("--init-std", type=float, default=0.01,
                        help="Gaussian A standard deviation; B is initialized to zero.")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--sampling", choices=("iid", "epoch"), default="iid",
                        help="IID batches with replacement or conventional shuffled epochs.")
    parser.add_argument("--weights", choices=("none", "imagenet"), default="none")
    parser.add_argument("--cifar-stem", action="store_true",
                        help="Explicit alternative: 3x3 stride-1 stem and no max-pooling.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1,
                        help="Exact norms at every k updates; initial and terminal states are always included.")
    parser.add_argument("--eval-every", type=int, default=1, help="Evaluate every k epochs; 0 disables evaluation.")
    parser.add_argument("--eval-limit", type=int, default=0, help="0 evaluates the complete selected datasets.")
    parser.add_argument("--max-train", type=int, default=0, help="0 uses the entire training set.")
    parser.add_argument("--max-test", type=int, default=0, help="0 uses the entire test set.")
    parser.add_argument("--max-steps", type=int, default=0, help="Optional early stop for a smoke run; 0 disables it.")
    parser.add_argument("--download", action="store_true", help="Allow downloading CIFAR-10.")
    parser.add_argument("--fake-data", action="store_true",
                        help="Offline execution check with synthetic images; not a CIFAR-10 experiment.")
    args = parser.parse_args()
    for name in ("rank", "epochs", "batch_size", "log_every"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("num_workers", "eval_every", "eval_limit", "max_train", "max_test", "max_steps"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative")
    if not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.init_std) or args.init_std <= 0:
        parser.error("--lr and --init-std must be finite and positive")
    return args


def main() -> None:
    args = parse_args()
    try:
        import numpy as np
        import torch
        import torchvision
        from torch import nn
        from torch.nn import functional as F
        from torch.utils.data import DataLoader, RandomSampler, Subset
        from torchvision import datasets, models, transforms
        from lora_growth.metrics import pair_metrics
        from lora_growth.torch_layers import LoRAConv2d, LoRALinear, lora_pairs
    except ImportError as exc:
        raise SystemExit("ResNet training requires torch, torchvision, and numpy. "
                         "Install requirements-vision.txt; see README.md.") from exc

    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit("--output must name a new or empty directory; existing results will not be overwritten.")
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)

    # Fixed preprocessing preserves a single finite training objective.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
    ])
    if args.fake_data:
        train_dataset = datasets.FakeData(size=args.max_train or 64, image_size=(3, 32, 32),
                                          num_classes=10, transform=transform, random_offset=args.seed)
        test_dataset = datasets.FakeData(size=args.max_test or 32, image_size=(3, 32, 32),
                                         num_classes=10, transform=transform, random_offset=args.seed + 100000)
    else:
        train_dataset = datasets.CIFAR10(str(args.data_root), train=True, download=args.download, transform=transform)
        test_dataset = datasets.CIFAR10(str(args.data_root), train=False, download=args.download, transform=transform)
    subset_rng = np.random.default_rng(args.seed)

    def select(dataset, maximum):
        indices = np.arange(len(dataset), dtype=np.int64)
        if maximum and maximum < len(dataset):
            indices = np.sort(subset_rng.choice(indices, maximum, replace=False))
        return Subset(dataset, indices.tolist()), indices

    train_dataset, train_indices = select(train_dataset, args.max_train)
    test_dataset, test_indices = select(test_dataset, args.max_test)
    np.savez_compressed(args.output / "dataset_indices.npz", train_indices=train_indices, test_indices=test_indices)
    steps_per_epoch = math.ceil(len(train_dataset) / args.batch_size)
    generator = torch.Generator().manual_seed(args.seed + 1)
    if args.sampling == "iid":
        sampler = RandomSampler(train_dataset, replacement=True,
                                num_samples=steps_per_epoch * args.batch_size, generator=generator)
    else:
        sampler = RandomSampler(train_dataset, replacement=False, generator=generator)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")

    weights = models.ResNet18_Weights.IMAGENET1K_V1 if args.weights == "imagenet" else None
    model = models.resnet18(weights=weights)
    if args.cifar_stem:
        model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    # The new 10-class base classifier is randomly initialized and then frozen.
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.requires_grad_(False)

    def inject(module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, args.rank, args.init_std))
            elif isinstance(child, nn.Conv2d):
                setattr(module, name, LoRAConv2d(child, args.rank, args.init_std))
            else:
                inject(child)

    inject(model)
    model.to(device)
    # eval() freezes BatchNorm running statistics; it does not disable autograd.
    model.eval()
    pairs = list(lora_pairs(model))
    if not pairs:
        raise RuntimeError("No LoRA factor pairs were created")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(trainable, lr=args.lr, momentum=0.0, weight_decay=0.0)
    manifest = {
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
        "torchvision": torchvision.__version__, "device": str(device),
        "optimizer": {"name": "SGD", "lr": args.lr, "momentum": 0.0, "weight_decay": 0.0,
                      "gradient_clipping": False, "normalization": False},
        "adapter_scaling": 1.0, "initialization": {"A": "Gaussian", "std": args.init_std, "B": "zeros"},
        "batchnorm": "eval mode: running statistics and affine parameters frozen",
        "data_preprocessing": "32x32 ToTensor plus fixed CIFAR-10 mean/std; no random augmentation",
        "sampling": args.sampling, "steps_per_epoch": steps_per_epoch,
        "dataset": "synthetic torchvision FakeData (execution check only)" if args.fake_data else "CIFAR-10",
        "train_size": len(train_dataset), "test_size": len(test_dataset),
        "trainable_parameters": sum(p.numel() for p in trainable),
        "pairs": [{"name": name, "A_shape": list(A.shape), "B_shape": list(B.shape)} for name, A, B in pairs],
        "metrics": "float64 CPU SVD/QR on stored factor matrices, not power iteration",
        "state_convention": "step=0 is initialization; step=t is after t joint optimizer updates",
        "loss_convention": "batch_loss at step=t was computed before update t",
        "reproduction_note": "Framework adaptation. Paper does not fully specify vision A std, backbone stem, or preprocessing.",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.output / "config.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    metric_keys = list(pair_metrics(np.zeros((1, 1)), np.zeros((1, 1))))
    fields = ["step", "epoch", "pair", "lr", "batch_loss", "batch_factor_gradient_sq"] + metric_keys + ["sampled_running_max_spectral_sum"]
    aggregate_fields = ["step", "epoch", "sum_pair_spectral_sum", "max_pair_spectral_sum", "sum_pair_frobenius_sum"]
    maxima = {name: 0.0 for name, _, _ in pairs}
    last_logged = -1
    start = time.perf_counter()
    step, epoch_value = 0, 0.0
    last_loss, last_gradient_sq = "", ""

    with (args.output / "pair_metrics.csv").open("x", newline="", encoding="utf-8") as stream, \
         (args.output / "aggregate.csv").open("x", newline="", encoding="utf-8") as aggregate_stream, \
         (args.output / "metrics.csv").open("x", newline="", encoding="utf-8") as scalar_stream, \
         (args.output / "evaluation.csv").open("x", newline="", encoding="utf-8") as evaluation_stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        aggregate_writer = csv.DictWriter(aggregate_stream, fieldnames=aggregate_fields)
        scalar_writer = csv.DictWriter(scalar_stream, fieldnames=["step", "epoch", "lr", "batch_loss", "batch_factor_gradient_sq"])
        evaluation_writer = csv.DictWriter(evaluation_stream, fieldnames=["step", "epoch", "split", "mean_loss", "accuracy", "n"])
        writer.writeheader()
        aggregate_writer.writeheader()
        scalar_writer.writeheader()
        evaluation_writer.writeheader()

        def log_state():
            nonlocal last_logged
            sums, frobenius_sums = [], []
            for name, A, B in pairs:
                metrics = pair_metrics(A.detach().cpu().double().numpy(), B.detach().cpu().double().numpy())
                maxima[name] = max(maxima[name], metrics["spectral_sum"])
                writer.writerow({"step": step, "epoch": epoch_value, "pair": name, "lr": args.lr,
                                 "batch_loss": last_loss, "batch_factor_gradient_sq": last_gradient_sq,
                                 **metrics, "sampled_running_max_spectral_sum": maxima[name]})
                sums.append(metrics["spectral_sum"])
                frobenius_sums.append(metrics["frobenius_sum"])
            aggregate_writer.writerow({"step": step, "epoch": epoch_value, "sum_pair_spectral_sum": sum(sums),
                                       "max_pair_spectral_sum": max(sums), "sum_pair_frobenius_sum": sum(frobenius_sums)})
            stream.flush()
            aggregate_stream.flush()
            last_logged = step

        def evaluate(dataset, split):
            if args.eval_limit and len(dataset) > args.eval_limit:
                dataset = Subset(dataset, list(range(args.eval_limit)))
            evaluation_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                           num_workers=args.num_workers, pin_memory=device.type == "cuda")
            loss_sum, correct, count = 0.0, 0, 0
            with torch.inference_mode():
                for images, labels in evaluation_loader:
                    images, labels = images.to(device), labels.to(device)
                    logits = model(images)
                    loss_sum += float(F.cross_entropy(logits, labels, reduction="sum").item())
                    correct += int((logits.argmax(1) == labels).sum().item())
                    count += len(labels)
            evaluation_writer.writerow({"step": step, "epoch": epoch_value, "split": split,
                                        "mean_loss": loss_sum / count, "accuracy": correct / count, "n": count})
            evaluation_stream.flush()

        log_state()
        scalar_writer.writerow({"step": 0, "epoch": 0.0, "lr": args.lr,
                                "batch_loss": "", "batch_factor_gradient_sq": ""})
        try:
            stop = False
            for epoch in range(args.epochs):
                for batch_index, (images, labels) in enumerate(loader):
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(model(images), labels)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("Non-finite loss; reduce --lr and inspect the last finite logs")
                    loss.backward()
                    gradient_sq = sum(float(p.grad.detach().to(device="cpu", dtype=torch.float64).square().sum().item())
                                      for p in trainable if p.grad is not None)
                    if not math.isfinite(gradient_sq):
                        raise FloatingPointError("Non-finite factor gradients; reduce --lr")
                    # Both factors use gradients from the same pre-update state.
                    optimizer.step()
                    step += 1
                    epoch_value = epoch + (batch_index + 1) / steps_per_epoch
                    last_loss, last_gradient_sq = float(loss.item()), gradient_sq
                    if not all(bool(torch.isfinite(p).all()) for p in trainable):
                        raise FloatingPointError("Non-finite updated factors; reduce --lr")
                    scalar_writer.writerow({"step": step, "epoch": epoch_value, "lr": args.lr,
                                            "batch_loss": last_loss, "batch_factor_gradient_sq": last_gradient_sq})
                    scalar_stream.flush()
                    if step % args.log_every == 0:
                        log_state()
                    if args.max_steps and step >= args.max_steps:
                        stop = True
                        break
                if args.eval_every and ((epoch + 1) % args.eval_every == 0 or stop or epoch + 1 == args.epochs):
                    evaluate(train_dataset, "train")
                    evaluate(test_dataset, "test")
                print(f"epoch={epoch_value:.4f} step={step} last_batch_loss={last_loss:.6g}", flush=True)
                if stop:
                    break
            if last_logged != step:
                log_state()
        except Exception as exc:
            failure = {"status": "failed", "step": step, "epoch": epoch_value,
                       "exception": type(exc).__name__, "message": str(exc)}
            (args.output / "failure.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
            raise

    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                "arguments": manifest["arguments"], "torch_rng_state": torch.get_rng_state(),
                "sampling_rng_state": generator.get_state()}, args.output / "final_checkpoint.pt")
    summary = {"status": "completed", "steps": step, "epochs_completed": epoch_value,
               "elapsed_seconds": time.perf_counter() - start, "n_pairs": len(pairs),
               "maximum_scope": "every optimizer state" if args.log_every == 1 else "logged checkpoints only",
               "per_pair_max_spectral_sum": maxima,
               "note": "Finite experiments diagnose factor growth; they do not prove a global bound or a convergence exponent."}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved per-pair metrics, aggregates, evaluations, and checkpoint in {args.output}")


if __name__ == "__main__":
    main()

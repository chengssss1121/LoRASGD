#!/usr/bin/env python3
"""Cache deterministic 512-dimensional ResNet-18 features for CIFAR-10.

The paper identifies the feature dimension and backbone, but does not specify
the exact feature checkpoint or image preprocessing. This implementation makes
both choices explicit; it is a documented adaptation, not a bitwise reproduction.
"""
from __future__ import annotations

import argparse
import json
import platform
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("data/cifar10_resnet18_features.npz"))
    parser.add_argument("--weights", choices=("imagenet", "none"), default="imagenet",
                        help="Frozen ImageNet checkpoint or a seeded random backbone.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-train", type=int, default=0, help="0 uses all 50000 training images.")
    parser.add_argument("--max-test", type=int, default=0, help="0 uses all 10000 test images.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--l2-normalize", action="store_true", help="Normalize each cached feature to unit norm.")
    parser.add_argument("--download", action="store_true", help="Allow torchvision to download CIFAR-10.")
    parser.add_argument("--fake-data", action="store_true",
                        help="Offline pipeline check with synthetic images; not a CIFAR-10 dataset.")
    args = parser.parse_args()
    for key in ("batch_size", "image_size"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    for key in ("max_train", "max_test", "num_workers"):
        if getattr(args, key) < 0:
            parser.error(f"--{key.replace('_', '-')} must be nonnegative")
    if args.output.suffix != ".npz":
        parser.error("--output must end in .npz")
    return args


def main() -> None:
    args = parse_args()
    try:
        import numpy as np
        import torch
        import torchvision
        from torch import nn
        from torch.utils.data import DataLoader, Subset
        from torchvision import datasets, models, transforms
    except ImportError as exc:
        raise SystemExit("Feature extraction requires torch, torchvision, and numpy. "
                         "Install requirements-vision.txt; see README.md.") from exc

    metadata_path = args.output.with_suffix(".json")
    if args.output.exists() or metadata_path.exists():
        raise SystemExit("The output or its metadata already exists; choose a new --output path.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    resize_size = max(args.image_size, round(args.image_size * 256 / 224))
    transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if args.weights == "imagenet" else None
    # ImageNet weights can be downloaded by torchvision when not already cached.
    model = models.resnet18(weights=weights)
    model.fc = nn.Identity()
    model.requires_grad_(False)
    model.eval().to(device)
    index_rng = np.random.default_rng(args.seed)

    def extract(train: bool, maximum: int) -> tuple[object, object, object]:
        if args.fake_data:
            dataset = datasets.FakeData(size=maximum or (64 if train else 32), image_size=(3, 32, 32),
                                        num_classes=10, transform=transform,
                                        random_offset=args.seed + (0 if train else 100000))
        else:
            dataset = datasets.CIFAR10(str(args.data_root), train=train, download=args.download,
                                       transform=transform)
        indices = np.arange(len(dataset), dtype=np.int64)
        if maximum and maximum < len(indices):
            indices = np.sort(index_rng.choice(indices, maximum, replace=False))
        selected = Subset(dataset, indices.tolist())
        loader = DataLoader(selected, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda")
        all_features, all_labels = [], []
        with torch.inference_mode():
            for batch_index, (images, labels) in enumerate(loader):
                features = model(images.to(device)).detach().float().cpu().numpy()
                if args.l2_normalize:
                    norms = np.linalg.norm(features, axis=1, keepdims=True)
                    features = features / np.maximum(norms, np.finfo(features.dtype).tiny)
                all_features.append(features)
                all_labels.append(labels.numpy().astype(np.int64, copy=False))
                if batch_index % 20 == 0:
                    print(f"{'train' if train else 'test'} batch {batch_index + 1}/{len(loader)}", flush=True)
        return np.concatenate(all_features), np.concatenate(all_labels), indices

    x_train, y_train, train_indices = extract(True, args.max_train)
    x_test, y_test, test_indices = extract(False, args.max_test)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, x_train=x_train, y_train=y_train,
                        x_test=x_test, y_test=y_test,
                        train_indices=train_indices, test_indices=test_indices)
    metadata = {
        "description": ("Synthetic-image ResNet-18 execution check" if args.fake_data
                        else "Frozen ResNet-18 CIFAR-10 features; documented framework adaptation"),
        "dataset": "synthetic torchvision FakeData (execution check only)" if args.fake_data else "CIFAR-10",
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "numpy": np.__version__, "python": platform.python_version(), "device": str(device),
        "backbone": "torchvision.models.resnet18", "feature_dimension": 512,
        "weights": "IMAGENET1K_V1" if weights is not None else "seeded random weights",
        "preprocessing": {"resize_short_edge": resize_size, "center_crop": args.image_size,
                          "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
                          "random_augmentation": False, "feature_l2_normalization": args.l2_normalize},
        "train_shape": list(x_train.shape), "test_shape": list(x_test.shape),
        "reproduction_note": "The paper does not fully specify its feature checkpoint or preprocessing.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output} and {metadata_path}")


if __name__ == "__main__":
    main()

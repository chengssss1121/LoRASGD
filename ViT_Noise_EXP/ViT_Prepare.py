#!/usr/bin/env python3
"""Cache the inputs needed for exact last-block ViT LoRA adaptation."""

import argparse
from pathlib import Path
import tempfile

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = "google/vit-base-patch16-224-in21k"
DEFAULT_DATA = HERE / "data"
EXISTING_CACHE = HERE.parent / "ViT_EXP" / "data" / "vit_cifar100.npz"
DEFAULT_CACHE = EXISTING_CACHE if EXISTING_CACHE.is_file() else DEFAULT_DATA / "vit_cifar100.npz"
PILOT_SIZE = 5_000
SPLIT_SEED = 13


def split_indices(size):
    if size <= PILOT_SIZE:
        raise ValueError("The training set must contain more than 5,000 images.")
    order = np.random.default_rng(SPLIT_SEED).permutation(size)
    return {"pilot": np.sort(order[:PILOT_SIZE]), "train": np.sort(order[PILOT_SIZE:])}


def final_mlp_output(model):
    try:
        if hasattr(model, "layers"):
            return model.layers[-1].mlp.fc2
        return model.encoder.layer[-1].output.dense
    except (AttributeError, IndexError) as exc:
        raise ValueError("Could not find the final MLP output projection in this ViT checkpoint.") from exc


class TokenCapture:
    def __init__(self, model):
        self.model = model
        self.projection = final_mlp_output(model)
        self.norm = model.layernorm
        self.projection_input = None
        self.residual = None
        self.checked_update = False

    def __enter__(self):
        self.handles = (
            self.projection.register_forward_pre_hook(self.capture_projection_input),
            self.norm.register_forward_pre_hook(self.capture_residual),
        )
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()

    def capture_projection_input(self, _module, inputs):
        self.projection_input = inputs[0][:, 0, :].detach().clone()

    def capture_residual(self, _module, inputs):
        self.residual = inputs[0][:, 0, :].detach().clone()

    def _forward(self, pixels):
        self.projection_input = None
        self.residual = None
        return self.model(pixel_values=pixels, return_dict=True).last_hidden_state[:, 0, :]

    def extract(self, pixels, torch):
        reference = self._forward(pixels)
        u, r0 = self.projection_input, self.residual
        if u is None or r0 is None:
            raise ValueError("ViT did not use the expected output projection and final LayerNorm.")
        self._check(torch, self.norm.forward(r0), reference)

        if not self.checked_update:
            self._check_rank_one_update(pixels, torch, u, r0)
            self.checked_update = True
        return u.cpu().numpy(), r0.cpu().numpy()

    def _check_rank_one_update(self, pixels, torch, u, r0):
        # Perturb the full projection once and compare its output with the cached path.
        a = torch.linspace(-1, 1, u.shape[-1], dtype=u.dtype, device=u.device)
        a = a / (u.shape[-1] ** 0.5)
        b = torch.linspace(-0.1, 0.1, r0.shape[-1], dtype=r0.dtype, device=r0.device)
        expected = self.norm.forward(r0 + (u @ a).unsqueeze(-1) * b)

        def inject_update(_module, inputs, output):
            return output + (inputs[0] @ a).unsqueeze(-1) * b

        handle = self.projection.register_forward_hook(inject_update)
        try:
            self._check(torch, expected, self._forward(pixels))
        finally:
            handle.remove()

    @staticmethod
    def _check(torch, expected, observed):
        try:
            torch.testing.assert_close(expected, observed, atol=5e-5, rtol=5e-5)
        except AssertionError as exc:
            raise ValueError("The cached CLS path did not reproduce the full ViT forward pass.") from exc


def extract_split(dataset, indices, capture, processor, device, batch_size, torch):
    input_dim = capture.projection.in_features
    hidden_dim = capture.projection.out_features
    u = np.empty((len(indices), input_dim), dtype=np.float32)
    r0 = np.empty((len(indices), hidden_dim), dtype=np.float32)
    y = np.empty(len(indices), dtype=np.int64)
    for start in range(0, len(indices), batch_size):
        stop = min(start + batch_size, len(indices))
        images, labels = zip(*(dataset[int(i)] for i in indices[start:stop]))
        pixels = processor(images=list(images), return_tensors="pt")["pixel_values"].to(device)
        u[start:stop], r0[start:stop] = capture.extract(pixels, torch)
        y[start:stop] = labels
        if (start // batch_size + 1) % 100 == 0 or stop == len(indices):
            print(f"  {stop:,}/{len(indices):,} images cached", flush=True)
    return u, r0, y


def save_cache(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as file:
        temporary = Path(file.name)
        try:
            np.savez(file, **values)
            file.flush()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def prepare(args):
    cache = args.cache.expanduser().resolve()
    if cache.is_file():
        print(f"Using existing ViT cache: {cache}", flush=True)
        return

    try:
        import torch
        from torchvision.datasets import CIFAR100
        from transformers import AutoImageProcessor, ViTModel
    except ImportError as exc:
        raise ValueError(f"Preparing the cache requires PyTorch, torchvision, and transformers. Import error: {exc}") from exc

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Use --device cpu or --device auto.")

    data_dir = args.data_dir.expanduser().resolve()
    train = CIFAR100(str(data_dir), train=True, download=args.download)
    test = CIFAR100(str(data_dir), train=False, download=args.download)
    indices = split_indices(len(train))
    processor = AutoImageProcessor.from_pretrained(args.model, local_files_only=not args.download)
    model = ViTModel.from_pretrained(args.model, add_pooling_layer=False, local_files_only=not args.download)
    model = model.eval().requires_grad_(False).float().to(device)

    values = {"checkpoint": np.array(args.model), "split_seed": np.array(SPLIT_SEED)}
    splits = (
        ("pilot", train, indices["pilot"]),
        ("train", train, indices["train"]),
        ("test", test, np.arange(len(test))),
    )
    with torch.inference_mode(), TokenCapture(model) as capture:
        for split, dataset, rows in splits:
            print(f"Extracting {split} CLS cache ({len(rows):,} images, {device})", flush=True)
            u, r0, y = extract_split(dataset, rows, capture, processor, device, args.batch_size, torch)
            values.update({f"{split}_u": u, f"{split}_r": r0, f"{split}_y": y})
        values["ln_weight"] = capture.norm.weight.cpu().numpy()
        values["ln_bias"] = capture.norm.bias.cpu().numpy()
        values["ln_eps"] = np.array(capture.norm.eps, dtype=np.float64)
    save_cache(cache, values)
    print(f"Saved ViT cache: {cache}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="Output .npz path")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA, help="CIFAR-100 dataset root")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face identifier or local checkpoint path")
    parser.add_argument("--download", action="store_true", help="Allow dataset and pretrained model downloads")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:<index>")
    parser.add_argument("--batch-size", type=int, default=32, help="Feature extraction batch size")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    if args.cache.suffix != ".npz":
        parser.error("--cache must end in .npz.")
    try:
        prepare(args)
    except (OSError, RuntimeError, ValueError) as exc:
        if not args.download:
            parser.exit(1, f"ViT preparation error: {exc}\nUse --download if local data or pretrained files are missing.\n")
        parser.exit(1, f"ViT preparation error: {exc}\n")


if __name__ == "__main__":
    main()

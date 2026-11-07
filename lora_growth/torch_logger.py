"""Read-only, exact numerical factor logging for custom and standard PEFT LoRA.

This module requires PyTorch, but neither imports nor requires Hugging Face PEFT.
It discovers its standard ``lora_A``/``lora_B`` module dictionaries by interface.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterator
from pathlib import Path

import torch
from torch import nn

from .metrics import pair_metrics


_METRIC_FIELDS = [
    "a_spectral_sq", "b_spectral_sq", "spectral_sum",
    "a_frobenius_sq", "b_frobenius_sq", "frobenius_sum",
    "product_spectral", "product_frobenius", "imbalance_spectral", "rank",
]
_CONTEXT_FIELDS = ["step", "epoch", "lr", "batch_loss"]
_PAIR_FIELDS = _CONTEXT_FIELDS + [
    "pair", "adapter_scaling", "adapter_active",
] + _METRIC_FIELDS + [
    "scaled_product_spectral", "scaled_product_frobenius",
    "sampled_running_max_spectral_sum",
]
_AGGREGATE_FIELDS = _CONTEXT_FIELDS + [
    "num_pairs", "log_every", "sum_a_spectral_sq", "sum_b_spectral_sq",
    "sum_spectral_sum", "max_pair_spectral_sum", "sum_frobenius_sum",
    "sampled_running_max_sum_spectral_sum",
    "sampled_running_max_pair_spectral_sum",
]


def _number(value: object, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be scalar")
        value = value.detach().cpu().item()
    try:
        result = float(value)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise ValueError(f"{name} must be finite" + (" and nonnegative" if nonnegative else ""))
    return result


def _weight(value: object, name: str) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else getattr(value, "weight", None)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a tensor or a module with a tensor weight")
    if tensor.is_complex():
        raise ValueError(f"{name} must contain real values")
    if tensor.ndim == 2:
        return tensor
    # Standard PEFT Conv1d/2d/3d stores its output/rank axis first.
    # Compatibility of the flattened A/B factors is checked by pair_metrics.
    if 3 <= tensor.ndim <= 5:
        return tensor.flatten(start_dim=1)
    raise ValueError(f"Unsupported {name} shape {tuple(tensor.shape)}")


def _factor_pairs(model: nn.Module) -> Iterator[tuple[str, torch.Tensor, torch.Tensor, float, bool]]:
    for name, module in model.named_modules():
        module_name = name or "root"
        embedding_a = getattr(module, "lora_embedding_A", None)
        if embedding_a is not None and len(embedding_a):
            raise ValueError(f"Embedding LoRA is not supported: {module_name}")
        A, B = getattr(module, "lora_A", None), getattr(module, "lora_B", None)
        if A is None and B is None:
            continue
        if A is None or B is None:
            raise ValueError(f"Incomplete LoRA factor pair at {module_name}")
        scaling = getattr(module, "scaling", 1.0)
        if isinstance(A, (nn.ModuleDict, nn.ParameterDict)):
            if not isinstance(B, (nn.ModuleDict, nn.ParameterDict)) or set(A) != set(B):
                raise ValueError(f"Mismatched adapter dictionaries at {module_name}")
            active = getattr(module, "active_adapters", list(A))
            if isinstance(active, str):
                active = [active]
            if getattr(module, "disable_adapters", False):
                active = []
            for adapter in A:
                use_dora = getattr(module, "use_dora", {})
                if hasattr(use_dora, "get") and use_dora.get(adapter, False):
                    raise ValueError(f"DoRA is not supported: {module_name}::{adapter}")
                scale = scaling.get(adapter, 1.0) if hasattr(scaling, "get") else scaling
                yield (
                    f"{module_name}::{adapter}",
                    _weight(A[adapter], "A"), _weight(B[adapter], "B"),
                    _number(scale, "adapter scaling"), adapter in active,
                )
        else:
            yield (
                module_name, _weight(A, "A"), _weight(B, "B"),
                _number(scaling, "adapter scaling"), True,
            )


class FactorGrowthLogger:
    """Log raw spectral growth without running a forward/backward pass.

    ``step`` counts completed optimizer updates: call with step=0 before the
    first update, then immediately after optimizer.step(). Initialization is
    always logged. Later calls are sampled at multiples of ``log_every``;
    ``force=True`` includes the final state when it is off cadence. Running
    maxima are explicitly sampled maxima when log_every > 1.

    Existing pair_metrics.csv/aggregate_metrics.csv are never overwritten.
    Model adapter names and factor shapes must remain fixed during a run.
    All configured adapters, including inactive PEFT adapters, are logged;
    ``adapter_active`` identifies active adapters. Aggregates cover all logged
    pairs and are diagnostics, not a single-factor theorem quantity.
    """

    def __init__(self, output_dir: str | Path, log_every: int = 1):
        if isinstance(log_every, bool) or not isinstance(log_every, int) or log_every < 1:
            raise ValueError("log_every must be a positive integer")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_every = log_every
        pair_path = self.output_dir / "pair_metrics.csv"
        aggregate_path = self.output_dir / "aggregate_metrics.csv"
        if pair_path.exists() or aggregate_path.exists():
            raise FileExistsError("Logger CSV files already exist; choose a new output directory")
        self._pair_file = pair_path.open("x", newline="", encoding="utf-8")
        try:
            self._aggregate_file = aggregate_path.open("x", newline="", encoding="utf-8")
        except Exception:
            self._pair_file.close()
            pair_path.unlink(missing_ok=True)
            raise
        self._pair_writer = csv.DictWriter(self._pair_file, fieldnames=_PAIR_FIELDS)
        self._aggregate_writer = csv.DictWriter(self._aggregate_file, fieldnames=_AGGREGATE_FIELDS)
        self._pair_writer.writeheader()
        self._aggregate_writer.writeheader()
        self._last_step: int | None = None
        self._shapes: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] | None = None
        self._pair_maxima: dict[str, float] = {}
        self._max_sum = 0.0
        self._max_pair = 0.0
        self._closed = False

    def log(
        self, model: nn.Module, step: int, epoch: float = 0,
        lr: float | None = None, batch_loss: float | None = None,
        force: bool = False,
    ) -> dict[str, float | int | str] | None:
        """Return the aggregate record, or None when the call is off cadence.

        ``batch_loss`` describes the batch used for the just-completed update;
        it is not a recomputed loss at the post-update factor state.
        """
        if self._closed:
            raise RuntimeError("Logger is closed")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("step must be a nonnegative integer")
        if self._last_step is None and step != 0:
            raise ValueError("Log initialization with step=0 before logging updates")
        if self._last_step is not None and step <= self._last_step:
            raise ValueError("Logged steps must be strictly increasing")
        if step != 0 and step % self.log_every != 0 and not force:
            return None
        context = {
            "step": step, "epoch": _number(epoch, "epoch", nonnegative=True),
            "lr": "" if lr is None else _number(lr, "lr", nonnegative=True),
            "batch_loss": "" if batch_loss is None else _number(batch_loss, "batch_loss"),
        }
        records: list[dict[str, float | int | str]] = []
        shapes = {}
        for name, A, B, scale, active in _factor_pairs(model):
            shapes[name] = (tuple(A.shape), tuple(B.shape))
            # Detached conversion creates no graph, changes no .grad, and uses no RNG.
            metrics = pair_metrics(
                A.detach().to(device="cpu", dtype=torch.float64).numpy(),
                B.detach().to(device="cpu", dtype=torch.float64).numpy(),
            )
            records.append({
                **context, "pair": name, "adapter_scaling": scale,
                "adapter_active": int(active), **metrics,
                "scaled_product_spectral": abs(scale) * metrics["product_spectral"],
                "scaled_product_frobenius": abs(scale) * metrics["product_frobenius"],
            })
        if not records:
            raise ValueError("No supported LoRA pairs were found in model")
        if len(shapes) != len(records):
            raise ValueError("Duplicate adapter pair names")
        if self._shapes is not None and self._shapes != shapes:
            raise ValueError("Adapter names or factor shapes changed during logging")
        self._shapes = shapes
        for row in records:
            name = str(row["pair"])
            self._pair_maxima[name] = max(
                self._pair_maxima.get(name, 0.0), float(row["spectral_sum"])
            )
            row["sampled_running_max_spectral_sum"] = self._pair_maxima[name]
        a_sum = sum(float(row["a_spectral_sq"]) for row in records)
        b_sum = sum(float(row["b_spectral_sq"]) for row in records)
        pair_max = max(float(row["spectral_sum"]) for row in records)
        self._max_sum = max(self._max_sum, a_sum + b_sum)
        self._max_pair = max(self._max_pair, pair_max)
        aggregate = {
            **context, "num_pairs": len(records), "log_every": self.log_every,
            "sum_a_spectral_sq": a_sum, "sum_b_spectral_sq": b_sum,
            "sum_spectral_sum": a_sum + b_sum, "max_pair_spectral_sum": pair_max,
            "sum_frobenius_sum": sum(float(row["frobenius_sum"]) for row in records),
            "sampled_running_max_sum_spectral_sum": self._max_sum,
            "sampled_running_max_pair_spectral_sum": self._max_pair,
        }
        self._pair_writer.writerows(records)
        self._aggregate_writer.writerow(aggregate)
        self._pair_file.flush()
        self._aggregate_file.flush()
        self._last_step = step
        return aggregate

    def close(self) -> None:
        """Flush and close both CSVs. Repeated close calls are harmless."""
        if not self._closed:
            self._pair_file.close()
            self._aggregate_file.close()
            self._closed = True

    def __enter__(self) -> "FactorGrowthLogger":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

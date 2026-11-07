"""Optional PyTorch LoRA layers with unit scaling and frozen base weights.

PyTorch is deliberately imported only by this optional module. The NumPy
logistic experiment and exact metric calculations do not require PyTorch.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import torch
from torch import nn
from torch.nn import functional as F


def _validate_initialization(rank: int, init_std: float) -> None:
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError("rank must be a positive integer")
    if not math.isfinite(init_std) or init_std <= 0:
        raise ValueError("init_std must be positive and finite")


class LoRALinear(nn.Module):
    """Replace a linear layer by W0 + BA, with A Gaussian and B zero.

    Bias and W0 are frozen buffers. No alpha/r scaling, factor weight decay,
    normalization, dropout, or clipping is applied by this layer. Adapter widths
    larger than min(in,out) are accepted, but are overparameterized.
    """

    def __init__(self, base_layer: nn.Linear, rank: int, init_std: float = 1e-3):
        super().__init__()
        _validate_initialization(rank, init_std)
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("base_layer must be torch.nn.Linear")
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = rank
        self.scaling = 1.0
        self.register_buffer("weight", base_layer.weight.detach().clone())
        self.register_buffer(
            "bias", None if base_layer.bias is None else base_layer.bias.detach().clone()
        )
        self.lora_A = nn.Parameter(self.weight.new_empty((rank, self.in_features)))
        self.lora_B = nn.Parameter(self.weight.new_zeros((self.out_features, rank)))
        nn.init.normal_(self.lora_A, mean=0.0, std=init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias) + F.linear(
            F.linear(x, self.lora_A), self.lora_B
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, scaling=1.0, frozen_bias={self.bias is not None}"
        )


class LoRAConv2d(nn.Module):
    """Factor a convolution's flattened weight matrix as W0 + BA.

    A has shape (r, in_channels*kH*kW), B has shape (out_channels,r).
    Only groups=1 is supported. Stride, padding, dilation and padding mode are
    copied from the original convolution; all base weights and biases are frozen.
    """

    def __init__(self, base_layer: nn.Conv2d, rank: int, init_std: float = 1e-3):
        super().__init__()
        _validate_initialization(rank, init_std)
        if not isinstance(base_layer, nn.Conv2d):
            raise TypeError("base_layer must be torch.nn.Conv2d")
        if base_layer.groups != 1:
            raise ValueError("LoRAConv2d supports groups=1 only")
        self.in_channels = base_layer.in_channels
        self.out_channels = base_layer.out_channels
        self.kernel_size = base_layer.kernel_size
        self.stride = base_layer.stride
        self.padding = base_layer.padding
        self.dilation = base_layer.dilation
        self.padding_mode = base_layer.padding_mode
        self._reversed_padding_repeated_twice = base_layer._reversed_padding_repeated_twice
        self.groups = 1
        self.rank = rank
        self.scaling = 1.0
        self.register_buffer("weight", base_layer.weight.detach().clone())
        self.register_buffer(
            "bias", None if base_layer.bias is None else base_layer.bias.detach().clone()
        )
        width = self.in_channels * math.prod(self.kernel_size)
        self.lora_A = nn.Parameter(self.weight.new_empty((rank, width)))
        self.lora_B = nn.Parameter(self.weight.new_zeros((self.out_channels, rank)))
        nn.init.normal_(self.lora_A, mean=0.0, std=init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight + (self.lora_B @ self.lora_A).reshape_as(self.weight)
        if self.padding_mode != "zeros":
            x = F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            padding = 0
        else:
            padding = self.padding
        return F.conv2d(x, weight, self.bias, self.stride, padding, self.dilation, 1)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, rank={self.rank}, scaling=1.0"
        )


def lora_pairs(model: nn.Module) -> Iterator[tuple[str, nn.Parameter, nn.Parameter]]:
    """Yield (module_name, A, B) for each custom LoRA module, once per module."""
    for name, module in model.named_modules():
        if isinstance(module, (LoRALinear, LoRAConv2d)):
            yield name or "root", module.lora_A, module.lora_B

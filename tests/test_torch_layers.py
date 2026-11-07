"""Optional runtime checks for the simultaneous PyTorch LoRA implementation."""
from __future__ import annotations

import unittest

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from lora_growth.torch_layers import LoRAConv2d, LoRALinear, lora_pairs
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Optional PyTorch dependency is not installed")
class TorchLayersTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def assertTensorClose(self, actual, expected, *, rtol=1e-10, atol=1e-10):
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)

    def test_linear_matches_merged_weight_and_initial_base(self):
        base = nn.Linear(4, 3, bias=True).double()
        layer = LoRALinear(base, rank=2, init_std=0.2)
        x = torch.randn(5, 4, dtype=torch.float64)
        self.assertTensorClose(layer(x), base(x))
        with torch.no_grad():
            layer.lora_B.normal_(std=0.3)
        merged = base.weight + layer.lora_B @ layer.lora_A
        self.assertTensorClose(layer(x), F.linear(x, merged, base.bias))
        self.assertEqual(layer.scaling, 1.0)

    def _check_finite_difference(self, layer, x):
        with torch.no_grad():
            layer.lora_B.normal_(std=0.15)
        loss = layer(x).square().sum() / 2
        loss.backward()
        epsilon = 1e-6
        for parameter in (layer.lora_A, layer.lora_B):
            analytic = parameter.grad.detach().clone()
            flat = parameter.view(-1)
            # Check the whole small parameter tensor, including both factors.
            for index in range(parameter.numel()):
                with torch.no_grad():
                    previous = flat[index].item()
                    flat[index] = previous + epsilon
                    plus = (layer(x).square().sum() / 2).item()
                    flat[index] = previous - epsilon
                    minus = (layer(x).square().sum() / 2).item()
                    flat[index] = previous
                numerical = (plus - minus) / (2 * epsilon)
                self.assertAlmostEqual(analytic.view(-1)[index].item(), numerical, delta=2e-6)

    def test_linear_factor_gradients_match_finite_differences(self):
        layer = LoRALinear(nn.Linear(3, 2, bias=False).double(), rank=2, init_std=0.2)
        self._check_finite_difference(layer, torch.randn(3, 3, dtype=torch.float64))

    def test_conv_factor_gradients_match_finite_differences(self):
        layer = LoRAConv2d(nn.Conv2d(1, 2, 2, bias=True).double(), rank=2, init_std=0.2)
        self._check_finite_difference(layer, torch.randn(1, 1, 4, 4, dtype=torch.float64))

    def test_joint_sgd_step_uses_both_preupdate_factors(self):
        layer = LoRALinear(nn.Linear(3, 2, bias=False).double(), rank=2, init_std=0.2)
        with torch.no_grad():
            layer.lora_B.normal_(std=0.2)
        A_before = layer.lora_A.detach().clone()
        B_before = layer.lora_B.detach().clone()
        W_before = layer.weight.detach().clone()
        x = torch.randn(5, 3, dtype=torch.float64)
        target = torch.randn(5, 2, dtype=torch.float64)
        residual = F.linear(x, W_before + B_before @ A_before) - target
        G = residual.T @ x / len(x)
        eta = 0.07
        expected_A = A_before - eta * B_before.T @ G
        expected_B = B_before - eta * G @ A_before.T
        optimizer = torch.optim.SGD(layer.parameters(), lr=eta, momentum=0.0, weight_decay=0.0)
        optimizer.zero_grad(set_to_none=True)
        loss = (layer(x) - target).square().sum() / (2 * len(x))
        loss.backward()
        optimizer.step()
        self.assertTensorClose(layer.lora_A, expected_A)
        self.assertTensorClose(layer.lora_B, expected_B)
        self.assertTensorClose(layer.weight, W_before)
        self.assertIsNone(layer.weight.grad)
        self.assertEqual(set(dict(layer.named_parameters())), {"lora_A", "lora_B"})

    def test_convolution_preserves_stride_padding_dilation(self):
        base = nn.Conv2d(2, 3, 3, stride=2, padding=2, dilation=2, bias=True,
                         padding_mode="reflect").double()
        layer = LoRAConv2d(base, rank=2, init_std=0.1)
        x = torch.randn(2, 2, 11, 11, dtype=torch.float64)
        with torch.no_grad():
            layer.lora_B.normal_(std=0.1)
        merged = base.weight + (layer.lora_B @ layer.lora_A).reshape_as(base.weight)
        expected = F.conv2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), merged,
                            base.bias, stride=2, padding=0, dilation=2)
        self.assertTensorClose(layer(x), expected)

    def test_overcomplete_classifier_width_and_pair_discovery(self):
        linear = LoRALinear(nn.Linear(32, 10), rank=20, init_std=0.01)
        model = nn.Sequential(linear)
        self.assertEqual(tuple(linear.lora_A.shape), (20, 32))
        self.assertEqual(tuple(linear.lora_B.shape), (10, 20))
        self.assertEqual(tuple(model(torch.randn(4, 32)).shape), (4, 10))
        discovered = list(lora_pairs(model))
        self.assertEqual(len(discovered), 1)
        self.assertEqual(discovered[0][0], "0")
        self.assertIs(discovered[0][1], linear.lora_A)
        self.assertIs(discovered[0][2], linear.lora_B)

    def test_eval_freezes_batchnorm_statistics_during_adapter_training(self):
        model = nn.Sequential(nn.Conv2d(2, 3, 3, padding=1), nn.BatchNorm2d(3),
                              nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        model.requires_grad_(False)
        model[0] = LoRAConv2d(model[0], rank=2, init_std=0.1)
        model.eval()
        mean_before = model[1].running_mean.clone()
        variance_before = model[1].running_var.clone()
        count_before = model[1].num_batches_tracked.clone()
        base_before = model[0].weight.clone()
        B_before = model[0].lora_B.detach().clone()
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            model(torch.randn(4, 2, 8, 8)).square().mean().backward()
            optimizer.step()
        self.assertTensorClose(model[1].running_mean, mean_before)
        self.assertTensorClose(model[1].running_var, variance_before)
        self.assertTensorClose(model[1].num_batches_tracked, count_before)
        self.assertTensorClose(model[0].weight, base_before)
        self.assertFalse(torch.equal(model[0].lora_B, B_before))
        self.assertTrue(model[0].lora_A.requires_grad)
        self.assertTrue(model[0].lora_B.requires_grad)
        self.assertFalse(model[1].weight.requires_grad)

    def test_grouped_convolution_is_rejected_explicitly(self):
        with self.assertRaisesRegex(ValueError, "groups=1"):
            LoRAConv2d(nn.Conv2d(4, 4, 3, groups=2), rank=2)


if __name__ == "__main__":
    unittest.main()

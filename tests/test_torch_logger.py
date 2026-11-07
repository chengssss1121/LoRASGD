"""Optional PyTorch integration checks; no PEFT/model downloads are required."""

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from torch import nn
    from lora_growth.torch_layers import LoRAConv2d, LoRALinear
    from lora_growth.torch_logger import FactorGrowthLogger


@unittest.skipUnless(HAS_TORCH, "optional PyTorch is not installed")
class TorchLoggerTests(unittest.TestCase):
    def _rows(self, path):
        with Path(path).open(newline="", encoding="utf-8") as source:
            return list(csv.DictReader(source))

    def test_logging_preserves_modes_gradients_parameters_and_rng(self):
        torch.manual_seed(13)
        model = nn.Sequential(LoRALinear(nn.Linear(5, 4), 2), nn.Dropout(0.2))
        model.train()
        model[0].eval()  # Preserve mixed per-module modes, not just root mode.
        model(torch.randn(3, 5)).square().mean().backward()
        modes = [m.training for m in model.modules()]
        parameters = {name: p.detach().clone() for name, p in model.named_parameters()}
        gradients = {name: None if p.grad is None else p.grad.clone() for name, p in model.named_parameters()}
        torch_state = torch.get_rng_state().clone()
        numpy_state = np.random.get_state()
        with tempfile.TemporaryDirectory() as directory:
            with FactorGrowthLogger(directory) as logger:
                record = logger.log(model, 0)
            self.assertEqual(record["num_pairs"], 1)
            self.assertEqual(self._rows(Path(directory) / "pair_metrics.csv")[0]["b_spectral_sq"], "0.0")
        self.assertEqual(modes, [m.training for m in model.modules()])
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        after_numpy = np.random.get_state()
        self.assertEqual(numpy_state[0], after_numpy[0])
        self.assertTrue(np.array_equal(numpy_state[1], after_numpy[1]))
        self.assertEqual(numpy_state[2:], after_numpy[2:])
        for name, p in model.named_parameters():
            self.assertTrue(torch.equal(parameters[name], p))
            if gradients[name] is None:
                self.assertIsNone(p.grad)
            else:
                self.assertTrue(torch.equal(gradients[name], p.grad))

    def test_cadence_and_forced_final_state(self):
        model = LoRALinear(nn.Linear(3, 2), 2)
        with tempfile.TemporaryDirectory() as directory:
            with FactorGrowthLogger(directory, log_every=3) as logger:
                logger.log(model, 0)
                self.assertIsNone(logger.log(model, 1))
                self.assertIsNone(logger.log(model, 2))
                logger.log(model, 3, lr=0.01, batch_loss=0.5)
                logger.log(model, 4, force=True)
            self.assertEqual(
                [int(row["step"]) for row in self._rows(Path(directory) / "aggregate_metrics.csv")],
                [0, 3, 4],
            )
            with self.assertRaises(FileExistsError):
                FactorGrowthLogger(directory)

    def test_fake_peft_multiple_adapters_and_scaling(self):
        class PeftLike(nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_A = nn.ModuleDict({
                    "default": nn.Linear(5, 2, bias=False),
                    "other": nn.Linear(5, 3, bias=False),
                })
                self.lora_B = nn.ModuleDict({
                    "default": nn.Linear(2, 4, bias=False),
                    "other": nn.Linear(3, 4, bias=False),
                })
                self.scaling = {"default": 2., "other": .5}
                self.active_adapters = ["default"]

        model = PeftLike()
        with tempfile.TemporaryDirectory() as directory:
            with FactorGrowthLogger(directory) as logger:
                aggregate = logger.log(model, 0)
            rows = self._rows(Path(directory) / "pair_metrics.csv")
        self.assertEqual(aggregate["num_pairs"], 2)
        self.assertEqual({row["pair"] for row in rows}, {"root::default", "root::other"})
        self.assertAlmostEqual(
            aggregate["sum_spectral_sum"], sum(float(row["spectral_sum"]) for row in rows)
        )
        for row in rows:
            self.assertAlmostEqual(
                float(row["scaled_product_spectral"]),
                abs(float(row["adapter_scaling"])) * float(row["product_spectral"]),
            )
            self.assertEqual(row["adapter_active"], "1" if row["pair"].endswith("default") else "0")

    def test_factor_rescaling_changes_raw_norms_not_product(self):
        model = LoRALinear(nn.Linear(5, 4), 2, init_std=.1)
        with torch.no_grad():
            model.lora_B.normal_()
        with tempfile.TemporaryDirectory() as directory:
            with FactorGrowthLogger(directory) as logger:
                logger.log(model, 0)
                with torch.no_grad():
                    model.lora_A.div_(3.)
                    model.lora_B.mul_(3.)
                logger.log(model, 1)
            before, after = self._rows(Path(directory) / "pair_metrics.csv")
        self.assertNotAlmostEqual(float(before["spectral_sum"]), float(after["spectral_sum"]))
        self.assertAlmostEqual(float(before["product_spectral"]), float(after["product_spectral"]), places=5)

    def test_conv_preserves_base_output_and_only_factors_are_parameters(self):
        for padding_mode in ["zeros", "reflect"]:
            with self.subTest(padding_mode=padding_mode):
                base = nn.Conv2d(3, 4, 3, padding=1, padding_mode=padding_mode)
                layer = LoRAConv2d(base, 2)
                x = torch.randn(2, 3, 7, 9)
                torch.testing.assert_close(layer(x), base(x))
                self.assertEqual(set(dict(layer.named_parameters())), {"lora_A", "lora_B"})
                with tempfile.TemporaryDirectory() as directory:
                    with FactorGrowthLogger(directory) as logger:
                        logger.log(layer, 0)

    def test_single_optimizer_uses_both_preupdate_gradients(self):
        layer = LoRALinear(nn.Linear(3, 2), 2, init_std=.1).double()
        with torch.no_grad():
            layer.lora_B.normal_(std=.2)
        x, target = torch.randn(4, 3, dtype=torch.float64), torch.randn(4, 2, dtype=torch.float64)
        optimizer = torch.optim.SGD(layer.parameters(), lr=.03, momentum=0., weight_decay=0.)
        optimizer.zero_grad(set_to_none=True)
        (layer(x) - target).square().mean().backward()
        A, B = layer.lora_A.detach().clone(), layer.lora_B.detach().clone()
        dA, dB = layer.lora_A.grad.clone(), layer.lora_B.grad.clone()
        optimizer.step()
        torch.testing.assert_close(layer.lora_A, A - .03 * dA)
        torch.testing.assert_close(layer.lora_B, B - .03 * dB)

    def test_peft_convolutional_factors_are_flattened(self):
        class PeftLikeConv(nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_A = nn.ModuleDict({"default": nn.Conv2d(3, 2, 3, bias=False)})
                self.lora_B = nn.ModuleDict({"default": nn.Conv2d(2, 4, 1, bias=False)})
                self.scaling = {"default": 3.}

        model = PeftLikeConv()
        product = (
            model.lora_B["default"].weight.detach().flatten(1)
            @ model.lora_A["default"].weight.detach().flatten(1)
        )
        with tempfile.TemporaryDirectory() as directory:
            with FactorGrowthLogger(directory) as logger:
                logger.log(model, 0)
            row = self._rows(Path(directory) / "pair_metrics.csv")[0]
        self.assertAlmostEqual(
            float(row["product_spectral"]), torch.linalg.matrix_norm(product, ord=2).item(), places=5
        )


if __name__ == "__main__":
    unittest.main()

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lora_growth.logistic import evaluate, initialize, simultaneous_step, weight_loss_gradient
from train_logistic import main


class LogisticTests(unittest.TestCase):
    def test_factor_gradients_match_finite_differences(self):
        rng = np.random.default_rng(20)
        x = rng.normal(size=(5, 3))
        y = np.array([0, 1, 0, 1, 1])
        W0, bias, A, B = initialize(3, 2, 2, 0.2, 13)
        B[:] = rng.normal(size=B.shape) * 0.1
        _, G, _ = weight_loss_gradient(x, y, W0, bias, A, B, 0.3)
        for matrix, analytical in [(A, B.T @ G), (B, G @ A.T)]:
            numerical = np.zeros_like(matrix)
            for index in np.ndindex(matrix.shape):
                old = matrix[index]
                matrix[index] = old + 1e-6
                plus = weight_loss_gradient(x, y, W0, bias, A, B, 0.3)[0]
                matrix[index] = old - 1e-6
                minus = weight_loss_gradient(x, y, W0, bias, A, B, 0.3)[0]
                matrix[index] = old
                numerical[index] = (plus - minus) / 2e-6
            np.testing.assert_allclose(analytical, numerical, rtol=2e-6, atol=2e-9)

    def test_joint_step_and_zero_B_initialization(self):
        W0, bias, A, B = initialize(3, 2, 1, 0.2, 13)
        x, y = np.eye(3), np.array([0, 1, 0])
        _, G, _ = weight_loss_gradient(x, y, W0, bias, A, B)
        A_new, B_new, *_ = simultaneous_step(x, y, W0, bias, A, B, 0.5)
        np.testing.assert_array_equal(A_new, A)  # A's first simultaneous gradient is zero.
        np.testing.assert_allclose(B_new, -0.5 * G @ A.T)
        self.assertGreater(np.linalg.norm(B_new), 0)
        np.testing.assert_array_equal(B, np.zeros_like(B))

    def test_chunked_evaluation_matches_full_batch(self):
        rng = np.random.default_rng(8)
        x, y = rng.normal(size=(17, 3)), rng.integers(2, size=17)
        W0, bias, A, B = initialize(3, 2, 2, 0.2, 13)
        B[:] = 0.1
        loss, G, acc = weight_loss_gradient(x, y, W0, bias, A, B, 0.3)
        result = evaluate(x, y, W0, bias, A, B, 0.3, chunk_size=4)
        self.assertAlmostEqual(loss, result["loss"], places=12)
        self.assertAlmostEqual(acc, result["accuracy"], places=12)
        self.assertAlmostEqual(np.sum((B.T @ G)**2) + np.sum((G @ A.T)**2),
                               result["full_factor_gradient_sq"], places=12)

    def test_logging_cadence_does_not_change_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            common = ["--synthetic", "--n-train", "32", "--n-test", "16", "--feature-dim", "5",
                      "--n-classes", "3", "--epochs", "4", "--batch-size", "8", "--max-steps", "12"]
            with contextlib.redirect_stdout(io.StringIO()):
                main(common + ["--output", str(root / "dense"), "--log-every", "1"])
                main(common + ["--output", str(root / "sparse"), "--log-every", "5"])
            with np.load(root / "dense/final_factors.npz") as a, np.load(root / "sparse/final_factors.npz") as b:
                for field in ["A", "B", "W0"]:
                    np.testing.assert_array_equal(a[field], b[field])

    def test_conservative_fixed_dataset_variance_bound(self):
        rng = np.random.default_rng(30)
        x, y = rng.normal(size=(12, 3)), rng.integers(2, size=12)
        W0, bias, A, B = initialize(3, 2, 2, 0.2, 13)
        B[:] = rng.normal(size=B.shape)
        gradients = np.stack([weight_loss_gradient(x[i:i+1], y[i:i+1], W0, bias, A, B, 1.5)[1]
                              for i in range(len(y))])
        measured = np.mean(np.sum((gradients - gradients.mean(axis=0))**2, axis=(1, 2)))
        self.assertLessEqual(measured, 2 * np.max(np.sum(x**2, axis=1)) + 1e-12)


if __name__ == "__main__":
    unittest.main()

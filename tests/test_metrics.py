"""Independent numerical checks for the spectral-norm measurements."""

import unittest

import numpy as np

from lora_growth.metrics import pair_metrics, spectral_norm_sq


class MetricsTests(unittest.TestCase):
    def test_diagonal_spectral_is_not_frobenius(self):
        A = np.diag([3.0, 2.0])
        B = np.diag([4.0, 1.0])
        metrics = pair_metrics(A, B)
        self.assertAlmostEqual(metrics["a_spectral_sq"], 9.0)
        self.assertAlmostEqual(metrics["b_spectral_sq"], 16.0)
        self.assertAlmostEqual(metrics["spectral_sum"], 25.0)
        self.assertEqual(metrics["frobenius_sum"], 30.0)

    def test_qr_core_matches_direct_product(self):
        rng = np.random.default_rng(13)
        # Includes overcomplete factors and rank-deficient factors.
        for out_dim, in_dim, rank in [(17, 23, 4), (2, 3, 6), (5, 2, 4), (9, 7, 1)]:
            with self.subTest(shape=(out_dim, in_dim, rank)):
                A = rng.normal(size=(rank, in_dim))
                B = rng.normal(size=(out_dim, rank))
                if rank > 1:
                    A[-1] = A[0]
                metrics = pair_metrics(A, B)
                product = B @ A
                self.assertAlmostEqual(
                    metrics["product_spectral"], np.linalg.norm(product, ord=2), places=10
                )
                self.assertAlmostEqual(
                    metrics["product_frobenius"], np.linalg.norm(product, ord="fro"), places=10
                )
                self.assertAlmostEqual(
                    metrics["imbalance_spectral"],
                    np.linalg.norm(A @ A.T - B.T @ B, ord=2), places=10
                )

    def test_zero_initial_B(self):
        A = np.array([[1., 2., 3.], [2., 1., 0.]])
        metrics = pair_metrics(A, np.zeros((4, 2)))
        self.assertEqual(metrics["b_spectral_sq"], 0.0)
        self.assertEqual(metrics["product_spectral"], 0.0)
        self.assertEqual(metrics["product_frobenius"], 0.0)
        self.assertEqual(metrics["spectral_sum"], spectral_norm_sq(A))

    def test_rescaling_preserves_product_but_not_factor_growth(self):
        rng = np.random.default_rng(5)
        A, B = rng.normal(size=(3, 8)), rng.normal(size=(6, 3))
        before = pair_metrics(A, B)
        after = pair_metrics(A / 7.0, B * 7.0)
        self.assertAlmostEqual(before["product_spectral"], after["product_spectral"], places=10)
        self.assertNotAlmostEqual(before["spectral_sum"], after["spectral_sum"])

    def test_bad_inputs(self):
        cases = [
            (np.ones(3), np.ones((3, 3))),
            (np.empty((0, 3)), np.empty((3, 0))),
            (np.ones((2, 3)), np.ones((4, 3))),
            (np.array([[np.inf]]), np.ones((1, 1))),
            (np.array([[np.nan]]), np.ones((1, 1))),
            (np.array([[1j]]), np.ones((1, 1))),
        ]
        for A, B in cases:
            with self.subTest(A=A):
                with self.assertRaises(ValueError):
                    pair_metrics(A, B)


if __name__ == "__main__":
    unittest.main()

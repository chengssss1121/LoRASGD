"""Frozen-base multiclass logistic regression with an unscaled BA update.

The classifier is logits(x) = x @ (W0 + B @ A).T + frozen_bias.
The objective adds lambda_product / 2 * ||B @ A||_F**2 if requested.
Both factor gradients are computed at the same pre-update (B, A).
"""

from __future__ import annotations

import numpy as np


def initialize(in_features: int, classes: int, rank: int, init_std: float, seed: int):
    if min(in_features, classes, rank) < 1 or init_std <= 0:
        raise ValueError("Dimensions/rank and init_std must be positive")
    rng = np.random.default_rng(seed)
    bound = 1.0 / np.sqrt(in_features)
    W0 = rng.uniform(-bound, bound, size=(classes, in_features))
    bias = np.zeros(classes, dtype=np.float64)
    A = rng.normal(0.0, init_std, size=(rank, in_features))
    B = np.zeros((classes, rank), dtype=np.float64)
    return W0, bias, A, B


def weight_loss_gradient(x, y, W0, bias, A, B, lambda_product=0.0):
    """Return mean objective, gradient with respect to BA, and accuracy.

    Uses log-sum-exp for a numerically stable softmax cross-entropy.
    The regularization term is on the PRODUCT, not on individual factors.
    """
    if lambda_product < 0:
        raise ValueError("lambda_product must be nonnegative")
    update = B @ A
    logits = x @ (W0 + update).T + bias
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    loss = np.mean(np.log(exp_logits.sum(axis=1)) - shifted[np.arange(len(y)), y])
    accuracy = np.mean(np.argmax(logits, axis=1) == y)
    probabilities[np.arange(len(y)), y] -= 1.0
    gradient = probabilities.T @ x / len(y)
    if lambda_product:
        loss += 0.5 * lambda_product * np.sum(update * update)
        gradient += lambda_product * update
    return float(loss), gradient, float(accuracy)


def simultaneous_step(x, y, W0, bias, A, B, lr, lambda_product=0.0):
    """Return new factors; neither input factor is mutated."""
    loss, G, accuracy = weight_loss_gradient(x, y, W0, bias, A, B, lambda_product)
    grad_A = B.T @ G
    grad_B = G @ A.T
    gradient_sq = float(np.sum(grad_A**2) + np.sum(grad_B**2))
    return A - lr * grad_A, B - lr * grad_B, loss, gradient_sq, accuracy


def evaluate(x, y, W0, bias, A, B, lambda_product=0.0, chunk_size=4096):
    """Compute exact full-dataset mean loss and factor gradient in chunks."""
    total_loss, total_accuracy = 0.0, 0.0
    G = np.zeros_like(W0)
    for start in range(0, len(y), chunk_size):
        xs, ys = x[start:start + chunk_size], y[start:start + chunk_size]
        loss, part, accuracy = weight_loss_gradient(xs, ys, W0, bias, A, B, 0.0)
        total_loss += len(ys) * loss
        total_accuracy += len(ys) * accuracy
        G += len(ys) * part
    G /= len(y)
    update = B @ A
    loss = total_loss / len(y) + 0.5 * lambda_product * np.sum(update**2)
    G += lambda_product * update
    grad_A, grad_B = B.T @ G, G @ A.T
    return {
        "loss": float(loss),
        "accuracy": float(total_accuracy / len(y)),
        "full_factor_gradient_sq": float(np.sum(grad_A**2) + np.sum(grad_B**2)),
    }


def make_synthetic(n_train=512, n_test=256, feature_dim=32, classes=10, seed=1729):
    """Offline smoke data only; these are NOT CIFAR-10 results."""
    rng = np.random.default_rng(seed)
    teacher = rng.normal(size=(classes, feature_dim))
    x = rng.normal(size=(n_train + n_test, feature_dim)) / np.sqrt(feature_dim)
    logits = 3.0 * x @ teacher.T + 0.15 * rng.normal(size=(len(x), classes))
    y = logits.argmax(axis=1).astype(np.int64)
    return x[:n_train], y[:n_train], x[n_train:], y[n_train:]

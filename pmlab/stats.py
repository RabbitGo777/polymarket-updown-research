"""Scores and uncertainty estimates."""

import numpy as np

FEE_RATE = 0.07


def taker_fee(price, shares=1.0):
    """Polymarket crypto taker fee in USDC. Makers pay none; quoted prices exclude it."""
    return shares * FEE_RATE * price * (1.0 - price)


def brier(p, y):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def log_loss(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def logit(p):
    p = np.clip(np.asarray(p, float), 0.005, 0.995)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))


def cluster_bootstrap(values, clusters, n=2000, seed=0):
    """Mean of `values` with a 95% interval, resampling whole clusters.

    Rows of one market share one outcome, so resampling rows would overstate the
    evidence many times over. The cluster is the unit of independent information.
    """
    ids, inv = np.unique(clusters, return_inverse=True)
    totals = np.bincount(inv, weights=values, minlength=len(ids))
    counts = np.bincount(inv, minlength=len(ids))
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(ids), size=(n, len(ids)))
    boot = totals[pick].sum(1) / counts[pick].sum(1)
    return float(totals.sum() / counts.sum()), *np.percentile(boot, [2.5, 97.5])

"""Paired bootstrap confidence intervals, exact McNemar tests, Holm correction and Wilson intervals."""
import math

import numpy as np
from scipy.stats import binomtest

N_BOOT = 2000
SEED = 553


def macro_f1(gold, pred, n_labels=77):
    """Macro-F1 over all labels (labels absent from gold and pred contribute 0, as sklearn zero_division=0)."""
    tp = np.bincount(gold[gold == pred], minlength=n_labels).astype(float)
    pc = np.bincount(pred, minlength=n_labels).astype(float)
    gc = np.bincount(gold, minlength=n_labels).astype(float)
    denom = pc + gc
    f1 = np.divide(2 * tp, denom, out=np.zeros(n_labels), where=denom > 0)
    return float(f1.mean())


def accuracy(gold, pred):
    return float((gold == pred).mean())


METRICS = {"macro_f1": macro_f1, "accuracy": accuracy}


def boot_indices(n, n_boot=N_BOOT, seed=SEED):
    return np.random.RandomState(seed).randint(0, n, size=(n_boot, n))


def paired_bootstrap(gold, preds, ref, metric="macro_f1", n_boot=N_BOOT, seed=SEED):
    """preds: {condition: pred array aligned with gold}. Same resamples for every condition.

    Returns {condition: {"value", "ci", "delta", "delta_ci"}} with deltas against preds[ref].
    """
    f = METRICS[metric]
    idx = boot_indices(len(gold), n_boot, seed)
    point = {c: f(gold, p) for c, p in preds.items()}
    dist = {c: np.array([f(gold[i], p[i]) for i in idx]) for c, p in preds.items()}
    out = {}
    for c in preds:
        d = dist[c] - dist[ref]
        out[c] = {"value": point[c], "ci": [float(np.percentile(dist[c], 2.5)), float(np.percentile(dist[c], 97.5))],
                  "delta": point[c] - point[ref],
                  "delta_ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    return out


def mcnemar_exact(correct_a, correct_b):
    """Exact McNemar test on paired correctness vectors; returns (b, c, p)."""
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    if b + c == 0:
        return b, c, 1.0
    return b, c, float(binomtest(b, b + c, 0.5).pvalue)


def holm(pvalues):
    """Holm step-down adjusted p-values (same order as input)."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adj[i] = running
    return adj


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (centre - half, centre + half)

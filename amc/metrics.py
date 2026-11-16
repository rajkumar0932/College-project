"""Evaluation metrics: accuracy vs SNR, confusion matrices, confused pairs."""
from __future__ import annotations

import numpy as np


def accuracy_by_snr(y_true, y_pred, snr):
    snrs = np.unique(snr)
    acc = np.array([np.mean(y_pred[snr == s] == y_true[snr == s]) for s in snrs])
    return snrs, acc


def confusion(y_true, y_pred, n_classes, normalize=True):
    cm = np.zeros((n_classes, n_classes), dtype=np.float64)
    np.add.at(cm, (y_true, y_pred), 1)
    if normalize:
        cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    return cm


def top_confusions(cm, classes, k=8):
    """Largest off-diagonal entries of a row-normalized confusion matrix."""
    pairs = []
    n = len(classes)
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                pairs.append((classes[i], classes[j], float(cm[i, j])))
    pairs.sort(key=lambda p: -p[2])
    return pairs[:k]


def summarize(y_true, y_pred, snr):
    snrs, acc = accuracy_by_snr(y_true, y_pred, snr)
    d = dict(zip(snrs.tolist(), acc.tolist()))
    hi = snr >= 0
    return {
        "overall_acc": float(np.mean(y_true == y_pred)),
        "acc_snr_ge_0": float(np.mean(y_true[hi] == y_pred[hi])) if hi.any() else None,
        "acc_snr_lt_0": float(np.mean(y_true[~hi] == y_pred[~hi])) if (~hi).any() else None,
        "peak_acc": float(acc.max()),
        "peak_snr": int(snrs[acc.argmax()]),
        "acc_at": {int(s): round(float(a), 4) for s, a in d.items()},
    }

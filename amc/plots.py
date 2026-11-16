"""All figures. Uses a non-interactive backend so scripts run headless."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3, "font.size": 10,
})

# Okabe-Ito palette: readable for colour-blind viewers and in grayscale prints.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def plot_acc_vs_snr(curves: dict, path: str, title="Classification accuracy vs SNR"):
    """curves: {label: (snrs, accs)}"""
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for i, (label, (s, a)) in enumerate(curves.items()):
        ax.plot(s, np.asarray(a) * 100, marker=MARKERS[i % len(MARKERS)], ms=4.5,
                lw=1.8, color=PALETTE[i % len(PALETTE)], label=label)
    ax.axhline(100 / 11, color="gray", ls=":", lw=1, label="Chance (1/11)")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(np.arange(-20, 20, 2))
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    fig.savefig(path)
    plt.close(fig)


def plot_confusion(cm, classes, path, title):
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    sns.heatmap(cm * 100, annot=True, fmt=".0f", cmap="Blues", vmin=0, vmax=100,
                xticklabels=classes, yticklabels=classes, cbar_kws={"label": "%"},
                annot_kws={"size": 7.5}, ax=ax, linewidths=0.3, linecolor="white")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    ax.grid(False)
    fig.savefig(path)
    plt.close(fig)


def plot_per_class_acc(per_class: dict, snrs, path, title):
    """per_class: {class: accs over snrs}"""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    cmap = plt.get_cmap("tab20")
    for i, (c, a) in enumerate(per_class.items()):
        ax.plot(snrs, np.asarray(a) * 100, marker=MARKERS[i % len(MARKERS)], ms=3.5,
                lw=1.4, color=cmap(i / max(len(per_class) - 1, 1)), label=c)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Per-class accuracy (%)")
    ax.set_ylim(0, 102)
    ax.set_title(title)
    ax.legend(ncol=2, fontsize=8, loc="lower right", framealpha=0.9)
    fig.savefig(path)
    plt.close(fig)


def plot_history(history: dict, path: str, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ep = np.arange(1, len(history["loss"]) + 1)
    axes[0].plot(ep, history["loss"], color=PALETTE[0], label="train")
    axes[0].plot(ep, history["val_loss"], color=PALETTE[1], label="validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend()
    axes[1].plot(ep, np.asarray(history["accuracy"]) * 100, color=PALETTE[0], label="train")
    axes[1].plot(ep, np.asarray(history["val_accuracy"]) * 100, color=PALETTE[1], label="validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    fig.suptitle(title)
    fig.savefig(path)
    plt.close(fig)


def plot_feature_importance(names, importances, path, title):
    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.barh(np.array(names)[order], np.asarray(importances)[order], color=PALETTE[0])
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_title(title)
    fig.savefig(path)
    plt.close(fig)


def plot_examples(X, y, snr, classes, path, snr_db=18, title=None):
    """One example per class: I/Q waveform (top) and constellation (bottom)."""
    n = len(classes)
    fig, axes = plt.subplots(2, n, figsize=(2.0 * n, 4.2))
    for c in range(n):
        idx = np.flatnonzero((y == c) & (snr == snr_db))
        if len(idx) == 0:
            continue
        x = X[idx[0]]
        x = x / (np.sqrt(np.mean(x[0] ** 2 + x[1] ** 2)) + 1e-12)
        a = axes[0, c]
        a.plot(x[0], lw=0.8, color=PALETTE[0], label="I")
        a.plot(x[1], lw=0.8, color=PALETTE[1], label="Q")
        a.set_title(classes[c], fontsize=9)
        a.set_xticks([])
        a.set_yticks([])
        b = axes[1, c]
        b.scatter(x[0], x[1], s=3, color=PALETTE[2])
        b.set_aspect("equal")
        lim = np.max(np.abs(x)) * 1.1
        b.set_xlim(-lim, lim)
        b.set_ylim(-lim, lim)
        b.set_xticks([])
        b.set_yticks([])
    axes[0, 0].legend(fontsize=7, loc="upper left")
    axes[0, 0].set_ylabel("I/Q vs time", fontsize=8)
    axes[1, 0].set_ylabel("Constellation", fontsize=8)
    fig.suptitle(title or f"One example per modulation at SNR = {snr_db} dB")
    fig.savefig(path)
    plt.close(fig)

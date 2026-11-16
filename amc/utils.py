"""Small helpers shared by the scripts."""
from __future__ import annotations

import json
import os
import random

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Order in which models appear in tables and plots.
MODEL_ORDER = ["feat_dt", "feat_rf", "vtcnn2", "rescnn", "rescnn_aug"]


def results_dir(dataset: str, root: str = "results") -> str:
    path = os.path.join(root, dataset)
    os.makedirs(path, exist_ok=True)
    return path


def model_dir(dataset: str, model: str, root: str = "results") -> str:
    path = os.path.join(results_dir(dataset, root), model)
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Not JSON serializable: {type(o)}")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        pass


def save_predictions(path, y_true, y_pred, probs, snr):
    np.savez(path, y_true=y_true, y_pred=y_pred, probs=probs.astype(np.float32), snr=snr)


def sorted_models(names):
    known = [m for m in MODEL_ORDER if m in names]
    return known + sorted(n for n in names if n not in MODEL_ORDER)

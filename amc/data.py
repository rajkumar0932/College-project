"""Dataset loading, splitting and normalization.

Supports two sources, both returned as the same arrays:

* ``rml2016a``: the DeepSig RadioML 2016.10a pickle (``RML2016.10a_dict.pkl``)
* ``synthetic``: a ``.npz`` file written by ``scripts/generate_synthetic.py``

Arrays returned by :func:`load_dataset`:

    X    float32 (N, 2, 128)   I and Q rows, as stored in RadioML
    y    int64   (N,)          index into ``classes``
    snr  int64   (N,)          SNR in dB
"""
from __future__ import annotations

import os
import pickle
import warnings
from dataclasses import dataclass

import numpy as np

from . import CLASSES, DEFAULT_SEED


# --------------------------------------------------------------------------- #
# Safe loading of the RadioML pickle
# --------------------------------------------------------------------------- #
class _NumpyOnlyUnpickler(pickle.Unpickler):
    """Unpickler that only allows what a pickled dict of NumPy arrays needs.

    A normal ``pickle.load`` can execute arbitrary code from the file. This
    whitelist means a tampered file fails to load instead of running anything.
    """

    _ALLOWED = {
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy.core.numeric", "_frombuffer"),
        ("numpy._core.numeric", "_frombuffer"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
    }

    def find_class(self, module, name):
        if (module, name) in self._ALLOWED:
            # numpy>=2 renamed numpy.core -> numpy._core; route to the new name.
            if module.startswith("numpy.core") and hasattr(np, "_core"):
                module = module.replace("numpy.core", "numpy._core", 1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"Refusing to load '{module}.{name}' from the dataset pickle "
            "(only NumPy arrays are allowed)."
        )


def load_rml2016a(path: str):
    """Load RML2016.10a_dict.pkl -> (X, y, snr, classes)."""
    with open(path, "rb") as f:
        # Python 2 pickle: latin1 is needed to decode the NumPy byte buffers.
        raw = _NumpyOnlyUnpickler(f, encoding="latin1").load()
    if not isinstance(raw, dict):
        raise ValueError("Unexpected pickle content: expected a dict")

    def _name(m):
        return m.decode() if isinstance(m, bytes) else str(m)

    mods = sorted({_name(k[0]) for k in raw})
    snrs = sorted({int(k[1]) for k in raw})
    unknown = set(mods) - set(CLASSES)
    if unknown:
        raise ValueError(f"Unknown modulations in file: {unknown}")

    xs, ys, ss = [], [], []
    for (m, s), arr in raw.items():
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[1:] != (2, 128):
            raise ValueError(f"Unexpected array shape {arr.shape} for {(m, s)}")
        xs.append(arr)
        ys.append(np.full(len(arr), CLASSES.index(_name(m)), dtype=np.int64))
        ss.append(np.full(len(arr), int(s), dtype=np.int64))
    X = np.concatenate(xs)
    y = np.concatenate(ys)
    snr = np.concatenate(ss)
    print(f"[data] RML2016.10a: {X.shape[0]} examples, {len(mods)} classes, "
          f"SNR {snrs[0]}..{snrs[-1]} dB")
    return X, y, snr, list(CLASSES)


def load_npz(path: str):
    """Load a synthetic dataset written by scripts/generate_synthetic.py."""
    d = np.load(path, allow_pickle=False)
    X = d["X"].astype(np.float32)
    y = d["y"].astype(np.int64)
    snr = d["snr"].astype(np.int64)
    classes = [str(c) for c in d["classes"]]
    if classes != CLASSES:
        raise ValueError("Synthetic dataset class order does not match amc.CLASSES")
    print(f"[data] {os.path.basename(path)}: {X.shape[0]} examples, "
          f"SNR {snr.min()}..{snr.max()} dB")
    return X, y, snr, classes


DEFAULT_PATHS = {
    "rml2016a": os.path.join("data", "RML2016.10a_dict.pkl"),
    "synthetic": os.path.join("data", "synthetic_rml.npz"),
}


def load_dataset(name: str, path: str | None = None):
    path = path or DEFAULT_PATHS.get(name)
    if path is None or not os.path.exists(path):
        hint = ("Download RML2016.10a.tar.bz2 from DeepSig and extract "
                "RML2016.10a_dict.pkl into data/" if name == "rml2016a" else
                "Run: python scripts/generate_synthetic.py")
        raise FileNotFoundError(f"Dataset file not found: {path}\n{hint}")
    if name == "rml2016a":
        return load_rml2016a(path)
    if name == "synthetic":
        return load_npz(path)
    raise ValueError(f"Unknown dataset '{name}'")


# --------------------------------------------------------------------------- #
# Split and preprocessing
# --------------------------------------------------------------------------- #
@dataclass
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def stratified_split(y, snr, frac=(0.6, 0.2, 0.2), seed=DEFAULT_SEED) -> Split:
    """Split indices so each (modulation, SNR) cell is divided in the same ratio.

    Deterministic for a given seed, so every script (CNNs, feature baseline,
    evaluation) sees exactly the same test set.
    """
    assert abs(sum(frac) - 1.0) < 1e-9
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for c in np.unique(y):
        for s in np.unique(snr):
            idx = np.flatnonzero((y == c) & (snr == s))
            idx = idx[rng.permutation(len(idx))]
            n_tr = int(round(frac[0] * len(idx)))
            n_va = int(round(frac[1] * len(idx)))
            tr.append(idx[:n_tr])
            va.append(idx[n_tr:n_tr + n_va])
            te.append(idx[n_tr + n_va:])
    return Split(*(np.sort(np.concatenate(p)) for p in (tr, va, te)))


def to_model_input(X: np.ndarray, normalize: str = "rms") -> np.ndarray:
    """(N, 2, 128) -> (N, 128, 2), optionally scaled to unit average power.

    RMS scaling only removes the arbitrary receiver gain; the waveform stays raw
    I/Q, so no hand-crafted features are introduced.
    """
    Xt = np.transpose(X, (0, 2, 1)).astype(np.float32)
    if normalize == "rms":
        p = np.sqrt(np.mean(np.sum(Xt ** 2, axis=2), axis=1, keepdims=True))
        Xt = Xt / np.maximum(p, 1e-12)[:, :, None]
    elif normalize != "none":
        raise ValueError(f"Unknown normalization '{normalize}'")
    return Xt


def to_complex(X: np.ndarray) -> np.ndarray:
    """(N, 2, 128) real -> (N, 128) complex64."""
    return (X[:, 0, :] + 1j * X[:, 1, :]).astype(np.complex64)


def get_split_data(dataset: str, path: str | None = None, seed: int = DEFAULT_SEED):
    """Convenience: load + split. Returns dict with arrays for each partition."""
    X, y, snr, classes = load_dataset(dataset, path)
    sp = stratified_split(y, snr, seed=seed)
    out = {"classes": classes, "split": sp}
    for part in ("train", "val", "test"):
        idx = getattr(sp, part)
        out[part] = (X[idx], y[idx], snr[idx])
    return out

"""Experiment E1: conventional feature-based classifier (proposal Section 2.2).

24 hand-crafted features (cumulants, amplitude/phase/frequency statistics,
spectral features) -> Decision Tree and Random Forest. Uses exactly the same
train/val/test split as the CNNs.

Example:
    python scripts/train_feature_baseline.py --dataset rml2016a
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from amc import DEFAULT_SEED  # noqa: E402
from amc.data import load_dataset, stratified_split  # noqa: E402
from amc.features import FEATURE_NAMES, extract_features  # noqa: E402
from amc.metrics import summarize  # noqa: E402
from amc.plots import plot_feature_importance  # noqa: E402
from amc.utils import model_dir, results_dir, save_json, save_predictions  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="rml2016a", choices=["rml2016a", "synthetic"])
    p.add_argument("--data-path")
    p.add_argument("--trees", type=int, default=200)
    p.add_argument("--max-train", type=int, default=0, help="subsample training set (smoke test)")
    p.add_argument("--results", default="results")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    a = p.parse_args()

    X, y, snr, classes = load_dataset(a.dataset, a.data_path)
    sp = stratified_split(y, snr, seed=a.seed)
    rdir = results_dir(a.dataset, a.results)

    cache = os.path.join(rdir, "features.npz")
    if os.path.exists(cache) and np.load(cache)["F"].shape[0] == len(X):
        F = np.load(cache)["F"]
        print(f"[features] loaded cache {cache}")
    else:
        t0 = time.time()
        F = extract_features(X)
        np.savez(cache, F=F)
        print(f"[features] extracted {F.shape} in {time.time() - t0:.1f} s")

    tr, va, te = sp.train, sp.val, sp.test
    if a.max_train:
        tr = np.random.default_rng(a.seed).choice(tr, size=min(a.max_train, len(tr)), replace=False)

    # Decision Tree: choose the depth on the validation set.
    best = None
    for depth in [6, 10, 14, 18, None]:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=a.seed).fit(F[tr], y[tr])
        acc = dt.score(F[va], y[va])
        print(f"[DT] max_depth={depth}: val acc {acc:.4f}")
        if best is None or acc > best[0]:
            best = (acc, depth, dt)
    _, depth, dt = best
    print(f"[DT] chosen max_depth={depth}")

    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=a.trees, min_samples_leaf=2, n_jobs=-1,
                                random_state=a.seed).fit(F[tr], y[tr])
    rf_time = time.time() - t0
    print(f"[RF] trained {a.trees} trees in {rf_time:.1f} s, val acc {rf.score(F[va], y[va]):.4f}")

    for key, clf, extra in [("feat_dt", dt, {"max_depth": depth}),
                            ("feat_rf", rf, {"n_estimators": a.trees, "train_time_s": rf_time})]:
        out = model_dir(a.dataset, key, a.results)
        probs = clf.predict_proba(F[te])
        pred = probs.argmax(axis=1)
        save_predictions(os.path.join(out, "test_predictions.npz"), y[te], pred, probs, snr[te])
        joblib.dump(clf, os.path.join(out, "model.joblib"))
        m = summarize(y[te], pred, snr[te])
        m.update({"model": key, "n_features": F.shape[1], "n_train": int(len(tr)), **extra})
        save_json(m, os.path.join(out, "metrics.json"))
        print(f"[{key}] test acc {m['overall_acc']:.4f} | SNR>=0 dB: {m['acc_snr_ge_0']:.4f}")

    plot_feature_importance(FEATURE_NAMES, rf.feature_importances_,
                            os.path.join(model_dir(a.dataset, "feat_rf", a.results), "feature_importance.png"),
                            "Random Forest: feature importance")


if __name__ == "__main__":
    main()

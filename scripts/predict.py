"""Live demo: classify random test signals the model has never seen.

Examples:
    python scripts/predict.py                      # 10 random clips, any SNR
    python scripts/predict.py --snr 10 --n 12      # 12 clips at +10 dB
    python scripts/predict.py --plot               # also save a picture
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402

from amc import CLASSES, DEFAULT_SEED  # noqa: E402
from amc.data import load_dataset, stratified_split, to_model_input  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="rml2016a", choices=["rml2016a", "synthetic"])
    p.add_argument("--data-path")
    p.add_argument("--model", default="rescnn")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--snr", type=int, help="only pick clips at this SNR (dB)")
    p.add_argument("--seed", type=int, help="fix the random pick (default: different every run)")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--results", default="results")
    a = p.parse_args()

    mpath = os.path.join(a.results, a.dataset, a.model, "model.keras")
    if not os.path.exists(mpath):
        sys.exit(f"Model not found: {mpath}. Train it first with scripts/train_cnn.py")
    model = keras.models.load_model(mpath, compile=False)
    infer = tf.function(lambda x: model(x, training=False), jit_compile=True)

    X, y, snr, _ = load_dataset(a.dataset, a.data_path)
    te = stratified_split(y, snr, seed=DEFAULT_SEED).test       # clips never used in training
    if a.snr is not None:
        te = te[snr[te] == a.snr]
    pick = np.random.default_rng(a.seed).choice(te, size=min(a.n, len(te)), replace=False)
    xin = to_model_input(X[pick])
    infer(tf.constant(xin[:1]))                                  # warm-up

    print(f"\nModel: {a.model}  |  {len(pick)} random test clips\n")
    print(f"{'#':>2}  {'SNR':>6}  {'True type':9}  {'Predicted':9}  {'Confidence':>10}  Result")
    print("-" * 58)
    correct, times, rows = 0, [], []
    for k, i in enumerate(pick):
        t0 = time.perf_counter()
        prob = infer(tf.constant(xin[k:k + 1])).numpy()[0]
        times.append(time.perf_counter() - t0)
        j = int(prob.argmax())
        ok = j == y[i]
        correct += ok
        rows.append((i, j, prob[j], ok))
        print(f"{k + 1:>2}  {snr[i]:>+4d} dB  {CLASSES[y[i]]:9}  {CLASSES[j]:9}  "
              f"{100 * prob[j]:>9.1f}%  {'correct' if ok else 'WRONG'}")
    print("-" * 58)
    print(f"Score: {correct}/{len(pick)} correct  |  average time per decision: "
          f"{1000 * np.mean(times):.2f} ms\n")

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = len(rows)
        cols = min(n, 5)
        fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(2.6 * cols, 2.9 * np.ceil(n / cols)),
                                 squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, (i, j, conf, ok) in zip(axes.ravel(), rows):
            ax.axis("on")
            x = X[i] / (np.sqrt(np.mean(X[i] ** 2) * 2) + 1e-12)
            ax.scatter(x[0], x[1], s=4, color="#0072B2")
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"True: {CLASSES[y[i]]} ({snr[i]:+d} dB)\nSaid: {CLASSES[j]} {100 * conf:.0f}%",
                         fontsize=8.5, color="#007A3D" if ok else "#C0392B")
        fig.suptitle(f"Live demo: {correct}/{n} correct", fontsize=11)
        path = os.path.join(a.results, a.dataset, "demo_predictions.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print("Saved picture:", path)


if __name__ == "__main__":
    main()

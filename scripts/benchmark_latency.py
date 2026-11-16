"""Measure inference speed and size of every trained model (proposal Section 4.4).

* single-example latency: one clip at a time, as a live receiver would work
* batch throughput: clips per second when classifying 1024 at once

Example:
    python scripts/benchmark_latency.py --dataset rml2016a
"""
import argparse
import os
import platform
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402

from amc import DEFAULT_SEED  # noqa: E402
from amc.data import load_dataset, stratified_split, to_model_input  # noqa: E402
from amc.features import extract_features  # noqa: E402
from amc.utils import results_dir, save_json, sorted_models  # noqa: E402


def cpu_name():
    try:
        import winreg
        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    except Exception:
        return platform.processor() or platform.machine()


def stats(times_s):
    t = np.asarray(times_s) * 1000
    return float(t.mean()), float(np.percentile(t, 50)), float(np.percentile(t, 95))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="rml2016a")
    p.add_argument("--data-path")
    p.add_argument("--runs", type=int, default=300)
    p.add_argument("--results", default="results")
    a = p.parse_args()

    X, y, snr, _ = load_dataset(a.dataset, a.data_path)
    te = stratified_split(y, snr, seed=DEFAULT_SEED).test
    Xraw = X[te[:2048]]
    Xin = to_model_input(Xraw)
    rdir = results_dir(a.dataset, a.results)
    out = {}

    for m in sorted_models(os.listdir(rdir)):
        mdir = os.path.join(rdir, m)
        kpath, jpath = os.path.join(mdir, "model.keras"), os.path.join(mdir, "model.joblib")
        extra = {}
        if os.path.exists(kpath):
            model = keras.models.load_model(kpath, compile=False)
            # XLA compilation fuses the layers into one program, the normal way to
            # deploy a TF model; without it per-layer overhead dominates at batch 1.
            infer = tf.function(lambda x: model(x, training=False), jit_compile=True)
            plain = tf.function(lambda x: model(x, training=False))
            one = lambda i: infer(tf.constant(Xin[i:i + 1])).numpy()           # noqa: E731
            batch = lambda: infer(tf.constant(Xin[:1024])).numpy()             # noqa: E731
            size, params = os.path.getsize(kpath), int(model.count_params())
            for i in range(20):
                plain(tf.constant(Xin[i:i + 1])).numpy()
            tp = []
            for i in range(a.runs):
                t0 = time.perf_counter()
                plain(tf.constant(Xin[i % len(Xin):i % len(Xin) + 1])).numpy()
                tp.append(time.perf_counter() - t0)
            extra = {"single_mean_ms_without_xla": stats(tp)[0], "compiled_with_xla": True}
        elif os.path.exists(jpath):
            clf = joblib.load(jpath)
            if hasattr(clf, "n_jobs"):
                clf.n_jobs = 1          # one clip: thread start-up would dominate
            one = lambda i: clf.predict(extract_features(Xraw[i:i + 1]))       # noqa: E731
            batch = lambda: clf.predict(extract_features(Xraw[:1024]))         # noqa: E731
            size, params = os.path.getsize(jpath), None
        else:
            continue

        for i in range(20):             # warm-up (graph tracing, caches)
            one(i)
        batch()
        t1 = []
        for i in range(a.runs):
            t0 = time.perf_counter()
            one(i % len(Xin))
            t1.append(time.perf_counter() - t0)
        tb = []
        for _ in range(5):
            t0 = time.perf_counter()
            batch()
            tb.append(time.perf_counter() - t0)
        mean, p50, p95 = stats(t1)
        out[m] = {"single_mean_ms": mean, "single_p50_ms": p50, "single_p95_ms": p95,
                  "batch_throughput": 1024 / float(np.median(tb)),
                  "size_mb": size / 1e6, "params": params, **extra}
        print(f"[latency] {m:10s} one clip: {mean:6.2f} ms (p95 {p95:6.2f}) | "
              f"batch: {out[m]['batch_throughput']:9,.0f} clips/s | {size / 1e6:6.2f} MB")

    info = {"cpu": cpu_name(), "tf_version": tf.__version__, "python": platform.python_version(),
            "gpu": bool(tf.config.list_physical_devices("GPU"))}
    save_json({"system": info, "models": out}, os.path.join(rdir, "latency.json"))
    print(f"[latency] wrote {os.path.join(rdir, 'latency.json')} ({info['cpu']})")


if __name__ == "__main__":
    main()

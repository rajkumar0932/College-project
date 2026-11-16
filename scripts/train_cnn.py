"""Experiments E2-E4: train a CNN on raw I/Q.

Examples:
    python scripts/train_cnn.py --dataset rml2016a --model vtcnn2
    python scripts/train_cnn.py --dataset rml2016a --model rescnn
    python scripts/train_cnn.py --dataset rml2016a --model rescnn --augment
    # quick smoke test
    python scripts/train_cnn.py --dataset synthetic --model rescnn --epochs 2 --max-train 5000
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

from amc import DEFAULT_SEED  # noqa: E402
from amc.data import load_dataset, stratified_split, to_model_input  # noqa: E402
from amc.metrics import summarize  # noqa: E402
from amc.models import build_model, display_name, random_rotation  # noqa: E402
from amc.plots import plot_history  # noqa: E402
from amc.utils import model_dir, results_dir, save_json, save_predictions, set_seed  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="rml2016a", choices=["rml2016a", "synthetic"])
    p.add_argument("--data-path")
    p.add_argument("--model", default="rescnn", choices=["vtcnn2", "rescnn"])
    p.add_argument("--augment", action="store_true", help="random phase-rotation augmentation")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--max-train", type=int, default=0, help="subsample training set (smoke test)")
    p.add_argument("--tag", help="results sub-folder name (default: model name)")
    p.add_argument("--results", default="results")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    a = p.parse_args()

    set_seed(a.seed)
    run = a.tag or (a.model + ("_aug" if a.augment else ""))
    out = model_dir(a.dataset, run, a.results)

    X, y, snr, classes = load_dataset(a.dataset, a.data_path)
    sp = stratified_split(y, snr, seed=a.seed)
    np.savez(os.path.join(results_dir(a.dataset, a.results), "split.npz"),
             train=sp.train, val=sp.val, test=sp.test)
    tr = sp.train
    if a.max_train:
        tr = np.random.default_rng(a.seed).choice(tr, size=min(a.max_train, len(tr)), replace=False)

    Xtr, ytr = to_model_input(X[tr]), y[tr]
    Xva, yva = to_model_input(X[sp.val]), y[sp.val]
    Xte, yte, ste = to_model_input(X[sp.test]), y[sp.test], snr[sp.test]
    del X
    print(f"[train] {run}: train {len(Xtr)}, val {len(Xva)}, test {len(Xte)}")

    ds = tf.data.Dataset.from_tensor_slices((Xtr, ytr))
    ds = ds.shuffle(len(Xtr), seed=a.seed, reshuffle_each_iteration=True).batch(a.batch_size)
    if a.augment:
        ds = ds.map(random_rotation, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    model = build_model(a.model, n_classes=len(classes))
    model.compile(optimizer=keras.optimizers.Adam(a.lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    lines = []
    model.summary(print_fn=lambda s, **k: lines.append(s), line_length=100)
    with open(os.path.join(out, "model_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    n_params = int(model.count_params())
    print(f"[train] {model.name}: {n_params:,} parameters (layer table in model_summary.txt)")

    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=a.patience,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=max(2, a.patience // 2), min_lr=1e-5, verbose=1),
        keras.callbacks.CSVLogger(os.path.join(out, "training_log.csv")),
    ]
    t0 = time.time()
    hist = model.fit(ds, validation_data=(Xva, yva), epochs=a.epochs, shuffle=False,
                     validation_batch_size=1024, callbacks=cbs, verbose=2)
    train_time = time.time() - t0

    model.save(os.path.join(out, "model.keras"))
    h = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    save_json(h, os.path.join(out, "history.json"))
    plot_history(h, os.path.join(out, "training_curves.png"), f"{display_name(run)}: training curves")

    probs = model.predict(Xte, batch_size=1024, verbose=0)
    pred = probs.argmax(axis=1)
    save_predictions(os.path.join(out, "test_predictions.npz"), yte, pred, probs, ste)
    m = summarize(yte, pred, ste)
    m.update({
        "model": run, "architecture": a.model, "params": n_params, "augment": a.augment,
        "epochs_run": len(h["loss"]), "best_epoch": int(np.argmin(h["val_loss"]) + 1),
        "train_time_s": round(train_time, 1), "n_train": int(len(Xtr)),
        "batch_size": a.batch_size, "lr": a.lr, "seed": a.seed,
        "tf_version": tf.__version__,
        "gpu": bool(tf.config.list_physical_devices("GPU")),
    })
    save_json(m, os.path.join(out, "metrics.json"))
    print(f"[done] {run}: test acc {m['overall_acc']:.4f} | SNR>=0 dB {m['acc_snr_ge_0']:.4f} | "
          f"peak {m['peak_acc']:.4f} @ {m['peak_snr']} dB | {train_time / 60:.1f} min")


if __name__ == "__main__":
    main()

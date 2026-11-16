"""Compare every trained model and write figures, CSVs and REPORT.md.

Reads results/<dataset>/<model>/test_predictions.npz for each model, plus
results/<dataset>/latency.json if the benchmark has been run.

Example:
    python scripts/evaluate.py --dataset rml2016a
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from amc import CLASSES  # noqa: E402
from amc.metrics import accuracy_by_snr, confusion, summarize, top_confusions  # noqa: E402
from amc.plots import plot_acc_vs_snr, plot_confusion, plot_per_class_acc  # noqa: E402
from amc.utils import load_json, results_dir, sorted_models  # noqa: E402

NAMES = {
    "feat_dt": "Traditional: features + Decision Tree",
    "feat_rf": "Traditional: features + Random Forest",
    "vtcnn2": "VT-CNN2 (published CNN, 2016)",
    "rescnn": "Our CNN (ResCNN-1D)",
    "rescnn_aug": "Our CNN + rotation augmentation",
}

WHY = {
    ("QAM16", "QAM64"): "dense constellations: the inner QAM64 points sit almost on top of QAM16's points",
    ("QAM64", "QAM16"): "dense constellations: the inner QAM64 points sit almost on top of QAM16's points",
    ("WBFM", "AM-DSB"): "during pauses in the speech both signals reduce to a plain carrier",
    ("AM-DSB", "WBFM"): "during pauses in the speech both signals reduce to a plain carrier",
    ("8PSK", "QPSK"): "QPSK's 4 phase points are a subset of 8PSK's 8 points",
    ("QPSK", "8PSK"): "QPSK's 4 phase points are a subset of 8PSK's 8 points",
}


def name(m):
    return NAMES.get(m, m)


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f} %"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="rml2016a")
    p.add_argument("--results", default="results")
    a = p.parse_args()

    rdir = results_dir(a.dataset, a.results)
    models = sorted_models([d for d in os.listdir(rdir)
                            if os.path.exists(os.path.join(rdir, d, "test_predictions.npz"))])
    if not models:
        sys.exit(f"No trained models found in {rdir}")
    print("[evaluate] models:", ", ".join(models))

    R = {}
    for m in models:
        d = np.load(os.path.join(rdir, m, "test_predictions.npz"))
        mpath = os.path.join(rdir, m, "metrics.json")
        R[m] = {"y": d["y_true"], "p": d["y_pred"], "snr": d["snr"],
                "meta": load_json(mpath) if os.path.exists(mpath) else {}}
        R[m]["sum"] = summarize(R[m]["y"], R[m]["p"], R[m]["snr"])
        R[m]["snrs"], R[m]["acc"] = accuracy_by_snr(R[m]["y"], R[m]["p"], R[m]["snr"])

    snrs = R[models[0]]["snrs"]
    lat_path = os.path.join(rdir, "latency.json")
    lat = load_json(lat_path)["models"] if os.path.exists(lat_path) else {}

    # 1) Accuracy vs SNR (all models) + CSV
    plot_acc_vs_snr({name(m): (R[m]["snrs"], R[m]["acc"]) for m in models},
                    os.path.join(rdir, "accuracy_vs_snr.png"))
    pd.DataFrame({name(m): np.round(R[m]["acc"] * 100, 2) for m in models},
                 index=pd.Index(snrs, name="SNR_dB")).to_csv(os.path.join(rdir, "accuracy_vs_snr.csv"))

    # 2) Confusion matrices per model
    pairs = {}
    for m in models:
        y, pr, s = R[m]["y"], R[m]["p"], R[m]["snr"]
        plot_confusion(confusion(y, pr, len(CLASSES)), CLASSES,
                       os.path.join(rdir, m, "confusion_all_snr.png"), f"{name(m)}: all SNRs")
        hi = s >= 0
        cm_hi = confusion(y[hi], pr[hi], len(CLASSES))
        plot_confusion(cm_hi, CLASSES, os.path.join(rdir, m, "confusion_snr_ge_0.png"),
                       f"{name(m)}: SNR ≥ 0 dB")
        pairs[m] = top_confusions(cm_hi, CLASSES, k=6)

    # 3) Extra views for the best model (highest accuracy at SNR >= 0 dB)
    best = max(models, key=lambda m: R[m]["sum"]["acc_snr_ge_0"] or 0)
    y, pr, s = R[best]["y"], R[best]["p"], R[best]["snr"]
    for snr_db in (int(snrs.max()), 0, -10):
        if (s == snr_db).any():
            k = s == snr_db
            plot_confusion(confusion(y[k], pr[k], len(CLASSES)), CLASSES,
                           os.path.join(rdir, f"best_confusion_{snr_db}dB.png"),
                           f"{name(best)}: SNR = {snr_db} dB")
    per_class = {c: [np.mean(pr[(y == i) & (s == v)] == i) for v in snrs] for i, c in enumerate(CLASSES)}
    plot_per_class_acc(per_class, snrs, os.path.join(rdir, "best_per_class_accuracy.png"),
                       f"{name(best)}: accuracy of each signal type vs SNR")

    # 4) Summary table
    rows = []
    for m in models:
        sm, meta, lt = R[m]["sum"], R[m]["meta"], lat.get(m, {})
        rows.append({
            "Method": name(m),
            "Parameters": meta.get("params", "-"),
            "Overall": pct(sm["overall_acc"]),
            "SNR ≥ 0 dB": pct(sm["acc_snr_ge_0"]),
            "SNR < 0 dB": pct(sm["acc_snr_lt_0"]),
            "At -10 dB": pct(sm["acc_at"].get(-10)),
            "At 0 dB": pct(sm["acc_at"].get(0)),
            "At +18 dB": pct(sm["acc_at"].get(18)),
            "Best": f"{pct(sm['peak_acc'])} @ {sm['peak_snr']} dB",
            "ms / decision": f"{lt['single_mean_ms']:.2f}" if lt else "-",
        })
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(rdir, "summary.csv"), index=False)

    # 5) Success criteria (only those we have data for)
    checks = []
    feats = [m for m in models if m.startswith("feat_")]
    if "rescnn" in R and feats:
        trad = np.max([R[f]["acc"] for f in feats], axis=0)
        rng = (snrs >= -10)
        wins = int(np.sum(R["rescnn"]["acc"][rng] > trad[rng]))
        checks.append(("Our CNN beats the traditional method at every SNR from -10 to +18 dB",
                       wins == int(rng.sum()), f"wins at {wins} of {int(rng.sum())} SNR levels"))
    if "rescnn" in R and "vtcnn2" in R:
        ours, ref = R["rescnn"]["sum"]["acc_snr_ge_0"], R["vtcnn2"]["sum"]["acc_snr_ge_0"]
        pr_ours, pr_ref = R["rescnn"]["meta"].get("params"), R["vtcnn2"]["meta"].get("params")
        ratio = (pr_ref / pr_ours) if pr_ours and pr_ref else None
        ok = ours >= ref - 0.005 and (ratio is None or ratio >= 10)
        checks.append(("Our CNN is at least as accurate as VT-CNN2 (SNR ≥ 0 dB) with ≥10× fewer parameters",
                       ok, f"{pct(ours)} vs {pct(ref)}; " + (f"{ratio:.1f}× fewer parameters" if ratio else "")))
    if "rescnn" in R:
        v = R["rescnn"]["sum"]["acc_snr_ge_0"]
        checks.append(("Our CNN reaches ≥ 85 % for SNR ≥ 0 dB", v >= 0.85, pct(v)))
    if "rescnn" in lat:
        v = lat["rescnn"]["single_mean_ms"]
        checks.append(("One decision takes < 5 ms on a laptop CPU", v < 5, f"{v:.2f} ms"))

    # 6) REPORT.md (plain language)
    L = []
    L.append(f"# Results: {a.dataset}\n")
    if a.dataset == "synthetic":
        L.append("> **Note:** these results come from our own generated test signals. They are only "
                 "a check that the code works end to end. The final results for the report must come "
                 "from RadioML 2016.10a.\n")
    L.append("All methods were tested on exactly the same test clips, which none of them saw during "
             f"training ({len(R[models[0]]['y']):,} clips, spread evenly across all 11 signal types "
             "and all 20 noise levels).\n")
    L.append("## 1. Main result: accuracy vs noise level\n")
    L.append("![Accuracy vs SNR](accuracy_vs_snr.png)\n")
    L.append("How to read it: left is very noisy, right is clean. Higher is better. The dotted line "
             "is pure guessing (1 in 11 = 9.1 %).\n")
    L.append("## 2. Summary table\n")
    L.append("| " + " | ".join(table.columns) + " |")
    L.append("|" + "---|" * len(table.columns))
    for _, r in table.iterrows():
        L.append("| " + " | ".join(f"{v:,}" if isinstance(v, int) else str(v) for v in r) + " |")
    L.append("")
    if checks:
        L.append("## 3. Did we meet our targets?\n")
        L.append("| Target | Met? | Measured |\n|---|---|---|")
        for t, ok, detail in checks:
            L.append(f"| {t} | {'Yes' if ok else '**No**'} | {detail} |")
        L.append("")
    L.append(f"## 4. Which signals get confused? ({name(best)}, SNR ≥ 0 dB)\n")
    L.append(f"![Confusion matrix]({best}/confusion_snr_ge_0.png)\n")
    L.append("Each row is the true signal type; each column is what the model said. A perfect model "
             "would only have numbers on the diagonal. Biggest mix-ups:\n")
    L.append("| True type | Mistaken for | How often | Likely reason |\n|---|---|---|---|")
    for t, q, v in pairs[best]:
        L.append(f"| {t} | {q} | {100 * v:.1f} % | {WHY.get((t, q), '-')} |")
    L.append("")
    L.append("## 5. More figures\n")
    L.append(f"- Accuracy of each signal type vs SNR (best model): `best_per_class_accuracy.png`")
    L.append(f"- Confusion at the cleanest, 0 dB and -10 dB levels: `best_confusion_*dB.png`")
    L.append("- Per-model confusion matrices: `<model>/confusion_all_snr.png`, `<model>/confusion_snr_ge_0.png`")
    L.append("- Training curves for the CNNs: `<model>/training_curves.png`")
    L.append("- Feature importance for the Random Forest: `feat_rf/feature_importance.png`")
    L.append("- Example signals of each type: `dataset_examples_*dB.png`\n")
    if lat:
        L.append("## 6. Speed and size\n")
        L.append("| Method | Parameters | Model file | One decision (mean) | One decision (95th pct) | Throughput |")
        L.append("|---|---|---|---|---|---|")
        for m in models:
            if m in lat:
                x = lat[m]
                L.append(f"| {name(m)} | {x.get('params', '-')} | {x['size_mb']:.2f} MB | "
                         f"{x['single_mean_ms']:.2f} ms | {x['single_p95_ms']:.2f} ms | "
                         f"{x['batch_throughput']:,.0f} clips/s |")
        info = load_json(lat_path).get("system", {})
        L.append(f"\nMeasured on: {info.get('cpu', 'unknown CPU')}, TensorFlow {info.get('tf_version', '?')}, "
                 "no GPU. CNN times use XLA compilation (the normal way to deploy a TensorFlow model); "
                 "traditional-method times include computing the 24 features.\n")

    with open(os.path.join(rdir, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(table.to_string(index=False))
    for t, ok, detail in checks:
        print(f"[{'MET' if ok else 'NOT MET'}] {t}: {detail}")
    print(f"[evaluate] wrote {os.path.join(rdir, 'REPORT.md')}")


if __name__ == "__main__":
    main()

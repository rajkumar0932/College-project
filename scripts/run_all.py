"""Run the whole project in one go.

    python scripts/run_all.py --dataset rml2016a           # the real experiment
    python scripts/run_all.py --dataset synthetic --quick  # 2-minute code check

Steps: example plots -> traditional baseline -> our CNN -> VT-CNN2
       -> speed test -> comparison report.
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def run(step, script, *args):
    print(f"\n{'=' * 70}\n>>> {step}\n{'=' * 70}", flush=True)
    t0 = time.time()
    subprocess.run([sys.executable, os.path.join(HERE, script), *args], check=True, cwd=ROOT)
    print(f">>> {step} finished in {(time.time() - t0) / 60:.1f} min", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="rml2016a", choices=["rml2016a", "synthetic"])
    p.add_argument("--data-path")
    p.add_argument("--quick", action="store_true", help="tiny run to check the code works")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--skip-vtcnn2", action="store_true", help="skip the slower VT-CNN2 baseline")
    p.add_argument("--results", help="results folder (default: results, or results_quick with --quick)")
    a = p.parse_args()

    results = a.results or ("results_quick" if a.quick else "results")
    common = ["--dataset", a.dataset, "--results", results]
    data = ["--data-path", a.data_path] if a.data_path else []
    small = ["--max-train", "4000"] if a.quick else []
    cnn = ["--epochs", "2" if a.quick else str(a.epochs), "--batch-size", str(a.batch_size)]

    if a.dataset == "synthetic" and not a.data_path and not os.path.exists(
            os.path.join(ROOT, "data", "synthetic_rml.npz")):
        run("Generate synthetic dataset", "generate_synthetic.py",
            *(["--n-per", "200"] if a.quick else []))

    t0 = time.time()
    run("1/6 Example signal plots", "visualize_dataset.py", *common, *data)
    run("2/6 Traditional method (features + DT / RF)", "train_feature_baseline.py",
        *common, *data, *small, *(["--trees", "50"] if a.quick else []))
    run("3/6 Our CNN (ResCNN-1D)", "train_cnn.py", *common, *data, *small, *cnn, "--model", "rescnn")
    if not a.skip_vtcnn2:
        run("4/6 VT-CNN2 (published baseline CNN, slower)", "train_cnn.py", *common, *data, *small, *cnn,
            "--model", "vtcnn2")
    run("5/6 Speed test", "benchmark_latency.py", *common, *data, *(["--runs", "50"] if a.quick else []))
    run("6/6 Comparison report", "evaluate.py", "--dataset", a.dataset, "--results", results)
    print(f"\nAll done in {(time.time() - t0) / 60:.1f} min. "
          f"Open {os.path.join(results, a.dataset, 'REPORT.md')}")


if __name__ == "__main__":
    main()

"""Plot one example per modulation (waveform + constellation) at chosen SNRs.

Example:
    python scripts/visualize_dataset.py --dataset rml2016a
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amc.data import load_dataset  # noqa: E402
from amc.plots import plot_examples  # noqa: E402
from amc.utils import results_dir  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="rml2016a", choices=["rml2016a", "synthetic"])
    p.add_argument("--data-path")
    p.add_argument("--snrs", type=int, nargs="+", default=[18, 0, -10])
    p.add_argument("--results", default="results")
    a = p.parse_args()

    X, y, snr, classes = load_dataset(a.dataset, a.data_path)
    out = results_dir(a.dataset, a.results)
    for s in a.snrs:
        path = os.path.join(out, f"dataset_examples_{s}dB.png")
        plot_examples(X, y, snr, classes, path, snr_db=s)
        print("wrote", path)


if __name__ == "__main__":
    main()

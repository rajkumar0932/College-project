"""Generate a RadioML-style synthetic dataset (same format as RML2016.10a).

Example:
    python scripts/generate_synthetic.py --n-per 1000 --out data/synthetic_rml.npz
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from amc import CLASSES  # noqa: E402
from amc.synth import generate  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-per", type=int, default=1000, help="examples per (modulation, SNR)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default=os.path.join("data", "synthetic_rml.npz"))
    a = p.parse_args()

    t0 = time.time()
    print(f"Generating {a.n_per} examples x 11 modulations x 20 SNRs ...")
    X, y, snr = generate(n_per=a.n_per, seed=a.seed)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez(a.out, X=X, y=y, snr=snr, classes=np.array(CLASSES))
    print(f"Saved {X.shape} to {a.out} in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()

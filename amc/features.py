"""Hand-crafted features for the conventional feature-based (FB) baseline.

This is the approach Section 2.2 of the proposal criticises: an expert picks
statistics of the signal (higher-order cumulants, instantaneous amplitude,
phase and frequency), then a simple classifier works on those numbers.

References for the feature set:
  * A. Swami, B. Sadler, "Hierarchical digital modulation classification
    using cumulants", IEEE Trans. Commun., 2000.
  * E. Azzouz, A. Nandi, "Automatic identification of digital modulation
    types", Signal Processing, 1995 (gamma_max, sigma_ap, sigma_dp, sigma_aa, sigma_af).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import detrend

FEATURE_NAMES = [
    "|C20|", "|C40|", "|C41|", "|C42|", "|C60|", "|C61|", "|C62|", "|C63|",
    "gamma_max", "sigma_aa", "sigma_a", "kurt_a",
    "sigma_ap", "sigma_dp", "sigma_af", "sigma_f",
    "|mean x|", "peak_psd_x", "peak_psd_x2", "peak_psd_x4",
    "spectral_flatness", "papr", "kurt_I", "kurt_Q",
]


def _masked_std(v, mask):
    cnt = np.maximum(mask.sum(axis=1), 1)
    m = np.sum(v * mask, axis=1) / cnt
    m2 = np.sum(v ** 2 * mask, axis=1) / cnt
    return np.sqrt(np.maximum(m2 - m ** 2, 0.0))


def _features_chunk(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.complex128)
    L = x.shape[1]
    x = x / (np.sqrt(np.mean(np.abs(x) ** 2, axis=1, keepdims=True)) + 1e-12)
    xc = np.conj(x)

    def M(p, q):
        return np.mean(x ** (p - q) * xc ** q, axis=1)

    M20, M21, M22 = M(2, 0), M(2, 1), M(2, 2)
    M40, M41, M42, M43 = M(4, 0), M(4, 1), M(4, 2), M(4, 3)
    M60, M61, M62, M63 = M(6, 0), M(6, 1), M(6, 2), M(6, 3)
    C20, C21 = M20, M21
    C40 = M40 - 3 * M20 ** 2
    C41 = M41 - 3 * M20 * M21
    C42 = M42 - np.abs(M20) ** 2 - 2 * M21 ** 2
    C60 = M60 - 15 * M20 * M40 + 30 * M20 ** 3
    C61 = M61 - 5 * M21 * M40 - 10 * M20 * M41 + 30 * M20 ** 2 * M21
    C62 = (M62 - 6 * M20 * M42 - 8 * M21 * M41 - M22 * M40
           + 6 * M20 ** 2 * M22 + 24 * M21 ** 2 * M20)
    C63 = (M63 - 9 * M21 * M42 + 12 * M21 ** 3 - 3 * M20 * M43
           - 3 * M22 * M41 + 18 * M20 * M21 * M22)
    c21 = np.abs(C21) + 1e-12

    # Instantaneous amplitude
    a = np.abs(x)
    a_n = a / (a.mean(axis=1, keepdims=True) + 1e-12)
    a_cn = a_n - 1.0
    gamma_max = np.max(np.abs(np.fft.fft(a_cn, axis=1)) ** 2, axis=1) / L
    sigma_aa = np.std(np.abs(a_cn), axis=1)
    sigma_a = np.std(a_cn, axis=1)
    kurt_a = np.mean(a_cn ** 4, axis=1) / (np.mean(a_cn ** 2, axis=1) ** 2 + 1e-12)

    # Instantaneous phase (linear trend removed = carrier offset removed),
    # evaluated only on non-weak samples (normalized amplitude above 1).
    phi = detrend(np.unwrap(np.angle(x), axis=1), axis=1, type="linear")
    strong = (a_n > 1.0).astype(float)
    sigma_ap = _masked_std(np.abs(phi), strong)
    sigma_dp = _masked_std(phi, strong)

    # Instantaneous frequency
    f = np.diff(np.unwrap(np.angle(x), axis=1), axis=1) / (2 * np.pi)
    f_c = f - f.mean(axis=1, keepdims=True)
    sigma_af = _masked_std(np.abs(f_c), strong[:, 1:])
    sigma_f = np.std(f_c, axis=1)

    # Spectral features
    def peak_psd(v):
        return np.max(np.abs(np.fft.fft(v, axis=1)) ** 2, axis=1) / L ** 2

    psd = np.abs(np.fft.fft(x, axis=1)) ** 2 + 1e-12
    flatness = np.exp(np.mean(np.log(psd), axis=1)) / np.mean(psd, axis=1)
    papr = np.max(np.abs(x) ** 2, axis=1)

    I, Q = x.real, x.imag
    kurt_I = np.mean(I ** 4, axis=1) / (np.mean(I ** 2, axis=1) ** 2 + 1e-12)
    kurt_Q = np.mean(Q ** 4, axis=1) / (np.mean(Q ** 2, axis=1) ** 2 + 1e-12)

    feats = np.stack([
        np.abs(C20), np.abs(C40) / c21 ** 2, np.abs(C41) / c21 ** 2, np.abs(C42) / c21 ** 2,
        np.abs(C60) / c21 ** 3, np.abs(C61) / c21 ** 3, np.abs(C62) / c21 ** 3, np.abs(C63) / c21 ** 3,
        gamma_max, sigma_aa, sigma_a, kurt_a,
        sigma_ap, sigma_dp, sigma_af, sigma_f,
        np.abs(x.mean(axis=1)), peak_psd(x), peak_psd(x ** 2), peak_psd(x ** 4),
        flatness, papr, kurt_I, kurt_Q,
    ], axis=1)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def extract_features(X: np.ndarray, chunk: int = 20000) -> np.ndarray:
    """X: (N, 2, 128) real I/Q -> (N, n_features) float32."""
    xc = X[:, 0, :] + 1j * X[:, 1, :]
    return np.concatenate([_features_chunk(xc[i:i + chunk])
                           for i in range(0, len(xc), chunk)])

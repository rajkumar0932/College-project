"""RadioML-style synthetic signal generator.

Produces examples with the same format as RadioML 2016.10a: 11 modulations,
2 x 128 I/Q samples, 8 samples per symbol, SNR -20..18 dB. The channel model
follows the same impairments the RadioML paper describes:

    multipath (Rician) fading -> sample-rate offset + random timing
    -> carrier-frequency offset + random phase -> AWGN at the target SNR

Uses:
  * run and test the full pipeline when the real dataset is not available
  * a "synthetic gap" experiment (train on one generator, test on another)

Note: this is NOT a copy of RadioML. The analog sources are synthetic
audio-like messages, not the real speech recording DeepSig used, so the
absolute accuracies differ from RML2016.10a. Report RML2016.10a numbers as
the main results.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve, firwin, hilbert

from . import CLASSES

SPS = 8          # samples per symbol (as in RML2016.10a)
N_OUT = 128      # samples per example
L_STREAM = 512   # length generated before windowing (avoids filter edge effects)


# --------------------------------------------------------------------------- #
# Pulse shapes and constellations
# --------------------------------------------------------------------------- #
def rrc_taps(beta: float = 0.35, sps: int = SPS, span: int = 8) -> np.ndarray:
    """Root-raised-cosine filter taps, unit energy."""
    t = np.arange(-span * sps // 2, span * sps // 2 + 1) / sps
    h = np.empty_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-9:
            h[i] = 1 - beta + 4 * beta / np.pi
        elif abs(abs(4 * beta * ti) - 1) < 1e-9:
            h[i] = beta / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                                       + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            h[i] = (np.sin(np.pi * ti * (1 - beta))
                    + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))) / (
                np.pi * ti * (1 - (4 * beta * ti) ** 2))
    return h / np.sqrt(np.sum(h ** 2))


def gaussian_taps(bt: float = 0.35, sps: int = SPS, span: int = 4) -> np.ndarray:
    """Gaussian frequency-pulse filter for GFSK, unit DC gain."""
    t = np.arange(-span * sps // 2, span * sps // 2 + 1) / sps
    sigma = np.sqrt(np.log(2)) / (2 * np.pi * bt)
    h = np.exp(-t ** 2 / (2 * sigma ** 2))
    return h / h.sum()


def _qam(m: int) -> np.ndarray:
    k = int(np.sqrt(m))
    lv = np.arange(-(k - 1), k, 2, dtype=float)
    pts = (lv[:, None] + 1j * lv[None, :]).ravel()
    return pts / np.sqrt(np.mean(np.abs(pts) ** 2))


CONSTELLATIONS = {
    "BPSK": np.array([-1.0, 1.0], dtype=complex),
    "QPSK": np.exp(1j * (np.pi / 4 + np.arange(4) * np.pi / 2)),
    "8PSK": np.exp(1j * 2 * np.pi * np.arange(8) / 8),
    "PAM4": np.array([-3.0, -1.0, 1.0, 3.0], dtype=complex) / np.sqrt(5),
    "QAM16": _qam(16),
    "QAM64": _qam(64),
}


# --------------------------------------------------------------------------- #
# Modulators (all vectorized over a batch of B examples)
# --------------------------------------------------------------------------- #
def _linear(points, B, L, rng):
    span = 8
    nsym = L // SPS + span + 2
    sym = points[rng.integers(0, len(points), size=(B, nsym))]
    up = np.zeros((B, nsym * SPS), dtype=complex)
    up[:, ::SPS] = sym
    x = fftconvolve(up, rrc_taps(span=span)[None, :], mode="same", axes=1)
    start = (span // 2 + 1) * SPS
    return x[:, start:start + L]


def _fsk(B, L, rng, h, gaussian):
    nsym = L // SPS + 8
    bits = rng.choice([-1.0, 1.0], size=(B, nsym))
    freq = np.repeat(bits, SPS, axis=1)
    if gaussian:
        freq = fftconvolve(freq, gaussian_taps()[None, :], mode="same", axes=1)
    phase = np.pi * h * np.cumsum(freq, axis=1) / SPS
    phase += rng.uniform(0, 2 * np.pi, size=(B, 1))
    x = np.exp(1j * phase)
    return x[:, 4 * SPS:4 * SPS + L]


def _audio_like(B, L, rng, p_silence=0.2):
    """Band-limited, speech-like message in [-1, 1].

    A mix of a few tones and low-pass noise with a slow syllabic envelope.
    Some examples are near-silent, like pauses in the speech RadioML used.
    """
    n = np.arange(L)[None, :]
    K = 4
    f = rng.uniform(0.002, 0.03, size=(B, K, 1))
    a = rng.uniform(0.2, 1.0, size=(B, K, 1))
    ph = rng.uniform(0, 2 * np.pi, size=(B, K, 1))
    m = np.sum(a * np.sin(2 * np.pi * f * n[:, None, :] + ph), axis=1)
    lp = firwin(65, 0.06)
    noise = fftconvolve(rng.standard_normal((B, L + 64)), lp[None, :], mode="same", axes=1)
    noise = noise[:, 32:32 + L]
    noise /= noise.std(axis=1, keepdims=True) + 1e-12
    m = m / (m.std(axis=1, keepdims=True) + 1e-12) + 0.5 * noise
    f_env = rng.uniform(0.0005, 0.003, size=(B, 1))
    env = 0.6 + 0.4 * np.sin(2 * np.pi * f_env * n + rng.uniform(0, 2 * np.pi, (B, 1)))
    m *= env
    m /= np.max(np.abs(m), axis=1, keepdims=True) + 1e-12
    silent = rng.random(B) < p_silence
    m[silent] *= 0.02
    return m


def modulate(mod: str, B: int, L: int, rng: np.random.Generator) -> np.ndarray:
    """Return B complex baseband streams of length L for one modulation."""
    if mod in CONSTELLATIONS:
        return _linear(CONSTELLATIONS[mod], B, L, rng)
    if mod == "CPFSK":
        return _fsk(B, L, rng, h=0.5, gaussian=False)
    if mod == "GFSK":
        return _fsk(B, L, rng, h=1.0, gaussian=True)
    m = _audio_like(B, L, rng)
    if mod == "AM-DSB":
        mu = rng.uniform(0.5, 0.9, size=(B, 1))
        return (1.0 + mu * m).astype(complex)
    if mod == "AM-SSB":
        return hilbert(m, axis=1)            # upper sideband, suppressed carrier
    if mod == "WBFM":
        dev = rng.uniform(0.05, 0.12, size=(B, 1))   # peak deviation, cycles/sample
        return np.exp(1j * 2 * np.pi * np.cumsum(dev * m, axis=1))
    raise ValueError(f"Unknown modulation {mod}")


# --------------------------------------------------------------------------- #
# Channel
# --------------------------------------------------------------------------- #
def channel(x: np.ndarray, snr_db: float, rng: np.random.Generator,
            n_out: int = N_OUT) -> np.ndarray:
    """Apply fading, timing/SRO, CFO, phase and AWGN; return (B, n_out) complex."""
    B, L = x.shape

    # 1) Multipath: 4-tap exponential power-delay profile, Rician first tap.
    n_taps = 4
    tau = rng.uniform(0.3, 1.5, size=(B, 1))
    p = np.exp(-np.arange(n_taps)[None, :] / tau)
    p /= p.sum(axis=1, keepdims=True)
    K = rng.uniform(0.0, 10.0, size=B)                         # Rician K-factor
    nlos = (rng.standard_normal((B, n_taps)) + 1j * rng.standard_normal((B, n_taps))) / np.sqrt(2)
    taps = np.sqrt(p) * nlos
    los = np.exp(1j * rng.uniform(0, 2 * np.pi, B))
    taps[:, 0] = np.sqrt(p[:, 0]) * (np.sqrt(K / (K + 1)) * los + np.sqrt(1 / (K + 1)) * nlos[:, 0])
    taps /= np.sqrt(np.sum(np.abs(taps) ** 2, axis=1, keepdims=True))
    y = np.zeros_like(x)
    for d in range(n_taps):
        y[:, d:] += taps[:, d:d + 1] * x[:, :L - d]

    # 2) Sample-rate offset + random fractional timing (linear interpolation).
    sro = rng.normal(0.0, 1e-4, size=(B, 1))
    margin = 32
    t0 = rng.uniform(margin, L - margin - n_out * 1.01, size=(B, 1))
    t = t0 + np.arange(n_out)[None, :] * (1.0 + sro)
    i0 = np.floor(t).astype(int)
    frac = t - i0
    rows = np.arange(B)[:, None]
    y = y[rows, i0] * (1 - frac) + y[rows, i0 + 1] * frac

    # 3) Carrier-frequency offset (random walk around a random offset) + phase.
    cfo = np.clip(rng.normal(0.0, 7e-4, size=(B, 1)), -2.5e-3, 2.5e-3)
    walk = np.cumsum(rng.normal(0.0, 1e-5, size=(B, n_out)), axis=1)
    phase = 2 * np.pi * np.cumsum(cfo + walk, axis=1) + rng.uniform(0, 2 * np.pi, (B, 1))
    y = y * np.exp(1j * phase)

    # 4) Unit signal power, then AWGN at the requested SNR.
    y /= np.sqrt(np.mean(np.abs(y) ** 2, axis=1, keepdims=True)) + 1e-12
    sigma = np.sqrt(10 ** (-snr_db / 10) / 2)
    y = y + sigma * (rng.standard_normal(y.shape) + 1j * rng.standard_normal(y.shape))
    return y


def generate(n_per: int = 1000, snrs=range(-20, 20, 2), mods=CLASSES,
             seed: int = 7, verbose: bool = True):
    """Generate a full dataset. Returns X (N,2,128) float32, y, snr."""
    rng = np.random.default_rng(seed)
    X, Y, S = [], [], []
    for mod in mods:
        ci = CLASSES.index(mod)
        for snr in snrs:
            x = modulate(mod, n_per, L_STREAM, rng)
            z = channel(x, snr, rng)
            X.append(np.stack([z.real, z.imag], axis=1).astype(np.float32))
            Y.append(np.full(n_per, ci, dtype=np.int64))
            S.append(np.full(n_per, snr, dtype=np.int64))
        if verbose:
            print(f"  generated {mod:7s} ({n_per} x {len(list(snrs))} SNRs)")
    return np.concatenate(X), np.concatenate(Y), np.concatenate(S)
